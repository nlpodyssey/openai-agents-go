package workflowrunner

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/memory"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

// ToolFactory creates an agents.Tool from the declaration.
type ToolFactory func(ctx context.Context, decl ToolDeclaration, env ToolFactoryEnv) (agents.Tool, error)

// ToolFactoryEnv provides context when constructing tools.
type ToolFactoryEnv struct {
	AgentName       string
	WorkflowName    string
	RequestMetadata map[string]any
}

// GuardrailFactory is a placeholder for future guardrail registry integration.
type GuardrailFactory func(ctx context.Context, decl GuardrailDeclaration) (agents.InputGuardrail, error)

// OutputTypeFactory produces custom output type implementations.
type OutputTypeFactory func(ctx context.Context, decl OutputTypeDeclaration) (agents.OutputTypeInterface, error)

// SessionFactory allocates or loads a conversational session.
type SessionFactory func(ctx context.Context, decl SessionDeclaration) (memory.Session, error)

// BuildResult contains the artifacts required to execute a workflow.
type BuildResult struct {
	StartingAgent *agents.Agent
	AgentMap      map[string]*agents.Agent
	Runner        agents.Runner
	Session       memory.Session
	WorkflowName  string
	TraceMetadata map[string]any
}

// Builder converts declarative workflow payloads into executable SDK primitives.
type Builder struct {
	ToolFactories       map[string]ToolFactory
	OutputTypeFactories map[string]OutputTypeFactory
	SessionFactory      SessionFactory
}

// NewDefaultBuilder returns a Builder with the builtin registries initialized.
func NewDefaultBuilder() *Builder {
	return &Builder{
		ToolFactories: map[string]ToolFactory{
			"web_search":       newWebSearchTool,
			"code_interpreter": newCodeInterpreterTool,
			"file_search":      newFileSearchTool,
			"image_generation": newImageGenerationTool,
			"hosted_mcp":       newHostedMCPTool,
		},
		OutputTypeFactories: map[string]OutputTypeFactory{
			"json_object": newJSONMapOutputType,
		},
		SessionFactory: NewSQLiteSessionFactory("workflowrunner_sessions"),
	}
}

// Build constructs agents, run configuration, and session resources from the request.
func (b *Builder) Build(ctx context.Context, req WorkflowRequest) (*BuildResult, error) {
	if err := ValidateWorkflowRequest(req); err != nil {
		return nil, err
	}

	sessionFactory := b.SessionFactory
	if sessionFactory == nil {
		sessionFactory = NewSQLiteSessionFactory("workflowrunner_sessions")
	}
	session, err := sessionFactory(ctx, req.Session)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	agentMap := make(map[string]*agents.Agent, len(req.Workflow.Agents))
	type pendingConfig struct {
		decl       AgentDeclaration
		agent      *agents.Agent
		agentTools []AgentToolReference
		toolDecls  []ToolDeclaration
	}
	pending := make([]pendingConfig, 0, len(req.Workflow.Agents))

	for _, decl := range req.Workflow.Agents {
		agent := agents.New(decl.Name)
		if decl.DisplayName != "" {
			agent.Name = decl.DisplayName
		}
		if decl.HandoffDescription != "" {
			agent.WithHandoffDescription(decl.HandoffDescription)
		}
		if strings.TrimSpace(decl.Instructions) != "" {
			agent.WithInstructions(decl.Instructions)
		}
		if decl.PromptID != "" {
			agent.WithPrompt(agents.Prompt{ID: decl.PromptID})
		}
		if decl.Model != nil {
			if err := applyModelDeclaration(agent, *decl.Model); err != nil {
				return nil, fmt.Errorf("agent %q model: %w", decl.Name, err)
			}
		}
		if decl.OutputType != nil {
			outputType, err := b.buildOutputType(ctx, *decl.OutputType)
			if err != nil {
				return nil, fmt.Errorf("agent %q output type: %w", decl.Name, err)
			}
			agent.WithOutputType(outputType)
		}
		if gr, err := buildInputGuardrails(ctx, decl.InputGuardrails); err != nil {
			return nil, fmt.Errorf("agent %q input guardrails: %w", decl.Name, err)
		} else if len(gr) > 0 {
			agent.WithInputGuardrails(gr)
		}
		if gr, err := buildOutputGuardrails(ctx, decl.OutputGuardrails); err != nil {
			return nil, fmt.Errorf("agent %q output guardrails: %w", decl.Name, err)
		} else if len(gr) > 0 {
			agent.WithOutputGuardrails(gr)
		}
		pending = append(pending, pendingConfig{
			decl:       decl,
			agent:      agent,
			agentTools: decl.AgentTools,
			toolDecls:  append(slices.Clone(decl.Tools), toolsFromMCP(decl.MCPServers)...),
		})
		agentMap[decl.Name] = agent
	}

	// Second pass: attach handoffs and tools.
	for _, item := range pending {
		agent := item.agent
		if len(item.decl.Handoffs) > 0 {
			handoffAgents := make([]*agents.Agent, 0, len(item.decl.Handoffs))
			for _, ref := range item.decl.Handoffs {
				target, ok := agentMap[ref]
				if !ok {
					return nil, fmt.Errorf("agent %q references unknown handoff agent %q", item.decl.Name, ref)
				}
				handoffAgents = append(handoffAgents, target)
			}
			agent.WithAgentHandoffs(handoffAgents...)
		}
		if len(item.agentTools) > 0 {
			for _, ref := range item.agentTools {
				target, ok := agentMap[ref.AgentName]
				if !ok {
					return nil, fmt.Errorf("agent %q agent_tool references unknown agent %q", item.decl.Name, ref.AgentName)
				}
				params := agents.AgentAsToolParams{
					ToolName:        ref.ToolName,
					ToolDescription: ref.Description,
				}
				agent.AddTool(target.AsTool(params))
			}
		}
		if len(item.toolDecls) > 0 {
			for _, toolDecl := range item.toolDecls {
				factory, ok := b.ToolFactories[toolDecl.Type]
				if !ok {
					return nil, fmt.Errorf("agent %q tool type %q not registered", item.decl.Name, toolDecl.Type)
				}
				tool, err := factory(ctx, toolDecl, ToolFactoryEnv{
					AgentName:       item.decl.Name,
					WorkflowName:    req.Workflow.Name,
					RequestMetadata: req.Metadata,
				})
				if err != nil {
					return nil, fmt.Errorf("agent %q tool %q: %w", item.decl.Name, toolDecl.Type, err)
				}
				agent.AddTool(tool)
			}
		}
	}

	startingAgent, ok := agentMap[req.Workflow.StartingAgent]
	if !ok {
		return nil, fmt.Errorf("starting agent %q missing", req.Workflow.StartingAgent)
	}

	runConfig := agents.RunConfig{
		WorkflowName: req.Workflow.Name,
	}
	if req.Session.MaxTurns > 0 {
		runConfig.MaxTurns = uint64(req.Session.MaxTurns)
	}
	runConfig.Session = session
	if req.Session.HistorySize > 0 {
		runConfig.LimitMemory = req.Session.HistorySize
	}
	runConfig.TracingDisabled = false
	runConfig.GroupID = req.Session.SessionID
	traceMetadata := composeTraceMetadata(req)
	runConfig.TraceMetadata = maps.Clone(traceMetadata)

	builderResult := &BuildResult{
		StartingAgent: startingAgent,
		AgentMap:      agentMap,
		Runner:        agents.Runner{Config: runConfig},
		Session:       session,
		WorkflowName:  req.Workflow.Name,
		TraceMetadata: traceMetadata,
	}
	return builderResult, nil
}

func applyModelDeclaration(agent *agents.Agent, decl ModelDeclaration) error {
	if strings.TrimSpace(decl.Provider) != "" && !strings.EqualFold(decl.Provider, "openai") {
		return fmt.Errorf("provider %q not supported (only openai is available in this build)", decl.Provider)
	}
	if strings.TrimSpace(decl.Model) == "" {
		return errors.New("model name cannot be empty")
	}
	agent.WithModel(decl.Model)
	settings := modelsettings.ModelSettings{}
	if decl.Temperature != nil {
		settings.Temperature = param.NewOpt(*decl.Temperature)
	}
	if decl.TopP != nil {
		settings.TopP = param.NewOpt(*decl.TopP)
	}
	if decl.MaxTokens != nil {
		settings.MaxTokens = param.NewOpt(*decl.MaxTokens)
	}
	if decl.Verbosity != "" {
		switch strings.ToLower(decl.Verbosity) {
		case "low":
			settings.Verbosity = param.NewOpt(modelsettings.VerbosityLow)
		case "medium":
			settings.Verbosity = param.NewOpt(modelsettings.VerbosityMedium)
		case "high":
			settings.Verbosity = param.NewOpt(modelsettings.VerbosityHigh)
		default:
			return fmt.Errorf("unsupported verbosity %q", decl.Verbosity)
		}
	}
	if decl.Metadata != nil {
		settings.Metadata = decl.Metadata
	}
	if decl.ExtraHeaders != nil {
		settings.ExtraHeaders = decl.ExtraHeaders
	}
	if decl.ExtraQuery != nil {
		settings.ExtraQuery = decl.ExtraQuery
	}
	if decl.Reasoning != nil {
		settings.Reasoning = buildReasoningParam(*decl.Reasoning)
	}
	if strings.TrimSpace(decl.ToolChoice) != "" {
		settings.ToolChoice = modelsettings.ToolChoiceString(decl.ToolChoice)
	}
	agent.WithModelSettings(settings)
	return nil
}

func buildReasoningParam(decl ReasoningDeclaration) openai.ReasoningParam {
	var result openai.ReasoningParam
	switch strings.ToLower(decl.Effort) {
	case "low":
		result.Effort = openai.ReasoningEffortLow
	case "medium":
		result.Effort = openai.ReasoningEffortMedium
	case "high":
		result.Effort = openai.ReasoningEffortHigh
	case "":
	default:
		result.Effort = openai.ReasoningEffort(decl.Effort)
	}
	switch strings.ToLower(decl.Summary) {
	case "auto":
		result.Summary = openai.ReasoningSummaryAuto
	case "concise":
		result.Summary = openai.ReasoningSummaryConcise
	case "detailed":
		result.Summary = openai.ReasoningSummaryDetailed
	case "":
	default:
		result.Summary = openai.ReasoningSummary(decl.Summary)
	}
	return result
}

func (b *Builder) buildOutputType(ctx context.Context, decl OutputTypeDeclaration) (agents.OutputTypeInterface, error) {
	if decl.Schema == nil {
		factory, ok := b.OutputTypeFactories[decl.Name]
		if !ok {
			return nil, fmt.Errorf("output type %q not registered", decl.Name)
		}
		return factory(ctx, decl)
	}
	name := decl.Name
	if name == "" {
		name = "inline_schema"
	}
	return newSchemaOutputType(name, decl.Strict, decl.Schema)
}

func toolsFromMCP(decls []MCPDeclaration) []ToolDeclaration {
	if len(decls) == 0 {
		return nil
	}
	out := make([]ToolDeclaration, 0, len(decls))
	for _, d := range decls {
		config := map[string]any{
			"server_label": d.ServerLabel,
			"server_url":   d.Address,
		}
		if d.RequireApproval != "" {
			config["require_approval"] = d.RequireApproval
		}
		for k, v := range d.Additional {
			config[k] = v
		}
		out = append(out, ToolDeclaration{
			Type:   "hosted_mcp",
			Name:   d.ServerLabel,
			Config: config,
		})
	}
	return out
}
