package workflowrunner

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
)

// WorkflowRequest represents the top-level payload describing a workflow run.
type WorkflowRequest struct {
	Query    string              `json:"query"`
	Session  SessionDeclaration  `json:"session"`
	Callback CallbackDeclaration `json:"callback"`
	Workflow WorkflowDeclaration `json:"workflow"`
	Metadata map[string]any      `json:"metadata,omitempty"`
	Context  map[string]any      `json:"context,omitempty"`
}

// SessionDeclaration carries caller-provided state and execution limits.
type SessionDeclaration struct {
	SessionID   string                `json:"session_id"`
	HistorySize int                   `json:"history_size,omitempty"`
	MaxTurns    int                   `json:"max_turns,omitempty"`
	Credentials CredentialDeclaration `json:"credentials"`
}

// CredentialDeclaration contains minimal identity data used for validation / logging.
type CredentialDeclaration struct {
	UserID       string         `json:"user_id"`
	AccountID    string         `json:"account_id"`
	Capabilities []string       `json:"capabilities,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

// CallbackDeclaration describes how streaming events should be published.
type CallbackDeclaration struct {
	Target string `json:"target"`
	Mode   string `json:"mode,omitempty"`
}

// UnmarshalJSON allows callback to be provided as string or object.
func (c *CallbackDeclaration) UnmarshalJSON(data []byte) error {
	type alias CallbackDeclaration
	var (
		asStr string
		asObj alias
	)
	if err := json.Unmarshal(data, &asStr); err == nil {
		c.Target = asStr
		c.Mode = ""
		return nil
	}
	if err := json.Unmarshal(data, &asObj); err != nil {
		return err
	}
	*c = CallbackDeclaration(asObj)
	return nil
}

// WorkflowDeclaration defines the agent graph that should be executed.
type WorkflowDeclaration struct {
	Name          string             `json:"name"`
	StartingAgent string             `json:"starting_agent"`
	Agents        []AgentDeclaration `json:"agents"`
	Metadata      map[string]any     `json:"metadata,omitempty"`
}

// AgentDeclaration captures the configuration of a single agent.
type AgentDeclaration struct {
	Name               string                 `json:"name"`
	DisplayName        string                 `json:"display_name,omitempty"`
	Instructions       string                 `json:"instructions,omitempty"`
	PromptID           string                 `json:"prompt_id,omitempty"`
	Model              *ModelDeclaration      `json:"model,omitempty"`
	Handoffs           []string               `json:"handoff,omitempty"`
	AgentTools         []AgentToolReference   `json:"agent_tools,omitempty"`
	Tools              []ToolDeclaration      `json:"tools,omitempty"`
	MCPServers         []MCPDeclaration       `json:"mcp,omitempty"`
	InputGuardrails    []GuardrailDeclaration `json:"input_guardrails,omitempty"`
	OutputGuardrails   []GuardrailDeclaration `json:"output_guardrails,omitempty"`
	OutputType         *OutputTypeDeclaration `json:"output_type,omitempty"`
	HandoffDescription string                 `json:"handoff_description,omitempty"`
	Annotations        map[string]any         `json:"annotations,omitempty"`
}

// AgentToolReference allows referencing another agent as a tool.
type AgentToolReference struct {
	AgentName   string `json:"agent_name"`
	ToolName    string `json:"tool_name,omitempty"`
	Description string `json:"description,omitempty"`
}

// ToolDeclaration represents a tool that should be attached to an agent.
type ToolDeclaration struct {
	Type   string         `json:"type"`
	Name   string         `json:"name,omitempty"`
	Config map[string]any `json:"config,omitempty"`
}

// MCPDeclaration configures hosted or stdio MCP servers.
type MCPDeclaration struct {
	Type            string         `json:"type,omitempty"`
	ServerLabel     string         `json:"server_label,omitempty"`
	Address         string         `json:"address"`
	RequireApproval string         `json:"require_approval,omitempty"`
	Additional      map[string]any `json:"additional,omitempty"`
}

// GuardrailDeclaration references a reusable guardrail preset.
type GuardrailDeclaration struct {
	Name   string         `json:"name"`
	Config map[string]any `json:"config,omitempty"`
	Target string         `json:"target,omitempty"`
}

// OutputTypeDeclaration describes the expected structured output.
type OutputTypeDeclaration struct {
	Name   string         `json:"name"`
	Strict bool           `json:"strict,omitempty"`
	Schema map[string]any `json:"schema,omitempty"`
}

// ModelDeclaration indicates which model/provider to use and optional settings.
type ModelDeclaration struct {
	Provider     string                `json:"provider,omitempty"`
	Model        string                `json:"model"`
	Temperature  *float64              `json:"temperature,omitempty"`
	TopP         *float64              `json:"top_p,omitempty"`
	MaxTokens    *int64                `json:"max_tokens,omitempty"`
	Reasoning    *ReasoningDeclaration `json:"reasoning,omitempty"`
	Verbosity    string                `json:"verbosity,omitempty"`
	Metadata     map[string]string     `json:"metadata,omitempty"`
	ExtraHeaders map[string]string     `json:"extra_headers,omitempty"`
	ExtraQuery   map[string]string     `json:"extra_query,omitempty"`
	ToolChoice   string                `json:"tool_choice,omitempty"`
}

// ReasoningDeclaration mirrors the subset of OpenAI reasoning parameters we support.
type ReasoningDeclaration struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// Validate performs shallow validation of the callback declaration.
func (c *CallbackDeclaration) Validate() error {
	mode := strings.ToLower(c.Mode)
	if mode == "stdout" {
		return nil
	}
	if strings.TrimSpace(c.Target) == "" {
		return fmt.Errorf("callback target is required")
	}
	if _, err := url.ParseRequestURI(c.Target); err != nil {
		return fmt.Errorf("callback target %q is not a valid URL: %w", c.Target, err)
	}
	return nil
}
