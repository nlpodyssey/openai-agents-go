package workflowrunner

import (
	"errors"
	"fmt"
	"strings"
)

// ValidateWorkflowRequest performs structural validation and returns an error
// describing the first issue encountered.
func ValidateWorkflowRequest(req WorkflowRequest) error {
	if strings.TrimSpace(req.Query) == "" {
		return errors.New("query is required")
	}
	if err := validateSession(req.Session); err != nil {
		return fmt.Errorf("session invalid: %w", err)
	}
	if err := req.Callback.Validate(); err != nil {
		return fmt.Errorf("callback invalid: %w", err)
	}
	if err := validateWorkflowDeclaration(req.Workflow); err != nil {
		return fmt.Errorf("workflow invalid: %w", err)
	}
	return nil
}

func validateSession(session SessionDeclaration) error {
	if session.SessionID == "" {
		return errors.New("session_id is required")
	}
	if session.Credentials.UserID == "" {
		return errors.New("credentials.user_id is required")
	}
	if session.Credentials.AccountID == "" {
		return errors.New("credentials.account_id is required")
	}
	if session.HistorySize < 0 {
		return errors.New("history_size cannot be negative")
	}
	if session.MaxTurns < 0 {
		return errors.New("max_turns cannot be negative")
	}
	return nil
}

func validateWorkflowDeclaration(workflow WorkflowDeclaration) error {
	if workflow.Name == "" {
		return errors.New("name is required")
	}
	if workflow.StartingAgent == "" {
		return errors.New("starting_agent is required")
	}
	if len(workflow.Agents) == 0 {
		return errors.New("agents cannot be empty")
	}

	seen := make(map[string]struct{}, len(workflow.Agents))
	for i, agent := range workflow.Agents {
		if agent.Name == "" {
			return fmt.Errorf("agents[%d] missing name", i)
		}
		if _, dup := seen[agent.Name]; dup {
			return fmt.Errorf("duplicate agent name %q", agent.Name)
		}
		seen[agent.Name] = struct{}{}
		if err := validateAgentDeclaration(agent); err != nil {
			return fmt.Errorf("agent %q invalid: %w", agent.Name, err)
		}
	}

	if _, ok := seen[workflow.StartingAgent]; !ok {
		return fmt.Errorf("starting_agent %q not found in agents", workflow.StartingAgent)
	}
	for _, agent := range workflow.Agents {
		for _, h := range agent.Handoffs {
			if _, ok := seen[h]; !ok {
				return fmt.Errorf("agent %q handoff %q not found", agent.Name, h)
			}
		}
		for _, tool := range agent.AgentTools {
			if _, ok := seen[tool.AgentName]; !ok {
				return fmt.Errorf("agent %q agent_tool references unknown agent %q", agent.Name, tool.AgentName)
			}
		}
	}
	return nil
}

func validateAgentDeclaration(agent AgentDeclaration) error {
	if agent.Model != nil {
		if agent.Model.Model == "" {
			return errors.New("model.model is required when model is present")
		}
	}
	for _, tool := range agent.Tools {
		if strings.TrimSpace(tool.Type) == "" {
			return fmt.Errorf("tool missing type")
		}
	}
	for _, mcp := range agent.MCPServers {
		if strings.TrimSpace(mcp.Address) == "" {
			return fmt.Errorf("mcp address is required")
		}
	}
	for _, gr := range append(agent.InputGuardrails, agent.OutputGuardrails...) {
		if strings.TrimSpace(gr.Name) == "" {
			return fmt.Errorf("guardrail missing name")
		}
	}
	return nil
}
