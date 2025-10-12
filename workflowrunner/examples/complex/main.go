package main

import (
	"context"
	"fmt"
	"time"

	"github.com/nlpodyssey/openai-agents-go/workflowrunner"
)

// This example demonstrates a richer workflow declaration with handoffs, tools,
// and hosted MCP integration. Callback events are printed to stdout.
func main() {
	builder := workflowrunner.NewDefaultBuilder()
	service := workflowrunner.NewRunnerService(builder)

	req := workflowrunner.WorkflowRequest{
		Query: "Draft a two paragraph executive summary about the latest reusable rocket breakthroughs.",
		Session: workflowrunner.SessionDeclaration{
			SessionID: "demo-complex",
			Credentials: workflowrunner.CredentialDeclaration{
				UserID:       "user-42",
				AccountID:    "enterprise-88",
				Capabilities: []string{"research", "summaries"},
			},
			HistorySize: 25,
			MaxTurns:    20,
		},
		Callback: workflowrunner.CallbackDeclaration{
			Mode: "stdout",
		},
		Workflow: workflowrunner.WorkflowDeclaration{
			Name:          "research_supervisor",
			StartingAgent: "supervisor",
			Agents: []workflowrunner.AgentDeclaration{
				{
					Name:         "supervisor",
					Instructions: "Coordinate expert agents to satisfy the user request. Decide whether to hand off to a specialist.",
					Handoffs:     []string{"researcher", "writer"},
					Model: &workflowrunner.ModelDeclaration{
						Model:       "gpt-4o-mini",
						Temperature: floatPtr(0.2),
					},
					HandoffDescription: "High-level orchestrator that delegates to experts.",
				},
				{
					Name:         "researcher",
					Instructions: "Gather concise findings and key points using search tools.",
					Model: &workflowrunner.ModelDeclaration{
						Model:       "gpt-4o-mini",
						Temperature: floatPtr(0.1),
					},
					InputGuardrails: []workflowrunner.GuardrailDeclaration{
						{Name: "math_homework_input"},
						{Name: "basic_profanity_input"},
					},
					Tools: []workflowrunner.ToolDeclaration{
						{
							Type: "web_search",
							Config: map[string]any{
								"user_location": map[string]any{
									"city": "San Francisco",
									"type": "approximate",
								},
								"search_context_size": "high",
							},
						},
					},
				},
				{
					Name:         "writer",
					Instructions: "Transform research bullet points into cohesive prose with executive tone.",
					Model: &workflowrunner.ModelDeclaration{
						Model:       "gpt-4o",
						Temperature: floatPtr(0.4),
					},
					OutputGuardrails: []workflowrunner.GuardrailDeclaration{
						{Name: "basic_profanity_output"},
						{Name: "phone_number_output"},
					},
					AgentTools: []workflowrunner.AgentToolReference{
						{
							AgentName:   "researcher",
							ToolName:    "invoke_research",
							Description: "Use to refresh or expand on research findings.",
						},
					},
					MCPServers: []workflowrunner.MCPDeclaration{
						{
							Address:         "https://gitmcp.io/openai/codex",
							ServerLabel:     "code_search",
							RequireApproval: "always",
						},
					},
				},
			},
		},
		Metadata: map[string]any{
			"requested_by": "example-complex",
			"timestamp":    time.Now().Format(time.RFC3339),
		},
	}

	ctx := context.Background()
	task, err := service.Execute(ctx, req)
	if err != nil {
		panic(err)
	}

	result := task.Await()
	if result.Error != nil {
		fmt.Printf("Run failed: %v\n", result.Error)
		return
	}

	fmt.Println("\nExecutive summary generated!")
}

func floatPtr(v float64) *float64 {
	return &v
}
