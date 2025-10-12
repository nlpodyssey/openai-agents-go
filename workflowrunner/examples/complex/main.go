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
					Name:         "researcher",
					Instructions: "You are a research specialist. Always call the web_search tool first, gather the most recent reusable rocket breakthroughs, and return 3-4 bullet points with citations.",
					Model: &workflowrunner.ModelDeclaration{
						Model:       "gpt-4o-mini",
						Temperature: floatPtr(0.1),
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
					Instructions: "You receive research bullet points and craft a concise two-paragraph executive summary that references those findings.",
					Model: &workflowrunner.ModelDeclaration{
						Model:       "gpt-4o",
						Temperature: floatPtr(0.3),
					},
					OutputGuardrails: []workflowrunner.GuardrailDeclaration{
						{Name: "basic_profanity_output"},
						{Name: "phone_number_output"},
					},
				},
				{
					Name:         "supervisor",
					Instructions: "You orchestrate reusable-rocket research. First, call the `research_insights` tool to gather bullet points. Then call the `exec_summary` tool to produce the final executive summary. After both tools run, deliver the writer's summary to the user.",
					Model: &workflowrunner.ModelDeclaration{
						Model:       "gpt-4o-mini",
						Temperature: floatPtr(0.2),
					},
					AgentTools: []workflowrunner.AgentToolReference{
						{
							AgentName:   "researcher",
							ToolName:    "research_insights",
							Description: "Gather the latest reusable rocket breakthroughs using web search.",
						},
						{
							AgentName:   "writer",
							ToolName:    "exec_summary",
							Description: "Produce the final executive summary based on research bullet points.",
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
