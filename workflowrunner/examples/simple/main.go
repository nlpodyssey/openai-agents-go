package main

import (
	"context"
	"fmt"
	"time"

	"github.com/nlpodyssey/openai-agents-go/workflowrunner"
)

func main() {
	builder := workflowrunner.NewDefaultBuilder()
	service := workflowrunner.NewRunnerService(builder)

	req := workflowrunner.WorkflowRequest{
		Query: "List three fun facts about Mars.",
		Session: workflowrunner.SessionDeclaration{
			SessionID: "demo-simple",
			Credentials: workflowrunner.CredentialDeclaration{
				UserID:    "user-123",
				AccountID: "acct-456",
			},
			HistorySize: 10,
			MaxTurns:    8,
		},
		Callback: workflowrunner.CallbackDeclaration{
			Mode: "stdout",
		},
		Workflow: workflowrunner.WorkflowDeclaration{
			Name:          "simple_assistant",
			StartingAgent: "assistant",
			Agents: []workflowrunner.AgentDeclaration{
				{
					Name:         "assistant",
					Instructions: "You are an enthusiastic planetary science assistant.",
					Model: &workflowrunner.ModelDeclaration{
						Model:       "gpt-4o-mini",
						Temperature: floatPtr(0.3),
					},
				},
			},
		},
		Metadata: map[string]any{
			"requested_by": "example-simple",
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

	fmt.Printf("\nFinal output:\n%v\n", result.Value.FinalOutput)
}

func floatPtr(v float64) *float64 {
	return &v
}
