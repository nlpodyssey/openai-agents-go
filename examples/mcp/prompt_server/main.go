// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"fmt"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

func main() {
	go runServer("localhost:8000")
	time.Sleep(2 * time.Second) // Give it 2 seconds to start

	server := agents.NewMCPServerStreamableHTTP(agents.MCPServerStreamableHTTPParams{
		Name: "Simple Prompt Server",
		URL:  "http://localhost:8000",
	})
	err := server.Run(context.Background(), func(ctx context.Context, server *agents.MCPServerWithClientSession) error {
		traceID := tracing.GenTraceID()
		return tracing.RunTrace(
			ctx, tracing.TraceParams{WorkflowName: "Simple Prompt Demo", TraceID: traceID},
			func(ctx context.Context, _ tracing.Trace) error {
				fmt.Printf("Trace: https://platform.openai.com/traces/trace?trace_id=%s\n", traceID)
				err := showAvailablePrompts(ctx, server)
				if err != nil {
					return err
				}
				return demoCodeReview(ctx, server)
			},
		)
	})
	if err != nil {
		panic(err)
	}
}

// showAvailablePrompts shows available prompts for user selection
func showAvailablePrompts(ctx context.Context, mcpServer agents.MCPServer) error {
	fmt.Println("=== AVAILABLE PROMPTS ===")
	promptResult, err := mcpServer.ListPrompts(ctx)
	if err != nil {
		return err
	}
	fmt.Println("User can select from these prompts:")
	for i, prompt := range promptResult.Prompts {
		fmt.Printf("  %d. %s - %s\n", i+1, prompt.Name, prompt.Description)
	}
	fmt.Println()
	return nil
}

// Demo: Code review with user-selected prompt.
func demoCodeReview(ctx context.Context, mcpServer agents.MCPServer) error {
	fmt.Println("=== CODE REVIEW DEMO ===")

	// User explicitly selects prompt and parameters
	instructions := getInstructionsFromPrompt(
		ctx,
		mcpServer,
		"generate_code_review_instructions",
		map[string]string{
			"focus":    "security vulnerabilities",
			"language": "python",
		},
	)

	agent := agents.New("Code Reviewer Agent").
		WithInstructions(instructions). // Instructions from MCP prompt
		WithModelSettings(modelsettings.ModelSettings{
			ToolChoice: modelsettings.ToolChoiceAuto,
		}).
		WithModel("gpt-4o")

	message := `Please review this code:

def process_user_input(user_input):
    command = f"echo {user_input}"
    os.system(command)
    return "Command executed"

`
	fmt.Printf("Running: %s...\n", message[:60])
	result, err := agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)
	fmt.Printf("\n==========\n\n")
	return nil
}

// Get agent instructions by calling MCP prompt endpoint (user-controlled).
func getInstructionsFromPrompt(
	ctx context.Context,
	mcpServer agents.MCPServer,
	promptName string,
	arguments map[string]string,
) string {
	fmt.Printf("Getting instructions from prompt: %s\n", promptName)

	promptResult, err := mcpServer.GetPrompt(ctx, promptName, arguments)
	if err != nil {
		fmt.Printf("Failed to get instructions: %v", err)
		return fmt.Sprintf("You are a helpful assistant. Error: %v", err)
	}

	var instructions string
	content := promptResult.Messages[0].Content
	if textContent, ok := content.(*mcp.TextContent); ok {
		instructions = textContent.Text
	} else if b, err := content.MarshalJSON(); err == nil {
		instructions = string(b)
	} else {
		instructions = fmt.Sprintf("%v", content)
	}

	fmt.Println("Generated instructions")
	return instructions
}
