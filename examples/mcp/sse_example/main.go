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

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

func main() {
	// We'll run the SSE server in a goroutine. Usually this would be a remote server, but for this
	// demo, we'll run it locally at http://localhost:8000
	// Proper server handling, including graceful shutdown, is omitted just to keep the example short.
	go runServer("localhost:8000")

	time.Sleep(2 * time.Second) // Give it 2 seconds to start

	server := agents.NewMCPServerSSE(agents.MCPServerSSEParams{
		Name:    "SSE Go Server",
		BaseURL: "http://localhost:8000",
	})

	err := server.Run(context.Background(), func(ctx context.Context, server *agents.MCPServerWithClientSession) error {
		traceID := tracing.GenTraceID()
		return tracing.RunTrace(
			ctx, tracing.TraceParams{WorkflowName: "SSE Example", TraceID: traceID},
			func(ctx context.Context, _ tracing.Trace) error {
				fmt.Printf("View trace: https://platform.openai.com/traces/trace?trace_id=%s\n", traceID)
				return run(ctx, server)
			},
		)
	})
	if err != nil {
		panic(err)
	}
}

func run(ctx context.Context, mcpServer agents.MCPServer) error {
	agent := agents.New("Assistant").
		WithInstructions("Use the tools to answer the questions.").
		AddMCPServer(mcpServer).
		WithModelSettings(modelsettings.ModelSettings{
			ToolChoice: modelsettings.ToolChoiceRequired,
		}).
		WithModel("gpt-5-chat-latest")

	// Use the `add` tool to add two numbers
	message := "Add these numbers: 7 and 22."
	fmt.Println("Running:", message)
	result, err := agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)

	// Run the `get_weather` tool
	message = "What's the weather in Tokyo?"
	fmt.Println("\nRunning:", message)
	result, err = agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)

	// Run the `get_secret_word` tool
	message = "What's the secret word?"
	fmt.Println("\nRunning:", message)
	result, err = agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)

	return nil
}
