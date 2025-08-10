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
	"log"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/nlpodyssey/openai-agents-go/tracing/wrappers/traceloop"
)

// Test tool for verifying tool call extraction
type EchoArgs struct {
	Message string `json:"message"`
}

func EchoTool(_ context.Context, args EchoArgs) (string, error) {
	fmt.Printf("[TOOL EXECUTED] Echo: %s\n", args.Message)
	return fmt.Sprintf("Echo: %s", args.Message), nil
}

var echoTool = agents.NewFunctionTool("echo", "Echoes the input message", EchoTool)

func testToolCalling() {
	apiKey := "tl_4be59d06bb644ced90f8b21e2924a31e"
	
	ctx := context.Background()

	// Create the Traceloop processor
	traceloopProcessor, err := traceloop.NewTracingProcessor(ctx, traceloop.ProcessorParams{
		APIKey:  apiKey,
		BaseURL: "api.traceloop.com",
		Metadata: map[string]any{
			"environment": "test",
			"version":     "1.0.0",
		},
		Tags: []string{"tool-test", "golang"},
	})
	if err != nil {
		log.Fatalf("Failed to create Traceloop processor: %v", err)
	}

	tracing.AddTraceProcessor(traceloopProcessor)

	// Create agent that is forced to use the tool
	agent := agents.New("Tool Test Agent").
		WithInstructions("You MUST use the echo tool to repeat any message the user gives you. Always call the tool.").
		WithTools(echoTool).
		WithModel("gpt-4o-mini")

	// Run the test
	err = tracing.RunTrace(ctx, tracing.TraceParams{
		WorkflowName: "Tool Calling Test",
		TraceID:      "test-tool-trace-" + fmt.Sprintf("%d", 123456),
	}, func(ctx context.Context, trace tracing.Trace) error {
		fmt.Println("Testing tool calling with traceloop integration...")
		
		result, err := agents.Run(ctx, agent, "Echo this message: 'Tool calling works!'")
		if err != nil {
			return fmt.Errorf("agent run failed: %w", err)
		}

		fmt.Printf("Agent Response: %s\n", result.FinalOutput)
		return nil
	})

	if err != nil {
		log.Fatalf("Test failed: %v", err)
	}

	fmt.Println("Tool calling test completed!")
}