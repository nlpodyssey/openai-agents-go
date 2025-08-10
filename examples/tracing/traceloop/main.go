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
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/nlpodyssey/openai-agents-go/tracing/wrappers/traceloop"
)

// Simple test tool for demonstration
type TestToolArgs struct {
	Message string `json:"message"`
}

func TestTool(_ context.Context, args TestToolArgs) (string, error) {
	return fmt.Sprintf("Tool executed with message: %s", args.Message), nil
}

var testTool = agents.NewFunctionTool("test_tool", "A simple test tool", TestTool)

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
		fmt.Printf("Failed to create Traceloop processor: %v\n", err)
		return
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
		TraceID:      "trace_" + fmt.Sprintf("test_tool_%d", 123456),
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
		fmt.Printf("Test failed: %v\n", err)
		return
	}

	fmt.Println("Tool calling test completed!")
}

func main() {
	// Set your Traceloop API key here or via environment variable
	apiKey := "tl_4be59d06bb644ced90f8b21e2924a31e"
	if apiKey == "" {
		fmt.Println("Warning: TRACELOOP_API_KEY environment variable not set.")
		fmt.Println("Please set your Traceloop API key to enable tracing.")
		fmt.Println("You can get an API key from: https://app.traceloop.com")
		return
	}

	ctx := context.Background()

	// Create the Traceloop processor
	traceloopProcessor, err := traceloop.NewTracingProcessor(ctx, traceloop.ProcessorParams{
		APIKey:  apiKey,
		BaseURL: "api.traceloop.com", // Use "api-staging.traceloop.com" for staging
		Metadata: map[string]any{
			"environment": "development",
			"version":     "1.0.0",
		},
		Tags: []string{"demo", "golang", "openai-agents"},
	})
	if err != nil {
		fmt.Printf("Failed to create Traceloop processor: %v\n", err)
		return
	}

	// Add the Traceloop processor to the tracing system
	// This will work alongside the default OpenAI backend processor
	tracing.AddTraceProcessor(traceloopProcessor)

	// Create agents for demonstration
	weatherAgent := agents.New("Weather Agent").
		WithInstructions("You are a helpful weather assistant. Always respond in a friendly manner.").
		WithTools(agents.WebSearchTool{}).
		WithModel("gpt-4o-mini")

	jokeAgent := agents.New("Joke Agent").
		WithInstructions("You tell funny jokes and make people laugh. Use the test tool if you need to send a message.").
		WithTools(testTool).
		WithModel("gpt-4o-mini")

	// Main agent with handoffs
	mainAgent := agents.New("Assistant").
		WithInstructions(`You are a helpful assistant. 
		- For weather questions, handoff to the Weather Agent
		- For jokes or funny requests, handoff to the Joke Agent
		- For other questions, answer directly`).
		WithAgentHandoffs(weatherAgent, jokeAgent).
		WithModel("gpt-4o-mini")

	// Run a trace that demonstrates various agent interactions
	err = tracing.RunTrace(
		ctx,
		tracing.TraceParams{
			WorkflowName: "Traceloop Integration Demo",
			GroupID:      "demo-conversation-001",
			Metadata: map[string]any{
				"user_id":    "demo-user",
				"session_id": "demo-session-001",
				"source":     "traceloop-example",
			},
		},
		func(ctx context.Context, trace tracing.Trace) error {
			fmt.Printf("Starting demo workflow - Trace ID: %s\n", trace.TraceID())

			// First interaction - simple question
			result1, err := agents.Run(ctx, mainAgent, "Hello! Can you introduce yourself?")
			if err != nil {
				return err
			}
			fmt.Printf("Introduction: %s\n\n", result1.FinalOutput)

			// Second interaction - weather query
			result2, err := agents.Run(ctx, mainAgent, "What's the weather like in New York today?")
			if err != nil {
				return err
			}
			fmt.Printf("Weather Response: %s\n\n", result2.FinalOutput)

			// Third interaction - joke request with explicit tool usage
			result3, err := agents.Run(ctx, mainAgent, "Tell me a programming joke! When you handoff to the joke agent, make sure they use the test tool to send a greeting message.")
			if err != nil {
				return err
			}
			fmt.Printf("Joke Response: %s\n\n", result3.FinalOutput)

			return nil
		},
	)

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Demo completed! Check your Traceloop dashboard for the trace data.")

	// Shutdown the processor to ensure all data is flushed
	traceloopProcessor.Shutdown(ctx)

	// Note about viewing traces
	fmt.Println("\nYou can view your traces at: https://app.traceloop.com")
	fmt.Println("The traces will appear as workflows with tasks for each agent interaction.")
	fmt.Println("LLM calls will be captured with full prompt and response data.")
	
	// Also run tool calling test
	fmt.Println("\n--- Running Tool Calling Test ---")
	testToolCalling()
}
