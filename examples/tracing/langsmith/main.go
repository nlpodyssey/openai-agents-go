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
	"github.com/nlpodyssey/openai-agents-go/tracing/wrappers/langsmith"
)

func main() {
	// Set your LangSmith API key here or via environment variable
	apiKey := "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"

	// Check if LangSmith API key is set
	if apiKey == "" {
		apiKey = os.Getenv("LANGSMITH_API_KEY")
		if apiKey == "" {
			fmt.Println("Warning: LANGSMITH_API_KEY not set.")
			fmt.Println("Traces will be logged to console but not sent to LangSmith.")
			fmt.Println("Set LANGSMITH_API_KEY to enable LangSmith integration.")
		}
	}

	// Create the LangSmith processor (note the corrected function name)
	langsmithProcessor := langsmith.NewTracingProcessor(langsmith.ProcessorParams{
		APIKey:      apiKey, // Pass API key from main
		ProjectName: "openai-agents-go-demo",
		Metadata: map[string]any{
			"environment": "development",
			"version":     "1.0.0",
		},
		Tags: []string{"demo", "golang", "openai-agents"},
		Name: "LangSmith Demo Workflow",
	})

	// Add the LangSmith processor to the tracing system
	// This will work alongside the default OpenAI backend processor
	tracing.AddTraceProcessor(langsmithProcessor)

	// Create agents (simplified to avoid the SDK bug)
	weatherAgent := agents.New("Weather Agent").
		WithInstructions("You are a helpful weather assistant. Always respond in a friendly manner.").
		WithTools(agents.WebSearchTool{}).
		WithModel("gpt-4o-mini")

	jokeAgent := agents.New("Joke Agent").
		WithInstructions("You tell funny jokes and make people laugh. Always respond with a joke.").
		WithModel("gpt-4o-mini")

	// Main agent
	mainAgent := agents.New("Assistant").
		WithInstructions(`You are a helpful assistant. 
		- For weather questions, handoff to the Weather Agent
		- For jokes or funny requests, handoff to the Joke Agent
		- For other questions, answer directly`).
		WithAgentHandoffs(weatherAgent, jokeAgent).
		WithModel("gpt-4o-mini")

	// Run a trace that demonstrates various agent interactions
	err := tracing.RunTrace(
		context.Background(),
		tracing.TraceParams{
			WorkflowName: "LangSmith Integration Demo",
			GroupID:      "demo-conversation-001",
			Metadata: map[string]any{
				"user_id":    "demo-user",
				"session_id": "demo-session-001",
			},
		},
		func(ctx context.Context, trace tracing.Trace) error {
			fmt.Printf("Starting demo workflow - Trace ID: %s\n", trace.TraceID())

			// First interaction - direct response
			result1, err := agents.Run(ctx, mainAgent, "Hello! Can you introduce yourself?")
			if err != nil {
				return err
			}
			fmt.Printf("Introduction: %s\n\n", result1.FinalOutput)

			// Second interaction - weather (should trigger handoff)
			// FIXED: Run each interaction separately to avoid the SDK bug with ToInputList()
			result2, err := agents.Run(ctx, mainAgent, "What's the weather like in San Francisco today?")
			if err != nil {
				return err
			}
			fmt.Printf("Weather Response: %s\n\n", result2.FinalOutput)

			// Third interaction - joke (should trigger handoff)
			result3, err := agents.Run(ctx, mainAgent, "Tell me a programming joke!")
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

	fmt.Println("Demo completed! Check your LangSmith project for the trace data.")

	// Note about viewing traces
	if apiKey != "" {
		fmt.Println("\nYou can view your traces at: https://smith.langchain.com/")
		fmt.Printf("Project: openai-agents-go-demo\n")
	}
}
