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
	"math/rand"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/nlpodyssey/openai-agents-go/types/optional"
	"github.com/openai/openai-go/packages/param"
)

func HowManyJokes() int64 {
	return rand.Int63n(10) + 1
}

var HowManyJokesTool = agents.FunctionTool{
	Name:        "how_many_jokes",
	Description: "",
	ParamsJSONSchema: map[string]any{
		"title":                "how_many_jokes_args",
		"type":                 "object",
		"required":             []string{},
		"additionalProperties": false,
		"properties":           map[string]any{},
	},
	OnInvokeTool: func(_ context.Context, _ *runcontext.RunContextWrapper, arguments string) (any, error) {
		return HowManyJokes(), nil
	},
	StrictJSONSchema: optional.Value(true),
}

func main() {
	agent := &agents.Agent{
		Name:         "Joker",
		Instructions: agents.StringInstructions("First call the `how_many_jokes` tool, then tell that many jokes."),
		Model:        param.NewOpt(agents.NewAgentModelName("gpt-4.1-nano")),
		Tools: []agents.Tool{
			HowManyJokesTool,
		},
	}

	ctx := context.Background()

	result, err := agents.Runner().RunStreamed(ctx, agents.RunStreamedParams{
		StartingAgent: agent,
		Input:         agents.InputString("Hello"),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("=== Run starting ===")
	err = result.StreamEvents(func(event agents.StreamEvent) error {
		// We'll ignore the raw responses event deltas
		switch e := event.(type) {
		case agents.RawResponsesStreamEvent:
			// Ignore
		case agents.AgentUpdatedStreamEvent:
			fmt.Printf("Agent updated: %s\n", e.NewAgent.Name)
		case agents.RunItemStreamEvent:
			switch item := e.Item.(type) {
			case agents.ToolCallItem:
				fmt.Println("-- Tool was called")
			case agents.ToolCallOutputItem:
				fmt.Printf("-- Tool output: %v\n", item.Output)
			case agents.MessageOutputItem:
				fmt.Printf("-- Message output:\n %s\n", agents.ItemHelpers().TextMessageOutput(item))
			default:
				// Ignore other event types
			}
		}
		return nil
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("=== Run complete ===")
}
