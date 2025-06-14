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
)

func HowManyJokes(_ context.Context, _ struct{}) (int64, error) {
	return rand.Int63n(10) + 1, nil
}

var HowManyJokesTool = agents.NewFunctionTool[struct{}, int64]("how_many_jokes", "", HowManyJokes)

func main() {
	agent := agents.New("Joker").
		WithInstructions("First call the `how_many_jokes` tool, then tell that many jokes.").
		WithModel("gpt-4.1-nano").
		WithTools(HowManyJokesTool)

	ctx := context.Background()

	result, err := agents.RunStreamed(ctx, agent, "Hello")
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
