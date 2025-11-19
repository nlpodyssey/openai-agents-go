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

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/memory"
	"github.com/openai/openai-go/v3/responses"
)

/*
Example demonstrating session memory functionality.

This example shows how to use session memory to maintain conversation history
across multiple agent runs without manually handling ToInputList().
*/

func main() {
	// Create an agent
	agent := agents.New("Assistant").
		WithInstructions("Reply very concisely.").
		WithModel("gpt-4o")

	ctx := context.Background()

	// Create a session instance that will persist across runs
	session, err := memory.NewSQLiteSession(ctx, memory.SQLiteSessionParams{
		SessionID: "conversation_123",
	})
	if err != nil {
		panic(err)
	}
	defer func() {
		err = session.Close()
		if err != nil {
			panic(err)
		}
	}()

	runner := agents.Runner{
		Config: agents.RunConfig{
			Session: session,
		},
	}

	fmt.Println("=== Session Example ===")
	fmt.Printf("The agent will remember previous messages automatically.\n\n")

	// First turn
	fmt.Println("First turn:")
	fmt.Println("User: What city is the Golden Gate Bridge in?")
	result, err := runner.Run(ctx, agent, "What city is the Golden Gate Bridge in?")
	if err != nil {
		panic(err)
	}
	fmt.Println("Assistant:", result.FinalOutput)
	fmt.Println()

	// Second turn - the agent will remember the previous conversation
	fmt.Println("Second turn:")
	fmt.Println("User: What state is it in?")
	result, err = runner.Run(ctx, agent, "What state is it in?")
	if err != nil {
		panic(err)
	}
	fmt.Println("Assistant:", result.FinalOutput)
	fmt.Println()

	// Third turn - continuing the conversation
	fmt.Println("Third turn:")
	fmt.Println("User: What's the population of that state?")
	result, err = runner.Run(ctx, agent, "What's the population of that state?")
	if err != nil {
		panic(err)
	}
	fmt.Println("Assistant:", result.FinalOutput)
	fmt.Println()

	fmt.Println("=== Conversation Complete ===")
	fmt.Println("Notice how the agent remembered the context from previous turns!")
	fmt.Println("Sessions automatically handles conversation history.")

	// Demonstrate the limit parameter - get only the latest 2 items
	fmt.Println("\n=== Latest Items Demo ===")
	latestItems, err := session.GetItems(ctx, 2)
	fmt.Println("Latest 2 items:")
	for i, msg := range latestItems {
		role := "unknown"
		if r := msg.GetRole(); r != nil {
			role = *r
		}
		var content string
		switch v := msg.GetContent().AsAny().(type) {
		case *string:
			content = *v
		case *responses.ResponseInputMessageContentListParam:
			content = (*v)[0].OfInputText.Text
		case *[]responses.ResponseOutputMessageContentUnionParam:
			content = (*v)[0].OfOutputText.Text
		default:
			panic("unexpected content")
		}
		fmt.Printf("  %d. %s: %s\n", i+1, role, content)
	}

	fmt.Printf("\nFetched %d out of total conversation history.\n", len(latestItems))

	// Get all items to show the difference
	allItems, err := session.GetItems(ctx, 0)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Total items in session: %d\n", len(allItems))
}
