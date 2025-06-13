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
)

/*
This demonstrates usage of the `PreviousResponseID` parameter to continue a conversation.
The second run passes the previous response ID to the model, which allows it to continue the
conversation without re-sending the previous messages.

Notes:
1. This only applies to the OpenAI Responses API. Other models will ignore this parameter.
2. Responses are only stored for 30 days as of this writing, so in production you should
store the response ID along with an expiration date; if the response is no longer valid,
you'll need to re-send the previous conversation history.
*/

func main() {
	fmt.Print("Run in stream mode? (y/n): ")
	_ = os.Stdout.Sync()

	var isStream string
	_, err := fmt.Scan(&isStream)
	if err != nil {
		panic(err)
	}

	agent := agents.New("Assistant").
		WithInstructions("You are a helpful assistant. Be VERY concise.").
		WithModel("gpt-4.1-nano")

	if isStream == "y" {
		fmt.Println("-- streaming mode enabled --")
		mainStream(agent)
	} else {
		fmt.Println("-- streaming mode disabled --")
		mainNoStream(agent)
	}
}

func mainNoStream(agent *agents.Agent) {
	ctx := context.Background()

	result, err := agents.Run(ctx, agent, agents.InputString("What is the largest country in South America?"))
	if err != nil {
		panic(err)
	}

	fmt.Println(result.FinalOutput)

	result, err = (agents.Runner{Config: agents.RunConfig{PreviousResponseID: result.LastResponseID()}}).
		Run(ctx, agent, agents.InputString("What is the capital of that country?"))
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
}

func mainStream(agent *agents.Agent) {
	ctx := context.Background()

	result, err := agents.RunStreamed(
		ctx, agent, agents.InputString("What is the largest country in South America?"),
	)
	if err != nil {
		panic(err)
	}

	err = result.StreamEvents(func(event agents.StreamEvent) error {
		if e, ok := event.(agents.RawResponsesStreamEvent); ok && e.Data.Type == "response.output_text.delta" {
			fmt.Print(e.Data.Delta.OfString)
			_ = os.Stdout.Sync()
		}
		return nil
	})
	if err != nil {
		panic(err)
	}
	fmt.Println()

	result, err = (agents.Runner{Config: agents.RunConfig{PreviousResponseID: result.LastResponseID()}}).
		RunStreamed(ctx, agent, agents.InputString("What is the capital of that country?"))
	if err != nil {
		panic(err)
	}

	err = result.StreamEvents(func(event agents.StreamEvent) error {
		if e, ok := event.(agents.RawResponsesStreamEvent); ok && e.Data.Type == "response.output_text.delta" {
			fmt.Print(e.Data.Delta.OfString)
			_ = os.Stdout.Sync()
		}
		return nil
	})
	if err != nil {
		panic(err)
	}
	fmt.Println()
}
