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

func main() {
	agent := agents.New("Joker").
		WithInstructions("You are a helpful assistant.").
		WithModel("gpt-4.1-nano")

	ctx := context.Background()

	result, err := agents.Runner{}.RunStreamed(ctx, agent, agents.InputString("Please tell me 5 jokes."))
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
}
