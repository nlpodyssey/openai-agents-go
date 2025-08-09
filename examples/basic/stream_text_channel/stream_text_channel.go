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

	events, errCh, err := agents.RunStreamedChan(ctx, agent, "Just write the word 'hi' and nothing else.")
	if err != nil {
		panic(err)
	}

	for event := range events {
		if e, ok := event.(agents.RawResponsesStreamEvent); ok && e.Data.Type == "response.output_text.delta" {
			fmt.Print(e.Data.Delta)
			_ = os.Stdout.Sync()
		}
	}

	if streamErr := <-errCh; streamErr != nil {
		panic(streamErr)
	}
}
