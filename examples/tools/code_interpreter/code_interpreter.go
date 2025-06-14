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
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
)

func main() {
	agent := agents.New("Code interpreter").
		WithInstructions("You love doing math.").
		WithTools(
			tools.CodeInterpreter{
				ToolConfig: responses.ToolCodeInterpreterParam{
					Container: responses.ToolCodeInterpreterContainerUnionParam{
						OfCodeInterpreterContainerAuto: &responses.ToolCodeInterpreterContainerCodeInterpreterContainerAutoParam{
							Type: constant.ValueOf[constant.Auto](),
						},
					},
					Type: constant.ValueOf[constant.CodeInterpreter](),
				},
			},
		).
		WithModel("gpt-4.1-nano")

	fmt.Println("Solving math problem...")
	result, err := agents.RunStreamed(context.Background(), agent, "What is the square root of 273 * 312821 plus 1782?")
	if err != nil {
		panic(err)
	}

	err = result.StreamEvents(func(event agents.StreamEvent) error {
		runItemStreamEvent, ok := event.(agents.RunItemStreamEvent)
		if !ok {
			return nil
		}

		if toolCallItem, ok := runItemStreamEvent.Item.(agents.ToolCallItem); ok {
			if codeInterpreterCall, ok := toolCallItem.RawItem.(agents.ResponseCodeInterpreterToolCall); ok {
				fmt.Printf("Code interpreter code:\n```\n%s\n```\n\n", codeInterpreterCall.Code)
				return nil
			}
		}

		fmt.Printf("Other event: %T\n", event)
		return nil
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Final output: %s\n", result.FinalOutput)
}
