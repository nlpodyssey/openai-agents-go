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
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

func main() {
	agent := agents.New("Code interpreter").
		WithInstructions("You love doing math.").
		WithTools(
			agents.CodeInterpreterTool{
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

	err := tracing.RunTrace(
		context.Background(),
		tracing.TraceParams{WorkflowName: "Code interpreter example"},
		func(ctx context.Context, _ tracing.Trace) error {
			fmt.Println("Solving math problem...")
			result, err := agents.RunStreamed(ctx, agent, "What is the square root of 273 * 312821 plus 1782?")
			if err != nil {
				return err
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
				return err
			}

			fmt.Printf("Final output: %s\n", result.FinalOutput())
			return nil
		},
	)
	if err != nil {
		panic(err)
	}
}
