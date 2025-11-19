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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"slices"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agents/extensions/handoff_filters"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

type RandomNumberArgs struct {
	Max int64 `json:"max"`
}

// RandomNumber returns a random integer between 0 and the given maximum.
func RandomNumber(_ context.Context, args RandomNumberArgs) (int64, error) {
	return rand.Int63n(args.Max + 1), nil
}

var RandomNumberTool = agents.NewFunctionTool("random_number", "Return a random integer between 0 and the given maximum.", RandomNumber)

func SpanishHandoffMessageFilter(_ context.Context, handoffMessageData agents.HandoffInputData) (agents.HandoffInputData, error) {
	// First, we'll remove any tool-related messages from the message history
	handoffMessageData = handoff_filters.RemoveAllTools(handoffMessageData)

	// Second, we'll also remove the first two items from the history, just for demonstration
	history := handoffMessageData.InputHistory
	if v, ok := history.(agents.InputItems); ok {
		history = v[2:]
	}

	return agents.HandoffInputData{
		InputHistory:    history,
		PreHandoffItems: slices.Clone(handoffMessageData.PreHandoffItems),
		NewItems:        slices.Clone(handoffMessageData.NewItems),
	}, nil
}

const Model = "gpt-4o-mini"

var (
	FirstAgent = agents.New("Assistant").
			WithInstructions("Be extremely concise.").
			WithTools(RandomNumberTool).
			WithModel(Model)

	SpanishAgent = agents.New("Spanish Assistant").
			WithInstructions("You only speak Spanish and are extremely concise.").
			WithHandoffDescription("A Spanish-speaking assistant.").
			WithModel(Model)

	SecondAgent = agents.New("Assistant").
			WithInstructions("Be a helpful assistant. If the user speaks Spanish, handoff to the Spanish assistant.").
			WithHandoffs(
			agents.HandoffFromAgent(agents.HandoffFromAgentParams{
				Agent:       SpanishAgent,
				InputFilter: SpanishHandoffMessageFilter,
			}),
		).
		WithModel(Model)
)

func main() {
	var result *agents.RunResult

	// Trace the entire run as a single workflow
	err := tracing.RunTrace(
		context.Background(), tracing.TraceParams{WorkflowName: "Message filtering"},
		func(ctx context.Context, _ tracing.Trace) error {
			// 1. Send a regular message to the first agent
			var err error
			result, err = agents.Run(ctx, FirstAgent, "Hi, my name is Sora.")
			if err != nil {
				return err
			}

			fmt.Println("Step 1 done")

			// 2. Ask it to generate a number
			result, err = agents.RunInputs(
				ctx,
				FirstAgent,
				append(
					result.ToInputList(),
					agents.TResponseInputItem{
						OfMessage: &responses.EasyInputMessageParam{
							Content: responses.EasyInputMessageContentUnionParam{
								OfString: param.NewOpt("Can you generate a random number between 0 and 100?"),
							},
							Role: responses.EasyInputMessageRoleUser,
							Type: responses.EasyInputMessageTypeMessage,
						},
					},
				),
			)
			if err != nil {
				return err
			}

			fmt.Println("Step 2 done")

			// 3. Call the second agent
			result, err = agents.RunInputs(
				ctx,
				SecondAgent,
				append(
					result.ToInputList(),
					agents.TResponseInputItem{
						OfMessage: &responses.EasyInputMessageParam{
							Content: responses.EasyInputMessageContentUnionParam{
								OfString: param.NewOpt("I live in New York City. Whats the population of the city?"),
							},
							Role: responses.EasyInputMessageRoleUser,
							Type: responses.EasyInputMessageTypeMessage,
						},
					},
				),
			)
			if err != nil {
				return err
			}

			fmt.Println("Step 3 done")

			// 4. Cause a handoff to occur
			result, err = agents.RunInputs(
				ctx,
				SecondAgent,
				append(
					result.ToInputList(),
					agents.TResponseInputItem{
						OfMessage: &responses.EasyInputMessageParam{
							Content: responses.EasyInputMessageContentUnionParam{
								OfString: param.NewOpt("Por favor habla en español. ¿Cuál es mi nombre y dónde vivo?"),
							},
							Role: responses.EasyInputMessageRoleUser,
							Type: responses.EasyInputMessageTypeMessage,
						},
					},
				),
			)
			if err != nil {
				return err
			}

			fmt.Println("Step 4 done")
			return nil
		},
	)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\n===Final messages===\n\n")

	// 5. That should have caused SpanishHandoffMessageFilter to be called, which means the
	// output should be missing the first two messages, and have no tool calls.
	// Let's print the messages to see what happened
	for _, message := range result.ToInputList() {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		err = enc.Encode(message)
		if err != nil {
			panic(err)
		}
		fmt.Println(buf.String())
	}
}
