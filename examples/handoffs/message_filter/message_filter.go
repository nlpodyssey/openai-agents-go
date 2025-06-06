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
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

type RandomNumberArgs struct {
	Max int64 `json:"max"`
}

// RandomNumber returns a random integer between 0 and the given maximum.
func RandomNumber(args RandomNumberArgs) int64 {
	return rand.Int63n(args.Max + 1)
}

var RandomNumberTool = agents.FunctionTool{
	Name:        "random_number",
	Description: "Return a random integer between 0 and the given maximum.",
	ParamsJSONSchema: map[string]any{
		"title":                "random_number_args",
		"type":                 "object",
		"required":             []string{"max"},
		"additionalProperties": false,
		"properties": map[string]any{
			"max": map[string]any{
				"title": "Max",
				"type":  "integer",
			},
		},
	},
	OnInvokeTool: func(_ context.Context, _ *runcontext.RunContextWrapper, arguments string) (any, error) {
		var args RandomNumberArgs
		err := json.Unmarshal([]byte(arguments), &args)
		if err != nil {
			return nil, err
		}
		return RandomNumber(args), nil
	},
}

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

var (
	Model = agents.NewAgentModelName("gpt-4o-mini")

	FirstAgent = &agents.Agent{
		Name:         "Assistant",
		Instructions: agents.StringInstructions("Be extremely concise."),
		Tools:        []agents.Tool{RandomNumberTool},
		Model:        param.NewOpt(Model),
	}

	SpanishAgent = &agents.Agent{
		Name:               "Spanish Assistant",
		Instructions:       agents.StringInstructions("You only speak Spanish and are extremely concise."),
		HandoffDescription: "A Spanish-speaking assistant.",
		Model:              param.NewOpt(Model),
	}

	SecondAgent = &agents.Agent{
		Name: "Assistant",
		Instructions: agents.StringInstructions(
			"Be a helpful assistant. If the user speaks Spanish, handoff to the Spanish assistant.",
		),
		Handoffs: []agents.AgentHandoff{
			agents.UnsafeHandoffFromAgent(agents.HandoffFromAgentParams{
				Agent:       SpanishAgent,
				InputFilter: SpanishHandoffMessageFilter,
			}),
		},
		Model: param.NewOpt(Model),
	}
)

func main() {
	// 1. Send a regular message to the first agent
	result, err := agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: FirstAgent,
		Input:         agents.InputString("Hi, my name is Sora."),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("Step 1 done")

	// 2. Ask it to generate a number
	result, err = agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: FirstAgent,
		Input: agents.InputItems(append(
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
		)),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("Step 2 done")

	// 3. Call the second agent
	result, err = agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: SecondAgent,
		Input: agents.InputItems(append(
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
		)),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("Step 3 done")

	// 4. Cause a handoff to occur
	result, err = agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: SecondAgent,
		Input: agents.InputItems(append(
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
		)),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("Step 4 done")

	fmt.Printf("\n===Final messages===\n\n")

	// 5. That should have caused SpanishHandoffMessageFilter to be called, which means the
	// output should be missing the first two messages, and have no tool calls.
	// Let's print the messages to see what happened
	for _, message := range result.ToInputList() {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		err := enc.Encode(message)
		if err != nil {
			panic(err)
		}
		fmt.Println(buf.String())
	}
}
