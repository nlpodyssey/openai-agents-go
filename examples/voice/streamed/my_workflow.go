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

//go:build portaudio

package main

import (
	"context"
	"fmt"
	"iter"
	"math/rand"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agents/extensions/handoff_prompt"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

type GetWeatherArgs struct {
	City string `json:"city"`
}

// GetWeather returns the weather for a given city.
func GetWeather(_ context.Context, args GetWeatherArgs) (string, error) {
	fmt.Printf("[debug] GetWeather called with city: %s\n", args.City)
	choices := []string{"sunny", "cloudy", "rainy", "snowy"}
	choice := choices[rand.Intn(len(choices))]
	return fmt.Sprintf("The weather in %s is %s.", args.City, choice), nil
}

var (
	GetWeatherTool = agents.NewFunctionTool(
		"get_weather",
		"Get the weather for a given city.",
		GetWeather,
	)

	AdvancedAgent = agents.New("Advanced").
			WithHandoffDescription("Advanced reasoning with GPT-4o for complex tasks.").
			WithInstructions(handoff_prompt.PromptWithHandoffInstructions(
			"You have advanced capabilities. Handle complex reasoning, code generation, and analysis.",
		)).
		WithModel("gpt-4o") // More expensive model

	Agent = agents.New("Assistant").
		WithInstructions(handoff_prompt.PromptWithHandoffInstructions(
			"You're a basic assistant. For simple queries, answer directly. If the user needs complex analysis, code generation, or mathematical reasoning, handoff to the advanced agent.",
		)).
		WithModel("gpt-4o-mini"). // Cheaper model
		WithAgentHandoffs(AdvancedAgent).
		AddTool(GetWeatherTool)
)

type MyWorkflow struct {
	inputHistory []agents.TResponseInputItem
	currentAgent *agents.Agent
	secretWord   string
	onStart      func(string)
}

func NewMyWorkflow(secretWord string, onStart func(string)) *MyWorkflow {
	return &MyWorkflow{
		inputHistory: nil,
		currentAgent: Agent,
		secretWord:   strings.ToLower(secretWord),
		onStart:      onStart,
	}
}

func (w *MyWorkflow) Run(ctx context.Context, transcription string) agents.VoiceWorkflowBaseRunResult {
	w.onStart(transcription)

	// Add the transcription to the input history
	w.inputHistory = append(w.inputHistory, agents.TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(transcription),
			},
			Role: responses.EasyInputMessageRoleUser,
			Type: responses.EasyInputMessageTypeMessage,
		},
	})

	return &myWorkflowRunResult{
		ctx:           ctx,
		w:             w,
		transcription: transcription,
	}
}

func (w *MyWorkflow) OnStart(context.Context) agents.VoiceWorkflowBaseOnStartResult {
	return agents.NoOpVoiceWorkflowBaseOnStartResult{}
}

type myWorkflowRunResult struct {
	ctx           context.Context
	w             *MyWorkflow
	transcription string
	err           error
}

func (r *myWorkflowRunResult) Seq() iter.Seq[string] {
	return func(yield func(string) bool) {
		w := r.w
		// If the user guessed the secret word, do alternate logic
		if strings.Contains(strings.ToLower(r.transcription), w.secretWord) {
			yield("You guessed the secret word!")
			w.inputHistory = append(w.inputHistory, agents.TResponseInputItem{
				OfMessage: &responses.EasyInputMessageParam{
					Content: responses.EasyInputMessageContentUnionParam{
						OfString: param.NewOpt("You guessed the secret word!"),
					},
					Role: responses.EasyInputMessageRoleAssistant,
					Type: responses.EasyInputMessageTypeMessage,
				},
			})
			return
		}

		// Otherwise, run the agent
		result, err := agents.RunInputsStreamed(r.ctx, w.currentAgent, w.inputHistory)
		if err != nil {
			r.err = err
			return
		}

		// Print the agent name to the terminal
		fmt.Printf("\n[%s]: ", w.currentAgent.Name)

		stream := agents.VoiceWorkflowHelper().StreamTextFrom(result)

		for chunk := range stream.Seq() {
			if !yield(chunk) {
				break
			}
		}
		r.err = stream.Error()

		// Update the input history and current agent
		w.inputHistory = result.ToInputList()
		w.currentAgent = result.LastAgent()
	}
}

func (r *myWorkflowRunResult) Error() error { return r.err }
