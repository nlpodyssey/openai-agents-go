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
	"github.com/nlpodyssey/openai-agents-go/agents/extensions/handoff_prompt"
)

/*
This is a simple example that uses a recorded audio buffer.

1. You can record an audio clip in the terminal.
2. The pipeline automatically transcribes the audio.
3. The agent workflow is a simple one that starts at the Assistant agent.
4. The output of the agent is streamed to the audio player.

Try examples like:
- Tell me a joke (will respond with a joke)
- What's the weather in Tokyo? (will call the `get_weather` tool and then speak)
- Hola, como estas? (will handoff to the spanish agent)
*/

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

var GetWeatherTool = agents.NewFunctionTool(
	"get_weather",
	"GetWeather returns the weather for a given city.",
	GetWeather,
)

var (
	SpanishAgent = agents.New("Spanish").
			WithHandoffDescription("A spanish speaking agent.").
			WithInstructions(handoff_prompt.PromptWithHandoffInstructions(
			"You're speaking to a human, so be polite and concise. Speak in Spanish.",
		)).
		WithModel("gpt-4o-mini")

	Agent = agents.New("Assistant").
		WithInstructions(handoff_prompt.PromptWithHandoffInstructions(
			"You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
		)).
		WithModel("gpt-4o-mini").
		WithAgentHandoffs(SpanishAgent).
		AddTool(GetWeatherTool)
)

type WorkflowCallbacks struct{}

func (WorkflowCallbacks) OnRun(_ context.Context, _ *agents.SingleAgentVoiceWorkflow, transcription string) error {
	fmt.Printf("[debug] on_run called with transcription: %s\n", transcription)
	return nil
}

var _ agents.SingleAgentWorkflowCallbacks = WorkflowCallbacks{}

func main() {
	ctx := context.Background()

	err := usingPortaudio(func() error {
		pipeline, err := agents.NewVoicePipeline(agents.VoicePipelineParams{
			Workflow: agents.NewSingleAgentVoiceWorkflow(Agent, WorkflowCallbacks{}),
		})
		if err != nil {
			return err
		}

		buffer, err := recordAudio()
		if err != nil {
			return err
		}
		audioInput := agents.AudioInput{
			Buffer: agents.AudioDataFloat32(buffer),
		}

		result, err := pipeline.Run(ctx, audioInput)
		if err != nil {
			return err
		}
		stream := result.Stream(ctx)

		return usingAudioPlayer(func(player *AudioPlayer) error {
			for event := range stream.Seq() {
				switch e := event.(type) {
				case agents.VoiceStreamEventAudio:
					if err = player.AddAudio(e.Data.Int16()); err != nil {
						return err
					}
					fmt.Println("Received audio")
				case agents.VoiceStreamEventLifecycle:
					fmt.Printf("Received lifecycle event: %s\n", e.Event)
				}
			}
			if err = stream.Error(); err != nil {
				return err
			}

			// Flush any remaining audio data
			if err = player.Flush(); err != nil {
				return err
			}

			// Add 1 second of silence to the end of the stream to avoid cutting off the last audio.
			return player.AddAudio(make([]int16, 24000))
		})
	})
	if err != nil {
		panic(err)
	}
}
