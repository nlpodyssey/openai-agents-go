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
This is a conversational voice example that maintains context across multiple turns.

1. You can have a multi-turn conversation using voice input/output.
2. The pipeline automatically transcribes your audio input.
3. The agent maintains conversation history across turns.
4. The output of the agent is streamed to the audio player.
5. Type 'quit' or 'exit' to end the conversation.

Try examples like:
- Tell me a joke (will respond with a joke)
- What's the weather in Tokyo? (will call the `get_weather` tool and then speak)
- Hola, como estas? (will handoff to the spanish agent)
- Follow-up questions that reference previous conversation context
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
	"Get the weather for a given city.",
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

	fmt.Println("=== Conversational Voice Assistant ===")
	fmt.Println("Press Enter to start recording each turn.")
	fmt.Println("After the assistant responds, you can:")
	fmt.Println("- Press Enter again for another voice turn")
	fmt.Println("- Type 'quit' or 'exit' to end the conversation")
	fmt.Println()

	// Create a single workflow instance to maintain conversation history
	workflow := agents.NewSingleAgentVoiceWorkflow(Agent, WorkflowCallbacks{})

	err := usingPortaudio(func() error {
		for {
			fmt.Print("Press <enter> to record, or type 'quit'/'exit' to end: ")

			// Check if user wants to quit before recording
			input, quit, err := checkForQuit()
			if err != nil {
				return err
			}
			if quit {
				fmt.Println("Goodbye!")
				break
			}
			if input != "" {
				fmt.Printf("Text input not supported in voice mode. Press Enter to record audio.\n")
				continue
			}

			fmt.Println("Recording started... Press <enter> to stop recording.")

			pipeline, err := agents.NewVoicePipeline(agents.VoicePipelineParams{
				Workflow: workflow,
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

			err = usingAudioPlayer(func(player *AudioPlayer) error {
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
			if err != nil {
				return err
			}

			fmt.Printf("Turn completed.\n\n")
		}
		return nil
	})
	if err != nil {
		panic(err)
	}
}
