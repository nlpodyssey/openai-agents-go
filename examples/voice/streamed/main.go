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
	"errors"
	"fmt"
	"os"
	"os/signal"
	"slices"
	"syscall"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	printStartupMessage()

	err := UsingPortaudio(func() error {
		audioInput := agents.NewStreamedAudioInput()
		onTranscription := func(transcription string) {
			fmt.Printf("\nTranscription: %s\n", transcription)
		}

		pipeline, err := agents.NewVoicePipeline(agents.VoicePipelineParams{
			Workflow: NewMyWorkflow("dog", onTranscription),
		})
		if err != nil {
			return err
		}

		errCh := make(chan error, 1)
		var player *AudioPlayer
		go func() {
			errCh <- UsingAudioInputStream(func(stream *AudioInputStream) error {
				for {
					select {
					case <-ctx.Done():
						return nil
					default:
						data, err := stream.Read()
						if err != nil {
							return err
						}

						if player != nil && !player.HasOutputFinished() {
							continue // Skip input if hardware is still playing AI audio
						}

						audioInput.AddAudio(agents.AudioDataInt16(slices.Clone(data)))
					}
				}
			})
		}()

		result, err := pipeline.Run(ctx, audioInput)
		if err != nil {
			stop()
			return errors.Join(err, <-errCh)
		}
		stream := result.Stream(ctx)

		err = UsingAudioPlayer(func(p *AudioPlayer) error {
			player = p
			eventCh := make(chan any, 1)
			done := make(chan struct{})
			var streamErr error

			// Process stream events asynchronously to prevent blocking the main event loop
			go func() {
				defer close(eventCh)
				defer close(done)
				for event := range stream.Seq() {
					select {
					case eventCh <- event:
					case <-ctx.Done():
						return
					}
				}
				streamErr = stream.Error()
			}()

			for {
				select {
				case <-ctx.Done():
					// Flush any remaining audio before shutdown
					if err := player.Flush(); err != nil {
						fmt.Printf("Warning: error flushing audio during shutdown: %v\n", err)
					}
					return nil
				case event, ok := <-eventCh:
					if !ok {
						return streamErr
					}

					switch e := event.(type) {
					case agents.VoiceStreamEventAudio:
						if err := player.AddAudio(e.Data.Int16()); err != nil {
							// Log error but continue to allow graceful shutdown
							fmt.Printf("Warning: error adding audio: %v\n", err)
							continue
						}
						fmt.Printf("Received audio: %d bytes\n", e.Data.Len()*2)
					case agents.VoiceStreamEventLifecycle:
						fmt.Printf("Lifecycle event: %v\n", e.Event)
						switch e.Event {
						case agents.VoiceStreamEventLifecycleEventTurnStarted:
							if player != nil {
								player.ResetPlaybackTracking()
							}
						case agents.VoiceStreamEventLifecycleEventTurnEnded, agents.VoiceStreamEventLifecycleEventSessionEnded:
							if err := player.Flush(); err != nil {
								fmt.Printf("Warning: error flushing audio: %v\n", err)
							}
						}
					case agents.VoiceStreamEventError:
						return e.Error
					}
				case <-done:
					return streamErr
				}
			}
		})

		stop()
		return errors.Join(err, <-errCh)
	})
	if err != nil {
		panic(err)
	}
}

func printStartupMessage() {
	fmt.Println("===========================================")
	fmt.Println("Voice Assistant Ready")
	fmt.Println("===========================================")
	fmt.Println()
	fmt.Println("What to expect:")
	fmt.Println("• Speak naturally - the system will transcribe your voice")
	fmt.Println("• Have a continuous conversation - context is maintained across turns")
	fmt.Println("• Responses are spoken back to you using text-to-speech")
	fmt.Println("• Simple queries are handled by GPT-4o-mini (cost-effective)")
	fmt.Println("• Complex tasks automatically escalate to GPT-4o")
	fmt.Println()
	fmt.Println("WARNING: Complex queries use GPT-4o which incurs higher costs!")
	fmt.Println()
	fmt.Println("Secret feature: There's a hidden word that triggers a special response!")
	fmt.Println()
	fmt.Println("Press Ctrl+C to stop")
	fmt.Println()
	fmt.Println("Listening for your voice input...")
	fmt.Println("-------------------------------------------")
}
