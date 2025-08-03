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

		var audioInputErr error
		go func() {
			audioInputErr = UsingAudioInputStream(func(stream *AudioInputStream) error {
				for {
					select {
					case <-ctx.Done():
						return nil
					default:
						data, err := stream.Read()
						if err != nil {
							return err
						}
						audioInput.AddAudio(agents.AudioDataInt16(slices.Clone(data)))
					}
				}
			})
		}()

		result, err := pipeline.Run(ctx, audioInput)
		if err != nil {
			return err
		}
		stream := result.Stream(ctx)

		err = UsingAudioPlayer(func(player *AudioPlayer) error {
			for event := range stream.Seq() {
				switch e := event.(type) {
				case agents.VoiceStreamEventAudio:
					err = player.AddAudio(e.Data.Int16())
					if err != nil {
						return err
					}
					fmt.Printf("Received audio: %d bytes\n", e.Data.Len()*2)
				case agents.VoiceStreamEventLifecycle:
					fmt.Printf("Lifecycle event: %v\n", e.Event)
				}
			}
			return stream.Error()
		})

		return errors.Join(err, audioInputErr)
	})
	if err != nil {
		panic(err)
	}
}
