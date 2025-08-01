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
	"bufio"
	"errors"
	"fmt"
	"os"
	"slices"

	"github.com/gordonklaus/portaudio"
)

func usingPortaudio(fn func() error) (err error) {
	if err = portaudio.Initialize(); err != nil {
		return fmt.Errorf("error initializing portaudio: %w", err)
	}
	defer func() {
		if e := portaudio.Terminate(); e != nil {
			err = errors.Join(err, fmt.Errorf("error terminating portaudio: %w", e))
		}
	}()
	return fn()
}

func recordAudio() (_ []float32, err error) {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("Press <enter> to start recording. Press <enter> again to stop recording.")
	if _, _, err = reader.ReadRune(); err != nil {
		return nil, err
	}

	in := make([]float32, 64)
	var buffer []float32
	stream, err := portaudio.OpenDefaultStream(1, 0, 24000, len(in), in)
	if err != nil {
		return nil, fmt.Errorf("error opening audio stream: %w", err)
	}
	defer func() {
		if e := stream.Close(); e != nil {
			err = errors.Join(err, fmt.Errorf("error closing audio stream: %w", e))
		}
	}()

	if err = stream.Start(); err != nil {
		return nil, fmt.Errorf("error starting audio stream: %w", err)
	}
	fmt.Println("Recording started...")

	keyPressChan := make(chan struct{})
	var readlineErr error
	go func() {
		_, _, readlineErr = reader.ReadRune()
		close(keyPressChan)
	}()

loop:
	for {
		select {
		case <-keyPressChan:
			break loop
		default:
			err = stream.Read()
			if err != nil {
				err = fmt.Errorf("error reading audio stream: %w", err)
				fmt.Println(err.Error())
				<-keyPressChan
				return nil, err
			}
			buffer = append(buffer, in...)
		}
	}

	if readlineErr != nil {
		return nil, readlineErr
	}
	if err = stream.Stop(); err != nil {
		return nil, fmt.Errorf("error stopping audio stream: %w", err)
	}

	fmt.Println("Recording stopped.")

	return buffer, nil
}

type AudioPlayer struct {
	out    []int16
	stream *portaudio.Stream
}

func usingAudioPlayer(fn func(*AudioPlayer) error) (err error) {
	out := make([]int16, 8192)
	stream, err := portaudio.OpenDefaultStream(0, 1, 24000, len(out), &out)
	if err != nil {
		return fmt.Errorf("error opening audio stream: %w", err)
	}
	defer func() {
		if e := stream.Close(); e != nil {
			err = errors.Join(err, fmt.Errorf("error closing audio stream: %w", e))
		}
	}()

	if err = stream.Start(); err != nil {
		return fmt.Errorf("error starting audio stream: %w", err)
	}
	defer func() {
		if e := stream.Stop(); e != nil {
			err = errors.Join(err, fmt.Errorf("error stopping audio stream: %w", e))
		}
	}()

	ap := &AudioPlayer{
		out:    out,
		stream: stream,
	}
	return fn(ap)
}

func (ap *AudioPlayer) AddAudio(buffer []int16) error {
	stream := ap.stream
	out := ap.out
	for chunk := range slices.Chunk(buffer, len(out)) {
		copy(out[:len(chunk)], chunk)
		clear(out[len(chunk):])
		if err := stream.Write(); err != nil {
			return fmt.Errorf("error writing audio stream: %w", err)
		}
	}
	return nil
}
