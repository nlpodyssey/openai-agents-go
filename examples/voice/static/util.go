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
	"strings"

	"github.com/gordonklaus/portaudio"
)

func UsingPortaudio(fn func() error) (err error) {
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
	out       []int16
	remainder []int16
	stream    *portaudio.Stream
	started   bool
}

func UsingAudioPlayer(fn func(*AudioPlayer) error) (err error) {
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

	ap := &AudioPlayer{
		out:       out,
		remainder: make([]int16, 0, len(out)),
		stream:    stream,
		started:   false,
	}

	defer func() {
		if e := ap.Flush(); e != nil {
			err = errors.Join(err, fmt.Errorf("error flushing audio player: %w", e))
		}
		if ap.started {
			if e := stream.Stop(); e != nil {
				err = errors.Join(err, fmt.Errorf("error stopping audio stream: %w", e))
			}
		}
	}()

	return fn(ap)
}

func (ap *AudioPlayer) AddAudio(buffer []int16) error {
	if len(buffer) == 0 {
		return nil
	}

	// Start stream on first audio data
	if !ap.started {
		if err := ap.stream.Start(); err != nil {
			return fmt.Errorf("error starting audio stream: %w", err)
		}
		ap.started = true
	}

	stream := ap.stream
	out := ap.out

	// Combine any remainder from previous calls with new buffer
	if len(ap.remainder) > 0 {
		buffer = append(ap.remainder, buffer...)
		ap.remainder = ap.remainder[:0]
	}

	for chunk := range slices.Chunk(buffer, len(out)) {
		if len(chunk) == len(out) {
			copy(out, chunk)
			if err := stream.Write(); err != nil {
				return fmt.Errorf("error writing audio stream: %w", err)
			}
		} else {
			// Store partial chunk for next call
			ap.remainder = ap.remainder[:len(chunk)]
			copy(ap.remainder, chunk)
		}
	}
	return nil
}

func (ap *AudioPlayer) Flush() error {
	if len(ap.remainder) > 0 && ap.started {
		// Pad remainder with zeros to fill buffer
		copy(ap.out[:len(ap.remainder)], ap.remainder)
		clear(ap.out[len(ap.remainder):])

		if err := ap.stream.Write(); err != nil {
			return fmt.Errorf("error writing remaining audio stream: %w", err)
		}
		ap.remainder = ap.remainder[:0]
	}
	return nil
}

func checkForQuit() (input string, quit bool, err error) {
	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	if err != nil {
		return "", false, err
	}

	input = strings.TrimSpace(line)
	quit = strings.ToLower(input) == "quit" || strings.ToLower(input) == "exit"

	return input, quit, nil
}
