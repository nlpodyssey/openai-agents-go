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
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sync/atomic"
	"time"

	"github.com/gordonklaus/portaudio"
)

const (
	SampleRate = 24000
	Channels   = 1
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

type AudioPlayer struct {
	out              []int16
	remainder        []int16
	stream           *portaudio.Stream
	started          bool
	lastWriteTime    atomic.Int64 // unix nano timestamp of last write
	lastChunkSamples atomic.Int64 // samples in the last chunk written
}

type AudioInputStream struct {
	data    []int16
	stream  *portaudio.Stream
	started bool
}

func UsingAudioInputStream(fn func(stream *AudioInputStream) error) (err error) {
	s := &AudioInputStream{
		data: make([]int16, SampleRate*0.02),
	}
	stream, err := portaudio.OpenDefaultStream(Channels, 0, SampleRate, len(s.data), &s.data)
	if err != nil {
		return fmt.Errorf("error opening audio stream: %w", err)
	}
	s.stream = stream
	defer func() {
		if s.started {
			if e := s.stream.Close(); e != nil {
				err = errors.Join(err, fmt.Errorf("error closing audio stream: %w", e))
			}
		}
	}()

	return fn(s)
}

func (s *AudioInputStream) Read() ([]int16, error) {
	if !s.started {
		if err := s.stream.Start(); err != nil {
			return nil, fmt.Errorf("error starting audio stream: %w", err)
		}
		s.started = true
	}
	if err := s.stream.Read(); err != nil {
		return nil, fmt.Errorf("error reading audio stream: %w", err)
	}
	return s.data, nil
}

func UsingAudioPlayer(fn func(*AudioPlayer) error) (err error) {
	out := make([]int16, 8192)
	stream, err := portaudio.OpenDefaultStream(0, Channels, SampleRate, len(out), &out)
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

	// Track when we write this chunk and how many samples it contains
	ap.lastWriteTime.Store(time.Now().UnixNano())
	ap.lastChunkSamples.Store(int64(len(buffer)))

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
				// Handle audio underflow gracefully - don't panic
				if errors.Is(err, portaudio.OutputUnderflowed) {
					slog.Debug("Audio output underflowed", slog.String("error", err.Error()))
					continue
				}
				return fmt.Errorf("error writing audio stream: %w", err)
			}
			// Update timing for this chunk write
			ap.lastWriteTime.Store(time.Now().UnixNano())
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
			// Handle audio underflow gracefully - don't panic
			if errors.Is(err, portaudio.OutputUnderflowed) {
				slog.Debug("Audio output underflowed", slog.String("error", err.Error()))
			} else {
				return fmt.Errorf("error writing remaining audio stream: %w", err)
			}
		}

		// Track final remainder flush
		ap.lastWriteTime.Store(time.Now().UnixNano())
		ap.lastChunkSamples.Store(int64(len(ap.remainder)))
		ap.remainder = ap.remainder[:0]
	}
	return nil
}

// HasOutputFinished checks if all audio data has been consumed by the output device
func (ap *AudioPlayer) HasOutputFinished() bool {
	if !ap.started {
		return true
	}

	info := ap.stream.Info()
	lastWrite := time.Unix(0, ap.lastWriteTime.Load())
	lastChunkSamples := ap.lastChunkSamples.Load()

	// If no samples written yet, we're done
	if lastChunkSamples == 0 {
		return true
	}

	// If no last write time recorded, we're done
	if ap.lastWriteTime.Load() == 0 {
		return true
	}

	// Calculate the expected duration for ONLY the last chunk written
	lastChunkDuration := time.Duration(lastChunkSamples) * time.Second / SampleRate
	outputLatency := info.OutputLatency
	totalExpectedTime := lastChunkDuration + outputLatency
	timeSinceLastWrite := time.Since(lastWrite)

	return timeSinceLastWrite >= totalExpectedTime
}

// ResetPlaybackTracking resets the sample tracking for a new turn
func (ap *AudioPlayer) ResetPlaybackTracking() {
	ap.lastChunkSamples.Store(0)
	ap.lastWriteTime.Store(0)
}
