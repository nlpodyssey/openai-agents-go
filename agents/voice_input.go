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

package agents

import (
	"encoding/base64"
	"fmt"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/nlpodyssey/openai-agents-go/asyncqueue"
	"github.com/nlpodyssey/openai-agents-go/util"
)

const (
	DefaultAudioSampleRate  = 24000
	DefaultAudioSampleWidth = 2
	DefaultAudioChannels    = 1
)

type AudioFile struct {
	Filename    string
	ContentType string
	Content     []byte
}

func bufferToAudioFile(buffer AudioData, sampleRate, sampleWidth, channels int) (*AudioFile, error) {
	if sampleRate <= 0 {
		sampleRate = DefaultAudioSampleRate
	}
	if sampleWidth <= 0 {
		sampleWidth = DefaultAudioSampleWidth
	}
	if channels <= 0 {
		channels = DefaultAudioChannels
	}

	var wavBuf util.WriteSeekerBuffer
	enc := wav.NewEncoder(
		&wavBuf,
		sampleRate,
		8*sampleWidth,
		channels,
		1, // PCM
	)

	intData := buffer.Int()

	err := enc.Write(&audio.IntBuffer{
		Format: &audio.Format{
			NumChannels: channels,
			SampleRate:  sampleRate,
		},
		Data:           intData,
		SourceBitDepth: 8 * sampleWidth,
	})
	if err != nil {
		return nil, fmt.Errorf("error writing WAV file: %w", err)
	}

	if err = enc.Close(); err != nil {
		return nil, fmt.Errorf("error closing WAV file: %w", err)
	}

	return &AudioFile{
		Filename:    "audio.wav",
		ContentType: "audio/wav",
		Content:     wavBuf.Bytes(),
	}, nil
}

// AudioInput represents static audio to be used as input for the VoicePipeline.
type AudioInput struct {
	// A buffer containing the audio data for the agent.
	Buffer AudioData

	// Optional sample rate of the audio data. Defaults to DefaultAudioSampleRate.
	SampleRate int

	// Optional sample width of the audio data. Defaults to DefaultAudioSampleWidth.
	SampleWidth int

	// Optional number of channels in the audio data. Defaults to DefaultAudioChannels.
	Channels int
}

func (ai AudioInput) ToAudioFile() (*AudioFile, error) {
	return bufferToAudioFile(ai.Buffer, ai.SampleRate, ai.SampleWidth, ai.Channels)
}

// ToBase64 returns the audio data as a base64 encoded string.
func (ai AudioInput) ToBase64() string {
	return base64.StdEncoding.EncodeToString(ai.Buffer.Int16().Bytes())
}

// StreamedAudioInput is an audio input represented as a stream of audio data.
// You can pass this to the VoicePipeline and then push audio data into the
// queue using the AddAudio method.
type StreamedAudioInput struct {
	Queue *asyncqueue.Queue[AudioData]
}

func NewStreamedAudioInput() StreamedAudioInput {
	return StreamedAudioInput{
		Queue: asyncqueue.New[AudioData](),
	}
}

// AddAudio adds more audio data to the stream.
func (s StreamedAudioInput) AddAudio(audio AudioData) {
	s.Queue.Put(audio)
}
