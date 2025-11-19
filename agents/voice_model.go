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
	"context"
	"iter"

	"github.com/openai/openai-go/v3/packages/param"
)

const DefaultTTSInstructions = "You will receive partial sentences. Do not complete the sentence, just read out the text."

type TTSVoice string

const (
	TTSVoiceAlloy   TTSVoice = "alloy"
	TTSVoiceAsh     TTSVoice = "ash"
	TTSVoiceCoral   TTSVoice = "coral"
	TTSVoiceEcho    TTSVoice = "echo"
	TTSVoiceFable   TTSVoice = "fable"
	TTSVoiceOnyx    TTSVoice = "onyx"
	TTSVoiceNova    TTSVoice = "nova"
	TTSVoiceSage    TTSVoice = "sage"
	TTSVoiceShimmer TTSVoice = "shimmer"
)

// TTSModelSettings provides settings for a TTS model.
type TTSModelSettings struct {
	// Optional voice to use for the TTS model.
	// If not provided, the default voice for the respective model will be used.
	Voice TTSVoice

	// Optional minimal size of the chunks of audio data that are being streamed out.
	// Default: 120.
	BufferSize int

	// Optional data type for the audio data to be returned in.
	// Default: AudioDataTypeInt16
	AudioDataType param.Opt[AudioDataType]

	// Optional function to transform the data from the TTS model.
	TransformData func(context.Context, AudioData) (AudioData, error)

	// Optional instructions to use for the TTS model.
	// This is useful if you want to control the tone of the audio output.
	// Default: DefaultTTSInstructions.
	Instructions param.Opt[string]

	// Optional function to split the text into chunks. This is useful if you want to split the text into
	// chunks before sending it to the TTS model rather than waiting for the whole text to be
	// processed.
	// Default: GetTTSSentenceBasedSplitter(20)
	TextSplitter TTSTextSplitterFunc

	// Optional speed with which the TTS model will read the text. Between 0.25 and 4.0.
	Speed param.Opt[float64]
}

// TTSTextSplitterFunc is a function to split the text into chunks.
// This is useful if you want to split the text into chunks before sending it
// to the TTS model rather than waiting for the whole text to be processed.
//
// It accepts the text to split and returns the text to process and the
// remaining text buffer.
type TTSTextSplitterFunc = func(textBuffer string) (textToProcess, remainingText string, err error)

// TTSModel interface is implemented by a text-to-speech model that can
// convert text into audio output.
type TTSModel interface {
	// ModelName returns the name of the TTS model.
	ModelName() string

	// Run accepts a text string and produces a stream of audio bytes, in PCM format.
	Run(ctx context.Context, text string, settings TTSModelSettings) TTSModelRunResult
}

type TTSModelRunResult interface {
	Seq() iter.Seq[[]byte]
	Error() error
}

// StreamedTranscriptionSession is a streamed transcription of audio input.
type StreamedTranscriptionSession interface {
	// TranscribeTurns yields a stream of text transcriptions.
	// Each transcription is a turn in the conversation.
	// This method is expected to return only after Close() is called.
	TranscribeTurns(ctx context.Context) StreamedTranscriptionSessionTranscribeTurns

	// Close the session.
	Close(ctx context.Context) error
}

type StreamedTranscriptionSessionTranscribeTurns interface {
	Seq() iter.Seq[string]
	Error() error
}

// STTModelSettings provides settings for a speech-to-text model.
type STTModelSettings struct {
	// Optional instructions for the model to follow.
	Prompt param.Opt[string]

	// Optional language of the audio input.
	Language param.Opt[string]

	// The temperature of the model.
	Temperature param.Opt[float64]

	// Optional turn detection settings for the model when using streamed audio input.
	TurnDetection map[string]any
}

// STTModel interface is implemented by a speech-to-text model that can
// convert audio input into text.
type STTModel interface {
	// ModelName returns the name of the STT model.
	ModelName() string

	// Transcribe accepts an audio input and produces a text transcription.
	Transcribe(ctx context.Context, params STTModelTranscribeParams) (string, error)

	// CreateSession creates a new transcription session, which you can push
	// audio to, and receive a stream of text transcriptions.
	CreateSession(ctx context.Context, params STTModelCreateSessionParams) (StreamedTranscriptionSession, error)
}

type STTModelTranscribeParams struct {
	// The audio input to transcribe.
	Input AudioInput
	// The settings to use for the transcription.
	Settings STTModelSettings
	// Whether to include sensitive data in traces.
	TraceIncludeSensitiveData bool
	// Whether to include sensitive audio data in traces.
	TraceIncludeSensitiveAudioData bool
}

type STTModelCreateSessionParams struct {
	// The audio input to transcribe.
	Input StreamedAudioInput
	// The settings to use for the transcription.
	Settings STTModelSettings
	// Whether to include sensitive data in traces.
	TraceIncludeSensitiveData bool
	// Whether to include sensitive audio data in traces.
	TraceIncludeSensitiveAudioData bool
}

// VoiceModelProvider is the base interface for a voice model provider.
//
// A model provider is responsible for creating speech-to-text and
// text-to-speech models, given a name.
type VoiceModelProvider interface {
	// GetSTTModel returns a speech-to-text model by name.
	GetSTTModel(modelName string) (STTModel, error)

	// GetTTSModel returns a text-to-speech model by name.
	GetTTSModel(modelName string) (TTSModel, error)
}
