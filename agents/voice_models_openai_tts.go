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
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"

	"github.com/openai/openai-go/v3"
)

const DefaultOpenAITTSModelVoice = openai.AudioSpeechNewParamsVoiceAsh

// OpenAITTSModel is a text-to-speech model for OpenAI.
type OpenAITTSModel struct {
	model  string
	client OpenaiClient
}

// NewOpenAITTSModel creates a new OpenAI text-to-speech model.
func NewOpenAITTSModel(modelName string, openAIClient OpenaiClient) *OpenAITTSModel {
	return &OpenAITTSModel{
		model:  modelName,
		client: openAIClient,
	}
}

func (m *OpenAITTSModel) ModelName() string {
	return m.model
}

func (m *OpenAITTSModel) Run(ctx context.Context, text string, settings TTSModelSettings) TTSModelRunResult {
	resp, err := m.client.Audio.Speech.New(ctx, openai.AudioSpeechNewParams{
		Model:          m.model,
		Voice:          cmp.Or(openai.AudioSpeechNewParamsVoice(settings.Voice), DefaultOpenAITTSModelVoice),
		Input:          text,
		Instructions:   settings.Instructions,
		Speed:          settings.Speed,
		ResponseFormat: openai.AudioSpeechNewParamsResponseFormatPCM,
		StreamFormat:   openai.AudioSpeechNewParamsStreamFormatAudio,
	})
	return &openAITTSModelRunResult{
		resp: resp,
		err:  err,
	}
}

type openAITTSModelRunResult struct {
	resp *http.Response
	err  error
}

func (r *openAITTSModelRunResult) Seq() iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		if r.err != nil {
			return
		}
		defer func() {
			if err := r.resp.Body.Close(); err != nil {
				r.err = errors.Join(r.err, fmt.Errorf("error closing response body: %w", err))
			}
		}()

		eof := false
		for !eof {
			chunk := make([]byte, 1024)
			n, err := r.resp.Body.Read(chunk)

			eof = errors.Is(err, io.EOF)
			if err != nil && !eof {
				r.err = fmt.Errorf("error reading response body: %w", err)
				break
			}

			if n > 0 {
				if !yield(chunk[:n]) {
					break
				}
			}
		}
	}
}

func (r *openAITTSModelRunResult) Error() error { return r.err }
