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
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
)

func Test_runner_getModel(t *testing.T) {
	t.Run("no prefix is OpenAI", func(t *testing.T) {
		agent := &Agent{
			Name:  "test",
			Model: param.NewOpt(NewAgentModelName("gpt-4o")),
		}
		model, err := Runner{}.getModel(agent, RunConfig{})
		assert.NoError(t, err)
		assert.IsType(t, OpenAIResponsesModel{}, model)
		assert.Equal(t, "gpt-4o", model.(OpenAIResponsesModel).Model)
	})

	t.Run("OpenAI prefix", func(t *testing.T) {
		agent := &Agent{
			Name:  "test",
			Model: param.NewOpt(NewAgentModelName("openai/gpt-4o")),
		}
		model, err := Runner{}.getModel(agent, RunConfig{})
		assert.NoError(t, err)
		assert.IsType(t, OpenAIResponsesModel{}, model)
		assert.Equal(t, "gpt-4o", model.(OpenAIResponsesModel).Model)
	})
}

type panicModel struct{}

func (m panicModel) GetResponse(context.Context, ModelResponseParams) (*ModelResponse, error) {
	panic("this is a test panic")
}

func (m panicModel) StreamResponse(context.Context, ModelResponseParams, ModelStreamResponseCallback) error {
	panic("this is a test panic")
}

func TestStreamHandlesPanicInModel(t *testing.T) {
	agent := &Agent{
		Name:  "test",
		Model: param.NewOpt(NewAgentModel(panicModel{})),
	}

	result, err := Runner{}.RunStreamed(context.Background(), agent, "test input")
	assert.NoError(t, err)

	// Use a timeout to ensure we don't hang forever
	done := make(chan struct{})
	var streamErr error

	go func() {
		defer close(done)
		streamErr = result.StreamEvents(func(event StreamEvent) error {
			return nil
		})
	}()

	select {
	case <-done:
		// Test passed - stream completed
		assert.Error(t, streamErr)
		assert.True(t, strings.Contains(streamErr.Error(), "this is a test panic"))
	case <-time.After(2 * time.Second):
		t.Fatal("Stream blocked forever - panic was not properly handled")
	}
}

