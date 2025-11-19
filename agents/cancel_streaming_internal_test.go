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
	"errors"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCancelCleansUpResources(t *testing.T) {
	agent := &Agent{
		Name:  "Joker",
		Model: param.NewOpt(NewAgentModel(FakeModel{})),
	}

	result, err := Runner{}.RunStreamed(t.Context(), agent, "Please tell me 5 jokes.")
	require.NoError(t, err)

	// Start streaming, then cancel
	_ = result.StreamEvents(func(event StreamEvent) error {
		result.Cancel()
		time.Sleep(100 * time.Millisecond)
		return nil
	})

	// After cancel, queues should be empty and is_complete True
	assert.True(t, result.IsComplete())
	assert.True(t, result.eventQueue.IsEmpty())
	assert.True(t, result.inputGuardrailQueue.IsEmpty())
}

type FakeModel struct{}

func (m FakeModel) GetResponse(context.Context, ModelResponseParams) (*ModelResponse, error) {
	return nil, errors.New("FakeModel.GetResponse not implemented")
}

func (m FakeModel) StreamResponse(context.Context, ModelResponseParams, ModelStreamResponseCallback) error {
	return errors.New("FakeModel.StreamResponse not implemented")
}
