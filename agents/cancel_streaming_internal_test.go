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
	"iter"
	"testing"

	"github.com/openai/openai-go/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCancelCleansUpResources(t *testing.T) {
	agent := &Agent{
		Name:  "Joker",
		Model: param.NewOpt(NewAgentModel(FakeModel{})),
	}

	result, err := Runner().RunStreamed(t.Context(), RunStreamedParams{
		StartingAgent: agent,
		Input:         InputString("Please tell me 5 jokes."),
	})
	require.NoError(t, err)

	stopErr := errors.New("stop")

	// Start streaming, then cancel
	err = result.StreamEvents(func(event StreamEvent) error {
		result.Cancel()
		return stopErr
	})

	require.ErrorIs(t, err, stopErr)

	// After cancel, queues should be empty and is_complete True
	assert.True(t, result.IsComplete)
	assert.True(t, result.eventQueue.IsEmpty())
	assert.True(t, result.inputGuardrailQueue.IsEmpty())
}

type FakeModel struct{}

func (m FakeModel) GetResponse(ctx context.Context, params ModelGetResponseParams) (*ModelResponse, error) {
	return nil, errors.New("not implemented")
}

func (m FakeModel) StreamResponse(ctx context.Context, params ModelStreamResponseParams) (iter.Seq2[*TResponseStreamEvent, error], error) {
	return nil, errors.New("not implemented")
}
