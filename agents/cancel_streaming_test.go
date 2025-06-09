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

package agents_test

import (
	"errors"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/openai/openai-go/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleStreamingWithCancel(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "Joker",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent,
		Input:         agents.InputString("Please tell me 5 jokes."),
	})
	require.NoError(t, err)

	numEvents := 0
	const stopAfter = 1 // There are two that the model gives back.

	err = result.StreamEvents(func(event agents.StreamEvent) error {
		numEvents += 1
		if numEvents == stopAfter {
			result.Cancel()
		}
		return nil
	})

	var target agents.CanceledError
	require.ErrorAs(t, err, &target)

	assert.Equal(t, stopAfter, numEvents)
}

func TestMultipleEventsStreamingWithCancel(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "Joker",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []tools.Tool{
			agentstesting.GetFunctionTool("foo", "tool_result"),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		// Second turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent,
		Input:         agents.InputString("Please tell me 5 jokes."),
	})
	require.NoError(t, err)

	numEvents := 0
	const stopAfter = 2

	err = result.StreamEvents(func(event agents.StreamEvent) error {
		numEvents += 1
		if numEvents == stopAfter {
			result.Cancel()
		}
		return nil
	})

	var target agents.CanceledError
	require.ErrorAs(t, err, &target)

	assert.Equal(t, stopAfter, numEvents)
}

func TestCancelPreventsFurtherEvents(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "Joker",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent,
		Input:         agents.InputString("Please tell me 5 jokes."),
	})
	require.NoError(t, err)

	stopErr := errors.New("stop")

	var events []agents.StreamEvent
	err = result.StreamEvents(func(event agents.StreamEvent) error {
		events = append(events, event)
		// Cancel after the first event
		result.Cancel()
		return stopErr
	})

	require.ErrorIs(t, err, stopErr)

	//  Try to get more events after cancel
	var moreEvents []agents.StreamEvent
	err = result.StreamEvents(func(event agents.StreamEvent) error {
		events = append(events, event)
		return nil
	})

	var target agents.CanceledError
	require.ErrorAs(t, err, &target)

	assert.Len(t, events, 1)
	assert.Empty(t, moreEvents)
}

func TestCancelIsIdempotent(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "Joker",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent,
		Input:         agents.InputString("Please tell me 5 jokes."),
	})
	require.NoError(t, err)

	stopErr := errors.New("stop")

	var events []agents.StreamEvent
	err = result.StreamEvents(func(event agents.StreamEvent) error {
		events = append(events, event)
		// Cancel after the first event
		result.Cancel()
		result.Cancel() // Call cancel again
		return stopErr
	})

	require.ErrorIs(t, err, stopErr)
	assert.Len(t, events, 1)
}

func TestCancelBeforeStreaming(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "Joker",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent,
		Input:         agents.InputString("Please tell me 5 jokes."),
	})
	require.NoError(t, err)

	result.Cancel() // Cancel before streaming

	var events []agents.StreamEvent
	err = result.StreamEvents(func(event agents.StreamEvent) error {
		events = append(events, event)
		return nil
	})

	var target agents.CanceledError
	require.ErrorAs(t, err, &target)

	assert.Empty(t, events)
}
