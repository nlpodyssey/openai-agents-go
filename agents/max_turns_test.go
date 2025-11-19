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
	"fmt"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNonStreamedMaxTurns(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
	}

	for i := range 5 {
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage(fmt.Sprintf("%d", i)),
				agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
			},
		})
	}

	_, err := agents.Runner{Config: agents.RunConfig{MaxTurns: 3}}.Run(t.Context(), agent, "user_message")
	assert.ErrorAs(t, err, &agents.MaxTurnsExceededError{})
}

func TestStreamedMaxTurns(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
	}

	for i := range 5 {
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage(fmt.Sprintf("%d", i)),
				agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
			},
		})
	}

	result, err := agents.Runner{Config: agents.RunConfig{MaxTurns: 3}}.
		RunStreamed(t.Context(), agent, "user_message")
	require.NoError(t, err)

	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	assert.ErrorAs(t, err, &agents.MaxTurnsExceededError{})
}

func TestStructuredOutputNonStreamedMaxTurns(t *testing.T) {
	type Foo struct {
		A string `json:"a"`
	}

	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:       "test_1",
		Model:      param.NewOpt(agents.NewAgentModel(model)),
		OutputType: agents.OutputType[Foo](),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("tool_1", "result"),
		},
	}

	for range 5 {
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetFunctionToolCall("tool_1", ""),
			},
		})
	}

	_, err := agents.Runner{Config: agents.RunConfig{MaxTurns: 3}}.Run(t.Context(), agent, "user_message")
	assert.ErrorAs(t, err, &agents.MaxTurnsExceededError{})
}

func TestStructuredOutputStreamedMaxTurns(t *testing.T) {
	type Foo struct {
		A string `json:"a"`
	}

	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:       "test_1",
		Model:      param.NewOpt(agents.NewAgentModel(model)),
		OutputType: agents.OutputType[Foo](),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("tool_1", "result"),
		},
	}

	for range 5 {
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetFunctionToolCall("tool_1", ""),
			},
		})
	}

	result, err := agents.Runner{Config: agents.RunConfig{MaxTurns: 3}}.
		RunStreamed(t.Context(), agent, "user_message")
	require.NoError(t, err)

	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	assert.ErrorAs(t, err, &agents.MaxTurnsExceededError{})
}
