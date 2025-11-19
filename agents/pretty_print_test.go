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
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPrettyResult(t *testing.T) {
	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Hi there"),
		},
	})
	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	result, err := agents.Runner{}.Run(t.Context(), agent, "Hello")
	require.NoError(t, err)
	require.NotNil(t, result)

	v := agents.PrettyPrintResult(*result)
	assert.Equal(t, `RunResult:
- Last agent: Agent(name="test_agent", ...)
- Final output (string):
    Hi there
- 1 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `+"`RunResult`"+` for more details)`, v)
}

func TestPrettyRunResultStreaming(t *testing.T) {
	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Hi there"),
		},
	})
	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	v := agents.PrettyPrintRunResultStreaming(*result)
	assert.Equal(t, `RunResultStreaming:
- Current agent: Agent(name="test_agent", ...)
- Current turn: 1
- Max turns: 10
- Is complete: true
- Final output (string):
    Hi there
- 1 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `+"`RunResultStreaming`"+` for more details)`, v)
}

func TestPrettyRunResultStructuredOutput(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Test"),
			agentstesting.GetFinalOutputMessage(`{"bar": "Hi there"}`),
		},
	})
	agent := &agents.Agent{
		Name:       "test_agent",
		Model:      param.NewOpt(agents.NewAgentModel(model)),
		OutputType: agents.OutputType[Foo](),
	}
	result, err := agents.Runner{}.Run(t.Context(), agent, "Hello")
	require.NoError(t, err)
	require.NotNil(t, result)

	v := agents.PrettyPrintResult(*result)
	assert.Equal(t, `RunResult:
- Last agent: Agent(name="test_agent", ...)
- Final output (agents_test.Foo):
    {
      "bar": "Hi there"
    }
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `+"`RunResult`"+` for more details)`, v)
}

func TestPrettyRunResultStreamingStructuredOutput(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Test"),
			agentstesting.GetFinalOutputMessage(`{"bar": "Hi there"}`),
		},
	})
	agent := &agents.Agent{
		Name:       "test_agent",
		Model:      param.NewOpt(agents.NewAgentModel(model)),
		OutputType: agents.OutputType[Foo](),
	}

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	v := agents.PrettyPrintRunResultStreaming(*result)
	assert.Equal(t, `RunResultStreaming:
- Current agent: Agent(name="test_agent", ...)
- Current turn: 1
- Max turns: 10
- Is complete: true
- Final output (agents_test.Foo):
    {
      "bar": "Hi there"
    }
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `+"`RunResultStreaming`"+` for more details)`, v)
}

func TestPrettyRunResultSliceStructuredOutput(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Test"),
			agentstesting.GetFinalOutputMessage(`{
				"response": [
					{ "bar": "Hi there" },
					{ "bar": "Hi there 2" }
				]
			}`),
		},
	})
	agent := &agents.Agent{
		Name:       "test_agent",
		Model:      param.NewOpt(agents.NewAgentModel(model)),
		OutputType: agents.OutputType[[]Foo](),
	}
	result, err := agents.Runner{}.Run(t.Context(), agent, "Hello")
	require.NoError(t, err)
	require.NotNil(t, result)

	v := agents.PrettyPrintResult(*result)
	assert.Equal(t, `RunResult:
- Last agent: Agent(name="test_agent", ...)
- Final output ([]agents_test.Foo):
    [
      {
        "bar": "Hi there"
      },
      {
        "bar": "Hi there 2"
      }
    ]
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `+"`RunResult`"+` for more details)`, v)
}

func TestPrettyRunResultStreamingListStructuredOutput(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Test"),
			agentstesting.GetFinalOutputMessage(`{
				"response": [
					{ "bar": "Test" },
					{ "bar": "Test 2" }
				]
			}`),
		},
	})
	agent := &agents.Agent{
		Name:       "test_agent",
		Model:      param.NewOpt(agents.NewAgentModel(model)),
		OutputType: agents.OutputType[[]Foo](),
	}

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "Hello")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	v := agents.PrettyPrintRunResultStreaming(*result)
	assert.Equal(t, `RunResultStreaming:
- Current agent: Agent(name="test_agent", ...)
- Current turn: 1
- Max turns: 10
- Is complete: true
- Final output ([]agents_test.Foo):
    [
      {
        "bar": "Test"
      },
      {
        "bar": "Test 2"
      }
    ]
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `+"`RunResultStreaming`"+` for more details)`, v)
}
