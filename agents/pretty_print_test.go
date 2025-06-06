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
	"encoding/json"
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPrettyResult(t *testing.T) {
	model := agentstesting.NewFakeModel(&agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Hi there"),
		},
	})
	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("Hello"),
	})
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
	model := agentstesting.NewFakeModel(&agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Hi there"),
		},
	})
	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent,
		Input:         agents.InputString("Hello"),
	})
	require.NoError(t, err)
	err = result.StreamEvents(func(event agents.StreamEvent) error { return nil })
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

type PrettyPrintTestFoo struct {
	Bar string `json:"bar"`
}

type PrettyPrintTestFooSchema struct{}

func (PrettyPrintTestFooSchema) Name() string             { return "Foo" }
func (PrettyPrintTestFooSchema) IsPlainText() bool        { return false }
func (PrettyPrintTestFooSchema) IsStrictJSONSchema() bool { return true }
func (PrettyPrintTestFooSchema) JSONSchema() map[string]any {
	return map[string]any{
		"title":                "Foo",
		"type":                 "object",
		"required":             []string{"bar"},
		"additionalProperties": false,
		"properties": map[string]any{
			"bar": map[string]any{
				"title": "Bar",
				"type":  "string",
			},
		},
	}
}
func (PrettyPrintTestFooSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var v PrettyPrintTestFoo
	err := dec.Decode(&v)
	return v, err
}

func TestPrettyRunResultStructuredOutput(t *testing.T) {
	model := agentstesting.NewFakeModel(&agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Test"),
			agentstesting.GetFinalOutputMessage(`{"bar": "Hi there"}`),
		},
	})
	agent := &agents.Agent{
		Name:         "test_agent",
		Model:        param.NewOpt(agents.NewAgentModel(model)),
		OutputSchema: PrettyPrintTestFooSchema{},
	}
	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("Hello"),
	})
	require.NoError(t, err)
	require.NotNil(t, result)

	v := agents.PrettyPrintResult(*result)
	assert.Equal(t, `RunResult:
- Last agent: Agent(name="test_agent", ...)
- Final output (agents_test.PrettyPrintTestFoo):
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
	model := agentstesting.NewFakeModel(&agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Test"),
			agentstesting.GetFinalOutputMessage(`{"bar": "Hi there"}`),
		},
	})
	agent := &agents.Agent{
		Name:         "test_agent",
		Model:        param.NewOpt(agents.NewAgentModel(model)),
		OutputSchema: PrettyPrintTestFooSchema{},
	}

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent,
		Input:         agents.InputString("Hello"),
	})
	require.NoError(t, err)
	err = result.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	v := agents.PrettyPrintRunResultStreaming(*result)
	assert.Equal(t, `RunResultStreaming:
- Current agent: Agent(name="test_agent", ...)
- Current turn: 1
- Max turns: 10
- Is complete: true
- Final output (agents_test.PrettyPrintTestFoo):
    {
      "bar": "Hi there"
    }
- 2 new item(s)
- 1 raw response(s)
- 0 input guardrail result(s)
- 0 output guardrail result(s)
(See `+"`RunResultStreaming`"+` for more details)`, v)
}
