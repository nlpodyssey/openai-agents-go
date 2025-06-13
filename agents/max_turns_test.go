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
	"fmt"
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/openai/openai-go/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNonStreamedMaxTurns(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []tools.Tool{
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

	_, err := agents.Runner{Config: agents.RunConfig{MaxTurns: 3}}.Run(
		t.Context(), agent, "user_message")

	var target agents.MaxTurnsExceededError
	assert.ErrorAs(t, err, &target)
}

func TestStreamedMaxTurns(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []tools.Tool{
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

	err = result.StreamEvents(func(event agents.StreamEvent) error { return nil })
	var target agents.MaxTurnsExceededError
	assert.ErrorAs(t, err, &target)
}

type MaxTurnsTestFoo struct {
	A string `json:"a"`
}

type MaxTurnsTestFooSchema struct{}

func (MaxTurnsTestFooSchema) Name() string             { return "Foo" }
func (MaxTurnsTestFooSchema) IsPlainText() bool        { return false }
func (MaxTurnsTestFooSchema) IsStrictJSONSchema() bool { return true }
func (MaxTurnsTestFooSchema) JSONSchema() map[string]any {
	return map[string]any{
		"title":                "Foo",
		"type":                 "object",
		"required":             []string{"a"},
		"additionalProperties": false,
		"properties": map[string]any{
			"a": map[string]any{
				"title": "A",
				"type":  "string",
			},
		},
	}
}
func (MaxTurnsTestFooSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var v MaxTurnsTestFoo
	err := dec.Decode(&v)
	return v, err
}

func TestStructuredOutputNonStreamedMaxTurns(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:         "test_1",
		Model:        param.NewOpt(agents.NewAgentModel(model)),
		OutputSchema: MaxTurnsTestFooSchema{},
		Tools: []tools.Tool{
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

	_, err := agents.Runner{Config: agents.RunConfig{MaxTurns: 3}}.Run(
		t.Context(), agent, "user_message")

	var target agents.MaxTurnsExceededError
	assert.ErrorAs(t, err, &target)
}

func TestStructuredOutputStreamedMaxTurns(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:         "test_1",
		Model:        param.NewOpt(agents.NewAgentModel(model)),
		OutputSchema: MaxTurnsTestFooSchema{},
		Tools: []tools.Tool{
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

	err = result.StreamEvents(func(event agents.StreamEvent) error { return nil })
	var target agents.MaxTurnsExceededError
	assert.ErrorAs(t, err, &target)
}
