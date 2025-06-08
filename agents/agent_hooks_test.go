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
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/openai/openai-go/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type AgentHooksForTests struct {
	Events map[string]int
}

func NewAgentHooksForTests() *AgentHooksForTests {
	return &AgentHooksForTests{
		Events: make(map[string]int),
	}
}

func (h *AgentHooksForTests) Reset() {
	clear(h.Events)
}

func (h *AgentHooksForTests) OnStart(context.Context, *runcontext.Wrapper, *agents.Agent) error {
	h.Events["OnStart"] += 1
	return nil
}

func (h *AgentHooksForTests) OnEnd(context.Context, *runcontext.Wrapper, *agents.Agent, any) error {
	h.Events["OnEnd"] += 1
	return nil
}

func (h *AgentHooksForTests) OnHandoff(context.Context, *runcontext.Wrapper, *agents.Agent, *agents.Agent) error {
	h.Events["OnHandoff"] += 1
	return nil
}

func (h *AgentHooksForTests) OnToolStart(context.Context, *runcontext.Wrapper, *agents.Agent, agents.Tool) error {
	h.Events["OnToolStart"] += 1
	return nil
}

func (h *AgentHooksForTests) OnToolEnd(context.Context, *runcontext.Wrapper, *agents.Agent, agents.Tool, any) error {
	h.Events["OnToolEnd"] += 1
	return nil
}

func TestNonStreamedAgentHooks(t *testing.T) {
	hooks := NewAgentHooksForTests()
	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "test_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent3 := &agents.Agent{
		Name:          "test_3",
		Model:         param.NewOpt(agents.NewAgentModel(model)),
		AgentHandoffs: []*agents.Agent{agent1, agent2},
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
		Hooks: hooks,
	}

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("user_message")},
	})
	output, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	assert.Equal(t, map[string]int{"OnStart": 1, "OnEnd": 1}, hooks.Events, output)
	hooks.Reset()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})
	_, err = agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)

	// Shouldn't have OnEnd because it's not the last agent
	assert.Equal(t, map[string]int{
		"OnStart":     1,
		"OnToolStart": 1,
		"OnToolEnd":   1,
		"OnHandoff":   1,
	}, hooks.Events)
	hooks.Reset()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message, another tool call, and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: a message and a handoff back to the orig agent
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent3, "", ""),
		}},
		// Fourth turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})
	_, err = agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		"OnStart":     2,
		"OnToolStart": 2,
		"OnToolEnd":   2,
		"OnHandoff":   1,
		"OnEnd":       1, // Agent 3 is the last agent
	}, hooks.Events)
}

func TestStreamedAgentHooks(t *testing.T) {
	hooks := NewAgentHooksForTests()
	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "test_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent3 := &agents.Agent{
		Name:          "test_3",
		Model:         param.NewOpt(agents.NewAgentModel(model)),
		AgentHandoffs: []*agents.Agent{agent1, agent2},
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
		Hooks: hooks,
	}

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("user_message")},
	})
	output, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	err = output.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)
	assert.Equal(t, map[string]int{"OnStart": 1, "OnEnd": 1}, hooks.Events, output)
	hooks.Reset()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})
	output, err = agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	err = output.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	// Shouldn't have OnEnd because it's not the last agent
	assert.Equal(t, map[string]int{
		"OnStart":     1,
		"OnToolStart": 1,
		"OnToolEnd":   1,
		"OnHandoff":   1,
	}, hooks.Events)
	hooks.Reset()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message, another tool call, and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: a message and a handoff back to the orig agent
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent3, "", ""),
		}},
		// Fourth turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})
	output, err = agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	err = output.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		"OnStart":     2,
		"OnToolStart": 2,
		"OnToolEnd":   2,
		"OnHandoff":   1,
		"OnEnd":       1, // Agent 3 is the last agent
	}, hooks.Events)
}

type AgentHooksTestFoo struct {
	A string `json:"a"`
}

type AgentHooksTestFooSchema struct{}

func (AgentHooksTestFooSchema) Name() string             { return "Foo" }
func (AgentHooksTestFooSchema) IsPlainText() bool        { return false }
func (AgentHooksTestFooSchema) IsStrictJSONSchema() bool { return true }
func (AgentHooksTestFooSchema) JSONSchema() map[string]any {
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
func (AgentHooksTestFooSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var v AgentHooksTestFoo
	err := dec.Decode(&v)
	return v, err
}

func TestStructuredOutputNonStreamedAgentHooks(t *testing.T) {
	hooks := NewAgentHooksForTests()
	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "test_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent3 := &agents.Agent{
		Name:          "test_3",
		Model:         param.NewOpt(agents.NewAgentModel(model)),
		AgentHandoffs: []*agents.Agent{agent1, agent2},
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
		Hooks:        hooks,
		OutputSchema: AgentHooksTestFooSchema{},
	}

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetFinalOutputMessage(`{"a": "b"}`)},
	})
	output, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	assert.Equal(t, map[string]int{"OnStart": 1, "OnEnd": 1}, hooks.Events, output)
	hooks.Reset()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: end message (for agent 1)
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})
	_, err = agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)

	// Shouldn't have OnEnd because it's not the last agent
	assert.Equal(t, map[string]int{
		"OnStart":     1,
		"OnToolStart": 1,
		"OnToolEnd":   1,
		"OnHandoff":   1,
	}, hooks.Events)
	hooks.Reset()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message, another tool call, and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: a message and a handoff back to the orig agent
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent3, "", ""),
		}},
		// Fourth turn: end message (for agent 3)
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFinalOutputMessage(`{"a": "b"}`),
		}},
	})
	_, err = agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		"OnStart":     2,
		"OnToolStart": 2,
		"OnToolEnd":   2,
		"OnHandoff":   1,
		"OnEnd":       1, // Agent 3 is the last agent
	}, hooks.Events)
}

func TestStructuredOutputStreamedAgentHooks(t *testing.T) {
	hooks := NewAgentHooksForTests()
	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "test_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent3 := &agents.Agent{
		Name:          "test_3",
		Model:         param.NewOpt(agents.NewAgentModel(model)),
		AgentHandoffs: []*agents.Agent{agent1, agent2},
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
		Hooks:        hooks,
		OutputSchema: AgentHooksTestFooSchema{},
	}

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetFinalOutputMessage(`{"a": "b"}`)},
	})
	output, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	err = output.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)
	assert.Equal(t, map[string]int{"OnStart": 1, "OnEnd": 1}, hooks.Events, output)
	hooks.Reset()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: end message (for agent 1)
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})
	output, err = agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	err = output.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	// Shouldn't have OnEnd because it's not the last agent
	assert.Equal(t, map[string]int{
		"OnStart":     1,
		"OnToolStart": 1,
		"OnToolEnd":   1,
		"OnHandoff":   1,
	}, hooks.Events)
	hooks.Reset()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message, another tool call, and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: a message and a handoff back to the orig agent
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent3, "", ""),
		}},
		// Fourth turn: end message (for agent 3)
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFinalOutputMessage(`{"a": "b"}`),
		}},
	})
	output, err = agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	err = output.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		"OnStart":     2,
		"OnToolStart": 2,
		"OnToolEnd":   2,
		"OnHandoff":   1,
		"OnEnd":       1, // Agent 3 is the last agent
	}, hooks.Events)
}
