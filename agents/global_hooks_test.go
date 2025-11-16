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
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type RunHooksForTests struct {
	Events map[string]int
}

func NewRunHooksForTests() *RunHooksForTests {
	return &RunHooksForTests{
		Events: make(map[string]int),
	}
}

func (h *RunHooksForTests) Reset() {
	clear(h.Events)
}

func (*RunHooksForTests) OnLLMStart(context.Context, *agents.Agent, param.Opt[string], []agents.TResponseInputItem) error {
	return nil
}

func (*RunHooksForTests) OnLLMEnd(context.Context, *agents.Agent, agents.ModelResponse) error {
	return nil
}

func (h *RunHooksForTests) OnAgentStart(context.Context, *agents.Agent) error {
	h.Events["OnAgentStart"] += 1
	return nil
}

func (h *RunHooksForTests) OnAgentEnd(context.Context, *agents.Agent, any) error {
	h.Events["OnAgentEnd"] += 1
	return nil
}

func (h *RunHooksForTests) OnHandoff(context.Context, *agents.Agent, *agents.Agent) error {
	h.Events["OnHandoff"] += 1
	return nil
}

func (h *RunHooksForTests) OnToolStart(context.Context, *agents.Agent, agents.Tool) error {
	h.Events["OnToolStart"] += 1
	return nil
}

func (h *RunHooksForTests) OnToolEnd(context.Context, *agents.Agent, agents.Tool, any) error {
	h.Events["OnToolEnd"] += 1
	return nil
}

func TestNonStreamedRuntHooks(t *testing.T) {
	hooks := NewRunHooksForTests()
	model := agentstesting.NewFakeModel(false, nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent3 := &agents.Agent{
		Name:          "agent_3",
		Model:         param.NewOpt(agents.NewAgentModel(model)),
		AgentHandoffs: []*agents.Agent{agent1, agent2},
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
	}

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("user_message"),
		},
	})

	output, err := (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		Run(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	assert.Equal(t, map[string]int{"OnAgentStart": 1, "OnAgentEnd": 1}, hooks.Events, output)
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

	_, err = (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		Run(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	assert.Equal(t, map[string]int{
		// We only invoke OnAgentStart when we begin executing a new agent.
		// Although agent3 runs two turns internally before handing off,
		// that's one logical agent segment, so OnAgentStart fires once.
		// Then we hand off to agent1, so OnAgentStart fires for that agent.
		"OnAgentStart": 2,
		"OnToolStart":  1,
		"OnToolEnd":    1,
		"OnHandoff":    1,
		"OnAgentEnd":   1, // Should always have one end
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

	_, err = (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		Run(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	assert.Equal(t, map[string]int{
		// agent3 starts (fires OnAgentStart), runs two turns and hands off.
		// agent1 starts (fires OnAgentStart), then hands back to agent_3.
		// agent3 starts again (fires OnAgentStart) to complete execution.
		"OnAgentStart": 3,
		"OnToolStart":  2,
		"OnToolEnd":    2,
		"OnHandoff":    2,
		"OnAgentEnd":   1, // Should always have one end
	}, hooks.Events)
}

func TestStreamedRuntHooks(t *testing.T) {
	hooks := NewRunHooksForTests()
	model := agentstesting.NewFakeModel(false, nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent3 := &agents.Agent{
		Name:          "agent_3",
		Model:         param.NewOpt(agents.NewAgentModel(model)),
		AgentHandoffs: []*agents.Agent{agent1, agent2},
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
	}

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("user_message"),
		},
	})

	output, err := (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		RunStreamed(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	err = output.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, map[string]int{"OnAgentStart": 1, "OnAgentEnd": 1}, hooks.Events, output)
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

	output, err = (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		RunStreamed(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	err = output.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		// As in the non-streamed case above, two logical agent segments:
		// starting agent3, then handoff to agent1.
		"OnAgentStart": 2,
		"OnToolStart":  1,
		"OnToolEnd":    1,
		"OnHandoff":    1,
		"OnAgentEnd":   1, // Should always have one end
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

	output, err = (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		RunStreamed(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	err = output.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		// Same three logical agent segments as in the non-streamed case,
		// so OnAgentStart fires three times.
		"OnAgentStart": 3,
		"OnToolStart":  2,
		"OnToolEnd":    2,
		"OnHandoff":    2,
		"OnAgentEnd":   1, // Should always have one end
	}, hooks.Events)
}

func TestStructuredOutputNonStreamedRunHooks(t *testing.T) {
	type Foo struct {
		A string `json:"a"`
	}

	hooks := NewRunHooksForTests()
	model := agentstesting.NewFakeModel(false, nil)
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
		OutputType: agents.OutputType[Foo](),
	}

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetFinalOutputMessage(`{"a": "b"}`)},
	})
	output, err := (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		Run(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	assert.Equal(t, map[string]int{"OnAgentStart": 1, "OnAgentEnd": 1}, hooks.Events, output)
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
	_, err = (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		Run(t.Context(), agent3, "user_message")
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		// As with unstructured output, we expect on_agent_start once for
		// agent3 and once for agent1.
		"OnAgentStart": 2,
		"OnToolStart":  1,
		"OnToolEnd":    1,
		"OnHandoff":    1,
		"OnAgentEnd":   1, // Should always have one end
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
	_, err = (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		Run(t.Context(), agent3, "user_message")
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		// We still expect three logical agent segments, as before.
		"OnAgentStart": 3,
		"OnToolStart":  2,
		"OnToolEnd":    2,
		"OnHandoff":    2,
		"OnAgentEnd":   1, // Should always have one end
	}, hooks.Events)
}

func TestStructuredOutputStreamedRunHooks(t *testing.T) {
	type Foo struct {
		A string `json:"a"`
	}

	hooks := NewRunHooksForTests()
	model := agentstesting.NewFakeModel(false, nil)
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
		OutputType: agents.OutputType[Foo](),
	}

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetFinalOutputMessage(`{"a": "b"}`)},
	})

	output, err := (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		RunStreamed(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	err = output.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, map[string]int{"OnAgentStart": 1, "OnAgentEnd": 1}, hooks.Events, output)
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

	output, err = (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		RunStreamed(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	err = output.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		// Two agent segments: agent3 and then agent1.
		"OnAgentStart": 2,
		"OnToolStart":  1,
		"OnToolEnd":    1,
		"OnHandoff":    1,
		"OnAgentEnd":   1, // Should always have one end
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

	output, err = (agents.Runner{Config: agents.RunConfig{Hooks: hooks}}).
		RunStreamed(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	err = output.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, map[string]int{
		// Three agent segments: agent3, agent1, agent3 again.
		"OnAgentStart": 3,
		"OnToolStart":  2,
		"OnToolEnd":    2,
		"OnHandoff":    2,
		"OnAgentEnd":   1, // Should always have one end
	}, hooks.Events)
}
