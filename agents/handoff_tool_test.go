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
	"encoding/json"
	"errors"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSingleHandoffSetup(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{
		Name:          "test_2",
		AgentHandoffs: []*Agent{agent1},
	}

	handoffs, err := Runner{}.getHandoffs(t.Context(), agent1)
	require.NoError(t, err)
	assert.Len(t, handoffs, 0)

	handoffs, err = Runner{}.getHandoffs(t.Context(), agent2)
	require.NoError(t, err)
	require.Len(t, handoffs, 1)

	obj := handoffs[0]
	assert.Equal(t, DefaultHandoffToolName(agent1), obj.ToolName)
	assert.Equal(t, DefaultHandoffToolDescription(agent1), obj.ToolDescription)
	assert.Equal(t, "test_1", obj.AgentName)
}

func TestMultipleHandoffsSetup(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:          "test_3",
		AgentHandoffs: []*Agent{agent1, agent2},
	}

	handoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
	require.NoError(t, err)
	require.Len(t, handoffs, 2)

	assert.Equal(t, DefaultHandoffToolName(agent1), handoffs[0].ToolName)
	assert.Equal(t, DefaultHandoffToolName(agent2), handoffs[1].ToolName)

	assert.Equal(t, DefaultHandoffToolDescription(agent1), handoffs[0].ToolDescription)
	assert.Equal(t, DefaultHandoffToolDescription(agent2), handoffs[1].ToolDescription)

	assert.Equal(t, "test_1", handoffs[0].AgentName)
	assert.Equal(t, "test_2", handoffs[1].AgentName)
}

func TestCustomHandoffSetup(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name: "test_3",
		AgentHandoffs: []*Agent{
			agent1,
		},
		Handoffs: []Handoff{
			HandoffFromAgent(HandoffFromAgentParams{
				Agent:                   agent2,
				ToolNameOverride:        "custom_tool_name",
				ToolDescriptionOverride: "custom tool description",
			}),
		},
	}

	handoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
	require.NoError(t, err)
	require.Len(t, handoffs, 2)

	assert.Equal(t, "custom_tool_name", handoffs[0].ToolName)
	assert.Equal(t, DefaultHandoffToolName(agent1), handoffs[1].ToolName)

	assert.Equal(t, "custom tool description", handoffs[0].ToolDescription)
	assert.Equal(t, DefaultHandoffToolDescription(agent1), handoffs[1].ToolDescription)

	assert.Equal(t, "test_2", handoffs[0].AgentName)
	assert.Equal(t, "test_1", handoffs[1].AgentName)
}

type HandoffToolTestFoo struct {
	Bar string `json:"bar"`
}

func TestHandoffInputType(t *testing.T) {
	onHandoff := func(context.Context, any) error {
		return nil
	}

	schema, err := OutputType[HandoffToolTestFoo]().JSONSchema()
	require.NoError(t, err)

	agent := &Agent{Name: "test"}
	obj, err := SafeHandoffFromAgent(HandoffFromAgentParams{
		Agent:           agent,
		OnHandoff:       OnHandoffWithInput(onHandoff),
		InputJSONSchema: schema,
	})
	require.NoError(t, err)

	// Invalid JSON should return an error
	_, err = obj.OnInvokeHandoff(t.Context(), "not json")
	require.Error(t, err)

	// Empty JSON should return an error
	_, err = obj.OnInvokeHandoff(t.Context(), "")
	require.Error(t, err)

	// Valid JSON should call the OnHandoff function
	invoked, err := obj.OnInvokeHandoff(t.Context(), `{"bar": "baz"}`)
	require.NoError(t, err)
	assert.Same(t, agent, invoked)
}

func TestOnHandoffCalled(t *testing.T) {
	wasCalled := false

	onHandoff := func(context.Context, any) error {
		wasCalled = true
		return nil
	}

	schema, err := OutputType[HandoffToolTestFoo]().JSONSchema()
	require.NoError(t, err)

	agent := &Agent{Name: "test"}
	obj, err := SafeHandoffFromAgent(HandoffFromAgentParams{
		Agent:           agent,
		OnHandoff:       OnHandoffWithInput(onHandoff),
		InputJSONSchema: schema,
	})
	require.NoError(t, err)

	// Valid JSON should call the OnHandoff function
	invoked, err := obj.OnInvokeHandoff(t.Context(), `{"bar": "baz"}`)
	require.NoError(t, err)
	assert.Same(t, agent, invoked)
	assert.True(t, wasCalled)
}

func TestOnHandoffError(t *testing.T) {
	handoffErr := errors.New("error")

	onHandoff := func(context.Context, any) error {
		return handoffErr
	}

	schema, err := OutputType[HandoffToolTestFoo]().JSONSchema()
	require.NoError(t, err)

	agent := &Agent{Name: "test"}
	obj, err := SafeHandoffFromAgent(HandoffFromAgentParams{
		Agent:           agent,
		OnHandoff:       OnHandoffWithInput(onHandoff),
		InputJSONSchema: schema,
	})
	require.NoError(t, err)

	// Valid JSON should call the OnHandoff function
	_, err = obj.OnInvokeHandoff(t.Context(), `{"bar": "baz"}`)
	assert.ErrorIs(t, err, handoffErr)
}

func TestOnHandoffWithoutInputCalled(t *testing.T) {
	wasCalled := false

	onHandoff := func(context.Context) error {
		wasCalled = true
		return nil
	}

	agent := &Agent{Name: "test"}
	obj, err := SafeHandoffFromAgent(HandoffFromAgentParams{
		Agent:     agent,
		OnHandoff: OnHandoffWithoutInput(onHandoff),
	})
	require.NoError(t, err)

	// Valid JSON should call the OnHandoff function
	invoked, err := obj.OnInvokeHandoff(t.Context(), "")
	require.NoError(t, err)
	assert.Same(t, agent, invoked)
	assert.True(t, wasCalled)
}

func TestOnHandoffWithoutInputError(t *testing.T) {
	handoffErr := errors.New("error")

	onHandoff := func(context.Context) error {
		return handoffErr
	}

	agent := &Agent{Name: "test"}
	obj, err := SafeHandoffFromAgent(HandoffFromAgentParams{
		Agent:     agent,
		OnHandoff: OnHandoffWithoutInput(onHandoff),
	})
	require.NoError(t, err)

	// Valid JSON should call the OnHandoff function
	_, err = obj.OnInvokeHandoff(t.Context(), `{"bar": "baz"}`)
	assert.ErrorIs(t, err, handoffErr)
}

func TestHandoffInputSchemaIsStrict(t *testing.T) {
	schema, err := OutputType[HandoffToolTestFoo]().JSONSchema()
	require.NoError(t, err)

	agent := &Agent{Name: "test"}
	obj, err := SafeHandoffFromAgent(HandoffFromAgentParams{
		Agent:           agent,
		InputJSONSchema: schema,
		OnHandoff: OnHandoffWithInput(func(context.Context, any) error {
			return nil
		}),
	})
	require.NoError(t, err)
	assert.Equal(t, param.NewOpt(true), obj.StrictJSONSchema)
}

func TestGetTransferMessageIsValidJson(t *testing.T) {
	agent := &Agent{Name: "foo"}
	obj, err := SafeHandoffFromAgent(HandoffFromAgentParams{Agent: agent})
	require.NoError(t, err)
	transfer := obj.GetTransferMessage(agent)

	var m map[string]any
	err = json.Unmarshal([]byte(transfer), &m)
	require.NoError(t, err)

	assert.Equal(t, map[string]any{"assistant": "foo"}, m)
}

func TestHandoff_IsEnabled(t *testing.T) {
	t.Run("enabled by default", func(t *testing.T) {
		agent := New("test")
		h := HandoffFromAgent(HandoffFromAgentParams{Agent: agent})
		isEnabled, err := h.IsEnabled.IsEnabled(t.Context(), agent)
		require.NoError(t, err)
		assert.True(t, isEnabled)
	})

	t.Run("explicitly enabled", func(t *testing.T) {
		agent := New("test")
		h := HandoffFromAgent(HandoffFromAgentParams{
			Agent:     agent,
			IsEnabled: HandoffEnabled(),
		})
		isEnabled, err := h.IsEnabled.IsEnabled(t.Context(), agent)
		require.NoError(t, err)
		assert.True(t, isEnabled)
	})

	t.Run("explicitly disabled", func(t *testing.T) {
		agent := New("test")
		h := HandoffFromAgent(HandoffFromAgentParams{
			Agent:     agent,
			IsEnabled: HandoffDisabled(),
		})
		isEnabled, err := h.IsEnabled.IsEnabled(t.Context(), agent)
		require.NoError(t, err)
		assert.False(t, isEnabled)
	})

	t.Run("enabler function returns true", func(t *testing.T) {
		agent := New("test")
		h := HandoffFromAgent(HandoffFromAgentParams{
			Agent: agent,
			IsEnabled: HandoffEnablerFunc(func(context.Context, *Agent) (bool, error) {
				return true, nil
			}),
		})
		isEnabled, err := h.IsEnabled.IsEnabled(t.Context(), agent)
		require.NoError(t, err)
		assert.True(t, isEnabled)
	})

	t.Run("enabler function returns false", func(t *testing.T) {
		agent := New("test")
		h := HandoffFromAgent(HandoffFromAgentParams{
			Agent: agent,
			IsEnabled: HandoffEnablerFunc(func(context.Context, *Agent) (bool, error) {
				return false, nil
			}),
		})
		isEnabled, err := h.IsEnabled.IsEnabled(t.Context(), agent)
		require.NoError(t, err)
		assert.False(t, isEnabled)
	})

	t.Run("agent filters handoffs", func(t *testing.T) {
		// Integration test to make sure that disabled handoffs are filtered out by the runner.
		agent1 := New("agent_1")
		agent2 := New("agent_2")
		agent3 := New("agent_3")
		agent4 := New("agent_4")

		mainAgent := New("main_agent").WithHandoffs(
			HandoffFromAgent(HandoffFromAgentParams{
				Agent:     agent1,
				IsEnabled: HandoffEnabled(),
			}),
			HandoffFromAgent(HandoffFromAgentParams{
				Agent:     agent2,
				IsEnabled: HandoffDisabled(),
			}),
			HandoffFromAgent(HandoffFromAgentParams{
				Agent: agent3,
				IsEnabled: HandoffEnablerFunc(func(context.Context, *Agent) (bool, error) {
					return true, nil
				}),
			}),
			HandoffFromAgent(HandoffFromAgentParams{
				Agent: agent4,
				IsEnabled: HandoffEnablerFunc(func(context.Context, *Agent) (bool, error) {
					return false, nil
				}),
			}),
		)

		filteredHandoffs, err := Runner{}.getHandoffs(t.Context(), mainAgent)
		require.NoError(t, err)
		assert.Len(t, filteredHandoffs, 2)

		agentNames := make(map[string]struct{})
		for _, h := range filteredHandoffs {
			agentNames[h.AgentName] = struct{}{}
		}
		assert.Contains(t, agentNames, "agent_1")
		assert.Contains(t, agentNames, "agent_3")
	})
}
