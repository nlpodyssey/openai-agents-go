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
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSystemInstructions(t *testing.T) {
	t.Run("StringInstructions", func(t *testing.T) {
		agent := &Agent{
			Name:         "test",
			Instructions: InstructionsStr("foo"),
		}
		prompt, err := agent.GetSystemPrompt(t.Context())
		require.NoError(t, err)
		assert.Equal(t, param.NewOpt("foo"), prompt)
	})

	t.Run("FunctionInstructions", func(t *testing.T) {
		agent := &Agent{
			Name: "test",
			Instructions: InstructionsFunc(
				func(context.Context, *Agent) (string, error) {
					return "bar", nil
				},
			),
		}
		prompt, err := agent.GetSystemPrompt(t.Context())
		require.NoError(t, err)
		assert.Equal(t, param.NewOpt("bar"), prompt)
	})
}

func TestHandoff(t *testing.T) {
	t.Run("with agents", func(t *testing.T) {
		agent1 := &Agent{Name: "agent_1"}
		agent2 := &Agent{Name: "agent_2"}
		agent3 := &Agent{
			Name:          "agent_3",
			AgentHandoffs: []*Agent{agent1, agent2},
		}

		handoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
		require.NoError(t, err)
		assert.Len(t, handoffs, 2)

		assert.Equal(t, "agent_1", handoffs[0].AgentName)
		assert.Equal(t, "agent_2", handoffs[1].AgentName)

		firstReturn, err := handoffs[0].OnInvokeHandoff(t.Context(), "")
		require.NoError(t, err)
		assert.Same(t, agent1, firstReturn)

		secondReturn, err := handoffs[1].OnInvokeHandoff(t.Context(), "")
		require.NoError(t, err)
		assert.Same(t, agent2, secondReturn)
	})

	t.Run("with handoff obj", func(t *testing.T) {
		agent1 := &Agent{Name: "agent_1"}
		agent2 := &Agent{Name: "agent_2"}
		agent3 := &Agent{
			Name: "agent_3",
			Handoffs: []Handoff{
				HandoffFromAgent(HandoffFromAgentParams{Agent: agent1}),
				HandoffFromAgent(HandoffFromAgentParams{
					Agent:                   agent2,
					ToolNameOverride:        "transfer_to_2",
					ToolDescriptionOverride: "description_2",
				}),
			},
		}

		handoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
		require.NoError(t, err)
		assert.Len(t, handoffs, 2)

		assert.Equal(t, "agent_1", handoffs[0].AgentName)
		assert.Equal(t, "agent_2", handoffs[1].AgentName)

		assert.Equal(t, DefaultHandoffToolName(agent1), handoffs[0].ToolName)
		assert.Equal(t, "transfer_to_2", handoffs[1].ToolName)

		assert.Equal(t, DefaultHandoffToolDescription(agent1), handoffs[0].ToolDescription)
		assert.Equal(t, "description_2", handoffs[1].ToolDescription)

		firstReturn, err := handoffs[0].OnInvokeHandoff(t.Context(), "")
		require.NoError(t, err)
		assert.Same(t, agent1, firstReturn)

		secondReturn, err := handoffs[1].OnInvokeHandoff(t.Context(), "")
		require.NoError(t, err)
		assert.Same(t, agent2, secondReturn)
	})

	t.Run("with handoff obj and agent", func(t *testing.T) {
		agent1 := &Agent{Name: "agent_1"}
		agent2 := &Agent{Name: "agent_2"}
		agent3 := &Agent{
			Name: "agent_3",
			Handoffs: []Handoff{
				HandoffFromAgent(HandoffFromAgentParams{Agent: agent1}),
			},
			AgentHandoffs: []*Agent{
				agent2,
			},
		}

		handoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
		require.NoError(t, err)
		assert.Len(t, handoffs, 2)

		assert.Equal(t, "agent_1", handoffs[0].AgentName)
		assert.Equal(t, "agent_2", handoffs[1].AgentName)

		assert.Equal(t, DefaultHandoffToolName(agent1), handoffs[0].ToolName)
		assert.Equal(t, DefaultHandoffToolName(agent2), handoffs[1].ToolName)

		assert.Equal(t, DefaultHandoffToolDescription(agent1), handoffs[0].ToolDescription)
		assert.Equal(t, DefaultHandoffToolDescription(agent2), handoffs[1].ToolDescription)

		firstReturn, err := handoffs[0].OnInvokeHandoff(t.Context(), "")
		require.NoError(t, err)
		assert.Same(t, agent1, firstReturn)

		secondReturn, err := handoffs[1].OnInvokeHandoff(t.Context(), "")
		require.NoError(t, err)
		assert.Same(t, agent2, secondReturn)
	})
}
