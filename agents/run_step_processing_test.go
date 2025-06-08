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
	"testing"

	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmptyResponse(t *testing.T) {
	agent := &Agent{Name: "test"}
	response := ModelResponse{
		Output:     nil,
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result, err := RunImpl().ProcessModelResponse(
		agent,
		nil,
		response,
		nil,
	)
	require.NoError(t, err)

	assert.Nil(t, result.Handoffs)
	assert.Nil(t, result.Functions)
}

func TestNoToolCalls(t *testing.T) {
	agent := &Agent{Name: "test"}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result, err := RunImpl().ProcessModelResponse(
		agent,
		nil,
		response,
		nil,
	)
	require.NoError(t, err)

	assert.Nil(t, result.Handoffs)
	assert.Nil(t, result.Functions)
}

func TestSingleToolCall(t *testing.T) {
	agent := &Agent{
		Name: "test",
		Tools: []Tool{
			getFunctionTool("test", ""),
		},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getFunctionToolCall("test", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result, err := RunImpl().ProcessModelResponse(
		agent,
		agent.GetAllTools(),
		response,
		nil,
	)
	require.NoError(t, err)

	assert.Nil(t, result.Handoffs)

	require.Len(t, result.Functions, 1)
	fn := result.Functions[0]
	assert.Equal(t, "test", fn.ToolCall.Name)
	assert.Equal(t, "", fn.ToolCall.Arguments)
}

func TestMissingToolCallRaisesError(t *testing.T) {
	agent := &Agent{
		Name: "test",
		Tools: []Tool{
			getFunctionTool("test", ""),
		},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getFunctionToolCall("missing", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	_, err := RunImpl().ProcessModelResponse(
		agent,
		agent.GetAllTools(),
		response,
		nil,
	)
	var target ModelBehaviorError
	assert.ErrorAs(t, err, &target)
}

func TestRunStepProcessingMultipleToolCalls(t *testing.T) {
	agent := &Agent{
		Name: "test",
		Tools: []Tool{
			getFunctionTool("test_1", ""),
			getFunctionTool("test_2", ""),
			getFunctionTool("test_3", ""),
		},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getFunctionToolCall("test_1", "abc"),
			getFunctionToolCall("test_2", "xyz"),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result, err := RunImpl().ProcessModelResponse(
		agent,
		agent.GetAllTools(),
		response,
		nil,
	)
	require.NoError(t, err)

	assert.Nil(t, result.Handoffs)

	require.Len(t, result.Functions, 2)

	func1 := result.Functions[0]
	assert.Equal(t, "test_1", func1.ToolCall.Name)
	assert.Equal(t, "abc", func1.ToolCall.Arguments)

	func2 := result.Functions[1]
	assert.Equal(t, "test_2", func2.ToolCall.Name)
	assert.Equal(t, "xyz", func2.ToolCall.Arguments)
}

func TestHandoffsParsedCorrectly(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:     "test_3",
		Handoffs: []AgentHandoff{agent1, agent2},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result, err := RunImpl().ProcessModelResponse(
		agent3,
		agent3.GetAllTools(),
		response,
		nil,
	)
	require.NoError(t, err)
	assert.Nil(t, result.Handoffs, "shouldn't have a handoff here")

	response = ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getHandoffToolCall(agent1, "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	runnerHandoffs, err := Runner().getHandoffs(agent3)
	require.NoError(t, err)
	result, err = RunImpl().ProcessModelResponse(
		agent3,
		agent3.GetAllTools(),
		response,
		runnerHandoffs,
	)
	require.NoError(t, err)

	require.Len(t, result.Handoffs, 1)
	handoff := result.Handoffs[0]
	assert.Equal(t, DefaultHandoffToolName(agent1), handoff.Handoff.ToolName)
	assert.Equal(t, DefaultHandoffToolDescription(agent1), handoff.Handoff.ToolDescription)
	assert.Equal(t, "test_1", handoff.Handoff.AgentName)

	handoffAgent, err := handoff.Handoff.OnInvokeHandoff(
		t.Context(),
		runcontext.NewWrapper(nil),
		handoff.ToolCall.Arguments,
	)
	require.NoError(t, err)
	assert.Same(t, agent1, handoffAgent)
}

func TestMissingHandoffFails(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:     "test_3",
		Handoffs: []AgentHandoff{agent1},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getHandoffToolCall(agent2, "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	runnerHandoffs, err := Runner().getHandoffs(agent3)
	require.NoError(t, err)
	_, err = RunImpl().ProcessModelResponse(
		agent3,
		agent3.GetAllTools(),
		response,
		runnerHandoffs,
	)
	var target ModelBehaviorError
	assert.ErrorAs(t, err, &target)
}

func TestMultipleHandoffsDoesntError(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:     "test_3",
		Handoffs: []AgentHandoff{agent1, agent2},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getHandoffToolCall(agent1, "", ""),
			getHandoffToolCall(agent2, "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	runnerHandoffs, err := Runner().getHandoffs(agent3)
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		agent3,
		agent3.GetAllTools(),
		response,
		runnerHandoffs,
	)
	require.NoError(t, err)
	assert.Len(t, result.Handoffs, 2)
}

func TestReasoningItemParsedCorrectly(t *testing.T) {
	// Verify that a Reasoning output item is converted into a ReasoningItem.
	agent := &Agent{Name: "test"}
	reasoning := TResponseOutputItem{
		ID:   "r1",
		Type: "reasoning",
		Summary: []responses.ResponseReasoningItemSummary{
			{
				Text: "why",
				Type: constant.ValueOf[constant.SummaryText](),
			},
		},
	}
	response := ModelResponse{
		Output:     []TResponseOutputItem{reasoning},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result, err := RunImpl().ProcessModelResponse(
		agent,
		agent.GetAllTools(),
		response,
		nil,
	)
	require.NoError(t, err)

	require.Len(t, result.NewItems, 1)

	item := result.NewItems[0]
	require.IsType(t, ReasoningItem{}, item)
	assert.Equal(t, responses.ResponseReasoningItem{
		ID: "r1",
		Summary: []responses.ResponseReasoningItemSummary{
			{
				Text: "why",
				Type: constant.ValueOf[constant.SummaryText](),
			},
		},
		Type:   constant.ValueOf[constant.Reasoning](),
		Status: "",
	}, item.(ReasoningItem).RawItem)
}

func TestToolAndHandoffParsedCorrectly(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name: "test_3",
		Tools: []Tool{
			getFunctionTool("test", ""),
		},
		Handoffs: []AgentHandoff{agent1, agent2},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getFunctionToolCall("test", "abc"),
			getHandoffToolCall(agent1, "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	runnerHandoffs, err := Runner().getHandoffs(agent3)
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		agent3,
		agent3.GetAllTools(),
		response,
		runnerHandoffs,
	)
	require.NoError(t, err)

	assert.Len(t, result.Functions, 1)
	require.Len(t, result.Handoffs, 1)

	handoff := result.Handoffs[0]
	assert.Equal(t, DefaultHandoffToolName(agent1), handoff.Handoff.ToolName)
	assert.Equal(t, DefaultHandoffToolDescription(agent1), handoff.Handoff.ToolDescription)
	assert.Equal(t, "test_1", handoff.Handoff.AgentName)
}
