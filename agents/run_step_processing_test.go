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
	"testing"

	"github.com/nlpodyssey/openai-agents-go/computer"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
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
		t.Context(),
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
		t.Context(),
		agent,
		nil,
		response,
		nil,
	)
	require.NoError(t, err)

	assert.Nil(t, result.Handoffs)
	assert.Nil(t, result.Functions)
}

func TestProcessModelResponsePreservesMCPSubunions(t *testing.T) {
	rawItems := []string{
		`{"id":"approval","type":"mcp_approval_request","arguments":"{\"city\":\"Tokyo\"}","name":"weather","server_label":"server"}`,
		`{"id":"list","type":"mcp_list_tools","server_label":"server","tools":[{"name":"weather","description":"Weather","input_schema":{"type":"object"},"annotations":null}],"error":null}`,
		`{"id":"call","type":"mcp_call","arguments":"{\"city\":\"Tokyo\"}","name":"weather","server_label":"server","output":"sunny","error":null,"status":"completed"}`,
	}
	output := make([]TResponseOutputItem, len(rawItems))
	for index, raw := range rawItems {
		require.NoError(t, json.Unmarshal([]byte(raw), &output[index]))
	}
	agent := &Agent{Name: "test", Tools: []Tool{HostedMCPTool{
		ToolConfig: responses.ToolMcpParam{ServerLabel: "server"},
	}}}
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(), agent, allTools,
		ModelResponse{Output: output, Usage: usage.NewUsage()}, nil,
	)
	require.NoError(t, err)
	require.Len(t, result.NewItems, 3)
	approval, ok := result.NewItems[0].(MCPApprovalRequestItem)
	require.True(t, ok)
	require.Equal(t, `{"city":"Tokyo"}`, approval.RawItem.Arguments)
	list, ok := result.NewItems[1].(MCPListToolsItem)
	require.True(t, ok)
	require.Len(t, list.RawItem.Tools, 1)
	require.Equal(t, "weather", list.RawItem.Tools[0].Name)
	call, ok := result.NewItems[2].(ToolCallItem)
	require.True(t, ok)
	mcpCall, ok := call.RawItem.(ResponseOutputItemMcpCall)
	require.True(t, ok)
	require.Equal(t, `{"city":"Tokyo"}`, mcpCall.Arguments)
}

func TestProcessModelResponseRejectsNonStringToolFields(t *testing.T) {
	tests := []struct {
		name        string
		rawItem     string
		agent       *Agent
		errorSubstr string
	}{
		{
			name:        "function call arguments",
			rawItem:     `{"id":"call","type":"function_call","arguments":{"city":"Tokyo"},"call_id":"call","name":"test","status":"completed"}`,
			agent:       &Agent{Name: "test", Tools: []Tool{getFunctionTool("test", "")}},
			errorSubstr: "function_call arguments must be a string",
		},
		{
			name:        "MCP approval request arguments",
			rawItem:     `{"id":"approval","type":"mcp_approval_request","arguments":{"city":"Tokyo"},"name":"weather","server_label":"server"}`,
			agent:       &Agent{Name: "test", Tools: []Tool{HostedMCPTool{ToolConfig: responses.ToolMcpParam{ServerLabel: "server"}}}},
			errorSubstr: "mcp_approval_request arguments must be a string",
		},
		{
			name:        "MCP call arguments",
			rawItem:     `{"id":"call","type":"mcp_call","arguments":{"city":"Tokyo"},"name":"weather","server_label":"server","output":"sunny","status":"completed"}`,
			agent:       &Agent{Name: "test"},
			errorSubstr: "mcp_call arguments must be a string",
		},
		{
			name:        "MCP call output",
			rawItem:     `{"id":"call","type":"mcp_call","arguments":"{}","name":"weather","server_label":"server","output":[{"type":"output_text","text":"sunny"}],"status":"completed"}`,
			agent:       &Agent{Name: "test"},
			errorSubstr: "mcp_call output must be a string",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var output TResponseOutputItem
			require.NoError(t, json.Unmarshal([]byte(tt.rawItem), &output))
			allTools, err := tt.agent.GetAllTools(t.Context())
			require.NoError(t, err)

			result, err := RunImpl().ProcessModelResponse(
				t.Context(), tt.agent, allTools,
				ModelResponse{Output: []TResponseOutputItem{output}, Usage: usage.NewUsage()}, nil,
			)

			require.Nil(t, result)
			assert.ErrorAs(t, err, &ModelBehaviorError{})
			assert.ErrorContains(t, err, tt.errorSubstr)
		})
	}
}

func TestProcessModelResponseRejectsDirectNonStringArguments(t *testing.T) {
	tests := []struct {
		name      string
		arguments any
	}{
		{name: "map", arguments: map[string]string{"city": "Tokyo"}},
		{name: "struct", arguments: struct{ City string }{City: "Tokyo"}},
		{name: "slice", arguments: []string{"Tokyo"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent := &Agent{Name: "test", Tools: []Tool{getFunctionTool("test", "")}}
			allTools, err := agent.GetAllTools(t.Context())
			require.NoError(t, err)
			output := TResponseOutputItem{
				ID:     "call",
				Type:   "function_call",
				CallID: "call",
				Name:   "test",
				Arguments: responses.ResponseOutputItemUnionArguments{
					OfResponseToolSearchCallArguments: tt.arguments,
				},
			}

			result, err := RunImpl().ProcessModelResponse(
				t.Context(), agent, allTools,
				ModelResponse{Output: []TResponseOutputItem{output}, Usage: usage.NewUsage()}, nil,
			)

			require.Nil(t, result)
			assert.ErrorAs(t, err, &ModelBehaviorError{})
			assert.ErrorContains(t, err, "function_call arguments must be a string")
		})
	}
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
			getFunctionToolCall("test", "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent,
		allTools,
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

func TestMissingToolCallReturnsError(t *testing.T) {
	agent := &Agent{
		Name: "test",
		Tools: []Tool{
			getFunctionTool("test", ""),
		},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getFunctionToolCall("missing", "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	_, err = RunImpl().ProcessModelResponse(
		t.Context(),
		agent,
		allTools,
		response,
		nil,
	)
	assert.ErrorAs(t, err, &ModelBehaviorError{})
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
			getFunctionToolCall("test_1", "abc", ""),
			getFunctionToolCall("test_2", "xyz", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent,
		allTools,
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
		Name:          "test_3",
		AgentHandoffs: []*Agent{agent1, agent2},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	allTools, err := agent3.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent3,
		allTools,
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
	runnerHandoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
	require.NoError(t, err)
	allTools, err = agent3.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err = RunImpl().ProcessModelResponse(
		t.Context(),
		agent3,
		allTools,
		response,
		runnerHandoffs,
	)
	require.NoError(t, err)

	require.Len(t, result.Handoffs, 1)
	handoff := result.Handoffs[0]
	assert.Equal(t, DefaultHandoffToolName(agent1), handoff.Handoff.ToolName)
	assert.Equal(t, DefaultHandoffToolDescription(agent1), handoff.Handoff.ToolDescription)
	assert.Equal(t, "test_1", handoff.Handoff.AgentName)

	handoffAgent, err := handoff.Handoff.OnInvokeHandoff(t.Context(), handoff.ToolCall.Arguments)
	require.NoError(t, err)
	assert.Same(t, agent1, handoffAgent)
}

func TestMissingHandoffFails(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:          "test_3",
		AgentHandoffs: []*Agent{agent1},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getHandoffToolCall(agent2, "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	runnerHandoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
	require.NoError(t, err)
	allTools, err := agent3.GetAllTools(t.Context())
	require.NoError(t, err)
	_, err = RunImpl().ProcessModelResponse(
		t.Context(),
		agent3,
		allTools,
		response,
		runnerHandoffs,
	)
	assert.ErrorAs(t, err, &ModelBehaviorError{})
}

func TestMultipleHandoffsDoesntError(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:          "test_3",
		AgentHandoffs: []*Agent{agent1, agent2},
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
	runnerHandoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
	require.NoError(t, err)
	allTools, err := agent3.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent3,
		allTools,
		response,
		runnerHandoffs,
	)
	require.NoError(t, err)
	assert.Len(t, result.Handoffs, 2)
}

func TestFileSearchToolCallParsedCorrectly(t *testing.T) {
	// Ensure that a ResponseFileSearchToolCall output is parsed into a ToolCallItem and that no tool
	// runs are scheduled.

	agent := &Agent{Name: "test"}
	fileSearchCall := TResponseOutputItem{
		ID:      "fs1",
		Queries: []string{"query"},
		Status:  "completed",
		Type:    "file_search_call",
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("hello"),
			fileSearchCall,
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent,
		allTools,
		response,
		nil,
	)
	require.NoError(t, err)
	require.Len(t, result.NewItems, 2)

	// The final item should be a ToolCallItem for the file search call
	item := result.NewItems[1]
	require.IsType(t, ToolCallItem{}, item)
	assert.Equal(t, ResponseFileSearchToolCall{
		ID:      "fs1",
		Queries: []string{"query"},
		Status:  responses.ResponseFileSearchToolCallStatusCompleted,
		Type:    constant.ValueOf[constant.FileSearchCall](),
		Results: nil,
	}, item.(ToolCallItem).RawItem)

	assert.Empty(t, result.Functions)
	assert.Empty(t, result.Handoffs)
}

func TestFunctionWebSearchToolCallParsedCorrectly(t *testing.T) {
	agent := &Agent{Name: "test"}
	webSearchCall := TResponseOutputItem{ // responses.ResponseFunctionWebSearch
		ID: "w1",
		Action: responses.ResponseOutputItemUnionAction{
			Type:  "search",
			Query: "query",
		},
		Status: "completed",
		Type:   "web_search_call",
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("hello"),
			webSearchCall,
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent,
		allTools,
		response,
		nil,
	)
	require.NoError(t, err)
	require.Len(t, result.NewItems, 2)

	item := result.NewItems[1]
	require.IsType(t, ToolCallItem{}, item)
	assert.Equal(t, ResponseFunctionWebSearch{
		ID: "w1",
		Action: responses.ResponseFunctionWebSearchActionUnion{
			Type:  "search",
			Query: "query",
		},
		Status: responses.ResponseFunctionWebSearchStatusCompleted,
		Type:   constant.ValueOf[constant.WebSearchCall](),
	}, item.(ToolCallItem).RawItem)

	assert.Empty(t, result.Functions)
	assert.Empty(t, result.Handoffs)
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
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent,
		allTools,
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

// DummyComputer is a minimal computer.Computer implementation for testing.
type DummyComputer struct{}

func (DummyComputer) Environment(context.Context) (computer.Environment, error) {
	return computer.EnvironmentLinux, nil
}
func (DummyComputer) Dimensions(context.Context) (computer.Dimensions, error) {
	return computer.Dimensions{Width: 0, Height: 0}, nil
}
func (DummyComputer) Screenshot(context.Context) (string, error)                 { return "", nil }
func (DummyComputer) Click(context.Context, int64, int64, computer.Button) error { return nil }
func (DummyComputer) DoubleClick(context.Context, int64, int64) error            { return nil }
func (DummyComputer) Scroll(context.Context, int64, int64, int64, int64) error   { return nil }
func (DummyComputer) Type(context.Context, string) error                         { return nil }
func (DummyComputer) Wait(context.Context) error                                 { return nil }
func (DummyComputer) Move(context.Context, int64, int64) error                   { return nil }
func (DummyComputer) Keypress(context.Context, []string) error                   { return nil }
func (DummyComputer) Drag(context.Context, []computer.Position) error            { return nil }

func TestComputerToolCallWithoutComputerToolReturnsError(t *testing.T) {
	// If the agent has no tools.ComputerTool in its tools, ProcessModelResponse should return a
	// ModelBehaviorError when encountering a ResponseComputerToolCall.
	agent := &Agent{Name: "test"}
	computerCall := TResponseOutputItem{ // responses.ResponseComputerToolCall
		ID:   "c1",
		Type: "computer_call",
		Action: responses.ResponseOutputItemUnionAction{
			Type: "click", X: 1, Y: 2, Button: "left",
		},
		CallID:              "c1",
		PendingSafetyChecks: nil,
		Status:              "completed",
	}
	response := ModelResponse{
		Output:     []TResponseOutputItem{computerCall},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	_, err = RunImpl().ProcessModelResponse(
		t.Context(),
		agent,
		allTools,
		response,
		nil,
	)
	assert.ErrorAs(t, err, &ModelBehaviorError{})
}

func TestComputerToolCallWithComputerToolParsedCorrectly(t *testing.T) {
	// If the agent contains a tools.ComputerTool, ensure that a ResponseComputerToolCall is parsed into a
	// ToolCallItem and scheduled to run in computer_actions.
	dummyComputer := DummyComputer{}
	computerTool := ComputerTool{Computer: dummyComputer}
	agent := &Agent{
		Name:  "test",
		Tools: []Tool{computerTool},
	}
	computerCall := TResponseOutputItem{ // responses.ResponseComputerToolCall
		ID:   "c1",
		Type: "computer_call",
		Action: responses.ResponseOutputItemUnionAction{
			Type: "click", X: 1, Y: 2, Button: "left",
		},
		CallID:              "c1",
		PendingSafetyChecks: nil,
		Status:              "completed",
	}
	response := ModelResponse{
		Output:     []TResponseOutputItem{computerCall},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	allTools, err := agent.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent,
		allTools,
		response,
		nil,
	)
	require.NoError(t, err)
	assert.Equal(t, []ToolRunComputerAction{
		{
			ToolCall: responses.ResponseComputerToolCall{
				ID: "c1",
				Action: responses.ResponseComputerToolCallActionUnion{
					Type: "click", X: 1, Y: 2, Button: "left",
				},
				CallID:              "c1",
				PendingSafetyChecks: nil,
				Status:              responses.ResponseComputerToolCallStatusCompleted,
				Type:                responses.ResponseComputerToolCallTypeComputerCall,
			},
			ComputerTool: computerTool,
		},
	}, result.ComputerActions)
}

func TestToolAndHandoffParsedCorrectly(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name: "test_3",
		Tools: []Tool{
			getFunctionTool("test", ""),
		},
		AgentHandoffs: []*Agent{agent1, agent2},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getFunctionToolCall("test", "abc", ""),
			getHandoffToolCall(agent1, "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	runnerHandoffs, err := Runner{}.getHandoffs(t.Context(), agent3)
	require.NoError(t, err)
	allTools, err := agent3.GetAllTools(t.Context())
	require.NoError(t, err)
	result, err := RunImpl().ProcessModelResponse(
		t.Context(),
		agent3,
		allTools,
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
