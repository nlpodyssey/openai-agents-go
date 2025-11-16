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

	"github.com/nlpodyssey/openai-agents-go/computer"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// LoggingComputer is a computer.Computer implementation that logs calls to
// its methods for verification in tests.
type LoggingComputer struct {
	calls            [][]any
	screenshotReturn string
}

func NewLoggingComputer(screenshotReturn string) *LoggingComputer {
	if screenshotReturn == "" {
		screenshotReturn = "screenshot"
	}
	return &LoggingComputer{
		screenshotReturn: screenshotReturn,
	}
}

func (lc *LoggingComputer) Environment(context.Context) (computer.Environment, error) {
	return computer.EnvironmentLinux, nil
}

func (lc *LoggingComputer) Dimensions(context.Context) (computer.Dimensions, error) {
	return computer.Dimensions{Width: 800, Height: 600}, nil
}

func (lc *LoggingComputer) Screenshot(context.Context) (string, error) {
	lc.calls = append(lc.calls, []any{"Screenshot"})
	return lc.screenshotReturn, nil
}

func (lc *LoggingComputer) Click(_ context.Context, x, y int64, button computer.Button) error {
	lc.calls = append(lc.calls, []any{"Click", x, y, button})
	return nil
}

func (lc *LoggingComputer) DoubleClick(_ context.Context, x, y int64) error {
	lc.calls = append(lc.calls, []any{"DoubleClick", x, y})
	return nil
}

func (lc *LoggingComputer) Scroll(_ context.Context, x, y int64, scrollX, scrollY int64) error {
	lc.calls = append(lc.calls, []any{"Scroll", x, y, scrollX, scrollY})
	return nil
}

func (lc *LoggingComputer) Type(_ context.Context, text string) error {
	lc.calls = append(lc.calls, []any{"Type", text})
	return nil
}

func (lc *LoggingComputer) Wait(context.Context) error {
	lc.calls = append(lc.calls, []any{"Wait"})
	return nil
}

func (lc *LoggingComputer) Move(_ context.Context, x, y int64) error {
	lc.calls = append(lc.calls, []any{"Move", x, y})
	return nil
}

func (lc *LoggingComputer) Keypress(_ context.Context, keys []string) error {
	lc.calls = append(lc.calls, []any{"Keypress", keys})
	return nil
}

func (lc *LoggingComputer) Drag(_ context.Context, path []computer.Position) error {
	lc.calls = append(lc.calls, []any{"Drag", path})
	return nil
}

func TestGetScreenshotExecutesActionAndTakesScreenshot(t *testing.T) {
	// For each action type, assert that the corresponding computer method is
	// invoked and that a screenshot is taken and returned.

	type Action = responses.ResponseComputerToolCallActionUnion
	type DragPath = responses.ResponseComputerToolCallActionDragPath

	testCases := []struct {
		action       Action
		expectedCall []any
	}{
		{
			Action{Type: "click", X: 10, Y: 21, Button: "left"},
			[]any{"Click", int64(10), int64(21), computer.ButtonLeft},
		},
		{
			Action{Type: "double_click", X: 42, Y: 47},
			[]any{"DoubleClick", int64(42), int64(47)},
		},
		{
			Action{Type: "drag", Path: []DragPath{{X: 1, Y: 2}, {X: 3, Y: 4}}},
			[]any{"Drag", []computer.Position{{X: 1, Y: 2}, {X: 3, Y: 4}}},
		},
		{
			Action{Type: "keypress", Keys: []string{"a", "b"}},
			[]any{"Keypress", []string{"a", "b"}},
		},
		{
			Action{Type: "move", X: 100, Y: 200},
			[]any{"Move", int64(100), int64(200)},
		},
		{
			Action{Type: "screenshot"},
			[]any{"Screenshot"},
		},
		{
			Action{Type: "scroll", X: 1, Y: 2, ScrollX: 3, ScrollY: 4},
			[]any{"Scroll", int64(1), int64(2), int64(3), int64(4)},
		},
		{
			Action{Type: "type", Text: "hello"},
			[]any{"Type", "hello"},
		},
		{
			Action{Type: "wait"},
			[]any{"Wait"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.action.Type, func(t *testing.T) {
			comp := NewLoggingComputer("synthetic")

			toolCall := responses.ResponseComputerToolCall{
				ID:                  "c1",
				Type:                responses.ResponseComputerToolCallTypeComputerCall,
				Action:              tc.action,
				CallID:              "c1",
				PendingSafetyChecks: nil,
				Status:              responses.ResponseComputerToolCallStatusCompleted,
			}

			result, err := ComputerAction().getScreenshot(t.Context(), comp, toolCall)
			require.NoError(t, err)
			assert.Equal(t, "synthetic", result)

			assert.Equal(t, [][]any{
				tc.expectedCall,
				{"Screenshot"}, // The last call is always to Screenshot()
			}, comp.calls)
		})
	}
}

// LoggingRunHooks captures OnToolStart and OnToolEnd invocations.
type LoggingRunHooks struct {
	Started [][]any
	Ended   [][]any
}

func (*LoggingRunHooks) OnLLMStart(context.Context, *Agent, param.Opt[string], []TResponseInputItem) error {
	return nil
}
func (*LoggingRunHooks) OnLLMEnd(context.Context, *Agent, ModelResponse) error { return nil }
func (*LoggingRunHooks) OnAgentStart(context.Context, *Agent) error            { return nil }
func (*LoggingRunHooks) OnAgentEnd(context.Context, *Agent, any) error         { return nil }
func (*LoggingRunHooks) OnHandoff(context.Context, *Agent, *Agent) error       { return nil }
func (h *LoggingRunHooks) OnToolStart(_ context.Context, agent *Agent, tool Tool) error {
	h.Started = append(h.Started, []any{agent, tool})
	return nil
}
func (h *LoggingRunHooks) OnToolEnd(_ context.Context, agent *Agent, tool Tool, result any) error {
	h.Ended = append(h.Ended, []any{agent, tool, result})
	return nil
}

// LoggingAgentHooks captures agent's tool hook invocations.
type LoggingAgentHooks struct {
	Started [][]any
	Ended   [][]any
}

func (h *LoggingAgentHooks) OnStart(context.Context, *Agent) error           { return nil }
func (h *LoggingAgentHooks) OnEnd(context.Context, *Agent, any) error        { return nil }
func (h *LoggingAgentHooks) OnHandoff(context.Context, *Agent, *Agent) error { return nil }
func (h *LoggingAgentHooks) OnToolStart(_ context.Context, agent *Agent, tool Tool, arguments any) error {
	h.Started = append(h.Started, []any{agent, tool})
	return nil
}
func (h *LoggingAgentHooks) OnToolEnd(_ context.Context, agent *Agent, tool Tool, result any) error {
	h.Ended = append(h.Ended, []any{agent, tool, result})
	return nil
}
func (*LoggingAgentHooks) OnLLMStart(context.Context, *Agent, param.Opt[string], []TResponseInputItem) error {
	return nil
}
func (*LoggingAgentHooks) OnLLMEnd(context.Context, *Agent, ModelResponse) error { return nil }

func TestExecuteInvokesHooksAndReturnsToolCallOutput(t *testing.T) {
	// ComputerAction().Execute() should invoke lifecycle hooks and return a proper ToolCallOutputItem.

	comp := NewLoggingComputer("xyz")
	compTool := ComputerTool{Computer: comp}

	// Create a dummy click action to trigger a click and screenshot.
	action := responses.ResponseComputerToolCallActionUnion{
		Type: "click", X: 1, Y: 2, Button: "left",
	}
	toolCall := responses.ResponseComputerToolCall{
		ID:                  "tool123",
		Type:                responses.ResponseComputerToolCallTypeComputerCall,
		Action:              action,
		CallID:              "tool123",
		PendingSafetyChecks: nil,
		Status:              responses.ResponseComputerToolCallStatusCompleted,
	}

	// Wrap tool call in ToolRunComputerAction
	toolRun := ToolRunComputerAction{
		ToolCall:     toolCall,
		ComputerTool: compTool,
	}
	// Setup agent and hooks.
	agent := &Agent{
		Name:  "test_agent",
		Tools: []Tool{compTool},
	}
	// Attach per-agent hooks as well as global run hooks.
	agentHooks := &LoggingAgentHooks{}
	agent.Hooks = agentHooks
	runHooks := &LoggingRunHooks{}
	// Execute the computer action.
	result, err := ComputerAction().Execute(t.Context(), agent, toolRun, runHooks, nil)
	require.NoError(t, err)

	// Both global and per-agent hooks should have been called once.
	require.Len(t, runHooks.Started, 1)
	require.Len(t, agentHooks.Started, 1)
	require.Len(t, runHooks.Ended, 1)
	require.Len(t, agentHooks.Ended, 1)

	// The hook invocations should refer to our agent and tool.
	assert.Same(t, agent, runHooks.Started[0][0])
	assert.Same(t, agent, runHooks.Ended[0][0])
	assert.Equal(t, compTool, runHooks.Started[0][1])
	assert.Equal(t, compTool, runHooks.Ended[0][1])

	// The result passed to on_tool_end should be the raw screenshot string.
	assert.Equal(t, "xyz", runHooks.Ended[0][2])
	assert.Equal(t, "xyz", agentHooks.Ended[0][2])

	// The computer should have performed a click then a screenshot.
	assert.Equal(t, [][]any{
		{"Click", int64(1), int64(2), computer.ButtonLeft},
		{"Screenshot"},
	}, comp.calls)

	// The returned item should include the agent, output string, and a ComputerCallOutput.
	require.IsType(t, ToolCallOutputItem{}, result)
	outputItem := result.(ToolCallOutputItem)
	assert.Same(t, agent, outputItem.Agent)
	assert.Equal(t, "data:image/png;base64,xyz", outputItem.Output)
	assert.Equal(t, ResponseInputItemComputerCallOutputParam{
		CallID: "tool123",
		Output: responses.ResponseComputerToolCallOutputScreenshotParam{
			ImageURL: param.NewOpt("data:image/png;base64,xyz"),
			Type:     constant.ValueOf[constant.ComputerScreenshot](),
		},
		ID:                       param.Opt[string]{},
		AcknowledgedSafetyChecks: nil,
		Status:                   "",
		Type:                     constant.ValueOf[constant.ComputerCallOutput](),
	}, outputItem.RawItem)
}

func TestPendingSafetyCheckAcknowledged(t *testing.T) {
	// Safety checks should be acknowledged via the callback.

	comp := NewLoggingComputer("img")
	var called []ComputerToolSafetyCheckData

	onSafetyCheck := func(_ context.Context, data ComputerToolSafetyCheckData) (bool, error) {
		called = append(called, data)
		return true, nil
	}

	tool := ComputerTool{
		Computer:      comp,
		OnSafetyCheck: onSafetyCheck,
	}
	safety := responses.ResponseComputerToolCallPendingSafetyCheck{
		ID:      "sc",
		Code:    "c",
		Message: "m",
	}
	toolCall := responses.ResponseComputerToolCall{
		ID:                  "t1",
		Action:              responses.ResponseComputerToolCallActionUnion{Type: "click", X: 1, Y: 2, Button: "left"},
		CallID:              "t1",
		PendingSafetyChecks: []responses.ResponseComputerToolCallPendingSafetyCheck{safety},
		Status:              responses.ResponseComputerToolCallStatusCompleted,
		Type:                responses.ResponseComputerToolCallTypeComputerCall,
	}
	runAction := ToolRunComputerAction{
		ToolCall:     toolCall,
		ComputerTool: tool,
	}
	agent := New("a").WithTools(tool)

	results, err := RunImpl().ExecuteComputerActions(t.Context(), agent, []ToolRunComputerAction{runAction}, NoOpRunHooks{})
	require.NoError(t, err)
	assert.Equal(t, []RunItem{
		ToolCallOutputItem{
			Agent: agent,
			RawItem: ResponseInputItemComputerCallOutputParam{
				CallID: "t1",
				Output: responses.ResponseComputerToolCallOutputScreenshotParam{
					ImageURL: param.NewOpt("data:image/png;base64,img"),
					Type:     constant.ValueOf[constant.ComputerScreenshot](),
				},
				ID: param.Opt[string]{},
				AcknowledgedSafetyChecks: []responses.ResponseInputItemComputerCallOutputAcknowledgedSafetyCheckParam{
					{
						ID:      "sc",
						Code:    param.NewOpt("c"),
						Message: param.NewOpt("m"),
					},
				},
				Status: "",
				Type:   constant.ValueOf[constant.ComputerCallOutput](),
			},
			Output: "data:image/png;base64,img",
			Type:   "tool_call_output_item",
		},
	}, results)
	assert.Equal(t, []ComputerToolSafetyCheckData{
		{
			Agent:       agent,
			ToolCall:    toolCall,
			SafetyCheck: safety,
		},
	}, called)
}
