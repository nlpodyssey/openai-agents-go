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
	"cmp"
	"context"
	"errors"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func getFunctionTool(
	name string,
	returnValue string,
) FunctionTool {
	return FunctionTool{
		Name: name,
		ParamsJSONSchema: map[string]any{
			"title":                name + "_args",
			"type":                 "object",
			"required":             []string{},
			"additionalProperties": false,
			"properties":           map[string]any{},
		},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return returnValue, nil
		},
	}
}

func makeFunctionToolResult(agent *Agent, output string, toolName string) FunctionToolResult {
	// Construct a FunctionToolResult with the given output using a simple function tool.
	return FunctionToolResult{
		Tool:   getFunctionTool(cmp.Or(toolName, "dummy"), output),
		Output: output,
		RunItem: ToolCallOutputItem{
			Agent: agent,
			RawItem: ResponseInputItemFunctionCallOutputParam{
				CallID: "1",
				Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
					OfString: param.NewOpt(output),
				},
				Type: constant.ValueOf[constant.FunctionCallOutput](),
			},
			Output: output,
			Type:   "tool_call_output_item",
		},
	}
}

func TestNoToolResultsReturnsNotFinalOutput(t *testing.T) {
	// If there are no tool results at all, ToolUseBehavior should not produce a final output.
	agent := &Agent{Name: "test"}
	result, err := RunImpl().checkForFinalOutputFromTools(t.Context(), agent, nil)
	require.NoError(t, err)
	assert.Equal(t, ToolsToFinalOutputResult{
		IsFinalOutput: false,
		FinalOutput:   param.Opt[any]{},
	}, result)
}

func TestRunLlmAgainBehavior(t *testing.T) {
	// With the default RunLLMAgain behavior, even with tools we still expect to keep running.
	agent := &Agent{
		Name:            "test",
		ToolUseBehavior: RunLLMAgain(),
	}
	toolResults := []FunctionToolResult{
		makeFunctionToolResult(agent, "ignored", ""),
	}
	result, err := RunImpl().checkForFinalOutputFromTools(t.Context(), agent, toolResults)
	require.NoError(t, err)
	assert.Equal(t, ToolsToFinalOutputResult{
		IsFinalOutput: false,
		FinalOutput:   param.Opt[any]{},
	}, result)
}

func TestStopOnFirstToolBehavior(t *testing.T) {
	// When ToolUseBehavior is StopOnFirstTool, we should surface first tool output as final.
	agent := &Agent{
		Name:            "test",
		ToolUseBehavior: StopOnFirstTool(),
	}
	toolResults := []FunctionToolResult{
		makeFunctionToolResult(agent, "first_tool_output", ""),
		makeFunctionToolResult(agent, "ignored", ""),
	}
	result, err := RunImpl().checkForFinalOutputFromTools(t.Context(), agent, toolResults)
	require.NoError(t, err)
	assert.Equal(t, ToolsToFinalOutputResult{
		IsFinalOutput: true,
		FinalOutput:   param.NewOpt[any]("first_tool_output"),
	}, result)
}

func TestCustomToolUseBehavior(t *testing.T) {
	// If ToolUseBehavior is a function, we should call it and propagate its return.
	behavior := func(ctx context.Context, results []FunctionToolResult) (ToolsToFinalOutputResult, error) {
		assert.Len(t, results, 3)
		return ToolsToFinalOutputResult{
			IsFinalOutput: true,
			FinalOutput:   param.NewOpt[any]("custom"),
		}, nil
	}
	agent := &Agent{
		Name:            "test",
		ToolUseBehavior: ToolsToFinalOutputFunction(behavior),
	}
	toolResults := []FunctionToolResult{
		makeFunctionToolResult(agent, "ignored1", ""),
		makeFunctionToolResult(agent, "ignored2", ""),
		makeFunctionToolResult(agent, "ignored3", ""),
	}
	result, err := RunImpl().checkForFinalOutputFromTools(t.Context(), agent, toolResults)
	require.NoError(t, err)
	assert.Equal(t, ToolsToFinalOutputResult{
		IsFinalOutput: true,
		FinalOutput:   param.NewOpt[any]("custom"),
	}, result)
}

func TestCustomToolUseBehaviorError(t *testing.T) {
	behaviorErr := errors.New("error")
	behavior := func(ctx context.Context, results []FunctionToolResult) (ToolsToFinalOutputResult, error) {
		return ToolsToFinalOutputResult{}, behaviorErr
	}
	agent := &Agent{
		Name:            "test",
		ToolUseBehavior: ToolsToFinalOutputFunction(behavior),
	}
	toolResults := []FunctionToolResult{
		makeFunctionToolResult(agent, "ignored1", ""),
		makeFunctionToolResult(agent, "ignored2", ""),
		makeFunctionToolResult(agent, "ignored3", ""),
	}
	_, err := RunImpl().checkForFinalOutputFromTools(t.Context(), agent, toolResults)
	require.ErrorIs(t, err, behaviorErr)
}

func TestToolNamesToStopAtBehavior(t *testing.T) {
	agent := &Agent{
		Name: "test",
		Tools: []Tool{
			getFunctionTool("tool1", "tool1_output"),
			getFunctionTool("tool2", "tool2_output"),
			getFunctionTool("tool3", "tool3_output"),
		},
		ToolUseBehavior: StopAtTools("tool1"),
	}
	toolResults := []FunctionToolResult{
		makeFunctionToolResult(agent, "ignored2", "tool2"),
		makeFunctionToolResult(agent, "ignored3", "tool3"),
	}
	result, err := RunImpl().checkForFinalOutputFromTools(t.Context(), agent, toolResults)
	require.NoError(t, err)
	assert.Equal(t, ToolsToFinalOutputResult{
		IsFinalOutput: false,
		FinalOutput:   param.Opt[any]{},
	}, result)

	// Now test with a tool that matches the list
	toolResults = []FunctionToolResult{
		makeFunctionToolResult(agent, "output1", "tool1"),
		makeFunctionToolResult(agent, "ignored2", "tool2"),
		makeFunctionToolResult(agent, "ignored3", "tool3"),
	}
	result, err = RunImpl().checkForFinalOutputFromTools(t.Context(), agent, toolResults)
	require.NoError(t, err)
	assert.Equal(t, ToolsToFinalOutputResult{
		IsFinalOutput: true,
		FinalOutput:   param.NewOpt[any]("output1"),
	}, result)
}
