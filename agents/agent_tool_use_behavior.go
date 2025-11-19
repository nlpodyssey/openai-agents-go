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
	"slices"

	"github.com/openai/openai-go/v3/packages/param"
)

// ToolUseBehavior lets you configure how tool use is handled.
// See Agent.ToolUseBehavior.
type ToolUseBehavior interface {
	ToolsToFinalOutput(context.Context, []FunctionToolResult) (ToolsToFinalOutputResult, error)
}

// RunLLMAgain returns a ToolUseBehavior that ignores any FunctionToolResults
// and always returns a non-final output result. With this behavior, the LLM
// receives the tool results and gets to respond.
func RunLLMAgain() ToolUseBehavior { return runLLMAgain{} }

type runLLMAgain struct{}

func (runLLMAgain) ToolsToFinalOutput(context.Context, []FunctionToolResult) (ToolsToFinalOutputResult, error) {
	return notFinalOutput, nil
}

// StopOnFirstTool returns a ToolUseBehavior which uses the output of the first
// tool call as the final output. This means that the LLM does not process the
// result of the tool call.
func StopOnFirstTool() ToolUseBehavior { return stopOnFirstTool{} }

type stopOnFirstTool struct{}

func (stopOnFirstTool) ToolsToFinalOutput(_ context.Context, toolResults []FunctionToolResult) (ToolsToFinalOutputResult, error) {
	if len(toolResults) == 0 {
		return notFinalOutput, nil
	}
	return ToolsToFinalOutputResult{
		IsFinalOutput: true,
		FinalOutput:   param.NewOpt(toolResults[0].Output),
	}, nil
}

// StopAtTools returns a ToolUseBehavior which causes the agent to stop running
// if any of the tools in the given list are called. The final output will be
// the output of the first matching tool call. The LLM does not process the
// result of the tool call.
func StopAtTools(toolNames ...string) ToolUseBehavior {
	return stopAtTools{names: toolNames}
}

type stopAtTools struct {
	names []string
}

func (sat stopAtTools) ToolsToFinalOutput(_ context.Context, toolResults []FunctionToolResult) (ToolsToFinalOutputResult, error) {
	for _, toolResult := range toolResults {
		if slices.Contains(sat.names, toolResult.Tool.Name) {
			return ToolsToFinalOutputResult{
				IsFinalOutput: true,
				FinalOutput:   param.NewOpt(toolResult.Output),
			}, nil
		}
	}
	return notFinalOutput, nil
}

// ToolsToFinalOutputFunction lets you implement a custom ToolUseBehavior.
type ToolsToFinalOutputFunction func(context.Context, []FunctionToolResult) (ToolsToFinalOutputResult, error)

func (f ToolsToFinalOutputFunction) ToolsToFinalOutput(ctx context.Context, toolResults []FunctionToolResult) (ToolsToFinalOutputResult, error) {
	return f(ctx, toolResults)
}

var notFinalOutput = ToolsToFinalOutputResult{
	IsFinalOutput: false,
	FinalOutput:   param.Opt[any]{},
}
