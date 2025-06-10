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

import "context"

// ToolUseBehavior lets you configure how tool use is handled.
// See Agent.ToolUseBehavior.
type ToolUseBehavior interface {
	isToolUseBehavior()
}

type RunLLMAgain struct{}

func (RunLLMAgain) isToolUseBehavior() {}

type StopOnFirstTool struct{}

func (StopOnFirstTool) isToolUseBehavior() {}

type StopAtTools struct {
	// A list of tool names, any of which will stop the agent from running further.
	StopAtToolNames []string
}

func (StopAtTools) isToolUseBehavior() {}

// ToolsToFinalOutputFunction is a function that takes a run context and a list
// of tool results, and returns a `ToolsToFinalOutputResult`.
type ToolsToFinalOutputFunction func(context.Context, []FunctionToolResult) (ToolsToFinalOutputResult, error)

func (ToolsToFinalOutputFunction) isToolUseBehavior() {}
