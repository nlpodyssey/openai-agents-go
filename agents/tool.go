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

	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/openai/openai-go/packages/param"
)

// FunctionTool is a tool that wraps a function.
type FunctionTool struct {
	// The name of the tool, as shown to the LLM. Generally the name of the function.
	Name string

	// A description of the tool, as shown to the LLM.
	Description string

	// The JSON schema for the tool's parameters.
	ParamsJSONSchema map[string]any

	// A function that invokes the tool with the given context and parameters. The params passed
	// are:
	// 1. The tool run context.
	// 2. The arguments from the LLM, as a JSON string.
	//
	// You must return a string representation of the tool output, or something we can call `str()` on.
	// In case of errors, you can either raise an Exception (which will cause the run to fail) or
	// return a string error message (which will be sent back to the LLM).
	OnInvokeTool func(ctx context.Context, contextWrapper *runcontext.Wrapper, arguments string) (any, error)

	// Whether the JSON schema is in strict mode.
	// We **strongly** recommend setting this to True, as it increases the likelihood of correct JSON input.
	// Defaults to true if omitted.
	StrictJSONSchema param.Opt[bool]
}

// A Tool that can be used in an agent.
type Tool interface {
	isTool()
}

// Other types that we don't need are omitted:
//  - FileSearchTool
//  - WebSearchTool
//  - ComputerTool

func (FunctionTool) isTool() {}

type FunctionToolResult struct {
	// The tool that was run.
	Tool FunctionTool

	// The output of the tool.
	Output any

	// The run item that was produced as a result of the tool call.
	RunItem RunItem
}
