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

package tools

import (
	"context"

	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
)

// Function Tool that wraps a function.
type Function struct {
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
	OnInvokeTool func(ctx context.Context, rcw *runcontext.Wrapper, arguments string) (any, error)

	// Whether the JSON schema is in strict mode.
	// We **strongly** recommend setting this to True, as it increases the likelihood of correct JSON input.
	// Defaults to true if omitted.
	StrictJSONSchema param.Opt[bool]
}

func (f Function) ToolName() string {
	return f.Name
}

func (f Function) ConvertToResponses() (*responses.ToolUnionParam, *responses.ResponseIncludable, error) {
	return &responses.ToolUnionParam{
		OfFunction: &responses.FunctionToolParam{
			Name:        f.Name,
			Parameters:  f.ParamsJSONSchema,
			Strict:      param.NewOpt(f.StrictJSONSchema.Or(true)),
			Description: param.NewOpt(f.Description),
			Type:        constant.ValueOf[constant.Function](),
		},
	}, nil, nil
}

func (f Function) ConvertToChatCompletions() (*openai.ChatCompletionToolParam, error) {
	description := param.Null[string]()
	if f.Description != "" {
		description = param.NewOpt(f.Description)
	}
	return &openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        f.Name,
			Description: description,
			Parameters:  f.ParamsJSONSchema,
		},
		Type: constant.ValueOf[constant.Function](),
	}, nil
}
