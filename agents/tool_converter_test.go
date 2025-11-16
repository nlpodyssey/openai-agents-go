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

package agents_test

import (
	"context"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToOpenaiWithFunctionTool(t *testing.T) {
	tool := agents.FunctionTool{
		Name:        "some_function",
		Description: "Function description.",
		ParamsJSONSchema: map[string]any{
			"title":                "some_function_args",
			"type":                 "object",
			"required":             []string{"a"},
			"additionalProperties": false,
			"properties": map[string]any{
				"a": map[string]any{"title": "A", "type": "string"},
			},
		},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return nil, nil
		},
	}

	result, err := agents.ChatCmplConverter().ToolToOpenai(tool)
	require.NoError(t, err)
	assert.Equal(t, openai.ChatCompletionFunctionTool(
		openai.FunctionDefinitionParam{
			Name:        "some_function",
			Strict:      param.Opt[bool]{},
			Description: param.NewOpt("Function description."),
			Parameters:  tool.ParamsJSONSchema,
		},
	), *result)
}

func TestConvertHandoffTool(t *testing.T) {
	agent := &agents.Agent{
		Name:               "test_1",
		HandoffDescription: "test_2",
	}
	handoff, err := agents.SafeHandoffFromAgent(agents.HandoffFromAgentParams{
		Agent: agent,
	})
	require.NoError(t, err)
	require.NotNil(t, handoff)

	result := agents.ChatCmplConverter().ConvertHandoffTool(*handoff)

	assert.Equal(t, openai.ChatCompletionFunctionTool(
		openai.FunctionDefinitionParam{
			Name:        agents.DefaultHandoffToolName(agent),
			Strict:      param.Opt[bool]{},
			Description: param.NewOpt(agents.DefaultHandoffToolDescription(agent)),
			Parameters:  handoff.InputJSONSchema,
		},
	), result)
}

func TestToolConverterHostedToolsErrors(t *testing.T) {
	tools := []agents.Tool{
		agents.CodeInterpreterTool{},
		agents.ComputerTool{},
		agents.FileSearchTool{},
		agents.ImageGenerationTool{},
		agents.LocalShellTool{},
		agents.WebSearchTool{},
	}

	for _, tool := range tools {
		t.Run(tool.ToolName(), func(t *testing.T) {
			_, err := agents.ChatCmplConverter().ToolToOpenai(tool)
			assert.ErrorAs(t, err, &agents.UserError{})
		})
	}
}
