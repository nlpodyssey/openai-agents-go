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
	"errors"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertToolChoiceStandardValues(t *testing.T) {
	// Make sure that the standard ToolChoice values map to themselves or
	// to "auto"/"required"/"none" as appropriate, and that special string
	// values map to the appropriate items.

	v := agents.ResponsesConverter().ConvertToolChoice("")
	assert.Equal(t, responses.ResponseNewParamsToolChoiceUnion{}, v)

	v = agents.ResponsesConverter().ConvertToolChoice("auto")
	assert.Equal(t, responses.ResponseNewParamsToolChoiceUnion{
		OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
	}, v)

	v = agents.ResponsesConverter().ConvertToolChoice("required")
	assert.Equal(t, responses.ResponseNewParamsToolChoiceUnion{
		OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
	}, v)

	v = agents.ResponsesConverter().ConvertToolChoice("none")
	assert.Equal(t, responses.ResponseNewParamsToolChoiceUnion{
		OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
	}, v)

	// Arbitrary string should be interpreted as a function name.
	v = agents.ResponsesConverter().ConvertToolChoice("my_function")
	assert.Equal(t, responses.ResponseNewParamsToolChoiceUnion{
		OfFunctionTool: &responses.ToolChoiceFunctionParam{
			Name: "my_function",
			Type: constant.ValueOf[constant.Function](),
		},
	}, v)
}

type PlainTextSchema struct{}

func (p PlainTextSchema) IsPlainText() bool                { return true }
func (p PlainTextSchema) Name() string                     { return "PlainText" }
func (p PlainTextSchema) JSONSchema() map[string]any       { return nil }
func (p PlainTextSchema) IsStrictJSONSchema() bool         { return false }
func (p PlainTextSchema) ValidateJSON(string) (any, error) { return nil, errors.New("not implemented") }

type FakeSchema struct{}

func (p FakeSchema) IsPlainText() bool { return false }
func (p FakeSchema) Name() string      { return "Fake" }
func (p FakeSchema) JSONSchema() map[string]any {
	return map[string]any{"title": "Fake"}
}
func (p FakeSchema) IsStrictJSONSchema() bool         { return true }
func (p FakeSchema) ValidateJSON(string) (any, error) { return nil, errors.New("not implemented") }

func TestGetResponseFormatPlainTextAndJsonSchema(t *testing.T) {
	// For plain text output, the converter should return a zero-value,
	// indicating no special response format constraint.
	// If an output schema is provided for a structured type, the converter
	// should return a ResponseTextConfigParam with the schema and strictness.

	// Default output (None) should be considered plain text.
	v := agents.ResponsesConverter().GetResponseFormat(nil)
	assert.Equal(t, responses.ResponseTextConfigParam{}, v)

	// An explicit plain-text schema (str) should also yield zero-value.
	v = agents.ResponsesConverter().GetResponseFormat(PlainTextSchema{})
	assert.Equal(t, responses.ResponseTextConfigParam{}, v)

	// A model-based schema should produce a format dict.
	v = agents.ResponsesConverter().GetResponseFormat(FakeSchema{})
	assert.Equal(t, responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "final_output",
				Schema: FakeSchema{}.JSONSchema(),
				Strict: param.NewOpt(true),
				Type:   constant.ValueOf[constant.JSONSchema](),
			},
		},
	}, v)
}

func TestConvertToolsBasicTypesAndIncludes(t *testing.T) {
	// Construct a variety of tool types and make sure `ConvertTools` returns
	// a matching list of tool param dicts and the expected includes.

	// Simple function tool
	toolFn := agents.FunctionTool{
		Name:             "fn",
		Description:      "...",
		ParamsJSONSchema: map[string]any{"title": "Fn"},
		OnInvokeTool: func(context.Context, *runcontext.RunContextWrapper, string) (any, error) {
			return nil, errors.New("not implemented")
		},
	}

	tools := []agents.Tool{toolFn}
	converted := agents.ResponsesConverter().ConvertTools(tools, nil)
	assert.Equal(t, agents.ConvertedTools{
		Tools: []responses.ToolUnionParam{
			{
				OfFunction: &responses.FunctionToolParam{
					Name:        "fn",
					Parameters:  toolFn.ParamsJSONSchema,
					Strict:      param.NewOpt(true),
					Description: param.NewOpt("..."),
					Type:        constant.ValueOf[constant.Function](),
				},
			},
		},
		Includes: nil,
	}, converted)
}

func TestConvertToolsIncludesHandoffs(t *testing.T) {
	//  When handoff objects are included, `ConvertTools` should append their
	//  tool param items after tools and include appropriate descriptions.

	agent := &agents.Agent{
		Name:               "support",
		HandoffDescription: "Handles support",
	}
	handoff, err := agents.HandoffFromAgent(agents.HandoffFromAgentParams{Agent: agent})
	require.NoError(t, err)
	require.NotNil(t, handoff)

	converted := agents.ResponsesConverter().ConvertTools(nil, []agents.Handoff{*handoff})
	assert.Equal(t, agents.ConvertedTools{
		Tools: []responses.ToolUnionParam{
			{
				OfFunction: &responses.FunctionToolParam{
					Name: agents.DefaultHandoffToolName(agent),
					Parameters: map[string]any{
						"type":                 "object",
						"additionalProperties": false,
						"properties":           map[string]any{},
						"required":             []string{},
					},
					Strict:      param.NewOpt(true),
					Description: param.NewOpt(agents.DefaultHandoffToolDescription(agent)),
					Type:        constant.ValueOf[constant.Function](),
				},
			},
		},
		Includes: nil,
	}, converted)
}
