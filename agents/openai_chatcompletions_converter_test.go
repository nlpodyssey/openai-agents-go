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
	"errors"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessageToOutputItemsWithTextOnly(t *testing.T) {
	// Make sure a simple ChatCompletionMessage with string content is converted
	// into a single ResponseOutputMessage containing one ResponseOutputText.

	msg := openai.ChatCompletionMessage{
		Content: "Hello",
		Role:    constant.ValueOf[constant.Assistant](),
	}

	items := agents.ChatCmplConverter().MessageToOutputItems(msg)
	assert.Equal(t, []agents.TResponseOutputItem{
		{
			ID: agents.FakeResponsesID,
			Content: []responses.ResponseOutputMessageContentUnion{
				{
					Annotations: nil,
					Text:        "Hello",
					Type:        "output_text",
				},
			},
			Role:   constant.ValueOf[constant.Assistant](),
			Status: "completed",
			Type:   "message",
		},
	}, items)
}

func TestMessageToOutputItemsWithRefusal(t *testing.T) {
	// Make sure a message with a refusal string produces a ResponseOutputMessage
	// with a ResponseOutputRefusal content part.

	msg := openai.ChatCompletionMessage{
		Refusal: "I'm sorry",
		Role:    constant.ValueOf[constant.Assistant](),
	}

	items := agents.ChatCmplConverter().MessageToOutputItems(msg)
	assert.Equal(t, []agents.TResponseOutputItem{
		{
			ID: agents.FakeResponsesID,
			Content: []responses.ResponseOutputMessageContentUnion{
				{
					Refusal: "I'm sorry",
					Type:    "refusal",
				},
			},
			Role:   constant.ValueOf[constant.Assistant](),
			Status: "completed",
			Type:   "message",
		},
	}, items)
}

func TestMessageToOutputItemsWithToolCall(t *testing.T) {
	// If the ChatCompletionMessage contains one or more tool calls, they should
	// be reflected as separate `ResponseFunctionToolCall` items appended after
	// the message item.

	toolCall := openai.ChatCompletionMessageToolCall{
		ID: "tool1",
		Function: openai.ChatCompletionMessageToolCallFunction{
			Name:      "my_func",
			Arguments: `{"x":1}`,
		},
		Type: constant.ValueOf[constant.Function](),
	}

	msg := openai.ChatCompletionMessage{
		Content:   "Hi",
		ToolCalls: []openai.ChatCompletionMessageToolCall{toolCall},
		Role:      constant.ValueOf[constant.Assistant](),
	}

	items := agents.ChatCmplConverter().MessageToOutputItems(msg)
	assert.Equal(t, []agents.TResponseOutputItem{
		{
			ID: agents.FakeResponsesID,
			Content: []responses.ResponseOutputMessageContentUnion{
				{
					Annotations: nil,
					Text:        "Hi",
					Type:        "output_text",
				},
			},
			Role:   constant.ValueOf[constant.Assistant](),
			Status: "completed",
			Type:   "message",
		},
		{
			ID:        agents.FakeResponsesID,
			CallID:    "tool1",
			Name:      "my_func",
			Arguments: `{"x":1}`,
			Type:      "function_call",
		},
	}, items)
}

func TestItemsToMessagesWithStringUserContent(t *testing.T) {
	// A simple string as the items argument should be converted into a user
	// message param with the same content.
	result, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputString("Ask me anything"))
	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: param.NewOpt("Ask me anything"),
				},
				Role: constant.ValueOf[constant.User](),
			},
		},
	}, result)
}

func TestItemsToMessagesWithEasyInputMessage(t *testing.T) {
	// Given an easy input message (just role/content), the converter should
	// produce the appropriate ChatCompletionMessageParam with the same content.

	messages, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("How are you?"),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
	})

	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: param.NewOpt("How are you?"),
				},
				Role: constant.ValueOf[constant.User](),
			},
		},
	}, messages)
}
func TestItemsToMessagesWithOutputMessageAndFunctionCall(t *testing.T) {
	// Given a sequence of one ResponseOutputMessageParam followed by a
	// ResponseFunctionToolCallParam, the converter should produce a single
	// ChatCompletionAssistantMessageParam that includes both the assistant's
	// textual content and a populated `ToolCalls` reflecting the function call.

	// Construct output message param with two content parts.
	outputText := responses.ResponseOutputMessageContentUnionParam{
		OfOutputText: &responses.ResponseOutputTextParam{
			Annotations: nil,
			Text:        "Part 1",
			Type:        constant.ValueOf[constant.OutputText](),
		},
	}
	refusal := responses.ResponseOutputMessageContentUnionParam{
		OfRefusal: &responses.ResponseOutputRefusalParam{
			Refusal: "won't do that",
			Type:    constant.ValueOf[constant.Refusal](),
		},
	}
	respMsg := agents.TResponseInputItem{
		OfOutputMessage: &responses.ResponseOutputMessageParam{
			ID:      "42",
			Type:    constant.ValueOf[constant.Message](),
			Role:    constant.ValueOf[constant.Assistant](),
			Status:  responses.ResponseOutputMessageStatusCompleted,
			Content: []responses.ResponseOutputMessageContentUnionParam{outputText, refusal},
		},
	}

	// Construct a function call item (as if returned from model)
	funcItem := agents.TResponseInputItem{
		OfFunctionCall: &responses.ResponseFunctionToolCallParam{
			ID:        param.NewOpt("99"),
			CallID:    "abc",
			Name:      "math",
			Arguments: "{}",
			Type:      constant.ValueOf[constant.FunctionCall](),
		},
	}

	messages, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{
		respMsg,
		funcItem,
	})

	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
		{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				Refusal: param.NewOpt("won't do that"),
				//Name:         param.Opt[string]{},
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: param.NewOpt("Part 1"),
				},
				ToolCalls: []openai.ChatCompletionMessageToolCallParam{
					{
						ID: "abc",
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      "math",
							Arguments: "{}",
						},
						Type: constant.ValueOf[constant.Function](),
					},
				},
				Role: constant.ValueOf[constant.Assistant](),
			},
		},
	}, messages)
}

func TestConvertToolChoiceHandlesStandardAndNamedOptions(t *testing.T) {
	// The `ConvertToolChoice` method should return false (not given)
	// if no choice is provided, pass through values like "auto", "required",
	// or "none" unchanged, and translate any other string into a function
	// selection object.

	_, ok := agents.ChatCmplConverter().ConvertToolChoice("")
	assert.False(t, ok)

	v, ok := agents.ChatCmplConverter().ConvertToolChoice("auto")
	assert.True(t, ok)
	assert.Equal(t, openai.ChatCompletionToolChoiceOptionUnionParam{
		OfAuto: param.NewOpt("auto"),
	}, v)

	v, ok = agents.ChatCmplConverter().ConvertToolChoice("required")
	assert.True(t, ok)
	assert.Equal(t, openai.ChatCompletionToolChoiceOptionUnionParam{
		OfAuto: param.NewOpt("required"),
	}, v)

	v, ok = agents.ChatCmplConverter().ConvertToolChoice("none")
	assert.True(t, ok)
	assert.Equal(t, openai.ChatCompletionToolChoiceOptionUnionParam{
		OfAuto: param.NewOpt("none"),
	}, v)

	v, ok = agents.ChatCmplConverter().ConvertToolChoice("mytool")
	assert.True(t, ok)
	assert.Equal(t, openai.ChatCompletionToolChoiceOptionUnionParam{
		OfChatCompletionNamedToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
			Function: openai.ChatCompletionNamedToolChoiceFunctionParam{
				Name: "mytool",
			},
			Type: constant.ValueOf[constant.Function](),
		},
	}, v)
}

type IntResponseSchema struct{}

func (IntResponseSchema) Name() string             { return "int" }
func (IntResponseSchema) IsPlainText() bool        { return false }
func (IntResponseSchema) IsStrictJSONSchema() bool { return true }
func (IntResponseSchema) JSONSchema() map[string]any {
	return map[string]any{
		"title":                "OutputType",
		"type":                 "object",
		"required":             []string{"response"},
		"additionalProperties": false,
		"properties": map[string]any{
			"a": map[string]any{
				"title": "Response",
				"type":  "integer",
			},
		},
	}
}
func (IntResponseSchema) ValidateJSON(string) (any, error) {
	return nil, errors.New("not implemented")
}

func TestConvertResponseFormatReturnsNotGivenForPlainTextAndObjectForSchemas(t *testing.T) {
	// The `ConvertResponseFormat` method should return false (not given)
	// when no output schema is provided or if the output schema indicates
	// plain text. For structured output schemas, it should return an object
	// with type `json_schema` and include the generated JSON schema and
	// strict flag from the provided `AgentOutputSchema`.

	_, ok := agents.ChatCmplConverter().ConvertResponseFormat(nil)
	assert.False(t, ok)

	v, ok := agents.ChatCmplConverter().ConvertResponseFormat(IntResponseSchema{})
	assert.True(t, ok)
	assert.Equal(t, openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
			JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        "final_output",
				Strict:      param.NewOpt(true),
				Description: param.NullOpt[string](),
				Schema:      IntResponseSchema{}.JSONSchema(),
			},
			Type: constant.ValueOf[constant.JSONSchema](),
		},
	}, v)
}

func TestItemsToMessagesWithFunctionOutputItem(t *testing.T) {
	// A function call output item should be converted into a tool role message
	// with the appropriate ToolCallID and content.
	v, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{
		{
			OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
				CallID: "some_call",
				Output: `{"foo": "bar"}`,
				Type:   constant.ValueOf[constant.FunctionCallOutput](),
			},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
		{
			OfTool: &openai.ChatCompletionToolMessageParam{
				Content: openai.ChatCompletionToolMessageParamContentUnion{
					OfString: param.NewOpt(`{"foo": "bar"}`),
				},
				ToolCallID: "some_call",
				Role:       constant.ValueOf[constant.Tool](),
			},
		},
	}, v)
}

func TestExtractAllAndTextContentForStringsAndLists(t *testing.T) {
	// The converter provides helpers for extracting user-supplied message content
	// either as a simple string or as a list of `input_text` objects.
	// When passed a bare string, both `ExtractAllContent` and
	// `ExtractTextContent` should return the string unchanged.
	// When passed a list of input objects, `ExtractAllContent` should
	// produce a list of `ChatCompletionContentPart` items, and `ExtractTextContent`
	// should filter to only the textual parts.

	v, err := agents.ChatCmplConverter().ExtractAllContentFromEasyInputMessageContentUnionParam(responses.EasyInputMessageContentUnionParam{
		OfString: param.NewOpt("hi"),
	})
	require.NoError(t, err)
	assert.Equal(t, &openai.ChatCompletionUserMessageParamContentUnion{
		OfString: param.NewOpt("hi"),
	}, v)

	s, params, err := agents.ChatCmplConverter().ExtractTextContentFromEasyInputMessageContentUnionParam(responses.EasyInputMessageContentUnionParam{
		OfString: param.NewOpt("hi"),
	})
	require.NoError(t, err)
	assert.Nil(t, params)
	assert.Equal(t, param.NewOpt("hi"), s)

	text1 := responses.ResponseInputTextParam{Text: "one", Type: constant.ValueOf[constant.InputText]()}
	text2 := responses.ResponseInputTextParam{Text: "two", Type: constant.ValueOf[constant.InputText]()}

	v, err = agents.ChatCmplConverter().ExtractAllContentFromEasyInputMessageContentUnionParam(responses.EasyInputMessageContentUnionParam{
		OfInputItemContentList: responses.ResponseInputMessageContentListParam{
			{OfInputText: &text1},
			{OfInputText: &text2},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, &openai.ChatCompletionUserMessageParamContentUnion{
		OfArrayOfContentParts: []openai.ChatCompletionContentPartUnionParam{
			{OfText: &openai.ChatCompletionContentPartTextParam{Text: "one", Type: constant.ValueOf[constant.Text]()}},
			{OfText: &openai.ChatCompletionContentPartTextParam{Text: "two", Type: constant.ValueOf[constant.Text]()}},
		},
	}, v)

	v, err = agents.ChatCmplConverter().ExtractAllContentFromResponseInputContentUnionParams([]responses.ResponseInputContentUnionParam{
		{OfInputText: &text1},
		{OfInputText: &text2},
	})
	require.NoError(t, err)
	assert.Equal(t, &openai.ChatCompletionUserMessageParamContentUnion{
		OfArrayOfContentParts: []openai.ChatCompletionContentPartUnionParam{
			{OfText: &openai.ChatCompletionContentPartTextParam{Text: "one", Type: constant.ValueOf[constant.Text]()}},
			{OfText: &openai.ChatCompletionContentPartTextParam{Text: "two", Type: constant.ValueOf[constant.Text]()}},
		},
	}, v)

	s, params, err = agents.ChatCmplConverter().ExtractTextContentFromEasyInputMessageContentUnionParam(responses.EasyInputMessageContentUnionParam{
		OfInputItemContentList: responses.ResponseInputMessageContentListParam{
			{OfInputText: &text1},
			{OfInputText: &text2},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, param.NullOpt[string](), s)
	assert.Equal(t, []openai.ChatCompletionContentPartTextParam{
		{Text: "one", Type: constant.ValueOf[constant.Text]()},
		{Text: "two", Type: constant.ValueOf[constant.Text]()},
	}, params)

	s, params, err = agents.ChatCmplConverter().ExtractTextContentFromResponseInputMessageContentListParams(responses.ResponseInputMessageContentListParam{
		{OfInputText: &text1},
		{OfInputText: &text2},
	})
	require.NoError(t, err)
	assert.Equal(t, param.NullOpt[string](), s)
	assert.Equal(t, []openai.ChatCompletionContentPartTextParam{
		{Text: "one", Type: constant.ValueOf[constant.Text]()},
		{Text: "two", Type: constant.ValueOf[constant.Text]()},
	}, params)

	params = agents.ChatCmplConverter().ExtractTextContentFromChatCompletionContentPartUnionParams([]openai.ChatCompletionContentPartUnionParam{
		{OfText: &openai.ChatCompletionContentPartTextParam{Text: "one", Type: constant.ValueOf[constant.Text]()}},
		{OfText: &openai.ChatCompletionContentPartTextParam{Text: "two", Type: constant.ValueOf[constant.Text]()}},
	})
	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionContentPartTextParam{
		{Text: "one", Type: constant.ValueOf[constant.Text]()},
		{Text: "two", Type: constant.ValueOf[constant.Text]()},
	}, params)
}

func TestItemsToMessagesHandlesSystemAndDeveloperRoles(t *testing.T) {
	// Roles other than `user` (e.g. `system` and `developer`) need to be
	// converted appropriately whether provided as simple objects or as full
	// `message` objects.

	sysItems := []agents.TResponseInputItem{
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("setup"),
				},
				Role: responses.EasyInputMessageRoleSystem,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
	}
	result, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems(sysItems))
	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
		{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: param.NewOpt("setup"),
				},
				Name: param.Opt[string]{},
				Role: constant.ValueOf[constant.System](),
			},
		},
	}, result)

	devItems := []agents.TResponseInputItem{
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("debug"),
				},
				Role: responses.EasyInputMessageRoleDeveloper,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
	}
	result, err = agents.ChatCmplConverter().ItemsToMessages(agents.InputItems(devItems))
	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
		{
			OfDeveloper: &openai.ChatCompletionDeveloperMessageParam{
				Content: openai.ChatCompletionDeveloperMessageParamContentUnion{
					OfString: param.NewOpt("debug"),
				},
				Name: param.Opt[string]{},
				Role: constant.ValueOf[constant.Developer](),
			},
		},
	}, result)
}

func TestToolCallConversion(t *testing.T) {
	// Test that tool calls are converted correctly.
	functionCall := agents.TResponseInputItem{
		OfFunctionCall: &responses.ResponseFunctionToolCallParam{
			ID:        param.NewOpt("tool1"),
			CallID:    "abc",
			Name:      "math",
			Arguments: "{}",
			Type:      constant.ValueOf[constant.FunctionCall](),
		},
	}
	result, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{functionCall})
	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
		{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				Name: param.Opt[string]{},
				ToolCalls: []openai.ChatCompletionMessageToolCallParam{
					{
						ID: "abc",
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      "math",
							Arguments: "{}",
						},
						Type: constant.ValueOf[constant.Function](),
					},
				},
				Role: constant.ValueOf[constant.Assistant](),
			},
		},
	}, result)
}

func TestInputMessageWithAllRoles(t *testing.T) {
	// Ensure that a message for each role is passed through by `ItemsToMessages`.

	t.Run("user", func(t *testing.T) {
		v, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{
			{
				OfMessage: &responses.EasyInputMessageParam{
					Content: responses.EasyInputMessageContentUnionParam{
						OfString: param.NewOpt("hi"),
					},
					Role: responses.EasyInputMessageRoleUser,
					Type: responses.EasyInputMessageTypeMessage,
				},
			},
		})
		require.NoError(t, err)
		assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: param.NewOpt("hi"),
					},
					Role: constant.ValueOf[constant.User](),
				},
			},
		}, v)
	})

	t.Run("system", func(t *testing.T) {
		v, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{
			{
				OfMessage: &responses.EasyInputMessageParam{
					Content: responses.EasyInputMessageContentUnionParam{
						OfString: param.NewOpt("hi"),
					},
					Role: responses.EasyInputMessageRoleSystem,
					Type: responses.EasyInputMessageTypeMessage,
				},
			},
		})
		require.NoError(t, err)
		assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
			{
				OfSystem: &openai.ChatCompletionSystemMessageParam{
					Content: openai.ChatCompletionSystemMessageParamContentUnion{
						OfString: param.NewOpt("hi"),
					},
					Role: constant.ValueOf[constant.System](),
				},
			},
		}, v)
	})

	t.Run("developer", func(t *testing.T) {
		v, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{
			{
				OfMessage: &responses.EasyInputMessageParam{
					Content: responses.EasyInputMessageContentUnionParam{
						OfString: param.NewOpt("hi"),
					},
					Role: responses.EasyInputMessageRoleDeveloper,
					Type: responses.EasyInputMessageTypeMessage,
				},
			},
		})
		require.NoError(t, err)
		assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
			{
				OfDeveloper: &openai.ChatCompletionDeveloperMessageParam{
					Content: openai.ChatCompletionDeveloperMessageParamContentUnion{
						OfString: param.NewOpt("hi"),
					},
					Role: constant.ValueOf[constant.Developer](),
				},
			},
		}, v)
	})
}

func TestItemReferenceErrors(t *testing.T) {
	_, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{
		{
			OfItemReference: &responses.ResponseInputItemItemReferenceParam{
				ID:   "item1",
				Type: constant.ValueOf[constant.ItemReference](),
			},
		},
	})
	var target agents.UserError
	assert.ErrorAs(t, err, &target)
}

func TestAssistantMessagesInHistory(t *testing.T) {
	// Test that assistant messages are added to the history.
	v, err := agents.ChatCmplConverter().ItemsToMessages(agents.InputItems{
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("Hello"),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("Hello?"),
				},
				Role: responses.EasyInputMessageRoleAssistant,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("What was my Name?"),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, []openai.ChatCompletionMessageParamUnion{
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: param.NewOpt("Hello"),
				},
				Role: constant.ValueOf[constant.User](),
			},
		},
		{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: param.NewOpt("Hello?"),
				},
				Role: constant.ValueOf[constant.Assistant](),
			},
		},
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: param.NewOpt("What was my Name?"),
				},
				Role: constant.ValueOf[constant.User](),
			},
		},
	}, v)
}
