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
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/openaitypes"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
	"github.com/openai/openai-go/shared/constant"
)

type chatCmplConverter struct{}

func ChatCmplConverter() chatCmplConverter { return chatCmplConverter{} }

func (chatCmplConverter) ConvertToolChoice(toolChoice string) (openai.ChatCompletionToolChoiceOptionUnionParam, bool) {
	switch toolChoice {
	case "":
		return openai.ChatCompletionToolChoiceOptionUnionParam{}, false
	case "auto", "required", "none":
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt(toolChoice),
		}, true
	default:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfChatCompletionNamedToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: toolChoice,
				},
				Type: constant.ValueOf[constant.Function](),
			},
		}, true
	}
}

func (chatCmplConverter) ConvertResponseFormat(
	finalOutputSchema AgentOutputSchemaInterface,
) (openai.ChatCompletionNewParamsResponseFormatUnion, bool) {
	if finalOutputSchema == nil || finalOutputSchema.IsPlainText() {
		return openai.ChatCompletionNewParamsResponseFormatUnion{}, false
	}
	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        "final_output",
				Strict:      param.NewOpt(finalOutputSchema.IsStrictJSONSchema()),
				Description: param.Null[string](),
				Schema:      finalOutputSchema.JSONSchema(),
			},
			Type: constant.ValueOf[constant.JSONSchema](),
		},
	}, true
}

func (chatCmplConverter) MessageToOutputItems(message openai.ChatCompletionMessage) []TResponseOutputItem {
	items := make([]TResponseOutputItem, 0)

	messageItem := responses.ResponseOutputItemUnion{
		ID:      FakeResponsesID,
		Content: nil,
		Role:    constant.ValueOf[constant.Assistant](),
		Status:  string(responses.ResponseOutputMessageStatusCompleted),
		Type:    "message",
	}
	if message.Content != "" {
		messageItem.Content = append(messageItem.Content, responses.ResponseOutputMessageContentUnion{
			Text:        message.Content,
			Type:        "output_text",
			Annotations: nil,
		})
	}
	if message.Refusal != "" {
		messageItem.Content = append(messageItem.Content, responses.ResponseOutputMessageContentUnion{
			Refusal: message.Refusal,
			Type:    "refusal",
		})
	}
	if !reflect.DeepEqual(message.Audio, openai.ChatCompletionAudio{}) {
		panic(errors.New("audio is not currently supported"))
	}

	if len(messageItem.Content) > 0 {
		items = append(items, messageItem)
	}

	for _, toolCall := range message.ToolCalls {
		items = append(items, responses.ResponseOutputItemUnion{
			ID:        FakeResponsesID,
			CallID:    toolCall.ID,
			Arguments: toolCall.Function.Arguments,
			Name:      toolCall.Function.Name,
			Type:      "function_call",
		})
	}

	return items
}

func (conv chatCmplConverter) ExtractTextContentFromEasyInputMessageContentUnionParam(
	content responses.EasyInputMessageContentUnionParam,
) (param.Opt[string], []openai.ChatCompletionContentPartTextParam, error) {
	allContent, err := conv.ExtractAllContentFromEasyInputMessageContentUnionParam(content)
	if err != nil {
		return param.Null[string](), nil, err
	}

	if !param.IsOmitted(allContent.OfString) {
		return allContent.OfString, nil, nil
	}
	if param.IsOmitted(allContent.OfArrayOfContentParts) {
		return param.Null[string](), nil, fmt.Errorf("unexpected .ChatCompletionUserMessageParamContentUnion %+v", allContent)
	}

	out := conv.ExtractTextContentFromChatCompletionContentPartUnionParams(allContent.OfArrayOfContentParts)
	return param.Null[string](), out, nil
}

func (conv chatCmplConverter) ExtractTextContentFromResponseInputMessageContentListParams(
	content responses.ResponseInputMessageContentListParam,
) (param.Opt[string], []openai.ChatCompletionContentPartTextParam, error) {
	allContent, err := conv.ExtractAllContentFromResponseInputContentUnionParams(content)
	if err != nil {
		return param.Null[string](), nil, err
	}

	if !param.IsOmitted(allContent.OfString) {
		return allContent.OfString, nil, nil
	}
	if param.IsOmitted(allContent.OfArrayOfContentParts) {
		return param.Null[string](), nil, fmt.Errorf("unexpected .ChatCompletionUserMessageParamContentUnion %+v", allContent)
	}

	out := conv.ExtractTextContentFromChatCompletionContentPartUnionParams(allContent.OfArrayOfContentParts)
	return param.Null[string](), out, nil
}

func (conv chatCmplConverter) ExtractTextContentFromChatCompletionContentPartUnionParams(
	arrayOfContentParts []openai.ChatCompletionContentPartUnionParam,
) []openai.ChatCompletionContentPartTextParam {
	out := make([]openai.ChatCompletionContentPartTextParam, 0)
	for _, c := range arrayOfContentParts {
		if !param.IsOmitted(c.OfText) {
			out = append(out, *c.OfText)
		}
	}
	return out
}

func (conv chatCmplConverter) ExtractAllContentFromEasyInputMessageContentUnionParam(
	content responses.EasyInputMessageContentUnionParam,
) (*openai.ChatCompletionUserMessageParamContentUnion, error) {
	if !param.IsOmitted(content.OfString) {
		return &openai.ChatCompletionUserMessageParamContentUnion{
			OfString: content.OfString,
		}, nil
	}
	if param.IsOmitted(content.OfInputItemContentList) {
		return nil, UserErrorf("unknown content: %+v", content)
	}
	return conv.ExtractAllContentFromResponseInputContentUnionParams(content.OfInputItemContentList)
}

func (chatCmplConverter) ExtractAllContentFromResponseInputContentUnionParams(
	inputItemContentList []responses.ResponseInputContentUnionParam,
) (*openai.ChatCompletionUserMessageParamContentUnion, error) {
	out := make([]openai.ChatCompletionContentPartUnionParam, len(inputItemContentList))

	for i, c := range inputItemContentList {
		if !param.IsOmitted(c.OfInputText) {
			out[i] = openai.ChatCompletionContentPartUnionParam{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Text: c.OfInputText.Text,
					Type: constant.ValueOf[constant.Text](),
				},
			}
		} else if !param.IsOmitted(c.OfInputImage) {
			if param.IsOmitted(c.OfInputImage.ImageURL) || c.OfInputImage.ImageURL.Value == "" {
				return nil, UserErrorf("only image URLs are supported for input_image %+v", c.OfInputImage)
			}
			detail := string(c.OfInputImage.Detail)
			if detail == "" {
				detail = "auto"
			}
			out[i] = openai.ChatCompletionContentPartUnionParam{
				OfImageURL: &openai.ChatCompletionContentPartImageParam{
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL:    c.OfInputImage.ImageURL.Value,
						Detail: detail,
					},
				},
			}
		} else if !param.IsOmitted(c.OfInputFile) {
			return nil, UserErrorf("file uploads are not supported for chat completions %+v", c)
		} else {
			return nil, UserErrorf("unknown content: %+v", c)
		}
	}

	return &openai.ChatCompletionUserMessageParamContentUnion{
		OfArrayOfContentParts: out,
	}, nil
}

// ItemsToMessages converts a sequence of 'Item' objects into a list of
// openai.ChatCompletionMessageParamUnion.
//
// Rules:
// - EasyInputMessage or InputMessage (role=user) => openai.ChatCompletionUserMessageParam
// - EasyInputMessage or InputMessage (role=system) => openai.ChatCompletionSystemMessageParam
// - EasyInputMessage or InputMessage (role=developer) => openai.ChatCompletionDeveloperMessageParam
// - InputMessage (role=assistant) => Start or flush an openai.ChatCompletionAssistantMessageParam
// - response_output_message => Also produces/flushes an openai.ChatCompletionAssistantMessageParam
// - tool calls get attached to the *current* assistant message, or create one if none.
// - tool outputs => openai.ChatCompletionToolMessageParam
func (conv chatCmplConverter) ItemsToMessages(items Input) ([]openai.ChatCompletionMessageParamUnion, error) {
	switch v := items.(type) {
	case InputString:
		return []openai.ChatCompletionMessageParamUnion{{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: param.NewOpt(v.String()),
				},
				Role: constant.ValueOf[constant.User](),
			},
		}}, nil
	case InputItems:
		return conv.itemsToMessages(v)
	default:
		// This would be an unrecoverable implementation bug, so a panic is appropriate.
		panic(fmt.Errorf("unexpected Input type %T", v))
	}
}

func (conv chatCmplConverter) itemsToMessages(items []TResponseInputItem) ([]openai.ChatCompletionMessageParamUnion, error) {
	var result []openai.ChatCompletionMessageParamUnion

	var currentAssistantMsg *openai.ChatCompletionAssistantMessageParam

	flushAssistantMessage := func() {
		if currentAssistantMsg != nil {
			result = append(result, openai.ChatCompletionMessageParamUnion{
				OfAssistant: currentAssistantMsg,
			})
			currentAssistantMsg = nil
		}
	}

	ensureAssistantMessage := func() *openai.ChatCompletionAssistantMessageParam {
		if currentAssistantMsg == nil {
			currentAssistantMsg = &openai.ChatCompletionAssistantMessageParam{
				Role: constant.ValueOf[constant.Assistant](),
			}
		}
		return currentAssistantMsg
	}

	for _, item := range items {
		if easyMsg := item.OfMessage; !param.IsOmitted(easyMsg) { // 1) Check easy input message
			role := easyMsg.Role
			content := easyMsg.Content

			switch role {
			case responses.EasyInputMessageRoleUser:
				flushAssistantMessage()
				c, err := conv.ExtractAllContentFromEasyInputMessageContentUnionParam(content)
				if err != nil {
					return nil, err
				}
				msgUser := openai.ChatCompletionMessageParamUnion{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: *c,
						Role:    constant.ValueOf[constant.User](),
					},
				}
				result = append(result, msgUser)
			case responses.EasyInputMessageRoleSystem:
				flushAssistantMessage()
				str, arr, err := conv.ExtractTextContentFromEasyInputMessageContentUnionParam(content)
				if err != nil {
					return nil, err
				}
				msgSystem := openai.ChatCompletionMessageParamUnion{
					OfSystem: &openai.ChatCompletionSystemMessageParam{
						Content: openai.ChatCompletionSystemMessageParamContentUnion{
							OfString:              str,
							OfArrayOfContentParts: arr,
						},
						Role: constant.ValueOf[constant.System](),
					},
				}
				result = append(result, msgSystem)
			case responses.EasyInputMessageRoleDeveloper:
				flushAssistantMessage()
				str, arr, err := conv.ExtractTextContentFromEasyInputMessageContentUnionParam(content)
				if err != nil {
					return nil, err
				}
				msgDeveloper := openai.ChatCompletionMessageParamUnion{
					OfDeveloper: &openai.ChatCompletionDeveloperMessageParam{
						Content: openai.ChatCompletionDeveloperMessageParamContentUnion{
							OfString:              str,
							OfArrayOfContentParts: arr,
						},
						Role: constant.ValueOf[constant.Developer](),
					},
				}
				result = append(result, msgDeveloper)
			case responses.EasyInputMessageRoleAssistant:
				flushAssistantMessage()
				str, arr, err := conv.ExtractTextContentFromEasyInputMessageContentUnionParam(content)
				if err != nil {
					return nil, err
				}
				msgAssistant := openai.ChatCompletionMessageParamUnion{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfString:              str,
							OfArrayOfContentParts: openaitypes.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnionSliceFromChatCompletionContentPartTextParamSlice(arr),
						},
						Role: constant.ValueOf[constant.Assistant](),
					},
				}
				result = append(result, msgAssistant)
			default:
				return nil, UserErrorf("unexpected rone in EasyInputMessageParam: %q", role)
			}
		} else if inMsg := item.OfInputMessage; !param.IsOmitted(inMsg) { // 2) Check input message
			role := inMsg.Role
			content := inMsg.Content
			flushAssistantMessage()

			switch role {
			case "user":
				c, err := conv.ExtractAllContentFromResponseInputContentUnionParams(content)
				if err != nil {
					return nil, err
				}
				msgUser := openai.ChatCompletionMessageParamUnion{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: *c,
						Role:    constant.ValueOf[constant.User](),
					},
				}
				result = append(result, msgUser)
			case "system":
				str, arr, err := conv.ExtractTextContentFromResponseInputMessageContentListParams(content)
				if err != nil {
					return nil, err
				}
				msgSystem := openai.ChatCompletionMessageParamUnion{
					OfSystem: &openai.ChatCompletionSystemMessageParam{
						Content: openai.ChatCompletionSystemMessageParamContentUnion{
							OfString:              str,
							OfArrayOfContentParts: arr,
						},
						Role: constant.ValueOf[constant.System](),
					},
				}
				result = append(result, msgSystem)
			case "developer":
				str, arr, err := conv.ExtractTextContentFromResponseInputMessageContentListParams(content)
				if err != nil {
					return nil, err
				}
				msgDeveloper := openai.ChatCompletionMessageParamUnion{
					OfDeveloper: &openai.ChatCompletionDeveloperMessageParam{
						Content: openai.ChatCompletionDeveloperMessageParamContentUnion{
							OfString:              str,
							OfArrayOfContentParts: arr,
						},
						Role: constant.ValueOf[constant.Developer](),
					},
				}
				result = append(result, msgDeveloper)
			default:
				return nil, UserErrorf("unexpected role in input_message: %+q", role)
			}
		} else if respMsg := item.OfOutputMessage; !param.IsOmitted(respMsg) { // 3) response output message => assistant
			flushAssistantMessage()
			newAsst := &openai.ChatCompletionAssistantMessageParam{
				Role: constant.ValueOf[constant.Assistant](),
			}
			contents := respMsg.Content

			var textSegments []string

			for _, c := range contents {
				switch {
				case !param.IsOmitted(c.OfOutputText):
					textSegments = append(textSegments, c.OfOutputText.Text)
				case !param.IsOmitted(c.OfRefusal):
					newAsst.Refusal = param.NewOpt(c.OfRefusal.Refusal)
				default:
					return nil, UserErrorf("unknown content type in ResponseOutputMessage: %+v", c)
				}
			}

			if len(textSegments) > 0 {
				newAsst.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: param.NewOpt(strings.Join(textSegments, "\n")),
				}
			}

			currentAssistantMsg = newAsst
		} else if funcCall := item.OfFunctionCall; !param.IsOmitted(funcCall) { // 4) function calls => attach to assistant
			asst := ensureAssistantMessage()
			toolCalls := slices.Clone(asst.ToolCalls)

			arguments := funcCall.Arguments
			if arguments == "" {
				arguments = "{}"
			}

			newToolCall := openai.ChatCompletionMessageToolCallParam{
				ID: funcCall.CallID,
				Function: openai.ChatCompletionMessageToolCallFunctionParam{
					Name:      funcCall.Name,
					Arguments: arguments,
				},
				Type: constant.ValueOf[constant.Function](),
			}
			toolCalls = append(toolCalls, newToolCall)
			asst.ToolCalls = toolCalls
		} else if funcOutput := item.OfFunctionCallOutput; !param.IsOmitted(funcOutput) { // 5) function call output => tool message
			flushAssistantMessage()
			msg := openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: param.NewOpt(funcOutput.Output),
					},
					ToolCallID: funcOutput.CallID,
					Role:       constant.ValueOf[constant.Tool](),
				},
			}
			result = append(result, msg)
		} else if itemRef := item.OfItemReference; !param.IsOmitted(itemRef) { // 6) item reference => handle or raise
			return nil, UserErrorf("encountered an item_reference, which is not supported: %+v", *itemRef)
		} else { // 7) If we haven't recognized it => fail or ignore
			return nil, UserErrorf("unhandled item type or structure: %+v", item)
		}
	}

	flushAssistantMessage()
	return result, nil
}

func (chatCmplConverter) ToolToOpenai(tool Tool) openai.ChatCompletionToolParam {
	switch tool := tool.(type) {
	case FunctionTool:
		description := param.Null[string]()
		if tool.Description != "" {
			description = param.NewOpt(tool.Description)
		}
		return openai.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: description,
				Parameters:  tool.ParamsJSONSchema,
			},
			Type: constant.ValueOf[constant.Function](),
		}
	default:
		// This would be an unrecoverable implementation bug, so a panic is appropriate.
		panic(fmt.Errorf("unexpected Tool type %T", tool))
	}
}

func (chatCmplConverter) ConvertHandoffTool(handoff Handoff) openai.ChatCompletionToolParam {
	description := param.Null[string]()
	if handoff.ToolDescription != "" {
		description = param.NewOpt(handoff.ToolDescription)
	}
	return openai.ChatCompletionToolParam{
		Function: shared.FunctionDefinitionParam{
			Name:        handoff.ToolName,
			Description: description,
			Parameters:  handoff.InputJSONSchema,
		},
		Type: constant.ValueOf[constant.Function](),
	}
}
