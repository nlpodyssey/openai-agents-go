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
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/openaitypes"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

type chatCmplConverter struct{}

func ChatCmplConverter() chatCmplConverter { return chatCmplConverter{} }

func (chatCmplConverter) ConvertToolChoice(toolChoice modelsettings.ToolChoice) (openai.ChatCompletionToolChoiceOptionUnionParam, error) {
	switch toolChoice := toolChoice.(type) {
	case nil:
		return openai.ChatCompletionToolChoiceOptionUnionParam{}, nil
	case modelsettings.ToolChoiceString:
		switch toolChoice {
		case "auto", "required", "none":
			return openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: param.NewOpt(toolChoice.String()),
			}, nil
		default:
			return openai.ChatCompletionToolChoiceOptionUnionParam{
				OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
					Function: openai.ChatCompletionNamedToolChoiceFunctionParam{
						Name: toolChoice.String(),
					},
					Type: constant.ValueOf[constant.Function](),
				},
			}, nil
		}
	case modelsettings.ToolChoiceMCP:
		return openai.ChatCompletionToolChoiceOptionUnionParam{},
			NewUserError("ToolChoiceMCP is not supported for Chat Completions models")
	default:
		// This would be an unrecoverable implementation bug, so a panic is appropriate.
		panic(fmt.Errorf("unexpected ToolChoice type %T", toolChoice))
	}
}

func (chatCmplConverter) ConvertResponseFormat(
	finalOutputType OutputTypeInterface,
) (openai.ChatCompletionNewParamsResponseFormatUnion, bool, error) {
	if finalOutputType == nil || finalOutputType.IsPlainText() {
		return openai.ChatCompletionNewParamsResponseFormatUnion{}, false, nil
	}
	schema, err := finalOutputType.JSONSchema()
	if err != nil {
		return openai.ChatCompletionNewParamsResponseFormatUnion{}, false, err
	}

	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
			JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        "final_output",
				Strict:      param.NewOpt(finalOutputType.IsStrictJSONSchema()),
				Description: param.Opt[string]{},
				Schema:      schema,
			},
			Type: constant.ValueOf[constant.JSONSchema](),
		},
	}, true, nil
}

func (chatCmplConverter) MessageToOutputItems(message openai.ChatCompletionMessage) ([]TResponseOutputItem, error) {
	items := make([]TResponseOutputItem, 0)

	// Build content array
	var content []responses.ResponseOutputMessageContentUnion
	if message.Content != "" {
		content = append(content, responses.ResponseOutputMessageContentUnion{
			Text:        message.Content,
			Type:        "output_text",
			Annotations: nil,
		})
	}
	if message.Refusal != "" {
		content = append(content, responses.ResponseOutputMessageContentUnion{
			Refusal: message.Refusal,
			Type:    "refusal",
		})
	}
	if !reflect.ValueOf(message.Audio).IsZero() {
		return nil, errors.New("audio is not currently supported")
	}

	// Create ResponseOutputItemUnion for message if we have content
	if len(content) > 0 {
		messageItem := responses.ResponseOutputItemUnion{
			ID:      FakeResponsesID,
			Content: content,
			Role:    constant.ValueOf[constant.Assistant](),
			Status:  string(responses.ResponseOutputMessageStatusCompleted),
			Type:    "message",
		}
		items = append(items, messageItem)
	}

	// Add function calls
	for _, toolCall := range message.ToolCalls {
		funcCall := responses.ResponseOutputItemUnion{
			ID:        FakeResponsesID,
			CallID:    toolCall.ID,
			Arguments: toolCall.Function.Arguments,
			Name:      toolCall.Function.Name,
			Type:      "function_call",
			Status:    string(responses.ResponseFunctionToolCallStatusCompleted),
		}
		items = append(items, funcCall)
	}

	return items, nil
}

func (conv chatCmplConverter) ExtractTextContentFromEasyInputMessageContentUnionParam(
	content responses.EasyInputMessageContentUnionParam,
) (param.Opt[string], []openai.ChatCompletionContentPartTextParam, error) {
	allContent, err := conv.ExtractAllContentFromEasyInputMessageContentUnionParam(content)
	if err != nil {
		return param.Opt[string]{}, nil, err
	}

	if !param.IsOmitted(allContent.OfString) {
		return allContent.OfString, nil, nil
	}
	if param.IsOmitted(allContent.OfArrayOfContentParts) {
		return param.Opt[string]{}, nil, fmt.Errorf("unexpected .ChatCompletionUserMessageParamContentUnion %+v", allContent)
	}

	out := conv.ExtractTextContentFromChatCompletionContentPartUnionParams(allContent.OfArrayOfContentParts)
	return param.Opt[string]{}, out, nil
}

func (conv chatCmplConverter) ExtractTextContentFromResponseInputMessageContentListParams(
	content responses.ResponseInputMessageContentListParam,
) (param.Opt[string], []openai.ChatCompletionContentPartTextParam, error) {
	allContent, err := conv.ExtractAllContentFromResponseInputContentUnionParams(content)
	if err != nil {
		return param.Opt[string]{}, nil, err
	}

	if !param.IsOmitted(allContent.OfString) {
		return allContent.OfString, nil, nil
	}
	if param.IsOmitted(allContent.OfArrayOfContentParts) {
		return param.Opt[string]{}, nil, fmt.Errorf("unexpected .ChatCompletionUserMessageParamContentUnion %+v", allContent)
	}

	out := conv.ExtractTextContentFromChatCompletionContentPartUnionParams(allContent.OfArrayOfContentParts)
	return param.Opt[string]{}, out, nil
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
			fileParam := c.OfInputFile
			if !fileParam.FileData.Valid() {
				return nil, UserErrorf("only FileData is supported for InputFile %#v", *fileParam)
			}
			if !fileParam.Filename.Valid() {
				return nil, UserErrorf("Filename must be provided for InputFile %#v", *fileParam)
			}
			out[i] = openai.ChatCompletionContentPartUnionParam{
				OfFile: &openai.ChatCompletionContentPartFileParam{
					File: openai.ChatCompletionContentPartFileFileParam{
						FileData: fileParam.FileData,
						Filename: fileParam.Filename,
					},
					Type: constant.ValueOf[constant.File](),
				},
			}
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
				return nil, UserErrorf("unexpected role in EasyInputMessageParam: %q", role)
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
				return nil, UserErrorf("unexpected role in ResponseInputItemMessageParam: %q", role)
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
		} else if fileSearch := item.OfFileSearchCall; !param.IsOmitted(fileSearch) { // 4) function/file-search calls => attach to assistant
			asst := ensureAssistantMessage()
			toolCalls := slices.Clone(asst.ToolCalls)

			queries := fileSearch.Queries
			if queries == nil {
				queries = make([]string, 0)
			}

			jsonArguments, err := json.Marshal(map[string]any{
				"queries": queries,
				"status":  fileSearch.Status,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to JSON-marshal file search call arguments: %w", err)
			}

			newToolCall := openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: fileSearch.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      "file_search_call",
						Arguments: string(jsonArguments),
					},
					Type: constant.ValueOf[constant.Function](),
				},
			}
			toolCalls = append(toolCalls, newToolCall)
			asst.ToolCalls = toolCalls
		} else if funcCall := item.OfFunctionCall; !param.IsOmitted(funcCall) {
			asst := ensureAssistantMessage()
			toolCalls := slices.Clone(asst.ToolCalls)

			arguments := funcCall.Arguments
			if arguments == "" {
				arguments = "{}"
			}

			newToolCall := openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: funcCall.CallID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      funcCall.Name,
						Arguments: arguments,
					},
					Type: constant.ValueOf[constant.Function](),
				},
			}
			toolCalls = append(toolCalls, newToolCall)
			asst.ToolCalls = toolCalls
		} else if funcOutput := item.OfFunctionCallOutput; !param.IsOmitted(funcOutput) {
			flushAssistantMessage()

			// Convert output to string
			var outputStr string
			if !param.IsOmitted(funcOutput.Output.OfString) {
				outputStr = funcOutput.Output.OfString.Value
			} else if !param.IsOmitted(funcOutput.Output.OfResponseFunctionCallOutputItemArray) {
				// Handle array output - serialize to JSON
				b, err := json.Marshal(funcOutput.Output.OfResponseFunctionCallOutputItemArray)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal function output array: %w", err)
				}
				outputStr = string(b)
			} else {
				return nil, UserErrorf("function call output has neither OfString nor OfResponseFunctionCallOutputItemArray set: %+v", funcOutput.Output)
			}

			msg := openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: param.NewOpt(outputStr),
					},
					ToolCallID: funcOutput.CallID,
					Role:       constant.ValueOf[constant.Tool](),
				},
			}
			result = append(result, msg)
		} else if itemRef := item.OfItemReference; !param.IsOmitted(itemRef) { // 6) item reference => handle or return error
			return nil, UserErrorf("encountered an item_reference, which is not supported: %+v", *itemRef)
		} else if !param.IsOmitted(item.OfReasoning) { // 7) reasoning message => not handled
			flushAssistantMessage()
			return nil, nil
		} else { // 8) If we haven't recognized it => fail or ignore
			return nil, UserErrorf("unhandled item type or structure: %+v", item)
		}
	}

	flushAssistantMessage()
	return result, nil
}

func (chatCmplConverter) ToolToOpenai(tool Tool) (*openai.ChatCompletionToolUnionParam, error) {
	functionTool, ok := tool.(FunctionTool)
	if !ok {
		return nil, UserErrorf("hosted tools are not supported with the ChatCompletions API. Got tool %#v", tool)
	}

	var description param.Opt[string]
	if functionTool.Description != "" {
		description = param.NewOpt(functionTool.Description)
	}

	funcTool := openai.ChatCompletionFunctionTool(
		openai.FunctionDefinitionParam{
			Name:        functionTool.Name,
			Description: description,
			Parameters:  functionTool.ParamsJSONSchema,
		},
	)
	return &funcTool, nil
}

func (chatCmplConverter) ConvertHandoffTool(handoff Handoff) openai.ChatCompletionToolUnionParam {
	var description param.Opt[string]
	if handoff.ToolDescription != "" {
		description = param.NewOpt(handoff.ToolDescription)
	}
	return openai.ChatCompletionFunctionTool(
		openai.FunctionDefinitionParam{
			Name:        handoff.ToolName,
			Description: description,
			Parameters:  handoff.InputJSONSchema,
		},
	)
}
