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

package openaitypes

import (
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

func ResponseInputItemUnionParamFromResponseOutputMessage(
	input responses.ResponseOutputMessage,
) responses.ResponseInputItemUnionParam {
	v := ResponseOutputMessageToParam(input)
	return responses.ResponseInputItemUnionParam{
		OfOutputMessage: &v,
	}
}

func ResponseOutputMessageToParam(
	input responses.ResponseOutputMessage,
) responses.ResponseOutputMessageParam {
	return responses.ResponseOutputMessageParam{
		ID:      input.ID,
		Content: ResponseOutputMessageContentUnionSliceToParams(input.Content),
		Status:  input.Status,
		Role:    input.Role,
		Type:    constant.ValueOf[constant.Message](),
	}
}

func ResponseOutputMessageContentUnionSliceToParams(
	input []responses.ResponseOutputMessageContentUnion,
) []responses.ResponseOutputMessageContentUnionParam {
	if input == nil {
		return nil
	}
	out := make([]responses.ResponseOutputMessageContentUnionParam, len(input))
	for i, item := range input {
		out[i] = ResponseOutputMessageContentUnionToParam(item)
	}
	return out
}

func ResponseOutputMessageContentUnionToParam(
	input responses.ResponseOutputMessageContentUnion,
) responses.ResponseOutputMessageContentUnionParam {
	switch input.Type {
	case "output_text":
		v := ResponseOutputTextParamFromResponseOutputMessageContentUnion(input)
		return responses.ResponseOutputMessageContentUnionParam{
			OfOutputText: &v,
		}
	case "refusal":
		v := ResponseOutputRefusalParamFromResponseOutputMessageContentUnion(input)
		return responses.ResponseOutputMessageContentUnionParam{
			OfRefusal: &v,
		}
	default:
		panic(fmt.Errorf("unexpected ResponseOutputMessageContentUnion type %q", input.Type))
	}
}

func ResponseOutputTextParamFromResponseOutputMessageContentUnion(
	input responses.ResponseOutputMessageContentUnion,
) responses.ResponseOutputTextParam {
	return responses.ResponseOutputTextParam{
		Annotations: ResponseOutputTextAnnotationUnionSliceToParams(input.Annotations),
		Text:        input.Text,
		Type:        constant.ValueOf[constant.OutputText](),
	}
}

func ResponseOutputRefusalParamFromResponseOutputMessageContentUnion(
	input responses.ResponseOutputMessageContentUnion,
) responses.ResponseOutputRefusalParam {
	return responses.ResponseOutputRefusalParam{
		Refusal: input.Refusal,
		Type:    constant.ValueOf[constant.Refusal](),
	}
}

func ResponseOutputTextAnnotationUnionSliceToParams(
	input []responses.ResponseOutputTextAnnotationUnion,
) []responses.ResponseOutputTextAnnotationUnionParam {
	if input == nil {
		return nil
	}
	out := make([]responses.ResponseOutputTextAnnotationUnionParam, len(input))
	for i, item := range input {
		out[i] = ResponseOutputTextAnnotationUnionToParam(item)
	}
	return out
}

func ResponseOutputTextAnnotationUnionToParam(
	input responses.ResponseOutputTextAnnotationUnion,
) responses.ResponseOutputTextAnnotationUnionParam {
	switch input.Type {
	case "file_citation":
		v := ResponseOutputTextAnnotationFileCitationParamFromResponseOutputTextAnnotationUnion(input)
		return responses.ResponseOutputTextAnnotationUnionParam{
			OfFileCitation: &v,
		}
	case "url_citation":
		v := ResponseOutputTextAnnotationURLCitationParamFromResponseOutputTextAnnotationUnion(input)
		return responses.ResponseOutputTextAnnotationUnionParam{
			OfURLCitation: &v,
		}
	case "file_path":
		v := ResponseOutputTextAnnotationFilePathParamFromResponseOutputTextAnnotationUnion(input)
		return responses.ResponseOutputTextAnnotationUnionParam{
			OfFilePath: &v,
		}
	default:
		panic(fmt.Errorf("unexpected ResponseOutputTextAnnotationUnion type %q", input.Type))
	}
}

func ResponseOutputTextAnnotationFileCitationParamFromResponseOutputTextAnnotationUnion(
	input responses.ResponseOutputTextAnnotationUnion,
) responses.ResponseOutputTextAnnotationFileCitationParam {
	return responses.ResponseOutputTextAnnotationFileCitationParam{
		FileID: input.FileID,
		Index:  input.Index,
		Type:   constant.ValueOf[constant.FileCitation](),
	}
}

func ResponseOutputTextAnnotationURLCitationParamFromResponseOutputTextAnnotationUnion(
	input responses.ResponseOutputTextAnnotationUnion,
) responses.ResponseOutputTextAnnotationURLCitationParam {
	return responses.ResponseOutputTextAnnotationURLCitationParam{
		EndIndex:   input.EndIndex,
		StartIndex: input.StartIndex,
		Title:      input.Title,
		URL:        input.URL,
		Type:       constant.ValueOf[constant.URLCitation](),
	}
}

func ResponseOutputTextAnnotationFilePathParamFromResponseOutputTextAnnotationUnion(
	input responses.ResponseOutputTextAnnotationUnion,
) responses.ResponseOutputTextAnnotationFilePathParam {
	return responses.ResponseOutputTextAnnotationFilePathParam{
		FileID: input.FileID,
		Index:  input.Index,
		Type:   constant.ValueOf[constant.FilePath](),
	}
}

func ResponseInputItemUnionParamFromResponseFunctionToolCall(
	input responses.ResponseFunctionToolCall,
) responses.ResponseInputItemUnionParam {
	v := ResponseFunctionToolCallToParam(input)
	return responses.ResponseInputItemUnionParam{
		OfFunctionCall: &v,
	}
}

func ResponseFunctionToolCallToParam(
	input responses.ResponseFunctionToolCall,
) responses.ResponseFunctionToolCallParam {
	return responses.ResponseFunctionToolCallParam{
		Arguments: input.Arguments,
		CallID:    input.CallID,
		Name:      input.Name,
		ID:        makeOpt(input.ID),
		Status:    input.Status,
		Type:      constant.ValueOf[constant.FunctionCall](),
	}
}

func ResponseInputItemUnionParamFromResponseInputItemFunctionCallOutputParam(
	input responses.ResponseInputItemFunctionCallOutputParam,
) responses.ResponseInputItemUnionParam {
	return responses.ResponseInputItemUnionParam{
		OfFunctionCallOutput: &input,
	}
}

func ResponseInputItemUnionParamFromResponseInputItemComputerCallOutputParam(
	input responses.ResponseInputItemComputerCallOutputParam,
) responses.ResponseInputItemUnionParam {
	return responses.ResponseInputItemUnionParam{
		OfComputerCallOutput: &input,
	}
}

func ResponseInputItemUnionParamFromResponseInputItemLocalShellCallOutputParam(
	input responses.ResponseInputItemLocalShellCallOutputParam,
) responses.ResponseInputItemUnionParam {
	return responses.ResponseInputItemUnionParam{
		OfLocalShellCallOutput: &input,
	}
}

func ResponseInputItemUnionParamFromResponseReasoningItem(
	input responses.ResponseReasoningItem,
) responses.ResponseInputItemUnionParam {
	v := ResponseReasoningItemToParam(input)
	return responses.ResponseInputItemUnionParam{
		OfReasoning: &v,
	}
}

func ResponseReasoningItemToParam(
	input responses.ResponseReasoningItem,
) responses.ResponseReasoningItemParam {
	var encryptedContent param.Opt[string]
	if input.EncryptedContent != "" {
		encryptedContent = param.NewOpt(input.EncryptedContent)
	}
	return responses.ResponseReasoningItemParam{
		ID:               input.ID,
		Summary:          ResponseReasoningItemSummarySliceToParams(input.Summary),
		Status:           input.Status,
		EncryptedContent: encryptedContent,
		Type:             constant.ValueOf[constant.Reasoning](),
	}
}

func ResponseReasoningItemSummarySliceToParams(
	input []responses.ResponseReasoningItemSummary,
) []responses.ResponseReasoningItemSummaryParam {
	if input == nil {
		return nil
	}
	out := make([]responses.ResponseReasoningItemSummaryParam, len(input))
	for i, item := range input {
		out[i] = ResponseReasoningItemSummaryToParam(item)
	}
	return out
}

func ResponseReasoningItemSummaryToParam(
	input responses.ResponseReasoningItemSummary,
) responses.ResponseReasoningItemSummaryParam {
	return responses.ResponseReasoningItemSummaryParam{
		Text: input.Text,
		Type: constant.ValueOf[constant.SummaryText](),
	}
}

func ResponseOutputItemUnionFromResponseOutputMessage(
	input responses.ResponseOutputMessage,
) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{
		ID:      input.ID,
		Content: input.Content,
		Role:    input.Role,
		Status:  string(input.Status),
		Type:    "message",
	}
}

func ResponseInputItemUnionParamFromResponseOutputItemUnion(
	input responses.ResponseOutputItemUnion,
) responses.ResponseInputItemUnionParam {
	switch input.Type {
	case "message":
		return ResponseInputItemUnionParamFromResponseOutputMessage(responses.ResponseOutputMessage{
			ID:      input.ID,
			Content: input.Content,
			Role:    input.Role,
			Status:  responses.ResponseOutputMessageStatus(input.Status),
			Type:    constant.ValueOf[constant.Message](),
		})
	case "file_search_call":
		return ResponseInputItemUnionParamFromResponseFileSearchToolCall(responses.ResponseFileSearchToolCall{
			ID:      input.ID,
			Queries: input.Queries,
			Status:  responses.ResponseFileSearchToolCallStatus(input.Status),
			Type:    constant.ValueOf[constant.FileSearchCall](),
			Results: input.Results,
		})
	case "function_call":
		return ResponseInputItemUnionParamFromResponseFunctionToolCall(responses.ResponseFunctionToolCall{
			Arguments: input.Arguments,
			CallID:    input.CallID,
			Name:      input.Name,
			Type:      constant.ValueOf[constant.FunctionCall](),
			ID:        input.ID,
			Status:    responses.ResponseFunctionToolCallStatus(input.Status),
		})
	case "web_search_call":
		return ResponseInputItemUnionParamFromResponseFunctionWebSearch(responses.ResponseFunctionWebSearch{
			ID:     input.ID,
			Action: ResponseFunctionWebSearchActionUnionFromResponseOutputItemUnionAction(input.Action),
			Status: responses.ResponseFunctionWebSearchStatus(input.Status),
			Type:   constant.ValueOf[constant.WebSearchCall](),
		})
	case "computer_call":
		return ResponseInputItemUnionParamFromResponseComputerToolCall(responses.ResponseComputerToolCall{
			ID:                  input.ID,
			Action:              ResponseComputerToolCallActionUnionFromResponseOutputItemUnionAction(input.Action),
			CallID:              input.CallID,
			PendingSafetyChecks: input.PendingSafetyChecks,
			Status:              responses.ResponseComputerToolCallStatus(input.Status),
			Type:                responses.ResponseComputerToolCallType(input.Type),
		})
	case "reasoning":
		return ResponseInputItemUnionParamFromResponseReasoningItem(responses.ResponseReasoningItem{
			ID:      input.ID,
			Summary: input.Summary,
			Type:    constant.ValueOf[constant.Reasoning](),
			Status:  responses.ResponseReasoningItemStatus(input.Status),
		})
	default:
		panic(fmt.Errorf("unexpected ResponseOutputItemUnion type %q", input.Type))
	}
}

func ResponseInputItemUnionParamFromResponseFileSearchToolCall(
	input responses.ResponseFileSearchToolCall,
) responses.ResponseInputItemUnionParam {
	v := ResponseFileSearchToolCallToParam(input)
	return responses.ResponseInputItemUnionParam{
		OfFileSearchCall: &v,
	}
}

func ResponseFileSearchToolCallToParam(
	input responses.ResponseFileSearchToolCall,
) responses.ResponseFileSearchToolCallParam {
	return responses.ResponseFileSearchToolCallParam{
		ID:      input.ID,
		Queries: input.Queries,
		Status:  input.Status,
		Results: ResponseFileSearchToolCallResultSliceToParams(input.Results),
		Type:    constant.ValueOf[constant.FileSearchCall](),
	}
}

func ResponseFileSearchToolCallResultSliceToParams(
	input []responses.ResponseFileSearchToolCallResult,
) []responses.ResponseFileSearchToolCallResultParam {
	if input == nil {
		return nil
	}
	out := make([]responses.ResponseFileSearchToolCallResultParam, len(input))
	for i, item := range input {
		out[i] = ResponseFileSearchToolCallResultToParam(item)
	}
	return out
}

func ResponseFileSearchToolCallResultToParam(
	input responses.ResponseFileSearchToolCallResult,
) responses.ResponseFileSearchToolCallResultParam {
	return responses.ResponseFileSearchToolCallResultParam{
		FileID:     makeOpt(input.FileID),
		Filename:   makeOpt(input.Filename),
		Score:      makeOpt(input.Score),
		Text:       makeOpt(input.Text),
		Attributes: ResponseFileSearchToolCallResultAttributeUnionMapToParamMap(input.Attributes),
	}
}

func ResponseFileSearchToolCallResultAttributeUnionMapToParamMap(
	input map[string]responses.ResponseFileSearchToolCallResultAttributeUnion,
) map[string]responses.ResponseFileSearchToolCallResultAttributeUnionParam {
	if input == nil {
		return nil
	}
	out := make(map[string]responses.ResponseFileSearchToolCallResultAttributeUnionParam)
	for k, v := range input {
		out[k] = ResponseFileSearchToolCallResultAttributeUnionToParam(v)
	}
	return out
}

func ResponseFileSearchToolCallResultAttributeUnionToParam(
	input responses.ResponseFileSearchToolCallResultAttributeUnion,
) responses.ResponseFileSearchToolCallResultAttributeUnionParam {
	return responses.ResponseFileSearchToolCallResultAttributeUnionParam{
		OfString: makeOpt(input.OfString),
		OfFloat:  makeOpt(input.OfFloat),
		OfBool:   makeOpt(input.OfBool),
	}
}

func ResponseInputItemUnionParamFromResponseFunctionWebSearch(
	input responses.ResponseFunctionWebSearch,
) responses.ResponseInputItemUnionParam {
	v := ResponseFunctionWebSearchToParam(input)
	return responses.ResponseInputItemUnionParam{
		OfWebSearchCall: &v,
	}
}

func ResponseFunctionWebSearchToParam(
	input responses.ResponseFunctionWebSearch,
) responses.ResponseFunctionWebSearchParam {
	return responses.ResponseFunctionWebSearchParam{
		ID:     input.ID,
		Action: ResponseFunctionWebSearchActionUnionToParam(input.Action),
		Status: input.Status,
		Type:   constant.ValueOf[constant.WebSearchCall](),
	}
}

func ResponseFunctionWebSearchActionUnionToParam(
	input responses.ResponseFunctionWebSearchActionUnion,
) responses.ResponseFunctionWebSearchActionUnionParam {
	switch input.Type {
	case "search":
		return responses.ResponseFunctionWebSearchActionUnionParam{
			OfSearch: &responses.ResponseFunctionWebSearchActionSearchParam{
				Query: input.Query,
				Type:  constant.ValueOf[constant.Search](),
			},
		}
	case "open_page":
		return responses.ResponseFunctionWebSearchActionUnionParam{
			OfOpenPage: &responses.ResponseFunctionWebSearchActionOpenPageParam{
				URL:  input.URL,
				Type: constant.ValueOf[constant.OpenPage](),
			},
		}
	case "find":
		return responses.ResponseFunctionWebSearchActionUnionParam{
			OfFind: &responses.ResponseFunctionWebSearchActionFindParam{
				Pattern: input.Pattern,
				URL:     input.URL,
				Type:    constant.ValueOf[constant.Find](),
			},
		}
	default:
		return responses.ResponseFunctionWebSearchActionUnionParam{}
	}
}

func ResponseFunctionWebSearchActionUnionFromResponseOutputItemUnionAction(
	input responses.ResponseOutputItemUnionAction,
) responses.ResponseFunctionWebSearchActionUnion {
	return responses.ResponseFunctionWebSearchActionUnion{
		Query:   input.Query,
		Type:    input.Type,
		URL:     input.URL,
		Pattern: input.Pattern,
	}
}

func ResponseInputItemUnionParamFromResponseComputerToolCall(
	input responses.ResponseComputerToolCall,
) responses.ResponseInputItemUnionParam {
	v := ResponseComputerToolCallToParam(input)
	return responses.ResponseInputItemUnionParam{
		OfComputerCall: &v,
	}
}

func ResponseInputItemUnionParamFromResponseOutputItemLocalShellCall(
	input responses.ResponseOutputItemLocalShellCall,
) responses.ResponseInputItemUnionParam {
	return responses.ResponseInputItemUnionParam{
		OfLocalShellCall: &responses.ResponseInputItemLocalShellCallParam{
			ID:     input.ID,
			Action: ResponseInputItemLocalShellCallActionParamFromResponseOutputItemLocalShellCallAction(input.Action),
			CallID: input.CallID,
			Status: input.Status,
			Type:   input.Type,
		},
	}
}

func ResponseInputItemLocalShellCallActionParamFromResponseOutputItemLocalShellCallAction(
	input responses.ResponseOutputItemLocalShellCallAction,
) responses.ResponseInputItemLocalShellCallActionParam {
	return responses.ResponseInputItemLocalShellCallActionParam{
		Command:          input.Command,
		Env:              input.Env,
		TimeoutMs:        param.NewOpt(input.TimeoutMs),
		User:             param.NewOpt(input.User),
		WorkingDirectory: param.NewOpt(input.WorkingDirectory),
		Type:             input.Type,
	}
}

func ResponseComputerToolCallActionUnionFromResponseOutputItemUnionAction(
	input responses.ResponseOutputItemUnionAction,
) responses.ResponseComputerToolCallActionUnion {
	return responses.ResponseComputerToolCallActionUnion{
		Button:  input.Button,
		Type:    input.Type,
		X:       input.X,
		Y:       input.Y,
		Path:    input.Path,
		Keys:    input.Keys,
		ScrollX: input.ScrollX,
		ScrollY: input.ScrollY,
		Text:    input.Text,
	}
}

func ResponseOutputItemLocalShellCallActionFromResponseOutputItemUnionAction(
	input responses.ResponseOutputItemUnionAction,
) responses.ResponseOutputItemLocalShellCallAction {
	return responses.ResponseOutputItemLocalShellCallAction{
		Command:          input.Command,
		Env:              input.Env,
		Type:             constant.ValueOf[constant.Exec](),
		TimeoutMs:        input.TimeoutMs,
		User:             input.User,
		WorkingDirectory: input.WorkingDirectory,
	}
}

func ResponseComputerToolCallToParam(
	input responses.ResponseComputerToolCall,
) responses.ResponseComputerToolCallParam {
	return responses.ResponseComputerToolCallParam{
		ID:                  input.ID,
		Action:              ResponseComputerToolCallActionUnionToParam(input.Action),
		CallID:              input.CallID,
		PendingSafetyChecks: ResponseComputerToolCallPendingSafetyCheckSliceToParams(input.PendingSafetyChecks),
		Status:              input.Status,
		Type:                input.Type,
	}
}

func ResponseComputerToolCallActionUnionToParam(
	input responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionUnionParam {
	switch input.Type {
	case "click":
		v := ResponseComputerToolCallActionClickParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfClick: &v,
		}
	case "double_click":
		v := ResponseComputerToolCallActionDoubleClickParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfDoubleClick: &v,
		}
	case "drag":
		v := ResponseComputerToolCallActionDragParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfDrag: &v,
		}
	case "keypress":
		v := ResponseComputerToolCallActionKeypressParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfKeypress: &v,
		}
	case "move":
		v := ResponseComputerToolCallActionMoveParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfMove: &v,
		}
	case "screenshot":
		v := ResponseComputerToolCallActionScreenshotParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfScreenshot: &v,
		}
	case "scroll":
		v := ResponseComputerToolCallActionScrollParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfScroll: &v,
		}
	case "type":
		v := ResponseComputerToolCallActionTypeParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfType: &v,
		}
	case "wait":
		v := ResponseComputerToolCallActionWaitParamFromResponseComputerToolCallActionUnion(input)
		return responses.ResponseComputerToolCallActionUnionParam{
			OfWait: &v,
		}
	default:
		panic(fmt.Errorf("unexpected ResponseComputerToolCallActionUnion type %q", input.Type))
	}
}

func ResponseComputerToolCallActionClickParamFromResponseComputerToolCallActionUnion(
	input responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionClickParam {
	return responses.ResponseComputerToolCallActionClickParam{
		Button: input.Button,
		X:      input.X,
		Y:      input.Y,
		Type:   constant.ValueOf[constant.Click](),
	}
}

func ResponseComputerToolCallActionDoubleClickParamFromResponseComputerToolCallActionUnion(
	input responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionDoubleClickParam {
	return responses.ResponseComputerToolCallActionDoubleClickParam{
		X:    input.X,
		Y:    input.Y,
		Type: constant.ValueOf[constant.DoubleClick](),
	}
}

func ResponseComputerToolCallActionDragParamFromResponseComputerToolCallActionUnion(
	input responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionDragParam {
	return responses.ResponseComputerToolCallActionDragParam{
		Path: ResponseComputerToolCallActionDragPathSliceToParams(input.Path),
		Type: constant.ValueOf[constant.Drag](),
	}
}

func ResponseComputerToolCallActionDragPathSliceToParams(
	input []responses.ResponseComputerToolCallActionDragPath,
) []responses.ResponseComputerToolCallActionDragPathParam {
	if input == nil {
		return nil
	}
	out := make([]responses.ResponseComputerToolCallActionDragPathParam, len(input))
	for i, item := range input {
		out[i] = ResponseComputerToolCallActionDragPathToParam(item)
	}
	return out
}

func ResponseComputerToolCallActionDragPathToParam(
	input responses.ResponseComputerToolCallActionDragPath,
) responses.ResponseComputerToolCallActionDragPathParam {
	return responses.ResponseComputerToolCallActionDragPathParam{
		X: input.X,
		Y: input.Y,
	}
}

func ResponseComputerToolCallActionKeypressParamFromResponseComputerToolCallActionUnion(
	input responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionKeypressParam {
	return responses.ResponseComputerToolCallActionKeypressParam{
		Keys: input.Keys,
		Type: constant.ValueOf[constant.Keypress](),
	}
}

func ResponseComputerToolCallActionMoveParamFromResponseComputerToolCallActionUnion(
	input responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionMoveParam {
	return responses.ResponseComputerToolCallActionMoveParam{
		X:    input.X,
		Y:    input.Y,
		Type: constant.ValueOf[constant.Move](),
	}
}

func ResponseComputerToolCallActionScreenshotParamFromResponseComputerToolCallActionUnion(
	_ responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionScreenshotParam {
	return responses.ResponseComputerToolCallActionScreenshotParam{
		Type: constant.ValueOf[constant.Screenshot](),
	}
}

func ResponseComputerToolCallActionScrollParamFromResponseComputerToolCallActionUnion(
	input responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionScrollParam {
	return responses.ResponseComputerToolCallActionScrollParam{
		ScrollX: input.ScrollX,
		ScrollY: input.ScrollY,
		X:       input.X,
		Y:       input.Y,
		Type:    constant.ValueOf[constant.Scroll](),
	}
}

func ResponseComputerToolCallActionTypeParamFromResponseComputerToolCallActionUnion(
	input responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionTypeParam {
	return responses.ResponseComputerToolCallActionTypeParam{
		Text: input.Text,
		Type: constant.ValueOf[constant.Type](),
	}
}

func ResponseComputerToolCallActionWaitParamFromResponseComputerToolCallActionUnion(
	_ responses.ResponseComputerToolCallActionUnion,
) responses.ResponseComputerToolCallActionWaitParam {
	return responses.ResponseComputerToolCallActionWaitParam{
		Type: constant.ValueOf[constant.Wait](),
	}
}

func ResponseComputerToolCallPendingSafetyCheckSliceToParams(
	input []responses.ResponseComputerToolCallPendingSafetyCheck,
) []responses.ResponseComputerToolCallPendingSafetyCheckParam {
	if input == nil {
		return nil
	}
	out := make([]responses.ResponseComputerToolCallPendingSafetyCheckParam, len(input))
	for i, item := range input {
		out[i] = ResponseComputerToolCallPendingSafetyCheckToParam(item)
	}
	return out
}

func ResponseComputerToolCallPendingSafetyCheckToParam(
	input responses.ResponseComputerToolCallPendingSafetyCheck,
) responses.ResponseComputerToolCallPendingSafetyCheckParam {
	return responses.ResponseComputerToolCallPendingSafetyCheckParam{
		ID:      input.ID,
		Code:    makeOpt(input.Code),
		Message: makeOpt(input.Message),
	}
}

func ChatCompletionAssistantMessageParamContentArrayOfContentPartUnionSliceFromChatCompletionContentPartTextParamSlice(
	input []openai.ChatCompletionContentPartTextParam,
) []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion {
	if input == nil {
		return nil
	}
	result := make([]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion, len(input))
	for i, item := range input {
		result[i] = ChatCompletionAssistantMessageParamContentArrayOfContentPartUnionFromChatCompletionContentPartTextParam(item)
	}
	return result
}

func ChatCompletionAssistantMessageParamContentArrayOfContentPartUnionFromChatCompletionContentPartTextParam(
	input openai.ChatCompletionContentPartTextParam,
) openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion {
	return openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
		OfText: &openai.ChatCompletionContentPartTextParam{
			Text: input.Text,
			Type: constant.ValueOf[constant.Text](),
		},
	}
}

func ReasoningFromParam(input responses.ReasoningParam) openai.Reasoning {
	return openai.Reasoning{
		Effort:  input.Effort,
		Summary: input.Summary,
	}
}

func ResponseOutputMessageContentUnionFromResponseStreamEventUnionPart(
	input responses.ResponseStreamEventUnionPart,
) responses.ResponseOutputMessageContentUnion {
	return responses.ResponseOutputMessageContentUnion{
		Annotations: input.Annotations,
		Text:        input.Text,
		Type:        input.Type,
		Refusal:     input.Refusal,
		JSON:        input.JSON,
	}
}

func ResponseOutputMessageFromResponseOutputItemUnion(
	input responses.ResponseOutputItemUnion,
) responses.ResponseOutputMessage {
	return responses.ResponseOutputMessage{
		ID:      input.ID,
		Content: input.Content,
		Role:    input.Role,
		Status:  responses.ResponseOutputMessageStatus(input.Status),
		Type:    constant.ValueOf[constant.Message](),
	}
}

func ResponseInputItemUnionParamFromResponseOutputItemMcpListTools(
	input responses.ResponseOutputItemMcpListTools,
) responses.ResponseInputItemUnionParam {
	v := ResponseInputItemMcpListToolsParamFromResponseOutputItemMcpListTools(input)
	return responses.ResponseInputItemUnionParam{
		OfMcpListTools: &v,
	}
}

func ResponseInputItemMcpListToolsParamFromResponseOutputItemMcpListTools(
	input responses.ResponseOutputItemMcpListTools,
) responses.ResponseInputItemMcpListToolsParam {
	return responses.ResponseInputItemMcpListToolsParam{
		ID:          input.ID,
		ServerLabel: input.ServerLabel,
		Tools:       ResponseInputItemMcpListToolsToolParamSliceFromResponseOutputItemMcpListToolsToolSlice(input.Tools),
		Error:       makeOpt(input.Error),
		Type:        constant.ValueOf[constant.McpListTools](),
	}
}
func ResponseInputItemMcpListToolsToolParamSliceFromResponseOutputItemMcpListToolsToolSlice(
	input []responses.ResponseOutputItemMcpListToolsTool,
) []responses.ResponseInputItemMcpListToolsToolParam {
	if input == nil {
		return nil
	}
	out := make([]responses.ResponseInputItemMcpListToolsToolParam, len(input))
	for i, item := range input {
		out[i] = ResponseInputItemMcpListToolsToolParamFromResponseOutputItemMcpListToolsTool(item)
	}
	return out
}

func ResponseInputItemMcpListToolsToolParamFromResponseOutputItemMcpListToolsTool(
	input responses.ResponseOutputItemMcpListToolsTool,
) responses.ResponseInputItemMcpListToolsToolParam {
	return responses.ResponseInputItemMcpListToolsToolParam{
		InputSchema: input.InputSchema,
		Name:        input.Name,
		Description: makeOpt(input.Description),
		Annotations: input.Annotations,
	}
}

func ResponseInputItemUnionParamFromResponseOutputItemMcpApprovalRequest(
	input responses.ResponseOutputItemMcpApprovalRequest,
) responses.ResponseInputItemUnionParam {
	v := ResponseInputItemMcpApprovalRequestParamFromResponseOutputItemMcpApprovalRequest(input)
	return responses.ResponseInputItemUnionParam{
		OfMcpApprovalRequest: &v,
	}
}

func ResponseInputItemMcpApprovalRequestParamFromResponseOutputItemMcpApprovalRequest(
	input responses.ResponseOutputItemMcpApprovalRequest,
) responses.ResponseInputItemMcpApprovalRequestParam {
	return responses.ResponseInputItemMcpApprovalRequestParam{
		ID:          input.ID,
		Arguments:   input.Arguments,
		Name:        input.Name,
		ServerLabel: input.ServerLabel,
		Type:        constant.ValueOf[constant.McpApprovalRequest](),
	}
}

func ResponseInputItemUnionParamFromResponseInputItemMcpApprovalResponseParam(
	input responses.ResponseInputItemMcpApprovalResponseParam,
) responses.ResponseInputItemUnionParam {
	return responses.ResponseInputItemUnionParam{
		OfMcpApprovalResponse: &input,
	}
}

func makeOpt[T comparable](v T) param.Opt[T] {
	var zero T
	if v == zero {
		return param.Opt[T]{}
	}
	return param.NewOpt(v)
}
