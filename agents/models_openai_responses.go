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
	"fmt"
	"iter"
	"log/slog"
	"reflect"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
)

// OpenAIResponsesModel is an implementation of Model that uses the OpenAI Responses API.
type OpenAIResponsesModel struct {
	Model  openai.ChatModel
	client OpenaiClient
}

func NewOpenAIResponsesModel(model openai.ChatModel, client OpenaiClient) OpenAIResponsesModel {
	return OpenAIResponsesModel{
		Model:  model,
		client: client,
	}
}

func (m OpenAIResponsesModel) GetResponse(
	ctx context.Context,
	params ModelGetResponseParams,
) (*ModelResponse, error) {
	body, opts, err := m.prepareRequest(
		ctx,
		params.SystemInstructions,
		params.Input,
		params.ModelSettings,
		params.Tools,
		params.OutputSchema,
		params.Handoffs,
		params.PreviousResponseID,
		false,
	)
	if err != nil {
		return nil, err
	}

	response, err := m.client.Responses.New(ctx, *body, opts...)
	if err != nil {
		slog.Error("error getting response", slog.String("error", err.Error()))
		return nil, err
	}

	if DontLogModelData {
		slog.Debug("LLM responded")
	} else {
		slog.Debug("LLM responded", slog.String("output", SimplePrettyJSONMarshal(response.Output)))
	}

	u := usage.NewUsage()
	if !reflect.ValueOf(response.Usage).IsZero() {
		*u = usage.Usage{
			Requests:            1,
			InputTokens:         uint64(response.Usage.InputTokens),
			InputTokensDetails:  response.Usage.InputTokensDetails,
			OutputTokens:        uint64(response.Usage.OutputTokens),
			OutputTokensDetails: response.Usage.OutputTokensDetails,
			TotalTokens:         uint64(response.Usage.TotalTokens),
		}
	}

	return &ModelResponse{
		Output:     response.Output,
		Usage:      u,
		ResponseID: response.ID,
	}, nil
}

// StreamResponse yields a partial message as it is generated, as well as the usage information.
func (m OpenAIResponsesModel) StreamResponse(
	ctx context.Context,
	params ModelStreamResponseParams,
) (iter.Seq2[*TResponseStreamEvent, error], error) {
	body, opts, err := m.prepareRequest(
		ctx,
		params.SystemInstructions,
		params.Input,
		params.ModelSettings,
		params.Tools,
		params.OutputSchema,
		params.Handoffs,
		params.PreviousResponseID,
		true,
	)
	if err != nil {
		return nil, err
	}

	stream := m.client.Responses.NewStreaming(ctx, *body, opts...)
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("error streaming response: %w", err)
	}

	return func(yield func(*TResponseStreamEvent, error) bool) {
		defer func() { _ = stream.Close() }()

		for stream.Next() {
			chunk := stream.Current()
			if !yield(&chunk, nil) {
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("error streaming response: %w", err))
		}
	}, nil
}

func (m OpenAIResponsesModel) prepareRequest(
	ctx context.Context,
	systemInstructions param.Opt[string],
	input Input,
	modelSettings modelsettings.ModelSettings,
	tools []Tool,
	outputSchema AgentOutputSchemaInterface,
	handoffs []Handoff,
	previousResponseID string,
	stream bool,
) (*responses.ResponseNewParams, []option.RequestOption, error) {
	listInput := ItemHelpers().InputToNewInputList(input)

	var parallelToolCalls param.Opt[bool]
	if modelSettings.ParallelToolCalls.Valid() {
		if modelSettings.ParallelToolCalls.Value && len(tools) > 0 {
			parallelToolCalls = param.NewOpt(true)
		} else if !modelSettings.ParallelToolCalls.Value {
			parallelToolCalls = param.NewOpt(false)
		}
	}

	toolChoice := ResponsesConverter().ConvertToolChoice(modelSettings.ToolChoice)
	convertedTools, err := ResponsesConverter().ConvertTools(ctx, tools, handoffs)
	if err != nil {
		return nil, nil, err
	}
	responseFormat := ResponsesConverter().GetResponseFormat(outputSchema)

	if DontLogModelData {
		slog.Debug("Calling LLM")
	} else {
		slog.Debug(
			"Calling LLM",
			slog.String("Input", SimplePrettyJSONMarshal(listInput)),
			slog.String("Tools", SimplePrettyJSONMarshal(convertedTools.Tools)),
			slog.Bool("Stream", stream),
			slog.String("Tool choice", SimplePrettyJSONMarshal(toolChoice)),
			slog.String("Response format", SimplePrettyJSONMarshal(responseFormat)),
			slog.String("Previous response ID", previousResponseID),
		)
	}

	var prevRespIDParam param.Opt[string]
	if previousResponseID != "" {
		prevRespIDParam = param.NewOpt(previousResponseID)
	}

	params := &responses.ResponseNewParams{
		PreviousResponseID: prevRespIDParam,
		Instructions:       systemInstructions,
		Model:              m.Model,
		Input:              responses.ResponseNewParamsInputUnion{OfInputItemList: listInput},
		Include:            convertedTools.Includes,
		Tools:              convertedTools.Tools,
		Temperature:        modelSettings.Temperature,
		TopP:               modelSettings.TopP,
		Truncation:         responses.ResponseNewParamsTruncation(modelSettings.Truncation.Or("")),
		MaxOutputTokens:    modelSettings.MaxTokens,
		ToolChoice:         toolChoice,
		ParallelToolCalls:  parallelToolCalls,
		Text:               responseFormat,
		Store:              modelSettings.Store,
		Reasoning:          modelSettings.Reasoning,
		Metadata:           modelSettings.Metadata,
	}

	var opts []option.RequestOption
	for k, v := range modelSettings.ExtraHeaders {
		opts = append(opts, option.WithHeader(k, v))
	}
	for k, v := range modelSettings.ExtraQuery {
		opts = append(opts, option.WithQuery(k, v))
	}
	return params, opts, nil
}

type ConvertedTools struct {
	Tools    []responses.ToolUnionParam
	Includes []responses.ResponseIncludable
}

type responsesConverter struct{}

func ResponsesConverter() responsesConverter { return responsesConverter{} }

func (responsesConverter) ConvertToolChoice(toolChoice string) responses.ResponseNewParamsToolChoiceUnion {
	switch toolChoice {
	case "":
		return responses.ResponseNewParamsToolChoiceUnion{}
	case "none", "auto", "required":
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions(toolChoice)),
		}
	case "file_search", "web_search_preview", "web_search_preview_2025_03_11",
		"computer_use_preview", "image_generation", "code_interpreter", "mcp":
		return responses.ResponseNewParamsToolChoiceUnion{
			OfHostedTool: &responses.ToolChoiceTypesParam{
				Type: responses.ToolChoiceTypesType(toolChoice),
			},
		}
	default:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfFunctionTool: &responses.ToolChoiceFunctionParam{
				Name: toolChoice,
				Type: constant.ValueOf[constant.Function](),
			},
		}
	}
}

func (responsesConverter) GetResponseFormat(
	outputSchema AgentOutputSchemaInterface,
) responses.ResponseTextConfigParam {
	if outputSchema == nil || outputSchema.IsPlainText() {
		return responses.ResponseTextConfigParam{}
	}
	return responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "final_output",
				Schema: outputSchema.JSONSchema(),
				Strict: param.NewOpt(outputSchema.IsStrictJSONSchema()),
				Type:   constant.ValueOf[constant.JSONSchema](),
			},
		},
	}
}

func (conv responsesConverter) ConvertTools(ctx context.Context, ts []Tool, handoffs []Handoff) (*ConvertedTools, error) {
	var convertedTools []responses.ToolUnionParam
	var includes []responses.ResponseIncludable

	var computerTools []ComputerTool
	for _, tool := range ts {
		if ct, ok := tool.(ComputerTool); ok {
			computerTools = append(computerTools, ct)
		}
	}
	if len(computerTools) > 1 {
		return nil, UserErrorf("you can only provide one computer tool, got %d", len(computerTools))
	}

	for _, tool := range ts {
		convertedTool, include, err := tool.ConvertToResponses(ctx)
		if err != nil {
			return nil, err
		}
		convertedTools = append(convertedTools, *convertedTool)
		if include != nil {
			includes = append(includes, *include)
		}
	}

	for _, handoff := range handoffs {
		convertedTools = append(convertedTools, conv.convertHandoffTool(handoff))
	}

	return &ConvertedTools{
		Tools:    convertedTools,
		Includes: includes,
	}, nil
}

func (responsesConverter) convertHandoffTool(handoff Handoff) responses.ToolUnionParam {
	return responses.ToolUnionParam{
		OfFunction: &responses.FunctionToolParam{
			Name:        handoff.ToolName,
			Parameters:  handoff.InputJSONSchema,
			Strict:      param.NewOpt(handoff.StrictJSONSchema.Or(true)),
			Description: param.NewOpt(handoff.ToolDescription),
			Type:        constant.ValueOf[constant.Function](),
		},
	}
}
