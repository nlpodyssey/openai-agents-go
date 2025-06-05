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
	"github.com/nlpodyssey/openai-agents-go/types/optional"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
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
	body, opts := m.prepareRequest(
		params.SystemInstructions,
		params.Input,
		params.ModelSettings,
		params.Tools,
		params.OutputSchema,
		params.Handoffs,
		params.PreviousResponseID,
		false,
	)

	response, err := m.client.Responses.New(ctx, body, opts...)
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
	if !reflect.DeepEqual(response.Usage, responses.ResponseUsage{}) {
		u.Requests = 1
		u.InputTokens = uint64(response.Usage.InputTokens)
		u.OutputTokens = uint64(response.Usage.OutputTokens)
		u.TotalTokens = uint64(response.Usage.TotalTokens)
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
	body, opts := m.prepareRequest(
		params.SystemInstructions,
		params.Input,
		params.ModelSettings,
		params.Tools,
		params.OutputSchema,
		params.Handoffs,
		params.PreviousResponseID,
		true,
	)

	stream := m.client.Responses.NewStreaming(ctx, body, opts...)
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
	systemInstructions optional.Optional[string],
	input Input,
	modelSettings modelsettings.ModelSettings,
	tools []Tool,
	outputSchema AgentOutputSchemaInterface,
	handoffs []Handoff,
	previousResponseID string,
	stream bool,
) (responses.ResponseNewParams, []option.RequestOption) {
	listInput := ItemHelpers().InputToNewInputList(input)

	var parallelToolCalls param.Opt[bool]
	if modelSettings.ParallelToolCalls.Present {
		if modelSettings.ParallelToolCalls.Value && len(tools) > 0 {
			parallelToolCalls = param.NewOpt(true)
		} else if !modelSettings.ParallelToolCalls.Value {
			parallelToolCalls = param.NewOpt(false)
		}
	}

	toolChoice := ResponsesConverter().ConvertToolChoice(modelSettings.ToolChoice)
	convertedTools := ResponsesConverter().ConvertTools(tools, handoffs)
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

	params := responses.ResponseNewParams{
		PreviousResponseID: prevRespIDParam,
		Instructions:       optional.ToParamOptOmitted(systemInstructions),
		Model:              m.Model,
		Input:              responses.ResponseNewParamsInputUnion{OfInputItemList: listInput},
		Include:            convertedTools.Includes,
		Tools:              convertedTools.Tools,
		Temperature:        optional.ToParamOptOmitted(modelSettings.Temperature),
		TopP:               optional.ToParamOptOmitted(modelSettings.TopP),
		Truncation:         responses.ResponseNewParamsTruncation(modelSettings.Truncation.ValueOrFallback("")),
		MaxOutputTokens:    optional.ToParamOptOmitted(modelSettings.MaxTokens),
		ToolChoice:         toolChoice,
		ParallelToolCalls:  parallelToolCalls,
		Text:               responseFormat,
		Store:              optional.ToParamOptOmitted(modelSettings.Store),
		Reasoning:          modelSettings.Reasoning.ValueOrFallback(shared.ReasoningParam{}),
		Metadata:           modelSettings.Metadata.ValueOrFallback(nil),
	}

	var opts []option.RequestOption
	if modelSettings.ExtraHeaders.Present {
		for k, v := range modelSettings.ExtraHeaders.Value {
			opts = append(opts, option.WithHeader(k, v))
		}
	}
	if modelSettings.ExtraQuery.Present {
		for k, v := range modelSettings.ExtraQuery.Value {
			opts = append(opts, option.WithQuery(k, v))
		}
	}

	return params, opts
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
	case "required":
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
		}
	case "auto":
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
		}
	case "none":
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
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

func (conv responsesConverter) ConvertTools(tools []Tool, handoffs []Handoff) ConvertedTools {
	var convertedTools []responses.ToolUnionParam
	var includes []responses.ResponseIncludable

	for _, tool := range tools {
		convertedTool, include := conv.convertTool(tool)
		convertedTools = append(convertedTools, convertedTool)
		if include != nil {
			includes = append(includes, *include)
		}
	}

	for _, handoff := range handoffs {
		convertedTools = append(convertedTools, conv.convertHandoffTool(handoff))
	}

	return ConvertedTools{
		Tools:    convertedTools,
		Includes: includes,
	}
}

// convertTool returns converted tool and includes.
func (responsesConverter) convertTool(tool Tool) (responses.ToolUnionParam, *responses.ResponseIncludable) {
	switch tool := tool.(type) {
	case FunctionTool:
		return responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.Name,
				Parameters:  tool.ParamsJSONSchema,
				Strict:      param.NewOpt(tool.StrictJSONSchema.ValueOrFallback(true)),
				Description: param.NewOpt(tool.Description),
				Type:        constant.ValueOf[constant.Function](),
			},
		}, nil
	default:
		// This would be an unrecoverable implementation bug, so a panic is appropriate.
		panic(fmt.Errorf("unexpected Tool type %T", tool))
	}
}

func (responsesConverter) convertHandoffTool(handoff Handoff) responses.ToolUnionParam {
	return responses.ToolUnionParam{
		OfFunction: &responses.FunctionToolParam{
			Name:        handoff.ToolName,
			Parameters:  handoff.InputJSONSchema,
			Strict:      param.NewOpt(handoff.StrictJSONSchema.ValueOrFallback(true)),
			Description: param.NewOpt(handoff.ToolDescription),
			Type:        constant.ValueOf[constant.Function](),
		},
	}
}
