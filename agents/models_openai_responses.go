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
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"slices"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
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
	params ModelResponseParams,
) (*ModelResponse, error) {
	var u *usage.Usage
	var response *responses.Response

	err := tracing.ResponseSpan(
		ctx, tracing.ResponseSpanParams{Disabled: params.Tracing.IsDisabled()},
		func(ctx context.Context, spanResponse tracing.Span) (err error) {
			defer func() {
				if err != nil {
					var v string
					if params.Tracing.IncludeData() {
						v = err.Error()
					} else {
						v = fmt.Sprintf("%T", err)
					}
					spanResponse.SetError(tracing.SpanError{
						Message: "Error getting response",
						Data:    map[string]any{"error": v},
					})
				}
			}()

			body, opts, err := m.prepareRequest(
				ctx,
				params.SystemInstructions,
				params.Input,
				params.ModelSettings,
				params.Tools,
				params.OutputType,
				params.Handoffs,
				params.PreviousResponseID,
				false,
				params.Prompt,
			)
			if err != nil {
				return err
			}

			response, err = m.client.Responses.New(ctx, *body, opts...)
			if err != nil {
				Logger().Error("error getting response", slog.String("error", err.Error()))
				return err
			}

			if DontLogModelData {
				Logger().Debug("LLM responded")
			} else {
				Logger().Debug("LLM responded", slog.String("output", SimplePrettyJSONMarshal(response.Output)))
			}

			u = usage.NewUsage()
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

			if params.Tracing.IncludeData() {
				spanData := spanResponse.SpanData().(*tracing.ResponseSpanData)
				spanData.Response = response
				spanData.Input = params.Input
			}

			return nil
		},
	)
	if err != nil {
		return nil, err
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
	params ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	return tracing.ResponseSpan(
		ctx, tracing.ResponseSpanParams{Disabled: params.Tracing.IsDisabled()},
		func(ctx context.Context, spanResponse tracing.Span) (err error) {
			defer func() {
				if err != nil {
					var v string
					if params.Tracing.IncludeData() {
						v = err.Error()
					} else {
						v = fmt.Sprintf("%T", err)
					}
					spanResponse.SetError(tracing.SpanError{
						Message: "Error streaming response",
						Data:    map[string]any{"error": v},
					})
					Logger().Error("error streaming response", slog.String("error", err.Error()))
				}
			}()

			body, opts, err := m.prepareRequest(
				ctx,
				params.SystemInstructions,
				params.Input,
				params.ModelSettings,
				params.Tools,
				params.OutputType,
				params.Handoffs,
				params.PreviousResponseID,
				true,
				params.Prompt,
			)
			if err != nil {
				return err
			}

			stream := m.client.Responses.NewStreaming(ctx, *body, opts...)
			defer func() {
				if e := stream.Close(); e != nil {
					err = errors.Join(err, fmt.Errorf("error closing stream: %w", e))
				}
			}()

			var finalResponse *responses.Response
			for stream.Next() {
				chunk := stream.Current()
				if chunk.Type == "response.completed" {
					finalResponse = &chunk.Response
				}
				if err = yield(ctx, chunk); err != nil {
					return err
				}
			}

			if err = stream.Err(); err != nil {
				return fmt.Errorf("error streaming response: %w", err)
			}

			if finalResponse != nil && params.Tracing.IncludeData() {
				spanData := spanResponse.SpanData().(*tracing.ResponseSpanData)
				spanData.Response = finalResponse
				spanData.Input = params.Input
			}
			return nil
		})
}

func (m OpenAIResponsesModel) prepareRequest(
	ctx context.Context,
	systemInstructions param.Opt[string],
	input Input,
	modelSettings modelsettings.ModelSettings,
	tools []Tool,
	outputType OutputTypeInterface,
	handoffs []Handoff,
	previousResponseID string,
	stream bool,
	prompt responses.ResponsePromptParam,
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
	responseFormat, err := ResponsesConverter().GetResponseFormat(outputType)
	if err != nil {
		return nil, nil, err
	}

	include := slices.Concat(convertedTools.Includes, modelSettings.ResponseInclude)
	if modelSettings.TopLogprobs.Valid() {
		include = append(include, "message.output_text.logprobs")
	}

	// Remove duplicates
	slices.Sort(include)
	include = slices.Compact(include)

	if modelSettings.Verbosity.Valid() {
		responseFormat.Verbosity = responses.ResponseTextConfigVerbosity(modelSettings.Verbosity.Value)
	}

	if DontLogModelData {
		Logger().Debug("Calling LLM")
	} else {
		Logger().Debug(
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
		Include:            include,
		Tools:              convertedTools.Tools,
		Prompt:             prompt,
		Temperature:        modelSettings.Temperature,
		TopP:               modelSettings.TopP,
		Truncation:         responses.ResponseNewParamsTruncation(modelSettings.Truncation.Or("")),
		MaxOutputTokens:    modelSettings.MaxTokens,
		ToolChoice:         toolChoice,
		ParallelToolCalls:  parallelToolCalls,
		Text:               responseFormat,
		Store:              modelSettings.Store,
		Reasoning:          modelSettings.Reasoning,
		TopLogprobs:        modelSettings.TopLogprobs,
		Metadata:           modelSettings.Metadata,
	}

	var opts []option.RequestOption
	for k, v := range modelSettings.ExtraHeaders {
		opts = append(opts, option.WithHeader(k, v))
	}
	for k, v := range modelSettings.ExtraQuery {
		opts = append(opts, option.WithQuery(k, v))
	}

	if modelSettings.CustomizeResponsesRequest != nil {
		return modelSettings.CustomizeResponsesRequest(ctx, params, opts)
	}

	return params, opts, nil
}

type ConvertedTools struct {
	Tools    []responses.ToolUnionParam
	Includes []responses.ResponseIncludable
}

type responsesConverter struct{}

func ResponsesConverter() responsesConverter { return responsesConverter{} }

func (responsesConverter) ConvertToolChoice(toolChoice modelsettings.ToolChoice) responses.ResponseNewParamsToolChoiceUnion {
	switch toolChoice := toolChoice.(type) {
	case nil:
		return responses.ResponseNewParamsToolChoiceUnion{}
	case modelsettings.ToolChoiceString:
		switch toolChoice {
		case "none", "auto", "required":
			return responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions(toolChoice)),
			}
		case "file_search", "web_search_preview", "web_search_preview_2025_03_11",
			"computer_use_preview", "image_generation", "code_interpreter":
			return responses.ResponseNewParamsToolChoiceUnion{
				OfHostedTool: &responses.ToolChoiceTypesParam{
					Type: responses.ToolChoiceTypesType(toolChoice),
				},
			}
		case "mcp":
			// Note that this is still here for backwards compatibility,
			// but migrating to ToolChoiceMCP is recommended.
			return responses.ResponseNewParamsToolChoiceUnion{
				OfMcpTool: &responses.ToolChoiceMcpParam{
					Type: constant.ValueOf[constant.Mcp](),
				},
			}
		default:
			return responses.ResponseNewParamsToolChoiceUnion{
				OfFunctionTool: &responses.ToolChoiceFunctionParam{
					Name: toolChoice.String(),
					Type: constant.ValueOf[constant.Function](),
				},
			}
		}
	case modelsettings.ToolChoiceMCP:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfMcpTool: &responses.ToolChoiceMcpParam{
				ServerLabel: toolChoice.ServerLabel,
				Name:        param.NewOpt(toolChoice.Name),
				Type:        constant.ValueOf[constant.Mcp](),
			},
		}
	default:
		// This would be an unrecoverable implementation bug, so a panic is appropriate.
		panic(fmt.Errorf("unexpected ToolChoice type %T", toolChoice))
	}
}

func (responsesConverter) GetResponseFormat(
	outputType OutputTypeInterface,
) (responses.ResponseTextConfigParam, error) {
	if outputType == nil || outputType.IsPlainText() {
		return responses.ResponseTextConfigParam{}, nil
	}
	schema, err := outputType.JSONSchema()
	if err != nil {
		return responses.ResponseTextConfigParam{}, err
	}
	return responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "final_output",
				Schema: schema,
				Strict: param.NewOpt(outputType.IsStrictJSONSchema()),
				Type:   constant.ValueOf[constant.JSONSchema](),
			},
		},
	}, nil
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
		convertedTool, include, err := conv.convertTool(ctx, tool)
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

// convertTool returns converted tool and includes.
func (conv responsesConverter) convertTool(
	ctx context.Context,
	tool Tool,
) (*responses.ToolUnionParam, *responses.ResponseIncludable, error) {
	var convertedTool *responses.ToolUnionParam
	var includes *responses.ResponseIncludable

	switch t := tool.(type) {
	case FunctionTool:
		convertedTool = &responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        t.Name,
				Parameters:  t.ParamsJSONSchema,
				Strict:      param.NewOpt(t.StrictJSONSchema.Or(true)),
				Description: param.NewOpt(t.Description),
				Type:        constant.ValueOf[constant.Function](),
			},
		}
		includes = nil
	case WebSearchTool:
		convertedTool = &responses.ToolUnionParam{
			OfWebSearch: &responses.WebSearchToolParam{
				Type:              responses.WebSearchToolTypeWebSearch,
				Filters:           t.Filters,
				UserLocation:      t.UserLocation,
				SearchContextSize: t.SearchContextSize,
			},
		}
		includes = nil
	case FileSearchTool:
		convertedTool = &responses.ToolUnionParam{
			OfFileSearch: &responses.FileSearchToolParam{
				VectorStoreIDs: t.VectorStoreIDs,
				MaxNumResults:  t.MaxNumResults,
				Filters:        t.Filters,
				RankingOptions: t.RankingOptions,
				Type:           constant.ValueOf[constant.FileSearch](),
			},
		}
		if t.IncludeSearchResults {
			includes = new(responses.ResponseIncludable)
			*includes = responses.ResponseIncludableFileSearchCallResults
		}
	case ComputerTool:
		environment, err := t.Computer.Environment(ctx)
		if err != nil {
			return nil, nil, err
		}

		dimensions, err := t.Computer.Dimensions(ctx)
		if err != nil {
			return nil, nil, err
		}

		convertedTool = &responses.ToolUnionParam{
			OfComputerUsePreview: &responses.ComputerToolParam{
				DisplayHeight: dimensions.Height,
				DisplayWidth:  dimensions.Width,
				Environment:   responses.ComputerToolEnvironment(environment),
				Type:          constant.ValueOf[constant.ComputerUsePreview](),
			},
		}
		includes = nil
	case HostedMCPTool:
		convertedTool = &responses.ToolUnionParam{
			OfMcp: &t.ToolConfig,
		}
		includes = nil
	case ImageGenerationTool:
		convertedTool = &responses.ToolUnionParam{
			OfImageGeneration: &t.ToolConfig,
		}
		includes = nil
	case CodeInterpreterTool:
		convertedTool = &responses.ToolUnionParam{
			OfCodeInterpreter: &t.ToolConfig,
		}
		includes = nil
	case LocalShellTool:
		convertedTool = &responses.ToolUnionParam{
			OfLocalShell: &responses.ToolLocalShellParam{
				Type: constant.ValueOf[constant.LocalShell](),
			},
		}
		includes = nil
	default:
		return nil, nil, UserErrorf("Unknown tool type: %T", tool)
	}

	return convertedTool, includes, nil
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
