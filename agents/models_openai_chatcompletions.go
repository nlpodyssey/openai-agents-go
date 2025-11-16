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
	"log/slog"
	"reflect"
	"slices"
	"time"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/openaitypes"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/nlpodyssey/openai-agents-go/util"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

type OpenAIChatCompletionsModel struct {
	Model  openai.ChatModel
	client OpenaiClient
}

func NewOpenAIChatCompletionsModel(model openai.ChatModel, client OpenaiClient) OpenAIChatCompletionsModel {
	return OpenAIChatCompletionsModel{
		Model:  model,
		client: client,
	}
}

func (m OpenAIChatCompletionsModel) GetResponse(
	ctx context.Context,
	params ModelResponseParams,
) (*ModelResponse, error) {
	var modelResponse *ModelResponse

	generationSpanParams, err := m.generationSpanParams(params)
	if err != nil {
		return nil, err
	}

	err = tracing.GenerationSpan(
		ctx, *generationSpanParams,
		func(ctx context.Context, spanGeneration tracing.Span) error {
			body, opts, err := m.prepareRequest(
				ctx,
				params.SystemInstructions,
				params.Input,
				params.ModelSettings,
				params.Tools,
				params.OutputType,
				params.Handoffs,
				spanGeneration,
				params.Tracing,
				false,
			)
			if err != nil {
				return err
			}

			response, err := m.client.Chat.Completions.New(ctx, *body, opts...)
			if err != nil {
				return err
			}

			var message *openai.ChatCompletionMessage
			var firstChoice *openai.ChatCompletionChoice

			if len(response.Choices) > 0 {
				firstChoice = &response.Choices[0]
				message = &firstChoice.Message
			}

			switch {
			case DontLogModelData:
				Logger().Debug("LLM responded")
			case message != nil:
				Logger().Debug("LLM responded", slog.String("message", SimplePrettyJSONMarshal(*message)))
			default:
				finishReason := "-"
				if firstChoice != nil {
					finishReason = firstChoice.FinishReason
				}
				Logger().Debug("LLM response", slog.String("finish_reason", finishReason))
			}

			u := usage.NewUsage()
			if !reflect.ValueOf(response.Usage).IsZero() {
				*u = usage.Usage{
					Requests:    1,
					InputTokens: uint64(response.Usage.PromptTokens),
					InputTokensDetails: responses.ResponseUsageInputTokensDetails{
						CachedTokens: response.Usage.PromptTokensDetails.CachedTokens,
					},
					OutputTokens: uint64(response.Usage.CompletionTokens),
					OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
						ReasoningTokens: response.Usage.CompletionTokensDetails.ReasoningTokens,
					},
					TotalTokens: uint64(response.Usage.TotalTokens),
				}
			}

			if params.Tracing.IncludeData() {
				var output []map[string]any
				if message != nil {
					v, err := util.JSONMap(*message)
					if err != nil {
						return fmt.Errorf("failed to convert message to JSON map: %w", err)
					}
					output = []map[string]any{v}
				}
				spanGeneration.SpanData().(*tracing.GenerationSpanData).Output = output
			}
			spanGeneration.SpanData().(*tracing.GenerationSpanData).Usage = map[string]any{
				"input_tokens":  u.InputTokens,
				"output_tokens": u.OutputTokens,
			}

			var items []TResponseOutputItem
			if message != nil {
				items, err = ChatCmplConverter().MessageToOutputItems(*message)
				if err != nil {
					return err
				}
			}
			modelResponse = &ModelResponse{
				Output:     items,
				Usage:      u,
				ResponseID: "",
			}
			return nil
		},
	)
	if err != nil {
		return nil, err
	}
	return modelResponse, nil
}

// StreamResponse yields a partial message as it is generated, as well as the usage information.
func (m OpenAIChatCompletionsModel) StreamResponse(
	ctx context.Context,
	params ModelResponseParams,
	yield ModelStreamResponseCallback,
) error {
	generationSpanParams, err := m.generationSpanParams(params)
	if err != nil {
		return err
	}

	return tracing.GenerationSpan(
		ctx, *generationSpanParams,
		func(ctx context.Context, spanGeneration tracing.Span) error {
			body, opts, err := m.prepareRequest(
				ctx,
				params.SystemInstructions,
				params.Input,
				params.ModelSettings,
				params.Tools,
				params.OutputType,
				params.Handoffs,
				spanGeneration,
				params.Tracing,
				true,
			)
			if err != nil {
				return err
			}

			stream := m.client.Chat.Completions.NewStreaming(ctx, *body, opts...)
			if err = stream.Err(); err != nil {
				return fmt.Errorf("error streaming response: %w", err)
			}

			response := responses.Response{
				ID:        FakeResponsesID,
				CreatedAt: float64(time.Now().Unix()),
				Model:     m.Model,
				Object:    constant.ValueOf[constant.Response](),
				Output:    nil,
				ToolChoice: responses.ResponseToolChoiceUnion{
					OfToolChoiceMode: responses.ToolChoiceOptions(body.ToolChoice.OfAuto.Or("auto")),
				},
				TopP:              params.ModelSettings.TopP.Or(0),
				Temperature:       params.ModelSettings.Temperature.Or(0),
				Tools:             nil,
				ParallelToolCalls: body.ParallelToolCalls.Or(false),
				Reasoning:         openaitypes.ReasoningFromParam(params.ModelSettings.Reasoning),
			}

			var finalResponse *responses.Response
			err = ChatCmplStreamHandler().HandleStream(response, stream, func(chunk TResponseStreamEvent) error {
				if chunk.Type == "response.completed" {
					finalResponse = &chunk.Response
				}
				return yield(ctx, chunk)
			})
			if err != nil {
				return err
			}

			if finalResponse != nil {
				spanData := spanGeneration.SpanData().(*tracing.GenerationSpanData)

				if params.Tracing.IncludeData() {
					out, err := util.JSONMap(*finalResponse)
					if err != nil {
						return fmt.Errorf("failed to convert final response to JSON map: %w", err)
					}
					spanData.Output = []map[string]any{out}
				}

				if u := finalResponse.Usage; !reflect.ValueOf(u).IsZero() {
					spanData.Usage = map[string]any{
						"input_tokens":  u.InputTokens,
						"output_tokens": u.OutputTokens,
					}
				}
			}
			return nil
		})
}

func (m OpenAIChatCompletionsModel) generationSpanParams(params ModelResponseParams) (*tracing.GenerationSpanParams, error) {
	modelConfig, err := util.JSONMap(params.ModelSettings)
	if err != nil {
		return nil, fmt.Errorf("failed to convert model settings to JSON map: %w", err)
	}
	if m.client.BaseURL.Valid() {
		modelConfig["base_url"] = m.client.BaseURL.Value
	}
	return &tracing.GenerationSpanParams{
		Model:       m.Model,
		ModelConfig: modelConfig,
		Disabled:    params.Tracing.IsDisabled(),
	}, nil
}

func (m OpenAIChatCompletionsModel) prepareRequest(
	ctx context.Context,
	systemInstructions param.Opt[string],
	input Input,
	modelSettings modelsettings.ModelSettings,
	tools []Tool,
	outputType OutputTypeInterface,
	handoffs []Handoff,
	span tracing.Span,
	modelTracing ModelTracing,
	stream bool,
) (*openai.ChatCompletionNewParams, []option.RequestOption, error) {
	convertedMessages, err := ChatCmplConverter().ItemsToMessages(input)
	if err != nil {
		return nil, nil, err
	}

	if systemInstructions.Valid() {
		convertedMessages = slices.Insert(convertedMessages, 0, openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: param.NewOpt(systemInstructions.Value),
				},
				Role: constant.ValueOf[constant.System](),
			},
		})
	}

	if modelTracing.IncludeData() {
		in, err := util.JSONMapSlice(convertedMessages)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to convert converted-messages to JSON []map: %w", err)
		}
		span.SpanData().(*tracing.GenerationSpanData).Input = in
	}

	var parallelToolCalls param.Opt[bool]
	if modelSettings.ParallelToolCalls.Valid() {
		if modelSettings.ParallelToolCalls.Value && len(tools) > 0 {
			parallelToolCalls = param.NewOpt(true)
		} else if !modelSettings.ParallelToolCalls.Value {
			parallelToolCalls = param.NewOpt(false)
		}
	}

	toolChoice, err := ChatCmplConverter().ConvertToolChoice(modelSettings.ToolChoice)
	if err != nil {
		return nil, nil, err
	}
	responseFormat, _, err := ChatCmplConverter().ConvertResponseFormat(outputType)
	if err != nil {
		return nil, nil, err
	}

	var convertedTools []openai.ChatCompletionToolUnionParam
	for _, tool := range tools {
		v, err := ChatCmplConverter().ToolToOpenai(tool)
		if err != nil {
			return nil, nil, err
		}
		convertedTools = append(convertedTools, *v)
	}

	for _, handoff := range handoffs {
		convertedTools = append(convertedTools, ChatCmplConverter().ConvertHandoffTool(handoff))
	}

	if DontLogModelData {
		Logger().Debug("Calling LLM")
	} else {
		Logger().Debug(
			"Calling LLM",
			slog.String("Messages", SimplePrettyJSONMarshal(convertedMessages)),
			slog.String("Tools", SimplePrettyJSONMarshal(convertedTools)),
			slog.Bool("Stream", stream),
			slog.String("Tool choice", SimplePrettyJSONMarshal(toolChoice)),
			slog.String("Response format", SimplePrettyJSONMarshal(responseFormat)),
		)
	}

	store := ChatCmplHelpers().GetStoreParam(m.client, modelSettings)

	streamOptions := ChatCmplHelpers().GetStreamOptionsParam(m.client, modelSettings, stream)

	params := &openai.ChatCompletionNewParams{
		Model:             m.Model,
		Messages:          convertedMessages,
		Tools:             convertedTools,
		Temperature:       modelSettings.Temperature,
		TopP:              modelSettings.TopP,
		FrequencyPenalty:  modelSettings.FrequencyPenalty,
		PresencePenalty:   modelSettings.PresencePenalty,
		MaxTokens:         modelSettings.MaxTokens,
		ToolChoice:        toolChoice,
		ResponseFormat:    responseFormat,
		ParallelToolCalls: parallelToolCalls,
		StreamOptions:     streamOptions,
		Store:             store,
		ReasoningEffort:   modelSettings.Reasoning.Effort,
		Verbosity:         openai.ChatCompletionNewParamsVerbosity(modelSettings.Verbosity.Or("")),
		TopLogprobs:       modelSettings.TopLogprobs,
		Metadata:          modelSettings.Metadata,
	}

	var opts []option.RequestOption
	for k, v := range modelSettings.ExtraHeaders {
		opts = append(opts, option.WithHeader(k, v))
	}
	for k, v := range modelSettings.ExtraQuery {
		opts = append(opts, option.WithQuery(k, v))
	}

	if modelSettings.CustomizeChatCompletionsRequest != nil {
		return modelSettings.CustomizeChatCompletionsRequest(ctx, params, opts)
	}

	return params, opts, nil
}
