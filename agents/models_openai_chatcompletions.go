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
	"slices"
	"time"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/openaitypes"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
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

	response, err := m.client.Chat.Completions.New(ctx, *body, opts...)
	if err != nil {
		return nil, err
	}

	if DontLogModelData {
		slog.Debug("LLM responded")
	} else {
		slog.Debug("LLM responded", slog.String("message", SimplePrettyJSONMarshal(response.Choices[0].Message)))
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

	items := ChatCmplConverter().MessageToOutputItems(response.Choices[0].Message)
	return &ModelResponse{
		Output:     items,
		Usage:      u,
		ResponseID: "",
	}, nil
}

// StreamResponse yields a partial message as it is generated, as well as the usage information.
func (m OpenAIChatCompletionsModel) StreamResponse(
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
		false,
	)
	if err != nil {
		return nil, err
	}

	stream := m.client.Chat.Completions.NewStreaming(ctx, *body, opts...)
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("error streaming response: %w", err)
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

	return ChatCmplStreamHandler().HandleStream(response, stream), nil
}

func (m OpenAIChatCompletionsModel) prepareRequest(
	ctx context.Context,
	systemInstructions param.Opt[string],
	input Input,
	modelSettings modelsettings.ModelSettings,
	tools []Tool,
	outputSchema AgentOutputSchemaInterface,
	handoffs []Handoff,
	previousResponseID string,
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

	var parallelToolCalls param.Opt[bool]
	if modelSettings.ParallelToolCalls.Valid() {
		if modelSettings.ParallelToolCalls.Value && len(tools) > 0 {
			parallelToolCalls = param.NewOpt(true)
		} else if !modelSettings.ParallelToolCalls.Value {
			parallelToolCalls = param.NewOpt(false)
		}
	}

	toolChoice, _ := ChatCmplConverter().ConvertToolChoice(modelSettings.ToolChoice)
	responseFormat, _ := ChatCmplConverter().ConvertResponseFormat(outputSchema)

	var convertedTools []openai.ChatCompletionToolParam
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
		slog.Debug("Calling LLM")
	} else {
		slog.Debug(
			"Calling LLM",
			slog.String("Messages", SimplePrettyJSONMarshal(convertedMessages)),
			slog.String("Tools", SimplePrettyJSONMarshal(convertedTools)),
			slog.Bool("Stream", stream),
			slog.String("Tool choice", SimplePrettyJSONMarshal(toolChoice)),
			slog.String("Response format", SimplePrettyJSONMarshal(responseFormat)),
			slog.String("Previous response ID", previousResponseID),
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
		Metadata:          modelSettings.Metadata,
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
