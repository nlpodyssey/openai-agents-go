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

package modelsettings

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Tests whether ModelSettings can be serialized to a JSON string.
func TestModelSettings_BasicSerialization(t *testing.T) {
	modelSettings := ModelSettings{
		Temperature: param.NewOpt(0.5),
		TopP:        param.NewOpt(0.9),
		MaxTokens:   param.NewOpt[int64](100),
	}
	res, err := json.Marshal(modelSettings)
	require.NoError(t, err)

	var got any
	err = unmarshal(res, &got)
	require.NoError(t, err)

	var want any = map[string]any{
		"temperature":         json.Number("0.5"),
		"top_p":               json.Number("0.9"),
		"frequency_penalty":   nil,
		"presence_penalty":    nil,
		"tool_choice":         nil,
		"parallel_tool_calls": nil,
		"truncation":          nil,
		"max_tokens":          json.Number("100"),
		"reasoning":           map[string]any{},
		"verbosity":           nil,
		"metadata":            nil,
		"store":               nil,
		"include_usage":       nil,
		"response_include":    nil,
		"top_logprobs":        nil,
		"extra_query":         nil,
		"extra_headers":       nil,
	}
	assert.Equal(t, want, got)
}

// Tests whether ModelSettings can be serialized to a JSON string.
func TestModelSettings_AllFieldsSerialization(t *testing.T) {
	modelSettings := ModelSettings{
		Temperature:       param.NewOpt(0.5),
		TopP:              param.NewOpt(0.9),
		FrequencyPenalty:  param.NewOpt(0.0),
		PresencePenalty:   param.NewOpt(0.0),
		ToolChoice:        ToolChoiceAuto,
		ParallelToolCalls: param.NewOpt(true),
		Truncation:        param.NewOpt(TruncationAuto),
		MaxTokens:         param.NewOpt[int64](100),
		Reasoning:         openai.ReasoningParam{},
		Verbosity:         param.NewOpt(VerbosityMedium),
		Metadata:          map[string]string{"foo": "bar"},
		Store:             param.NewOpt(false),
		IncludeUsage:      param.NewOpt(false),
		ResponseInclude:   []responses.ResponseIncludable{responses.ResponseIncludableFileSearchCallResults},
		TopLogprobs:       param.NewOpt(int64(1)),
		ExtraQuery:        map[string]string{"foo": "bar"},
		ExtraHeaders:      map[string]string{"foo": "bar"},
	}
	res, err := json.Marshal(modelSettings)
	require.NoError(t, err)

	var got any
	err = unmarshal(res, &got)
	require.NoError(t, err)

	var want any = map[string]any{
		"temperature":         json.Number("0.5"),
		"top_p":               json.Number("0.9"),
		"frequency_penalty":   json.Number("0"),
		"presence_penalty":    json.Number("0"),
		"tool_choice":         "auto",
		"parallel_tool_calls": true,
		"truncation":          "auto",
		"max_tokens":          json.Number("100"),
		"reasoning":           map[string]any{},
		"verbosity":           "medium",
		"metadata":            map[string]any{"foo": "bar"},
		"store":               false,
		"include_usage":       false,
		"response_include":    []any{"file_search_call.results"},
		"top_logprobs":        json.Number("1"),
		"extra_query":         map[string]any{"foo": "bar"},
		"extra_headers":       map[string]any{"foo": "bar"},
	}
	assert.Equal(t, want, got)
}

func TestModelSettings_ToolChoiceMCPSerialization(t *testing.T) {
	modelSettings := ModelSettings{
		Temperature: param.NewOpt(0.5),
		ToolChoice: ToolChoiceMCP{
			ServerLabel: "mcp",
			Name:        "mcp_tool",
		},
	}
	res, err := json.Marshal(modelSettings)
	require.NoError(t, err)

	var got any
	err = unmarshal(res, &got)
	require.NoError(t, err)

	var want any = map[string]any{
		"temperature":       json.Number("0.5"),
		"top_p":             nil,
		"frequency_penalty": nil,
		"presence_penalty":  nil,
		"tool_choice": map[string]any{
			"server_label": "mcp",
			"name":         "mcp_tool",
		},
		"parallel_tool_calls": nil,
		"truncation":          nil,
		"max_tokens":          nil,
		"reasoning":           map[string]any{},
		"verbosity":           nil,
		"metadata":            nil,
		"store":               nil,
		"include_usage":       nil,
		"response_include":    nil,
		"top_logprobs":        nil,
		"extra_query":         nil,
		"extra_headers":       nil,
	}
	assert.Equal(t, want, got)
}

func unmarshal(data []byte, v any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	err := d.Decode(v)
	return err
}

func TestModelSettings_Resolve(t *testing.T) {
	base := ModelSettings{
		Temperature:       param.NewOpt(0.5),
		TopP:              param.NewOpt(0.9),
		FrequencyPenalty:  param.NewOpt(0.0),
		PresencePenalty:   param.NewOpt[float64](0.0),
		ToolChoice:        ToolChoiceAuto,
		ParallelToolCalls: param.NewOpt(true),
		Truncation:        param.NewOpt(TruncationAuto),
		MaxTokens:         param.NewOpt[int64](100),
		Reasoning: openai.ReasoningParam{
			Effort:  openai.ReasoningEffortLow,
			Summary: openai.ReasoningSummaryConcise,
		},
		Verbosity:                       param.NewOpt(VerbosityMedium),
		Metadata:                        map[string]string{"foo": "bar"},
		Store:                           param.NewOpt(false),
		IncludeUsage:                    param.NewOpt(false),
		ResponseInclude:                 []responses.ResponseIncludable{responses.ResponseIncludableFileSearchCallResults},
		TopLogprobs:                     param.NewOpt(int64(1)),
		ExtraQuery:                      map[string]string{"foo": "bar"},
		ExtraHeaders:                    map[string]string{"foo": "bar"},
		CustomizeResponsesRequest:       nil,
		CustomizeChatCompletionsRequest: nil,
	}

	t.Run("overriding first set of properties", func(t *testing.T) {
		override := ModelSettings{
			Temperature:      param.NewOpt(0.4),
			FrequencyPenalty: param.NewOpt(0.1),
			ToolChoice:       ToolChoiceRequired,
			Truncation:       param.NewOpt(TruncationDisabled),
			Reasoning: openai.ReasoningParam{
				Effort:  openai.ReasoningEffortMedium,
				Summary: openai.ReasoningSummaryDetailed,
			},
			Verbosity:  param.NewOpt(VerbosityHigh),
			Store:      param.NewOpt(true),
			ExtraQuery: map[string]string{"a": "b"},
			CustomizeResponsesRequest: func(context.Context, *responses.ResponseNewParams, []option.RequestOption) (*responses.ResponseNewParams, []option.RequestOption, error) {
				return nil, nil, nil
			},
		}

		resolved := base.Resolve(override)

		assert.Equal(t, param.NewOpt(0.4), resolved.Temperature)
		assert.Equal(t, param.NewOpt(0.9), resolved.TopP)
		assert.Equal(t, param.NewOpt(0.1), resolved.FrequencyPenalty)
		assert.Equal(t, param.NewOpt(0.0), resolved.PresencePenalty)
		assert.Equal(t, ToolChoiceRequired, resolved.ToolChoice)
		assert.Equal(t, param.NewOpt(true), resolved.ParallelToolCalls)
		assert.Equal(t, param.NewOpt(TruncationDisabled), resolved.Truncation)
		assert.Equal(t, param.NewOpt[int64](100), resolved.MaxTokens)
		assert.Equal(t, openai.ReasoningParam{
			Effort:  openai.ReasoningEffortMedium,
			Summary: openai.ReasoningSummaryDetailed,
		}, resolved.Reasoning)
		assert.Equal(t, param.NewOpt(VerbosityHigh), resolved.Verbosity)
		assert.Equal(t, map[string]string{"foo": "bar"}, resolved.Metadata)
		assert.Equal(t, param.NewOpt(true), resolved.Store)
		assert.Equal(t, param.NewOpt(false), resolved.IncludeUsage)
		assert.Equal(t, []responses.ResponseIncludable{responses.ResponseIncludableFileSearchCallResults}, resolved.ResponseInclude)
		assert.Equal(t, param.NewOpt(int64(1)), resolved.TopLogprobs)
		assert.Equal(t, map[string]string{"a": "b"}, resolved.ExtraQuery)
		assert.Equal(t, map[string]string{"foo": "bar"}, resolved.ExtraHeaders)
		assert.NotNil(t, resolved.CustomizeResponsesRequest)
		assert.Nil(t, resolved.CustomizeChatCompletionsRequest)
	})

	t.Run("overriding second set of properties", func(t *testing.T) {
		override := ModelSettings{
			TopP:              param.NewOpt(0.8),
			PresencePenalty:   param.NewOpt(0.2),
			ParallelToolCalls: param.NewOpt(false),
			MaxTokens:         param.NewOpt[int64](42),
			Metadata:          map[string]string{"a": "b"},
			IncludeUsage:      param.NewOpt(true),
			ResponseInclude:   []responses.ResponseIncludable{responses.ResponseIncludableMessageInputImageImageURL},
			TopLogprobs:       param.NewOpt(int64(2)),
			ExtraHeaders:      map[string]string{"c": "d"},
			CustomizeChatCompletionsRequest: func(context.Context, *openai.ChatCompletionNewParams, []option.RequestOption) (*openai.ChatCompletionNewParams, []option.RequestOption, error) {
				return nil, nil, nil
			},
		}

		resolved := base.Resolve(override)

		assert.Equal(t, param.NewOpt(0.5), resolved.Temperature)
		assert.Equal(t, param.NewOpt(0.8), resolved.TopP)
		assert.Equal(t, param.NewOpt(0.0), resolved.FrequencyPenalty)
		assert.Equal(t, param.NewOpt(0.2), resolved.PresencePenalty)
		assert.Equal(t, ToolChoiceAuto, resolved.ToolChoice)
		assert.Equal(t, param.NewOpt(false), resolved.ParallelToolCalls)
		assert.Equal(t, param.NewOpt(TruncationAuto), resolved.Truncation)
		assert.Equal(t, param.NewOpt[int64](42), resolved.MaxTokens)
		assert.Equal(t, openai.ReasoningParam{
			Effort:  openai.ReasoningEffortLow,
			Summary: openai.ReasoningSummaryConcise,
		}, resolved.Reasoning)
		assert.Equal(t, param.NewOpt(VerbosityMedium), resolved.Verbosity)
		assert.Equal(t, map[string]string{"a": "b"}, resolved.Metadata)
		assert.Equal(t, param.NewOpt(false), resolved.Store)
		assert.Equal(t, param.NewOpt(true), resolved.IncludeUsage)
		assert.Equal(t, []responses.ResponseIncludable{responses.ResponseIncludableMessageInputImageImageURL}, resolved.ResponseInclude)
		assert.Equal(t, param.NewOpt(int64(2)), resolved.TopLogprobs)
		assert.Equal(t, map[string]string{"foo": "bar"}, resolved.ExtraQuery)
		assert.Equal(t, map[string]string{"c": "d"}, resolved.ExtraHeaders)
		assert.Nil(t, resolved.CustomizeResponsesRequest)
		assert.NotNil(t, resolved.CustomizeChatCompletionsRequest)
	})
}
