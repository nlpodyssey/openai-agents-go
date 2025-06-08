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
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
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
		"tool_choice":         "",
		"parallel_tool_calls": nil,
		"truncation":          nil,
		"max_tokens":          json.Number("100"),
		"reasoning":           map[string]any{},
		"metadata":            nil,
		"store":               nil,
		"include_usage":       nil,
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
		ToolChoice:        "auto",
		ParallelToolCalls: param.NewOpt(true),
		Truncation:        param.NewOpt(TruncationAuto),
		MaxTokens:         param.NewOpt[int64](100),
		Reasoning:         openai.ReasoningParam{},
		Metadata:          map[string]string{"foo": "bar"},
		Store:             param.NewOpt(false),
		IncludeUsage:      param.NewOpt(false),
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
		"metadata":            map[string]any{"foo": "bar"},
		"store":               false,
		"include_usage":       false,
		"extra_query":         map[string]any{"foo": "bar"},
		"extra_headers":       map[string]any{"foo": "bar"},
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
		ToolChoice:        "auto",
		ParallelToolCalls: param.NewOpt(true),
		Truncation:        param.NewOpt(TruncationAuto),
		MaxTokens:         param.NewOpt[int64](100),
		Reasoning: openai.ReasoningParam{
			Effort:  openai.ReasoningEffortLow,
			Summary: openai.ReasoningSummaryConcise,
		},
		Metadata:     map[string]string{"foo": "bar"},
		Store:        param.NewOpt(false),
		IncludeUsage: param.NewOpt(false),
		ExtraQuery:   map[string]string{"foo": "bar"},
		ExtraHeaders: map[string]string{"foo": "bar"},
	}

	t.Run("overriding first set of properties", func(t *testing.T) {
		override := ModelSettings{
			Temperature:      param.NewOpt(0.4),
			FrequencyPenalty: param.NewOpt(0.1),
			ToolChoice:       "required",
			Truncation:       param.NewOpt(TruncationDisabled),
			Reasoning: openai.ReasoningParam{
				Effort:  openai.ReasoningEffortMedium,
				Summary: openai.ReasoningSummaryDetailed,
			},
			Store:      param.NewOpt(true),
			ExtraQuery: map[string]string{"a": "b"},
		}

		resolved := base.Resolve(override)

		want := ModelSettings{
			Temperature:       param.NewOpt(0.4),
			TopP:              param.NewOpt(0.9),
			FrequencyPenalty:  param.NewOpt(0.1),
			PresencePenalty:   param.NewOpt(0.0),
			ToolChoice:        "required",
			ParallelToolCalls: param.NewOpt(true),
			Truncation:        param.NewOpt(TruncationDisabled),
			MaxTokens:         param.NewOpt[int64](100),
			Reasoning: openai.ReasoningParam{
				Effort:  openai.ReasoningEffortMedium,
				Summary: openai.ReasoningSummaryDetailed,
			},
			Metadata:     map[string]string{"foo": "bar"},
			Store:        param.NewOpt(true),
			IncludeUsage: param.NewOpt(false),
			ExtraQuery:   map[string]string{"a": "b"},
			ExtraHeaders: map[string]string{"foo": "bar"},
		}

		assert.Equal(t, want, resolved)
	})

	t.Run("overriding second set of properties", func(t *testing.T) {
		override := ModelSettings{
			TopP:              param.NewOpt(0.8),
			PresencePenalty:   param.NewOpt(0.2),
			ParallelToolCalls: param.NewOpt(false),
			MaxTokens:         param.NewOpt[int64](42),
			Metadata:          map[string]string{"a": "b"},
			IncludeUsage:      param.NewOpt(true),
			ExtraHeaders:      map[string]string{"c": "d"},
		}

		resolved := base.Resolve(override)

		want := ModelSettings{
			Temperature:       param.NewOpt(0.5),
			TopP:              param.NewOpt(0.8),
			FrequencyPenalty:  param.NewOpt(0.0),
			PresencePenalty:   param.NewOpt(0.2),
			ToolChoice:        "auto",
			ParallelToolCalls: param.NewOpt(false),
			Truncation:        param.NewOpt(TruncationAuto),
			MaxTokens:         param.NewOpt[int64](42),
			Reasoning: openai.ReasoningParam{
				Effort:  openai.ReasoningEffortLow,
				Summary: openai.ReasoningSummaryConcise,
			},
			Metadata:     map[string]string{"a": "b"},
			Store:        param.NewOpt(false),
			IncludeUsage: param.NewOpt(true),
			ExtraQuery:   map[string]string{"foo": "bar"},
			ExtraHeaders: map[string]string{"c": "d"},
		}

		assert.Equal(t, want, resolved)
	})
}
