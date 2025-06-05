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

	"github.com/nlpodyssey/openai-agents-go/types/optional"
	"github.com/openai/openai-go/shared"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Tests whether ModelSettings can be serialized to a JSON string.
func TestModelSettings_BasicSerialization(t *testing.T) {
	modelSettings := ModelSettings{
		Temperature: optional.Value(0.5),
		TopP:        optional.Value(0.9),
		MaxTokens:   optional.Value[int64](100),
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
		"reasoning":           nil,
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
		Temperature:       optional.Value(0.5),
		TopP:              optional.Value(0.9),
		FrequencyPenalty:  optional.Value(0.0),
		PresencePenalty:   optional.Value(0.0),
		ToolChoice:        "auto",
		ParallelToolCalls: optional.Value(true),
		Truncation:        optional.Value(TruncationAuto),
		MaxTokens:         optional.Value[int64](100),
		Reasoning:         optional.Value(shared.ReasoningParam{}),
		Metadata:          optional.Value(map[string]string{"foo": "bar"}),
		Store:             optional.Value(false),
		IncludeUsage:      optional.Value(false),
		ExtraQuery:        optional.Value(map[string]string{"foo": "bar"}),
		ExtraHeaders:      optional.Value(map[string]string{"foo": "bar"}),
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
		Temperature:       optional.Value[float64](0.5),
		TopP:              optional.Value[float64](0.9),
		FrequencyPenalty:  optional.Value[float64](0.0),
		PresencePenalty:   optional.Value[float64](0.0),
		ToolChoice:        "auto",
		ParallelToolCalls: optional.Value(true),
		Truncation:        optional.Value(TruncationAuto),
		MaxTokens:         optional.Value[int64](100),
		Reasoning: optional.Value(shared.ReasoningParam{
			Effort:          shared.ReasoningEffortLow,
			GenerateSummary: shared.ReasoningGenerateSummaryConcise,
		}),
		Metadata:     optional.Value(map[string]string{"foo": "bar"}),
		Store:        optional.Value(false),
		IncludeUsage: optional.Value(false),
		ExtraQuery:   optional.Value(map[string]string{"foo": "bar"}),
		ExtraHeaders: optional.Value(map[string]string{"foo": "bar"}),
	}

	t.Run("overriding first set of properties", func(t *testing.T) {
		override := ModelSettings{
			Temperature:      optional.Value(0.4),
			FrequencyPenalty: optional.Value(0.1),
			ToolChoice:       "required",
			Truncation:       optional.Value(TruncationDisabled),
			Reasoning: optional.Value(shared.ReasoningParam{
				Effort:          shared.ReasoningEffortMedium,
				GenerateSummary: shared.ReasoningGenerateSummaryDetailed,
			}),
			Store:      optional.Value(true),
			ExtraQuery: optional.Value(map[string]string{"a": "b"}),
		}

		resolved := base.Resolve(optional.Value(override))

		want := ModelSettings{
			Temperature:       optional.Value(0.4),
			TopP:              optional.Value(0.9),
			FrequencyPenalty:  optional.Value(0.1),
			PresencePenalty:   optional.Value(0.0),
			ToolChoice:        "required",
			ParallelToolCalls: optional.Value(true),
			Truncation:        optional.Value(TruncationDisabled),
			MaxTokens:         optional.Value[int64](100),
			Reasoning: optional.Value(shared.ReasoningParam{
				Effort:          shared.ReasoningEffortMedium,
				GenerateSummary: shared.ReasoningGenerateSummaryDetailed,
			}),
			Metadata:     optional.Value(map[string]string{"foo": "bar"}),
			Store:        optional.Value(true),
			IncludeUsage: optional.Value(false),
			ExtraQuery:   optional.Value(map[string]string{"a": "b"}),
			ExtraHeaders: optional.Value(map[string]string{"foo": "bar"}),
		}

		assert.Equal(t, want, resolved)
	})

	t.Run("overriding second set of properties", func(t *testing.T) {
		override := ModelSettings{
			TopP:              optional.Value(0.8),
			PresencePenalty:   optional.Value(0.2),
			ParallelToolCalls: optional.Value(false),
			MaxTokens:         optional.Value[int64](42),
			Metadata:          optional.Value(map[string]string{"a": "b"}),
			IncludeUsage:      optional.Value(true),
			ExtraHeaders:      optional.Value(map[string]string{"c": "d"}),
		}

		resolved := base.Resolve(optional.Value(override))

		want := ModelSettings{
			Temperature:       optional.Value(0.5),
			TopP:              optional.Value(0.8),
			FrequencyPenalty:  optional.Value(0.0),
			PresencePenalty:   optional.Value(0.2),
			ToolChoice:        "auto",
			ParallelToolCalls: optional.Value(false),
			Truncation:        optional.Value(TruncationAuto),
			MaxTokens:         optional.Value[int64](42),
			Reasoning: optional.Value(shared.ReasoningParam{
				Effort:          shared.ReasoningEffortLow,
				GenerateSummary: shared.ReasoningGenerateSummaryConcise,
			}),
			Metadata:     optional.Value(map[string]string{"a": "b"}),
			Store:        optional.Value(false),
			IncludeUsage: optional.Value(true),
			ExtraQuery:   optional.Value(map[string]string{"foo": "bar"}),
			ExtraHeaders: optional.Value(map[string]string{"c": "d"}),
		}

		assert.Equal(t, want, resolved)
	})
}
