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
	"maps"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

// ModelSettings holds settings to use when calling an LLM.
//
// This type holds optional model configuration parameters (e.g. temperature,
// top-p, penalties, truncation, etc.).
//
// Not all models/providers support all of these parameters, so please check
// the API documentation for the specific model and provider you are using.
type ModelSettings struct {
	// The temperature to use when calling the model.
	Temperature param.Opt[float64] `json:"temperature"`

	// The top_p to use when calling the model.
	TopP param.Opt[float64] `json:"top_p"`

	// The frequency penalty to use when calling the model.
	FrequencyPenalty param.Opt[float64] `json:"frequency_penalty"`

	// The presence penalty to use when calling the model.
	PresencePenalty param.Opt[float64] `json:"presence_penalty"`

	// Optional tool choice to use when calling the model.
	// Well-known values are: "auto", "required", "none".
	ToolChoice string `json:"tool_choice"`

	// Whether to use parallel tool calls when calling the model.
	// Defaults to false if not provided.
	ParallelToolCalls param.Opt[bool] `json:"parallel_tool_calls"`

	// The truncation strategy to use when calling the model.
	Truncation param.Opt[Truncation] `json:"truncation"`

	// The maximum number of output tokens to generate.
	MaxTokens param.Opt[int64] `json:"max_tokens"`

	// Optional configuration options for reasoning models
	// (see https://platform.openai.com/docs/guides/reasoning).
	Reasoning openai.ReasoningParam `json:"reasoning"`

	// Optional metadata to include with the model response call.
	Metadata map[string]string `json:"metadata"`

	// Whether to store the generated model response for later retrieval.
	// Defaults to true if not provided.
	Store param.Opt[bool] `json:"store"`

	// Whether to include usage chunk.
	// Defaults to true if not provided.
	IncludeUsage param.Opt[bool] `json:"include_usage"`

	// Optional additional query fields to provide with the request.
	ExtraQuery map[string]string `json:"extra_query"`

	// Optional additional headers to provide with the request.
	ExtraHeaders map[string]string `json:"extra_headers"`
}

type Truncation string

const (
	TruncationAuto     Truncation = "auto"
	TruncationDisabled Truncation = "disabled"
)

// Resolve produces a new ModelSettings by overlaying any present values from
// the override on top of this instance.
func (ms ModelSettings) Resolve(override ModelSettings) ModelSettings {
	newSettings := ms
	resolveOpt(&newSettings.Temperature, override.Temperature)
	resolveOpt(&newSettings.TopP, override.TopP)
	resolveOpt(&newSettings.FrequencyPenalty, override.FrequencyPenalty)
	resolveOpt(&newSettings.PresencePenalty, override.PresencePenalty)
	resolveComparable(&newSettings.ToolChoice, override.ToolChoice)
	resolveOpt(&newSettings.ParallelToolCalls, override.ParallelToolCalls)
	resolveOpt(&newSettings.Truncation, override.Truncation)
	resolveOpt(&newSettings.MaxTokens, override.MaxTokens)
	resolveComparable(&newSettings.Reasoning, override.Reasoning)
	resolveMap(&newSettings.Metadata, override.Metadata)
	resolveOpt(&newSettings.Store, override.Store)
	resolveOpt(&newSettings.IncludeUsage, override.IncludeUsage)
	resolveMap(&newSettings.ExtraQuery, override.ExtraQuery)
	resolveMap(&newSettings.ExtraHeaders, override.ExtraHeaders)
	return newSettings
}

func resolveOpt[T comparable](base *param.Opt[T], override param.Opt[T]) {
	if override.Valid() {
		*base = override
	}
}

func resolveComparable[T comparable](base *T, override T) {
	var zero T
	if override != zero {
		*base = override
	}
}

func resolveMap[M ~map[K]V, K comparable, V any](base *M, override M) {
	if len(override) > 0 {
		*base = maps.Clone(override)
	}
}
