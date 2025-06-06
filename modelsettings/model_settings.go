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
	"github.com/nlpodyssey/openai-agents-go/types/optional"
	"github.com/openai/openai-go"
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
	Temperature optional.Optional[float64] `json:"temperature"`

	// The top_p to use when calling the model.
	TopP optional.Optional[float64] `json:"top_p"`

	// The frequency penalty to use when calling the model.
	FrequencyPenalty optional.Optional[float64] `json:"frequency_penalty"`

	// The presence penalty to use when calling the model.
	PresencePenalty optional.Optional[float64] `json:"presence_penalty"`

	// Optional tool choice to use when calling the model.
	// Well-known values are: "auto", "required", "none".
	ToolChoice string `json:"tool_choice"`

	// Whether to use parallel tool calls when calling the model.
	// Defaults to false if not provided.
	ParallelToolCalls optional.Optional[bool] `json:"parallel_tool_calls"`

	// The truncation strategy to use when calling the model.
	Truncation optional.Optional[Truncation] `json:"truncation"`

	// The maximum number of output tokens to generate.
	MaxTokens optional.Optional[int64] `json:"max_tokens"`

	// Configuration options for reasoning models
	// (see https://platform.openai.com/docs/guides/reasoning).
	Reasoning optional.Optional[openai.ReasoningParam] `json:"reasoning"`

	// Metadata to include with the model response call.
	Metadata optional.Optional[map[string]string] `json:"metadata"`

	// Whether to store the generated model response for later retrieval.
	// Defaults to true if not provided.
	Store optional.Optional[bool] `json:"store"`

	// Whether to include usage chunk.
	// Defaults to true if not provided.
	IncludeUsage optional.Optional[bool] `json:"include_usage"`

	// Additional query fields to provide with the request.
	ExtraQuery optional.Optional[map[string]string] `json:"extra_query"`

	// Additional headers to provide with the request.
	ExtraHeaders optional.Optional[map[string]string] `json:"extra_headers"`
}

type Truncation string

const (
	TruncationAuto     Truncation = "auto"
	TruncationDisabled Truncation = "disabled"
)

// Resolve produces a new ModelSettings by overlaying any present values from
// the override on top of this instance.
func (ms ModelSettings) Resolve(override optional.Optional[ModelSettings]) ModelSettings {
	newSettings := ms

	if !override.Present {
		return newSettings
	}

	resolveOptionalValue(&newSettings.Temperature, override.Value.Temperature)
	resolveOptionalValue(&newSettings.TopP, override.Value.TopP)
	resolveOptionalValue(&newSettings.FrequencyPenalty, override.Value.FrequencyPenalty)
	resolveOptionalValue(&newSettings.PresencePenalty, override.Value.PresencePenalty)
	resolveOptionalZeroValue(&newSettings.ToolChoice, override.Value.ToolChoice)
	resolveOptionalValue(&newSettings.ParallelToolCalls, override.Value.ParallelToolCalls)
	resolveOptionalValue(&newSettings.Truncation, override.Value.Truncation)
	resolveOptionalValue(&newSettings.MaxTokens, override.Value.MaxTokens)
	resolveOptionalValue(&newSettings.Reasoning, override.Value.Reasoning)
	resolveOptionalValue(&newSettings.Metadata, override.Value.Metadata)
	resolveOptionalValue(&newSettings.Store, override.Value.Store)
	resolveOptionalValue(&newSettings.IncludeUsage, override.Value.IncludeUsage)
	resolveOptionalValue(&newSettings.ExtraQuery, override.Value.ExtraQuery)
	resolveOptionalValue(&newSettings.ExtraHeaders, override.Value.ExtraHeaders)

	return newSettings
}

func resolveOptionalValue[T any](base *optional.Optional[T], override optional.Optional[T]) {
	if override.Present {
		*base = override
	}
}

func resolveOptionalZeroValue[T comparable](base *T, override T) {
	var zero T
	if override != zero {
		*base = override
	}
}
