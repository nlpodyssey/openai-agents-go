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

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

type ModelTracing uint8

const (
	// ModelTracingDisabled means that tracing is disabled entirely.
	ModelTracingDisabled ModelTracing = iota
	// ModelTracingEnabled means that tracing is enabled, and all data is included.
	ModelTracingEnabled
	// ModelTracingEnabledWithoutData means that tracing is enabled, but inputs/outputs are not included.
	ModelTracingEnabledWithoutData
)

func (mt ModelTracing) IsDisabled() bool  { return mt == ModelTracingDisabled }
func (mt ModelTracing) IncludeData() bool { return mt == ModelTracingEnabled }

// Model is the base interface for calling an LLM.
type Model interface {
	// GetResponse returns the full model response from the model.
	GetResponse(context.Context, ModelResponseParams) (*ModelResponse, error)

	// StreamResponse streams a response from the model.
	StreamResponse(context.Context, ModelResponseParams, ModelStreamResponseCallback) error
}

type ModelStreamResponseCallback = func(context.Context, TResponseStreamEvent) error

type ModelResponseParams struct {
	// The system instructions to use.
	SystemInstructions param.Opt[string]

	// The input items to the model, in OpenAI Responses format.
	Input Input

	// The model settings to use.
	ModelSettings modelsettings.ModelSettings

	// The tools available to the model.
	Tools []Tool

	// Optional output type to use.
	OutputType OutputTypeInterface

	// The handoffs available to the model.
	Handoffs []Handoff

	// Tracing configuration.
	Tracing ModelTracing

	// Optional ID of the previous response. Generally not used by the model,
	// except for the OpenAI Responses API.
	PreviousResponseID string

	// Optional prompt config to use for the model.
	Prompt responses.ResponsePromptParam
}

// ModelProvider is the base interface for a model provider.
// It is responsible for looking up Models by name.
type ModelProvider interface {
	// GetModel returns a model by name.
	GetModel(modelName string) (Model, error)
}
