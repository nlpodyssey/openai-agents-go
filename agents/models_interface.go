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
	"iter"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/openai/openai-go/packages/param"
)

// Model is the base interface for calling an LLM.
type Model interface {
	// GetResponse returns the full model response from the model.
	GetResponse(context.Context, ModelGetResponseParams) (*ModelResponse, error)

	// StreamResponse streams a response from the model.
	StreamResponse(context.Context, ModelStreamResponseParams) (iter.Seq2[*TResponseStreamEvent, error], error)
}

type ModelGetResponseParams struct {
	// The system instructions to use.
	SystemInstructions param.Opt[string]

	// The input items to the model, in OpenAI Responses format.
	Input Input

	// The model settings to use.
	ModelSettings modelsettings.ModelSettings

	// The tools available to the model.
	Tools []tools.Tool

	// Optional output schema to use.
	OutputSchema AgentOutputSchemaInterface

	// The handoffs available to the model.
	Handoffs []Handoff

	// Optional ID of the previous response. Generally not used by the model,
	// except for the OpenAI Responses API.
	PreviousResponseID string
}

type ModelStreamResponseParams struct {
	// The system instructions to use.
	SystemInstructions param.Opt[string]

	// The input items to the model, in OpenAI Responses format.
	Input Input

	// The model settings to use.
	ModelSettings modelsettings.ModelSettings

	// The tools available to the model.
	Tools []tools.Tool

	// Optional output schema to use.
	OutputSchema AgentOutputSchemaInterface

	// The handoffs available to the model.
	Handoffs []Handoff

	// Optional ID of the previous response. Generally not used by the model, except for the OpenAI Responses API.
	PreviousResponseID string
}

// ModelProvider is the base interface for a model provider.
// It is responsible for looking up Models by name.
type ModelProvider interface {
	// GetModel returns a model by name.
	GetModel(modelName string) (Model, error)
}
