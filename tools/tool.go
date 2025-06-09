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

package tools

import (
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
)

// A Tool that can be used in an Agent.
type Tool interface {
	// ToolName returns the name of the tool.
	ToolName() string

	// ConvertToResponses converts a Tool to an OpenAI Responses API object,
	// optionally providing the name of an additional output data to include
	// in the model response.
	//
	// If you don't plan to use Responses API, this function can panic or
	// always return an error (preferred).
	ConvertToResponses() (*responses.ToolUnionParam, *responses.ResponseIncludable, error)

	// ConvertToChatCompletions converts a Tool to an OpenAI Chat Completions
	// API object.
	//
	// If you don't plan to use Chat Completions API, this function can panic or
	// always return an error (preferred).
	ConvertToChatCompletions() (*openai.ChatCompletionToolParam, error)
}
