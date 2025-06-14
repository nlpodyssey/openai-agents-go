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
	"context"
	"errors"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
)

// WebSearchTool is a hosted tool that lets the LLM search the web.
// Currently only supported with OpenAI models, using the Responses API.
type WebSearchTool struct {
	// Optional location for the search. Lets you customize results to be relevant to a location.
	UserLocation responses.WebSearchToolUserLocationParam

	// Optional amount of context to use for the search. Default: "medium".
	SearchContextSize responses.WebSearchToolSearchContextSize
}

func (t WebSearchTool) ToolName() string {
	return "web_search_preview"
}

func (t WebSearchTool) ConvertToResponses(context.Context) (*responses.ToolUnionParam, *responses.ResponseIncludable, error) {
	return &responses.ToolUnionParam{
		OfWebSearchPreview: &responses.WebSearchToolParam{
			Type:              responses.WebSearchToolTypeWebSearchPreview,
			UserLocation:      t.UserLocation,
			SearchContextSize: t.SearchContextSize,
		},
	}, nil, nil
}

func (t WebSearchTool) ConvertToChatCompletions(context.Context) (*openai.ChatCompletionToolParam, error) {
	return nil, errors.New("WebSearchTool.ConvertToChatCompletions not implemented")
}
