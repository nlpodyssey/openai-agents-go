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
	"errors"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
)

// FileSearchTool is a hosted tool that lets the LLM search through a vector store.
// Currently only supported with OpenAI models, using the Responses API.
type FileSearchTool struct {
	// The IDs of the vector stores to search.
	VectorStoreIDs []string

	// The maximum number of results to return.
	MaxNumResults param.Opt[int64]

	// Whether to include the search results in the output produced by the LLM.
	IncludeSearchResults bool

	// Optional ranking options for search.
	RankingOptions responses.FileSearchToolRankingOptionsParam

	// Optional filter to apply based on file attributes.
	Filters responses.FileSearchToolFiltersUnionParam
}

func (t FileSearchTool) ToolName() string {
	return "file_search"
}

func (t FileSearchTool) ConvertToResponses(context.Context) (*responses.ToolUnionParam, *responses.ResponseIncludable, error) {
	convertedTool := &responses.ToolUnionParam{
		OfFileSearch: &responses.FileSearchToolParam{
			VectorStoreIDs: t.VectorStoreIDs,
			MaxNumResults:  t.MaxNumResults,
			Filters:        t.Filters,
			RankingOptions: t.RankingOptions,
			Type:           constant.ValueOf[constant.FileSearch](),
		},
	}

	var includes *responses.ResponseIncludable
	if t.IncludeSearchResults {
		includes = new(responses.ResponseIncludable)
		*includes = responses.ResponseIncludableFileSearchCallResults
	}

	return convertedTool, includes, nil
}

func (t FileSearchTool) ConvertToChatCompletions(context.Context) (*openai.ChatCompletionToolParam, error) {
	return nil, errors.New("FileSearchTool.ConvertToChatCompletions not implemented")
}
