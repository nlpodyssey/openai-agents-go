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
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
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

func (t FileSearchTool) isTool() {}
