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

package main

import (
	"context"
	"fmt"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
)

func main() {
	agent := agents.New("Web searcher").
		WithInstructions("You are a helpful agent.").
		WithTools(tools.WebSearchTool{
			UserLocation: responses.WebSearchToolUserLocationParam{
				City: param.NewOpt("New York"),
				Type: constant.ValueOf[constant.Approximate](),
			},
		}).
		WithModel("gpt-4o-mini")

	result, err := agents.Run(
		context.Background(),
		agent,
		"Search the web for 'local sports news' and give me 1 interesting update in a sentence.",
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(result.FinalOutput)
}
