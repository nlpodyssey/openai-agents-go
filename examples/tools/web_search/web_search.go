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
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

func main() {
	agent := agents.New("Web searcher").
		WithInstructions("You are a helpful agent.").
		WithTools(agents.WebSearchTool{
			UserLocation: responses.WebSearchToolUserLocationParam{
				City: param.NewOpt("New York"),
				Type: string(constant.ValueOf[constant.Approximate]()),
			},
		}).
		WithModel("gpt-4o-mini")

	err := tracing.RunTrace(
		context.Background(), tracing.TraceParams{WorkflowName: "Web search example"},
		func(ctx context.Context, _ tracing.Trace) error {

			result, err := agents.Run(
				ctx, agent,
				"Search the web for 'local sports news' and give me 1 interesting update in a sentence.",
			)
			if err != nil {
				panic(err)
			}

			fmt.Println(result.FinalOutput)
			return nil
		},
	)
	if err != nil {
		panic(err)
	}
}
