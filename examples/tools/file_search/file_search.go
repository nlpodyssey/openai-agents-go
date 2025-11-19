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
)

func main() {
	agent := agents.New("File searcher").
		WithInstructions("You are a helpful agent.").
		WithTools(agents.FileSearchTool{
			VectorStoreIDs:       []string{"vs_67bf88953f748191be42b462090e53e7"},
			MaxNumResults:        param.NewOpt[int64](3),
			IncludeSearchResults: true,
		}).
		WithModel("gpt-4o-mini")

	err := tracing.RunTrace(
		context.Background(), tracing.TraceParams{WorkflowName: "File search example"},
		func(ctx context.Context, _ tracing.Trace) error {
			result, err := agents.Run(ctx, agent, "Be concise, and tell me 1 sentence about Arrakis I might not know.")
			if err != nil {
				return err
			}

			fmt.Println(result.FinalOutput)

			for _, item := range result.NewItems {
				fmt.Printf("%+v\n", item)
			}
			return nil
		},
	)
	if err != nil {
		panic(err)
	}
}
