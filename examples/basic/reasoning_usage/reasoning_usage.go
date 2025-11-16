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
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/v3"
)

func main() {
	agent := agents.New("Reasoner").
		WithInstructions("Solve the math problem step by step.").
		WithModel("gpt-5-mini").
		WithModelSettings(modelsettings.ModelSettings{
			// Note: Summary MUST be specified or reasoning tokens won't be emitted.
			Reasoning: openai.ReasoningParam{
				Effort:  openai.ReasoningEffortMedium,
				Summary: openai.ReasoningSummaryAuto,
			},
		})

	u := usage.NewUsage()
	ctx := usage.NewContext(context.Background(), u)
	events, errs, err := agents.RunStreamedChan(ctx, agent, "What is 81 multiplied by 17?")
	if err != nil {
		panic(err)
	}

	for event := range events {
		if e, ok := event.(agents.RawResponsesStreamEvent); ok {
			switch e.Data.Type {
			case "response.reasoning_summary_text.delta":
				fmt.Print(e.Data.Delta)
				_ = os.Stdout.Sync()
			case "response.reasoning_summary_text.done":
				fmt.Printf("\n--- Reasoning done ---\n\n")
				_ = os.Stdout.Sync()
			case "response.output_text.delta":
				fmt.Print(e.Data.Delta)
				_ = os.Stdout.Sync()
			case "response.output_text.done":
				fmt.Printf("\n")
				_ = os.Stdout.Sync()
			}
		}
	}

	if streamErr := <-errs; streamErr != nil {
		panic(streamErr)
	}

	printTokenUsage(u)
}

func printTokenUsage(u *usage.Usage) {
	reasoningPct := float64(u.OutputTokensDetails.ReasoningTokens) / float64(u.OutputTokens) * 100
	completionTokens := int64(u.OutputTokens) - u.OutputTokensDetails.ReasoningTokens

	fmt.Printf(`
┌─────────────────────────────┐
│     TOKEN USAGE REPORT      │
├─────────────────────────────┤
│ Requests:          %8d │
│ Input Tokens:      %8d │
│ Output Tokens:     %8d │
│   ├─ Reasoning:    %8d │
│   └─ Completion:   %8d │
│ Total Tokens:      %8d │
│ Reasoning:          %6.1f%% │
└─────────────────────────────┘
`, u.Requests, u.InputTokens, u.OutputTokens,
		u.OutputTokensDetails.ReasoningTokens, completionTokens,
		u.TotalTokens, reasoningPct)
}
