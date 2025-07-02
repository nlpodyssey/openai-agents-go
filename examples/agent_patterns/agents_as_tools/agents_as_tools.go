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
	"bufio"
	"context"
	"fmt"
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

/*
This example shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools. In this case, it picks from a set of translation
agents.
*/

const Model = "gpt-4o-mini"

var (
	SpanishAgent = agents.New("spanish_agent").
			WithInstructions("You translate the user's message to Spanish").
			WithHandoffDescription("An English to Spanish translator").
			WithModel(Model)

	FrenchAgent = agents.New("french_agent").
			WithInstructions("You translate the user's message to French").
			WithHandoffDescription("An English to French translator").
			WithModel(Model)

	ItalianAgent = agents.New("italian_agent").
			WithInstructions("You translate the user's message to Italian").
			WithHandoffDescription("An English to Italian translator").
			WithModel(Model)

	OrchestratorAgent = agents.New("orchestrator_agent").
				WithInstructions(
			"You are a translation agent. You use the tools given to you to translate. "+
				"If asked for multiple translations, you call the relevant tools in order. "+
				"You never translate on your own, you always use the provided tools.",
		).
		WithTools(
			SpanishAgent.AsTool(agents.AgentAsToolParams{
				ToolName:        "translate_to_spanish",
				ToolDescription: "Translate the user's message to Spanish",
			}),
			FrenchAgent.AsTool(agents.AgentAsToolParams{
				ToolName:        "translate_to_french",
				ToolDescription: "Translate the user's message to French",
			}),
			ItalianAgent.AsTool(agents.AgentAsToolParams{
				ToolName:        "translate_to_italian",
				ToolDescription: "Translate the user's message to Italian",
			}),
		).
		WithModel(Model)

	SynthesizerAgent = agents.New("synthesizer_agent").
				WithInstructions("You inspect translations, correct them if needed, and produce a final concatenated response.").
				WithModel(Model)
)

func main() {
	fmt.Print("Hi! What would you like translated, and to which languages? ")
	_ = os.Stdout.Sync()

	line, _, err := bufio.NewReader(os.Stdin).ReadLine()
	if err != nil {
		panic(err)
	}
	msg := string(line)

	// Run the entire orchestration in a single trace
	err = tracing.RunTrace(
		context.Background(),
		tracing.TraceParams{WorkflowName: "Orchestrator evaluator"},
		func(ctx context.Context, _ tracing.Trace) error {
			orchestratorResult, err := agents.Run(ctx, OrchestratorAgent, msg)
			if err != nil {
				return err
			}

			for _, item := range orchestratorResult.NewItems {
				if item, ok := item.(agents.MessageOutputItem); ok {
					text := agents.ItemHelpers().TextMessageOutput(item)
					if text != "" {
						fmt.Printf("  - Translation step: %s\n", text)
					}
				}
			}

			synthesizerResult, err := agents.RunInputs(ctx, SynthesizerAgent, orchestratorResult.ToInputList())
			if err != nil {
				return err
			}

			fmt.Printf("\nFinal response:\n%s\n", synthesizerResult.FinalOutput)
			return nil
		})
	if err != nil {
		panic(err)
	}
}
