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
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/openai/openai-go/packages/param"
)

/*
This example shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools. In this case, it picks from a set of translation
agents.
*/

var (
	Model = agents.NewAgentModelName("gpt-4o-mini")

	SpanishAgent = &agents.Agent{
		Name:               "spanish_agent",
		Instructions:       agents.InstructionsStr("You translate the user's message to Spanish"),
		HandoffDescription: "An English to Spanish translator",
		Model:              param.NewOpt(Model),
	}

	FrenchAgent = &agents.Agent{
		Name:               "french_agent",
		Instructions:       agents.InstructionsStr("You translate the user's message to French"),
		HandoffDescription: "An English to French translator",
		Model:              param.NewOpt(Model),
	}

	ItalianAgent = &agents.Agent{
		Name:               "italian_agent",
		Instructions:       agents.InstructionsStr("You translate the user's message to Italian"),
		HandoffDescription: "An English to Italian translator",
		Model:              param.NewOpt(Model),
	}

	OrchestratorAgent = &agents.Agent{
		Name: "orchestrator_agent",
		Instructions: agents.InstructionsStr(
			"You are a translation agent. You use the tools given to you to translate. " +
				"If asked for multiple translations, you call the relevant tools in order. " +
				"You never translate on your own, you always use the provided tools.",
		),
		Tools: []tools.Tool{
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
		},
		Model: param.NewOpt(Model),
	}

	SynthesizerAgent = &agents.Agent{
		Name: "synthesizer_agent",
		Instructions: agents.InstructionsStr(
			"You inspect translations, correct them if needed, and produce a final concatenated response.",
		),
		Model: param.NewOpt(Model),
	}
)

func main() {
	fmt.Print("Hi! What would you like translated, and to which languages? ")
	_ = os.Stdout.Sync()

	line, _, err := bufio.NewReader(os.Stdin).ReadLine()
	if err != nil {
		panic(err)
	}
	msg := string(line)

	orchestratorResult, err := agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: OrchestratorAgent,
		Input:         agents.InputString(msg),
	})
	if err != nil {
		panic(err)
	}

	for _, item := range orchestratorResult.NewItems {
		if item, ok := item.(agents.MessageOutputItem); ok {
			text := agents.ItemHelpers().TextMessageOutput(item)
			if text != "" {
				fmt.Printf("  - Translation step: %s\n", text)
			}
		}
	}

	synthesizerResult, err := agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: SynthesizerAgent,
		Input:         agents.InputItems(orchestratorResult.ToInputList()),
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("\nFinal response:\n%s\n", synthesizerResult.FinalOutput)
}
