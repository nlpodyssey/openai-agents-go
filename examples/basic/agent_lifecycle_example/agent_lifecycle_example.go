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
	"math/rand"
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/v3/packages/param"
)

type CustomAgentHooks struct {
	eventCounter int
	displayName  string
}

func NewCustomAgentHooks(displayName string) *CustomAgentHooks {
	return &CustomAgentHooks{displayName: displayName}
}

func (h *CustomAgentHooks) OnStart(_ context.Context, agent *agents.Agent) error {
	h.eventCounter += 1
	fmt.Printf(
		"### (%s) %d: Agent %s started\n",
		h.displayName, h.eventCounter, agent.Name,
	)
	return nil
}

func (h *CustomAgentHooks) OnEnd(_ context.Context, agent *agents.Agent, output any) error {
	h.eventCounter += 1
	fmt.Printf(
		"### (%s) %d: Agent %s ended with output %#v\n",
		h.displayName, h.eventCounter, agent.Name, output,
	)
	return nil
}

func (h *CustomAgentHooks) OnHandoff(_ context.Context, agent, source *agents.Agent) error {
	h.eventCounter += 1
	fmt.Printf(
		"### (%s) %d: Agent %s handed off to %s\n",
		h.displayName, h.eventCounter, source.Name, agent.Name,
	)
	return nil
}

func (h *CustomAgentHooks) OnToolStart(_ context.Context, agent *agents.Agent, tool agents.Tool, arguments any) error {
	h.eventCounter += 1
	fmt.Printf(
		"### (%s) %d: Agent %s started tool %s\n",
		h.displayName, h.eventCounter, agent.Name, tool.ToolName(),
	)
	return nil
}

func (h *CustomAgentHooks) OnToolEnd(_ context.Context, agent *agents.Agent, tool agents.Tool, result any) error {
	h.eventCounter += 1
	fmt.Printf(
		"### (%s) %d: Agent %s ended tool %s with result %#v\n",
		h.displayName, h.eventCounter, agent.Name, tool.ToolName(), result,
	)
	return nil
}

func (*CustomAgentHooks) OnLLMStart(context.Context, *agents.Agent, param.Opt[string], []agents.TResponseInputItem) error {
	return nil
}

func (*CustomAgentHooks) OnLLMEnd(context.Context, *agents.Agent, agents.ModelResponse) error {
	return nil
}

type RandomNumberArgs struct {
	Max int64 `json:"max"`
}

// RandomNumber generates a random number from 0 to max (inclusive).
func RandomNumber(_ context.Context, args RandomNumberArgs) (int64, error) {
	return rand.Int63n(args.Max + 1), nil
}

type MultiplyByTwoArgs struct {
	X int64 `json:"x"`
}

// MultiplyByTwo calculates a simple multiplication by two.
func MultiplyByTwo(_ context.Context, args MultiplyByTwoArgs) (int64, error) {
	return args.X * 2, nil
}

type FinalResult struct {
	Number int64 `json:"number"`
}

var (
	RandomNumberTool = agents.NewFunctionTool("random_number", "Generate a random number from 0 to max (inclusive).", RandomNumber)

	MultiplyByTwoTool = agents.NewFunctionTool("multiply_by_two", "Simple multiplication by two.", MultiplyByTwo)

	MultiplyAgent = agents.New("Multiply Agent").
			WithInstructions("Multiply the number by 2 and then return the final result.").
			WithTools(MultiplyByTwoTool).
			WithOutputType(agents.OutputType[FinalResult]()).
			WithHooks(NewCustomAgentHooks("Multiply Agent")).
			WithModel("gpt-4o-mini")

	StartAgent = agents.New("Start Agent").
			WithInstructions("Generate a random number. If it's even, stop. If it's odd, hand off to the multiply agent.").
			WithTools(RandomNumberTool).
			WithOutputType(agents.OutputType[FinalResult]()).
			WithAgentHandoffs(MultiplyAgent).
			WithHooks(NewCustomAgentHooks("Start Agent")).
			WithModel("gpt-4o-mini")
)

func main() {
	fmt.Print("Enter a max number: ")
	_ = os.Stdout.Sync()

	var userInput string
	_, err := fmt.Scan(&userInput)
	if err != nil {
		panic(err)
	}

	ctx := context.Background()

	_, err = agents.Run(
		ctx, StartAgent,
		fmt.Sprintf("Generate a random number between 0 and %s.", userInput),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println("Done!")
}
