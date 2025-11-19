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
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/v3/packages/param"
)

type ExampleHooks struct {
	eventCounter int
}

func (*ExampleHooks) usageToStr(u *usage.Usage) string {
	return fmt.Sprintf(
		"%d requests, %d input tokens, %d output tokens, %d total tokens",
		u.Requests, u.InputTokens, u.OutputTokens, u.TotalTokens,
	)
}

func (*ExampleHooks) OnLLMStart(context.Context, *agents.Agent, param.Opt[string], []agents.TResponseInputItem) error {
	return nil
}

func (*ExampleHooks) OnLLMEnd(context.Context, *agents.Agent, agents.ModelResponse) error {
	return nil
}

func (e *ExampleHooks) OnAgentStart(ctx context.Context, agent *agents.Agent) error {
	e.eventCounter += 1
	u, _ := usage.FromContext(ctx)
	fmt.Printf(
		"### %d: Agent %s started. Usage: %s\n",
		e.eventCounter, agent.Name, e.usageToStr(u),
	)
	return nil
}

func (e *ExampleHooks) OnAgentEnd(ctx context.Context, agent *agents.Agent, output any) error {
	e.eventCounter += 1
	u, _ := usage.FromContext(ctx)
	fmt.Printf(
		"### %d: Agent %s ended with output %#v. Usage: %s\n",
		e.eventCounter, agent.Name, output, e.usageToStr(u),
	)
	return nil
}

func (e *ExampleHooks) OnToolStart(ctx context.Context, _ *agents.Agent, tool agents.Tool) error {
	e.eventCounter += 1
	u, _ := usage.FromContext(ctx)
	fmt.Printf(
		"### %d: Tool %s started. Usage: %s\n",
		e.eventCounter, tool.ToolName(), e.usageToStr(u),
	)
	return nil
}

func (e *ExampleHooks) OnToolEnd(ctx context.Context, _ *agents.Agent, tool agents.Tool, result any) error {
	e.eventCounter += 1
	u, _ := usage.FromContext(ctx)
	fmt.Printf(
		"### %d: Tool %s ended with result %#v. Usage: %s\n",
		e.eventCounter, tool.ToolName(), result, e.usageToStr(u),
	)
	return nil
}

func (e *ExampleHooks) OnHandoff(ctx context.Context, fromAgent, toAgent *agents.Agent) error {
	e.eventCounter += 1
	u, _ := usage.FromContext(ctx)
	fmt.Printf(
		"### %d: Handoff from %s to %s. Usage: %s\n",
		e.eventCounter, fromAgent.Name, toAgent.Name, e.usageToStr(u),
	)
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

// MultiplyByTwo returns x times two.
func MultiplyByTwo(_ context.Context, args MultiplyByTwoArgs) (int64, error) {
	return args.X * 2, nil
}

type FinalResult struct {
	Number int64 `json:"number"`
}

var (
	Hooks = &ExampleHooks{}

	RandomNumberTool = agents.NewFunctionTool("random_number", "Generate a random number from 0 to max (inclusive).", RandomNumber)

	MultiplyByTwoTool = agents.NewFunctionTool("multiply_by_two", "Return x times two.", MultiplyByTwo)

	MultiplyAgent = agents.New("Multiply Agent").
			WithInstructions("Multiply the number by 2 and then return the final result.").
			WithTools(MultiplyByTwoTool).
			WithOutputType(agents.OutputType[FinalResult]()).
			WithModel("gpt-4o-mini")

	StartAgent = agents.New("Start Agent").
			WithInstructions("Generate a random number. If it's even, stop. If it's odd, hand off to the multiplier agent.").
			WithTools(RandomNumberTool).
			WithOutputType(agents.OutputType[FinalResult]()).
			WithAgentHandoffs(MultiplyAgent).
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

	_, err = (agents.Runner{Config: agents.RunConfig{Hooks: Hooks}}).Run(
		ctx, StartAgent,
		fmt.Sprintf("Generate a random number between 0 and %s.", userInput),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println("Done!")
}
