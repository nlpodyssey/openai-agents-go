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
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/nlpodyssey/openai-agents-go/usage"
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

func (e *ExampleHooks) OnToolStart(ctx context.Context, _ *agents.Agent, tool tools.Tool) error {
	e.eventCounter += 1
	u, _ := usage.FromContext(ctx)
	fmt.Printf(
		"### %d: Tool %s started. Usage: %s\n",
		e.eventCounter, tool.ToolName(), e.usageToStr(u),
	)
	return nil
}

func (e *ExampleHooks) OnToolEnd(ctx context.Context, _ *agents.Agent, tool tools.Tool, result any) error {
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

// RandomNumber generates a random number up to the provided max.
func RandomNumber(_ context.Context, args RandomNumberArgs) (int64, error) {
	return rand.Int63n(args.Max + 1), nil
}

type MultiplyByTwoArgs struct {
	X int64 `json:"x"`
}

// MultiplyByTwo return x times two.
func MultiplyByTwo(_ context.Context, args MultiplyByTwoArgs) (int64, error) {
	return args.X * 2, nil
}

type FinalResult struct {
	Number int64 `json:"number"`
}

type FinalResultOutputSchema struct{}

func (f FinalResultOutputSchema) Name() string             { return "FinalResult" }
func (f FinalResultOutputSchema) IsPlainText() bool        { return false }
func (f FinalResultOutputSchema) IsStrictJSONSchema() bool { return true }
func (f FinalResultOutputSchema) JSONSchema() map[string]any {
	return map[string]any{
		"title":                "FinalResult",
		"type":                 "object",
		"required":             []string{"number"},
		"additionalProperties": false,
		"properties": map[string]any{
			"number": map[string]any{
				"title": "Number",
				"type":  "integer",
			},
		},
	}
}
func (f FinalResultOutputSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()

	var v FinalResult
	err := dec.Decode(&v)
	return v, err
}

var (
	Hooks = &ExampleHooks{}

	RandomNumberTool = tools.NewFunctionTool("random_number", "Generate a random number up to the provided max.", RandomNumber)

	MultiplyByTwoTool = tools.NewFunctionTool("multiply_by_two", "Return x times two.", MultiplyByTwo)

	MultiplyAgent = agents.NewAgent().
			WithName("Multiply Agent").
			WithInstructions("Multiply the number by 2 and then return the final result.").
			WithTools(MultiplyByTwoTool).
			WithOutputSchema(FinalResultOutputSchema{}).
			WithModel("gpt-4o-mini")

	StartAgent = agents.NewAgent().
			WithName("Start Agent").
			WithInstructions("Generate a random number. If it's even, stop. If it's odd, hand off to the multiplier agent.").
			WithTools(RandomNumberTool).
			WithOutputSchema(FinalResultOutputSchema{}).
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

	_, err = agents.Runner().Run(ctx, agents.RunParams{
		StartingAgent: StartAgent,
		Hooks:         Hooks,
		Input: agents.InputString(
			fmt.Sprintf("Generate a random number between 0 and %s.", userInput),
		),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("Done!")
}
