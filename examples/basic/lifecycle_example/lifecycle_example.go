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
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/nlpodyssey/openai-agents-go/types/optional"
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

func (e *ExampleHooks) OnAgentStart(_ context.Context, cw *runcontext.RunContextWrapper, agent *agents.Agent) error {
	e.eventCounter += 1
	fmt.Printf(
		"### %d: Agent %s started. Usage: %s\n",
		e.eventCounter, agent.Name, e.usageToStr(cw.Usage),
	)
	return nil
}

func (e *ExampleHooks) OnAgentEnd(_ context.Context, cw *runcontext.RunContextWrapper, agent *agents.Agent, output any) error {
	e.eventCounter += 1
	fmt.Printf(
		"### %d: Agent %s ended with output %#v. Usage: %s\n",
		e.eventCounter, agent.Name, output, e.usageToStr(cw.Usage),
	)
	return nil
}

func (e *ExampleHooks) OnToolStart(_ context.Context, cw *runcontext.RunContextWrapper, _ *agents.Agent, tool agents.Tool) error {
	e.eventCounter += 1
	funcTool := tool.(agents.FunctionTool)
	fmt.Printf(
		"### %d: Tool %s started. Usage: %s\n",
		e.eventCounter, funcTool.Name, e.usageToStr(cw.Usage),
	)
	return nil
}

func (e *ExampleHooks) OnToolEnd(_ context.Context, cw *runcontext.RunContextWrapper, _ *agents.Agent, tool agents.Tool, result any) error {
	e.eventCounter += 1
	funcTool := tool.(agents.FunctionTool)
	fmt.Printf(
		"### %d: Tool %s ended with result %#v. Usage: %s\n",
		e.eventCounter, funcTool.Name, result, e.usageToStr(cw.Usage),
	)
	return nil
}

func (e *ExampleHooks) OnHandoff(_ context.Context, cw *runcontext.RunContextWrapper, fromAgent, toAgent *agents.Agent) error {
	e.eventCounter += 1
	fmt.Printf(
		"### %d: Handoff from %s to %s. Usage: %s\n",
		e.eventCounter, fromAgent.Name, toAgent.Name, e.usageToStr(cw.Usage),
	)
	return nil
}

type RandomNumberArgs struct {
	Max int64 `json:"max"`
}

// RandomNumber generates a random number up to the provided max.
func RandomNumber(args RandomNumberArgs) int64 {
	return rand.Int63n(args.Max + 1)
}

type MultiplyByTwoArgs struct {
	X int64 `json:"x"`
}

// MultiplyByTwo return x times two.
func MultiplyByTwo(args MultiplyByTwoArgs) int64 {
	return args.X * 2
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

	RandomNumberTool = agents.FunctionTool{
		Name:        "random_number",
		Description: "Generate a random number up to the provided max.",
		ParamsJSONSchema: map[string]any{
			"title":                "random_number_args",
			"type":                 "object",
			"required":             []string{"max"},
			"additionalProperties": false,
			"properties": map[string]any{
				"max": map[string]any{
					"title": "Max",
					"type":  "integer",
				},
			},
		},
		OnInvokeTool: func(_ context.Context, _ *runcontext.RunContextWrapper, arguments string) (any, error) {
			var args RandomNumberArgs
			err := json.Unmarshal([]byte(arguments), &args)
			if err != nil {
				return nil, err
			}
			return RandomNumber(args), nil
		},
	}

	MultiplyByTwoTool = agents.FunctionTool{
		Name:        "multiply_by_two",
		Description: "Return x times two.",
		ParamsJSONSchema: map[string]any{
			"title":                "multiply_by_two_args",
			"type":                 "object",
			"required":             []string{"x"},
			"additionalProperties": false,
			"properties": map[string]any{
				"x": map[string]any{
					"title": "X",
					"type":  "integer",
				},
			},
		},
		OnInvokeTool: func(_ context.Context, _ *runcontext.RunContextWrapper, arguments string) (any, error) {
			var args MultiplyByTwoArgs
			err := json.Unmarshal([]byte(arguments), &args)
			if err != nil {
				return nil, err
			}
			return MultiplyByTwo(args), nil
		},
	}

	MultiplyAgent = &agents.Agent{
		Name: "Multiply Agent",
		Instructions: agents.StringInstructions(
			"Multiply the number by 2 and then return the final result.",
		),
		Tools:        []agents.Tool{MultiplyByTwoTool},
		OutputSchema: FinalResultOutputSchema{},
		Model:        optional.Value(agents.NewAgentModelName("gpt-4o-mini")),
	}

	StartAgent = &agents.Agent{
		Name: "Start Agent",
		Instructions: agents.StringInstructions(
			"Generate a random number. If it's even, stop. If it's odd, hand off to the multiplier agent.",
		),
		Tools:        []agents.Tool{RandomNumberTool},
		OutputSchema: FinalResultOutputSchema{},
		Handoffs:     []agents.AgentHandoff{MultiplyAgent},
		Model:        optional.Value(agents.NewAgentModelName("gpt-4o-mini")),
	}
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
