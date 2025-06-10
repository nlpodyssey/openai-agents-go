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
	"github.com/openai/openai-go/packages/param"
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

func (h *CustomAgentHooks) OnToolStart(_ context.Context, agent *agents.Agent, tool tools.Tool) error {
	h.eventCounter += 1
	fmt.Printf(
		"### (%s) %d: Agent %s started tool %s\n",
		h.displayName, h.eventCounter, agent.Name, tool.ToolName(),
	)
	return nil
}

func (h *CustomAgentHooks) OnToolEnd(_ context.Context, agent *agents.Agent, tool tools.Tool, result any) error {
	h.eventCounter += 1
	fmt.Printf(
		"### (%s) %d: Agent %s ended tool %s with result %#v\n",
		h.displayName, h.eventCounter, agent.Name, tool.ToolName(), result,
	)
	return nil
}

type RandomNumberArgs struct {
	Max int64 `json:"max"`
}

// RandomNumber generates a random number up to the provided maximum.
func RandomNumber(args RandomNumberArgs) int64 {
	return rand.Int63n(args.Max + 1)
}

type MultiplyByTwoArgs struct {
	X int64 `json:"x"`
}

// MultiplyByTwo calculates a simple multiplication by two.
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
	RandomNumberTool = tools.Function{
		Name:        "random_number",
		Description: "Generate a random number up to the provided maximum.",
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
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			var args RandomNumberArgs
			err := json.Unmarshal([]byte(arguments), &args)
			if err != nil {
				return nil, err
			}
			return RandomNumber(args), nil
		},
	}

	MultiplyByTwoTool = tools.Function{
		Name:        "multiply_by_two",
		Description: "Simple multiplication by two.",
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
		OnInvokeTool: func(_ context.Context, arguments string) (any, error) {
			var args MultiplyByTwoArgs
			err := json.Unmarshal([]byte(arguments), &args)
			if err != nil {
				return nil, err
			}
			return MultiplyByTwo(args), nil
		},
	}

	MultiplyAgent = &agents.Agent{
		Name:         "Multiply Agent",
		Instructions: agents.InstructionsStr("Multiply the number by 2 and then return the final result."),
		Tools:        []tools.Tool{MultiplyByTwoTool},
		OutputSchema: FinalResultOutputSchema{},
		Hooks:        NewCustomAgentHooks("Multiply Agent"),
		Model:        param.NewOpt(agents.NewAgentModelName("gpt-4o-mini")),
	}

	StartAgent = &agents.Agent{
		Name: "Start Agent",
		Instructions: agents.InstructionsStr(
			"Generate a random number. If it's even, stop. If it's odd, hand off to the multiply agent.",
		),
		Tools:         []tools.Tool{RandomNumberTool},
		OutputSchema:  FinalResultOutputSchema{},
		AgentHandoffs: []*agents.Agent{MultiplyAgent},
		Hooks:         NewCustomAgentHooks("Start Agent"),
		Model:         param.NewOpt(agents.NewAgentModelName("gpt-4o-mini")),
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
		Input: agents.InputString(
			fmt.Sprintf("Generate a random number between 0 and %s.", userInput),
		),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("Done!")
}
