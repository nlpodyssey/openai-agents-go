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
	"maps"
	"slices"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

/*
This example demonstrates how to use an output type that is not in strict mode. Strict mode
allows us to guarantee valid JSON output, but some schemas are not strict-compatible.

In this example, we define an output type that is not strict-compatible, and then we run the
agent with StrictJSONSchema=False.

We also demonstrate a custom output type.

To understand which schemas are strict-compatible, see:
https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#supported-schemas
*/

type OutputType struct {
	// A list of jokes, indexed by joke number.
	Jokes map[int]string `json:"jokes"`
}

type OutputTypeSchema struct {
	isStrictJSONSchema bool
}

func (OutputTypeSchema) Name() string               { return "OutputType" }
func (OutputTypeSchema) IsPlainText() bool          { return false }
func (s OutputTypeSchema) IsStrictJSONSchema() bool { return s.isStrictJSONSchema }
func (OutputTypeSchema) JSONSchema() map[string]any {
	type m = map[string]any
	return m{
		"title":    "OutputType",
		"type":     "object",
		"required": []string{"jokes"},
		"properties": m{
			"jokes": m{
				"title": "Jokes",
				"type":  "object",
				"additionalProperties": m{
					"type": "string",
				},
			},
		},
	}
}
func (OutputTypeSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()

	var v OutputType
	err := dec.Decode(&v)
	return v, err
}

// CustomOutputSchema is a demonstration of a custom output schema.
type CustomOutputSchema struct{}

type CustomOutputType struct {
	Jokes map[string]string `json:"jokes"`
}

func (CustomOutputSchema) Name() string             { return "CustomOutputType" }
func (CustomOutputSchema) IsPlainText() bool        { return false }
func (CustomOutputSchema) IsStrictJSONSchema() bool { return false }
func (CustomOutputSchema) JSONSchema() map[string]any {
	type m = map[string]any
	return m{
		"title": "CustomOutputType",
		"type":  "object",
		"properties": m{
			"jokes": m{
				"type": "object",
				"properties": m{
					"joke": m{
						"type": "string",
					},
				},
			},
		},
	}
}
func (CustomOutputSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()

	var v CustomOutputType
	err := dec.Decode(&v)
	if err != nil {
		return nil, err
	}

	// Just for demonstration, we'll return a list.
	return slices.Collect(maps.Values(v.Jokes)), nil
}

func main() {
	ctx := context.Background()

	agent := agents.New("Assistant").
		WithInstructions("You are a helpful assistant.").
		WithModel("gpt-4o")

	input := "Tell me 3 short jokes."

	// First, let's try with a strict output type. This should raise an exception.
	agent.OutputSchema = OutputTypeSchema{isStrictJSONSchema: true}
	_, err := agents.Run(ctx, agent, agents.InputString(input))
	if err == nil {
		panic("Should have raised an exception")
	}
	fmt.Printf("Error (always expected): %s\n", err)

	// Now let's try again with a non-strict output type. This should work.
	// In some cases, it will raise an error - the schema isn't strict, so the model may
	// produce an invalid JSON object.
	agent.OutputSchema = OutputTypeSchema{isStrictJSONSchema: false}
	result, err := agents.Run(ctx, agent, agents.InputString(input))
	if err != nil {
		fmt.Printf("Error (expected occasionally, try again and you might get a good result): %s\n", err)
	} else {
		fmt.Printf("%T: %+v\n", result.FinalOutput, result.FinalOutput)
	}

	// Finally, let's try a custom output type.
	agent.OutputSchema = CustomOutputSchema{}
	result, err = agents.Run(ctx, agent, agents.InputString(input))
	if err != nil {
		fmt.Printf("Error (expected occasionally, try again and you might get a good result): %s\n", err)
	} else {
		fmt.Printf("%T: %+v\n", result.FinalOutput, result.FinalOutput)
	}
}
