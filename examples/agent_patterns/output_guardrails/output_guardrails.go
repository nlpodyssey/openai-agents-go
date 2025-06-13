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
	"errors"
	"fmt"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

/*
This example shows how to use output guardrails.

Output guardrails are checks that run on the final output of an agent.
They can be used to do things like:
- Check if the output contains sensitive data
- Check if the output is a valid response to the user's message

In this example, we'll use a (contrived) example where we check if the
agent's response contains a phone number.
*/

// MessageOutput is the agent's output type.
type MessageOutput struct {
	// Thoughts on how to respond to the user's message
	Reasoning string `json:"reasoning"`

	// The response to the user's message
	Response string `json:"response"`

	// The name of the user who sent the message, if known
	UserName *string `json:"user_name"`
}

type MessageOutputSchema struct{}

func (MessageOutputSchema) Name() string             { return "MessageOutput" }
func (MessageOutputSchema) IsPlainText() bool        { return false }
func (MessageOutputSchema) IsStrictJSONSchema() bool { return true }
func (MessageOutputSchema) JSONSchema() map[string]any {
	type m = map[string]any
	return m{
		"title":                "MessageOutput",
		"type":                 "object",
		"additionalProperties": false,
		"properties": m{
			"reasoning": m{
				"title":       "Reasoning",
				"description": "Thoughts on how to respond to the user's message",
				"type":        "string",
			},
			"response": m{
				"title":       "Response",
				"description": "The response to the user's message",
				"type":        "string",
			},
			"user_name": m{
				"title":       "User Name",
				"description": "The name of the user who sent the message, if known",
				"anyOf":       []any{m{"type": "string"}, m{"type": "null"}},
			},
		},
		"required": []string{"reasoning", "response", "user_name"},
	}
}
func (MessageOutputSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()

	var v MessageOutput
	err := dec.Decode(&v)
	return v, err
}

type SensitiveDataCheckInfo struct {
	PhoneNumberInResponse  bool
	PhoneNumberInReasoning bool
}

func SensitiveDataCheckFunction(
	_ context.Context,
	_ *agents.Agent,
	anyOutput any,
) (agents.GuardrailFunctionOutput, error) {
	output := anyOutput.(MessageOutput)
	phoneNumberInResponse := strings.Contains(output.Response, "650")
	phoneNumberInReasoning := strings.Contains(output.Reasoning, "650")

	return agents.GuardrailFunctionOutput{
		OutputInfo: SensitiveDataCheckInfo{
			PhoneNumberInResponse:  phoneNumberInResponse,
			PhoneNumberInReasoning: phoneNumberInReasoning,
		},
		TripwireTriggered: phoneNumberInResponse || phoneNumberInReasoning,
	}, nil
}

var SensitiveDataCheck = agents.OutputGuardrail{
	GuardrailFunction: SensitiveDataCheckFunction,
	Name:              "sensitive_data_check",
}

var Agent = agents.New("Assistant").
	WithInstructions("You are a helpful assistant.").
	WithOutputSchema(MessageOutputSchema{}).
	WithOutputGuardrails([]agents.OutputGuardrail{SensitiveDataCheck}).
	WithModel("gpt-4.1-nano")

func main() {
	// This should be ok
	_, err := agents.Runner{}.Run(context.Background(), Agent, agents.InputString("What's the capital of California?"))
	if err != nil {
		panic(err)
	}
	fmt.Println("First message passed")

	// This should trip the guardrail
	result, err := agents.Runner{}.Run(
		context.Background(), Agent,
		agents.InputString(
			"My phone number is 650-123-4567. Where do you think I live?",
		),
	)
	if err == nil {
		fmt.Printf("Guardrail didn't trip - this is unexpected. Output: %#v\n", result.FinalOutput)
		return
	}
	var tripwireError agents.OutputGuardrailTripwireTriggeredError
	if !errors.As(err, &tripwireError) {
		panic(err)
	}
	fmt.Printf("Guardrail tripped. Info: %#v\n", tripwireError.GuardrailResult.Output.OutputInfo)
}
