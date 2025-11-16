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
	"errors"
	"fmt"
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

/*
This example shows how to use guardrails.

Guardrails are checks that run in parallel to the agent's execution.
They can be used to do things like:
- Check if input messages are off-topic
- Check that input messages don't violate any policies
- Take over control of the agent's execution if an unexpected input is detected

In this example, we'll set up an input guardrail that trips if the user is
asking to do math homework.
If the guardrail trips, we'll respond with a refusal message.
*/

// 1. An agent-based guardrail that is triggered if the user is asking to do math homework

type MathHomeworkOutput struct {
	Reasoning      string `json:"reasoning"`
	IsMathHomework bool   `json:"is_math_homework"`
}

var GuardrailAgent = agents.New("Guardrail check").
	WithInstructions("Check if the user is asking you to do their math homework.").
	WithOutputType(agents.OutputType[MathHomeworkOutput]()).
	WithModel("gpt-4.1-nano")

// MathGuardrailFunction is an input guardrail function, which happens to call
// an agent to check if the input is a math homework question.
func MathGuardrailFunction(
	ctx context.Context,
	_ *agents.Agent,
	input agents.Input,
) (agents.GuardrailFunctionOutput, error) {
	var (
		result *agents.RunResult
		err    error
	)
	switch v := input.(type) {
	case agents.InputString:
		result, err = agents.Run(ctx, GuardrailAgent, v.String())
	case agents.InputItems:
		result, err = agents.RunInputs(ctx, GuardrailAgent, v)
	default:
		panic(fmt.Errorf("unexpected input type %T", v))
	}
	if err != nil {
		return agents.GuardrailFunctionOutput{}, err
	}
	finalOutput := result.FinalOutput.(MathHomeworkOutput)

	return agents.GuardrailFunctionOutput{
		OutputInfo:        finalOutput,
		TripwireTriggered: finalOutput.IsMathHomework,
	}, nil
}

var MathGuardrail = agents.InputGuardrail{
	GuardrailFunction: MathGuardrailFunction,
	Name:              "math_guardrail",
}

// 2. The run loop

func main() {
	agent := agents.New("Customer support agent").
		WithInstructions("You are a customer support agent. You help customers with their questions.").
		WithInputGuardrails([]agents.InputGuardrail{MathGuardrail}).
		WithModel("gpt-4.1-nano")

	var inputData []agents.TResponseInputItem

	for {
		fmt.Print("Enter a message: ")
		_ = os.Stdout.Sync()
		line, _, err := bufio.NewReader(os.Stdin).ReadLine()
		if err != nil {
			panic(err)
		}
		userInput := string(line)
		inputData = append(inputData, agents.TResponseInputItem{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt(userInput),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		})

		result, err := agents.RunInputs(context.Background(), agent, inputData)
		if err != nil {
			var tripwireError agents.InputGuardrailTripwireTriggeredError
			if errors.As(err, &tripwireError) {
				// If the guardrail triggered, we add a refusal message to the input
				message := "Sorry, I can't help you with your math homework."
				fmt.Println(message)
				inputData = append(inputData, agents.TResponseInputItem{
					OfMessage: &responses.EasyInputMessageParam{
						Content: responses.EasyInputMessageContentUnionParam{
							OfString: param.NewOpt(message),
						},
						Role: responses.EasyInputMessageRoleAssistant,
						Type: responses.EasyInputMessageTypeMessage,
					},
				})
				continue
			} else {
				panic(err)
			}
		}

		fmt.Println(result.FinalOutput)

		// If the guardrail didn't trigger, we use the result as the input for the next run
		inputData = result.ToInputList()
	}
}
