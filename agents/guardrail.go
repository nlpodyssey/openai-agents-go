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

package agents

import (
	"context"
)

// An InputGuardrail is a check that runs in parallel to the agent's execution.
//
// Input guardrails can be used to do things like:
//   - Check if input messages are off-topic
//   - Take over control of the agent's execution if an unexpected input is detected
//
// Guardrails return an InputGuardrailResult. If GuardrailFunctionOutput.TripwireTriggered is true,
// the agent's execution will immediately stop, and an InputGuardrailTripwireTriggeredError will be returned.
type InputGuardrail struct {
	// A function that receives the agent input and the context, and returns a
	// GuardrailFunctionOutput. The result marks whether the tripwire was
	// triggered, and can optionally include information about the guardrail's output.
	GuardrailFunction InputGuardrailFunction

	// The name of the guardrail, used for tracing.
	Name string
}

type InputGuardrailFunction = func(context.Context, *Agent, Input) (GuardrailFunctionOutput, error)

func (ig InputGuardrail) Run(ctx context.Context, agent *Agent, input Input) (InputGuardrailResult, error) {
	output, err := ig.GuardrailFunction(ctx, agent, input)
	result := InputGuardrailResult{
		Guardrail: ig,
		Output:    output,
	}
	return result, err
}

// InputGuardrailResult is the result of a guardrail run.
type InputGuardrailResult struct {
	// The guardrail that was run.
	Guardrail InputGuardrail

	// The output of the guardrail function.
	Output GuardrailFunctionOutput
}

// GuardrailFunctionOutput is the output of a guardrail function.
type GuardrailFunctionOutput struct {
	// Optional information about the guardrail's output. For example, the guardrail could include
	// information about the checks it performed and granular results.
	OutputInfo any

	// Whether the tripwire was triggered. If triggered, the agent's execution will be halted.
	TripwireTriggered bool
}

// An OutputGuardrail is a check that runs on the final output of an agent.
// Output guardrails can be used to do check if the output passes certain validation criteria.
//
// Guardrails return an OutputGuardrailResult. If GuardrailFunctionOutput.TripwireTriggered is true,
// an OutputGuardrailTripwireTriggeredError will be returned.
type OutputGuardrail struct {
	// A function that receives the final agent, its output, and the context, and returns a
	// GuardrailFunctionOutput. The result marks whether the tripwire was triggered, and can optionally
	// include information about the guardrail's output.
	GuardrailFunction OutputGuardrailFunction

	// The name of the guardrail, used for tracing.
	Name string
}

type OutputGuardrailFunction = func(ctx context.Context, agent *Agent, agentOutput any) (GuardrailFunctionOutput, error)

func (og OutputGuardrail) Run(ctx context.Context, agent *Agent, agentOutput any) (OutputGuardrailResult, error) {
	output, err := og.GuardrailFunction(ctx, agent, agentOutput)
	result := OutputGuardrailResult{
		Guardrail:   og,
		Agent:       agent,
		AgentOutput: agentOutput,
		Output:      output,
	}
	return result, err
}

// OutputGuardrailResult is the result of a guardrail run.
type OutputGuardrailResult struct {
	// The guardrail that was run.
	Guardrail OutputGuardrail

	// The output of the agent that was checked by the guardrail.
	AgentOutput any

	// The agent that was checked by the guardrail.
	Agent *Agent

	// The output of the guardrail function.
	Output GuardrailFunctionOutput
}
