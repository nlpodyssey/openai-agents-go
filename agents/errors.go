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
	"errors"
	"fmt"
)

// RunErrorDetails provides data collected from an agent run when an error occurs.
type RunErrorDetails struct {
	Context                context.Context
	Input                  Input
	NewItems               []RunItem
	RawResponses           []ModelResponse
	LastAgent              *Agent
	InputGuardrailResults  []InputGuardrailResult
	OutputGuardrailResults []OutputGuardrailResult
}

func (d RunErrorDetails) String() string {
	return PrettyPrintRunErrorDetails(d)
}

// AgentsError is the base object wrapped by all other errors in the Agents SDK.
type AgentsError struct {
	Err     error
	RunData *RunErrorDetails
}

func (err *AgentsError) Error() string {
	if err.Err == nil {
		return "AgentsError"
	}
	return err.Err.Error()
}

func (err *AgentsError) Unwrap() error {
	return err.Err
}

func NewAgentsError(message string) *AgentsError {
	return &AgentsError{Err: errors.New(message)}
}

func AgentsErrorf(format string, a ...any) *AgentsError {
	return &AgentsError{Err: fmt.Errorf(format, a...)}
}

// MaxTurnsExceededError is returned when the maximum number of turns is exceeded.
type MaxTurnsExceededError struct {
	*AgentsError
}

func (err MaxTurnsExceededError) Error() string {
	if err.AgentsError == nil {
		return "MaxTurnsExceededError"
	}
	return err.AgentsError.Error()
}

func (err MaxTurnsExceededError) Unwrap() error {
	return err.AgentsError
}

func NewMaxTurnsExceededError(message string) MaxTurnsExceededError {
	return MaxTurnsExceededError{AgentsError: NewAgentsError(message)}
}

func MaxTurnsExceededErrorf(format string, a ...any) MaxTurnsExceededError {
	return MaxTurnsExceededError{AgentsError: AgentsErrorf(format, a...)}
}

// ModelBehaviorError is returned when the model does something unexpected,
// e.g. calling a tool that doesn't exist, or providing malformed JSON.
type ModelBehaviorError struct {
	*AgentsError
}

func (err ModelBehaviorError) Error() string {
	if err.AgentsError == nil {
		return "ModelBehaviorError"
	}
	return err.AgentsError.Error()
}

func (err ModelBehaviorError) Unwrap() error {
	return err.AgentsError
}

func NewModelBehaviorError(message string) ModelBehaviorError {
	return ModelBehaviorError{AgentsError: NewAgentsError(message)}
}

func ModelBehaviorErrorf(format string, a ...any) ModelBehaviorError {
	return ModelBehaviorError{AgentsError: AgentsErrorf(format, a...)}
}

// UserError is returned when the user makes an error using the SDK.
type UserError struct {
	*AgentsError
}

func (err UserError) Error() string {
	if err.AgentsError == nil {
		return "UserError"
	}
	return err.AgentsError.Error()
}

func (err UserError) Unwrap() error {
	return err.AgentsError
}

func NewUserError(message string) UserError {
	return UserError{AgentsError: NewAgentsError(message)}
}

func UserErrorf(format string, a ...any) UserError {
	return UserError{AgentsError: AgentsErrorf(format, a...)}
}

// InputGuardrailTripwireTriggeredError is returned when an input guardrail tripwire is triggered.
type InputGuardrailTripwireTriggeredError struct {
	*AgentsError
	// The result data of the guardrail that was triggered.
	GuardrailResult InputGuardrailResult
}

func (err InputGuardrailTripwireTriggeredError) Error() string {
	if err.AgentsError == nil {
		return "InputGuardrailTripwireTriggeredError"
	}
	return err.AgentsError.Error()
}

func (err InputGuardrailTripwireTriggeredError) Unwrap() error {
	return err.AgentsError
}

func NewInputGuardrailTripwireTriggeredError(guardrailResult InputGuardrailResult) InputGuardrailTripwireTriggeredError {
	return InputGuardrailTripwireTriggeredError{
		AgentsError:     AgentsErrorf("input guardrail %s triggered tripwire", guardrailResult.Guardrail.Name),
		GuardrailResult: guardrailResult,
	}
}

// OutputGuardrailTripwireTriggeredError is returned when an output guardrail tripwire is triggered.
type OutputGuardrailTripwireTriggeredError struct {
	*AgentsError
	// The result data of the guardrail that was triggered.
	GuardrailResult OutputGuardrailResult
}

func (err OutputGuardrailTripwireTriggeredError) Error() string {
	if err.AgentsError == nil {
		return "OutputGuardrailTripwireTriggeredError"
	}
	return err.AgentsError.Error()
}

func (err OutputGuardrailTripwireTriggeredError) Unwrap() error {
	return err.AgentsError
}

func NewOutputGuardrailTripwireTriggeredError(guardrailResult OutputGuardrailResult) OutputGuardrailTripwireTriggeredError {
	return OutputGuardrailTripwireTriggeredError{
		AgentsError:     AgentsErrorf("output guardrail %s triggered tripwire", guardrailResult.Guardrail.Name),
		GuardrailResult: guardrailResult,
	}
}

// TaskCanceledError is returned when a task has been canceled.
type TaskCanceledError struct {
	*AgentsError
}

func (err TaskCanceledError) Error() string {
	if err.AgentsError == nil {
		return "TaskCanceledError"
	}
	return err.AgentsError.Error()
}

func (err TaskCanceledError) Unwrap() error {
	return err.AgentsError
}

func NewTaskCanceledError(message string) TaskCanceledError {
	return TaskCanceledError{AgentsError: NewAgentsError(message)}
}

func TaskCanceledErrorf(format string, a ...any) TaskCanceledError {
	return TaskCanceledError{AgentsError: AgentsErrorf(format, a...)}
}
