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
	"errors"
	"fmt"
)

// MaxTurnsExceededError is returned when the maximum number of turns is exceeded.
type MaxTurnsExceededError error

func NewMaxTurnsExceededError(message string) MaxTurnsExceededError {
	return MaxTurnsExceededError(errors.New(message))
}

func MaxTurnsExceededErrorf(format string, a ...any) MaxTurnsExceededError {
	return MaxTurnsExceededError(fmt.Errorf(format, a...))
}

// ModelBehaviorError is returned when the model does something unexpected,
// e.g. calling a tool that doesn't exist, or providing malformed JSON.
type ModelBehaviorError error

func NewModelBehaviorError(message string) ModelBehaviorError {
	return ModelBehaviorError(errors.New(message))
}

func ModelBehaviorErrorf(format string, a ...any) ModelBehaviorError {
	return ModelBehaviorError(fmt.Errorf(format, a...))
}

// UserError is returned when the user makes an error using the SDK.
type UserError error

func NewUserError(message string) UserError {
	return UserError(errors.New(message))
}

func UserErrorf(format string, a ...any) UserError {
	return UserError(fmt.Errorf(format, a...))
}

// InputGuardrailTripwireTriggeredError is returned when a guardrail tripwire is triggered.
type InputGuardrailTripwireTriggeredError struct {
	// The result data of the guardrail that was triggered.
	GuardrailResult InputGuardrailResult
}

func (err InputGuardrailTripwireTriggeredError) Error() string {
	return fmt.Sprintf("input guardrail %s triggered tripwire", err.GuardrailResult.Guardrail.Name)
}

func NewInputGuardrailTripwireTriggeredError(guardrailResult InputGuardrailResult) InputGuardrailTripwireTriggeredError {
	return InputGuardrailTripwireTriggeredError{
		GuardrailResult: guardrailResult,
	}
}

// OutputGuardrailTripwireTriggeredError is returned when a guardrail tripwire is triggered.
type OutputGuardrailTripwireTriggeredError struct {
	GuardrailName string

	// The result data of the guardrail that was triggered.
	GuardrailResult OutputGuardrailResult
}

func (err OutputGuardrailTripwireTriggeredError) Error() string {
	return fmt.Sprintf("output guardrail %s triggered tripwire", err.GuardrailName)
}

func NewOutputGuardrailTripwireTriggeredError(guardrailName string, guardrailResult OutputGuardrailResult) OutputGuardrailTripwireTriggeredError {
	return OutputGuardrailTripwireTriggeredError{
		GuardrailName:   guardrailName,
		GuardrailResult: guardrailResult,
	}
}

type CanceledError error

func NewCanceledError(message string) CanceledError {
	return CanceledError(errors.New(message))
}

func CanceledErrorf(format string, a ...any) CanceledError {
	return CanceledError(fmt.Errorf(format, a...))
}
