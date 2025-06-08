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

package agents_test

import (
	"context"
	"errors"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInputGuardrail(t *testing.T) {
	getGuardrail := func(triggers bool, outputInfo any, err error) agents.InputGuardrailFunction {
		return func(context.Context, *runcontext.Wrapper, *agents.Agent, agents.Input) (agents.GuardrailFunctionOutput, error) {
			return agents.GuardrailFunctionOutput{
				OutputInfo:        outputInfo,
				TripwireTriggered: triggers,
			}, err
		}
	}

	t.Run("tripwire not triggered, nil output info", func(t *testing.T) {
		guardrail := agents.InputGuardrail{
			GuardrailFunction: getGuardrail(false, nil, nil),
			Name:              "guardrail_function",
		}
		result, err := guardrail.Run(
			t.Context(),
			&agents.Agent{Name: "test"},
			agents.InputString("test"),
			runcontext.NewWrapper(nil),
		)
		require.NoError(t, err)
		assert.False(t, result.Output.TripwireTriggered)
		assert.Nil(t, result.Output.OutputInfo)
	})

	t.Run("tripwire triggered, nil output info", func(t *testing.T) {
		guardrail := agents.InputGuardrail{
			GuardrailFunction: getGuardrail(true, nil, nil),
			Name:              "guardrail_function",
		}
		result, err := guardrail.Run(
			t.Context(),
			&agents.Agent{Name: "test"},
			agents.InputString("test"),
			runcontext.NewWrapper(nil),
		)
		require.NoError(t, err)
		assert.True(t, result.Output.TripwireTriggered)
		assert.Nil(t, result.Output.OutputInfo)
	})

	t.Run("tripwire triggered, some output info", func(t *testing.T) {
		guardrail := agents.InputGuardrail{
			GuardrailFunction: getGuardrail(true, "test", nil),
			Name:              "guardrail_function",
		}
		result, err := guardrail.Run(
			t.Context(),
			&agents.Agent{Name: "test"},
			agents.InputString("test"),
			runcontext.NewWrapper(nil),
		)
		require.NoError(t, err)
		assert.True(t, result.Output.TripwireTriggered)
		assert.Equal(t, "test", result.Output.OutputInfo)
	})

	t.Run("error", func(t *testing.T) {
		e := errors.New("error")
		guardrail := agents.InputGuardrail{
			GuardrailFunction: getGuardrail(false, "test", e),
			Name:              "guardrail_function",
		}
		_, err := guardrail.Run(
			t.Context(),
			&agents.Agent{Name: "test"},
			agents.InputString("test"),
			runcontext.NewWrapper(nil),
		)
		require.ErrorIs(t, err, e)
	})
}

func TestOutputGuardrail(t *testing.T) {
	getGuardrail := func(triggers bool, outputInfo any, err error) agents.OutputGuardrailFunction {
		return func(context.Context, *runcontext.Wrapper, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
			return agents.GuardrailFunctionOutput{
				OutputInfo:        outputInfo,
				TripwireTriggered: triggers,
			}, err
		}
	}

	t.Run("tripwire not triggered, nil output info", func(t *testing.T) {
		guardrail := agents.OutputGuardrail{
			GuardrailFunction: getGuardrail(false, nil, nil),
			Name:              "guardrail_function",
		}
		result, err := guardrail.Run(
			t.Context(),
			runcontext.NewWrapper(nil),
			&agents.Agent{Name: "test"},
			agents.InputString("test"),
		)
		require.NoError(t, err)
		assert.False(t, result.Output.TripwireTriggered)
		assert.Nil(t, result.Output.OutputInfo)
	})

	t.Run("tripwire triggered, nil output info", func(t *testing.T) {
		guardrail := agents.OutputGuardrail{
			GuardrailFunction: getGuardrail(true, nil, nil),
			Name:              "guardrail_function",
		}
		result, err := guardrail.Run(
			t.Context(),
			runcontext.NewWrapper(nil),
			&agents.Agent{Name: "test"},
			agents.InputString("test"),
		)
		require.NoError(t, err)
		assert.True(t, result.Output.TripwireTriggered)
		assert.Nil(t, result.Output.OutputInfo)
	})

	t.Run("tripwire triggered, some output info", func(t *testing.T) {
		guardrail := agents.OutputGuardrail{
			GuardrailFunction: getGuardrail(true, "test", nil),
			Name:              "guardrail_function",
		}
		result, err := guardrail.Run(
			t.Context(),
			runcontext.NewWrapper(nil),
			&agents.Agent{Name: "test"},
			agents.InputString("test"),
		)
		require.NoError(t, err)
		assert.True(t, result.Output.TripwireTriggered)
		assert.Equal(t, "test", result.Output.OutputInfo)
	})

	t.Run("error", func(t *testing.T) {
		e := errors.New("error")
		guardrail := agents.OutputGuardrail{
			GuardrailFunction: getGuardrail(false, "test", e),
			Name:              "guardrail_function",
		}
		_, err := guardrail.Run(
			t.Context(),
			runcontext.NewWrapper(nil),
			&agents.Agent{Name: "test"},
			agents.InputString("test"),
		)
		require.ErrorIs(t, err, e)
	})
}
