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
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestShouldResetToolChoiceDirect(t *testing.T) {
	// Test various inputs to ensure it correctly identifies cases where reset is needed.

	agent := &agents.Agent{Name: "test_agent"}

	t.Run(`empty tool use tracker should not change the nil tool choice`, func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{ToolChoice: nil}
		tracker := agents.NewAgentToolUseTracker()
		newSettings := agents.RunImpl().MaybeResetToolChoice(agent, tracker, modelSettings)
		assert.Nil(t, newSettings.ToolChoice)
	})

	t.Run(`empty tool use tracker should not change the "none" tool choice`, func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceNone}
		tracker := agents.NewAgentToolUseTracker()
		newSettings := agents.RunImpl().MaybeResetToolChoice(agent, tracker, modelSettings)
		assert.Equal(t, modelsettings.ToolChoiceNone, newSettings.ToolChoice)
	})

	t.Run(`empty tool use tracker should not change the "auto" tool choice`, func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceAuto}
		tracker := agents.NewAgentToolUseTracker()
		newSettings := agents.RunImpl().MaybeResetToolChoice(agent, tracker, modelSettings)
		assert.Equal(t, modelsettings.ToolChoiceAuto, newSettings.ToolChoice)
	})

	t.Run(`empty tool use tracker should not change the "required" tool choice`, func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceRequired}
		tracker := agents.NewAgentToolUseTracker()
		newSettings := agents.RunImpl().MaybeResetToolChoice(agent, tracker, modelSettings)
		assert.Equal(t, modelsettings.ToolChoiceRequired, newSettings.ToolChoice)
	})

	t.Run(`ToolChoice = "required" with one tool should reset`, func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceRequired}
		tracker := agents.NewAgentToolUseTracker()
		tracker.AddToolUse(agent, []string{"tool1"})
		newSettings := agents.RunImpl().MaybeResetToolChoice(agent, tracker, modelSettings)
		assert.Nil(t, newSettings.ToolChoice)
	})

	t.Run(`ToolChoice = "required" with multiple tools should reset`, func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceRequired}
		tracker := agents.NewAgentToolUseTracker()
		tracker.AddToolUse(agent, []string{"tool1", "tool2"})
		newSettings := agents.RunImpl().MaybeResetToolChoice(agent, tracker, modelSettings)
		assert.Nil(t, newSettings.ToolChoice)
	})

	t.Run(`tool usage on a different agent should not affect the tool choice`, func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceString("foo_bar")}
		tracker := agents.NewAgentToolUseTracker()
		tracker.AddToolUse(&agents.Agent{Name: "other_agent"}, []string{"foo_bar", "baz"})
		newSettings := agents.RunImpl().MaybeResetToolChoice(agent, tracker, modelSettings)
		assert.Equal(t, modelsettings.ToolChoiceString("foo_bar"), newSettings.ToolChoice)
	})

	t.Run(`ToolChoice = "foo_bar" with multiple tools should reset`, func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{ToolChoice: modelsettings.ToolChoiceString("foo_bar")}
		tracker := agents.NewAgentToolUseTracker()
		tracker.AddToolUse(agent, []string{"foo_bar", "baz"})
		newSettings := agents.RunImpl().MaybeResetToolChoice(agent, tracker, modelSettings)
		assert.Nil(t, newSettings.ToolChoice)
	})
}

func TestRequiredToolChoiceWithMultipleRuns(t *testing.T) {
	// Test scenario 1: When multiple runs are executed with ToolChoice="required", ensure each
	// run works correctly and doesn't get stuck in an infinite loop. Also verify that ToolChoice
	// remains "required" between runs.

	// Set up our fake model with responses for two runs
	fakeModel := agentstesting.NewFakeModel(false, nil)
	fakeModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("First run response")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Second run response")}},
	})

	// Create agent with a custom tool and ToolChoice="required"
	customTool := agentstesting.GetFunctionTool("custom_tool", "tool_result")
	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(fakeModel)),
		Tools: []agents.Tool{customTool},
		ModelSettings: modelsettings.ModelSettings{
			ToolChoice: modelsettings.ToolChoiceRequired,
		},
	}

	// First run should work correctly and preserve ToolChoice
	result1, err := agents.Runner{}.Run(t.Context(), agent, "first run")
	require.NoError(t, err)
	assert.Equal(t, "First run response", result1.FinalOutput)
	assert.Equal(t, modelsettings.ToolChoiceRequired, fakeModel.LastTurnArgs.ModelSettings.ToolChoice)

	// Second run should also work correctly with ToolChoice still required
	result2, err := agents.Runner{}.Run(t.Context(), agent, "second run")
	require.NoError(t, err)
	assert.Equal(t, "Second run response", result2.FinalOutput)
	assert.Equal(t, modelsettings.ToolChoiceRequired, fakeModel.LastTurnArgs.ModelSettings.ToolChoice)
}

func TestRequiredWithStopAtToolName(t *testing.T) {
	// Test scenario 2: When using required ToolChoice with StopAtToolNames behavior, ensure
	// it correctly stops at the specified tool.

	// Set up fake model to return a tool call for second_tool
	fakeModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("second_tool", "{}"),
		},
	})

	// Create agent with two tools and ToolChoice="required" and StopAtTool behavior
	firstTool := agentstesting.GetFunctionTool("first_tool", "first tool result")
	secondTool := agentstesting.GetFunctionTool("second_tool", "second tool result")

	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(fakeModel)),
		Tools: []agents.Tool{firstTool, secondTool},
		ModelSettings: modelsettings.ModelSettings{
			ToolChoice: modelsettings.ToolChoiceRequired,
		},
		ToolUseBehavior: agents.StopAtTools("second_tool"),
	}

	// Run should stop after using second_tool
	result, err := agents.Runner{}.Run(t.Context(), agent, "run test")
	require.NoError(t, err)
	assert.Equal(t, "second tool result", result.FinalOutput)
}

func TestSpecificToolChoice(t *testing.T) {
	// Test scenario 3: When using a specific tool choice name, ensure it doesn't cause infinite
	// loops.

	// Set up fake model to return a text message
	fakeModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Test message"),
		},
	})

	// Create agent with specific ToolChoice
	tool1 := agentstesting.GetFunctionTool("tool1", "result1")
	tool2 := agentstesting.GetFunctionTool("tool2", "result2")
	tool3 := agentstesting.GetFunctionTool("tool3", "result3")

	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(fakeModel)),
		Tools: []agents.Tool{tool1, tool2, tool3},
		ModelSettings: modelsettings.ModelSettings{
			ToolChoice: modelsettings.ToolChoiceString("tool1"), // Specific tool
		},
	}

	// Run should complete without infinite loops
	result, err := agents.Runner{}.Run(t.Context(), agent, "run test")
	require.NoError(t, err)
	assert.Equal(t, "Test message", result.FinalOutput)
}

func TestRequiredWithSingleTool(t *testing.T) {
	// Test scenario 4: When using required ToolChoice with only one tool, ensure it doesn't cause
	// infinite loops.

	// Set up fake model to return a tool call followed by a text message
	fakeModel := agentstesting.NewFakeModel(false, nil)
	fakeModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First call returns a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("custom_tool", "{}"),
		}},
		// Second call returns a text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Final response"),
		}},
	})

	// Create agent with a single tool and ToolChoice="required"
	customTool := agentstesting.GetFunctionTool("custom_tool", "tool result")
	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(fakeModel)),
		Tools: []agents.Tool{customTool},
		ModelSettings: modelsettings.ModelSettings{
			ToolChoice: modelsettings.ToolChoiceRequired,
		},
	}

	// Run should complete without infinite loops
	result, err := agents.Runner{}.Run(t.Context(), agent, "run test")
	require.NoError(t, err)
	assert.Equal(t, "Final response", result.FinalOutput)
}

func TestDontResetToolChoiceIfNotRequired(t *testing.T) {
	// Test scenario 5: When the agent's ResetToolChoice is false, ensure ToolChoice is not reset.

	// Set up fake model to return a tool call followed by a text message
	fakeModel := agentstesting.NewFakeModel(false, nil)
	fakeModel.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First call returns a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("custom_tool", "{}"),
		}},
		// Second call returns a text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Final response"),
		}},
	})

	// Create agent with a single tool and ToolChoice="required" and ResetToolChoice=False
	customTool := agentstesting.GetFunctionTool("custom_tool", "tool result")
	agent := &agents.Agent{
		Name:  "test_agent",
		Model: param.NewOpt(agents.NewAgentModel(fakeModel)),
		Tools: []agents.Tool{customTool},
		ModelSettings: modelsettings.ModelSettings{
			ToolChoice: modelsettings.ToolChoiceRequired,
		},
		ResetToolChoice: param.NewOpt(false),
	}

	_, err := agents.Runner{}.Run(t.Context(), agent, "run test")
	require.NoError(t, err)
	assert.Equal(t, modelsettings.ToolChoiceRequired, fakeModel.LastTurnArgs.ModelSettings.ToolChoice)
}
