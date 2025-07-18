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

package agentstesting_test

import (
	"fmt"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFakeMCPPromptServer(t *testing.T) {
	t.Run("ListPrompts", func(t *testing.T) {
		// Test listing available prompts
		server := agentstesting.NewFakeMCPPromptServer("")
		server.AddPrompt(
			"generate_code_review_instructions",
			"Generate agent instructions for code review tasks",
			nil,
		)
		result, err := server.ListPrompts(t.Context())
		require.NoError(t, err)
		require.Len(t, result.Prompts, 1)
		assert.Equal(t, "generate_code_review_instructions", result.Prompts[0].Name)
		assert.Contains(t, result.Prompts[0].Description, "code review")
	})

	t.Run("GetPrompt without arguments", func(t *testing.T) {
		// Test getting a prompt without arguments
		server := agentstesting.NewFakeMCPPromptServer("")
		server.AddPrompt("simple_prompt", "A simple prompt", nil)
		server.SetPromptResult("simple_prompt", "You are a helpful assistant.")

		result, err := server.GetPrompt(t.Context(), "simple_prompt", nil)
		require.NoError(t, err)
		require.Len(t, result.Messages, 1)
		assert.Equal(t, &mcp.TextContent{Text: "You are a helpful assistant."}, result.Messages[0].Content)
	})

	t.Run("GetPrompt with arguments", func(t *testing.T) {
		// Test getting a prompt with arguments
		server := agentstesting.NewFakeMCPPromptServer("")
		server.AddPrompt(
			"generate_code_review_instructions",
			"Generate agent instructions for code review tasks",
			nil,
		)
		server.SetPromptResult(
			"generate_code_review_instructions",
			"You are a senior {{.language}} code review specialist. Focus on {{.focus}}.",
		)

		result, err := server.GetPrompt(
			t.Context(),
			"generate_code_review_instructions",
			map[string]string{"focus": "security vulnerabilities", "language": "python"},
		)
		require.NoError(t, err)
		assert.Equal(t, &mcp.TextContent{
			Text: "You are a senior python code review specialist. Focus on security vulnerabilities.",
		}, result.Messages[0].Content)
	})

	t.Run("prompt not found", func(t *testing.T) {
		// Test getting a prompt that doesn't exist
		server := agentstesting.NewFakeMCPPromptServer("")
		_, err := server.GetPrompt(t.Context(), "nonexistent", nil)
		assert.EqualError(t, err, `prompt "nonexistent" not found`)
	})

	t.Run("multiple prompts", func(t *testing.T) {
		// Test server with multiple prompts
		server := agentstesting.NewFakeMCPPromptServer("")

		// Add multiple prompts
		server.AddPrompt(
			"generate_code_review_instructions",
			"Generate agent instructions for code review tasks",
			nil,
		)
		server.AddPrompt(
			"generate_testing_instructions",
			"Generate agent instructions for testing tasks",
			nil,
		)

		server.SetPromptResult("generate_code_review_instructions", "You are a code reviewer.")
		server.SetPromptResult("generate_testing_instructions", "You are a test engineer.")

		ctx := t.Context()

		// Test listing prompts
		promptsResult, err := server.ListPrompts(ctx)
		require.NoError(t, err)
		require.Len(t, promptsResult.Prompts, 2)

		promptNames := make([]string, len(promptsResult.Prompts))
		for i, promptResult := range promptsResult.Prompts {
			promptNames[i] = promptResult.Name
		}
		assert.Contains(t, promptNames, "generate_code_review_instructions")
		assert.Contains(t, promptNames, "generate_testing_instructions")

		// Test getting each prompt
		reviewResult, err := server.GetPrompt(ctx, "generate_code_review_instructions", nil)
		assert.Equal(t, &mcp.TextContent{Text: "You are a code reviewer."}, reviewResult.Messages[0].Content)

		testingResult, err := server.GetPrompt(ctx, "generate_testing_instructions", nil)
		assert.Equal(t, &mcp.TextContent{Text: "You are a test engineer."}, testingResult.Messages[0].Content)
	})

	t.Run("prompt with complex arguments", func(t *testing.T) {
		// Test prompt with complex argument formatting
		server := agentstesting.NewFakeMCPPromptServer("")
		server.AddPrompt(
			"generate_detailed_instructions",
			"Generate detailed instructions with multiple parameters",
			nil,
		)
		server.SetPromptResult(
			"generate_detailed_instructions",
			"You are a {{.role}} specialist. Your focus is on {{.focus}}. "+
				"You work with {{.language}} code. Your experience level is {{.level}}.",
		)
		arguments := map[string]string{
			"role":     "security",
			"focus":    "vulnerability detection",
			"language": "Python",
			"level":    "senior",
		}

		result, err := server.GetPrompt(t.Context(), "generate_detailed_instructions", arguments)
		require.NoError(t, err)

		assert.Equal(t, &mcp.TextContent{
			Text: "You are a security specialist. Your focus is on vulnerability detection. " +
				"You work with Python code. Your experience level is senior.",
		}, result.Messages[0].Content)
	})

	t.Run("prompt with invalid template", func(t *testing.T) {
		// Test prompt with missing arguments in template string
		server := agentstesting.NewFakeMCPPromptServer("")
		server.AddPrompt("foo", "", nil)
		server.SetPromptResult("foo", "Hello {{{.name}}!")

		// Only provide one of the required arguments
		args := map[string]string{"name": "Bar"}
		result, err := server.GetPrompt(t.Context(), "foo", args)
		require.NoError(t, err)

		// Should return the original string since formatting fails
		assert.Equal(t, &mcp.TextContent{Text: "Hello {{{.name}}!"}, result.Messages[0].Content)
	})

	t.Run("prompt with missing arguments", func(t *testing.T) {
		// Test prompt with missing arguments in template string
		server := agentstesting.NewFakeMCPPromptServer("")
		server.AddPrompt("incomplete_prompt", "Prompt with missing arguments", nil)
		server.SetPromptResult("incomplete_prompt", "You are a {{.role}} working on {{.task}}.")

		// Only provide one of the required arguments
		args := map[string]string{"role": "developer"}
		result, err := server.GetPrompt(t.Context(), "incomplete_prompt", args)
		require.NoError(t, err)

		assert.Equal(t, &mcp.TextContent{Text: "You are a developer working on <no value>."}, result.Messages[0].Content)
	})

	t.Run("cleanup", func(t *testing.T) {
		// Test that prompt server cleanup works correctly
		server := agentstesting.NewFakeMCPPromptServer("")
		server.AddPrompt("test_prompt", "Test prompt", nil)
		server.SetPromptResult("test_prompt", "Test result")

		ctx := t.Context()

		// Test that server works before cleanup
		result, err := server.GetPrompt(ctx, "test_prompt", nil)
		require.NoError(t, err)
		assert.Equal(t, &mcp.TextContent{Text: "Test result"}, result.Messages[0].Content)

		// Cleanup should not raise any errors
		require.NoError(t, server.Cleanup(ctx))

		// Server should still work after cleanup (in this fake implementation)
		result, err = server.GetPrompt(ctx, "test_prompt", nil)
		require.NoError(t, err)
		assert.Equal(t, &mcp.TextContent{Text: "Test result"}, result.Messages[0].Content)
	})
}

func TestAgentWithPromptInstructions(t *testing.T) {
	// Test using prompt-generated instructions with an agent
	server := agentstesting.NewFakeMCPPromptServer("")
	server.AddPrompt(
		"generate_code_review_instructions",
		"Generate agent instructions for code review tasks",
		nil,
	)
	server.SetPromptResult(
		"generate_code_review_instructions",
		"You are a code reviewer. Analyze the provided code for security issues.",
	)

	// Get instructions from prompt
	promptResult, err := server.GetPrompt(t.Context(), "generate_code_review_instructions", nil)
	require.NoError(t, err)
	instructions := promptResult.Messages[0].Content.(*mcp.TextContent).Text

	// Create agent with prompt-generated instructions
	model := agentstesting.NewFakeModel(false, nil)
	agent := agents.New("prompt_agent").
		WithInstructions(instructions).
		WithModelInstance(model).
		AddMCPServer(server)

	// Mock model response
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("Code analysis complete. Found security vulnerability."),
		}},
	})

	// Run the agent
	result, err := agents.Run(t.Context(), agent, "Review this code: def unsafe_exec(cmd): os.system(cmd)")
	require.NoError(t, err)
	assert.Equal(t, "Code analysis complete. Found security vulnerability.", result.FinalOutput)
	assert.Equal(t, agents.InstructionsStr("You are a code reviewer. Analyze the provided code for security issues."), agent.Instructions)
}

func TestAgentWithPromptInstructionsStreaming(t *testing.T) {
	// Test using prompt-generated instructions with streaming and non-streaming
	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			server := agentstesting.NewFakeMCPPromptServer("")
			server.AddPrompt(
				"generate_code_review_instructions",
				"Generate agent instructions for code review tasks",
				nil,
			)
			server.SetPromptResult(
				"generate_code_review_instructions",
				"You are a {{.language}} code reviewer focusing on {{.focus}}.",
			)

			// Get instructions from prompt with arguments
			promptResult, err := server.GetPrompt(
				t.Context(),
				"generate_code_review_instructions",
				map[string]string{"language": "Python", "focus": "security"},
			)
			require.NoError(t, err)
			instructions := promptResult.Messages[0].Content.(*mcp.TextContent).Text

			// Create agent
			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("streaming_prompt_agent").
				WithInstructions(instructions).
				WithModelInstance(model).
				AddMCPServer(server)

			model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("Security analysis complete."),
				}},
			})

			var finalResult any
			if streaming {
				result, err := agents.RunStreamed(t.Context(), agent, "Review code")
				require.NoError(t, err)
				err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
				require.NoError(t, err)
				finalResult = result.FinalOutput()
			} else {
				result, err := agents.Run(t.Context(), agent, "Review code")
				require.NoError(t, err)
				finalResult = result.FinalOutput
			}
			assert.Equal(t, "Security analysis complete.", finalResult)
			assert.Equal(t, agents.InstructionsStr("You are a Python code reviewer focusing on security."), agent.Instructions)
		})
	}
}
