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
	"sync"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/util/transforms"
	"github.com/openai/openai-go/packages/param"
)

type ToolsToFinalOutputResult struct {
	// Whether this is the final output.
	// If false, the LLM will run again and receive the tool call output.
	IsFinalOutput bool

	// The final output.
	// Can be missing if `IsFinalOutput` is false.
	FinalOutput param.Opt[any]
}

// An Agent is an AI model configured with instructions, tools, guardrails, handoffs and more.
//
// We strongly recommend passing `Instructions`, which is the "system prompt" for the agent. In
// addition, you can pass `HandoffDescription`, which is a human-readable description of the
// agent, used when the agent is used inside tools/handoffs.
type Agent struct {
	// The name of the agent.
	Name string

	// Optional instructions for the agent. Will be used as the "system prompt" when this agent is
	// invoked. Describes what the agent should do, and how it responds.
	Instructions InstructionsGetter

	// Optional description of the agent. This is used when the agent is used as a handoff, so that an
	// LLM knows what it does and when to invoke it.
	HandoffDescription string

	// Handoffs are sub-agents that the agent can delegate to. You can provide a list of handoffs,
	// and the agent can choose to delegate to them if relevant. Allows for separation of concerns and
	// modularity.
	//
	// Here you can provide a list of Handoff objects. In order to use Agent objects as
	// handoffs, add them to AgentHandoffs.
	Handoffs []Handoff

	// List of Agent objects to be used as handoffs. They will be converted to Handoff objects
	// before use. If you already have a Handoff, add it to Handoffs.
	AgentHandoffs []*Agent

	// The model implementation to use when invoking the LLM.
	Model param.Opt[AgentModel]

	// Configures model-specific tuning parameters (e.g. temperature, top_p).
	ModelSettings modelsettings.ModelSettings

	// A list of tools that the agent can use.
	Tools []Tool

	// A list of checks that run in parallel to the agent's execution, before generating a
	// response. Runs only if the agent is the first agent in the chain.
	InputGuardrails []InputGuardrail

	// A list of checks that run on the final output of the agent, after generating a response.
	// Runs only if the agent produces a final output.
	OutputGuardrails []OutputGuardrail

	// Optional output schema object describing the output. If not provided, the output will be a simple string.
	OutputSchema AgentOutputSchemaInterface

	// Optional object that receives callbacks on various lifecycle events for this agent.
	Hooks AgentHooks

	// Optional property which lets you configure how tool use is handled.
	// - RunLLMAgain: The default behavior. Tools are run, and then the LLM receives the results
	//   and gets to respond.
	// - StopOnFirstTool: The output of the first tool call is used as the final output. This
	//   means that the LLM does not process the result of the tool call.
	// - StopAtTools: The agent will stop running if any of the tools in the list are called.
	//   The final output will be the output of the first matching tool call. The LLM does not
	//   process the result of the tool call.
	// - ToolsToFinalOutputFunction: If you pass a function, it will be called with the run context and the list of
	//   tool results. It must return a `ToolsToFinalOutputResult`, which determines whether the tool
	//   calls result in a final output.
	//
	// NOTE: This configuration is specific to function tools. Hosted tools, such as file search,
	// web search, etc. are always processed by the LLM.
	ToolUseBehavior ToolUseBehavior

	// Whether to reset the tool choice to the default value after a tool has been called.
	// Defaults to true.
	// This ensures that the agent doesn't enter an infinite loop of tool usage.
	ResetToolChoice param.Opt[bool]
}

type AgentAsToolParams struct {
	// Optional name of the tool. If not provided, the agent's name will be used.
	ToolName string

	// Optional description of the tool, which should indicate what it does and when to use it.
	ToolDescription string

	// Optional function that extracts the output from the agent.
	// If not provided, the last message from the agent will be used.
	CustomOutputExtractor func(RunResult) (string, error)
}

// AsTool transforms this agent into a tool, callable by other agents.
//
// This is different from handoffs in two ways:
//  1. In handoffs, the new agent receives the conversation history. In this tool, the new agent
//     receives generated input.
//  2. In handoffs, the new agent takes over the conversation. In this tool, the new agent is
//     called as a tool, and the conversation is continued by the original agent.
func (a *Agent) AsTool(params AgentAsToolParams) Tool {
	name := params.ToolName
	if name == "" {
		name = transforms.TransformStringFunctionStyle(a.Name)
	}

	type argsType struct {
		Input string `json:"input"`
	}

	runAgent := func(ctx context.Context, args argsType) (string, error) {
		output, err := DefaultRunner.Run(ctx, a, args.Input)
		if err != nil {
			return "", fmt.Errorf("failed to run agent %s as tool: %w", a.Name, err)
		}
		if params.CustomOutputExtractor != nil {
			return params.CustomOutputExtractor(*output)
		}

		return ItemHelpers().TextMessageOutputs(output.NewItems), nil
	}

	return NewFunctionTool(name, params.ToolDescription, runAgent)
}

// GetSystemPrompt returns the system prompt for the agent.
func (a *Agent) GetSystemPrompt(ctx context.Context) (param.Opt[string], error) {
	if a.Instructions == nil {
		return param.Null[string](), nil
	}
	v, err := a.Instructions.GetInstructions(ctx, a)
	if err != nil {
		return param.Null[string](), err
	}
	return param.NewOpt(v), nil
}

// GetAllTools returns all agent tools.
// It only includes function tools, since other types are omitted, as we don't need them.
func (a *Agent) GetAllTools(ctx context.Context) ([]Tool, error) {
	isEnabledResults := make([]bool, len(a.Tools))
	isEnabledErrors := make([]error, len(a.Tools))

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(len(a.Tools))

	for i, tool := range a.Tools {
		go func() {
			defer wg.Done()

			functionTool, ok := tool.(FunctionTool)
			if !ok || functionTool.IsEnabled == nil {
				isEnabledResults[i] = true
				return
			}

			isEnabledResults[i], isEnabledErrors[i] = functionTool.IsEnabled.IsEnabled(childCtx, a)
			if isEnabledErrors[i] != nil {
				cancel()
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(isEnabledErrors...); err != nil {
		return nil, err
	}

	var enabledTools []Tool
	for i, tool := range a.Tools {
		if isEnabledResults[i] {
			enabledTools = append(enabledTools, tool)
		}
	}

	return enabledTools, nil
}
