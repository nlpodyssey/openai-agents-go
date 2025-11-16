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
	"slices"
	"sync"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/util/transforms"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

type ToolsToFinalOutputResult struct {
	// Whether this is the final output.
	// If false, the LLM will run again and receive the tool call output.
	IsFinalOutput bool

	// The final output.
	// Can be missing if `IsFinalOutput` is false.

	// The final output. Can be Null if `IsFinalOutput` is false, otherwise must match the
	// `OutputType` of the agent.
	FinalOutput param.Opt[any]
}

// MCPConfig provides configuration parameters for MCP servers.
type MCPConfig struct {
	// If true, we will attempt to convert the MCP schemas to strict-mode schemas.
	// This is a best-effort conversion, so some schemas may not be convertible.
	// Defaults to false.
	ConvertSchemasToStrict bool
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

	// Optional Prompter object. Prompts allow you to dynamically configure the instructions,
	// tools and other config for an agent outside your code.
	// Only usable with OpenAI models, using the Responses API.
	Prompt Prompter

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

	// Optional list of Model Context Protocol (https://modelcontextprotocol.io) servers that
	// the agent can use. Every time the agent runs, it will include tools from these servers in the
	// list of available tools.
	//
	// NOTE: You are expected to manage the lifecycle of these servers. Specifically, you must call
	// `MCPServer.Connect()` before passing it to the agent, and `MCPServer.Cleanup()` when the server is no
	// longer needed.
	MCPServers []MCPServer

	// Optional configuration for MCP servers.
	MCPConfig MCPConfig

	// A list of checks that run in parallel to the agent's execution, before generating a
	// response. Runs only if the agent is the first agent in the chain.
	InputGuardrails []InputGuardrail

	// A list of checks that run on the final output of the agent, after generating a response.
	// Runs only if the agent produces a final output.
	OutputGuardrails []OutputGuardrail

	// Optional output type describing the output. If not provided, the output will be a simple string.
	OutputType OutputTypeInterface

	// Optional object that receives callbacks on various lifecycle events for this agent.
	Hooks AgentHooks

	// Optional property which lets you configure how tool use is handled.
	// - RunLLMAgain: The default behavior. Tools are run, and then the LLM receives the results
	//   and gets to respond.
	// - StopOnFirstTool: The output from the first tool call is treated as the final result.
	//   In other words, it isn’t sent back to the LLM for further processing but is used directly
	//   as the final output.
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
	CustomOutputExtractor func(context.Context, RunResult) (string, error)

	// Optional static or dynamic flag reporting whether the tool is enabled.
	// If omitted, the tool is enabled by default.
	IsEnabled FunctionToolEnabler
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
			return params.CustomOutputExtractor(ctx, *output)
		}

		return ItemHelpers().TextMessageOutputs(output.NewItems), nil
	}

	tool := NewFunctionTool(name, params.ToolDescription, runAgent)
	tool.IsEnabled = params.IsEnabled
	return tool
}

// GetSystemPrompt returns the system prompt for the agent.
func (a *Agent) GetSystemPrompt(ctx context.Context) (param.Opt[string], error) {
	if a.Instructions == nil {
		return param.Opt[string]{}, nil
	}
	v, err := a.Instructions.GetInstructions(ctx, a)
	if err != nil {
		return param.Opt[string]{}, err
	}
	return param.NewOpt(v), nil
}

// GetPrompt returns the prompt for the agent.
func (a *Agent) GetPrompt(ctx context.Context) (responses.ResponsePromptParam, bool, error) {
	return PromptUtil().ToModelInput(ctx, a.Prompt, a)
}

// GetMCPTools fetches the available tools from the MCP servers.
func (a *Agent) GetMCPTools(ctx context.Context) ([]Tool, error) {
	return MCPUtil().GetAllFunctionTools(ctx, a.MCPServers, a.MCPConfig.ConvertSchemasToStrict, a)
}

// GetAllTools returns all agent tools, including MCP tools and function tools.
func (a *Agent) GetAllTools(ctx context.Context) ([]Tool, error) {
	mcpTools, err := a.GetMCPTools(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get MCP tools: %w", err)
	}

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

	return slices.Concat(mcpTools, enabledTools), nil
}
