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

	"github.com/openai/openai-go/v3/packages/param"
)

// RunHooks is implemented by an object that receives callbacks on various
// lifecycle events in an agent run.
type RunHooks interface {
	// OnLLMStart is called just before invoking the LLM for this agent.
	OnLLMStart(ctx context.Context, agent *Agent, systemPrompt param.Opt[string], inputItems []TResponseInputItem) error

	// OnLLMEnd is called immediately after the LLM call returns for this agent.
	OnLLMEnd(ctx context.Context, agent *Agent, response ModelResponse) error

	// OnAgentStart is called before the agent is invoked. Called each time the current agent changes.
	OnAgentStart(ctx context.Context, agent *Agent) error

	// OnAgentEnd is called when the agent produces a final output.
	OnAgentEnd(ctx context.Context, agent *Agent, output any) error

	// OnHandoff is called when a handoff occurs.
	OnHandoff(ctx context.Context, fromAgent, toAgent *Agent) error

	// OnToolStart is called concurrently with tool invocation.
	OnToolStart(ctx context.Context, agent *Agent, tool Tool) error

	// OnToolEnd is called after a tool is invoked.
	OnToolEnd(ctx context.Context, agent *Agent, tool Tool, result any) error
}

type NoOpRunHooks struct{}

func (NoOpRunHooks) OnLLMStart(context.Context, *Agent, param.Opt[string], []TResponseInputItem) error {
	return nil
}
func (NoOpRunHooks) OnLLMEnd(context.Context, *Agent, ModelResponse) error {
	return nil
}
func (NoOpRunHooks) OnAgentStart(context.Context, *Agent) error {
	return nil
}
func (NoOpRunHooks) OnAgentEnd(context.Context, *Agent, any) error {
	return nil
}
func (NoOpRunHooks) OnHandoff(context.Context, *Agent, *Agent) error {
	return nil
}
func (NoOpRunHooks) OnToolStart(context.Context, *Agent, Tool) error {
	return nil
}
func (NoOpRunHooks) OnToolEnd(context.Context, *Agent, Tool, any) error {
	return nil
}

// AgentHooks is implemented by an object that receives callbacks on various
// lifecycle events for a specific agent.
// You can set this on `Agent.Hooks` to receive events for that specific agent.
type AgentHooks interface {
	// OnStart is called before the agent is invoked. Called each time the running agent is changed to this agent.
	OnStart(ctx context.Context, agent *Agent) error

	// OnEnd is called when the agent produces a final output.
	OnEnd(ctx context.Context, agent *Agent, output any) error

	// OnHandoff is called when the agent is being handed off to.
	// The `source` is the agent that is handing off to this agent.
	OnHandoff(ctx context.Context, agent, source *Agent) error

	// OnToolStart is called concurrently with tool invocation.
	OnToolStart(ctx context.Context, agent *Agent, tool Tool, arguments any) error

	// OnToolEnd is called after a tool is invoked.
	OnToolEnd(ctx context.Context, agent *Agent, tool Tool, result any) error

	// OnLLMStart is called immediately before the agent issues an LLM call.
	OnLLMStart(ctx context.Context, agent *Agent, systemPrompt param.Opt[string], inputItems []TResponseInputItem) error

	// OnLLMEnd is called immediately after the agent receives the LLM response.
	OnLLMEnd(ctx context.Context, agent *Agent, response ModelResponse) error
}
