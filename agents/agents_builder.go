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
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/openai/openai-go/packages/param"
)

// NewAgent returns an empty Agent that can be configured using the builder
// methods.
func NewAgent() *Agent {
	return &Agent{}
}

// WithName sets the Agent name.
func (a *Agent) WithName(name string) *Agent {
	a.Name = name
	return a
}

// WithInstructions sets the Agent instructions.
func (a *Agent) WithInstructions(instr string) *Agent {
	a.Instructions = InstructionsStr(instr)
	return a
}

// WithInstructionsFunc sets dynamic instructions using an InstructionsFunc.
func (a *Agent) WithInstructionsFunc(fn InstructionsFunc) *Agent {
	a.Instructions = fn
	return a
}

// WithInstructionsGetter sets custom instructions implementing InstructionsGetter.
func (a *Agent) WithInstructionsGetter(g InstructionsGetter) *Agent {
	a.Instructions = g
	return a
}

// WithHandoffDescription sets the handoff description.
func (a *Agent) WithHandoffDescription(desc string) *Agent {
	a.HandoffDescription = desc
	return a
}

// WithHandoffs sets the agent handoffs.
func (a *Agent) WithHandoffs(handoffs ...Handoff) *Agent {
	a.Handoffs = handoffs
	return a
}

// WithAgentHandoffs sets the agent handoffs using Agent pointers.
func (a *Agent) WithAgentHandoffs(agents ...*Agent) *Agent {
	a.AgentHandoffs = agents
	return a
}

// WithModel sets the model to use by name.
func (a *Agent) WithModel(name string) *Agent {
	a.Model = param.NewOpt(NewAgentModelName(name))
	return a
}

// WithModelInstance sets the model using a Model implementation.
func (a *Agent) WithModelInstance(m Model) *Agent {
	a.Model = param.NewOpt(NewAgentModel(m))
	return a
}

// WithModelOpt sets the model using an AgentModel wrapped in param.Opt.
func (a *Agent) WithModelOpt(model param.Opt[AgentModel]) *Agent {
	a.Model = model
	return a
}

// WithModelSettings sets model-specific settings.
func (a *Agent) WithModelSettings(settings modelsettings.ModelSettings) *Agent {
	a.ModelSettings = settings
	return a
}

// WithTools sets the list of tools available to the agent.
func (a *Agent) WithTools(t ...tools.Tool) *Agent {
	a.Tools = append([]tools.Tool{}, t...)
	return a
}

// AddTool appends a tool to the agent's tool list.
func (a *Agent) AddTool(t tools.Tool) *Agent {
	a.Tools = append(a.Tools, t)
	return a
}

// WithInputGuardrails sets the input guardrails.
func (a *Agent) WithInputGuardrails(gr []InputGuardrail) *Agent {
	a.InputGuardrails = gr
	return a
}

// WithOutputGuardrails sets the output guardrails.
func (a *Agent) WithOutputGuardrails(gr []OutputGuardrail) *Agent {
	a.OutputGuardrails = gr
	return a
}

// WithOutputSchema sets the output schema.
func (a *Agent) WithOutputSchema(schema AgentOutputSchemaInterface) *Agent {
	a.OutputSchema = schema
	return a
}

// WithHooks sets the lifecycle hooks for the agent.
func (a *Agent) WithHooks(hooks AgentHooks) *Agent {
	a.Hooks = hooks
	return a
}

// WithToolUseBehavior configures how tool use is handled.
func (a *Agent) WithToolUseBehavior(b ToolUseBehavior) *Agent {
	a.ToolUseBehavior = b
	return a
}

// WithResetToolChoice sets whether tool choice is reset after use.
func (a *Agent) WithResetToolChoice(v param.Opt[bool]) *Agent {
	a.ResetToolChoice = v
	return a
}
