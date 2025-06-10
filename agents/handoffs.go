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
	"strings"

	"github.com/nlpodyssey/openai-agents-go/util/transforms"
	"github.com/openai/openai-go/packages/param"
	"github.com/xeipuuv/gojsonschema"
)

// A Handoff is when an agent delegates a task to another agent.
//
// For example, in a customer support scenario you might have a "triage agent" that determines
// which agent should handle the user's request, and sub-agents that specialize in different
// areas like billing, account management, etc.
type Handoff struct {
	// The name of the tool that represents the handoff.
	ToolName string

	// The description of the tool that represents the handoff.
	ToolDescription string

	// The JSON schema for the handoff input. Can be empty if the handoff does not take an input.
	InputJSONSchema map[string]any

	// The function that invokes the handoff. The parameters passed are:
	// 1. The handoff run context
	// 2. The arguments from the LLM, as a JSON string. Empty string if InputJSONSchema is empty.
	//
	//Must return an agent.
	OnInvokeHandoff func(context.Context, string) (*Agent, error)

	// The name of the agent that is being handed off to.
	AgentName string

	// Optional function that filters the inputs that are passed to the next agent. By default, the new
	// agent sees the entire conversation history. In some cases, you may want to filter inputs e.g.
	// to remove older inputs, or remove tools from existing inputs.
	//
	// The function will receive the entire conversation history so far, including the input item
	// that triggered the handoff and a tool call output item representing the handoff tool's output.
	//
	// You are free to modify the input history or new items as you see fit. The next agent that
	// runs will receive `handoff_input_data.all_items`.
	//
	// IMPORTANT: in streaming mode, we will not stream anything as a result of this function. The
	// items generated before will already have been streamed.
	InputFilter HandoffInputFilter

	// Whether the input JSON schema is in strict mode. We **strongly** recommend setting this to
	// true, as it increases the likelihood of correct JSON input.
	// Defaults to true if omitted.
	StrictJSONSchema param.Opt[bool]
}

func (h Handoff) GetTransferMessage(agent *Agent) string {
	return fmt.Sprintf(`{{'assistant': '%s'}}`, agent.Name)
}

func DefaultHandoffToolName(agent *Agent) string {
	return transforms.TransformStringFunctionStyle("transfer_to_" + agent.Name)
}

func DefaultHandoffToolDescription(agent *Agent) string {
	return fmt.Sprintf(
		"Handoff to the %s agent to handle the request. %s",
		agent.Name,
		agent.HandoffDescription,
	)
}

// HandoffInputFilter is a function that filters the input data passed to the next agent.
type HandoffInputFilter = func(context.Context, HandoffInputData) (HandoffInputData, error)

type HandoffInputData struct {
	// The input history before `Runner().Run()` was called.
	InputHistory Input

	// The items generated before the agent turn where the handoff was invoked.
	PreHandoffItems []RunItem

	// The new items generated during the current agent turn, including the item that triggered the
	// handoff and the tool output message representing the response from the handoff output.
	NewItems []RunItem
}

type OnHandoff interface {
	isOnHandoff()
}

type OnHandoffWithInput func(ctx context.Context, jsonInput any) error

func (OnHandoffWithInput) isOnHandoff() {}

type OnHandoffWithoutInput func(context.Context) error

func (OnHandoffWithoutInput) isOnHandoff() {}

type HandoffFromAgentParams struct {
	// The agent to hand off to.
	Agent *Agent

	// Optional override for the name of the tool that represents the handoff.
	ToolNameOverride string

	// Optional override for the description of the tool that represents the handoff.
	ToolDescriptionOverride string

	// Optional function that runs when the handoff is invoked.
	OnHandoff OnHandoff

	// Optional JSON schema describing the type of the input to the handoff.
	// If provided, the input will be validated against this type.
	// Only relevant if you pass a function that takes an input.
	InputJSONSchema map[string]any

	// Optional function that filters the inputs that are passed to the next agent.
	InputFilter HandoffInputFilter
}

// HandoffFromAgent creates a Handoff from an Agent. It panics in case of problems.
func HandoffFromAgent(params HandoffFromAgentParams) Handoff {
	h, err := SafeHandoffFromAgent(params)
	if err != nil {
		panic(err)
	}
	return *h
}

// SafeHandoffFromAgent creates a Handoff from an Agent. It returns an error in case of problems.
func SafeHandoffFromAgent(params HandoffFromAgentParams) (*Handoff, error) {
	var rawInputJSONSchema map[string]any

	if len(params.InputJSONSchema) > 0 {
		rawInputJSONSchema = params.InputJSONSchema
		if params.OnHandoff == nil {
			return nil, errors.New("OnHandoff must be present since InputJSONSchema is given")
		}
		if _, ok := params.OnHandoff.(OnHandoffWithInput); !ok {
			return nil, errors.New("OnHandoff must be of type OnHandoffWithInput")
		}
	} else {
		rawInputJSONSchema = map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"properties":           map[string]any{},
			"required":             []string{},
		}
		if params.OnHandoff != nil {
			if _, ok := params.OnHandoff.(OnHandoffWithoutInput); !ok {
				return nil, errors.New("OnHandoff must be of type OnHandoffWithoutInput")
			}
		}
	}

	inputJSONSchemaLoader := gojsonschema.NewGoLoader(rawInputJSONSchema)
	inputJSONSchema, err := gojsonschema.NewSchema(inputJSONSchemaLoader)
	if err != nil {
		return nil, fmt.Errorf("failed to load and compile JSON schema: %w", err)
	}

	invokeHandoff := func(ctx context.Context, jsonInput string) (*Agent, error) {
		if len(params.InputJSONSchema) > 0 {
			inputJSONLoader := gojsonschema.NewStringLoader(jsonInput)
			result, err := inputJSONSchema.Validate(inputJSONLoader)
			if err != nil {
				return params.Agent, ModelBehaviorErrorf("failed to load and validate handoff JSON input: %w", err)
			}
			if !result.Valid() {
				var sb strings.Builder
				sb.WriteString("handoff input JSON validation failed with the following errors:\n")
				for _, e := range result.Errors() {
					_, _ = fmt.Fprintf(&sb, "- %s\n", e)
				}
				return params.Agent, NewModelBehaviorError(sb.String())
			}

			inputFunc := params.OnHandoff.(OnHandoffWithInput)
			err = inputFunc(ctx, jsonInput)
			if err != nil {
				return params.Agent, err
			}
		} else if params.OnHandoff != nil {
			noInputFunc := params.OnHandoff.(OnHandoffWithoutInput)
			err = noInputFunc(ctx)
			if err != nil {
				return params.Agent, err
			}
		}

		return params.Agent, nil
	}

	toolName := params.ToolNameOverride
	if toolName == "" {
		toolName = DefaultHandoffToolName(params.Agent)
	}

	toolDescription := params.ToolDescriptionOverride
	if toolDescription == "" {
		toolDescription = DefaultHandoffToolDescription(params.Agent)
	}

	return &Handoff{
		ToolName:         toolName,
		ToolDescription:  toolDescription,
		InputJSONSchema:  rawInputJSONSchema,
		OnInvokeHandoff:  invokeHandoff,
		AgentName:        params.Agent.Name,
		InputFilter:      params.InputFilter,
		StrictJSONSchema: param.NewOpt(true),
	}, nil
}
