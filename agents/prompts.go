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
	"github.com/openai/openai-go/v3/responses"
)

// Prompt configuration to use for interacting with an OpenAI model.
type Prompt struct {
	// The unique ID of the prompt.
	ID string

	// Optional version of the prompt.
	Version param.Opt[string]

	// Optional variables to substitute into the prompt.
	Variables map[string]responses.ResponsePromptVariableUnionParam
}

// Prompt satisfies the Prompter interface, allowing you to define a static
// prompt. It returns the Prompt itself and nil error.
func (p Prompt) Prompt(context.Context, *Agent) (Prompt, error) {
	return p, nil
}

// Prompter is implemented by objects that can dynamically generate a prompt.
type Prompter interface {
	Prompt(context.Context, *Agent) (Prompt, error)
}

// DynamicPromptFunction is function that dynamically generates a prompt,
// satisfying the Prompter interface.
type DynamicPromptFunction func(context.Context, *Agent) (Prompt, error)

func (f DynamicPromptFunction) Prompt(ctx context.Context, agent *Agent) (Prompt, error) {
	return f(ctx, agent)
}

type promptUtil struct{}

func PromptUtil() promptUtil { return promptUtil{} }

func (promptUtil) ToModelInput(
	ctx context.Context,
	prompter Prompter,
	agent *Agent,
) (responses.ResponsePromptParam, bool, error) {
	if prompter == nil {
		return responses.ResponsePromptParam{}, false, nil
	}

	prompt, err := prompter.Prompt(ctx, agent)
	if err != nil {
		return responses.ResponsePromptParam{}, false, err
	}
	return responses.ResponsePromptParam{
		ID:        prompt.ID,
		Version:   prompt.Version,
		Variables: prompt.Variables,
	}, true, nil
}
