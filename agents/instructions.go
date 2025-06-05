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

	"github.com/nlpodyssey/openai-agents-go/runcontext"
)

// The Instructions for the agent.
//
// Can either be a string (StringInstructions), or a function (FunctionInstructions)
// that dynamically generates instructions for the agent.
// If you provide a function, it will be called with the context and the agent instance.
// It must return a string.
type Instructions interface {
	isInstructions()
}

type StringInstructions string

func (StringInstructions) isInstructions() {}

func (si StringInstructions) String() string {
	return string(si)
}

type FunctionInstructions func(context.Context, *runcontext.RunContextWrapper, *Agent) (string, error)

func (FunctionInstructions) isInstructions() {}
