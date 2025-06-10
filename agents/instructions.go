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
)

// InstructionsGetter interface is implemented by objects that can provide instructions to an Agent.
type InstructionsGetter interface {
	GetInstructions(context.Context, *Agent) (string, error)
}

// InstructionsStr satisfies InstructionsGetter providing a simple constant string value.
type InstructionsStr string

// GetInstructions returns the string value and always nil error.
func (s InstructionsStr) GetInstructions(context.Context, *Agent) (string, error) {
	return s.String(), nil
}

func (s InstructionsStr) String() string {
	return string(s)
}

// InstructionsFunc lets you implement a function that dynamically generates instructions for an Agent.
type InstructionsFunc func(context.Context, *Agent) (string, error)

// GetInstructions returns the string value and always nil error.
func (fn InstructionsFunc) GetInstructions(ctx context.Context, a *Agent) (string, error) {
	return fn(ctx, a)
}
