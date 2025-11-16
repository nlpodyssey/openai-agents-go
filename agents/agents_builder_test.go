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
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
)

func TestAgentBuilder_Chaining(t *testing.T) {
	instr := "hello"
	tool := FunctionTool{Name: "t"}

	agent := New("agent").
		WithInstructions(instr).
		WithHandoffDescription("desc").
		WithTools(tool).
		AddTool(tool).
		WithModel("model")

	assert.Equal(t, "agent", agent.Name)
	assert.Equal(t, InstructionsStr(instr), agent.Instructions)
	assert.Equal(t, "desc", agent.HandoffDescription)
	assert.Len(t, agent.Tools, 2)
	assert.Equal(t, "t", agent.Tools[0].(FunctionTool).Name)
	assert.Equal(t, param.NewOpt(NewAgentModelName("model")), agent.Model)
}

func TestAgentBuilder_ReturnsSamePointer(t *testing.T) {
	agent := New("foo")
	returned := agent.WithHandoffDescription("bar")
	assert.Same(t, agent, returned)
}

func TestAgentBuilder_WithInstructionsFunc(t *testing.T) {
	agent := New("").WithInstructionsFunc(func(ctx context.Context, a *Agent) (string, error) {
		return "dynamic", nil
	})
	v, err := agent.Instructions.GetInstructions(context.Background(), agent)
	assert.NoError(t, err)
	assert.Equal(t, "dynamic", v)
}
