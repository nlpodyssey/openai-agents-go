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

	"github.com/nlpodyssey/openai-agents-go/computer"
	"github.com/openai/openai-go/v3/responses"
)

// ComputerTool is a hosted tool that lets the LLM control a computer.
type ComputerTool struct {
	// The Computer implementation, which describes the environment and
	// dimensions of the computer, as well as implements the computer actions
	// like click, screenshot, etc.
	Computer computer.Computer

	// Optional callback to acknowledge computer tool safety checks.
	OnSafetyCheck func(context.Context, ComputerToolSafetyCheckData) (bool, error)
}

func (t ComputerTool) ToolName() string {
	return "computer_use_preview"
}

func (t ComputerTool) isTool() {}

// ComputerToolSafetyCheckData provides information about a computer tool safety check.
type ComputerToolSafetyCheckData struct {
	// The agent performing the computer action.
	Agent *Agent

	// The computer tool call.
	ToolCall responses.ResponseComputerToolCall

	// The pending safety check to acknowledge.
	SafetyCheck responses.ResponseComputerToolCallPendingSafetyCheck
}
