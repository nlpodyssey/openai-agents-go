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

	"github.com/openai/openai-go/v3/responses"
)

// LocalShellCommandRequest is a request to execute a command on a shell.
type LocalShellCommandRequest struct {
	// The data from the local shell tool call.
	Data responses.ResponseOutputItemLocalShellCall
}

// LocalShellExecutor is a function that executes a command on a shell.
type LocalShellExecutor = func(context.Context, LocalShellCommandRequest) (string, error)

// LocalShellTool is a tool that allows the LLM to execute commands on a shell.
type LocalShellTool struct {
	// A function that executes a command on a shell.
	Executor LocalShellExecutor
}

func (t LocalShellTool) ToolName() string {
	return "local_shell"
}

func (t LocalShellTool) isTool() {}
