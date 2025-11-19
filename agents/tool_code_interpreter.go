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
	"github.com/openai/openai-go/v3/responses"
)

// CodeInterpreterTool is a tool that allows the LLM to execute code in a sandboxed environment.
type CodeInterpreterTool struct {
	// The tool config, which includes the container and other settings.
	ToolConfig responses.ToolCodeInterpreterParam
}

func (t CodeInterpreterTool) ToolName() string {
	return "code_interpreter"
}

func (t CodeInterpreterTool) isTool() {}
