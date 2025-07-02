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

// MCPTool is a hosted tool that lets the LLM use remote MCP servers to perform tasks.
// Currently only supported with OpenAI models, using the Responses API.
type MCPTool struct {
	// Required label for the MCP server.
	ServerLabel string

	// Required URL of the MCP server.
	ServerURL string

	// Optional approval requirement for tool calls.
	// Can be "never" to skip approvals for all tools, or an object specifying which tools to skip approvals for.
	RequireApproval any

	// Optional headers to include in requests to the MCP server.
	// Can be used for authentication (e.g., {"Authorization": "Bearer API_KEY"}).
	Headers map[string]string

	// Optional list of allowed tools from the MCP server.
	// If provided, only these tools will be imported.
	AllowedTools []string
}

func (t MCPTool) ToolName() string {
	return "mcp"
}

func (t MCPTool) isTool() {}
