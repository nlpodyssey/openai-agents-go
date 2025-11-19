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

// MCPToolApprovalFunctionResult the result of an MCP tool approval function.
type MCPToolApprovalFunctionResult struct {
	// Whether to approve the tool call.
	Approve bool

	// An optional reason, if rejected.
	Reason string
}

// MCPToolApprovalFunction is a function that approves or rejects a tool call.
type MCPToolApprovalFunction = func(context.Context, responses.ResponseOutputItemMcpApprovalRequest) (MCPToolApprovalFunctionResult, error)

// HostedMCPTool is a tool that allows the LLM to use a remote MCP server.
// The LLM will automatically list and call tools, without requiring a round trip back to your code.
// If you want to run MCP servers locally via stdio, in a VPC or other non-publicly-accessible
// environment, or you just prefer to run tool calls locally, then you can instead use an MCPServer
// and pass it to the agent.
type HostedMCPTool struct {
	// The MCP tool config, which includes the server URL and other settings.
	ToolConfig responses.ToolMcpParam

	// An optional function that will be called if approval is requested for an MCP tool.
	// If not provided, you will need to manually add approvals/rejections to the input and call
	// `Run(...)` again.
	OnApprovalRequest MCPToolApprovalFunction
}

func (t HostedMCPTool) ToolName() string {
	return "hosted_mcp"
}

func (t HostedMCPTool) isTool() {}
