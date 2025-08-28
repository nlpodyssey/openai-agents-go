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

package agentstesting

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
)

type FakeMCPServer struct {
	name        string
	Tools       []*mcp.Tool
	ToolCalls   []string
	ToolResults []string
	ToolFilter  agents.MCPToolFilter
}

func NewFakeMCPServer(
	tools []*mcp.Tool,
	toolFilter agents.MCPToolFilter,
	name string,
) *FakeMCPServer {
	return &FakeMCPServer{
		name:       cmp.Or(name, "fake_mcp_server"),
		Tools:      tools,
		ToolFilter: toolFilter,
	}
}

func (s *FakeMCPServer) AddTool(name string, inputSchema *jsonschema.Schema) {
	s.Tools = append(s.Tools, &mcp.Tool{
		Name:        name,
		InputSchema: inputSchema,
	})
}

func (s *FakeMCPServer) Connect(context.Context) error { return nil }
func (s *FakeMCPServer) Cleanup(context.Context) error { return nil }
func (s *FakeMCPServer) Name() string                  { return s.name }
func (s *FakeMCPServer) UseStructuredContent() bool    { return false }

func (s *FakeMCPServer) ListTools(ctx context.Context, agent *agents.Agent) ([]*mcp.Tool, error) {
	tools := s.Tools

	// Apply tool filtering using the REAL implementation
	if s.ToolFilter != nil {
		filterContext := agents.MCPToolFilterContext{
			Agent:      agent,
			ServerName: s.name,
		}
		tools = agents.ApplyMCPToolFilter(ctx, filterContext, s.ToolFilter, tools, agent)
	}
	return tools, nil
}

func (s *FakeMCPServer) CallTool(_ context.Context, toolName string, arguments map[string]any) (*mcp.CallToolResult, error) {
	s.ToolCalls = append(s.ToolCalls, toolName)
	b, err := json.Marshal(arguments)
	if err != nil {
		return nil, err
	}
	result := fmt.Sprintf("result_%s_%s", toolName, string(b))
	s.ToolResults = append(s.ToolResults, result)
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: result}}}, nil
}

// ListPrompts returns empty list of prompts for fake server.
func (s *FakeMCPServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{}, nil
}

// GetPrompt returns a simple prompt result for fake server.
func (s *FakeMCPServer) GetPrompt(_ context.Context, name string, _ map[string]string) (*mcp.GetPromptResult, error) {
	content := fmt.Sprintf("Fake prompt content for %s", name)
	message := &mcp.PromptMessage{
		Content: &mcp.TextContent{Text: content},
		Role:    "user",
	}
	return &mcp.GetPromptResult{
		Description: fmt.Sprintf("Fake prompt: %s", name),
		Messages:    []*mcp.PromptMessage{message},
	}, nil
}
