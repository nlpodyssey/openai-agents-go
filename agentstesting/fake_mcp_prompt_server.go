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
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"strings"
	"text/template"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
)

// FakeMCPPromptServer is a fake MCP server for testing prompt functionality.
type FakeMCPPromptServer struct {
	Prompts       []*mcp.Prompt
	PromptResults map[string]string
	serverName    string
}

func NewFakeMCPPromptServer(serverName string) *FakeMCPPromptServer {
	return &FakeMCPPromptServer{
		Prompts:       nil,
		PromptResults: make(map[string]string),
		serverName:    cmp.Or(serverName, "fake_prompt_server"),
	}
}

// AddPrompt adds a prompt to the fake server
func (s *FakeMCPPromptServer) AddPrompt(name, description string, arguments []*mcp.PromptArgument) {
	s.Prompts = append(s.Prompts, &mcp.Prompt{
		Name:        name,
		Description: description,
		Arguments:   arguments,
	})
}

// SetPromptResult sets the result that should be returned for a prompt.
func (s *FakeMCPPromptServer) SetPromptResult(name, result string) {
	s.PromptResults[name] = result
}

func (s *FakeMCPPromptServer) Connect(context.Context) error { return nil }
func (s *FakeMCPPromptServer) Cleanup(context.Context) error { return nil }
func (s *FakeMCPPromptServer) Name() string                  { return s.serverName }
func (s *FakeMCPPromptServer) UseStructuredContent() bool    { return false }

func (s *FakeMCPPromptServer) ListTools(context.Context, *agents.Agent) ([]*mcp.Tool, error) {
	return nil, nil
}

func (s *FakeMCPPromptServer) CallTool(context.Context, string, map[string]any) (*mcp.CallToolResult, error) {
	return nil, errors.New("this fake server doesn't support tools")
}

func (s *FakeMCPPromptServer) ListPrompts(context.Context) (*mcp.ListPromptsResult, error) {
	return &mcp.ListPromptsResult{Prompts: s.Prompts}, nil
}

func (s *FakeMCPPromptServer) GetPrompt(_ context.Context, name string, arguments map[string]string) (*mcp.GetPromptResult, error) {
	content, ok := s.PromptResults[name]
	if !ok {
		return nil, fmt.Errorf("prompt %q not found", name)
	}

	// If it's a template, try to execute it with arguments
	if len(arguments) > 0 && strings.Contains(content, "{{") {
		tmpl, err := template.New(name).Parse(content)
		if err == nil { // Use original content if template parsing fails
			var buf bytes.Buffer
			err = tmpl.Execute(&buf, arguments)
			if err == nil { // Use original content if template execution fails
				content = buf.String()
			}
		}
	}

	message := &mcp.PromptMessage{
		Content: &mcp.TextContent{Text: content},
		Role:    "user",
	}
	return &mcp.GetPromptResult{
		Description: fmt.Sprintf("Generated prompt for %s", name),
		Messages:    []*mcp.PromptMessage{message},
	}, nil
}
