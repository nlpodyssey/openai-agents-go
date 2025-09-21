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
	"log"
	"os"
	"os/exec"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func runMCPServer() {
	ctx := context.Background()
	server := mcp.NewServer(&mcp.Implementation{Name: "test"}, nil)

	mcp.AddTool(
		server, &mcp.Tool{Name: "add_nop_tool", Description: "Adds nop_tool"},
		func(ctx context.Context, session *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, *struct{}, error) {
			mcp.AddTool(
				server, &mcp.Tool{Name: "nop_tool"},
				func(ctx context.Context, session *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, *struct{}, error) {
					return &mcp.CallToolResult{}, nil, nil
				},
			)
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "OK"}}}, nil, nil
		},
	)

	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}

func createMCPServerCommand(t *testing.T) *exec.Cmd {
	t.Helper()

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe)
	cmd.Env = append(os.Environ(), runAsMCPServer+"=true")

	return cmd
}
