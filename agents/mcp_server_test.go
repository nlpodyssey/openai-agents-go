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
	"errors"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPServerWithClientSessionCaching(t *testing.T) {
	t.Run("CacheToolsList true", func(t *testing.T) {
		server := NewMCPServerStdio(MCPServerStdioParams{
			Command:        createMCPServerCommand(t),
			CacheToolsList: true,
		})

		err := server.Run(t.Context(), func(ctx context.Context, server *MCPServerWithClientSession) error {
			agent := New("test_agent").WithInstructions("Test agent")

			// First call should get tools and cache them. The NOP tool must not be present.
			tools, err := server.ListTools(ctx, agent)
			require.NoError(t, err)
			toolNames := collectMCPToolNames(tools)
			require.NotContains(t, toolNames, "nop_tool")

			// Tell the server to add the NOP tool.
			_, err = server.CallTool(ctx, "add_nop_tool", nil)
			require.NoError(t, err)

			// Tools should be cached, so the NOP tool must not be present yet.
			tools, err = server.ListTools(ctx, agent)
			require.NoError(t, err)
			toolNames = collectMCPToolNames(tools)
			require.NotContains(t, toolNames, "nop_tool")

			// Invalidate the cache and list the tools again. Now the NOP tool should be there.
			server.InvalidateToolsCache()
			tools, err = server.ListTools(ctx, agent)
			require.NoError(t, err)
			toolNames = collectMCPToolNames(tools)
			require.Contains(t, toolNames, "nop_tool")

			return nil
		})
		require.NoError(t, err)
	})

	t.Run("CacheToolsList false", func(t *testing.T) {
		server := NewMCPServerStdio(MCPServerStdioParams{
			Command:        createMCPServerCommand(t),
			CacheToolsList: false,
		})

		err := server.Run(t.Context(), func(ctx context.Context, server *MCPServerWithClientSession) error {
			agent := New("test_agent").WithInstructions("Test agent")

			// First call should get tools and cache them. The NOP tool must not be present.
			tools, err := server.ListTools(ctx, agent)
			require.NoError(t, err)
			toolNames := collectMCPToolNames(tools)
			require.NotContains(t, toolNames, "nop_tool")

			// Tell the server to add the NOP tool.
			_, err = server.CallTool(ctx, "add_nop_tool", nil)
			require.NoError(t, err)

			// Tools should not be cached, so the NOP must be already present.
			tools, err = server.ListTools(ctx, agent)
			require.NoError(t, err)
			toolNames = collectMCPToolNames(tools)
			require.Contains(t, toolNames, "nop_tool")

			return nil
		})
		require.NoError(t, err)
	})
}

func TestMCPServerWithClientSessionSession(t *testing.T) {
	t.Run("using Run", func(t *testing.T) {
		server := NewMCPServerStdio(MCPServerStdioParams{
			Command: createMCPServerCommand(t),
		})
		require.Nil(t, server.session)
		err := server.Run(t.Context(), func(ctx context.Context, server *MCPServerWithClientSession) error {
			require.NotNil(t, server.session)
			return nil
		})
		require.NoError(t, err)
		require.Nil(t, server.session)
	})

	t.Run("with Connect Cleanup", func(t *testing.T) {
		server := NewMCPServerStdio(MCPServerStdioParams{
			Command: createMCPServerCommand(t),
		})
		ctx := t.Context()

		require.Nil(t, server.session)

		require.NoError(t, server.Connect(ctx))
		require.NotNil(t, server.session)

		require.NoError(t, server.Cleanup(ctx))
		require.Nil(t, server.session)
	})
}

type failingMCPTransport struct {
	err error
}

func (t failingMCPTransport) Connect(ctx context.Context) (mcp.Connection, error) { return nil, t.err }

func TestMCPServerWithClientSessionErrors(t *testing.T) {
	t.Run("transport connection error", func(t *testing.T) {
		testErr := errors.New("error")
		server := NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
			Name:      "test",
			Transport: failingMCPTransport{err: testErr},
		})
		ctx := t.Context()

		err := server.Connect(ctx)
		require.ErrorIs(t, err, testErr)
		require.Nil(t, server.session)

		err = server.Run(ctx, func(context.Context, *MCPServerWithClientSession) error {
			t.Fatal("run callback should never be called")
			return nil
		})
		require.ErrorIs(t, err, testErr)
		require.Nil(t, server.session)
	})

	t.Run("not calling Connect", func(t *testing.T) {
		agent := New("test_agent").WithInstructions("Test agent")
		server := NewMCPServerStdio(MCPServerStdioParams{
			Command:        createMCPServerCommand(t),
			CacheToolsList: true,
		})
		ctx := t.Context()

		_, err := server.ListTools(ctx, agent)
		assert.ErrorAs(t, err, &UserError{})

		_, err = server.CallTool(ctx, "add_nop_tool", nil)
		assert.ErrorAs(t, err, &UserError{})
	})
}

func collectMCPToolNames(tools []*mcp.Tool) []string {
	names := make([]string, len(tools))
	for i, tool := range tools {
		names[i] = tool.Name
	}
	return names
}
