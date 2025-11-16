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

package agents_test

import (
	"cmp"
	"context"
	"errors"
	"log/slog"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPUtil_GetAllFunctionTools(t *testing.T) {
	// Test that the GetAllFunctionTools function returns all function tools
	// from a list of MCP servers.

	testTools := []struct {
		name   string
		schema *jsonschema.Schema
	}{
		{"test_tool_1", nil},
		{"test_tool_2", nil},
		{"test_tool_3", &jsonschema.Schema{Type: "object"}},
		{"test_tool_4", &jsonschema.Schema{
			Type:     "object",
			Required: []string{"bar", "baz"},
			Properties: map[string]*jsonschema.Schema{
				"bar": {Type: "string"},
				"baz": {Type: "integer"},
			},
		}},
		{"test_tool_5", &jsonschema.Schema{
			Type:     "object",
			Required: []string{"qux"},
			Properties: map[string]*jsonschema.Schema{
				"qux": {
					Type:                 "object",
					AdditionalProperties: &jsonschema.Schema{Type: "string"},
				},
			},
		}},
	}

	server1 := agentstesting.NewFakeMCPServer(nil, nil, "server1")
	server1.AddTool(testTools[0].name, testTools[0].schema)
	server1.AddTool(testTools[1].name, testTools[1].schema)

	server2 := agentstesting.NewFakeMCPServer(nil, nil, "server2")
	server2.AddTool(testTools[2].name, testTools[2].schema)
	server2.AddTool(testTools[3].name, testTools[3].schema)

	server3 := agentstesting.NewFakeMCPServer(nil, nil, "server3")
	server3.AddTool(testTools[4].name, testTools[4].schema)

	servers := []agents.MCPServer{server1, server2, server3}
	agent := agents.New("test_agent")

	t.Run("convertSchemasToStrict false", func(t *testing.T) {
		tools, err := agents.MCPUtil().GetAllFunctionTools(t.Context(), servers, false, agent)
		require.NoError(t, err)
		require.Len(t, tools, 5)

		assert.Equal(t, "test_tool_1", tools[0].ToolName())
		assert.Equal(t, "test_tool_2", tools[1].ToolName())
		assert.Equal(t, "test_tool_3", tools[2].ToolName())
		assert.Equal(t, "test_tool_4", tools[3].ToolName())
		assert.Equal(t, "test_tool_5", tools[4].ToolName())

		funcTools := make([]agents.FunctionTool, len(tools))
		for i, tool := range tools {
			require.IsType(t, agents.FunctionTool{}, tool)
			funcTools[i] = tool.(agents.FunctionTool)
		}

		type m = map[string]any
		assert.Equal(t, m{"properties": m{}}, funcTools[0].ParamsJSONSchema)
		assert.Equal(t, m{"properties": m{}}, funcTools[1].ParamsJSONSchema)
		assert.Equal(t, m{"type": "object", "properties": m{}}, funcTools[2].ParamsJSONSchema)
		assert.Equal(t, m{
			"type":     "object",
			"required": []any{"bar", "baz"},
			"properties": m{
				"bar": m{"type": "string"},
				"baz": m{"type": "integer"},
			},
		}, funcTools[3].ParamsJSONSchema)
		assert.Equal(t, m{
			"type":     "object",
			"required": []any{"qux"},
			"properties": m{
				"qux": m{
					"type":                 "object",
					"additionalProperties": m{"type": "string"},
				},
			},
		}, funcTools[4].ParamsJSONSchema)
	})

	t.Run("convertSchemasToStrict true", func(t *testing.T) {
		tools, err := agents.MCPUtil().GetAllFunctionTools(t.Context(), servers, true, agent)
		require.NoError(t, err)
		require.Len(t, tools, 5)

		assert.Equal(t, "test_tool_1", tools[0].ToolName())
		assert.Equal(t, "test_tool_2", tools[1].ToolName())
		assert.Equal(t, "test_tool_3", tools[2].ToolName())
		assert.Equal(t, "test_tool_4", tools[3].ToolName())
		assert.Equal(t, "test_tool_5", tools[4].ToolName())

		funcTools := make([]agents.FunctionTool, len(tools))
		for i, tool := range tools {
			require.IsType(t, agents.FunctionTool{}, tool)
			funcTools[i] = tool.(agents.FunctionTool)
		}

		type m = map[string]any
		assert.Equal(t, m{"properties": m{}, "required": []any{}}, funcTools[0].ParamsJSONSchema)
		assert.Equal(t, m{"properties": m{}, "required": []any{}}, funcTools[1].ParamsJSONSchema)
		assert.Equal(t, m{
			"type":                 "object",
			"additionalProperties": false,
			"properties":           m{},
			"required":             []any{},
		}, funcTools[2].ParamsJSONSchema)
		assert.Equal(t, m{
			"type":                 "object",
			"required":             []any{"bar", "baz"},
			"additionalProperties": false,
			"properties": m{
				"bar": m{"type": "string"},
				"baz": m{"type": "integer"},
			},
		}, funcTools[3].ParamsJSONSchema)
		assert.Equal(t, m{
			"type":                 "object",
			"required":             []any{"qux"},
			"additionalProperties": false,
			"properties": m{
				"qux": m{
					"type":                 "object",
					"additionalProperties": m{"type": "string"},
				},
			},
		}, funcTools[4].ParamsJSONSchema)
	})
}

type CrashingFakeMCPServer struct {
	*agentstesting.FakeMCPServer
	err error
}

func (s *CrashingFakeMCPServer) CallTool(context.Context, string, map[string]any) (*mcp.CallToolResult, error) {
	return nil, s.err
}

func TestMCPUtil_InvokeMCPTool(t *testing.T) {
	t.Run("no errors", func(t *testing.T) {
		// Test that the invoke_mcp_tool function invokes an MCP tool and returns the result.
		server := agentstesting.NewFakeMCPServer(nil, nil, "")
		server.AddTool("test_tool_1", nil)

		tool := &mcp.Tool{Name: "test_tool_1"}

		// Just making sure it doesn't return an error
		_, err := agents.MCPUtil().InvokeMCPTool(t.Context(), server, tool, "")
		require.NoError(t, err)
	})

	t.Run("bad JSON input", func(t *testing.T) {
		// Test that bad JSON input errors are logged and re-raised.
		var sbLog strings.Builder
		agents.SetLogger(
			slog.New(slog.NewTextHandler(&sbLog, &slog.HandlerOptions{Level: slog.LevelDebug})),
		)
		t.Cleanup(agents.ResetLogger)

		server := agentstesting.NewFakeMCPServer(nil, nil, "")
		server.AddTool("test_tool_1", nil)

		tool := &mcp.Tool{Name: "test_tool_1"}

		_, err := agents.MCPUtil().InvokeMCPTool(t.Context(), server, tool, "not_json")
		assert.ErrorAs(t, err, &agents.ModelBehaviorError{})

		assert.Regexp(t, regexp.MustCompile("Invalid JSON input.+toolName=test_tool_1"), sbLog.String())
	})

	t.Run("server call tool error", func(t *testing.T) {
		// Test that bad JSON input errors are logged and re-raised.
		var sbLog strings.Builder
		agents.SetLogger(
			slog.New(slog.NewTextHandler(&sbLog, &slog.HandlerOptions{Level: slog.LevelDebug})),
		)
		t.Cleanup(agents.ResetLogger)

		server := &CrashingFakeMCPServer{
			FakeMCPServer: agentstesting.NewFakeMCPServer(nil, nil, ""),
			err:           errors.New("error"),
		}
		server.AddTool("test_tool_1", nil)

		tool := &mcp.Tool{Name: "test_tool_1"}

		_, err := agents.MCPUtil().InvokeMCPTool(t.Context(), server, tool, "")
		assert.ErrorIs(t, err, server.err)

		assert.Regexp(t, regexp.MustCompile("Error invoking MCP tool.+toolName=test_tool_1"), sbLog.String())
	})
}

func TestAgentConvertSchemasToStrict(t *testing.T) {
	strictSchema := &jsonschema.Schema{
		Type:     "object",
		Required: []string{"bar", "baz"},
		Properties: map[string]*jsonschema.Schema{
			"bar": {Type: "string"},
			"baz": {Type: "integer"},
		},
	}
	nonStrictSchema := &jsonschema.Schema{
		Type: "object",
		AdditionalProperties: &jsonschema.Schema{
			Type: "string",
		},
	}
	possibleToConvertSchema := &jsonschema.Schema{
		Type:                 "object",
		Required:             []string{"bar", "baz"},
		AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
		Properties: map[string]*jsonschema.Schema{
			"bar": {Type: "string"},
			"baz": {Type: "integer"},
		},
	}

	server := agentstesting.NewFakeMCPServer(nil, nil, "")
	server.AddTool("foo", strictSchema)
	server.AddTool("bar", nonStrictSchema)
	server.AddTool("baz", possibleToConvertSchema)

	t.Run("ConvertSchemasToStrict true", func(t *testing.T) {
		// Test that setting ConvertSchemasToStrict to true converts non-strict schemas to strict.
		// - 'foo' tool is already strict and remains strict
		// - 'bar' tool is non-strict and becomes strict (additionalProperties set to false, etc.)

		mcpConfig := agents.MCPConfig{ConvertSchemasToStrict: true}
		agent := agents.New("test_agent").AddMCPServer(server).WithMCPConfig(mcpConfig)

		ctx := t.Context()
		tools, err := agent.GetMCPTools(ctx)
		require.NoError(t, err)

		funcTools := make(map[string]agents.FunctionTool)
		for _, tool := range tools {
			require.IsType(t, agents.FunctionTool{}, tool)
			funcTool := tool.(agents.FunctionTool)
			funcTools[funcTool.Name] = funcTool
		}

		type m = map[string]any

		// Checks that additionalProperties is set to false
		assert.Equal(t, m{
			"type":                 "object",
			"required":             []any{"bar", "baz"},
			"additionalProperties": false,
			"properties": m{
				"bar": m{"type": "string"},
				"baz": m{"type": "integer"},
			},
		}, funcTools["foo"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(true), funcTools["foo"].StrictJSONSchema)

		// Checks that additionalProperties is set to False
		assert.Equal(t, m{
			"type":                 "object",
			"additionalProperties": m{"type": "string"},
			"properties":           m{},
		}, funcTools["bar"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(false), funcTools["bar"].StrictJSONSchema)

		// Checks that additionalProperties is set to False
		assert.Equal(t, m{
			"type":                 "object",
			"required":             []any{"bar", "baz"},
			"additionalProperties": false,
			"properties": m{
				"bar": m{"type": "string"},
				"baz": m{"type": "integer"},
			},
		}, funcTools["baz"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(true), funcTools["baz"].StrictJSONSchema)
	})

	t.Run("ConvertSchemasToStrict false", func(t *testing.T) {
		// Test that setting ConvertSchemasToStrict to false leaves tool schemas as non-strict.
		// - 'foo' tool remains strict
		// - 'bar' tool remains non-strict

		mcpConfig := agents.MCPConfig{ConvertSchemasToStrict: false}
		agent := agents.New("test_agent").AddMCPServer(server).WithMCPConfig(mcpConfig)

		ctx := t.Context()
		tools, err := agent.GetMCPTools(ctx)
		require.NoError(t, err)

		funcTools := make(map[string]agents.FunctionTool)
		for _, tool := range tools {
			require.IsType(t, agents.FunctionTool{}, tool)
			funcTool := tool.(agents.FunctionTool)
			funcTools[funcTool.Name] = funcTool
		}

		type m = map[string]any

		assert.Equal(t, m{
			"type":     "object",
			"required": []any{"bar", "baz"},
			"properties": m{
				"bar": m{"type": "string"},
				"baz": m{"type": "integer"},
			},
		}, funcTools["foo"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(false), funcTools["foo"].StrictJSONSchema)

		assert.Equal(t, m{
			"type":                 "object",
			"additionalProperties": m{"type": "string"},
			"properties":           m{},
		}, funcTools["bar"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(false), funcTools["bar"].StrictJSONSchema)

		assert.Equal(t, m{
			"type":                 "object",
			"required":             []any{"bar", "baz"},
			"additionalProperties": false,
			"properties": m{
				"bar": m{"type": "string"},
				"baz": m{"type": "integer"},
			},
		}, funcTools["baz"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(false), funcTools["baz"].StrictJSONSchema)
	})

	t.Run("MCP config not set", func(t *testing.T) {
		// Test that setting ConvertSchemasToStrict to false leaves tool schemas as non-strict.
		// - 'foo' tool remains strict
		// - 'bar' tool remains non-strict

		agent := agents.New("test_agent").AddMCPServer(server)

		ctx := t.Context()
		tools, err := agent.GetMCPTools(ctx)
		require.NoError(t, err)

		funcTools := make(map[string]agents.FunctionTool)
		for _, tool := range tools {
			require.IsType(t, agents.FunctionTool{}, tool)
			funcTool := tool.(agents.FunctionTool)
			funcTools[funcTool.Name] = funcTool
		}

		type m = map[string]any

		assert.Equal(t, m{
			"type":     "object",
			"required": []any{"bar", "baz"},
			"properties": m{
				"bar": m{"type": "string"},
				"baz": m{"type": "integer"},
			},
		}, funcTools["foo"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(false), funcTools["foo"].StrictJSONSchema)

		assert.Equal(t, m{
			"type":                 "object",
			"additionalProperties": m{"type": "string"},
			"properties":           m{},
		}, funcTools["bar"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(false), funcTools["bar"].StrictJSONSchema)

		assert.Equal(t, m{
			"type":                 "object",
			"required":             []any{"bar", "baz"},
			"additionalProperties": false,
			"properties": m{
				"bar": m{"type": "string"},
				"baz": m{"type": "integer"},
			},
		}, funcTools["baz"].ParamsJSONSchema)
		assert.Equal(t, param.NewOpt(false), funcTools["baz"].StrictJSONSchema)
	})
}

func TestMCPUtilAddsPropertiesToJSONSchema(t *testing.T) {
	// The MCP spec doesn't require the input schema to have `properties`,
	// so we need to add it if it's missing.

	schema := &jsonschema.Schema{
		Type:        "object",
		Description: "Test tool",
	}

	server := agentstesting.NewFakeMCPServer(nil, nil, "")
	server.AddTool("test_tool", schema)

	agent := agents.New("test_agent")
	tools, err := agents.MCPUtil().GetAllFunctionTools(t.Context(), []agents.MCPServer{server}, false, agent)
	require.NoError(t, err)

	require.Len(t, tools, 1)
	tool := tools[0]

	require.IsType(t, agents.FunctionTool{}, tool)
	funcTool := tool.(agents.FunctionTool)

	assert.Equal(t, map[string]any{
		"type":        "object",
		"description": "Test tool",
		"properties":  map[string]any{},
	}, funcTool.ParamsJSONSchema)
}

func TestMCPToolFilter(t *testing.T) {
	newTestAgent := func(name string) *agents.Agent {
		return agents.New(cmp.Or(name, "test_agent"))
	}

	collectMCPToolNames := func(tools []*mcp.Tool) []string {
		names := make([]string, len(tools))
		for i, tool := range tools {
			names[i] = tool.Name
		}
		sort.Strings(names)
		return names
	}

	collectAgentsToolNames := func(tools []agents.Tool) []string {
		names := make([]string, len(tools))
		for i, tool := range tools {
			names[i] = tool.ToolName()
		}
		sort.Strings(names)
		return names
	}

	t.Run("static tool filtering", func(t *testing.T) {
		// Test all static tool filtering scenarios: allowed, blocked, both, none, etc.

		server := agentstesting.NewFakeMCPServer(nil, nil, "test_server")
		server.AddTool("tool1", nil)
		server.AddTool("tool2", nil)
		server.AddTool("tool3", nil)
		server.AddTool("tool4", nil)

		ctx := t.Context()
		agent := newTestAgent("")

		t.Run("AllowedToolNames only", func(t *testing.T) {
			server.ToolFilter = agents.MCPToolFilterStatic{
				AllowedToolNames: []string{"tool1", "tool3"},
				BlockedToolNames: nil,
			}
			tools, err := server.ListTools(ctx, agent)
			require.NoError(t, err)
			names := collectMCPToolNames(tools)
			assert.Equal(t, []string{"tool1", "tool3"}, names)
		})

		t.Run("BlockedToolNames only", func(t *testing.T) {
			server.ToolFilter = agents.MCPToolFilterStatic{
				AllowedToolNames: nil,
				BlockedToolNames: []string{"tool2", "tool3"},
			}
			tools, err := server.ListTools(ctx, agent)
			require.NoError(t, err)
			names := collectMCPToolNames(tools)
			assert.Equal(t, []string{"tool1", "tool4"}, names)
		})

		t.Run("both filters together", func(t *testing.T) {
			server.ToolFilter = agents.MCPToolFilterStatic{
				AllowedToolNames: []string{"tool1", "tool2", "tool3"},
				BlockedToolNames: []string{"tool3"},
			}
			tools, err := server.ListTools(ctx, agent)
			require.NoError(t, err)
			names := collectMCPToolNames(tools)
			assert.Equal(t, []string{"tool1", "tool2"}, names)
		})

		t.Run("no filter", func(t *testing.T) {
			server.ToolFilter = nil
			tools, err := server.ListTools(ctx, agent)
			require.NoError(t, err)
			names := collectMCPToolNames(tools)
			assert.Equal(t, []string{"tool1", "tool2", "tool3", "tool4"}, names)
		})
	})

	t.Run("dynamic tool filtering", func(t *testing.T) {
		server := agentstesting.NewFakeMCPServer(nil, nil, "test_server")
		server.AddTool("foo1", nil)
		server.AddTool("foo2", nil)
		server.AddTool("bar1", nil)
		server.AddTool("bar2", nil)

		ctx := t.Context()
		agent := newTestAgent("")

		syncFilter := func(_ context.Context, _ agents.MCPToolFilterContext, tool *mcp.Tool) (bool, error) {
			return strings.HasPrefix(tool.Name, "foo"), nil
		}
		server.ToolFilter = agents.MCPToolFilterFunc(syncFilter)

		tools, err := server.ListTools(ctx, agent)
		require.NoError(t, err)
		names := collectMCPToolNames(tools)
		assert.Equal(t, []string{"foo1", "foo2"}, names)
	})

	t.Run("dynamic tool filtering context handling", func(t *testing.T) {
		server := agentstesting.NewFakeMCPServer(nil, nil, "test_server")
		server.AddTool("admin_tool", nil)
		server.AddTool("user_tool", nil)
		server.AddTool("guest_tool", nil)

		t.Run("context-independent filter", func(t *testing.T) {
			contextIndependentFilter := func(_ context.Context, _ agents.MCPToolFilterContext, tool *mcp.Tool) (bool, error) {
				return !strings.HasPrefix(tool.Name, "admin"), nil
			}
			server.ToolFilter = agents.MCPToolFilterFunc(contextIndependentFilter)

			ctx := t.Context()
			agent := newTestAgent("")

			tools, err := server.ListTools(ctx, agent)
			require.NoError(t, err)
			names := collectMCPToolNames(tools)
			assert.Equal(t, []string{"guest_tool", "user_tool"}, names)
		})

		t.Run("context-dependent filter", func(t *testing.T) {
			contextDependentFilter := func(_ context.Context, filterCtx agents.MCPToolFilterContext, tool *mcp.Tool) (bool, error) {
				assert.NotNil(t, filterCtx.Agent)
				assert.Equal(t, "test_server", filterCtx.ServerName)

				// Only admin tools for agents with "admin" in name
				return strings.Contains(strings.ToLower(filterCtx.Agent.Name), "admin") || !strings.HasPrefix(tool.Name, "admin"), nil
			}
			server.ToolFilter = agents.MCPToolFilterFunc(contextDependentFilter)

			ctx := t.Context()

			regularAgent := newTestAgent("regular_user")
			tools, err := server.ListTools(ctx, regularAgent)
			require.NoError(t, err)
			names := collectMCPToolNames(tools)
			assert.Equal(t, []string{"guest_tool", "user_tool"}, names)

			adminAgent := newTestAgent("admin_user")
			tools, err = server.ListTools(ctx, adminAgent)
			require.NoError(t, err)
			names = collectMCPToolNames(tools)
			assert.Equal(t, []string{"admin_tool", "guest_tool", "user_tool"}, names)
		})
	})

	t.Run("dynamic tool filtering error", func(t *testing.T) {
		server := agentstesting.NewFakeMCPServer(nil, nil, "test_server")
		server.AddTool("good_tool", nil)
		server.AddTool("error_tool", nil)
		server.AddTool("another_good_tool", nil)

		errorProneFilter := func(_ context.Context, _ agents.MCPToolFilterContext, tool *mcp.Tool) (bool, error) {
			if tool.Name == "error_tool" {
				return false, errors.New("tool error")
			}
			return true, nil
		}
		server.ToolFilter = agents.MCPToolFilterFunc(errorProneFilter)

		ctx := t.Context()
		agent := newTestAgent("")

		tools, err := server.ListTools(ctx, agent)
		require.NoError(t, err)
		names := collectMCPToolNames(tools)
		assert.Equal(t, []string{"another_good_tool", "good_tool"}, names)
	})

	t.Run("agent dynamic filtering integratin", func(t *testing.T) {
		// Test dynamic filtering integration with Agent methods

		server := agentstesting.NewFakeMCPServer(nil, nil, "")
		server.AddTool("file_read", &jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"path": {Type: "string"}},
		})
		server.AddTool("file_write", &jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"path":    {Type: "string"},
				"content": {Type: "string"},
			},
		})
		server.AddTool("database_query", &jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"query": {Type: "string"}},
		})
		server.AddTool("network_request", &jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"url": {Type: "string"}},
		})

		// Role-based filter for comprehensive testing
		roleBasedFilter := func(_ context.Context, filterCtx agents.MCPToolFilterContext, tool *mcp.Tool) (bool, error) {
			agentName := strings.ToLower(filterCtx.Agent.Name)
			switch {
			case strings.Contains(agentName, "admin"):
				return true, nil
			case strings.Contains(agentName, "readonly"):
				return strings.Contains(tool.Name, "read") || strings.Contains(tool.Name, "query"), nil
			default:
				return strings.HasPrefix(tool.Name, "file_"), nil
			}
		}
		server.ToolFilter = agents.MCPToolFilterFunc(roleBasedFilter)

		ctx := t.Context()

		t.Run("admin agent", func(t *testing.T) {
			agent := agents.New("admin_user").AddMCPServer(server)
			tools, err := agent.GetMCPTools(ctx)
			require.NoError(t, err)
			names := collectAgentsToolNames(tools)
			assert.Equal(t, []string{"database_query", "file_read", "file_write", "network_request"}, names)
		})

		t.Run("readonly agent", func(t *testing.T) {
			agent := agents.New("readonly_viewer").AddMCPServer(server)
			tools, err := agent.GetMCPTools(ctx)
			require.NoError(t, err)
			names := collectAgentsToolNames(tools)
			assert.Equal(t, []string{"database_query", "file_read"}, names)
		})

		t.Run("regular agent", func(t *testing.T) {
			agent := agents.New("regular_user").AddMCPServer(server)
			tools, err := agent.GetMCPTools(ctx)
			require.NoError(t, err)
			names := collectAgentsToolNames(tools)
			assert.Equal(t, []string{"file_read", "file_write"}, names)
		})

		t.Run("GetAllTools", func(t *testing.T) {
			agent := agents.New("regular_user").AddMCPServer(server).AddTool(agentstesting.GetFunctionTool("foo", ""))
			tools, err := agent.GetAllTools(ctx)
			require.NoError(t, err)
			names := collectAgentsToolNames(tools)
			assert.Equal(t, []string{"file_read", "file_write", "foo"}, names)
		})
	})
}

func TestCreateMCPStaticToolFilter(t *testing.T) {
	t.Run("nil values", func(t *testing.T) {
		result, ok := agents.CreateMCPStaticToolFilter(nil, nil)
		assert.False(t, ok)
		assert.Equal(t, agents.MCPToolFilterStatic{}, result)
	})

	t.Run("empty values", func(t *testing.T) {
		result, ok := agents.CreateMCPStaticToolFilter([]string{}, []string{})
		assert.False(t, ok)
		assert.Equal(t, agents.MCPToolFilterStatic{}, result)
	})

	t.Run("allowed tools only", func(t *testing.T) {
		result, ok := agents.CreateMCPStaticToolFilter([]string{"foo", "bar"}, nil)
		assert.True(t, ok)
		assert.Equal(t, agents.MCPToolFilterStatic{
			AllowedToolNames: []string{"foo", "bar"},
			BlockedToolNames: nil,
		}, result)
	})

	t.Run("blocked tools only", func(t *testing.T) {
		result, ok := agents.CreateMCPStaticToolFilter(nil, []string{"baz", "quux"})
		assert.True(t, ok)
		assert.Equal(t, agents.MCPToolFilterStatic{
			AllowedToolNames: nil,
			BlockedToolNames: []string{"baz", "quux"},
		}, result)
	})

	t.Run("allowed and blocked tools", func(t *testing.T) {
		result, ok := agents.CreateMCPStaticToolFilter([]string{"foo", "bar"}, []string{"baz", "quux"})
		assert.True(t, ok)
		assert.Equal(t, agents.MCPToolFilterStatic{
			AllowedToolNames: []string{"foo", "bar"},
			BlockedToolNames: []string{"baz", "quux"},
		}, result)
	})
}
