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
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"slices"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/nlpodyssey/openai-agents-go/util"
	"github.com/openai/openai-go/v3/packages/param"
)

// MCPToolFilterContext provides context information available to tool filter functions.
type MCPToolFilterContext struct {
	// The agent that is requesting the tool list.
	Agent *Agent

	// The name of the MCP server.
	ServerName string
}

type MCPToolFilter interface {
	// FilterMCPTool determines whether a tool should be available (true) or
	// filtered out (false).
	FilterMCPTool(context.Context, MCPToolFilterContext, *mcp.Tool) (bool, error)
}

type MCPToolFilterFunc func(context.Context, MCPToolFilterContext, *mcp.Tool) (bool, error)

func (f MCPToolFilterFunc) FilterMCPTool(ctx context.Context, filterCtx MCPToolFilterContext, t *mcp.Tool) (bool, error) {
	return f(ctx, filterCtx, t)
}

// MCPToolFilterStatic is a static tool filter configuration using allowlists and blocklists.
type MCPToolFilterStatic struct {
	// Optional list of tool names to allow (whitelist).
	// If set (not nil), only these tools will be available.
	AllowedToolNames []string

	// Optional list of tool names to exclude (blacklist).
	// If set (not nil), these tools will be filtered out.
	BlockedToolNames []string
}

func (f MCPToolFilterStatic) FilterMCPTool(_ context.Context, _ MCPToolFilterContext, t *mcp.Tool) (bool, error) {
	return (f.AllowedToolNames == nil || slices.Contains(f.AllowedToolNames, t.Name)) &&
			(f.BlockedToolNames == nil || !slices.Contains(f.BlockedToolNames, t.Name)),
		nil
}

// CreateMCPStaticToolFilter creates a static tool filter from allowlist and blocklist parameters.
// This is a convenience function for creating a MCPToolFilterStatic.
// It returns a MCPToolFilterStatic if any filtering is specified, None otherwise.
func CreateMCPStaticToolFilter(allowedToolNames, blockedToolNames []string) (MCPToolFilterStatic, bool) {
	if len(allowedToolNames) == 0 && len(blockedToolNames) == 0 {
		return MCPToolFilterStatic{}, false
	}
	return MCPToolFilterStatic{
		AllowedToolNames: allowedToolNames,
		BlockedToolNames: blockedToolNames,
	}, true
}

type mcpUtil struct{}

// MCPUtil provides a set of utilities for interop between MCP and Agents SDK tools.
func MCPUtil() mcpUtil { return mcpUtil{} }

// GetAllFunctionTools returns all function tools from a list of MCP servers.
func (u mcpUtil) GetAllFunctionTools(
	ctx context.Context,
	servers []MCPServer,
	convertSchemasToStrict bool,
	agent *Agent,
) ([]Tool, error) {
	var tools []Tool
	toolNames := make(map[string]struct{})
	for _, server := range servers {
		serverTools, err := u.GetFunctionTools(ctx, server, convertSchemasToStrict, agent)
		if err != nil {
			return nil, err
		}

		serverToolNames := make(map[string]struct{}, len(serverTools))
		for _, serverTool := range serverTools {
			toolName := serverTool.ToolName()
			if _, ok := toolNames[toolName]; ok {
				return nil, UserErrorf("duplicate tool name found across MCP servers: %q", toolName)
			}
			serverToolNames[toolName] = struct{}{}
		}

		maps.Copy(toolNames, serverToolNames)
		tools = append(tools, serverTools...)
	}
	return tools, nil
}

// GetFunctionTools returns all function tools from a single MCP server.
func (u mcpUtil) GetFunctionTools(
	ctx context.Context,
	server MCPServer,
	convertSchemasToStrict bool,
	agent *Agent,
) ([]Tool, error) {
	var mcpTools []*mcp.Tool
	err := tracing.MCPToolsSpan(
		ctx, tracing.MCPToolsSpanParams{Server: server.Name()},
		func(ctx context.Context, span tracing.Span) error {
			var err error
			mcpTools, err = server.ListTools(ctx, agent)
			if err != nil {
				return err
			}

			toolNames := make([]string, len(mcpTools))
			for i, tool := range mcpTools {
				toolNames[i] = tool.Name
			}
			span.SpanData().(*tracing.MCPListToolsSpanData).Result = toolNames
			return nil
		})
	if err != nil {
		return nil, err
	}

	functionTools := make([]Tool, len(mcpTools))
	for i, mcpTool := range mcpTools {
		funcTool, err := u.ToFunctionTool(mcpTool, server, convertSchemasToStrict)
		if err != nil {
			return nil, err
		}
		functionTools[i] = funcTool
	}
	return functionTools, nil
}

// ToFunctionTool converts an MCP tool to an Agents SDK function tool.
func (u mcpUtil) ToFunctionTool(
	tool *mcp.Tool,
	server MCPServer,
	convertSchemasToStrict bool,
) (FunctionTool, error) {
	invokeFunc := func(ctx context.Context, arguments string) (any, error) {
		return u.InvokeMCPTool(ctx, server, tool, arguments)
	}

	schema, err := util.JSONMap(tool.InputSchema)
	if err != nil {
		return FunctionTool{}, fmt.Errorf("failed to convert MCP tool input schema to map: %w", err)
	}

	if schema == nil {
		schema = make(map[string]any)
	}

	// MCP spec doesn't require the inputSchema to have `properties`, but OpenAI spec does.
	if _, ok := schema["properties"]; !ok {
		schema["properties"] = map[string]any{}
	}

	isStrict := false
	if convertSchemasToStrict {
		converted, err := EnsureStrictJSONSchema(schema)
		if err != nil {
			Logger().Info("Error converting MCP schema to strict mode", slog.String("error", err.Error()))
		} else {
			schema = converted
			isStrict = true
		}
	}

	return FunctionTool{
		Name:             tool.Name,
		Description:      tool.Description,
		ParamsJSONSchema: schema,
		OnInvokeTool:     invokeFunc,
		StrictJSONSchema: param.NewOpt(isStrict),
	}, nil
}

// InvokeMCPTool invokes an MCP tool and returns the result as a string.
func (mcpUtil) InvokeMCPTool(
	ctx context.Context,
	server MCPServer,
	tool *mcp.Tool,
	jsonInput string,
) (string, error) {
	var jsonData map[string]any
	if jsonInput != "" {
		err := json.Unmarshal([]byte(jsonInput), &jsonData)
		if err != nil {
			if DontLogToolData {
				Logger().Debug("Invalid JSON input", slog.String("toolName", tool.Name))
			} else {
				Logger().Debug("Invalid JSON input",
					slog.String("toolName", tool.Name),
					slog.String("jsonInput", jsonInput))
			}
			return "", ModelBehaviorErrorf("invalid JSON input for tool %s - %s: %w",
				tool.Name, jsonInput, err)
		}
	}

	if DontLogToolData {
		Logger().Debug("Invoking MCP tool", slog.String("toolName", tool.Name))
	} else {
		Logger().Debug("Invoking MCP tool",
			slog.String("toolName", tool.Name),
			slog.String("input", jsonInput))
	}

	result, err := server.CallTool(ctx, tool.Name, jsonData)
	if err != nil {
		Logger().Error("Error invoking MCP tool",
			slog.String("toolName", tool.Name),
			slog.String("error", err.Error()))
		return "", AgentsErrorf("error invoking MCP tool %s: %w", tool.Name, err)
	}

	if DontLogToolData {
		Logger().Debug("MCP tool completed", slog.String("toolName", tool.Name))
	} else {
		Logger().Debug("MCP tool completed",
			slog.String("toolName", tool.Name),
			slog.Any("result", *result))
	}

	var toolOutput string

	// If structured content is requested and available, use it exclusively
	if server.UseStructuredContent() && result.StructuredContent != nil {
		b, err := json.Marshal(result.StructuredContent)
		if err != nil {
			return "", fmt.Errorf("failed to JSON-marshal result structured content of MCP tool %s: %w", tool.Name, err)
		}
		toolOutput = string(b)
	} else {
		// Fall back to regular text content processing
		// The MCP tool result is a list of content items, whereas OpenAI tool
		// outputs are a single string. We'll try to convert.
		if len(result.Content) == 1 {
			b, err := json.Marshal(result.Content[0])
			if err != nil {
				return "", fmt.Errorf("failed to JSON-marshal result content of MCP tool %s: %w", tool.Name, err)
			}
			toolOutput = string(b)
		} else if len(result.Content) > 1 {
			b, err := json.Marshal(result.Content)
			if err != nil {
				return "", fmt.Errorf("failed to JSON-marshal result content of MCP tool %s: %w", tool.Name, err)
			}
			toolOutput = string(b)
		} else {
			// Empty content is a valid result (e.g., "no results found")
			toolOutput = "[]"
		}
	}

	currentSpan := tracing.GetCurrentSpan(ctx)
	if currentSpan != nil {
		if spanData, ok := currentSpan.SpanData().(*tracing.FunctionSpanData); ok {
			spanData.Output = toolOutput
			spanData.MCPData = map[string]any{"server": server.Name()}
		} else {
			Logger().Warn(fmt.Sprintf("Current span is not a FunctionSpanData, skipping tool output: %#v", currentSpan))
		}
	}

	return toolOutput, nil
}

// ApplyMCPToolFilter applies the tool filter to the list of tools.
func ApplyMCPToolFilter(
	ctx context.Context,
	filterContext MCPToolFilterContext,
	toolFilter MCPToolFilter,
	tools []*mcp.Tool,
	agent *Agent,
) []*mcp.Tool {
	if toolFilter == nil {
		return tools
	}

	var filteredTools []*mcp.Tool
	for _, tool := range tools {
		shouldInclude, err := toolFilter.FilterMCPTool(ctx, filterContext, tool)
		if err != nil {
			Logger().Error("Error applying tool filter",
				slog.String("toolName", tool.Name),
				slog.String("serverName", filterContext.ServerName),
				slog.String("error", err.Error()),
			)
			// On error, exclude the tool for safety
			continue
		}
		if shouldInclude {
			filteredTools = append(filteredTools, tool)
		}
	}

	return filteredTools
}
