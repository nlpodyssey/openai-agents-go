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
	"fmt"
	"log/slog"
	"os/exec"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPServer is implemented by Model Context Protocol servers.
type MCPServer interface {
	// Connect to the server.
	//
	// For example, this might mean spawning a subprocess or opening a network connection.
	// The server is expected to remain connected until Cleanup is called.
	Connect(context.Context) error

	// Cleanup the server.
	//
	// For example, this might mean closing a subprocess or closing a network connection.
	Cleanup(context.Context) error

	// Name returns a readable name for the server.
	Name() string

	// UseStructuredContent reports  whether to use a tool result's
	// `StructuredContent` when calling an MCP tool.
	UseStructuredContent() bool

	// ListTools lists the tools available on the server.
	ListTools(context.Context, *Agent) ([]*mcp.Tool, error)

	// CallTool invokes a tool on the server.
	CallTool(ctx context.Context, toolName string, arguments map[string]any) (*mcp.CallToolResult, error)

	// ListPrompts lists the prompts available on the server.
	ListPrompts(ctx context.Context) (*mcp.ListPromptsResult, error)

	// GetPrompt returns a specific prompt from the server.
	GetPrompt(ctx context.Context, name string, arguments map[string]string) (*mcp.GetPromptResult, error)
}

// MCPServerWithClientSession is a base type for MCP servers that uses an
// mcp.ClientSession to communicate with the server.
type MCPServerWithClientSession struct {
	transport            mcp.Transport
	session              *mcp.ClientSession
	cleanupMu            sync.Mutex
	cacheToolsList       bool
	cacheDirty           bool
	toolsList            []*mcp.Tool
	toolFilter           MCPToolFilter
	name                 string
	useStructuredContent bool
}

type MCPServerWithClientSessionParams struct {
	Name      string
	Transport mcp.Transport

	// Whether to cache the tools list. If `true`, the tools list will be
	// cached and only fetched from the server once. If `false`, the tools list will be
	// fetched from the server on each call to `ListTools()`. The cache can be invalidated
	// by calling `InvalidateToolsCache()`. You should set this to `true` if you know the
	// server will not change its tools list, because it can drastically improve latency
	// (by avoiding a round-trip to the server every time).
	CacheToolsList bool

	// The tool filter to use for filtering tools.
	ToolFilter MCPToolFilter

	// Whether to use `StructuredContent` when calling an MCP tool.
	// Defaults to false for backwards compatibility - most MCP servers still include
	// the structured content in `Content`, and using it by default will cause duplicate
	// content. You can set this to true if you know the server will not duplicate
	// the structured content in `Content`.
	UseStructuredContent bool
}

func NewMCPServerWithClientSession(params MCPServerWithClientSessionParams) *MCPServerWithClientSession {
	return &MCPServerWithClientSession{
		transport:      params.Transport,
		cacheToolsList: params.CacheToolsList,
		// The cache is always dirty at startup, so that we fetch tools at least once
		cacheDirty:           true,
		toolFilter:           params.ToolFilter,
		name:                 params.Name,
		useStructuredContent: params.UseStructuredContent,
	}
}

func (s *MCPServerWithClientSession) Connect(ctx context.Context) (err error) {
	defer func() {
		if err != nil {
			Logger().Error("Error initializing MCP server", slog.String("error", err.Error()))
			if e := s.Cleanup(ctx); e != nil {
				err = errors.Join(err, fmt.Errorf("MCP server cleanup error: %w", e))
			}
		}
	}()

	client := mcp.NewClient(&mcp.Implementation{Name: s.name}, nil)
	session, err := client.Connect(ctx, s.transport)
	if err != nil {
		return fmt.Errorf("MCP client connection error: %w", err)
	}
	s.session = session
	return nil
}

func (s *MCPServerWithClientSession) Cleanup(context.Context) error {
	s.cleanupMu.Lock()
	defer func() {
		s.session = nil
		s.cleanupMu.Unlock()
	}()

	if s.session != nil {
		err := s.session.Close()
		if err != nil {
			Logger().Error("Error cleaning up server", slog.String("error", err.Error()))
		}
		return err
	}
	return nil
}

func (s *MCPServerWithClientSession) Name() string {
	return s.name
}

func (s *MCPServerWithClientSession) UseStructuredContent() bool {
	return s.useStructuredContent
}

func (s *MCPServerWithClientSession) ListTools(ctx context.Context, agent *Agent) ([]*mcp.Tool, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}

	var tools []*mcp.Tool
	// Return from cache if caching is enabled, we have tools, and the cache is not dirty
	if s.cacheToolsList && !s.cacheDirty && len(s.toolsList) > 0 {
		tools = s.toolsList
	} else {
		s.cacheDirty = false
		listToolsResults, err := s.session.ListTools(ctx, nil)
		if err != nil {
			return nil, fmt.Errorf("MCP list tools error: %w", err)
		}
		tools = listToolsResults.Tools
		s.toolsList = tools
	}

	filteredTools := tools
	if s.toolFilter != nil {
		if agent == nil {
			return nil, UserErrorf("agent is required for dynamic tool filtering")
		}
		filterContext := MCPToolFilterContext{
			Agent:      agent,
			ServerName: s.name,
		}
		filteredTools = ApplyMCPToolFilter(ctx, filterContext, s.toolFilter, filteredTools, agent)
	}
	return filteredTools, nil
}

func (s *MCPServerWithClientSession) CallTool(ctx context.Context, toolName string, arguments map[string]any) (*mcp.CallToolResult, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	return s.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      toolName,
		Arguments: arguments,
	})
}

func (s *MCPServerWithClientSession) ListPrompts(ctx context.Context) (*mcp.ListPromptsResult, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	return s.session.ListPrompts(ctx, nil)
}

func (s *MCPServerWithClientSession) GetPrompt(ctx context.Context, name string, arguments map[string]string) (*mcp.GetPromptResult, error) {
	if s.session == nil {
		return nil, NewUserError("server not initialized: make sure you call `Connect()` first")
	}
	return s.session.GetPrompt(ctx, &mcp.GetPromptParams{
		Name:      name,
		Arguments: arguments,
	})
}

func (s *MCPServerWithClientSession) Run(ctx context.Context, fn func(context.Context, *MCPServerWithClientSession) error) (err error) {
	err = s.Connect(ctx)
	if err != nil {
		return fmt.Errorf("MCP server connection error: %w", err)
	}
	defer func() {
		if e := s.Cleanup(ctx); e != nil {
			err = errors.Join(err, fmt.Errorf("MCP server cleanup error: %w", e))
		}
	}()
	return fn(ctx, s)
}

// InvalidateToolsCache invalidates the tools cache.
func (s *MCPServerWithClientSession) InvalidateToolsCache() {
	s.cacheDirty = true
}

type MCPServerStdioParams struct {
	// The command to run to start the server.
	Command *exec.Cmd

	// Whether to cache the tools list. If `true`, the tools list will be
	// cached and only fetched from the server once. If `false`, the tools list will be
	// fetched from the server on each call to `ListTools()`. The cache can be
	// invalidated by calling `InvalidateToolsCache()`. You should set this to `true`
	// if you know the server will not change its tools list, because it can drastically
	// improve latency (by avoiding a round-trip to the server every time).
	CacheToolsList bool

	// A readable name for the server. If not provided, we'll create one from the command.
	Name string

	// Optional tool filter to use for filtering tools
	ToolFilter MCPToolFilter

	// Whether to use `StructuredContent` when calling an MCP tool.
	// Defaults to false for backwards compatibility - most MCP servers still include
	// the structured content in `Content`, and using it by default will cause duplicate
	// content. You can set this to true if you know the server will not duplicate
	// the structured content in `Content`.
	UseStructuredContent bool
}

// MCPServerStdio is an MCP server implementation that uses the stdio transport.
//
// See: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
type MCPServerStdio struct {
	*MCPServerWithClientSession
}

// NewMCPServerStdio creates a new MCP server based on the stdio transport.
func NewMCPServerStdio(params MCPServerStdioParams) *MCPServerStdio {
	name := params.Name
	if name == "" {
		name = fmt.Sprintf("stdio: %s", params.Command.Path)
	}

	return &MCPServerStdio{
		MCPServerWithClientSession: NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
			Name:                 name,
			Transport:            mcp.NewCommandTransport(params.Command),
			CacheToolsList:       params.CacheToolsList,
			ToolFilter:           params.ToolFilter,
			UseStructuredContent: params.UseStructuredContent,
		}),
	}
}

type MCPServerSSEParams struct {
	BaseURL       string
	TransportOpts *mcp.SSEClientTransportOptions

	// Whether to cache the tools list. If `true`, the tools list will be
	// cached and only fetched from the server once. If `false`, the tools list will be
	// fetched from the server on each call to `ListTools()`. The cache can be
	// invalidated by calling `InvalidateToolsCache()`. You should set this to `true`
	// if you know the server will not change its tools list, because it can drastically
	// improve latency (by avoiding a round-trip to the server every time).
	CacheToolsList bool

	// A readable name for the server. If not provided, we'll create one from the base URL.
	Name string

	// Optional tool filter to use for filtering tools
	ToolFilter MCPToolFilter

	// Whether to use `StructuredContent` when calling an MCP tool.
	// Defaults to false for backwards compatibility - most MCP servers still include
	// the structured content in `Content`, and using it by default will cause duplicate
	// content. You can set this to true if you know the server will not duplicate
	// the structured content in `Content`.
	UseStructuredContent bool
}

// MCPServerSSE is an MCP server implementation that uses the HTTP with SSE transport.
//
// See: https://modelcontextprotocol.io/specification/2024-11-05/basic/transports#http-with-sse
//
// Deprecated: SSE as a standalone transport is deprecated as of MCP protocol version 2024-11-05.
// It has been replaced by Streamable HTTP, which incorporates SSE as an optional streaming mechanism.
// See: https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse-deprecated
type MCPServerSSE struct {
	*MCPServerWithClientSession
}

// NewMCPServerSSE creates a new MCP server based on the HTTP with SSE transport.
//
// Deprecated: SSE as a standalone transport is deprecated as of MCP protocol version 2024-11-05.
// It has been replaced by Streamable HTTP, which incorporates SSE as an optional streaming mechanism.
// See: https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse-deprecated
func NewMCPServerSSE(params MCPServerSSEParams) *MCPServerSSE {
	name := params.Name
	if name == "" {
		name = fmt.Sprintf("sse: %s", params.BaseURL)
	}

	return &MCPServerSSE{
		MCPServerWithClientSession: NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
			Name:                 name,
			Transport:            mcp.NewSSEClientTransport(params.BaseURL, params.TransportOpts),
			CacheToolsList:       params.CacheToolsList,
			ToolFilter:           params.ToolFilter,
			UseStructuredContent: params.UseStructuredContent,
		}),
	}
}

type MCPServerStreamableHTTPParams struct {
	URL           string
	TransportOpts *mcp.StreamableClientTransportOptions

	// Whether to cache the tools list. If `true`, the tools list will be
	// cached and only fetched from the server once. If `false`, the tools list will be
	// fetched from the server on each call to `ListTools()`. The cache can be
	// invalidated by calling `InvalidateToolsCache()`. You should set this to `true`
	// if you know the server will not change its tools list, because it can drastically
	// improve latency (by avoiding a round-trip to the server every time).
	CacheToolsList bool

	// A readable name for the server. If not provided, we'll create one from the URL.
	Name string

	// Optional tool filter to use for filtering tools
	ToolFilter MCPToolFilter

	// Whether to use `StructuredContent` when calling an MCP tool.
	// Defaults to false for backwards compatibility - most MCP servers still include
	// the structured content in `Content`, and using it by default will cause duplicate
	// content. You can set this to true if you know the server will not duplicate
	// the structured content in `Content`.
	UseStructuredContent bool
}

// MCPServerStreamableHTTP is an MCP server implementation that uses the Streamable HTTP transport.
//
// See: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http
type MCPServerStreamableHTTP struct {
	*MCPServerWithClientSession
}

func NewMCPServerStreamableHTTP(params MCPServerStreamableHTTPParams) *MCPServerStreamableHTTP {
	name := params.Name
	if name == "" {
		name = fmt.Sprintf("streamable_http: %s", params.URL)
	}

	return &MCPServerStreamableHTTP{
		MCPServerWithClientSession: NewMCPServerWithClientSession(MCPServerWithClientSessionParams{
			Name:                 name,
			Transport:            mcp.NewStreamableClientTransport(params.URL, params.TransportOpts),
			CacheToolsList:       params.CacheToolsList,
			ToolFilter:           params.ToolFilter,
			UseStructuredContent: params.UseStructuredContent,
		}),
	}
}
