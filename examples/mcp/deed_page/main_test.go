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

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/stretchr/testify/require"
)

type fixture struct {
	mu      sync.Mutex
	calls   []string
	live    bool
	pageURL string
}

func newFixture(t *testing.T) (*fixture, *httptest.Server, *httptest.Server) {
	f := &fixture{}
	pages := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()
		if !f.live {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte("<h1>Published from openai-agents-go</h1>"))
	}))
	f.pageURL = pages.URL + "/"

	server := mcp.NewServer(&mcp.Implementation{Name: "deed-page-fixture"}, nil)
	type publishArgs struct {
		Files      map[string]any `json:"files"`
		TTLSeconds int            `json:"ttl_seconds,omitempty"`
	}
	type siteArgs struct {
		Slug  string `json:"slug"`
		Token string `json:"token,omitempty"`
	}
	text := func(v any) *mcp.CallToolResult {
		b, _ := json.Marshal(v)
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: string(b)}}}
	}
	mcp.AddTool(server, &mcp.Tool{Name: "publish_site"}, func(ctx context.Context, req *mcp.CallToolRequest, args publishArgs) (*mcp.CallToolResult, any, error) {
		require.Contains(t, args.Files, "index.html")
		require.Equal(t, 600, args.TTLSeconds)
		f.mu.Lock()
		f.calls = append(f.calls, "publish_site")
		f.live = true
		f.mu.Unlock()
		return text(map[string]any{"ok": true, "url": f.pageURL, "slug": "test-slug", "token": "deed_tok_test"}), nil, nil
	})
	mcp.AddTool(server, &mcp.Tool{Name: "get_site"}, func(ctx context.Context, req *mcp.CallToolRequest, args siteArgs) (*mcp.CallToolResult, any, error) {
		return text(map[string]any{"ok": true, "slug": args.Slug}), nil, nil
	})
	mcp.AddTool(server, &mcp.Tool{Name: "delete_site"}, func(ctx context.Context, req *mcp.CallToolRequest, args siteArgs) (*mcp.CallToolResult, any, error) {
		require.Equal(t, "test-slug", args.Slug)
		require.Equal(t, "deed_tok_test", args.Token)
		f.mu.Lock()
		f.calls = append(f.calls, "delete_site")
		f.live = false
		f.mu.Unlock()
		return text(map[string]any{"ok": true}), nil, nil
	})
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	mcpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, userAgent, r.UserAgent())
		require.Empty(t, r.Header.Get("Authorization"))
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(func() { mcpServer.Close(); pages.Close() })
	return f, mcpServer, pages
}

func TestAgentPublishesThroughMCP(t *testing.T) {
	tracing.SetTracingDisabled(true)
	defer tracing.SetTracingDisabled(false)
	f, mcpServer, _ := newFixture(t)
	client := newServer(mcpServer.URL, http.DefaultTransport)
	err := client.Run(t.Context(), func(ctx context.Context, client *agents.MCPServerWithClientSession) error {
		agent := newAgent(client, 600)
		tools, err := agents.MCPUtil().GetFunctionTools(ctx, client, false, agent)
		require.NoError(t, err)
		var names []string
		for _, tool := range tools {
			names = append(names, tool.ToolName())
		}
		require.ElementsMatch(t, []string{"publish_site", "get_site"}, names, "delete_site must not reach the model")

		model := agentstesting.NewFakeModel(false, nil)
		model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
			{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("publish_site", `{"files":{"index.html":"<h1>Hi</h1>"},"ttl_seconds":600}`)}},
			{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Published: " + f.pageURL)}},
		})
		result, err := agents.Run(ctx, agent.WithModelInstance(model), "A test page")
		require.NoError(t, err)
		require.Equal(t, "Published: "+f.pageURL, result.FinalOutput)
		require.Contains(t, model.LastTurnArgs.SystemInstructions.Value, "ttl_seconds=600")
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, []string{"publish_site"}, f.calls)
}

func TestDirectPublishCheckDelete(t *testing.T) {
	f, mcpServer, pages := newFixture(t)
	client := newServer(mcpServer.URL, http.DefaultTransport)
	err := client.Run(t.Context(), func(ctx context.Context, client *agents.MCPServerWithClientSession) error {
		return runDirect(ctx, client, 600, pages.Client())
	})
	require.NoError(t, err)
	require.Equal(t, []string{"publish_site", "delete_site"}, f.calls)
	require.False(t, f.live)
}

func TestCanceledConnection(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	server := newServer("https://example.org/mcp", http.DefaultTransport)
	require.Error(t, server.Connect(ctx))
}
