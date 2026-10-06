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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/stretchr/testify/require"
)

func TestSearchAndFetchThroughAgent(t *testing.T) {
	tracing.SetTracingDisabled(true)
	defer tracing.SetTracingDisabled(false)
	server := mcp.NewServer(&mcp.Implementation{Name: "search-fixture"}, nil)
	var mu sync.Mutex
	var methods []string
	var calls []string
	type searchArgs struct {
		Objective     string   `json:"objective"`
		SearchQueries []string `json:"search_queries"`
		SessionID     string   `json:"session_id"`
	}
	type fetchArgs struct {
		URLs      []string `json:"urls"`
		SessionID string   `json:"session_id"`
	}
	mcp.AddTool(server, &mcp.Tool{Name: "web_search"}, func(ctx context.Context, req *mcp.CallToolRequest, args searchArgs) (*mcp.CallToolResult, any, error) {
		require.Equal(t, "Go SDK MCP", args.Objective)
		require.Equal(t, []string{"Go SDK MCP"}, args.SearchQueries)
		require.Equal(t, "test-conversation", args.SessionID)
		mu.Lock()
		calls = append(calls, "web_search")
		mu.Unlock()
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "Source: https://example.org/sdk"}}}, nil, nil
	})
	mcp.AddTool(server, &mcp.Tool{Name: "web_fetch"}, func(ctx context.Context, req *mcp.CallToolRequest, args fetchArgs) (*mcp.CallToolResult, any, error) {
		require.Equal(t, []string{"https://example.org/sdk"}, args.URLs)
		require.Equal(t, "test-conversation", args.SessionID)
		mu.Lock()
		calls = append(calls, "web_fetch")
		mu.Unlock()
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "SDK supports Streamable HTTP."}}}, nil, nil
	})
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, userAgent, r.UserAgent())
		require.Empty(t, r.Header.Get("Authorization"))
		if r.Method == http.MethodPost {
			body, err := io.ReadAll(r.Body)
			require.NoError(t, err)
			r.Body = io.NopCloser(strings.NewReader(string(body)))
			var msg struct {
				Method string `json:"method"`
			}
			require.NoError(t, json.Unmarshal(body, &msg))
			mu.Lock()
			methods = append(methods, msg.Method)
			mu.Unlock()
		}
		handler.ServeHTTP(w, r)
	}))
	defer httpServer.Close()
	client := newServer(httpServer.URL, http.DefaultTransport)
	err := client.Run(t.Context(), func(ctx context.Context, client *agents.MCPServerWithClientSession) error {
		model := agentstesting.NewFakeModel(false, nil)
		model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
			{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("web_search", `{"objective":"Go SDK MCP","search_queries":["Go SDK MCP"],"session_id":"test-conversation"}`)}},
			{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("web_fetch", `{"urls":["https://example.org/sdk"],"session_id":"test-conversation"}`)}},
			{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Answer: https://example.org/sdk")}},
		})
		agent := newAgent(client, "test-conversation").WithModelInstance(model)
		result, err := agents.Run(ctx, agent, "Go SDK MCP")
		require.NoError(t, err)
		require.Equal(t, "Answer: https://example.org/sdk", result.FinalOutput)
		input, err := json.Marshal(model.LastTurnArgs.Input)
		require.NoError(t, err)
		require.Contains(t, string(input), "SDK supports Streamable HTTP.")
		require.Contains(t, model.LastTurnArgs.SystemInstructions.Value, "test-conversation")
		require.NoError(t, runDirect(ctx, agent, client, "Go SDK MCP", "https://example.org/sdk", "test-conversation"))
		return nil
	})
	require.NoError(t, err)
	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, []string{"web_search", "web_fetch", "web_search", "web_fetch"}, calls)
	require.Contains(t, methods, "initialize")
	require.Contains(t, methods, "tools/list")
	require.Contains(t, methods, "tools/call")
}

func TestCanceledConnection(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	server := newServer("https://example.org/mcp", http.DefaultTransport)
	require.Error(t, server.Connect(ctx))
}
