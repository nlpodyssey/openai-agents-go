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
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/google/uuid"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

const parallelURL = "https://search.parallel.ai/mcp"
const userAgent = "openai-agents-go/parallel-search-example"

// Identify this example on every MCP request, including discovery and tool calls.
type userAgentTransport struct{ base http.RoundTripper }

func (t userAgentTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set("User-Agent", userAgent)
	return t.base.RoundTrip(req)
}

func newServer(endpoint string, transport http.RoundTripper) *agents.MCPServerStreamableHTTP {
	return agents.NewMCPServerStreamableHTTP(agents.MCPServerStreamableHTTPParams{
		Name:           "Parallel Search",
		URL:            endpoint,
		CacheToolsList: true,
		TransportOpts: &mcp.StreamableClientTransport{
			HTTPClient: &http.Client{Transport: userAgentTransport{base: transport}, Timeout: 60 * time.Second},
		},
	})
}

func newAgent(server agents.MCPServer, sessionID string) *agents.Agent {
	return agents.New("Web researcher").
		WithInstructions("Use web_search to research the question and web_fetch to read relevant sources. Cite source URLs in your answer. Use this session_id on all tool calls: " + sessionID).
		AddMCPServer(server).
		WithModel("gpt-4o")
}

func main() {
	direct := flag.Bool("direct", false, "Call search and fetch through SDK function tools without a model")
	query := flag.String("query", "What is the OpenAI Agents Go SDK?", "Research question")
	url := flag.String("url", "https://github.com/nlpodyssey/openai-agents-go", "URL to fetch in direct mode")
	flag.Parse()

	// Tracing is disabled so direct mode needs no OpenAI credentials either.
	tracing.SetTracingDisabled(true)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	server := newServer(parallelURL, http.DefaultTransport)
	sessionID := uuid.NewString()
	err := server.Run(ctx, func(ctx context.Context, server *agents.MCPServerWithClientSession) error {
		agent := newAgent(server, sessionID)
		if *direct {
			return runDirect(ctx, agent, server, *query, *url, sessionID)
		}
		result, err := agents.Run(ctx, agent, *query)
		if err != nil {
			return err
		}
		fmt.Println(result.FinalOutput)
		return nil
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runDirect(ctx context.Context, agent *agents.Agent, server agents.MCPServer, query, url, sessionID string) error {
	// Use the same MCP-to-function-tool conversion used by the agent runner.
	tools, err := agents.MCPUtil().GetFunctionTools(ctx, server, false, agent)
	if err != nil {
		return err
	}
	for _, call := range []struct {
		name string
		args map[string]any
	}{
		{"web_search", map[string]any{"objective": query, "search_queries": []string{query}, "session_id": sessionID}},
		{"web_fetch", map[string]any{"urls": []string{url}, "session_id": sessionID}},
	} {
		var found bool
		for _, tool := range tools {
			if tool.ToolName() != call.name {
				continue
			}
			found = true
			args, err := json.Marshal(call.args)
			if err != nil {
				return err
			}
			output, err := tool.(agents.FunctionTool).OnInvokeTool(ctx, string(args))
			if err != nil {
				return err
			}
			fmt.Printf("%s:\n%v\n", call.name, output)
		}
		if !found {
			return fmt.Errorf("server did not advertise %s", call.name)
		}
	}
	return nil
}
