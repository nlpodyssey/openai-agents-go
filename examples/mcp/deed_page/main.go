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
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

const deedPageURL = "https://deed.page/mcp"
const userAgent = "openai-agents-go/deed-page-example"

// Identify this example on every MCP request, including discovery and tool calls.
type userAgentTransport struct{ base http.RoundTripper }

func (t userAgentTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set("User-Agent", userAgent)
	return t.base.RoundTrip(req)
}

func newServer(endpoint string, transport http.RoundTripper) *agents.MCPServerStreamableHTTP {
	return agents.NewMCPServerStreamableHTTP(agents.MCPServerStreamableHTTPParams{
		Name:           "deed.page",
		URL:            endpoint,
		CacheToolsList: true,
		// Only the publishing tools are exposed to the model.
		ToolFilter: agents.MCPToolFilterStatic{AllowedToolNames: []string{"publish_site", "get_site"}},
		TransportOpts: &mcp.StreamableClientTransport{
			HTTPClient: &http.Client{Transport: userAgentTransport{base: transport}, Timeout: 60 * time.Second},
		},
	})
}

func newAgent(server agents.MCPServer, ttlSeconds int) *agents.Agent {
	instructions := "Write one self-contained HTML page (inline CSS, no external scripts) that answers the request, " +
		"then publish it with publish_site using files {\"index.html\": <page>}. " +
		"Reply with the url from the result. Never repeat the token from the result."
	if ttlSeconds > 0 {
		instructions += fmt.Sprintf(" Pass ttl_seconds=%d so the page expires.", ttlSeconds)
	}
	return agents.New("Page publisher").
		WithInstructions(instructions).
		AddMCPServer(server).
		WithModel("gpt-4o")
}

func main() {
	direct := flag.Bool("direct", false, "Publish a sample page, check it and delete it without a model")
	prompt := flag.String("prompt", "A one-page cheat sheet for Go's context package", "What the page should contain")
	ttl := flag.Int("ttl", 600, "Seconds until the published page expires (0 keeps it)")
	flag.Parse()

	// Tracing is disabled so direct mode needs no OpenAI credentials either.
	tracing.SetTracingDisabled(true)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	server := newServer(deedPageURL, http.DefaultTransport)
	err := server.Run(ctx, func(ctx context.Context, server *agents.MCPServerWithClientSession) error {
		if *direct {
			return runDirect(ctx, server, *ttl, http.DefaultClient)
		}
		result, err := agents.Run(ctx, newAgent(server, *ttl), *prompt)
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

type publishResult struct {
	OK    bool   `json:"ok"`
	URL   string `json:"url"`
	Slug  string `json:"slug"`
	Token string `json:"token"`
	Error string `json:"error"`
}

// callTool calls a tool on the MCP session and decodes the JSON text it returns.
func callTool(ctx context.Context, server agents.MCPServer, name string, args map[string]any, out any) error {
	result, err := server.CallTool(ctx, name, args)
	if err != nil {
		return err
	}
	for _, c := range result.Content {
		if text, ok := c.(*mcp.TextContent); ok {
			if result.IsError {
				return fmt.Errorf("%s: %s", name, text.Text)
			}
			return json.Unmarshal([]byte(text.Text), out)
		}
	}
	return fmt.Errorf("%s returned no text content", name)
}

// runDirect publishes a sample page, fetches it, then deletes it with the token
// from the publish result. The token is kept in memory and never printed.
func runDirect(ctx context.Context, server agents.MCPServer, ttlSeconds int, client *http.Client) error {
	page := "<!doctype html><meta charset=utf-8><title>openai-agents-go</title>" +
		"<h1>Published from openai-agents-go</h1><p>This page was published over MCP and is deleted right away.</p>"
	args := map[string]any{"files": map[string]any{"index.html": page}}
	if ttlSeconds > 0 {
		args["ttl_seconds"] = ttlSeconds
	}
	var pub publishResult
	if err := callTool(ctx, server, "publish_site", args, &pub); err != nil {
		return err
	}
	if !pub.OK || pub.URL == "" {
		return fmt.Errorf("publish_site failed: %s", pub.Error)
	}
	fmt.Println("published:", pub.URL)

	status, body, err := get(ctx, client, pub.URL)
	if err != nil {
		return err
	}
	fmt.Println("GET", pub.URL, "->", status)
	if status != http.StatusOK || !strings.Contains(body, "Published from openai-agents-go") {
		return fmt.Errorf("unexpected page: HTTP %d", status)
	}

	var del struct {
		OK bool `json:"ok"`
	}
	if err := callTool(ctx, server, "delete_site", map[string]any{"slug": pub.Slug, "token": pub.Token}, &del); err != nil {
		return err
	}
	if !del.OK {
		return errors.New("delete_site did not confirm the deletion")
	}
	status, _, err = get(ctx, client, pub.URL)
	if err != nil {
		return err
	}
	fmt.Println("deleted; GET", pub.URL, "->", status)
	return nil
}

func get(ctx context.Context, client *http.Client, url string) (int, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, "", err
	}
	return resp.StatusCode, string(body), nil
}
