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
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strconv"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func runServer(addr string) {
	server := mcp.NewServer(&mcp.Implementation{Name: "Agents MCP SSE Example"}, nil)

	type addParams struct {
		A int `json:"a"`
		B int `json:"b"`
	}

	mcp.AddTool(
		server, &mcp.Tool{Name: "add", Description: "Add two numbers"},
		func(_ context.Context, _ *mcp.ServerSession, params *mcp.CallToolParamsFor[addParams]) (*mcp.CallToolResultFor[int], error) {
			fmt.Printf("[debug-server] add(%d, %d)\n", params.Arguments.A, params.Arguments.B)
			result := params.Arguments.A + params.Arguments.B
			return &mcp.CallToolResultFor[int]{
				Content: []mcp.Content{&mcp.TextContent{Text: strconv.Itoa(result)}},
			}, nil
		},
	)

	mcp.AddTool(
		server, &mcp.Tool{Name: "get_secret_word"},
		func(_ context.Context, _ *mcp.ServerSession, _ *mcp.CallToolParamsFor[struct{}]) (*mcp.CallToolResultFor[string], error) {
			fmt.Println("[debug-server] get_secret_word()")
			choice := []string{"apple", "banana", "cherry"}[rand.Intn(3)]
			return &mcp.CallToolResultFor[string]{
				Content: []mcp.Content{&mcp.TextContent{Text: choice}},
			}, nil
		},
	)

	type getCurrentWeatherParams struct {
		City string `json:"city"`
	}

	mcp.AddTool(
		server, &mcp.Tool{Name: "get_current_weather"},
		func(ctx context.Context, _ *mcp.ServerSession, params *mcp.CallToolParamsFor[getCurrentWeatherParams]) (*mcp.CallToolResultFor[string], error) {
			fmt.Printf("[debug-server] get_current_weather(%q)\n", params.Arguments.City)

			resp, err := http.Get("https://wttr.in/" + params.Arguments.City)
			if err != nil {
				return nil, fmt.Errorf("HTTP request to wttr.in error: %w", err)
			}
			defer func() {
				_ = resp.Body.Close()
			}()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("failed to read wttr.in body response: %w", err)
			}

			return &mcp.CallToolResultFor[string]{
				Content: []mcp.Content{&mcp.TextContent{Text: string(body)}},
			}, nil
		},
	)

	handler := mcp.NewSSEHandler(func(request *http.Request) *mcp.Server {
		return server
	})

	fmt.Printf("Starting SSE server at %s ...\n", addr)
	err := http.ListenAndServe(addr, handler)
	if err != nil {
		panic(err)
	}
}
