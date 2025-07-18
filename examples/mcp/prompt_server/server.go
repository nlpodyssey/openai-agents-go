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
	"cmp"
	"context"
	"fmt"
	"net/http"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func runServer(addr string) {
	server := mcp.NewServer(&mcp.Implementation{Name: "Prompt Server"}, nil)

	server.AddPrompt(
		&mcp.Prompt{
			Name:        "generate_code_review_instructions",
			Description: "Generate agent instructions for code review tasks",
			Arguments: []*mcp.PromptArgument{
				{Name: "focus", Required: false},
				{Name: "language", Required: false},
			},
		},
		func(ctx context.Context, session *mcp.ServerSession, params *mcp.GetPromptParams) (*mcp.GetPromptResult, error) {
			focus := cmp.Or(params.Arguments["focus"], "general code quality")
			language := cmp.Or(params.Arguments["language"], "python")
			fmt.Printf("[debug-server] generate_code_review_instructions(%q, %q)\n", focus, language)

			text := `You are a senior ` + language + ` code review specialist. Your role is to provide comprehensive code analysis with focus on ` + focus + `.

INSTRUCTIONS:
- Analyze code for quality, security, performance, and best practices
- Provide specific, actionable feedback with examples
- Identify potential bugs, vulnerabilities, and optimization opportunities
- Suggest improvements with code examples when applicable
- Be constructive and educational in your feedback
- Focus particularly on ` + focus + ` aspects

RESPONSE FORMAT:
1. Overall Assessment
2. Specific Issues Found
3. Security Considerations
4. Performance Notes
5. Recommended Improvements
6. Best Practices Suggestions

Use the available tools to check current time if you need timestamps for your analysis.`

			return &mcp.GetPromptResult{Messages: []*mcp.PromptMessage{{Content: &mcp.TextContent{Text: text}}}}, nil
		},
	)

	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server {
		return server
	}, nil)

	fmt.Printf("Starting Simple Prompt Server at %s ...\n", addr)
	err := http.ListenAndServe(addr, handler)
	if err != nil {
		panic(err)
	}
}
