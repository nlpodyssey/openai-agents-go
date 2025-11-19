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
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

// This example demonstrates how to use the hosted MCP support in the OpenAI
// Responses API, with approval callbacks.

func approvalCallback(_ context.Context, req responses.ResponseOutputItemMcpApprovalRequest) (agents.MCPToolApprovalFunctionResult, error) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Printf("Approve running the tool `%s`? (y/n) ", req.Name)
	_ = os.Stdout.Sync()
	r, _, err := reader.ReadRune()
	fmt.Println()
	if err != nil {
		return agents.MCPToolApprovalFunctionResult{}, fmt.Errorf("error reading rune: %w", err)
	}

	if r == 'y' {
		return agents.MCPToolApprovalFunctionResult{Approve: true}, nil
	}
	return agents.MCPToolApprovalFunctionResult{
		Approve: false,
		Reason:  "User denied",
	}, nil
}

func main() {
	verbose := flag.Bool("verbose", false, "")
	stream := flag.Bool("stream", false, "")
	flag.Parse()

	agent := agents.New("Assistant").WithModel("gpt-5-chat-latest").WithTools(
		agents.HostedMCPTool{
			ToolConfig: responses.ToolMcpParam{
				ServerLabel: "gitmcp",
				ServerURL:   param.NewOpt("https://gitmcp.io/openai/codex"),
				RequireApproval: responses.ToolMcpRequireApprovalUnionParam{
					OfMcpToolApprovalSetting: param.NewOpt("always"),
				},
				Type: constant.ValueOf[constant.Mcp](),
			},
			OnApprovalRequest: approvalCallback,
		},
	)

	ctx := context.Background()
	input := "Which language is this repo written in?"
	var newItems []agents.RunItem

	if *stream {
		result, err := agents.RunStreamed(ctx, agent, input)
		if err != nil {
			panic(err)
		}
		err = result.StreamEvents(func(event agents.StreamEvent) error {
			if e, ok := event.(agents.RunItemStreamEvent); ok {
				fmt.Printf("Got event of type %T\n", e.Item)
			}
			return nil
		})
		if err != nil {
			panic(err)
		}
		fmt.Printf("Done streaming; final result: %v\n", result.FinalOutput())
		newItems = result.NewItems()
	} else {
		result, err := agents.Run(ctx, agent, input)
		if err != nil {
			panic(err)
		}
		fmt.Println(result.FinalOutput)
		newItems = result.NewItems
	}

	if *verbose {
		for _, item := range newItems {
			fmt.Println(item)
		}
	}
}
