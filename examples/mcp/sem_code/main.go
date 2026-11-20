package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

func semCommand(ctx context.Context, executable, repo string) (*exec.Cmd, error) {
	if repo == "" {
		return nil, fmt.Errorf("-repo must name an existing Git checkout")
	}
	abs, err := filepath.Abs(repo)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("repository path is not a directory")
	}
	cmd := exec.CommandContext(ctx, executable, "mcp")
	cmd.Dir = abs
	cmd.Env = append(cmd.Environ(), "SEM_REPO="+abs)
	return cmd, nil
}

func newAgent(server agents.MCPServer, model string) *agents.Agent {
	return agents.New("Code navigator").
		WithInstructions("Use sem_find for symbols and sem_grep for source text. Request context as needed; distinguish observed references from unresolved relationships. Do not claim tests passed from reading source alone.").
		AddMCPServer(server).WithModel(model)
}

func runDirect(ctx context.Context, agent *agents.Agent, server agents.MCPServer, query string) (any, error) {
	tools, err := agents.MCPUtil().GetFunctionTools(ctx, server, false, agent)
	if err != nil {
		return nil, err
	}
	for _, tool := range tools {
		if tool.ToolName() == "sem_find" {
			args, err := json.Marshal(map[string]any{"query": query, "mode": "context", "token_budget": 1200})
			if err != nil {
				return nil, err
			}
			return tool.(agents.FunctionTool).OnInvokeTool(ctx, string(args))
		}
	}
	return nil, fmt.Errorf("server did not advertise sem_find")
}

func main() {
	repo := flag.String("repo", "", "Existing local Git checkout (required)")
	query := flag.String("query", "", "Symbol name in direct mode, or question in agent mode (required)")
	executable := flag.String("sem", "sem", "Sem executable")
	direct := flag.Bool("direct", false, "Call sem_find through SDK function tools without model credentials")
	model := flag.String("model", "gpt-4o", "Model for agent mode")
	flag.Parse()
	tracing.SetTracingDisabled(true)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	err := func() error {
		if *query == "" {
			return fmt.Errorf("-query is required")
		}
		cmd, err := semCommand(ctx, *executable, *repo)
		if err != nil {
			return err
		}
		server := agents.NewMCPServerStdio(agents.MCPServerStdioParams{
			Name: "Sem local code", Command: cmd, CacheToolsList: true,
			ToolFilter: agents.MCPToolFilterStatic{AllowedToolNames: []string{"sem_find", "sem_grep"}},
		})
		return server.Run(ctx, func(ctx context.Context, server *agents.MCPServerWithClientSession) error {
			agent := newAgent(server, *model)
			if *direct {
				output, err := runDirect(ctx, agent, server, *query)
				if err != nil {
					return err
				}
				fmt.Println(output)
				return nil
			}
			result, err := agents.Run(ctx, agent, *query)
			if err != nil {
				return err
			}
			fmt.Println(result.FinalOutput)
			return nil
		})
	}()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
