package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/stretchr/testify/require"
)

func TestCommandUsesExplicitCheckout(t *testing.T) {
	repo := t.TempDir()
	cmd, err := semCommand(t.Context(), "sem", repo)
	require.NoError(t, err)
	require.Equal(t, repo, cmd.Dir)
	require.Equal(t, []string{"sem", "mcp"}, cmd.Args)
	require.Contains(t, cmd.Env, "SEM_REPO="+repo)
	_, err = semCommand(t.Context(), "sem", "")
	require.Error(t, err)
	file := filepath.Join(repo, "file")
	require.NoError(t, os.WriteFile(file, nil, 0600))
	_, err = semCommand(t.Context(), "sem", file)
	require.Error(t, err)
}

func TestDirectUsesSDKToolsAndHandlesMissingTool(t *testing.T) {
	tracing.SetTracingDisabled(true)
	defer tracing.SetTracingDisabled(false)
	server := agentstesting.NewFakeMCPServer(nil, agents.MCPToolFilterStatic{
		AllowedToolNames: []string{"sem_find", "sem_grep"},
	}, "sem-fixture")
	agent := newAgent(server, "gpt-4o")
	_, err := runDirect(t.Context(), agent, server, "parse_config")
	require.ErrorContains(t, err, "did not advertise sem_find")
	server.AddTool("sem_find", &jsonschema.Schema{Type: "object"})
	server.AddTool("sem_validate", &jsonschema.Schema{Type: "object"})
	output, err := runDirect(t.Context(), agent, server, "parse_config")
	require.NoError(t, err)
	require.Equal(t, []string{"sem_find"}, server.ToolCalls)
	require.Contains(t, output, "parse_config")
	require.Contains(t, server.ToolResults[0], `"query":"parse_config"`)
	require.Contains(t, server.ToolResults[0], `"token_budget":1200`)
	tools, err := server.ListTools(t.Context(), agent)
	require.NoError(t, err)
	require.Len(t, tools, 1)
}
