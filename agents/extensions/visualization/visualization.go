package visualization

import (
	"fmt"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

// GetMainGraph generates the main graph structure in DOT format for the given agent.
func GetMainGraph(agent *agents.Agent) string {
	var sb strings.Builder
	sb.WriteString(`digraph G {
   graph [splines=true];
   node [fontname="Arial"];
   edge [penwidth=1.5];
`)
	sb.WriteString(GetAllNodes(agent))
	sb.WriteString(GetAllEdges(agent))
	sb.WriteString("}\n")
	return sb.String()
}

// GetAllNodes recursively generates the nodes for the given agent and its handoffs in DOT format.
func GetAllNodes(agent *agents.Agent) string {
	return getAllNodes(agent, nil, make(map[string]struct{}))
}

func getAllNodes(agent, parent *agents.Agent, visited map[string]struct{}) string {
	if _, ok := visited[agent.Name]; ok {
		return ""
	}
	visited[agent.Name] = struct{}{}

	var sb strings.Builder

	// Start and end the graph
	if parent == nil {
		sb.WriteString(`"__start__" [label="__start__", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];
"__end__" [label="__end__", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];
`)
		// Ensure parent agent node is colored
		_, _ = fmt.Fprintf(&sb,
			"\"%s\" [label=\"%s\", shape=box, style=filled, fillcolor=lightyellow, width=1.5, height=0.8];\n",
			agent.Name, agent.Name,
		)
	}

	for _, tool := range agent.Tools {
		_, _ = fmt.Fprintf(&sb,
			"\"%s\" [label=\"%s\", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];\n",
			tool.ToolName(), tool.ToolName(),
		)
	}

	for _, handoff := range agent.Handoffs {
		_, _ = fmt.Fprintf(&sb,
			"\"%s\" [label=\"%s\", shape=box, style=filled, style=rounded, fillcolor=lightyellow, width=1.5, height=0.8];\n",
			handoff.AgentName, handoff.AgentName,
		)
	}

	for _, handoff := range agent.AgentHandoffs {
		if _, ok := visited[handoff.Name]; !ok {
			_, _ = fmt.Fprintf(&sb,
				"\"%s\" [label=\"%s\", shape=box, style=filled, style=rounded, fillcolor=lightyellow, width=1.5, height=0.8];\n",
				handoff.Name, handoff.Name,
			)
		}
		sb.WriteString(getAllNodes(handoff, agent, visited))
	}

	return sb.String()
}

// GetAllEdges recursively generates the edges for the given agent and its handoffs in DOT format.
func GetAllEdges(agent *agents.Agent) string {
	return getAllEdges(agent, nil, make(map[string]struct{}))
}

func getAllEdges(agent, parent *agents.Agent, visited map[string]struct{}) string {
	if _, ok := visited[agent.Name]; ok {
		return ""
	}
	visited[agent.Name] = struct{}{}

	var sb strings.Builder

	if parent == nil {
		_, _ = fmt.Fprintf(&sb, "\"__start__\" -> \"%s\";\n", agent.Name)
	}

	for _, tool := range agent.Tools {
		_, _ = fmt.Fprintf(&sb,
			"\"%s\" -> \"%s\" [style=dotted, penwidth=1.5];\n",
			agent.Name, tool.ToolName(),
		)
		_, _ = fmt.Fprintf(&sb,
			"\"%s\" -> \"%s\" [style=dotted, penwidth=1.5];\n",
			tool.ToolName(), agent.Name,
		)
	}

	for _, handoff := range agent.Handoffs {
		_, _ = fmt.Fprintf(&sb, "\"%s\" -> \"%s\";\n", agent.Name, handoff.AgentName)
	}

	for _, handoff := range agent.AgentHandoffs {
		_, _ = fmt.Fprintf(&sb, "\"%s\" -> \"%s\";\n", agent.Name, handoff.Name)
		sb.WriteString(getAllEdges(handoff, agent, visited))
	}

	if len(agent.Handoffs) == 0 {
		_, _ = fmt.Fprintf(&sb, "\"%s\" -> \"__end__\";\n", agent.Name)
	}

	return sb.String()
}
