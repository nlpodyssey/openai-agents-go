package visualization_test

import (
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agents/extensions/visualization"
	"github.com/stretchr/testify/assert"
)

var testingAgent = agents.New("Agent1").
	WithTools(
		agents.FunctionTool{Name: "Tool1"},
		agents.FunctionTool{Name: "Tool2"},
	).
	WithAgentHandoffs(
		agents.New("Handoff1"),
	)

func TestGetMainGraph(t *testing.T) {
	result := visualization.GetMainGraph(testingAgent)
	items := []string{
		`digraph G`,
		`graph [splines=true];`,
		`node [fontname="Arial"];`,
		`edge [penwidth=1.5];`,
		`"__start__" [label="__start__", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];`,
		`"__end__" [label="__end__", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];`,
		`"Agent1" [label="Agent1", shape=box, style=filled, fillcolor=lightyellow, width=1.5, height=0.8];`,
		`"Tool1" [label="Tool1", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];`,
		`"Tool2" [label="Tool2", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];`,
		`"Handoff1" [label="Handoff1", shape=box, style=filled, style=rounded, fillcolor=lightyellow, width=1.5, height=0.8];`,
	}
	for _, item := range items {
		assert.Contains(t, result, item)
	}
}

func TestGetAllNodes(t *testing.T) {
	result := visualization.GetAllNodes(testingAgent)
	items := []string{
		`"__start__" [label="__start__", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];`,
		`"__end__" [label="__end__", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];`,
		`"Agent1" [label="Agent1", shape=box, style=filled, fillcolor=lightyellow, width=1.5, height=0.8];`,
		`"Tool1" [label="Tool1", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];`,
		`"Tool2" [label="Tool2", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];`,
		`"Handoff1" [label="Handoff1", shape=box, style=filled, style=rounded, fillcolor=lightyellow, width=1.5, height=0.8];`,
	}
	for _, item := range items {
		assert.Contains(t, result, item)
	}
}

func TestGetAllEdges(t *testing.T) {
	result := visualization.GetAllEdges(testingAgent)
	items := []string{
		`"__start__" -> "Agent1";`,
		`"Agent1" -> "__end__";`,
		`"Agent1" -> "Tool1" [style=dotted, penwidth=1.5];`,
		`"Tool1" -> "Agent1" [style=dotted, penwidth=1.5];`,
		`"Agent1" -> "Tool2" [style=dotted, penwidth=1.5];`,
		`"Tool2" -> "Agent1" [style=dotted, penwidth=1.5];`,
		`"Agent1" -> "Handoff1";`,
	}
	for _, item := range items {
		assert.Contains(t, result, item)
	}
}

func TestCycleDetection(t *testing.T) {
	agentA := agents.New("A")
	agentB := agents.New("B")
	agentA.AgentHandoffs = []*agents.Agent{agentB}
	agentB.AgentHandoffs = []*agents.Agent{agentA}

	nodes := visualization.GetAllNodes(agentA)
	assert.Equal(t, 1, strings.Count(nodes, `"A" [label="A"`))
	assert.Equal(t, 1, strings.Count(nodes, `"B" [label="B"`))

	edges := visualization.GetAllEdges(agentA)
	assert.Equal(t, 1, strings.Count(edges, `"A" -> "B"`))
	assert.Equal(t, 1, strings.Count(edges, `"B" -> "A"`))
}
