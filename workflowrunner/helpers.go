package workflowrunner

import "github.com/nlpodyssey/openai-agents-go/agents"

func displayAgentName(agent *agents.Agent) string {
	if agent == nil {
		return ""
	}
	return agent.Name
}
