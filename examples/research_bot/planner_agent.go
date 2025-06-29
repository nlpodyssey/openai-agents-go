package main

import (
	"github.com/nlpodyssey/openai-agents-go/agents"
)

const PlannerAgentPrompt = "You are a helpful research assistant. Given a query, come up with a set of web searches " +
	"to perform to best answer the query. Output between 5 and 20 terms to query for."

type WebSearchItem struct {
	Reason string `json:"reason" jsonschema_description:"Your reasoning for why this search is important to the query."`
	Query  string `json:"query" jsonschema_description:"The search term to use for the web search."`
}

type WebSearchPlan struct {
	Searches []WebSearchItem `json:"searches" jsonschema_description:"A list of web searches to perform to best answer the query."`
}

var PlannerAgent = agents.New("PlannerAgent").
	WithInstructions(PlannerAgentPrompt).
	WithOutputType(agents.OutputType[WebSearchPlan]()).
	WithModel("gpt-4o")
