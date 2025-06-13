package main

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
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

type WebSearchPlanSchema struct{}

func (WebSearchPlanSchema) Name() string             { return "WebSearchPlan" }
func (WebSearchPlanSchema) IsPlainText() bool        { return false }
func (WebSearchPlanSchema) IsStrictJSONSchema() bool { return true }
func (WebSearchPlanSchema) JSONSchema() map[string]any {
	reflector := &jsonschema.Reflector{ExpandedStruct: true}
	schema := reflector.Reflect(WebSearchPlan{})
	b, err := json.Marshal(schema)
	if err != nil {
		panic(err) // This should never happen
	}
	var result map[string]any
	err = json.Unmarshal(b, &result)
	if err != nil {
		panic(err) // This should never happen
	}
	return result
}
func (WebSearchPlanSchema) ValidateJSON(jsonStr string) (any, error) {
	var v WebSearchPlan
	err := json.Unmarshal([]byte(jsonStr), &v)
	return v, err
}

var PlannerAgent = agents.New("PlannerAgent").
	WithInstructions(PlannerAgentPrompt).
	WithModel("gpt-4o").
	WithOutputSchema(WebSearchPlanSchema{})
