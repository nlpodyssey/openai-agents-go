package main

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
	"github.com/nlpodyssey/openai-agents-go/agents"
)

// Generate a plan of searches to ground the financial analysis.
// For a given financial question or company, we want to search for
// recent news, official filings, analyst commentary, and other
// relevant background.

const PlannerPrompt = "You are a financial research planner. Given a request for financial analysis, " +
	"produce a set of web searches to gather the context needed. Aim for recent " +
	"headlines, earnings calls or 10‑K snippets, analyst commentary, and industry background. " +
	"Output between 5 and 15 search terms to query for."

type FinancialSearchItem struct {
	Reason string `json:"reason" jsonschema_description:"Your reasoning for why this search is relevant."`
	Query  string `json:"query" jsonschema_description:"The search term to feed into a web (or file) search."`
}

type FinancialSearchPlan struct {
	Searches []FinancialSearchItem `json:"searches" jsonschema_description:"A list of searches to perform."`
}

type FinancialSearchPlanSchema struct{}

func (FinancialSearchPlanSchema) Name() string             { return "FinancialSearchPlan" }
func (FinancialSearchPlanSchema) IsPlainText() bool        { return false }
func (FinancialSearchPlanSchema) IsStrictJSONSchema() bool { return true }
func (FinancialSearchPlanSchema) JSONSchema() map[string]any {
	reflector := &jsonschema.Reflector{ExpandedStruct: true}
	schema := reflector.Reflect(FinancialSearchPlan{})
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
func (FinancialSearchPlanSchema) ValidateJSON(jsonStr string) (any, error) {
	var v FinancialSearchPlan
	err := json.Unmarshal([]byte(jsonStr), &v)
	return v, err
}

var PlannerAgent = agents.New("FinancialPlannerAgent").
	WithInstructions(PlannerPrompt).
	WithOutputSchema(FinancialSearchPlanSchema{}).
	WithModel("o3-mini")
