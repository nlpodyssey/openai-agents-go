package main

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
	"github.com/nlpodyssey/openai-agents-go/agents"
)

// Writer agent brings together the raw search results and optionally calls out
// to sub‑analyst tools for specialized commentary, then returns a cohesive markdown report.

const WriterPrompt = "You are a senior financial analyst. You will be provided with the original query and " +
	"a set of raw search summaries. Your task is to synthesize these into a long‑form markdown " +
	"report (at least several paragraphs) including a short executive summary and follow‑up " +
	"questions. If needed, you can call the available analysis tools (e.g. fundamentals_analysis, " +
	"risk_analysis) to get short specialist write‑ups to incorporate."

type FinancialReportData struct {
	ShortSummary      string   `json:"short_summary" jsonschema_description:"A short 2‑3 sentence executive summary."`
	MarkdownReport    string   `json:"markdown_report" jsonschema_description:"The full markdown report."`
	FollowUpQuestions []string `json:"follow_up_questions" jsonschema_description:"Suggested follow‑up questions for further research."`
}

type FinancialReportDataSchema struct{}

func (FinancialReportDataSchema) Name() string             { return "FinancialReportData" }
func (FinancialReportDataSchema) IsPlainText() bool        { return false }
func (FinancialReportDataSchema) IsStrictJSONSchema() bool { return true }
func (FinancialReportDataSchema) JSONSchema() map[string]any {
	reflector := &jsonschema.Reflector{ExpandedStruct: true}
	schema := reflector.Reflect(FinancialReportData{})
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
func (FinancialReportDataSchema) ValidateJSON(jsonStr string) (any, error) {
	var v FinancialReportData
	err := json.Unmarshal([]byte(jsonStr), &v)
	return v, err
}

// Note: We will attach handoffs to specialist analyst agents at runtime in the manager.
// This shows how an agent can use handoffs to delegate to specialized subagents.

var WriterAgent = agents.New("FinancialWriterAgent").
	WithInstructions(WriterPrompt).
	WithOutputSchema(FinancialReportDataSchema{}).
	WithModel("gpt-4.5-preview-2025-02-27")
