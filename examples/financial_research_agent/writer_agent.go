package main

import (
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

// Note: We will attach handoffs to specialist analyst agents at runtime in the manager.
// This shows how an agent can use handoffs to delegate to specialized subagents.

var WriterAgent = agents.New("FinancialWriterAgent").
	WithInstructions(WriterPrompt).
	WithOutputType(agents.OutputType[FinancialReportData]()).
	WithModel("gpt-4.1")
