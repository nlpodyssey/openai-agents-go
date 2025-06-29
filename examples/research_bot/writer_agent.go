package main

import (
	"github.com/nlpodyssey/openai-agents-go/agents"
)

// Agent used to synthesize a final report from the individual summaries.

const WriterAgentPrompt = "You are a senior researcher tasked with writing a cohesive report for a research query. " +
	"You will be provided with the original query, and some initial research done by a research " +
	"assistant.\n" +
	"You should first come up with an outline for the report that describes the structure and " +
	"flow of the report. Then, generate the report and return that as your final output.\n" +
	"The final output should be in markdown format, and it should be lengthy and detailed. Aim " +
	"for 5-10 pages of content, at least 1000 words."

type ReportData struct {
	ShortSummary      string   `json:"short_summary" jsonschema_description:"A short 2-3 sentence summary of the findings."`
	MarkdownReport    string   `json:"markdown_report" jsonschema_description:"The final report."`
	FollowUpQuestions []string `json:"follow_up_questions" jsonschema_description:"Suggested topics to research further."`
}

var WriterAgent = agents.New("WriterAgent").
	WithInstructions(WriterAgentPrompt).
	WithOutputType(agents.OutputType[ReportData]()).
	WithModel("o3-mini")
