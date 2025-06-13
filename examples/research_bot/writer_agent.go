package main

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
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

type ReportDataSchema struct{}

func (ReportDataSchema) Name() string             { return "ReportData" }
func (ReportDataSchema) IsPlainText() bool        { return false }
func (ReportDataSchema) IsStrictJSONSchema() bool { return true }
func (ReportDataSchema) JSONSchema() map[string]any {
	reflector := &jsonschema.Reflector{ExpandedStruct: true}
	schema := reflector.Reflect(ReportData{})
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
func (ReportDataSchema) ValidateJSON(jsonStr string) (any, error) {
	var v ReportData
	err := json.Unmarshal([]byte(jsonStr), &v)
	return v, err
}

var WriterAgent = agents.New("WriterAgent").
	WithInstructions(WriterAgentPrompt).
	WithModel("o3-mini").
	WithOutputSchema(ReportDataSchema{})
