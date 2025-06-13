package main

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
)

type AnalysisSummary struct {
	Summary string `json:"summary" jsonschema_description:"Short text summary for this aspect of the analysis."`
}

type AnalysisSummarySchema struct{}

func (AnalysisSummarySchema) Name() string             { return "AnalysisSummary" }
func (AnalysisSummarySchema) IsPlainText() bool        { return false }
func (AnalysisSummarySchema) IsStrictJSONSchema() bool { return true }
func (AnalysisSummarySchema) JSONSchema() map[string]any {
	reflector := &jsonschema.Reflector{ExpandedStruct: true}
	schema := reflector.Reflect(AnalysisSummary{})
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
func (AnalysisSummarySchema) ValidateJSON(jsonStr string) (any, error) {
	var v AnalysisSummary
	err := json.Unmarshal([]byte(jsonStr), &v)
	return v, err
}
