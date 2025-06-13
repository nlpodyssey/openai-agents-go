package main

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
	"github.com/nlpodyssey/openai-agents-go/agents"
)

// Agent to sanity‑check a synthesized report for consistency and recall.
// This can be used to flag potential gaps or obvious mistakes.

const VerifierPrompt = "You are a meticulous auditor. You have been handed a financial analysis report. " +
	"Your job is to verify the report is internally consistent, clearly sourced, and makes " +
	"no unsupported claims. Point out any issues or uncertainties."

type VerificationResult struct {
	Verified bool   `json:"verified" jsonschema_description:"Whether the report seems coherent and plausible."`
	Issues   string `json:"issues" jsonschema_description:"If not verified, describe the main issues or concerns."`
}

type VerificationResultSchema struct{}

func (VerificationResultSchema) Name() string             { return "VerificationResult" }
func (VerificationResultSchema) IsPlainText() bool        { return false }
func (VerificationResultSchema) IsStrictJSONSchema() bool { return true }
func (VerificationResultSchema) JSONSchema() map[string]any {
	reflector := &jsonschema.Reflector{ExpandedStruct: true}
	schema := reflector.Reflect(VerificationResult{})
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
func (VerificationResultSchema) ValidateJSON(jsonStr string) (any, error) {
	var v VerificationResult
	err := json.Unmarshal([]byte(jsonStr), &v)
	return v, err
}

var VerifierAgent = agents.New("VerificationAgent").
	WithInstructions(VerifierPrompt).
	WithOutputSchema(VerificationResultSchema{}).
	WithModel("gpt-4o")
