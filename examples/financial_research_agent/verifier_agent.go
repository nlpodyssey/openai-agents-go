package main

import (
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

var VerifierAgent = agents.New("VerificationAgent").
	WithInstructions(VerifierPrompt).
	WithOutputType(agents.OutputType[VerificationResult]()).
	WithModel("gpt-4o")
