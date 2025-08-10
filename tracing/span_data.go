// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tracing

import (
	"fmt"

	"github.com/openai/openai-go/responses"
)

// SpanData represents span data in the trace.
type SpanData interface {
	// The Type of the span.
	Type() string

	// Export the span data as a map.
	Export() map[string]any
}

// AgentSpanData represents an Agent Span in the trace.
// Includes name, handoffs, tools, and output type.
type AgentSpanData struct {
	// Mandatory name.
	Name string
	// Optional handoffs.
	Handoffs []string
	// Optional tools.
	Tools []string
	// Optional output type.
	OutputType string
}

func (AgentSpanData) Type() string { return "agent" }

func (sd AgentSpanData) Export() map[string]any {
	var outputType any
	if sd.OutputType != "" {
		outputType = sd.OutputType
	}
	return map[string]any{
		"type":        sd.Type(),
		"name":        sd.Name,
		"handoffs":    sd.Handoffs,
		"tools":       sd.Tools,
		"output_type": outputType,
	}
}

// FunctionSpanData represents a Function Span in the trace.
// Includes input, output and MCP data (if applicable).
type FunctionSpanData struct {
	// Mandatory name.
	Name string
	// Optional input.
	Input string
	// Optional output.
	Output any
	// Optional MCP data.
	MCPData map[string]any
}

func (FunctionSpanData) Type() string { return "function" }

func (sd FunctionSpanData) Export() map[string]any {
	var input any
	if sd.Input != "" {
		input = sd.Input
	}
	var output any
	if sd.Output != nil {
		output = fmt.Sprintf("%+v", sd.Output)
	}
	return map[string]any{
		"type":     sd.Type(),
		"name":     sd.Name,
		"input":    input,
		"output":   output,
		"mcp_data": sd.MCPData,
	}
}

// GenerationSpanData represents a Generation Span in the trace.
// Includes input, output, model, model configuration, and usage.
type GenerationSpanData struct {
	// Optional input.
	Input []map[string]any
	// Optional output.
	Output []map[string]any
	// Optional model.
	Model string
	// Optional model configuration.
	ModelConfig map[string]any
	// Optional usage.
	Usage map[string]any
}

func (GenerationSpanData) Type() string { return "generation" }

func (sd GenerationSpanData) Export() map[string]any {
	var model any
	if sd.Model != "" {
		model = sd.Model
	}
	return map[string]any{
		"type":         sd.Type(),
		"input":        sd.Input,
		"output":       sd.Output,
		"model":        model,
		"model_config": sd.ModelConfig,
		"usage":        sd.Usage,
	}
}

// ResponseSpanData represents a Response Span in the trace.
// Includes response, input, request model information, and tool definitions.
type ResponseSpanData struct {
	// Optional response.
	Response *responses.Response

	// Optional input.
	// This is not used by the OpenAI trace processors, but is useful for
	// other tracing processor implementations.
	Input any

	// Optional request model name.
	// This is useful for tracing processors to track what model was requested,
	// especially when the response model might be different or missing.
	Model string

	// Optional tool definitions available during this request.
	// This should be a serializable representation of tool schemas.
	Tools []map[string]interface{}
}

func (ResponseSpanData) Type() string { return "response" }

func (sd ResponseSpanData) Export() map[string]any {
	var responseID any
	if sd.Response != nil {
		responseID = sd.Response.ID
	}
	var model any
	if sd.Model != "" {
		model = sd.Model
	}
	return map[string]any{
		"type":        sd.Type(),
		"response_id": responseID,
		"model":       model,
		"tools":       sd.Tools,
	}
}

// HandoffSpanData represents a Handoff Span in the trace.
// Includes source and destination agents.
type HandoffSpanData struct {
	// Optional source agent.
	FromAgent string
	// Optional destination agent.
	ToAgent string
}

func (HandoffSpanData) Type() string { return "handoff" }

func (sd HandoffSpanData) Export() map[string]any {
	var fromAgent any
	if sd.FromAgent != "" {
		fromAgent = sd.FromAgent
	}
	var toAgent any
	if sd.ToAgent != "" {
		toAgent = sd.ToAgent
	}
	return map[string]any{
		"type":       sd.Type(),
		"from_agent": fromAgent,
		"to_agent":   toAgent,
	}
}

// CustomSpanData represents a Custom Span in the trace.
// Includes name and data property bag.
type CustomSpanData struct {
	Name string
	Data map[string]any
}

func (CustomSpanData) Type() string { return "custom" }

func (sd CustomSpanData) Export() map[string]any {
	return map[string]any{
		"type": sd.Type(),
		"name": sd.Name,
		"data": sd.Data,
	}
}

// GuardrailSpanData represents a Guardrail Span in the trace.
// Includes name and triggered status.
type GuardrailSpanData struct {
	// Mandatory name.
	Name string
	// Optional triggered flag (default: false)
	Triggered bool
}

func (GuardrailSpanData) Type() string { return "guardrail" }

func (sd GuardrailSpanData) Export() map[string]any {
	return map[string]any{
		"type":      sd.Type(),
		"name":      sd.Name,
		"triggered": sd.Triggered,
	}
}

// TranscriptionSpanData represents a Transcription Span in the trace.
// Includes input, output, model, and model configuration.
type TranscriptionSpanData struct {
	// Optional input.
	Input string
	// Optional input format.
	InputFormat string
	// Optional output.
	Output string
	// Optional model.
	Model string
	// Optional model configuration.
	ModelConfig map[string]any
}

func (TranscriptionSpanData) Type() string { return "transcription" }

func (sd TranscriptionSpanData) Export() map[string]any {
	var inputFormat any
	if sd.InputFormat != "" {
		inputFormat = sd.InputFormat
	}
	var output any
	if sd.Output != "" {
		output = sd.Output
	}
	var model any
	if sd.Model != "" {
		model = sd.Model
	}
	return map[string]any{
		"type": sd.Type(),
		"input": map[string]any{
			"data":   sd.Input,
			"format": inputFormat,
		},
		"output":       output,
		"model":        model,
		"model_config": sd.ModelConfig,
	}
}

// SpeechSpanData represents a Speech Span in the trace.
// Includes input, output, model, model configuration, and first content timestamp.
type SpeechSpanData struct {
	// Optional input.
	Input string
	// Optional output.
	Output string
	// Optional output format.
	OutputFormat string
	// Optional model.
	Model string
	// Optional model configuration.
	ModelConfig map[string]any
	// Optional first content timestamp.
	FirstContentAt string
}

func (SpeechSpanData) Type() string { return "speech" }

func (sd SpeechSpanData) Export() map[string]any {
	var input any
	if sd.Input != "" {
		input = sd.Input
	}
	var outputFormat any
	if sd.OutputFormat != "" {
		outputFormat = sd.OutputFormat
	}
	var model any
	if sd.Model != "" {
		model = sd.Model
	}
	var firstContentAt any
	if sd.FirstContentAt != "" {
		firstContentAt = sd.FirstContentAt
	}
	return map[string]any{
		"type":  sd.Type(),
		"input": input,
		"output": map[string]any{
			"data":   sd.Output,
			"format": outputFormat,
		},
		"model":            model,
		"model_config":     sd.ModelConfig,
		"first_content_at": firstContentAt,
	}
}

// SpeechGroupSpanData represents a Speech Group Span in the trace.
type SpeechGroupSpanData struct {
	// Optional input.
	Input string
}

func (SpeechGroupSpanData) Type() string { return "speech_group" }

func (sd SpeechGroupSpanData) Export() map[string]any {
	var input any
	if sd.Input != "" {
		input = sd.Input
	}
	return map[string]any{
		"type":  sd.Type(),
		"input": input,
	}
}

// MCPListToolsSpanData represents an MCP List Tools Span in the trace.
// Includes server and result.
type MCPListToolsSpanData struct {
	// Optional server.
	Server string
	// Optional result.
	Result []string
}

func (MCPListToolsSpanData) Type() string { return "mcp_tools" }

func (sd MCPListToolsSpanData) Export() map[string]any {
	var server any
	if sd.Server != "" {
		server = sd.Server
	}
	return map[string]any{
		"type":   sd.Type(),
		"server": server,
		"result": sd.Result,
	}
}
