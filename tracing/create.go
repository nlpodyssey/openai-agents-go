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
	"cmp"
	"context"

	"github.com/openai/openai-go/v3/responses"
)

type TraceParams struct {
	// The name of the logical app or workflow.
	// For example, you might provide "code_bot" for a coding agent,
	// or "customer_support_agent" for a customer support agent.
	WorkflowName string

	// The ID of the trace. Optional. If not provided, we will generate an ID.
	// We recommend using GenTraceID to generate a trace ID, to guarantee that
	// IDs are correctly formatted.
	TraceID string

	// Optional grouping identifier to link multiple traces from the same
	// conversation or process. For instance, you might use a chat thread ID.
	GroupID string

	// Optional dictionary of additional metadata to attach to the trace.
	Metadata map[string]any

	// If true, we will return a Trace but the Trace will not be recorded.
	Disabled bool
}

// NewTrace creates a new trace.
//
// The trace will not be started automatically; you should either use RunTrace,
// Trace.Run, or call Trace.Start and Trace.Finish manually.
//
// In addition to the workflow name and optional grouping identifier, you can provide
// an arbitrary metadata dictionary to attach additional user-defined information to
// the trace.
func NewTrace(ctx context.Context, params TraceParams) Trace {
	currentTrace := GetTraceProvider().GetCurrentTrace(ctx)
	if currentTrace != nil {
		Logger().Warn("Trace already exists. Creating a new trace, but this is probably a mistake.")
	}
	return GetTraceProvider().CreateTrace(
		params.WorkflowName,
		params.TraceID,
		params.GroupID,
		params.Metadata,
		params.Disabled,
	)
}

func RunTrace(ctx context.Context, params TraceParams, fn func(context.Context, Trace) error) error {
	return NewTrace(ctx, params).Run(ctx, fn)
}

// GetCurrentTrace returns the currently active trace, if present.
func GetCurrentTrace(ctx context.Context) Trace {
	return GetTraceProvider().GetCurrentTrace(ctx)
}

// GetCurrentSpan returns the currently active span, if present.
func GetCurrentSpan(ctx context.Context) Span {
	return GetTraceProvider().GetCurrentSpan(ctx)
}

type AgentSpanParams struct {
	// The name of the agent.
	Name string
	// Optional list of agent names to which this agent could hand off control.
	Handoffs []string
	// Optional list of tool names available to this agent.
	Tools []string
	// Optional name of the output type produced by the agent.
	OutputType string
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewAgentSpan creates a new agent span.
//
// The span will not be started automatically, you should either use AgentSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewAgentSpan(ctx context.Context, params AgentSpanParams) Span {
	spanData := &AgentSpanData{
		Name:       params.Name,
		Handoffs:   params.Handoffs,
		Tools:      params.Tools,
		OutputType: params.OutputType,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func AgentSpan(ctx context.Context, params AgentSpanParams, fn func(context.Context, Span) error) error {
	return NewAgentSpan(ctx, params).Run(ctx, fn)
}

type FunctionSpanParams struct {
	// The name of the function.
	Name string
	// The input to the function.
	Input string
	// The output of the function.
	Output string
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewFunctionSpan create a new function span.
//
// The span will not be started automatically, you should either use FunctionSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewFunctionSpan(ctx context.Context, params FunctionSpanParams) Span {
	spanData := &FunctionSpanData{
		Name:   params.Name,
		Input:  params.Input,
		Output: params.Output,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func FunctionSpan(ctx context.Context, params FunctionSpanParams, fn func(context.Context, Span) error) error {
	return NewFunctionSpan(ctx, params).Run(ctx, fn)
}

type GenerationSpanParams struct {
	// The sequence of input messages sent to the model.
	Input []map[string]any
	// The sequence of output messages received from the model.
	Output []map[string]any
	// The model identifier used for the generation.
	Model string
	// The model configuration (hyperparameters) used.
	ModelConfig map[string]any
	// A map of usage information (input tokens, output tokens, etc.).
	Usage map[string]any
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewGenerationSpan creates a new generation span.
//
// The span will not be started automatically, you should either use GenerationSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
//
// This span captures the details of a model generation, including the
// input message sequence, any generated outputs, the model name and
// configuration, and usage data. If you only need to capture a model
// response identifier, use NewResponseSpan instead.
func NewGenerationSpan(ctx context.Context, params GenerationSpanParams) Span {
	spanData := &GenerationSpanData{
		Input:       params.Input,
		Output:      params.Output,
		Model:       params.Model,
		ModelConfig: params.ModelConfig,
		Usage:       params.Usage,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func GenerationSpan(ctx context.Context, params GenerationSpanParams, fn func(context.Context, Span) error) error {
	return NewGenerationSpan(ctx, params).Run(ctx, fn)
}

type ResponseSpanParams struct {
	// The OpenAI Response object.
	Response *responses.Response
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewResponseSpan creates a new response span.
//
// The span will not be started automatically, you should either use ResponseSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewResponseSpan(ctx context.Context, params ResponseSpanParams) Span {
	spanData := &ResponseSpanData{
		Response: params.Response,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func ResponseSpan(ctx context.Context, params ResponseSpanParams, fn func(context.Context, Span) error) error {
	return NewResponseSpan(ctx, params).Run(ctx, fn)
}

type HandoffSpanParams struct {
	// The name of the agent that is handing off.
	FromAgent string
	// The name of the agent that is receiving the handoff.
	ToAgent string
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewHandoffSpan creates a new handoff span.
//
// The span will not be started automatically, you should either use HandoffSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewHandoffSpan(ctx context.Context, params HandoffSpanParams) Span {
	spanData := &HandoffSpanData{
		FromAgent: params.FromAgent,
		ToAgent:   params.ToAgent,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func HandoffSpan(ctx context.Context, params HandoffSpanParams, fn func(context.Context, Span) error) error {
	return NewHandoffSpan(ctx, params).Run(ctx, fn)
}

type CustomSpanParams struct {
	// The name of the custom span.
	Name string
	// Arbitrary structured data to associate with the span.
	Data map[string]any
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewCustomSpan creates a new custom span, to which you can add your own metadata.
//
// The span will not be started automatically, you should either use CustomSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewCustomSpan(ctx context.Context, params CustomSpanParams) Span {
	spanData := &CustomSpanData{
		Name: params.Name,
		Data: params.Data,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func CustomSpan(ctx context.Context, params CustomSpanParams, fn func(context.Context, Span) error) error {
	return NewCustomSpan(ctx, params).Run(ctx, fn)
}

type GuardrailSpanParams struct {
	// The name of the guardrail.
	Name string
	// Whether the guardrail was triggered.
	Triggered bool
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewGuardrailSpan creates a new guardrail span.
//
// The span will not be started automatically, you should either use GuardrailSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewGuardrailSpan(ctx context.Context, params GuardrailSpanParams) Span {
	spanData := &GuardrailSpanData{
		Name:      params.Name,
		Triggered: params.Triggered,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func GuardrailSpan(ctx context.Context, params GuardrailSpanParams, fn func(context.Context, Span) error) error {
	return NewGuardrailSpan(ctx, params).Run(ctx, fn)
}

type TranscriptionSpanParams struct {
	// The name of the model used for the speech-to-text.
	Model string
	// The audio input of the speech-to-text transcription, as a base64 encoded string of audio bytes.
	Input string
	// The format of the audio input (defaults to "pcm").
	InputFormat string
	// The output of the speech-to-text transcription.
	Output string
	// The model configuration (hyperparameters) used.
	ModelConfig map[string]any
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewTranscriptionSpan creates a new transcription span.
//
// The span will not be started automatically, you should either use TranscriptionSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewTranscriptionSpan(ctx context.Context, params TranscriptionSpanParams) Span {
	spanData := &TranscriptionSpanData{
		Input:       params.Input,
		InputFormat: cmp.Or(params.InputFormat, "pcm"),
		Output:      params.Output,
		Model:       params.Model,
		ModelConfig: params.ModelConfig,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func TranscriptionSpan(ctx context.Context, params TranscriptionSpanParams, fn func(context.Context, Span) error) error {
	return NewTranscriptionSpan(ctx, params).Run(ctx, fn)
}

type SpeechSpanParams struct {
	// The name of the model used for the text-to-speech.
	Model string
	// The text input of the text-to-speech.
	Input string
	// The audio output of the text-to-speech as base64 encoded string of PCM audio bytes.
	Output string
	// The format of the audio output (defaults to "pcm").
	OutputFormat string
	// The model configuration (hyperparameters) used.
	ModelConfig map[string]any
	// The time of the first byte of the audio output.
	FirstContentAt string
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewSpeechSpan creates a new speech span.
//
// The span will not be started automatically, you should either use SpeechSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewSpeechSpan(ctx context.Context, params SpeechSpanParams) Span {
	spanData := &SpeechSpanData{
		Input:          params.Input,
		Output:         params.Output,
		OutputFormat:   cmp.Or(params.OutputFormat, "pcm"),
		Model:          params.Model,
		ModelConfig:    params.ModelConfig,
		FirstContentAt: params.FirstContentAt,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func SpeechSpan(ctx context.Context, params SpeechSpanParams, fn func(context.Context, Span) error) error {
	return NewSpeechSpan(ctx, params).Run(ctx, fn)
}

type SpeechGroupSpanParams struct {
	// The input text used for the speech request.
	Input string
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewSpeechGroupSpan creates a new speech group span.
//
// The span will not be started automatically, you should either use SpeechGroupSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewSpeechGroupSpan(ctx context.Context, params SpeechGroupSpanParams) Span {
	spanData := &SpeechGroupSpanData{
		Input: params.Input,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func SpeechGroupSpan(ctx context.Context, params SpeechGroupSpanParams, fn func(context.Context, Span) error) error {
	return NewSpeechGroupSpan(ctx, params).Run(ctx, fn)
}

type MCPToolsSpanParams struct {
	// The name of the MCP server.
	Server string
	// The result of the MCP list tools call.
	Result []string
	// The ID of the span. Optional. If not provided, we will generate an ID.
	// We recommend using GenSpanID to generate a span ID, to guarantee that
	// IDs are correctly formatted.
	SpanID string
	// The parent span or trace. If not provided, we will automatically use
	// the current trace/span as the parent.
	Parent any
	// If true, we will return a Span but the Span will not be recorded.
	Disabled bool
}

// NewMCPToolsSpan creates a new MCP list tools span.
//
// The span will not be started automatically, you should either use MCPToolsSpan,
// Span.Run, or call Span.Start and Span.Finish manually.
func NewMCPToolsSpan(ctx context.Context, params MCPToolsSpanParams) Span {
	spanData := &MCPListToolsSpanData{
		Server: params.Server,
		Result: params.Result,
	}
	return GetTraceProvider().CreateSpan(
		ctx,
		spanData,
		params.SpanID,
		params.Parent,
		params.Disabled,
	)
}

func MCPToolsSpan(ctx context.Context, params MCPToolsSpanParams, fn func(context.Context, Span) error) error {
	return NewMCPToolsSpan(ctx, params).Run(ctx, fn)
}
