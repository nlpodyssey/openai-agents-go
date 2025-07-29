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

package langsmith

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

// TracingProcessor implements tracing.Processor to send traces to LangSmith
type TracingProcessor struct {
	client      *http.Client
	apiKey      string
	apiURL      string
	projectName string
	metadata    map[string]any
	tags        []string
	name        string

	// Track runs for parent-child relationships
	runs   map[string]*RunData
	runsMu sync.RWMutex

	// Track first inputs and last outputs for traces
	firstResponseInputs map[string]map[string]any
	lastResponseOutputs map[string]map[string]any
	responsesMu         sync.RWMutex
}

// RunData tracks LangSmith run information
type RunData struct {
	ID          string         `json:"id"`
	StartTime   time.Time      `json:"start_time"`
	ParentRunID *string        `json:"parent_run_id,omitempty"`
	Name        string         `json:"name"`
	RunType     string         `json:"run_type"`
	Inputs      map[string]any `json:"inputs"`
	Outputs     map[string]any `json:"outputs,omitempty"`
	Error       *string        `json:"error,omitempty"`
	EndTime     *time.Time     `json:"end_time,omitempty"`
	Extra       map[string]any `json:"extra,omitempty"`
	Tags        []string       `json:"tags,omitempty"`
	SessionName string         `json:"session_name,omitempty"`
}

// ProcessorParams configuration for the LangSmith processor
type ProcessorParams struct {
	// LangSmith API key. Required - pass from main
	APIKey string
	// LangSmith API URL. Defaults to https://api.smith.langchain.com
	APIURL string
	// LangSmith project name. Defaults to LANGSMITH_PROJECT environment variable
	ProjectName string
	// Optional metadata to attach to all traces
	Metadata map[string]any
	// Optional tags to attach to all traces
	Tags []string
	// Optional name for the root trace
	Name string
	// Optional custom HTTP client
	HTTPClient *http.Client
}

// NewTracingProcessor creates a new LangSmith tracing processor
func NewTracingProcessor(params ProcessorParams) *TracingProcessor {
	apiURL := params.APIURL
	if apiURL == "" {
		apiURL = "https://api.smith.langchain.com"
	}

	projectName := params.ProjectName
	if projectName == "" {
		projectName = os.Getenv("LANGSMITH_PROJECT")
		if projectName == "" {
			projectName = "default"
		}
	}

	client := params.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}

	return &TracingProcessor{
		client:              client,
		apiKey:              params.APIKey, // API key comes from main
		apiURL:              apiURL,
		projectName:         projectName,
		metadata:            params.Metadata,
		tags:                params.Tags,
		name:                params.Name,
		runs:                make(map[string]*RunData),
		firstResponseInputs: make(map[string]map[string]any),
		lastResponseOutputs: make(map[string]map[string]any),
	}
}

// OnTraceStart implements tracing.Processor
func (p *TracingProcessor) OnTraceStart(ctx context.Context, trace tracing.Trace) error {
	if p.apiKey == "" {
		fmt.Fprintf(os.Stderr, "LangSmith API key not set, skipping trace export\n")
		return nil
	}

	runName := p.name
	if runName == "" && trace.Name() != "" {
		runName = trace.Name()
	}
	if runName == "" {
		runName = "Agent workflow"
	}

	traceRunID := uuid.New().String()
	startTime := time.Now().UTC()

	// Build extra metadata
	extra := make(map[string]any)
	if p.metadata != nil {
		for k, v := range p.metadata {
			extra[k] = v
		}
	}

	traceDict := trace.Export()
	if traceDict != nil {
		if groupID, ok := traceDict["group_id"].(string); ok && groupID != "" {
			extra["thread_id"] = groupID
		}
		if metadata, ok := traceDict["metadata"].(map[string]any); ok {
			for k, v := range metadata {
				extra[k] = v
			}
		}
	}

	p.runsMu.Lock()
	p.runs[trace.TraceID()] = &RunData{
		ID:          traceRunID,
		StartTime:   startTime,
		Name:        runName,
		RunType:     "chain",
		Inputs:      make(map[string]any),
		Extra:       extra,
		Tags:        p.tags,
		SessionName: p.projectName,
	}
	p.runsMu.Unlock()

	// Create run data for API request
	runData := map[string]any{
		"id":           traceRunID,
		"name":         runName,
		"run_type":     "chain",
		"inputs":       make(map[string]any),
		"start_time":   startTime.Format(time.RFC3339),
		"session_name": p.projectName,
	}

	if len(p.tags) > 0 {
		runData["tags"] = p.tags
	}

	if len(extra) > 0 {
		runData["extra"] = extra
	}

	return p.createRun(ctx, runData)
}

// OnTraceEnd implements tracing.Processor
func (p *TracingProcessor) OnTraceEnd(ctx context.Context, trace tracing.Trace) error {
	if p.apiKey == "" {
		return nil
	}

	p.runsMu.Lock()
	run, exists := p.runs[trace.TraceID()]
	if exists {
		delete(p.runs, trace.TraceID())
	}
	p.runsMu.Unlock()

	if !exists {
		return nil
	}

	// Get first inputs and last outputs
	p.responsesMu.RLock()
	inputs := p.firstResponseInputs[trace.TraceID()]
	outputs := p.lastResponseOutputs[trace.TraceID()]
	delete(p.firstResponseInputs, trace.TraceID())
	delete(p.lastResponseOutputs, trace.TraceID())
	p.responsesMu.RUnlock()

	if inputs == nil {
		inputs = make(map[string]any)
	}
	if outputs == nil {
		outputs = make(map[string]any)
	}

	updateData := map[string]any{
		"outputs":  outputs,
		"end_time": time.Now().UTC().Format(time.RFC3339),
	}

	// Update inputs if we have them
	if len(inputs) > 0 {
		updateData["inputs"] = inputs
	}

	return p.updateRun(ctx, run.ID, updateData)
}

// OnSpanStart implements tracing.Processor
func (p *TracingProcessor) OnSpanStart(ctx context.Context, span tracing.Span) error {
	if p.apiKey == "" {
		return nil
	}

	// Find parent run
	p.runsMu.RLock()
	var parentRun *RunData
	if span.ParentID() != "" {
		parentRun = p.runs[span.ParentID()]
	}
	if parentRun == nil {
		parentRun = p.runs[span.TraceID()]
	}
	p.runsMu.RUnlock()

	if parentRun == nil {
		fmt.Fprintf(os.Stderr, "No trace info found for span, skipping: %s\n", span.SpanID())
		return nil
	}

	spanRunID := uuid.New().String()
	spanStartTime := span.StartedAt()
	if spanStartTime.IsZero() {
		spanStartTime = time.Now().UTC()
	}

	runName := p.getRunName(span)
	runType := p.getRunType(span)
	extracted := p.extractSpanData(span)

	p.runsMu.Lock()
	p.runs[span.SpanID()] = &RunData{
		ID:          spanRunID,
		StartTime:   spanStartTime,
		ParentRunID: &parentRun.ID,
		Name:        runName,
		RunType:     runType,
		SessionName: p.projectName,
	}
	p.runsMu.Unlock()

	runData := map[string]any{
		"id":            spanRunID,
		"name":          runName,
		"run_type":      runType,
		"inputs":        extracted["inputs"],
		"start_time":    spanStartTime.Format(time.RFC3339),
		"parent_run_id": parentRun.ID,
		"session_name":  p.projectName,
	}

	return p.createRun(ctx, runData)
}

// OnSpanEnd implements tracing.Processor
func (p *TracingProcessor) OnSpanEnd(ctx context.Context, span tracing.Span) error {
	if p.apiKey == "" {
		return nil
	}

	p.runsMu.Lock()
	run, exists := p.runs[span.SpanID()]
	if exists {
		delete(p.runs, span.SpanID())
	}
	p.runsMu.Unlock()

	if !exists {
		return nil
	}

	extracted := p.extractSpanData(span)

	outputs := make(map[string]any)
	if extractedOutputs, ok := extracted["outputs"].(map[string]any); ok {
		outputs = extractedOutputs
	}

	updateData := map[string]any{
		"outputs": outputs,
	}

	if span.Error() != nil {
		updateData["error"] = span.Error().Error()
	}

	if !span.EndedAt().IsZero() {
		updateData["end_time"] = span.EndedAt().Format(time.RFC3339)
	} else {
		updateData["end_time"] = time.Now().UTC().Format(time.RFC3339)
	}

	// Handle response span data for first/last tracking
	if p.isResponseSpanData(span) {
		if inputs, ok := extracted["inputs"].(map[string]any); ok {
			p.responsesMu.Lock()
			if _, exists := p.firstResponseInputs[span.TraceID()]; !exists {
				p.firstResponseInputs[span.TraceID()] = inputs
			}
			p.lastResponseOutputs[span.TraceID()] = outputs
			p.responsesMu.Unlock()
		}
	}

	return p.updateRun(ctx, run.ID, updateData)
}

// Shutdown implements tracing.Processor
func (p *TracingProcessor) Shutdown(ctx context.Context) error {
	// In a real implementation, you might want to flush any pending requests
	return nil
}

// ForceFlush implements tracing.Processor
func (p *TracingProcessor) ForceFlush(ctx context.Context) error {
	// In a real implementation, you might want to flush any pending requests
	return nil
}

// Helper methods

func (p *TracingProcessor) getRunName(span tracing.Span) string {
	spanData := span.SpanData()
	if spanData == nil {
		return "Unknown"
	}

	switch data := spanData.(type) {
	case *tracing.AgentSpanData:
		return data.Name
	case *tracing.FunctionSpanData:
		return data.Name
	case *tracing.GenerationSpanData:
		if data.Model != "" {
			return fmt.Sprintf("LLM (%s)", data.Model)
		}
		return "LLM Generation"
	case *tracing.ResponseSpanData:
		// Response spans represent LLM calls in the newer API
		return "LLM Response"
	default:
		return spanData.Type()
	}
}

func (p *TracingProcessor) getRunType(span tracing.Span) string {
	spanData := span.SpanData()
	if spanData == nil {
		return "chain"
	}

	switch spanData.Type() {
	case "agent":
		return "chain"
	case "function":
		return "tool"
	case "generation":
		return "llm"
	case "response":
		return "llm" // ResponseSpanData also represents LLM calls
	default:
		return "chain"
	}
}

func (p *TracingProcessor) extractSpanData(span tracing.Span) map[string]any {
	result := map[string]any{
		"inputs":   make(map[string]any),
		"outputs":  make(map[string]any),
		"metadata": make(map[string]any),
	}

	spanData := span.SpanData()
	if spanData == nil {
		return result
	}

	switch data := spanData.(type) {
	case *tracing.AgentSpanData:
		inputs := make(map[string]any)
		inputs["name"] = data.Name
		if len(data.Handoffs) > 0 {
			inputs["handoffs"] = data.Handoffs
		}
		if len(data.Tools) > 0 {
			inputs["tools"] = data.Tools
		}
		if data.OutputType != "" {
			inputs["output_type"] = data.OutputType
		}
		result["inputs"] = inputs

	case *tracing.FunctionSpanData:
		inputs := make(map[string]any)
		if data.Input != "" {
			inputs["input"] = data.Input
		}
		result["inputs"] = inputs

		outputs := make(map[string]any)
		if data.Output != nil {
			outputs["output"] = fmt.Sprintf("%v", data.Output)
		}
		result["outputs"] = outputs

		if data.MCPData != nil {
			result["metadata"] = map[string]any{"mcp_data": data.MCPData}
		}

	case *tracing.GenerationSpanData:
		if data.Input != nil {
			result["inputs"] = map[string]any{"messages": data.Input}
		}
		if data.Output != nil {
			result["outputs"] = map[string]any{"generations": data.Output}
		}

		metadata := make(map[string]any)
		if data.Model != "" {
			metadata["model"] = data.Model
		}
		if data.Usage != nil {
			metadata["usage"] = data.Usage
		}
		if data.ModelConfig != nil {
			metadata["model_configuration"] = data.ModelConfig
		}
		result["metadata"] = metadata

	case *tracing.ResponseSpanData:
		// Handle ResponseSpanData for LLM calls
		inputs := make(map[string]any)
		outputs := make(map[string]any)
		metadata := make(map[string]any)

		if data.Input != nil {
			inputs["input"] = data.Input
		}

		if data.Response != nil {
			outputs["response_id"] = data.Response.ID
			outputs["response"] = data.Response
			metadata["response_id"] = data.Response.ID
		}

		result["inputs"] = inputs
		result["outputs"] = outputs
		result["metadata"] = metadata

	default:
		// Handle other span types
		if exported := spanData.Export(); exported != nil {
			for k, v := range exported {
				if k == "type" {
					continue
				}
				result["metadata"].(map[string]any)[k] = v
			}
		}
	}

	return result
}

func (p *TracingProcessor) isResponseSpanData(span tracing.Span) bool {
	spanData := span.SpanData()
	if spanData == nil {
		return false
	}

	// Check if it's ResponseSpanData (LLM calls) or AgentSpanData
	switch spanData.(type) {
	case *tracing.ResponseSpanData:
		return true
	case *tracing.AgentSpanData:
		return true
	default:
		return false
	}
}

func (p *TracingProcessor) createRun(ctx context.Context, runData map[string]any) error {
	return p.sendRequest(ctx, "POST", "/runs", runData)
}

func (p *TracingProcessor) updateRun(ctx context.Context, runID string, runData map[string]any) error {
	return p.sendRequest(ctx, "PATCH", fmt.Sprintf("/runs/%s", runID), runData)
}

func (p *TracingProcessor) sendRequest(ctx context.Context, method, path string, data map[string]any) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	url := strings.TrimSuffix(p.apiURL, "/") + path
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Use lowercase header as per LangSmith API documentation
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		// Read response body for debugging
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("LangSmith API error: %d %s - %s", resp.StatusCode, resp.Status, string(respBody))
	}

	return nil
}
