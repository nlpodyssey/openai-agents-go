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

package traceloop

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/nlpodyssey/openai-agents-go/tracing"
	sdk "github.com/traceloop/go-openllmetry/traceloop-sdk"
)

// TracingProcessor implements tracing.Processor to send traces to Traceloop
type TracingProcessor struct {
	client *sdk.Traceloop

	// Track workflows and tasks for parent-child relationships
	workflows map[string]*sdk.Workflow
	tasks     map[string]*sdk.Task
	llmSpans  map[string]*sdk.LLMSpan
	mu        sync.RWMutex
}

// ProcessorParams configuration for the Traceloop processor
type ProcessorParams struct {
	// Traceloop API key. Required - pass from main
	APIKey string
	// Traceloop Base URL. Defaults to api.traceloop.com
	BaseURL string
	// Optional metadata to attach to all workflows
	Metadata map[string]any
	// Optional tags to attach to all workflows
	Tags []string
}

// NewTracingProcessor creates a new Traceloop tracing processor
func NewTracingProcessor(ctx context.Context, params ProcessorParams) (*TracingProcessor, error) {
	baseURL := params.BaseURL
	if baseURL == "" {
		baseURL = "api.traceloop.com"
	}

	client, err := sdk.NewClient(ctx, sdk.Config{
		BaseURL: baseURL,
		APIKey:  params.APIKey,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Traceloop client: %w", err)
	}

	return &TracingProcessor{
		client:    client,
		workflows: make(map[string]*sdk.Workflow),
		tasks:     make(map[string]*sdk.Task),
		llmSpans:  make(map[string]*sdk.LLMSpan),
	}, nil
}

// OnTraceStart implements tracing.Processor
func (p *TracingProcessor) OnTraceStart(ctx context.Context, trace tracing.Trace) error {
	if p.client == nil {
		fmt.Fprintf(os.Stderr, "Traceloop client not initialized, skipping trace export\n")
		return nil
	}

	workflowName := trace.Name()
	if workflowName == "" {
		workflowName = "Agent workflow"
	}

	// Create workflow attributes
	attrs := sdk.WorkflowAttributes{
		Name: workflowName,
	}

	// Add metadata from trace
	if traceDict := trace.Export(); traceDict != nil {
		if metadata, ok := traceDict["metadata"].(map[string]string); ok {
			// Convert metadata to workflow attributes if needed
			for k, v := range metadata {
				attrs.AssociationProperties[k] = v
			}
		}
	}

	workflow := p.client.NewWorkflow(ctx, attrs)

	p.mu.Lock()
	p.workflows[trace.TraceID()] = workflow
	p.mu.Unlock()

	return nil
}

// OnTraceEnd implements tracing.Processor
func (p *TracingProcessor) OnTraceEnd(ctx context.Context, trace tracing.Trace) error {
	if p.client == nil {
		return nil
	}

	p.mu.Lock()
	workflow, exists := p.workflows[trace.TraceID()]
	if exists {
		delete(p.workflows, trace.TraceID())
	}
	p.mu.Unlock()

	if exists && workflow != nil {
		workflow.End()
	}

	return nil
}

// OnSpanStart implements tracing.Processor
func (p *TracingProcessor) OnSpanStart(ctx context.Context, span tracing.Span) error {
	if p.client == nil {
		return nil
	}

	// Find parent workflow
	p.mu.RLock()
	workflow := p.workflows[span.TraceID()]
	p.mu.RUnlock()

	if workflow == nil {
		fmt.Fprintf(os.Stderr, "No workflow found for span, skipping: %s\n", span.SpanID())
		return nil
	}

	taskName := p.getTaskName(span)
	task := workflow.NewTask(taskName)

	p.mu.Lock()
	p.tasks[span.SpanID()] = task
	p.mu.Unlock()

	// For LLM spans (generation/response), start logging the prompt
	if p.isLLMSpan(span) {
		prompt := p.extractPrompt(span)
		if prompt != nil {
			llmSpan, err := task.LogPrompt(*prompt)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Failed to log prompt: %v\n", err)
				return err
			}

			p.mu.Lock()
			p.llmSpans[span.SpanID()] = &llmSpan
			p.mu.Unlock()
		}
	}

	return nil
}

// OnSpanEnd implements tracing.Processor
func (p *TracingProcessor) OnSpanEnd(ctx context.Context, span tracing.Span) error {
	if p.client == nil {
		return nil
	}

	p.mu.Lock()
	task, taskExists := p.tasks[span.SpanID()]
	llmSpan, llmExists := p.llmSpans[span.SpanID()]
	if taskExists {
		delete(p.tasks, span.SpanID())
	}
	if llmExists {
		delete(p.llmSpans, span.SpanID())
	}
	p.mu.Unlock()

	// Log completion for LLM spans
	if llmExists && llmSpan != nil && p.isLLMSpan(span) {
		completion := p.extractCompletion(span)
		usage := p.extractUsage(span)

		if completion != nil {
			llmSpan.LogCompletion(ctx, *completion, usage)
		}
	}

	// End the task
	if taskExists && task != nil {
		task.End()
	}

	return nil
}

// Shutdown implements tracing.Processor
func (p *TracingProcessor) Shutdown(ctx context.Context) error {
	if p.client != nil {
		p.client.Shutdown(ctx)
	}
	return nil
}

// ForceFlush implements tracing.Processor
func (p *TracingProcessor) ForceFlush(ctx context.Context) error {
	// Traceloop SDK handles flushing internally
	return nil
}

// Helper methods

func (p *TracingProcessor) getTaskName(span tracing.Span) string {
	spanData := span.SpanData()
	if spanData == nil {
		return "unknown_task"
	}

	switch data := spanData.(type) {
	case *tracing.AgentSpanData:
		return fmt.Sprintf("agent_%s", data.Name)
	case *tracing.FunctionSpanData:
		return fmt.Sprintf("function_%s", data.Name)
	case *tracing.GenerationSpanData:
		if data.Model != "" {
			return fmt.Sprintf("llm_%s", data.Model)
		}
		return "llm_generation"
	case *tracing.ResponseSpanData:
		return "llm_response"
	default:
		return spanData.Type()
	}
}

func (p *TracingProcessor) isLLMSpan(span tracing.Span) bool {
	spanData := span.SpanData()
	if spanData == nil {
		return false
	}

	switch spanData.(type) {
	case *tracing.GenerationSpanData, *tracing.ResponseSpanData:
		return true
	default:
		return false
	}
}

func (p *TracingProcessor) extractPrompt(span tracing.Span) *sdk.Prompt {
	spanData := span.SpanData()
	if spanData == nil {
		return nil
	}

	switch data := spanData.(type) {
	case *tracing.GenerationSpanData:
		prompt := &sdk.Prompt{
			Vendor: "openai",
			Mode:   "chat",
		}

		if data.Model != "" {
			prompt.Model = data.Model
		}

		if data.Input != nil {
			messages := p.convertMessagesToTraceloop(data.Input)
			prompt.Messages = messages
		}

		return prompt

	case *tracing.ResponseSpanData:
		prompt := &sdk.Prompt{
			Vendor: "openai",
			Mode:   "chat",
			Model:  "gpt-4", // Default, will be updated if available
		}

		if data.Input != nil {
			// Try to extract messages from input
			if inputSlice, ok := data.Input.([]map[string]any); ok {
				messages := make([]sdk.Message, len(inputSlice))
				for i, msg := range inputSlice {
					if content, ok := msg["content"].(string); ok {
						role := "user" // default
						if r, ok := msg["role"].(string); ok {
							role = r
						}
						messages[i] = sdk.Message{
							Index:   i,
							Content: content,
							Role:    role,
						}
					}
				}
				prompt.Messages = messages
			}
		}

		return prompt
	}

	return nil
}

func (p *TracingProcessor) extractCompletion(span tracing.Span) *sdk.Completion {
	spanData := span.SpanData()
	if spanData == nil {
		return nil
	}

	switch data := spanData.(type) {
	case *tracing.GenerationSpanData:
		completion := &sdk.Completion{}

		if data.Model != "" {
			completion.Model = data.Model
		}

		if data.Output != nil {
			messages := p.convertMessagesToTraceloop(data.Output)
			completion.Messages = messages
		}

		return completion

	case *tracing.ResponseSpanData:
		completion := &sdk.Completion{
			Model: "gpt-4", // Default
		}

		if data.Response != nil {
			// Extract response content
			// This would need to be adapted based on the actual Response structure
			completion.Messages = []sdk.Message{
				{
					Index:   0,
					Content: fmt.Sprintf("Response ID: %v", data.Response),
					Role:    "assistant",
				},
			}
		}

		return completion
	}

	return nil
}

func (p *TracingProcessor) extractUsage(span tracing.Span) sdk.Usage {
	spanData := span.SpanData()
	if spanData == nil {
		return sdk.Usage{}
	}

	if data, ok := spanData.(*tracing.GenerationSpanData); ok && data.Usage != nil {
		// Convert usage data if available
		usage := sdk.Usage{}

		// Try to extract usage information from the usage data
		if totalTokens, ok := data.Usage["total_tokens"].(int); ok {
			usage.TotalTokens = totalTokens
		}
		if promptTokens, ok := data.Usage["prompt_tokens"].(int); ok {
			usage.PromptTokens = promptTokens
		}
		if completionTokens, ok := data.Usage["completion_tokens"].(int); ok {
			usage.CompletionTokens = completionTokens
		}

		return usage
	}

	return sdk.Usage{}
}

func (p *TracingProcessor) convertMessagesToTraceloop(input any) []sdk.Message {
	if input == nil {
		return nil
	}

	// Try to convert input to message format
	if inputSlice, ok := input.([]map[string]any); ok {
		messages := make([]sdk.Message, len(inputSlice))
		for i, msg := range inputSlice {
			content := ""
			role := "user"

			if c, ok := msg["content"].(string); ok {
				content = c
			}
			if r, ok := msg["role"].(string); ok {
				role = r
			}

			messages[i] = sdk.Message{
				Index:   i,
				Content: content,
				Role:    role,
			}
		}
		return messages
	}

	return nil
}
