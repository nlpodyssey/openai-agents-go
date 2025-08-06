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
	"reflect"
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

		// Prefer request model over response model for more accurate reporting
		if data.Model != "" {
			prompt.Model = data.Model
		} else if data.Response != nil && data.Response.Model != "" {
			prompt.Model = data.Response.Model
		}

		if data.Input != nil {
			// Handle InputItems from agents package using reflection
			messages := p.extractMessagesFromInputUsingReflection(data.Input)
			prompt.Messages = messages
			
			// Fallback to legacy format if no messages extracted
			if len(messages) == 0 {
				if inputSlice, ok := data.Input.([]map[string]any); ok {
					// Handle legacy []map[string]any format
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

		// Prefer request model over response model for more accurate reporting
		if data.Model != "" {
			completion.Model = data.Model
		} else if data.Response != nil && data.Response.Model != "" {
			completion.Model = data.Response.Model
		}

		if data.Response != nil {
			
			// Extract response content from the actual response output
			if len(data.Response.Output) > 0 {
				messages := make([]sdk.Message, 0)
				for i, output := range data.Response.Output {
					if len(output.Content) > 0 {
						for j, content := range output.Content {
							if content.Text != "" {
								messages = append(messages, sdk.Message{
									Index:   i*100 + j, // Simple indexing
									Content: content.Text,
									Role:    "assistant",
								})
							}
						}
					}
				}
				completion.Messages = messages
			} else {
				// Fallback to simple response ID
				completion.Messages = []sdk.Message{
					{
						Index:   0,
						Content: fmt.Sprintf("Response ID: %s", data.Response.ID),
						Role:    "assistant",
					},
				}
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
		// Check for standard OpenAI field names first
		if totalTokens, ok := data.Usage["total_tokens"].(int); ok {
			usage.TotalTokens = totalTokens
		}
		
		if promptTokens, ok := data.Usage["prompt_tokens"].(int); ok {
			usage.PromptTokens = promptTokens
		} else if inputTokens, ok := data.Usage["input_tokens"].(int); ok {
			usage.PromptTokens = inputTokens
		}
		
		if completionTokens, ok := data.Usage["completion_tokens"].(int); ok {
			usage.CompletionTokens = completionTokens
		} else if outputTokens, ok := data.Usage["output_tokens"].(int); ok {
			usage.CompletionTokens = outputTokens
		}
		
		// Calculate total if not provided
		if usage.TotalTokens == 0 && (usage.PromptTokens > 0 || usage.CompletionTokens > 0) {
			usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		}

		return usage
	}

	if data, ok := spanData.(*tracing.ResponseSpanData); ok && data.Response != nil {
		usage := sdk.Usage{}
		
		// Extract usage from OpenAI response usage fields
		if data.Response.Usage.InputTokens > 0 {
			usage.PromptTokens = int(data.Response.Usage.InputTokens)
		}
		if data.Response.Usage.OutputTokens > 0 {
			usage.CompletionTokens = int(data.Response.Usage.OutputTokens)
		}
		
		// Calculate total tokens
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		
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

// extractMessagesFromInputUsingReflection extracts messages from any input format using reflection
func (p *TracingProcessor) extractMessagesFromInputUsingReflection(input interface{}) []sdk.Message {
	if input == nil {
		return nil
	}
	
	// Use reflection to handle the InputItems type
	v := reflect.ValueOf(input)
	if v.Kind() == reflect.Slice {
		var messages []sdk.Message
		
		for i := 0; i < v.Len(); i++ {
			item := v.Index(i).Interface()
			if msg := p.extractMessageFromInputItem(item, i); msg != nil {
				messages = append(messages, *msg)
			}
		}
		return messages
	}
	
	return nil
}

// extractMessageFromInputItem extracts a message from a TResponseInputItem using reflection
func (p *TracingProcessor) extractMessageFromInputItem(item interface{}, index int) *sdk.Message {
	if item == nil {
		return nil
	}
	
	// Use reflection to access the OfMessage field
	v := reflect.ValueOf(item)
	if v.Kind() == reflect.Struct {
		// Look for the OfMessage field
		ofMessageField := v.FieldByName("OfMessage")
		if ofMessageField.IsValid() && !ofMessageField.IsNil() {
			// Get the actual message object
			messageValue := ofMessageField.Elem() // Dereference the pointer
			if messageValue.IsValid() && messageValue.Kind() == reflect.Struct {
				// Look for Content and Role fields in the message
				contentField := messageValue.FieldByName("Content")
				roleField := messageValue.FieldByName("Role")
				
				content := ""
				role := "user" // default
				
				if contentField.IsValid() {
					// Content is a union type, try to extract string value
					if contentField.Kind() == reflect.String {
						content = contentField.String()
					} else {
						// Try to extract from union type by examining its structure
						contentStr := fmt.Sprintf("%v", contentField.Interface())
						// Look for the actual text content in the string representation
						// From debug: Content = {Hello! Can you introduce yourself? [] {{<nil>}}}
						// Need to extract the full text before the array brackets
						if len(contentStr) > 2 && contentStr[0] == '{' {
							// Find the end of the text content (before " []" or similar)
							end := len(contentStr)
							bracketStart := -1
							
							// Look for " [" which indicates the start of the array part
							for i := 1; i < len(contentStr)-1; i++ {
								if contentStr[i] == ' ' && contentStr[i+1] == '[' {
									bracketStart = i
									break
								}
							}
							
							if bracketStart > 1 {
								end = bracketStart
							} else {
								// Fallback: find the last } before the end
								for i := len(contentStr) - 1; i > 1; i-- {
									if contentStr[i] == '}' {
										end = i
										break
									}
								}
							}
							
							if end > 1 {
								content = contentStr[1:end]
							}
						}
					}
				}
				
				if roleField.IsValid() && roleField.Kind() == reflect.String {
					role = roleField.String()
				}
				
				if content != "" {
					return &sdk.Message{
						Index:   index,
						Content: content,
						Role:    role,
					}
				}
			}
		}
	}
	
	return nil
}
