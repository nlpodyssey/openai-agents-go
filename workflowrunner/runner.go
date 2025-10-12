package workflowrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/asynctask"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

// RunnerService orchestrates building and executing workflow requests.
type RunnerService struct {
	Builder         *Builder
	CallbackFactory func(ctx context.Context, decl CallbackDeclaration) (CallbackPublisher, error)
	StateStore      ExecutionStateStore
}

// RunSummary holds metadata about a completed run.
type RunSummary struct {
	WorkflowName   string           `json:"workflow_name"`
	SessionID      string           `json:"session_id"`
	FinalOutput    any              `json:"final_output"`
	NewItems       []agents.RunItem `json:"-"`
	LastResponseID string           `json:"last_response_id"`
	Error          error            `json:"error,omitempty"`
}

// NewRunnerService constructs a RunnerService with sensible defaults.
func NewRunnerService(builder *Builder) *RunnerService {
	if builder == nil {
		builder = NewDefaultBuilder()
	}
	defaultStore := NewInMemoryExecutionStateStore()
	return &RunnerService{
		Builder: builder,
		CallbackFactory: func(ctx context.Context, decl CallbackDeclaration) (CallbackPublisher, error) {
			switch decl.Mode {
			case "", "http":
				return NewHTTPCallbackPublisher(decl.Target, nil), nil
			case "stdout":
				return StdoutCallbackPublisher{}, nil
			default:
				return nil, fmt.Errorf("unsupported callback mode %q", decl.Mode)
			}
		},
		StateStore: defaultStore,
	}
}

// Execute validates, builds, and runs the workflow asynchronously.
func (s *RunnerService) Execute(ctx context.Context, req WorkflowRequest) (*asynctask.Task[RunSummary], error) {
	if s.Builder == nil {
		return nil, errors.New("RunnerService missing Builder")
	}
	buildResult, err := s.Builder.Build(ctx, req)
	if err != nil {
		return nil, err
	}

	callbackFactory := s.CallbackFactory
	if callbackFactory == nil {
		callbackFactory = func(ctx context.Context, decl CallbackDeclaration) (CallbackPublisher, error) {
			return StdoutCallbackPublisher{}, nil
		}
	}
	publisher, err := callbackFactory(ctx, req.Callback)
	if err != nil {
		return nil, fmt.Errorf("create callback publisher: %w", err)
	}

	stateStore := s.StateStore
	if stateStore == nil {
		stateStore = NewInMemoryExecutionStateStore()
	}
	tracker := newExecutionStateTracker(stateStore, req.Session.SessionID, req.Workflow.Name)

	return asynctask.CreateTask(ctx, func(taskCtx context.Context) (RunSummary, error) {
		defer func() {
			if closer, ok := buildResult.Session.(interface{ Close() error }); ok {
				_ = closer.Close()
			}
		}()

		summary := RunSummary{
			WorkflowName: req.Workflow.Name,
			SessionID:    req.Session.SessionID,
		}
		traceMetadata := buildResult.TraceMetadata
		if traceMetadata == nil {
			traceMetadata = composeTraceMetadata(req)
		}
		traceID := tracing.GenTraceID()
		buildResult.Runner.Config.TraceID = traceID

		traceErr := tracing.RunTrace(taskCtx, tracing.TraceParams{
			WorkflowName: req.Workflow.Name,
			TraceID:      traceID,
			GroupID:      req.Session.SessionID,
			Metadata:     traceMetadata,
		}, func(ctx context.Context, _ tracing.Trace) error {
			if err := tracker.OnRunStarted(ctx, req.Query); err != nil {
				return err
			}
			startEvent := CallbackEvent{
				Type:      "run.started",
				Timestamp: time.Now().UTC(),
				Payload: map[string]any{
					"workflow": req.Workflow.Name,
					"session":  req.Session.SessionID,
					"query":    req.Query,
				},
			}
			_ = publisher.Publish(ctx, startEvent)

			result, err := buildResult.Runner.RunStreamed(ctx, buildResult.StartingAgent, req.Query)
			if err != nil {
				runErr := wrapRunError(err)
				summary.Error = runErr
				_ = publisher.Publish(ctx, CallbackEvent{
					Type:      "run.failed",
					Timestamp: time.Now().UTC(),
					Payload: map[string]any{
						"error": runErr.Error(),
					},
				})
				_ = tracker.OnRunFailed(ctx, runErr)
				return runErr
			}

			streamErr := result.StreamEvents(func(ev agents.StreamEvent) error {
				event := CallbackEvent{
					Type:      "run.event",
					Timestamp: time.Now().UTC(),
					Payload:   serializeStreamEvent(ev),
				}
				if err := tracker.OnStreamEvent(ctx, ev); err != nil {
					return err
				}
				return publisher.Publish(ctx, event)
			})
			if streamErr != nil {
				streamErr = wrapRunError(streamErr)
				summary.Error = streamErr
				_ = publisher.Publish(ctx, CallbackEvent{
					Type:      "run.failed",
					Timestamp: time.Now().UTC(),
					Payload: map[string]any{
						"error": streamErr.Error(),
					},
				})
				_ = tracker.OnRunFailed(ctx, streamErr)
				return streamErr
			}

			final := result.FinalOutput()
			summary.FinalOutput = final
			summary.NewItems = result.NewItems()
			summary.LastResponseID = result.LastResponseID()

			completeEvent := CallbackEvent{
				Type:      "run.completed",
				Timestamp: time.Now().UTC(),
				Payload: map[string]any{
					"final_output":     final,
					"last_response_id": result.LastResponseID(),
				},
			}
			_ = publisher.Publish(ctx, completeEvent)
			_ = tracker.OnRunCompleted(ctx, result.LastResponseID(), final)
			return nil
		})

		if traceErr != nil && summary.Error == nil {
			summary.Error = traceErr
			_ = tracker.OnRunFailed(taskCtx, traceErr)
		}

		return summary, summary.Error
	}), nil
}

func wrapRunError(err error) error {
	var agentsErr *agents.AgentsError
	if errors.As(err, &agentsErr) && agentsErr.RunData != nil {
		if agentsErr.RunData.LastAgent != nil {
			return fmt.Errorf("%w (last agent: %s)", err, agentsErr.RunData.LastAgent.Name)
		}
	}
	return err
}

func serializeStreamEvent(event agents.StreamEvent) map[string]any {
	switch ev := event.(type) {
	case agents.RawResponsesStreamEvent:
		payload := map[string]any{
			"event_kind": "raw",
			"type":       ev.Data.Type,
		}
		if raw, err := json.Marshal(ev.Data); err == nil {
			payload["data"] = json.RawMessage(raw)
		} else {
			payload["marshal_error"] = err.Error()
		}
		return payload
	case agents.AgentUpdatedStreamEvent:
		agentName := ""
		if ev.NewAgent != nil {
			agentName = ev.NewAgent.Name
		}
		return map[string]any{
			"event_kind": "agent_updated",
			"agent_name": agentName,
		}
	case agents.RunItemStreamEvent:
		return map[string]any{
			"event_kind": "run_item",
			"name":       string(ev.Name),
			"item":       summarizeRunItem(ev.Item),
		}
	default:
		return map[string]any{
			"event_kind": "unknown",
			"type":       fmt.Sprintf("%T", event),
		}
	}
}

func summarizeRunItem(item agents.RunItem) map[string]any {
	switch v := item.(type) {
	case agents.MessageOutputItem:
		return map[string]any{
			"type":  v.Type,
			"agent": displayAgentName(v.Agent),
			"text":  agents.ItemHelpers().TextMessageOutput(v),
		}
	case agents.ToolCallItem:
		payload := map[string]any{
			"type":      v.Type,
			"agent":     displayAgentName(v.Agent),
			"tool_call": fmt.Sprintf("%T", v.RawItem),
		}
		switch raw := v.RawItem.(type) {
		case agents.ResponseFunctionToolCall:
			payload["function_name"] = raw.Name
		case agents.ResponseFunctionWebSearch:
			payload["web_search_status"] = raw.Status
		case agents.ResponseFileSearchToolCall:
			payload["file_search_status"] = raw.Status
		}
		return payload
	case agents.ToolCallOutputItem:
		return map[string]any{
			"type":   v.Type,
			"agent":  displayAgentName(v.Agent),
			"output": v.Output,
		}
	case agents.HandoffOutputItem:
		return map[string]any{
			"type":         v.Type,
			"agent":        displayAgentName(v.Agent),
			"source_agent": displayAgentName(v.SourceAgent),
			"target_agent": displayAgentName(v.TargetAgent),
		}
	default:
		return map[string]any{
			"type": fmt.Sprintf("%T", item),
		}
	}
}
