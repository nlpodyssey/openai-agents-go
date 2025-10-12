package workflowrunner

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

type ExecutionStatus string

const (
	ExecutionStatusIdle            ExecutionStatus = "idle"
	ExecutionStatusRunning         ExecutionStatus = "running"
	ExecutionStatusWaitingApproval ExecutionStatus = "waiting_approval"
	ExecutionStatusCompleted       ExecutionStatus = "completed"
	ExecutionStatusFailed          ExecutionStatus = "failed"
)

type ApprovalRequestState struct {
	RequestID   string    `json:"request_id"`
	AgentName   string    `json:"agent_name"`
	ToolName    string    `json:"tool_name"`
	ServerLabel string    `json:"server_label"`
	Arguments   string    `json:"arguments"`
	CreatedAt   time.Time `json:"created_at"`
}

type WorkflowExecutionState struct {
	SessionID        string                 `json:"session_id"`
	WorkflowName     string                 `json:"workflow_name"`
	Status           ExecutionStatus        `json:"status"`
	LastAgent        string                 `json:"last_agent"`
	LastResponseID   string                 `json:"last_response_id"`
	LastQuery        string                 `json:"last_query"`
	LastError        string                 `json:"last_error"`
	PendingApprovals []ApprovalRequestState `json:"pending_approvals"`
	FinalOutput      any                    `json:"final_output,omitempty"`
	UpdatedAt        time.Time              `json:"updated_at"`
}

type ExecutionStateStore interface {
	Save(ctx context.Context, state WorkflowExecutionState) error
	Load(ctx context.Context, sessionID string) (WorkflowExecutionState, bool, error)
	Clear(ctx context.Context, sessionID string) error
}

type InMemoryExecutionStateStore struct {
	mu   sync.RWMutex
	data map[string]WorkflowExecutionState
}

func NewInMemoryExecutionStateStore() *InMemoryExecutionStateStore {
	return &InMemoryExecutionStateStore{data: make(map[string]WorkflowExecutionState)}
}

func (s *InMemoryExecutionStateStore) Save(_ context.Context, state WorkflowExecutionState) error {
	if state.SessionID == "" {
		return errors.New("missing session id")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	copyState := state
	if len(state.PendingApprovals) > 0 {
		copyState.PendingApprovals = append([]ApprovalRequestState(nil), state.PendingApprovals...)
	}
	s.data[state.SessionID] = copyState
	return nil
}

func (s *InMemoryExecutionStateStore) Load(_ context.Context, sessionID string) (WorkflowExecutionState, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	state, ok := s.data[sessionID]
	if !ok {
		return WorkflowExecutionState{}, false, nil
	}
	if len(state.PendingApprovals) > 0 {
		state.PendingApprovals = append([]ApprovalRequestState(nil), state.PendingApprovals...)
	}
	return state, true, nil
}

func (s *InMemoryExecutionStateStore) Clear(_ context.Context, sessionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.data, sessionID)
	return nil
}

type executionStateTracker struct {
	store ExecutionStateStore
	state WorkflowExecutionState
}

func newExecutionStateTracker(store ExecutionStateStore, sessionID, workflowName string) *executionStateTracker {
	return &executionStateTracker{
		store: store,
		state: WorkflowExecutionState{
			SessionID:    sessionID,
			WorkflowName: workflowName,
			Status:       ExecutionStatusIdle,
		},
	}
}

func (t *executionStateTracker) OnRunStarted(ctx context.Context, query string) error {
	t.state.Status = ExecutionStatusRunning
	t.state.LastQuery = query
	t.state.PendingApprovals = nil
	t.state.LastError = ""
	t.state.FinalOutput = nil
	t.state.UpdatedAt = time.Now().UTC()
	return t.store.Save(ctx, t.state)
}

func (t *executionStateTracker) OnStreamEvent(ctx context.Context, event agents.StreamEvent) error {
	switch ev := event.(type) {
	case agents.AgentUpdatedStreamEvent:
		if ev.NewAgent != nil {
			t.state.LastAgent = ev.NewAgent.Name
			t.state.UpdatedAt = time.Now().UTC()
			return t.store.Save(ctx, t.state)
		}
	case agents.RunItemStreamEvent:
		switch item := ev.Item.(type) {
		case agents.MessageOutputItem:
			if item.Agent != nil {
				t.state.LastAgent = item.Agent.Name
				t.state.UpdatedAt = time.Now().UTC()
				return t.store.Save(ctx, t.state)
			}
		case agents.MCPApprovalRequestItem:
			req := ApprovalRequestState{
				RequestID:   item.RawItem.ID,
				AgentName:   displayAgentName(item.Agent),
				ToolName:    item.RawItem.Name,
				ServerLabel: item.RawItem.ServerLabel,
				Arguments:   item.RawItem.Arguments,
				CreatedAt:   time.Now().UTC(),
			}
			t.state.LastAgent = req.AgentName
			t.state.PendingApprovals = append(t.state.PendingApprovals, req)
			t.state.Status = ExecutionStatusWaitingApproval
			t.state.UpdatedAt = time.Now().UTC()
			return t.store.Save(ctx, t.state)
		case agents.MCPApprovalResponseItem:
			id := item.RawItem.ApprovalRequestID
			if id == "" {
				break
			}
			filtered := t.state.PendingApprovals[:0]
			for _, req := range t.state.PendingApprovals {
				if req.RequestID != id {
					filtered = append(filtered, req)
				}
			}
			if len(filtered) != len(t.state.PendingApprovals) {
				t.state.PendingApprovals = append([]ApprovalRequestState(nil), filtered...)
				if len(t.state.PendingApprovals) == 0 && t.state.Status == ExecutionStatusWaitingApproval {
					t.state.Status = ExecutionStatusRunning
				}
				t.state.UpdatedAt = time.Now().UTC()
				return t.store.Save(ctx, t.state)
			}
		}
	}
	return nil
}

func (t *executionStateTracker) OnRunCompleted(ctx context.Context, lastResponseID string, finalOutput any) error {
	t.state.Status = ExecutionStatusCompleted
	t.state.LastResponseID = lastResponseID
	t.state.FinalOutput = finalOutput
	t.state.PendingApprovals = nil
	t.state.LastError = ""
	t.state.UpdatedAt = time.Now().UTC()
	return t.store.Save(ctx, t.state)
}

func (t *executionStateTracker) OnRunFailed(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}
	t.state.LastError = err.Error()
	if len(t.state.PendingApprovals) > 0 {
		t.state.Status = ExecutionStatusWaitingApproval
	} else {
		t.state.Status = ExecutionStatusFailed
	}
	t.state.UpdatedAt = time.Now().UTC()
	return t.store.Save(ctx, t.state)
}
