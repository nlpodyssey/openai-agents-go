package workflowrunner

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// CallbackPublisher publishes streaming events to an external sink.
type CallbackPublisher interface {
	Publish(ctx context.Context, event CallbackEvent) error
}

// CallbackEvent describes an update emitted during a workflow run.
type CallbackEvent struct {
	Type      string         `json:"type"`
	Timestamp time.Time      `json:"timestamp"`
	Payload   any            `json:"payload,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// HTTPCallbackPublisher POSTs events to a configured endpoint as JSON.
type HTTPCallbackPublisher struct {
	client *http.Client
	URL    string
}

// NewHTTPCallbackPublisher constructs an HTTP publisher with an optional custom client.
func NewHTTPCallbackPublisher(url string, client *http.Client) *HTTPCallbackPublisher {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	return &HTTPCallbackPublisher{
		client: client,
		URL:    url,
	}
}

func (p *HTTPCallbackPublisher) Publish(ctx context.Context, event CallbackEvent) error {
	body, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.URL, strings.NewReader(string(body)))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("post callback: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("callback returned status %s", resp.Status)
	}
	return nil
}

// StdoutCallbackPublisher prints events to stdout (useful for local testing).
type StdoutCallbackPublisher struct{}

func (StdoutCallbackPublisher) Publish(_ context.Context, event CallbackEvent) error {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(event)
}
