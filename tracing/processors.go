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
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
)

// ConsoleSpanExporter is an Exporter that prints the traces and spans to the console.
type ConsoleSpanExporter struct{}

func (c ConsoleSpanExporter) Export(items []any) error {
	for _, item := range items {
		switch v := item.(type) {
		case Trace:
			fmt.Printf("[Exporter] Export trace_id=%s, name=%s\n", v.TraceID(), v.Name())
		case Span:
			fmt.Printf("[Exporter] Export span: %+v", v.Export())
		default:
			return fmt.Errorf("ConsoleSpanExporter: unexpected item type %T", item)
		}
	}
	return nil
}

const DefaultBackendSpanExporterEndpoint = "https://api.openai.com/v1/traces/ingest"

type BackendSpanExporter struct {
	apiKey       atomic.Pointer[string]
	organization string
	project      string
	Endpoint     string
	MaxRetries   int
	BaseDelay    time.Duration
	MaxDelay     time.Duration
	client       *http.Client
}

type BackendSpanExporterParams struct {
	// The API key for the "Authorization" header
	// Defaults to OPENAI_API_KEY environment variable if not provided.
	APIKey string
	// The OpenAI organization to use.
	// Defaults to OPENAI_ORG_ID environment variable if not provided.
	Organization string
	// The OpenAI project to use.
	// Defaults to OPENAI_PROJECT_ID environment variable if not provided.
	Project string
	// The HTTP endpoint to which traces/spans are posted.
	// Defaults to DefaultBackendSpanExporterEndpoint if not provided.
	Endpoint string
	// Maximum number of retries upon failures.
	// Default: 3.
	MaxRetries param.Opt[int]
	// Base delay for the first backoff.
	// Default: 1 second.
	BaseDelay param.Opt[time.Duration]
	// Maximum delay for backoff growth.
	// Default: 30 seconds.
	MaxDelay param.Opt[time.Duration]
	// Optional custom http.Client.
	HTTPClient *http.Client
}

func NewBackendSpanExporter(params BackendSpanExporterParams) *BackendSpanExporter {
	b := &BackendSpanExporter{
		organization: params.Organization,
		project:      params.Project,
		Endpoint:     cmp.Or(params.Endpoint, DefaultBackendSpanExporterEndpoint),
		MaxRetries:   params.MaxRetries.Or(3),
		BaseDelay:    params.BaseDelay.Or(1 * time.Second),
		MaxDelay:     params.MaxDelay.Or(30 * time.Second),
		client:       cmp.Or(params.HTTPClient, &http.Client{Timeout: 60 * time.Second}),
	}
	if params.APIKey != "" {
		b.apiKey.Store(&params.APIKey)
	}
	return b
}

// SetAPIKey sets the OpenAI API key for the exporter.
func (b *BackendSpanExporter) SetAPIKey(apiKey string) {
	b.apiKey.Store(&apiKey)
}

func (b *BackendSpanExporter) APIKey() string {
	if v := b.apiKey.Load(); v != nil && *v != "" {
		return *v
	}
	return os.Getenv("OPENAI_API_KEY")
}

func (b *BackendSpanExporter) Organization() string {
	if b.organization == "" {
		return os.Getenv("OPENAI_ORG_ID")
	}
	return b.organization
}

func (b *BackendSpanExporter) Project() string {
	if b.project == "" {
		return os.Getenv("OPENAI_PROJECT_ID")
	}
	return b.project
}

func (b *BackendSpanExporter) Export(ctx context.Context, items []any) error {
	if len(items) == 0 {
		return nil
	}

	if b.APIKey() == "" {
		Logger().Warn("BackendSpanExporter: OpenAI API key is not set, skipping trace export")
		return nil
	}

	data := make([]map[string]any, len(items))
	for i, item := range items {
		switch v := item.(type) {
		case Trace:
			data[i] = v.Export()
		case Span:
			data[i] = v.Export()
		default:
			return fmt.Errorf("BackendSpanExporter: unexpected item type %T", item)
		}
	}

	payload := map[string]any{
		"data": data,
	}

	header := make(http.Header)
	header.Set("Authorization", "Bearer "+b.APIKey())
	header.Set("Content-Type", "application/json")
	header.Set("OpenAI-Beta", "traces=v1")
	if b.Organization() != "" {
		header.Set("OpenAI-Organization", b.Organization())
	}
	if b.Project() != "" {
		header.Set("OpenAI-Project", b.Project())
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to JSON-marshal tracing payload: %w", err)
	}

	// Exponential backoff loop
	attempt := 0
	delay := b.BaseDelay
	for {
		attempt += 1

		request, err := http.NewRequestWithContext(ctx, http.MethodPost, b.Endpoint, bytes.NewReader(jsonPayload))
		if err != nil {
			return fmt.Errorf("failed to initialize new tracing request: %w", err)
		}
		request.Header = header

		response, err := b.client.Do(request)

		if err != nil {
			Logger().Warn("[non-fatal] Tracing: request failed", slog.String("error", err.Error()))
		} else {
			// If the response is successful, break out of the loop
			if response.StatusCode < 300 {
				_ = response.Body.Close()
				Logger().Debug(fmt.Sprintf("Exported %d items", len(items)))
				return nil
			}

			// If the response is a client error (4xx), we won't retry
			if response.StatusCode >= 400 && response.StatusCode < 500 {
				body, err := io.ReadAll(response.Body)
				if err != nil {
					Logger().Warn("failed to read tracing response body", slog.String("error", err.Error()))
				}
				_ = response.Body.Close()
				Logger().Warn(
					"[non-fatal] Tracing client error",
					slog.Int("statusCode", response.StatusCode),
					slog.String("response", string(body)),
				)
				return nil
			}
			_ = response.Body.Close()

			// For 5xx or other unexpected codes, treat it as transient and retry
			Logger().Warn("[non-fatal] Tracing: server error, retrying.", slog.Int("statusCode", response.StatusCode))
		}

		//# If we reach here, we need to retry or give up
		if attempt >= b.MaxRetries {
			Logger().Error("[non-fatal] Tracing: max retries reached, giving up on this batch.")
			return nil
		}

		// Exponential backoff + jitter
		sleepTime := delay + time.Duration(rand.Int64N(int64(delay/10))) // 10% jitter
		time.Sleep(sleepTime)
		delay = min(delay*2, b.MaxDelay)
	}
}

// Close the underlying HTTP client's idle connections.
func (b *BackendSpanExporter) Close() {
	b.client.CloseIdleConnections()
}

type BatchTraceProcessor struct {
	exporter      Exporter
	maxQueueSize  int
	maxBatchSize  int
	scheduleDelay time.Duration
	// The queue size threshold at which we export immediately.
	exportTriggerSize int
	// Track when we next *must* perform a scheduled export.
	nextExportTime time.Time
	shutdownCalled atomic.Bool
	workerRunning  atomic.Bool
	workerDoneChan chan struct{}
	workerMu       sync.RWMutex

	queueMu   sync.RWMutex
	queueChan chan any
	queueSize int
}

type BatchTraceProcessorParams struct {
	// The exporter to use.
	Exporter Exporter
	// The maximum number of spans to store in the queue.
	// After this, we will start dropping spans.
	// Default: 8192.
	MaxQueueSize param.Opt[int]
	// The maximum number of spans to export in a single batch.
	// Default: 128.
	MaxBatchSize param.Opt[int]
	// The delay between checks for new spans to export.
	// Default: 5 seconds.
	ScheduleDelay param.Opt[time.Duration]
	// The ratio of the queue size at which we will trigger an export.
	// Default: 0.7.
	ExportTriggerRatio param.Opt[float64]
}

func NewBatchTraceProcessor(params BatchTraceProcessorParams) *BatchTraceProcessor {
	maxQueueSize := params.MaxQueueSize.Or(8192)
	scheduleDelay := params.ScheduleDelay.Or(5 * time.Second)
	exportTriggerRatio := params.ExportTriggerRatio.Or(0.7)

	return &BatchTraceProcessor{
		exporter:          params.Exporter,
		maxQueueSize:      maxQueueSize,
		maxBatchSize:      params.MaxBatchSize.Or(128),
		scheduleDelay:     scheduleDelay,
		exportTriggerSize: max(1, int(float64(maxQueueSize)*exportTriggerRatio)),
		nextExportTime:    time.Now().Add(scheduleDelay),
		queueChan:         make(chan any, maxQueueSize),
		queueSize:         0,
	}
}

func (b *BatchTraceProcessor) OnTraceStart(ctx context.Context, trace Trace) error {
	// Ensure the background worker is running before we enqueue anything.
	b.ensureWorkerStarted(ctx)

	b.queueMu.Lock()
	defer b.queueMu.Unlock()

	select {
	case b.queueChan <- trace:
		b.queueSize += 1
	default:
		Logger().Warn("Queue is full, dropping trace.")
	}
	return nil
}

func (b *BatchTraceProcessor) OnTraceEnd(ctx context.Context, trace Trace) error {
	// We send traces via OnTraceStart, so we don't need to do anything here.
	return nil
}

func (b *BatchTraceProcessor) OnSpanStart(ctx context.Context, span Span) error {
	// We send spans via OnSpanEnd, so we don't need to do anything here.
	return nil
}

func (b *BatchTraceProcessor) OnSpanEnd(ctx context.Context, span Span) error {
	// Ensure the background worker is running before we enqueue anything.
	b.ensureWorkerStarted(ctx)

	b.queueMu.Lock()
	defer b.queueMu.Unlock()

	select {
	case b.queueChan <- span:
		b.queueSize += 1
	default:
		Logger().Warn("Queue is full, dropping span.")
	}
	return nil
}

// Shutdown is called when the application stops.
// We signal our worker goroutine to stop, then wait for its completion.
func (b *BatchTraceProcessor) Shutdown(ctx context.Context) error {
	b.shutdownCalled.Store(true)

	// Only wait if we ever started the background worker; otherwise flush synchronously.
	if b.workerRunning.Load() {
		<-b.workerDoneChan
		return nil
	}

	// No background goroutine: process any remaining items synchronously.
	return b.exportBatches(ctx, true)
}

// ForceFlush forces an immediate flush of all queued spans.
func (b *BatchTraceProcessor) ForceFlush(ctx context.Context) error {
	return b.exportBatches(ctx, true)
}

func (b *BatchTraceProcessor) ensureWorkerStarted(ctx context.Context) {
	// Fast path without holding the lock.
	if b.workerRunning.Load() {
		return
	}

	b.workerMu.Lock()
	defer b.workerMu.Unlock()
	if b.workerRunning.Load() {
		return
	}

	b.workerDoneChan = make(chan struct{})
	b.workerRunning.Store(true)

	go func() {
		defer func() {
			b.workerMu.Lock()
			defer b.workerMu.Unlock()
			b.workerRunning.Store(false)
			close(b.workerDoneChan)
		}()

		err := b.run(ctx)
		if err != nil {
			Logger().Error("BatchTraceProcessor worker error", slog.String("error", err.Error()))
		}
	}()
}

func (b *BatchTraceProcessor) run(ctx context.Context) error {
	for !b.shutdownCalled.Load() {
		currentTime := time.Now()

		b.queueMu.RLock()
		queueSize := b.queueSize
		b.queueMu.RUnlock()

		// TODO: this could be improved using sync.Cond, avoiding sleep

		// If it's time for a scheduled flush or queue is above the trigger threshold
		if currentTime.After(b.nextExportTime) || queueSize >= b.exportTriggerSize {
			err := b.exportBatches(ctx, false)
			if err != nil {
				return err
			}
			// Reset the next scheduled flush time
			b.nextExportTime = time.Now().Add(b.scheduleDelay)
		} else {
			// Sleep a short interval so we don't busy-wait.
			time.Sleep(200 * time.Millisecond)
		}
	}

	// Final drain after shutdown
	return b.exportBatches(ctx, true)
}

// exportBatches drains the queue and exports in batches. If force=true, export everything.
// Otherwise, export up to `maxBatchSize` repeatedly until the queue is completely empty.
func (b *BatchTraceProcessor) exportBatches(ctx context.Context, force bool) error {
	for {
		var itemsToExport []any

		// Gather a batch of spans up to maxBatchSize
	queueLoop:
		for {
			b.queueMu.Lock()
			queueSize := b.queueSize
			if !(queueSize > 0 && (force || len(itemsToExport) < b.maxBatchSize)) {
				b.queueMu.Unlock()
				break queueLoop
			}

			select {
			case item := <-b.queueChan:
				b.queueSize -= 1
				b.queueMu.Unlock()
				itemsToExport = append(itemsToExport, item)
			default:
				b.queueMu.Unlock()
				// Another goroutine might have emptied the queue between checks
				break queueLoop
			}
		}

		// If we collected nothing, we're done
		if len(itemsToExport) == 0 {
			break
		}

		// Export the batch
		err := b.exporter.Export(ctx, itemsToExport)
		if err != nil {
			return err
		}
	}

	return nil
}

var globalExporter atomic.Pointer[BackendSpanExporter]
var globalProcessor atomic.Pointer[BatchTraceProcessor]

func init() {
	exporter := NewBackendSpanExporter(BackendSpanExporterParams{})
	processor := NewBatchTraceProcessor(BatchTraceProcessorParams{
		Exporter: exporter,
	})

	globalExporter.Store(exporter)
	globalProcessor.Store(processor)
}

// DefaultExporter returns the default exporter, which exports traces and
// spans to the backend in batches.
func DefaultExporter() *BackendSpanExporter {
	return globalExporter.Load()
}

// DefaultProcessor returns the default processor, which exports traces and
// spans to the backend in batches.
func DefaultProcessor() *BatchTraceProcessor {
	return globalProcessor.Load()
}
