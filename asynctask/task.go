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

package asynctask

import (
	"context"
	"sync"
	"sync/atomic"
)

type Task[T any] struct {
	ctx           context.Context
	ctxCancelFunc context.CancelFunc
	doneCh        chan struct{}
	result        *T
	canceled      *atomic.Bool
	mu            sync.Mutex
	closed        bool
}

type TaskResult[T any] struct {
	Result   *T
	Canceled bool
}

func (t *Task[T]) Await() TaskResult[T] {
	<-t.doneCh
	return TaskResult[T]{
		Result:   t.result,
		Canceled: t.canceled.Load(),
	}
}

func (t *Task[T]) IsDone() bool {
	select {
	case <-t.doneCh:
		return true
	default:
		return false
	}
}

func (t *Task[T]) closeDoneCh() {
	if !t.IsDone() {
		t.mu.Lock()
		defer t.mu.Unlock()
		if !t.closed {
			close(t.doneCh)
			t.closed = true
		}
	}
}

func (t *Task[T]) Cancel() bool {
	if t.IsDone() {
		return false
	}

	t.closeDoneCh()
	t.canceled.Store(true)
	t.ctxCancelFunc()
	return true
}

func CreateTask[T any](
	baseCtx context.Context,
	fn func(context.Context) T,
) *Task[T] {
	ctx, cancel := context.WithCancel(baseCtx)

	t := &Task[T]{
		ctx:           nil,
		ctxCancelFunc: cancel,
		doneCh:        make(chan struct{}),
		canceled:      new(atomic.Bool),
	}

	go func() {
		defer cancel()
		defer t.closeDoneCh()
		result := fn(ctx)
		if !t.canceled.Load() {
			t.result = &result
		}
	}()

	return t
}
