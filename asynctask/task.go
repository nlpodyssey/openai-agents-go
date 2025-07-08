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
	"errors"
	"fmt"
	"sync"
)

type Task[T any] struct {
	mu       *sync.RWMutex
	cond     *sync.Cond
	cancel   context.CancelFunc
	canceled bool
	done     bool
	result   Result[T]
}

type Result[T any] struct {
	Value T
	Error error
}

var taskCanceledErr = errors.New("task has been canceled")

func TaskCanceledErr() error { return taskCanceledErr }

func (t *Task[T]) Await() Result[T] {
	t.cond.L.Lock()
	for !t.done {
		t.cond.Wait()
	}
	t.cond.L.Unlock()
	return t.result
}

func (t *Task[T]) IsDone() bool {
	t.mu.RLock()
	done := t.done
	t.mu.RUnlock()
	return done
}

func (t *Task[T]) IsCanceled() bool {
	t.mu.RLock()
	canceled := t.canceled
	t.mu.RUnlock()
	return canceled
}

func (t *Task[T]) Cancel() {
	t.mu.Lock()
	if !t.done && !t.canceled {
		t.cancel()
		t.canceled = true
	}
	t.mu.Unlock()
}

type TaskFunc[T any] = func(context.Context) (T, error)

func CreateTask[T any](ctx context.Context, fn TaskFunc[T]) *Task[T] {
	var cancel context.CancelFunc
	ctx, cancel = context.WithCancel(ctx)
	mu := new(sync.RWMutex)
	t := &Task[T]{
		mu:       mu,
		cond:     sync.NewCond(mu),
		cancel:   cancel,
		canceled: false,
		done:     false,
	}

	go func() {
		var value T
		var err error

		defer func() {
			if r := recover(); r != nil {
				err = errors.Join(err, fmt.Errorf("task panicked: %v", r))
			}

			t.cond.L.Lock()
			if t.canceled {
				err = errors.Join(err, TaskCanceledErr())
			}
			t.result = Result[T]{Value: value, Error: err}
			t.done = true
			t.cond.L.Unlock()
			t.cond.Broadcast()

			cancel()
		}()

		value, err = fn(ctx)
	}()

	return t
}

type TaskNoValue = Task[struct{}]

func CreateTaskNoValue(ctx context.Context, fn func(context.Context) error) *TaskNoValue {
	return CreateTask[struct{}](ctx, func(ctx context.Context) (struct{}, error) {
		return struct{}{}, fn(ctx)
	})
}
