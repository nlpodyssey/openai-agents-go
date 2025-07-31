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

package asyncqueue

import (
	"sync"
	"time"
)

type Queue[T any] struct {
	cond   *sync.Cond
	values []T
}

func New[T any]() *Queue[T] {
	return &Queue[T]{
		cond: sync.NewCond(&sync.Mutex{}),
	}
}

func (q *Queue[T]) Put(v T) {
	q.cond.L.Lock()
	q.put(v)
	q.cond.L.Unlock()
}

func (q *Queue[T]) Get() T {
	q.cond.L.Lock()
	defer q.cond.L.Unlock()
	for len(q.values) == 0 {
		q.cond.Wait()
	}
	return q.get()
}

func (q *Queue[T]) GetTimeout(timeout time.Duration) (T, bool) {
	timedOut := false
	timer := time.AfterFunc(timeout, func() {
		q.cond.L.Lock()
		timedOut = true
		q.cond.L.Unlock()
		q.cond.Broadcast()
	})
	defer timer.Stop()

	q.cond.L.Lock()
	defer q.cond.L.Unlock()
	for len(q.values) == 0 && !timedOut {
		q.cond.Wait()
	}

	if timedOut {
		var zero T
		return zero, false
	}
	return q.get(), true
}

func (q *Queue[T]) GetNoWait() (T, bool) {
	q.cond.L.Lock()
	defer q.cond.L.Unlock()

	var zero T
	if len(q.values) == 0 {
		return zero, false
	}

	return q.get(), true
}

func (q *Queue[T]) IsEmpty() bool {
	q.cond.L.Lock()
	defer q.cond.L.Unlock()
	return len(q.values) == 0
}

func (q *Queue[T]) put(v T) {
	q.values = append(q.values, v)
	q.cond.Broadcast()
}

func (q *Queue[T]) get() T {
	v := q.values[0]
	copy(q.values[:len(q.values)-1], q.values[1:])
	clear(q.values[len(q.values)-1:]) // helps GC
	q.values = q.values[:len(q.values)-1]
	q.cond.Broadcast()
	return v
}
