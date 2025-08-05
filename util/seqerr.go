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

package util

import "iter"

type SeqErr[T any] interface {
	Seq() iter.Seq[T]
	Error() error
}

func SeqErrFunc[T any](fn func(yield func(T) bool) error) SeqErr[T] {
	return &seqErrFunc[T]{fn: fn}
}

type seqErrFunc[T any] struct {
	fn  func(yield func(T) bool) error
	err error
}

func (s *seqErrFunc[T]) seq(yield func(T) bool) { s.err = s.fn(yield) }
func (s *seqErrFunc[T]) Seq() iter.Seq[T]       { return s.seq }
func (s *seqErrFunc[T]) Error() error           { return s.err }
