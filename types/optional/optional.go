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

package optional

import (
	"encoding/json"

	"github.com/openai/openai-go/packages/param"
)

type Optional[T any] struct {
	Present bool
	Value   T
}

func (o Optional[T]) Get() (T, bool) {
	return o.Value, o.Present
}

func (o Optional[T]) ValueOrFallback(fallback T) T {
	if o.Present {
		return o.Value
	}
	return fallback
}

func (o Optional[T]) ValueOrFallbackFunc(fallbackFunc func() T) T {
	if o.Present {
		return o.Value
	}
	return fallbackFunc()
}

func (o Optional[T]) MarshalJSON() ([]byte, error) {
	if !o.Present {
		return []byte("null"), nil
	}
	return json.Marshal(o.Value)
}

func (o *Optional[T]) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		o.Present = false
		return nil
	}
	o.Present = true
	return json.Unmarshal(data, &o.Value)
}

func Value[T any](v T) Optional[T] {
	return Optional[T]{Present: true, Value: v}
}

func None[T any]() Optional[T] {
	return Optional[T]{Present: false}
}

func ToParamOptNull[T comparable](o Optional[T]) param.Opt[T] {
	if o.Present {
		return param.NewOpt(o.Value)
	}
	return param.Null[T]()
}

func ToParamOptOmitted[T comparable](o Optional[T]) param.Opt[T] {
	if o.Present {
		return param.NewOpt(o.Value)
	}
	return param.Opt[T]{}
}
