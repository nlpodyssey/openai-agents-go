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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOptional_MarshalJSON(t *testing.T) {
	t.Run("value present", func(t *testing.T) {
		v := Value("foo")
		res, err := json.Marshal(v)
		require.NoError(t, err)
		assert.Equal(t, `"foo"`, string(res))
	})

	t.Run("value not present", func(t *testing.T) {
		v := None[string]()
		res, err := json.Marshal(v)
		require.NoError(t, err)
		assert.Equal(t, `null`, string(res))
	})

	t.Run("marshaling error", func(t *testing.T) {
		v := Value(2i)
		_, err := json.Marshal(v)
		assert.Error(t, err)
	})
}

func TestOptional_UnmarshalJSON(t *testing.T) {
	t.Run("not null", func(t *testing.T) {
		var v Optional[string]
		err := json.Unmarshal([]byte(`"foo"`), &v)
		require.NoError(t, err)
		want := Value("foo")
		assert.Equal(t, want, v)
	})

	t.Run("null", func(t *testing.T) {
		var v Optional[string]
		err := json.Unmarshal([]byte(`null`), &v)
		require.NoError(t, err)
		want := None[string]()
		assert.Equal(t, want, v)
	})

	t.Run("unmarshaling error", func(t *testing.T) {
		var v Optional[string]
		err := json.Unmarshal([]byte(`hey`), &v)
		assert.Error(t, err)
	})
}

func TestOptional_ValueOrFallback(t *testing.T) {
	t.Run("value present", func(t *testing.T) {
		v := Value("foo")
		got := v.ValueOrFallback("bar")
		assert.Equal(t, "foo", got)
	})

	t.Run("value not present", func(t *testing.T) {
		v := None[string]()
		got := v.ValueOrFallback("bar")
		assert.Equal(t, "bar", got)
	})
}

func TestOptional_ValueOrFallbackFunc(t *testing.T) {
	t.Run("value present", func(t *testing.T) {
		v := Value("foo")
		got := v.ValueOrFallbackFunc(func() string { return "bar" })
		assert.Equal(t, "foo", got)
	})
	t.Run("value not present", func(t *testing.T) {
		v := None[string]()
		got := v.ValueOrFallbackFunc(func() string { return "bar" })
		assert.Equal(t, "bar", got)
	})
}
