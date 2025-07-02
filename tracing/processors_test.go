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

package tracing_test

import (
	"testing"

	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/stretchr/testify/assert"
)

func TestBackendSpanExporter_APIKey(t *testing.T) {
	t.Run("SetAPIKey", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "")

		// If the API key is not set, it should stay empty string
		processor := tracing.NewBackendSpanExporter(tracing.BackendSpanExporterParams{})
		assert.Equal(t, "", processor.APIKey())

		// If we set it afterward, it should be the new value
		processor.SetAPIKey("test_api_key")
		assert.Equal(t, "test_api_key", processor.APIKey())
	})

	t.Run("from env", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "")

		// If the API key is not set at creation time but set before access
		// time, it should be the new value.
		processor := tracing.NewBackendSpanExporter(tracing.BackendSpanExporterParams{})
		assert.Equal(t, "", processor.APIKey())

		// If we set it afterward, it should be the new value
		t.Setenv("OPENAI_API_KEY", "foo_bar_123")
		assert.Equal(t, "foo_bar_123", processor.APIKey())
	})
}
