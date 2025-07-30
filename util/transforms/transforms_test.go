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

package transforms

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTransformStringFunctionStyle(t *testing.T) {
	input := "Foo Bar 123?Baz Quux!"
	result := TransformStringFunctionStyle(input)
	assert.Equal(t, "foo_bar_123_baz_quux_", result)
}

func TestToCamelCase(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"snake_case", "snakeCase"},
		{"PascalCase", "pascalCase"},
		{"camelCase", "camelCase"},
		{"", ""},
		{"single", "single"},
		{"UPPER_CASE", "upperCase"},
		{"multiple_word_example", "multipleWordExample"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := ToCamelCase(tt.input)
			if result != tt.expected {
				t.Errorf("ToCamelCase(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestToSnakeCase(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"camelCase", "camel_case"},
		{"PascalCase", "pascal_case"},
		{"snake_case", "snake_case"},
		{"", ""},
		{"single", "single"},
		{"HTTPRequest", "httprequest"}, // Note: consecutive capitals
		{"getHTTPResponse", "get_httpresponse"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := ToSnakeCase(tt.input)
			if result != tt.expected {
				t.Errorf("ToSnakeCase(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
