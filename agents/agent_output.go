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

package agents

// AgentOutputSchemaInterface is implemented by an object that captures the JSON
// schema of the output, as well as validating/parsing JSON produced by the
// LLM into the output type.
type AgentOutputSchemaInterface interface {
	// IsPlainText reports whether the output type is plain text (versus a JSON object).
	IsPlainText() bool

	// The Name of the output type.
	Name() string

	// JSONSchema returns the JSON schema of the output. Will only be called if the output type is not plain text.
	JSONSchema() map[string]any

	// IsStrictJSONSchema reports whether the JSON schema is in strict mode.
	// Strict mode constrains the JSON schema features, but guarantees valis JSON.
	//
	// See here for details: https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
	IsStrictJSONSchema() bool

	// ValidateJSON validates a JSON string against the output type.
	// You must return the validated object, or a `ModelBehaviorError` if the JSON is invalid.
	ValidateJSON(jsonStr string) (any, error)
}
