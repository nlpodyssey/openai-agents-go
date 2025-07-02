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

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/invopop/jsonschema"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/xeipuuv/gojsonschema"
)

// OutputTypeInterface is implemented by an object that describes an agent's output type.
// Unless the output type is plain text (string), it captures the JSON schema of the output,
// as well as validating/parsing JSON produced by the LLM into the output type.
type OutputTypeInterface interface {
	// IsPlainText reports whether the output type is plain text (versus a JSON object).
	IsPlainText() bool

	// The Name of the output type.
	Name() string

	// JSONSchema returns the JSON schema of the output.
	// It will only be called if the output type is not plain text.
	JSONSchema() (map[string]any, error)

	// IsStrictJSONSchema reports whether the JSON schema is in strict mode.
	// Strict mode constrains the JSON schema features, but guarantees valid JSON.
	//
	// For more details, see https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
	IsStrictJSONSchema() bool

	// ValidateJSON validates a JSON string against the output type.
	// You must return the validated object, or a `ModelBehaviorError` if the JSON is invalid.
	// It will only be called if the output type is not plain text.
	ValidateJSON(ctx context.Context, jsonStr string) (any, error)
}

type outputTypeImpl[T any] struct {
	// Whether the output type is wrapped in a dictionary. This is generally done if the base
	// output type cannot be represented as a JSON Schema object.
	isWrapped bool

	// The JSON schema of the output.
	outputSchema map[string]any

	// Whether the JSON schema is in strict mode. We **strongly** recommend setting this to true,
	// as it increases the likelihood of correct JSON input.
	strictJSONSchema bool

	isPlainText bool
	name        string
}

type wrappedOutputType[T any] struct {
	Response T `json:"response"`
}

// OutputType creates a new output type for T with default options (strict schema).
// It panics in case of errors. For a safer variant, see SafeOutputType.
func OutputType[T any]() OutputTypeInterface {
	result, err := SafeOutputType[T](defaultOutputTypeOpts)
	if err != nil {
		panic(err)
	}
	return result
}

type OutputTypeOpts struct {
	StrictJSONSchema bool
}

var defaultOutputTypeOpts = OutputTypeOpts{
	StrictJSONSchema: true,
}

// OutputTypeWithOpts creates a new output type for T with custom options.
// It panics in case of errors. For a safer variant, see SafeOutputType.
func OutputTypeWithOpts[T any](opts OutputTypeOpts) OutputTypeInterface {
	result, err := SafeOutputType[T](opts)
	if err != nil {
		panic(err)
	}
	return result
}

// SafeOutputType creates a new output type for T with custom options.
func SafeOutputType[T any](opts OutputTypeOpts) (OutputTypeInterface, error) {
	var isWrapped bool
	var outputSchema map[string]any

	var zero T
	_, isPlainText := any(zero).(string)

	if isPlainText {
		isWrapped = false
		outputSchema = map[string]any{"type": "string"}
	} else {
		// We should wrap for things that are not plain text, and for things
		// that would definitely not be a JSON Schema object.
		isWrapped = !isStruct[T]()

		var schema *jsonschema.Schema
		reflector := jsonschema.Reflector{
			Anonymous:                 true,
			AllowAdditionalProperties: !opts.StrictJSONSchema,
			ExpandedStruct:            true,
		}

		var valueToReflect any
		if isWrapped {
			valueToReflect = wrappedOutputType[T]{}
		} else {
			valueToReflect = zero
		}

		schema = reflector.Reflect(valueToReflect)
		b, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to JSON-marshal JSON schema: %w", err)
		}
		err = json.Unmarshal(b, &outputSchema)
		if err != nil {
			return nil, fmt.Errorf("failed to JSON-unmarshal JSON schema: %w", err)
		}

		if opts.StrictJSONSchema {
			outputSchema, err = EnsureStrictJSONSchema(outputSchema)
			if err != nil {
				var userError UserError
				if errors.As(err, &userError) {
					return nil, UserErrorf(
						"Strict JSON schema is enabled, but the output type is not valid. Either make the "+
							"output type strict, or disable strict JSON schema in your Agent(). Error: %w", userError,
					)
				} else {
					return nil, err
				}
			}
		}
	}

	return outputTypeImpl[T]{
		isWrapped:        isWrapped,
		outputSchema:     outputSchema,
		strictJSONSchema: opts.StrictJSONSchema,
		isPlainText:      isPlainText,
		name:             fmt.Sprintf("%T", zero),
	}, nil
}

// isStruct reports whether v is a struct or pointer to struct.
func isStruct[T any]() bool {
	var zero T
	val := reflect.ValueOf(zero)
	kind := val.Type().Kind()
	return (kind == reflect.Struct) || (kind == reflect.Ptr && val.Elem().Kind() == reflect.Struct)
}

func (t outputTypeImpl[T]) IsPlainText() bool        { return t.isPlainText }
func (t outputTypeImpl[T]) Name() string             { return t.name }
func (t outputTypeImpl[T]) IsStrictJSONSchema() bool { return t.strictJSONSchema }

func (t outputTypeImpl[T]) JSONSchema() (map[string]any, error) {
	if t.isPlainText {
		return nil, NewUserError("output type is plain text, so no JSON schema is available")
	}
	return t.outputSchema, nil
}

func (t outputTypeImpl[T]) ValidateJSON(ctx context.Context, jsonStr string) (_ any, err error) {
	if t.isPlainText {
		return nil, NewUserError("output type is plain text, so JSON validation is not available")
	}

	schemaLoader := gojsonschema.NewGoLoader(t.outputSchema)
	schema, err := gojsonschema.NewSchema(schemaLoader)
	if err != nil {
		return nil, ModelBehaviorErrorf("failed to load and compile output JSON schema: %w", err)
	}

	err = ValidateJSON(ctx, schema, jsonStr)
	if err != nil {
		return nil, fmt.Errorf("output type validation error: %w", err)
	}

	defer func() {
		if err != nil {
			AttachErrorToCurrentSpan(ctx, tracing.SpanError{
				Message: "Invalid JSON",
				Data:    map[string]any{"details": err.Error()},
			})
		}
	}()

	if t.isWrapped {
		var wrappedOutput wrappedOutputType[T]
		err = json.Unmarshal([]byte(jsonStr), &wrappedOutput)
		if err != nil {
			return nil, ModelBehaviorErrorf("failed to unmarshal JSON output (wrapped): %w", err)
		}
		return wrappedOutput.Response, nil
	} else {
		var output T
		err = json.Unmarshal([]byte(jsonStr), &output)
		if err != nil {
			return nil, ModelBehaviorErrorf("failed to unmarshal JSON output: %w", err)
		}
		return output, nil
	}
}
