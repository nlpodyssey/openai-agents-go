// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agents

import (
	"context"
	"fmt"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/xeipuuv/gojsonschema"
)

func ValidateJSON(ctx context.Context, schema *gojsonschema.Schema, jsonValue string) (err error) {
	defer func() {
		if err != nil {
			AttachErrorToCurrentSpan(ctx, tracing.SpanError{Message: "Invalid JSON provided"})
		}
	}()

	loader := gojsonschema.NewStringLoader(jsonValue)
	result, err := schema.Validate(loader)
	if err != nil {
		return ModelBehaviorErrorf("failed to load and validate JSON: %w", err)
	}

	if result.Valid() {
		return nil
	}

	var sb strings.Builder
	sb.WriteString("JSON validation failed with the following errors:\n")
	for _, e := range result.Errors() {
		_, _ = fmt.Fprintf(&sb, "- %s\n", e)
	}
	return NewModelBehaviorError(sb.String())
}
