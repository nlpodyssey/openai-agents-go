package workflowrunner

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/xeipuuv/gojsonschema"
)

type schemaOutputType struct {
	name     string
	schema   map[string]any
	strict   bool
	compiled *gojsonschema.Schema
}

func newSchemaOutputType(name string, strict bool, schema map[string]any) (agents.OutputTypeInterface, error) {
	if len(schema) == 0 {
		return nil, fmt.Errorf("output type schema cannot be empty")
	}
	compiled, err := gojsonschema.NewSchema(gojsonschema.NewGoLoader(schema))
	if err != nil {
		return nil, fmt.Errorf("compile json schema: %w", err)
	}
	return &schemaOutputType{
		name:     name,
		schema:   schema,
		strict:   strict,
		compiled: compiled,
	}, nil
}

func (s *schemaOutputType) IsPlainText() bool        { return false }
func (s *schemaOutputType) Name() string             { return s.name }
func (s *schemaOutputType) IsStrictJSONSchema() bool { return s.strict }

func (s *schemaOutputType) JSONSchema() (map[string]any, error) {
	copy := make(map[string]any, len(s.schema))
	for k, v := range s.schema {
		copy[k] = v
	}
	return copy, nil
}

func (s *schemaOutputType) ValidateJSON(ctx context.Context, jsonStr string) (any, error) {
	if err := agents.ValidateJSON(ctx, s.compiled, jsonStr); err != nil {
		return nil, err
	}
	var parsed any
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return nil, agents.ModelBehaviorErrorf("failed to decode validated json: %w", err)
	}
	return parsed, nil
}
