package agents

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
)

func newEmptyJSONSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties":           map[string]any{},
		"required":             []string{},
	}
}

// EnsureStrictJSONSchema mutates the given JSON schema to ensure it conforms
// to the `strict` standard that the OpenAI API expects.
func EnsureStrictJSONSchema(schema map[string]any) (map[string]any, error) {
	if len(schema) == 0 {
		return newEmptyJSONSchema(), nil
	}
	return ensureStrictJSONSchema(schema, nil, schema)
}

func ensureStrictJSONSchema(rawJSONSchema any, path []string, root map[string]any) (map[string]any, error) {
	jsonSchema, ok := rawJSONSchema.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("expected %#v to be a map[string]any, path=%+v", rawJSONSchema, path)
	}

	for _, defKey := range []string{"$defs", "definitions"} {
		if defs, ok := jsonSchema[defKey].(map[string]any); ok {
			for defName, defSchema := range defs {
				_, err := ensureStrictJSONSchema(defSchema, slices.Concat(path, []string{defKey, defName}), root)
				if err != nil {
					return nil, err
				}
			}
		}
	}

	additionalProperties, hasAdditionalProperties := jsonSchema["additionalProperties"]

	if typ, _ := jsonSchema["type"].(string); typ == "object" {
		if !hasAdditionalProperties {
			jsonSchema["additionalProperties"] = false
		} else if additionalProperties == true {
			return nil, NewUserError(
				"additionalProperties should not be set for object types. " +
					"This could be because you configured additional properties to be allowed. " +
					"If you really need this, update the function or output tool to not use a strict schema.",
			)
		}
	}

	// object types
	// { 'type': 'object', 'properties': { 'a':  {...} } }
	if properties, ok := jsonSchema["properties"].(map[string]any); ok {
		jsonSchema["required"] = slices.Collect(maps.Keys(properties))

		newProperties := make(map[string]any, len(properties))
		for key, propSchema := range properties {
			var err error
			newProperties[key], err = ensureStrictJSONSchema(propSchema, slices.Concat(path, []string{"properties", key}), root)
			if err != nil {
				return nil, err
			}
		}
		jsonSchema["properties"] = newProperties
	}

	//arrays
	// { 'type': 'array', 'items': {...} }
	if items, ok := jsonSchema["items"].(map[string]any); ok {
		var err error
		jsonSchema["items"], err = ensureStrictJSONSchema(items, slices.Concat(path, []string{"items"}), root)
		if err != nil {
			return nil, err
		}
	}

	// unions
	if anyOf, ok := jsonSchema["anyOf"].([]any); ok {
		newAnyOf := make([]any, len(anyOf))
		for i, variant := range anyOf {
			var err error
			newAnyOf[i], err = ensureStrictJSONSchema(variant, slices.Concat(path, []string{"anyOf", strconv.FormatInt(int64(i), 10)}), root)
			if err != nil {
				return nil, err
			}
		}
		jsonSchema["anyOf"] = newAnyOf
	}

	// intersections
	if allOf, ok := jsonSchema["allOf"].([]any); ok {
		if len(allOf) == 1 {
			result, err := ensureStrictJSONSchema(allOf[0], slices.Concat(path, []string{"allOf", "0"}), root)
			if err != nil {
				return nil, err
			}
			delete(jsonSchema, "allOf")
			maps.Copy(jsonSchema, result)
		} else {
			newAllOf := make([]any, len(allOf))
			for i, variant := range allOf {
				var err error
				newAllOf[i], err = ensureStrictJSONSchema(variant, slices.Concat(path, []string{"allOf", strconv.FormatInt(int64(i), 10)}), root)
				if err != nil {
					return nil, err
				}
			}
			jsonSchema["allOf"] = newAllOf
		}
	}

	// strip `nil` defaults as there's no meaningful distinction here
	// the schema will still be `nullable` and the model will default
	// to using `nil` anyway
	if d, ok := jsonSchema["default"]; ok && d == nil {
		delete(jsonSchema, "default")
	}

	// we can't use `$ref`s if there are other properties defined, e.g.
	// `{"$ref": "...", "description": "my description"}`
	// so we unravel the ref
	// `{"type": "string", "description": "my description"}`
	if rawRef, ok := jsonSchema["$ref"]; ok && len(jsonSchema) > 1 {
		ref, ok := rawRef.(string)
		if !ok {
			return nil, fmt.Errorf("received non-string $ref: %#v", rawRef)
		}
		resolved, err := resolveJONSchemaRef(root, ref)
		if err != nil {
			return nil, err
		}

		delete(jsonSchema, "$ref")
		// properties from the json schema take priority over the ones on the `$ref`
		for k, v := range resolved {
			if _, ok := jsonSchema[k]; !ok {
				jsonSchema[k] = v
			}
		}
		// Since the schema expanded from `$ref` might not have `additionalProperties: false` applied
		// we call `ensureStrictJSONSchema` again to fix the inlined schema and ensure it's valid
		return ensureStrictJSONSchema(jsonSchema, path, root)
	}

	return jsonSchema, nil
}

func resolveJONSchemaRef(root map[string]any, ref string) (map[string]any, error) {
	if !strings.HasPrefix(ref, "#/") {
		return nil, fmt.Errorf("unexpected $ref format: expected `#/` prefix in $ref value %q", ref)
	}

	path := strings.Split(ref[2:], "/")
	resolved := root

	for _, key := range path {
		var ok bool
		resolved, ok = resolved[key].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("encountered non-dictionary entry while resolving $ref %q: %#v", ref, resolved)
		}
	}

	return resolved, nil
}
