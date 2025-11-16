package workflowrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/memory"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

func newWebSearchTool(_ context.Context, decl ToolDeclaration, _ ToolFactoryEnv) (agents.Tool, error) {
	var tool agents.WebSearchTool
	if loc, ok := getMap(decl.Config, "user_location"); ok {
		tool.UserLocation = responses.WebSearchToolUserLocationParam{}
		if city, ok := getString(loc, "city"); ok {
			tool.UserLocation.City = param.NewOpt(city)
		}
		if region, ok := getString(loc, "region"); ok {
			tool.UserLocation.Region = param.NewOpt(region)
		}
		if country, ok := getString(loc, "country"); ok {
			tool.UserLocation.Country = param.NewOpt(country)
		}
		if t, ok := getString(loc, "type"); ok {
			tool.UserLocation.Type = t
		}
		if tool.UserLocation.Type == "" {
			tool.UserLocation.Type = string(constant.ValueOf[constant.Approximate]())
		}
	}
	if ctxSize, ok := getString(decl.Config, "search_context_size"); ok {
		switch strings.ToLower(ctxSize) {
		case "low", "medium", "high":
			tool.SearchContextSize = responses.WebSearchToolSearchContextSize(ctxSize)
		default:
			tool.SearchContextSize = responses.WebSearchToolSearchContextSize("medium")
		}
	} else {
		tool.SearchContextSize = responses.WebSearchToolSearchContextSize("medium")
	}
	return tool, nil
}

func newCodeInterpreterTool(_ context.Context, _ ToolDeclaration, _ ToolFactoryEnv) (agents.Tool, error) {
	return agents.CodeInterpreterTool{
		ToolConfig: responses.ToolCodeInterpreterParam{
			Container: responses.ToolCodeInterpreterContainerUnionParam{
				OfCodeInterpreterContainerAuto: &responses.ToolCodeInterpreterContainerCodeInterpreterContainerAutoParam{
					Type: constant.ValueOf[constant.Auto](),
				},
			},
			Type: constant.ValueOf[constant.CodeInterpreter](),
		},
	}, nil
}

func newFileSearchTool(_ context.Context, decl ToolDeclaration, _ ToolFactoryEnv) (agents.Tool, error) {
	cfg := agents.FileSearchTool{}
	if vectorIDs, ok := getSlice[string](decl.Config, "vector_store_ids"); ok {
		cfg.VectorStoreIDs = vectorIDs
	}
	if limit, ok := getFloat(decl.Config, "max_num_results"); ok {
		cfg.MaxNumResults = param.NewOpt[int64](int64(limit))
	}
	if includeResults, ok := getBool(decl.Config, "include_search_results"); ok {
		cfg.IncludeSearchResults = includeResults
	}
	return cfg, nil
}

func newImageGenerationTool(_ context.Context, decl ToolDeclaration, _ ToolFactoryEnv) (agents.Tool, error) {
	tool := agents.ImageGenerationTool{}
	if cfg, ok := getMap(decl.Config, "tool_config"); ok {
		param := responses.ToolImageGenerationParam{}
		if quality, ok := getString(cfg, "quality"); ok {
			param.Quality = quality
		}
		param.Type = constant.ValueOf[constant.ImageGeneration]()
		tool.ToolConfig = param
	} else {
		tool.ToolConfig = responses.ToolImageGenerationParam{
			Type: constant.ValueOf[constant.ImageGeneration](),
		}
	}
	return tool, nil
}

func newHostedMCPTool(_ context.Context, decl ToolDeclaration, env ToolFactoryEnv) (agents.Tool, error) {
	url, ok := getString(decl.Config, "server_url")
	if !ok || strings.TrimSpace(url) == "" {
		return nil, errors.New("server_url is required for hosted_mcp tool")
	}
	label := decl.Name
	if label == "" {
		label, _ = getString(decl.Config, "server_label")
		if label == "" {
			label = fmt.Sprintf("%s_%s", env.WorkflowName, env.AgentName)
		}
	}
	requireApproval := "never"
	if v, ok := getString(decl.Config, "require_approval"); ok && v != "" {
		requireApproval = v
	}
	return agents.HostedMCPTool{
		ToolConfig: responses.ToolMcpParam{
			ServerLabel: label,
			ServerURL:   param.NewOpt(url),
			RequireApproval: responses.ToolMcpRequireApprovalUnionParam{
				OfMcpToolApprovalSetting: param.NewOpt(requireApproval),
			},
			Type: constant.ValueOf[constant.Mcp](),
		},
	}, nil
}

func newJSONMapOutputType(_ context.Context, decl OutputTypeDeclaration) (agents.OutputTypeInterface, error) {
	schema := decl.Schema
	if schema == nil {
		schema = map[string]any{
			"type": "object",
		}
	}
	name := decl.Name
	if name == "" {
		name = "json_object"
	}
	return newSchemaOutputType(name, decl.Strict, schema)
}

// NewSQLiteSessionFactory stores sessions on-disk inside baseDir (created if needed).
func NewSQLiteSessionFactory(baseDir string) SessionFactory {
	return func(ctx context.Context, decl SessionDeclaration) (memory.Session, error) {
		if err := os.MkdirAll(baseDir, 0o755); err != nil {
			return nil, fmt.Errorf("create session dir: %w", err)
		}
		dbPath := filepath.Join(baseDir, fmt.Sprintf("%s.db", sanitizeFileName(decl.SessionID)))
		return memory.NewSQLiteSession(ctx, memory.SQLiteSessionParams{
			SessionID:        decl.SessionID,
			DBDataSourceName: dbPath,
		})
	}
}

func sanitizeFileName(input string) string {
	var b strings.Builder
	for _, r := range input {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	return b.String()
}

func getMap(source map[string]any, key string) (map[string]any, bool) {
	if source == nil {
		return nil, false
	}
	raw, ok := source[key]
	if !ok || raw == nil {
		return nil, false
	}
	switch v := raw.(type) {
	case map[string]any:
		return v, true
	case map[string]string:
		result := make(map[string]any, len(v))
		for k, val := range v {
			result[k] = val
		}
		return result, true
	case []byte:
		var m map[string]any
		if err := json.Unmarshal(v, &m); err != nil {
			return nil, false
		}
		return m, true
	default:
		return nil, false
	}
}

func getString(source map[string]any, key string) (string, bool) {
	if source == nil {
		return "", false
	}
	raw, ok := source[key]
	if !ok || raw == nil {
		return "", false
	}
	switch v := raw.(type) {
	case string:
		return v, true
	case fmt.Stringer:
		return v.String(), true
	default:
		return fmt.Sprintf("%v", raw), true
	}
}

func getFloat(source map[string]any, key string) (float64, bool) {
	if source == nil {
		return 0, false
	}
	raw, ok := source[key]
	if !ok || raw == nil {
		return 0, false
	}
	switch v := raw.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case json.Number:
		f, err := v.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		return 0, false
	}
}

func getBool(source map[string]any, key string) (bool, bool) {
	if source == nil {
		return false, false
	}
	raw, ok := source[key]
	if !ok || raw == nil {
		return false, false
	}
	switch v := raw.(type) {
	case bool:
		return v, true
	case string:
		switch strings.ToLower(v) {
		case "true", "1", "yes", "y":
			return true, true
		case "false", "0", "no", "n":
			return false, true
		default:
			return false, false
		}
	default:
		return false, false
	}
}

func getSlice[T any](source map[string]any, key string) ([]T, bool) {
	if source == nil {
		return nil, false
	}
	raw, ok := source[key]
	if !ok || raw == nil {
		return nil, false
	}
	items, ok := raw.([]any)
	if !ok {
		return nil, false
	}
	result := make([]T, 0, len(items))
	for _, item := range items {
		switch v := item.(type) {
		case T:
			result = append(result, v)
		case string:
			val, ok := any(v).(T)
			if !ok {
				return nil, false
			}
			result = append(result, val)
		default:
			if converted, ok := any(fmt.Sprintf("%v", v)).(T); ok {
				result = append(result, converted)
			} else {
				return nil, false
			}
		}
	}
	return result, true
}
