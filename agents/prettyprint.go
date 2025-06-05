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
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

func indent(text string, indentLevel int) string {
	indentString := strings.Repeat("  ", indentLevel)

	var sb strings.Builder
	for line := range strings.Lines(text) {
		sb.WriteString(indentString)
		sb.WriteString(line)
	}
	return sb.String()
}

func finalOutputStr(result RunResultBase) string {
	switch finalOutput := result.FinalOutput.(type) {
	case nil:
		return "None"
	case string:
		return finalOutput
	case []byte:
		return string(finalOutput)
	default:
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		err := enc.Encode(finalOutput)
		if err != nil {
			return fmt.Sprintf("%+v", finalOutput)
		}
		return buf.String()
	}
}

func PrettyPrintResult(result RunResult) string {
	var sb strings.Builder

	sb.WriteString("RunResult:")
	_, _ = fmt.Fprintf(&sb, "\n- Last agent: Agent(name=%q, ...)", result.LastAgent().Name)

	_, _ = fmt.Fprintf(&sb, "\n- Final output (%T):\n", result.FinalOutput)
	sb.WriteString(indent(strings.TrimSuffix(finalOutputStr(result.RunResultBase), "\n"), 2))

	_, _ = fmt.Fprintf(&sb, "\n- %d new item(s)", len(result.NewItems))
	_, _ = fmt.Fprintf(&sb, "\n- %d raw response(s)", len(result.RawResponses))
	_, _ = fmt.Fprintf(&sb, "\n- %d input guardrail result(s)", len(result.InputGuardrailResults))
	_, _ = fmt.Fprintf(&sb, "\n- %d output guardrail result(s)", len(result.OutputGuardrailResults))
	sb.WriteString("\n(See `RunResult` for more details)")

	return sb.String()
}

func PrettyPrintRunResultStreaming(result RunResultStreaming) string {
	var sb strings.Builder

	sb.WriteString("RunResultStreaming:")
	_, _ = fmt.Fprintf(&sb, "\n- Current agent: Agent(name=%q, ...)", result.LastAgent().Name)
	_, _ = fmt.Fprintf(&sb, "\n- Current turn: %d", result.CurrentTurn)
	_, _ = fmt.Fprintf(&sb, "\n- Max turns: %d", result.MaxTurns)
	_, _ = fmt.Fprintf(&sb, "\n- Is complete: %v", result.IsComplete)

	_, _ = fmt.Fprintf(&sb, "\n- Final output (%T):\n", result.FinalOutput)
	sb.WriteString(indent(strings.TrimSuffix(finalOutputStr(result.RunResultBase), "\n"), 2))

	_, _ = fmt.Fprintf(&sb, "\n- %d new item(s)", len(result.NewItems))
	_, _ = fmt.Fprintf(&sb, "\n- %d raw response(s)", len(result.RawResponses))
	_, _ = fmt.Fprintf(&sb, "\n- %d input guardrail result(s)", len(result.InputGuardrailResults))
	_, _ = fmt.Fprintf(&sb, "\n- %d output guardrail result(s)", len(result.OutputGuardrailResults))
	sb.WriteString("\n(See `RunResultStreaming` for more details)")

	return sb.String()
}

func SimplePrettyJSONMarshal(v any) string {
	s, err := PrettyJSONMarshal(v)
	if err != nil {
		return fmt.Sprintf("<<%s>>", err)
	}
	return s
}

func PrettyJSONMarshal(v any) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	err := enc.Encode(v)
	return buf.String(), err
}
