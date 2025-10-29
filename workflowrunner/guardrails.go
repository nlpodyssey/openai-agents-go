package workflowrunner

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

type inputGuardrailBuilder func(context.Context, GuardrailDeclaration) (agents.InputGuardrail, error)
type outputGuardrailBuilder func(context.Context, GuardrailDeclaration) (agents.OutputGuardrail, error)

var inputGuardrailRegistry = map[string]inputGuardrailBuilder{
	"math_homework_input":   newMathHomeworkInputGuardrail,
	"basic_profanity_input": newProfanityInputGuardrail,
}

var outputGuardrailRegistry = map[string]outputGuardrailBuilder{
	"phone_number_output":    newPhoneNumberOutputGuardrail,
	"basic_profanity_output": newProfanityOutputGuardrail,
}

func buildInputGuardrails(ctx context.Context, decls []GuardrailDeclaration) ([]agents.InputGuardrail, error) {
	if len(decls) == 0 {
		return nil, nil
	}
	guardrails := make([]agents.InputGuardrail, 0, len(decls))
	for _, decl := range decls {
		builder, ok := inputGuardrailRegistry[strings.ToLower(decl.Name)]
		if !ok {
			return nil, fmt.Errorf("unknown input guardrail %q", decl.Name)
		}
		gr, err := builder(ctx, decl)
		if err != nil {
			return nil, fmt.Errorf("build input guardrail %q: %w", decl.Name, err)
		}
		guardrails = append(guardrails, gr)
	}
	return guardrails, nil
}

func buildOutputGuardrails(ctx context.Context, decls []GuardrailDeclaration) ([]agents.OutputGuardrail, error) {
	if len(decls) == 0 {
		return nil, nil
	}
	guardrails := make([]agents.OutputGuardrail, 0, len(decls))
	for _, decl := range decls {
		builder, ok := outputGuardrailRegistry[strings.ToLower(decl.Name)]
		if !ok {
			return nil, fmt.Errorf("unknown output guardrail %q", decl.Name)
		}
		gr, err := builder(ctx, decl)
		if err != nil {
			return nil, fmt.Errorf("build output guardrail %q: %w", decl.Name, err)
		}
		guardrails = append(guardrails, gr)
	}
	return guardrails, nil
}

func newMathHomeworkInputGuardrail(_ context.Context, _ GuardrailDeclaration) (agents.InputGuardrail, error) {
	keywords := []string{"math homework", "solve this math", "algebra homework", "calculus homework"}
	return agents.InputGuardrail{
		Name: "math_homework_input",
		GuardrailFunction: func(_ context.Context, _ *agents.Agent, input agents.Input) (agents.GuardrailFunctionOutput, error) {
			text := normalizeInput(input)
			for _, keyword := range keywords {
				if strings.Contains(text, keyword) {
					return agents.GuardrailFunctionOutput{
						TripwireTriggered: true,
						OutputInfo: map[string]any{
							"keyword": keyword,
						},
					}, nil
				}
			}
			return agents.GuardrailFunctionOutput{TripwireTriggered: false}, nil
		},
	}, nil
}

func newProfanityInputGuardrail(_ context.Context, _ GuardrailDeclaration) (agents.InputGuardrail, error) {
	return agents.InputGuardrail{
		Name: "basic_profanity_input",
		GuardrailFunction: func(_ context.Context, _ *agents.Agent, input agents.Input) (agents.GuardrailFunctionOutput, error) {
			text := normalizeInput(input)
			triggered := containsProfanity(text)
			return agents.GuardrailFunctionOutput{
				TripwireTriggered: triggered,
				OutputInfo: map[string]any{
					"detected": triggered,
				},
			}, nil
		},
	}, nil
}

func newPhoneNumberOutputGuardrail(_ context.Context, decl GuardrailDeclaration) (agents.OutputGuardrail, error) {
	regexPattern := `\b(\+?\d{1,3}[-.\s]?)?(\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b`
	if custom, ok := decl.Config["pattern"].(string); ok && custom != "" {
		regexPattern = custom
	}
	expr, err := regexp.Compile(regexPattern)
	if err != nil {
		return agents.OutputGuardrail{}, fmt.Errorf("invalid regex pattern: %w", err)
	}
	return agents.OutputGuardrail{
		Name: "phone_number_output",
		GuardrailFunction: func(_ context.Context, _ *agents.Agent, output any) (agents.GuardrailFunctionOutput, error) {
			str, ok := output.(string)
			if !ok {
				return agents.GuardrailFunctionOutput{TripwireTriggered: false}, nil
			}
			triggered := expr.MatchString(str)
			return agents.GuardrailFunctionOutput{
				TripwireTriggered: triggered,
				OutputInfo: map[string]any{
					"pattern": regexPattern,
				},
			}, nil
		},
	}, nil
}

func newProfanityOutputGuardrail(_ context.Context, _ GuardrailDeclaration) (agents.OutputGuardrail, error) {
	return agents.OutputGuardrail{
		Name: "basic_profanity_output",
		GuardrailFunction: func(_ context.Context, _ *agents.Agent, output any) (agents.GuardrailFunctionOutput, error) {
			str, ok := output.(string)
			if !ok {
				return agents.GuardrailFunctionOutput{TripwireTriggered: false}, nil
			}
			triggered := containsProfanity(strings.ToLower(str))
			return agents.GuardrailFunctionOutput{
				TripwireTriggered: triggered,
				OutputInfo: map[string]any{
					"detected": triggered,
				},
			}, nil
		},
	}, nil
}

func normalizeInput(input agents.Input) string {
	switch v := input.(type) {
	case agents.InputString:
		return strings.ToLower(v.String())
	case agents.InputItems:
		var sb strings.Builder
		for _, item := range v {
			if msg := item.OfMessage; msg != nil {
				if msg.Content.OfString.Valid() {
					text := msg.Content.OfString.Value
					if text != "" {
						if sb.Len() > 0 {
							sb.WriteString(" ")
						}
						sb.WriteString(strings.ToLower(text))
					}
				}
			}
		}
		return sb.String()
	default:
		return ""
	}
}

func containsProfanity(text string) bool {
	profanities := []string{"damn", "hell", "shit", "fuck"}
	for _, bad := range profanities {
		if strings.Contains(text, bad) {
			return true
		}
	}
	return false
}
