package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

/*
NOTE: This example will not work out of the box, because the default prompt ID will not be available
in your project.

To use it, please:
1. Go to https://platform.openai.com/playground/prompts
2. Create a new prompt variable, `poem_style`.
3. Create a system prompt with the content:
```
Write a poem in {{poem_style}}
```
4. Run the example with the `--prompt-id` flag.
*/

const DefaultPromptID = "pmpt_6850729e8ba481939fd439e058c69ee004afaa19c520b78b"

type PoemStyle string

func (ps PoemStyle) String() string { return string(ps) }

const (
	PoemStyleLimerick PoemStyle = "limerick"
	PoemStyleHaiku    PoemStyle = "haiku"
	PoemStyleBallad   PoemStyle = "ballad"
)

var PoemStyles = []PoemStyle{
	PoemStyleLimerick,
	PoemStyleHaiku,
	PoemStyleBallad,
}

type DynamicContext struct {
	PromptID  string
	PoemStyle PoemStyle
}

type dynamicContextKey struct{}

func NewDynamicContext(promptID string) *DynamicContext {
	poemStyle := PoemStyles[rand.Intn(len(PoemStyles))]
	fmt.Printf("[debug] DynamicContext initialized with poem style: %s\n", poemStyle)
	return &DynamicContext{
		PromptID:  promptID,
		PoemStyle: poemStyle,
	}
}

func getDynamicPrompt(ctx context.Context, agent *agents.Agent) (agents.Prompt, error) {
	dynamicContext := ctx.Value(dynamicContextKey{}).(*DynamicContext)
	return agents.Prompt{
		ID:      dynamicContext.PromptID,
		Version: param.NewOpt("1"),
		Variables: map[string]responses.ResponsePromptVariableUnionParam{
			"poem_style": {
				OfString: param.NewOpt(dynamicContext.PoemStyle.String()),
			},
		},
	}, nil
}

func runDynamicPromptExample(promptID string) {
	dynamicContext := NewDynamicContext(promptID)
	ctx := context.WithValue(context.Background(), dynamicContextKey{}, dynamicContext)

	agent := agents.New("Assistant").
		WithPrompt(agents.DynamicPromptFunction(getDynamicPrompt)).
		WithModel("gpt-4o")

	result, err := agents.Run(ctx, agent, "Tell me about recursion in programming.")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
}

func runStaticPromptExample(promptID string) {
	agent := agents.New("Assistant").
		WithPrompt(agents.Prompt{
			ID:      promptID,
			Version: param.NewOpt("1"),
			Variables: map[string]responses.ResponsePromptVariableUnionParam{
				"poem_style": {
					OfString: param.NewOpt(PoemStyleLimerick.String()),
				},
			},
		}).
		WithModel("gpt-4o")

	result, err := agents.Run(context.Background(), agent, "Tell me about recursion in programming.")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
}

func main() {
	dynamic := flag.Bool("dynamic", false, "")
	promptID := flag.String("prompt-id", DefaultPromptID, "")
	flag.Parse()

	if *dynamic {
		runDynamicPromptExample(*promptID)
	} else {
		runStaticPromptExample(*promptID)
	}
}
