package main

import (
	"context"
	"flag"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

func main() {
	noStream := flag.Bool("no-stream", false, "")
	model := flag.String("model", "gpt-4o", "")
	instructions := flag.String("instructions", "You are a helpful assistant", "")
	flag.Parse()

	agent := agents.New("Assistant").
		WithInstructions(*instructions).
		WithModel(*model)

	err := agents.RunDemoLoop(context.Background(), agent, !*noStream)
	if err != nil {
		panic(err)
	}
}
