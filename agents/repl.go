package agents

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

// RunDemoLoop runs a simple REPL loop with the given agent.
//
// This utility allows quick manual testing and debugging of an agent from the
// command line. Conversation state is preserved across turns. Enter "exit"
// or "quit" to stop the loop.
func RunDemoLoop(ctx context.Context, agent *Agent, stream bool) error {
	return RunDemoLoopRW(ctx, agent, stream, os.Stdin, os.Stdout)
}

func RunDemoLoopRW(ctx context.Context, agent *Agent, stream bool, r io.Reader, w io.Writer) error {
	currentAgent := agent
	var inputItems []TResponseInputItem

	writeAndFlush := func(s string) (err error) {
		if _, err = w.Write([]byte(s)); err != nil {
			return err
		}
		if flusher, ok := w.(interface{ Flush() error }); ok {
			_ = flusher.Flush()
		} else if syncer, ok := w.(interface{ Sync() error }); ok {
			_ = syncer.Sync()
		}
		return nil
	}

	bufReader := bufio.NewReader(r)

	for {
		if err := writeAndFlush("> "); err != nil {
			return err
		}

		line, _, err := bufReader.ReadLine()
		if errors.Is(err, io.EOF) {
			if err = writeAndFlush("\n"); err != nil {
				return err
			}
			break
		}
		if err != nil {
			return err
		}
		userInput := string(line)

		if v := strings.ToLower(strings.TrimSpace(userInput)); v == "exit" || v == "quit" {
			break
		} else if v == "" {
			continue
		}

		inputItems = append(inputItems, TResponseInputItem{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt(userInput),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		})

		if stream {
			result, err := RunInputsStreamed(ctx, currentAgent, inputItems)
			if err != nil {
				return err
			}

			err = result.StreamEvents(func(event StreamEvent) error {
				switch e := event.(type) {
				case RawResponsesStreamEvent:
					if e.Data.Type == "response.output_text.delta" {
						return writeAndFlush(e.Data.Delta)
					}
				case RunItemStreamEvent:
					switch item := e.Item.(type) {
					case ToolCallItem:
						return writeAndFlush("\n[tool called]\n")
					case ToolCallOutputItem:
						return writeAndFlush(fmt.Sprintf("\n[tool output: %+v]\n", item.Output))
					}
				case AgentUpdatedStreamEvent:
					return writeAndFlush(fmt.Sprintf("\n[Agent updated: %s]\n", e.NewAgent.Name))
				}
				return nil
			})
			if err != nil {
				return err
			}
			if err = writeAndFlush("\n"); err != nil {
				return err
			}

			currentAgent = result.LastAgent()
			inputItems = result.ToInputList()
		} else {
			result, err := RunInputs(ctx, currentAgent, inputItems)
			if err != nil {
				return err
			}
			if result.FinalOutput != nil {
				if err = writeAndFlush(fmt.Sprintln(result.FinalOutput)); err != nil {
					return err
				}
			}

			currentAgent = result.LastAgent
			inputItems = result.ToInputList()
		}
	}

	return nil
}
