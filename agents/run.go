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
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/packages/param"
)

const DefaultMaxTurns = 10

// DefaultRunner is the default Runner instance used by package-level Run
// helpers.
var DefaultRunner = Runner{}

// Runner executes agents using the configured RunConfig.
//
// The zero value is valid.
type Runner struct {
	Config RunConfig
}

// RunConfig configures settings for the entire agent run.
type RunConfig struct {
	// The model to use for the entire agent run. If set, will override the model set on every
	// agent. The model_provider passed in below must be able to resolve this model name.
	Model param.Opt[AgentModel]

	// Optional model provider to use when looking up string model names. Defaults to OpenAI (MultiProvider).
	ModelProvider ModelProvider

	// Optional global model settings. Any non-null or non-zero values will
	// override the agent-specific model settings.
	ModelSettings modelsettings.ModelSettings

	// Optional global input filter to apply to all handoffs. If `Handoff.InputFilter` is set, then that
	// will take precedence. The input filter allows you to edit the inputs that are sent to the new
	// agent. See the documentation in `Handoff.InputFilter` for more details.
	HandoffInputFilter HandoffInputFilter

	// A list of input guardrails to run on the initial run input.
	InputGuardrails []InputGuardrail

	// A list of output guardrails to run on the final output of the run.
	OutputGuardrails []OutputGuardrail

	// Optional maximum number of turns to run the agent for.
	// A turn is defined as one AI invocation (including any tool calls that might occur).
	// Default (when left zero): DefaultMaxTurns.
	MaxTurns uint64

	// Optional object that receives callbacks on various lifecycle events.
	Hooks RunHooks

	// Optional ID of the previous response, if using OpenAI models via the Responses API,
	// this allows you to skip passing in input from the previous turn.
	PreviousResponseID string
}

// Run executes startingAgent with the provided input using the DefaultRunner.
func Run(ctx context.Context, startingAgent *Agent, input string) (*RunResult, error) {
	return DefaultRunner.Run(ctx, startingAgent, input)
}

// RunStreamed executes startingAgent with the provided input using the
// DefaultRunner and returns a streaming result.
func RunStreamed(ctx context.Context, startingAgent *Agent, input string) (*RunResultStreaming, error) {
	return DefaultRunner.RunStreamed(ctx, startingAgent, input)
}

// RunResponseInputs executes startingAgent with the provided list of input items using the DefaultRunner.
func RunResponseInputs(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*RunResult, error) {
	return DefaultRunner.RunResponseInputs(ctx, startingAgent, input)
}

// RunResponseInputsStreamed executes startingAgent with the provided list of input items using the DefaultRunner
// and returns a streaming result.
func RunResponseInputsStreamed(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*RunResultStreaming, error) {
	return DefaultRunner.RunResponseInputsStreamed(ctx, startingAgent, input)
}

// Run executes startingAgent with the provided input string using the Runner configuration.
func (r Runner) Run(ctx context.Context, startingAgent *Agent, input string) (*RunResult, error) {
	return r.run(ctx, startingAgent, InputString(input))
}

// RunStreamed executes startingAgent with the provided input string using the Runner configuration and returns a streaming result.
func (r Runner) RunStreamed(ctx context.Context, startingAgent *Agent, input string) (*RunResultStreaming, error) {
	return r.runStreamed(ctx, startingAgent, InputString(input))
}

// RunResponseInputs executes startingAgent with the provided list of input items using the Runner configuration.
func (r Runner) RunResponseInputs(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*RunResult, error) {
	return r.run(ctx, startingAgent, InputItems(input))
}

// RunResponseInputsStreamed executes startingAgent with the provided list of input items using the Runner configuration and returns a streaming result.
func (r Runner) RunResponseInputsStreamed(ctx context.Context, startingAgent *Agent, input []TResponseInputItem) (*RunResultStreaming, error) {
	return r.runStreamed(ctx, startingAgent, InputItems(input))
}

// Run a workflow starting at the given agent. The agent will run in a loop until a final
// output is generated.
//
// The loop runs like so:
//  1. The agent is invoked with the given input.
//  2. If there is a final output, the loop terminates.
//  3. If there's a handoff, we run the loop again, with the new agent.
//  4. Else, we run tool calls (if any), and re-run the loop.
//
// In two cases, the agent run may return an error:
//  1. If the maxTurns is exceeded, a MaxTurnsExceededError is returned.
//  2. If a guardrail tripwire is triggered, a *GuardrailTripwireTriggeredError is returned.
//
// Note that only the first agent's input guardrails are run.
//
// It returns a run result containing all the inputs, guardrail results and the output of the last
// agent. Agents may perform handoffs, so we don't know the specific type of the output.
func (r Runner) run(ctx context.Context, startingAgent *Agent, input Input) (_ *RunResult, err error) {
	hooks := r.Config.Hooks
	if hooks == nil {
		hooks = NoOpRunHooks{}
	}

	toolUseTracker := NewAgentToolUseTracker()
	originalInput := CopyGeneralInput(input)
	currentTurn := uint64(0)

	maxTurns := r.Config.MaxTurns
	if maxTurns == 0 {
		maxTurns = DefaultMaxTurns
	}

	var (
		generatedItems         []RunItem
		modelResponses         []ModelResponse
		inputGuardrailResults  []InputGuardrailResult
		outputGuardrailResults []OutputGuardrailResult
	)

	ctx = usage.NewContext(ctx, usage.NewUsage())

	if startingAgent == nil {
		return nil, fmt.Errorf("StartingAgent must not be nil")
	}
	currentAgent := startingAgent
	shouldRunAgentStartHooks := true

	shouldGetAgentTools := true
	var allTools []Tool

	defer func() {
		if err != nil {
			var agentsErr *AgentsError
			if errors.As(err, &agentsErr) {
				agentsErr.RunData = &RunErrorDetails{
					Context:                ctx,
					Input:                  originalInput,
					NewItems:               generatedItems,
					RawResponses:           modelResponses,
					LastAgent:              currentAgent,
					InputGuardrailResults:  inputGuardrailResults,
					OutputGuardrailResults: outputGuardrailResults,
				}
			}
		}
	}()

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	for {
		if shouldGetAgentTools {
			var err error
			allTools, err = r.getAllTools(childCtx, currentAgent)
			if err != nil {
				return nil, err
			}
			shouldGetAgentTools = false
		}

		currentTurn += 1
		if currentTurn > maxTurns {
			return nil, MaxTurnsExceededErrorf("max turns %d exceeded", maxTurns)
		}
		slog.Debug(
			"Running agent",
			slog.String("agentName", currentAgent.Name),
			slog.Uint64("turn", currentTurn),
		)

		var turnResult *SingleStepResult

		if currentTurn == 1 {
			var wg sync.WaitGroup
			wg.Add(2)

			var guardrailsError error
			go func() {
				defer wg.Done()
				inputGuardrailResults, guardrailsError = r.runInputGuardrails(
					childCtx,
					startingAgent,
					slices.Concat(startingAgent.InputGuardrails, r.Config.InputGuardrails),
					CopyGeneralInput(input),
				)
				if guardrailsError != nil {
					cancel()
				}
			}()

			var turnError error
			go func() {
				defer wg.Done()
				turnResult, turnError = r.runSingleTurn(
					childCtx,
					currentAgent,
					allTools,
					originalInput,
					generatedItems,
					hooks,
					r.Config,
					shouldRunAgentStartHooks,
					toolUseTracker,
					r.Config.PreviousResponseID,
				)
				if turnError != nil {
					cancel()
				}
			}()

			wg.Wait()
			if err := errors.Join(turnError, guardrailsError); err != nil {
				return nil, err
			}
		} else {
			var err error
			turnResult, err = r.runSingleTurn(
				childCtx,
				currentAgent,
				allTools,
				originalInput,
				generatedItems,
				hooks,
				r.Config,
				shouldRunAgentStartHooks,
				toolUseTracker,
				r.Config.PreviousResponseID,
			)
			if err != nil {
				return nil, err
			}
		}

		shouldRunAgentStartHooks = false

		modelResponses = append(modelResponses, turnResult.ModelResponse)
		originalInput = turnResult.OriginalInput
		generatedItems = turnResult.GeneratedItems()

		switch nextStep := turnResult.NextStep.(type) {
		case NextStepFinalOutput:
			outputGuardrailResults, err = r.runOutputGuardrails(
				childCtx,
				slices.Concat(currentAgent.OutputGuardrails, r.Config.OutputGuardrails),
				currentAgent,
				nextStep.Output,
			)
			if err != nil {
				return nil, err
			}
			return &RunResult{
				Input:                  originalInput,
				NewItems:               generatedItems,
				RawResponses:           modelResponses,
				FinalOutput:            nextStep.Output,
				InputGuardrailResults:  inputGuardrailResults,
				OutputGuardrailResults: outputGuardrailResults,
				LastAgent:              currentAgent,
			}, nil
		case NextStepHandoff:
			currentAgent = nextStep.NewAgent
			allTools = nil
			shouldGetAgentTools = true
			shouldRunAgentStartHooks = true
		case NextStepRunAgain:
			// Nothing to do
		default:
			// This would be an unrecoverable implementation bug, so a panic is appropriate.
			panic(fmt.Errorf("unexpected NextStep type %T", nextStep))
		}
	}
}

// RunStreamed run a workflow starting at the given agent in streaming mode.
// The returned result object contains a method you can use to stream semantic
// events as they are generated.
//
// The agent will run in a loop until a final output is generated. The loop runs like so:
//  1. The agent is invoked with the given input.
//  2. If there is a final output, the loop terminates.
//  3. If there's a handoff, we run the loop again, with the new agent.
//  4. Else, we run tool calls (if any), and re-run the loop.
//
// In two cases, the agent run may return an error:
//  1. If the max_turns is exceeded, a MaxTurnsExceededError is returned.
//  2. If a guardrail tripwire is triggered, a *GuardrailTripwireTriggeredError is returned.
//
// Note that only the first agent's input guardrails are run.
//
// It returns a result object that contains data about the run, as well as a method to stream events.
func (r Runner) runStreamed(ctx context.Context, startingAgent *Agent, input Input) (*RunResultStreaming, error) {
	hooks := r.Config.Hooks
	if hooks == nil {
		hooks = NoOpRunHooks{}
	}

	maxTurns := r.Config.MaxTurns
	if maxTurns == 0 {
		maxTurns = DefaultMaxTurns
	}

	if startingAgent == nil {
		return nil, fmt.Errorf("StartingAgent must not be nil")
	}

	outputSchema := startingAgent.OutputSchema
	ctx = usage.NewContext(ctx, usage.NewUsage())

	streamedResult := newRunResultStreaming(ctx)
	streamedResult.setInput(CopyGeneralInput(input))
	streamedResult.setCurrentAgent(startingAgent)
	streamedResult.setMaxTurns(maxTurns)
	streamedResult.setCurrentAgentOutputSchema(outputSchema)

	// Kick off the actual agent loop in the background and return the streamed result object.
	streamedResult.createRunImplTask(ctx, func(ctx context.Context) error {
		return r.runStreamedImpl(
			ctx,
			input,
			streamedResult,
			startingAgent,
			maxTurns,
			hooks,
			r.Config,
			r.Config.PreviousResponseID,
		)
	})

	return streamedResult, nil
}

func (r Runner) runInputGuardrailsWithQueue(
	ctx context.Context,
	agent *Agent,
	guardrails []InputGuardrail,
	input Input,
	streamedResult *RunResultStreaming,
) error {
	queue := streamedResult.inputGuardrailQueue

	guardrailResults := make([]InputGuardrailResult, len(guardrails))
	guardrailErrors := make([]error, len(guardrails))

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(len(guardrails))

	// We'll run the guardrails and push them onto the queue as they complete
	for i, guardrail := range guardrails {
		go func() {
			defer wg.Done()

			result, err := RunImpl().RunSingleInputGuardrail(childCtx, agent, guardrail, input)
			if err != nil {
				cancel()
				guardrailErrors[i] = fmt.Errorf("failed to run input guardrail %s: %w", guardrail.Name, err)
			} else {
				guardrailResults[i] = result
				queue.Put(result)
			}
		}()
	}

	wg.Wait()
	if err := errors.Join(guardrailErrors...); err != nil {
		return err
	}

	streamedResult.setInputGuardrailResults(guardrailResults)
	return nil
}

func (r Runner) runStreamedImpl(
	ctx context.Context,
	startingInput Input,
	streamedResult *RunResultStreaming,
	startingAgent *Agent,
	maxTurns uint64,
	hooks RunHooks,
	runConfig RunConfig,
	previousResponseID string,
) (err error) {
	currentAgent := startingAgent

	defer func() {
		if err != nil {
			var agentsErr *AgentsError
			if errors.As(err, &agentsErr) {
				agentsErr.RunData = &RunErrorDetails{
					Context:                ctx,
					Input:                  streamedResult.Input(),
					NewItems:               streamedResult.NewItems(),
					RawResponses:           streamedResult.RawResponses(),
					LastAgent:              currentAgent,
					InputGuardrailResults:  streamedResult.InputGuardrailResults(),
					OutputGuardrailResults: streamedResult.OutputGuardrailResults(),
				}
			}

			streamedResult.markAsComplete()
			streamedResult.eventQueue.Put(queueCompleteSentinel{})
		}
	}()

	currentTurn := uint64(0)
	shouldRunAgentStartHooks := true
	toolUseTracker := NewAgentToolUseTracker()

	streamedResult.eventQueue.Put(AgentUpdatedStreamEvent{
		NewAgent: currentAgent,
		Type:     "agent_updated_stream_event",
	})

	shouldGetAgentTools := true
	var allTools []Tool

	for !streamedResult.IsComplete() {
		if shouldGetAgentTools {
			allTools, err = r.getAllTools(ctx, currentAgent)
			if err != nil {
				return err
			}
			shouldGetAgentTools = false
		}

		currentTurn += 1
		streamedResult.setCurrentTurn(currentTurn)

		if currentTurn > maxTurns {
			streamedResult.eventQueue.Put(queueCompleteSentinel{})
			break
		}

		if currentTurn == 1 {
			// Run the input guardrails in the background and put the results on the queue
			streamedResult.createInputGuardrailsTask(ctx, func(ctx context.Context) error {
				return r.runInputGuardrailsWithQueue(
					ctx,
					startingAgent,
					slices.Concat(startingAgent.InputGuardrails, runConfig.InputGuardrails),
					InputItems(ItemHelpers().InputToNewInputList(startingInput)),
					streamedResult,
				)
			})
		}

		turnResult, err := r.runSingleTurnStreamed(
			ctx,
			streamedResult,
			currentAgent,
			hooks,
			runConfig,
			shouldRunAgentStartHooks,
			toolUseTracker,
			allTools,
			previousResponseID,
		)
		if err != nil {
			return err
		}
		shouldRunAgentStartHooks = false

		streamedResult.appendRawResponses(turnResult.ModelResponse)
		streamedResult.setInput(turnResult.OriginalInput)
		streamedResult.setNewItems(turnResult.GeneratedItems())

		switch nextStep := turnResult.NextStep.(type) {
		case NextStepFinalOutput:
			streamedResult.createOutputGuardrailsTask(ctx, func(ctx context.Context) outputGuardrailsTaskResult {
				result, err := r.runOutputGuardrails(
					ctx,
					slices.Concat(currentAgent.OutputGuardrails, runConfig.OutputGuardrails),
					currentAgent,
					nextStep.Output,
				)
				return outputGuardrailsTaskResult{Result: result, Err: err}
			})

			taskResult := streamedResult.getOutputGuardrailsTask().Await()

			var outputGuardrailResults []OutputGuardrailResult
			if taskResult.Canceled {
				return NewTaskCanceledError("output guardrails task has been canceled")
			}
			if taskResult.Result.Err == nil {
				// Errors will be checked in the stream-events loop
				outputGuardrailResults = taskResult.Result.Result
			}

			streamedResult.setOutputGuardrailResults(outputGuardrailResults)
			streamedResult.setFinalOutput(nextStep.Output)
			streamedResult.markAsComplete()
			streamedResult.eventQueue.Put(queueCompleteSentinel{})
		case NextStepHandoff:
			currentAgent = nextStep.NewAgent
			allTools = nil
			shouldGetAgentTools = true
			shouldRunAgentStartHooks = true
			streamedResult.eventQueue.Put(AgentUpdatedStreamEvent{
				NewAgent: currentAgent,
				Type:     "agent_updated_stream_event",
			})
		case NextStepRunAgain:
			// Nothing to do
		default:
			// This would be an unrecoverable implementation bug, so a panic is appropriate.
			panic(fmt.Errorf("unexpected NextStep type %T", nextStep))
		}
	}

	streamedResult.markAsComplete()
	return nil
}

func (r Runner) runSingleTurnStreamed(
	ctx context.Context,
	streamedResult *RunResultStreaming,
	agent *Agent,
	hooks RunHooks,
	runConfig RunConfig,
	shouldRunAgentStartHooks bool,
	toolUseTracker *AgentToolUseTracker,
	allTools []Tool,
	previousResponseID string,
) (*SingleStepResult, error) {
	if shouldRunAgentStartHooks {
		childCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		var wg sync.WaitGroup
		var hooksErrors [2]error

		wg.Add(1)
		go func() {
			defer wg.Done()
			err := hooks.OnAgentStart(childCtx, agent)
			if err != nil {
				cancel()
				hooksErrors[0] = fmt.Errorf("RunHooks.OnAgentStart failed: %w", err)
			}
		}()

		if agent.Hooks != nil {
			wg.Add(1)
			go func() {
				defer wg.Done()
				err := agent.Hooks.OnStart(childCtx, agent)
				if err != nil {
					cancel()
					hooksErrors[1] = fmt.Errorf("AgentHooks.OnStart failed: %w", err)
				}
			}()
		}

		wg.Wait()
		if err := errors.Join(hooksErrors[:]...); err != nil {
			return nil, err
		}
	}

	outputSchema := agent.OutputSchema

	streamedResult.setCurrentAgent(agent)
	streamedResult.setCurrentAgentOutputSchema(outputSchema)

	systemPrompt, err := agent.GetSystemPrompt(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get system prompt: %w", err)
	}

	handoffs, err := r.getHandoffs(agent)
	if err != nil {
		return nil, err
	}

	model, err := r.getModel(agent, runConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get model: %w", err)
	}
	modelSettings := agent.ModelSettings.Resolve(runConfig.ModelSettings)
	modelSettings = RunImpl().MaybeResetToolChoice(agent, toolUseTracker, modelSettings)

	var finalResponse *ModelResponse

	input := ItemHelpers().InputToNewInputList(streamedResult.Input())
	for _, item := range streamedResult.NewItems() {
		input = append(input, item.ToInputItem())
	}

	// 1. Stream the output events
	stream, err := model.StreamResponse(ctx, ModelStreamResponseParams{
		SystemInstructions: systemPrompt,
		Input:              InputItems(input),
		ModelSettings:      modelSettings,
		Tools:              allTools,
		OutputSchema:       outputSchema,
		Handoffs:           handoffs,
		PreviousResponseID: previousResponseID,
	})
	if err != nil {
		return nil, fmt.Errorf("stream response error: %w", err)
	}

	eventErrors := make([]error, 0)
	for event, eventErr := range stream {
		if eventErr != nil {
			eventErrors = append(eventErrors, eventErr)
			continue
		}
		if event.Type == "response.completed" {
			u := usage.NewUsage()
			if !reflect.ValueOf(event.Response.Usage).IsZero() {
				*u = usage.Usage{
					Requests:            1,
					InputTokens:         uint64(event.Response.Usage.InputTokens),
					InputTokensDetails:  event.Response.Usage.InputTokensDetails,
					OutputTokens:        uint64(event.Response.Usage.OutputTokens),
					OutputTokensDetails: event.Response.Usage.OutputTokensDetails,
					TotalTokens:         uint64(event.Response.Usage.TotalTokens),
				}
			}
			finalResponse = &ModelResponse{
				Output:     event.Response.Output,
				Usage:      u,
				ResponseID: event.Response.ID,
			}
			if contextUsage, _ := usage.FromContext(ctx); contextUsage != nil {
				contextUsage.Add(u)
			}
		}
		streamedResult.eventQueue.Put(RawResponsesStreamEvent{
			Data: *event,
			Type: "raw_response_event",
		})
	}
	if err = errors.Join(eventErrors...); err != nil {
		return nil, fmt.Errorf("stream event errors: %w", err)
	}

	// 2. At this point, the streaming is complete for this turn of the agent loop.
	if finalResponse == nil {
		return nil, NewModelBehaviorError("Model did not produce a final response!")
	}

	// 3. Now, we can process the turn as we do in the non-streaming case
	singleStepResult, err := r.getSingleStepResultFromResponse(
		ctx,
		agent,
		allTools,
		streamedResult.Input(),
		streamedResult.NewItems(),
		*finalResponse,
		outputSchema,
		handoffs,
		hooks,
		runConfig,
		toolUseTracker,
	)
	if err != nil {
		return nil, err
	}

	RunImpl().StreamStepResultToQueue(*singleStepResult, streamedResult.eventQueue)
	return singleStepResult, nil
}

func (r Runner) runSingleTurn(
	ctx context.Context,
	agent *Agent,
	allTools []Tool,
	originalInput Input,
	generatedItems []RunItem,
	hooks RunHooks,
	runConfig RunConfig,
	shouldRunAgentStartHooks bool,
	toolUseTracker *AgentToolUseTracker,
	previousResponseID string,
) (*SingleStepResult, error) {
	// Ensure we run the hooks before anything else
	if shouldRunAgentStartHooks {
		childCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		var wg sync.WaitGroup
		var hooksErrors [2]error

		wg.Add(1)
		go func() {
			defer wg.Done()
			err := hooks.OnAgentStart(childCtx, agent)
			if err != nil {
				cancel()
				hooksErrors[0] = fmt.Errorf("RunHooks.OnAgentStart failed: %w", err)
			}
		}()

		if agent.Hooks != nil {
			wg.Add(1)
			go func() {
				defer wg.Done()
				err := agent.Hooks.OnStart(childCtx, agent)
				if err != nil {
					cancel()
					hooksErrors[1] = fmt.Errorf("AgentHooks.OnStart failed: %w", err)
				}
			}()
		}

		wg.Wait()
		if err := errors.Join(hooksErrors[:]...); err != nil {
			return nil, err
		}
	}

	systemPrompt, err := agent.GetSystemPrompt(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get system prompt: %w", err)
	}

	handoffs, err := r.getHandoffs(agent)
	if err != nil {
		return nil, err
	}

	input := ItemHelpers().InputToNewInputList(originalInput)
	for _, generatedItem := range generatedItems {
		input = append(input, generatedItem.ToInputItem())
	}

	newResponse, err := r.getNewResponse(
		ctx,
		agent,
		systemPrompt,
		input,
		agent.OutputSchema,
		allTools,
		handoffs,
		runConfig,
		toolUseTracker,
		previousResponseID,
	)
	if err != nil {
		return nil, err
	}

	return r.getSingleStepResultFromResponse(
		ctx,
		agent,
		allTools,
		originalInput,
		generatedItems,
		*newResponse,
		agent.OutputSchema,
		handoffs,
		hooks,
		runConfig,
		toolUseTracker,
	)
}

func (Runner) getSingleStepResultFromResponse(
	ctx context.Context,
	agent *Agent,
	allTools []Tool,
	originalInput Input,
	preStepItems []RunItem,
	newResponse ModelResponse,
	outputSchema AgentOutputSchemaInterface,
	handoffs []Handoff,
	hooks RunHooks,
	runConfig RunConfig,
	toolUseTracker *AgentToolUseTracker,
) (*SingleStepResult, error) {
	processedResponse, err := RunImpl().ProcessModelResponse(
		agent,
		allTools,
		newResponse,
		handoffs,
	)
	if err != nil {
		return nil, err
	}

	toolUseTracker.AddToolUse(agent, processedResponse.ToolsUsed)

	return RunImpl().ExecuteToolsAndSideEffects(
		ctx,
		agent,
		originalInput,
		preStepItems,
		newResponse,
		*processedResponse,
		outputSchema,
		hooks,
		runConfig,
	)
}

func (Runner) runInputGuardrails(
	ctx context.Context,
	agent *Agent,
	guardrails []InputGuardrail,
	input Input,
) ([]InputGuardrailResult, error) {
	if len(guardrails) == 0 {
		return nil, nil
	}

	guardrailResults := make([]InputGuardrailResult, len(guardrails))
	guardrailErrors := make([]error, len(guardrails))
	var tripwireErr atomic.Pointer[InputGuardrailTripwireTriggeredError]

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(len(guardrails))

	for i, guardrail := range guardrails {
		go func() {
			defer wg.Done()

			result, err := RunImpl().RunSingleInputGuardrail(childCtx, agent, guardrail, input)
			if err != nil {
				cancel()
				guardrailErrors[i] = fmt.Errorf("failed to run input guardrail %s: %w", guardrail.Name, err)
			} else if result.Output.TripwireTriggered {
				cancel() // Cancel all guardrail tasks if a tripwire is triggered.
				err := NewInputGuardrailTripwireTriggeredError(result)
				tripwireErr.Store(&err)
			} else {
				guardrailResults[i] = result
			}
		}()
	}

	wg.Wait()

	if err := tripwireErr.Load(); err != nil {
		return nil, *err
	}
	if err := errors.Join(guardrailErrors...); err != nil {
		return nil, err
	}

	return guardrailResults, nil
}

func (Runner) runOutputGuardrails(
	ctx context.Context,
	guardrails []OutputGuardrail,
	agent *Agent,
	agentOutput any,
) ([]OutputGuardrailResult, error) {
	if len(guardrails) == 0 {
		return nil, nil
	}

	guardrailResults := make([]OutputGuardrailResult, len(guardrails))
	guardrailErrors := make([]error, len(guardrails))
	var tripwireErr atomic.Pointer[OutputGuardrailTripwireTriggeredError]

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(len(guardrails))

	for i, guardrail := range guardrails {
		go func() {
			defer wg.Done()

			result, err := RunImpl().RunSingleOutputGuardrail(childCtx, guardrail, agent, agentOutput)
			if err != nil {
				cancel()
				guardrailErrors[i] = fmt.Errorf("failed to run output guardrail %s: %w", guardrail.Name, err)
			} else if result.Output.TripwireTriggered {
				cancel() // Cancel all guardrail tasks if a tripwire is triggered.
				err := NewOutputGuardrailTripwireTriggeredError(result)
				tripwireErr.Store(&err)
			} else {
				guardrailResults[i] = result
			}
		}()
	}

	wg.Wait()

	if err := tripwireErr.Load(); err != nil {
		return nil, *err
	}
	if err := errors.Join(guardrailErrors...); err != nil {
		return nil, err
	}

	return guardrailResults, nil
}

func (r Runner) getNewResponse(
	ctx context.Context,
	agent *Agent,
	systemPrompt param.Opt[string],
	input []TResponseInputItem,
	outputSchema AgentOutputSchemaInterface,
	allTools []Tool,
	handoffs []Handoff,
	runConfig RunConfig,
	toolUseTracker *AgentToolUseTracker,
	previousResponseID string,
) (*ModelResponse, error) {
	model, err := r.getModel(agent, runConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get model: %w", err)
	}

	modelSettings := agent.ModelSettings.Resolve(runConfig.ModelSettings)
	modelSettings = RunImpl().MaybeResetToolChoice(agent, toolUseTracker, modelSettings)

	newResponse, err := model.GetResponse(ctx, ModelGetResponseParams{
		SystemInstructions: systemPrompt,
		Input:              InputItems(input),
		ModelSettings:      modelSettings,
		Tools:              allTools,
		OutputSchema:       outputSchema,
		Handoffs:           handoffs,
		PreviousResponseID: previousResponseID,
	})
	if err != nil {
		return nil, err
	}

	if contextUsage, _ := usage.FromContext(ctx); contextUsage != nil {
		contextUsage.Add(newResponse.Usage)
	}

	return newResponse, err
}

func (Runner) getHandoffs(agent *Agent) ([]Handoff, error) {
	handoffs := make([]Handoff, 0, len(agent.Handoffs)+len(agent.AgentHandoffs))
	for _, h := range agent.Handoffs {
		handoffs = append(handoffs, h)
	}
	for _, a := range agent.AgentHandoffs {
		h, err := SafeHandoffFromAgent(HandoffFromAgentParams{Agent: a})
		if err != nil {
			return nil, fmt.Errorf("failed to make Handoff from Agent %q: %w", a.Name, err)
		}
		handoffs = append(handoffs, *h)
	}
	return handoffs, nil
}

func (Runner) getAllTools(ctx context.Context, agent *Agent) ([]Tool, error) {
	return agent.GetAllTools(ctx)
}

func (r Runner) getModel(agent *Agent, runConfig RunConfig) (Model, error) {
	modelProvider := runConfig.ModelProvider
	if modelProvider == nil {
		modelProvider = NewMultiProvider(NewMultiProviderParams{})
	}

	if runConfig.Model.Valid() {
		runConfigModel := runConfig.Model.Value
		if v, ok := runConfigModel.SafeModel(); ok {
			return v, nil
		}
		return modelProvider.GetModel(runConfigModel.ModelName())
	}

	if agent.Model.Valid() {
		agentModel := agent.Model.Value
		if v, ok := agentModel.SafeModel(); ok {
			return v, nil
		}
		return modelProvider.GetModel(agentModel.ModelName())
	}

	return modelProvider.GetModel("")
}
