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

	"github.com/nlpodyssey/openai-agents-go/asyncqueue"
	"github.com/nlpodyssey/openai-agents-go/asynctask"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

type runner struct{}

func Runner() runner { return runner{} }

const DefaultMaxTurns = 10

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
}

type RunParams struct {
	// The starting agent to run.
	StartingAgent *Agent

	// The initial input to the agent.
	// You can pass a single string for a user message, or a list of input items.
	Input Input

	// Optional context to run the agent with.
	Context any

	// Optional maximum number of turns to run the agent for.
	// A turn is defined as one AI invocation (including any tool calls that might occur).
	// Default (when left zero): DefaultMaxTurns.
	MaxTurns uint64

	// Optional object that receives callbacks on various lifecycle events.
	Hooks RunHooks

	// Optional global settings for the entire agent run.
	RunConfig RunConfig

	// Optional ID of the previous response, if using OpenAI models via the Responses API,
	// this allows you to skip passing in input from the previous turn.
	PreviousResponseID string
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
// In two cases, the agent may raise an exception:
//  1. If the maxTurns is exceeded, a MaxTurnsExceededError is returned.
//  2. If a guardrail tripwire is triggered, a *GuardrailTripwireTriggeredError is returned.
//
// Note that only the first agent's input guardrails are run.
//
// It returns a run result containing all the inputs, guardrail results and the output of the last
// agent. Agents may perform handoffs, so we don't know the specific type of the output.
func (r runner) Run(ctx context.Context, params RunParams) (*RunResult, error) {
	hooks := params.Hooks
	if hooks == nil {
		hooks = NoOpRunHooks{}
	}

	toolUseTracker := NewAgentToolUseTracker()
	originalInput := CopyGeneralInput(params.Input)
	currentTurn := uint64(0)

	maxTurns := params.MaxTurns
	if maxTurns == 0 {
		maxTurns = DefaultMaxTurns
	}

	var (
		generatedItems        []RunItem
		modelResponses        []ModelResponse
		inputGuardrailResults []InputGuardrailResult
	)

	contextWrapper := runcontext.NewWrapper(params.Context)

	if params.StartingAgent == nil {
		return nil, fmt.Errorf("StartingAgent must not be nil")
	}
	currentAgent := params.StartingAgent
	shouldRunAgentStartHooks := true

	shouldGetAgentTools := true
	var allTools []Tool

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	for {
		if shouldGetAgentTools {
			allTools = r.getAllTools(currentAgent)
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
					params.StartingAgent,
					slices.Concat(params.StartingAgent.InputGuardrails, params.RunConfig.InputGuardrails),
					CopyGeneralInput(params.Input),
					contextWrapper,
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
					contextWrapper,
					params.RunConfig,
					shouldRunAgentStartHooks,
					toolUseTracker,
					params.PreviousResponseID,
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
				contextWrapper,
				params.RunConfig,
				shouldRunAgentStartHooks,
				toolUseTracker,
				params.PreviousResponseID,
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
			outputGuardrailResults, err := r.runOutputGuardrails(
				childCtx,
				slices.Concat(currentAgent.OutputGuardrails, params.RunConfig.OutputGuardrails),
				currentAgent,
				nextStep.Output,
				contextWrapper,
			)
			if err != nil {
				return nil, err
			}
			return &RunResult{
				RunResultBase: RunResultBase{
					Input:                  originalInput,
					NewItems:               generatedItems,
					RawResponses:           modelResponses,
					FinalOutput:            nextStep.Output,
					InputGuardrailResults:  inputGuardrailResults,
					OutputGuardrailResults: outputGuardrailResults,
					ContextWrapper:         contextWrapper,
				},
				lastAgent: currentAgent,
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

type RunStreamedParams struct {
	// The starting agent to run.
	StartingAgent *Agent

	// The initial input to the agent.
	// You can pass a single string for a user message, or a list of input items.
	Input Input

	// Optional context to run the agent with.
	Context any

	// Optional maximum number of turns to run the agent for.
	// A turn is defined as one AI invocation (including any tool calls that might occur).
	// Default (when left zero): DefaultMaxTurns.
	MaxTurns uint64

	// An object that receives callbacks on various lifecycle events.
	Hooks RunHooks

	// Optional global settings for the entire agent run.
	RunConfig RunConfig

	// Optional ID of the previous response, if using OpenAI models via the Responses API,
	// this allows you to skip passing in input from the previous turn.
	PreviousResponseID string
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
// In two cases, the agent may raise an exception:
//  1. If the max_turns is exceeded, a MaxTurnsExceededError is raised.
//  2. If a guardrail tripwire is triggered, a *GuardrailTripwireTriggeredError is returned.
//
// Note that only the first agent's input guardrails are run.
//
// It returns a result object that contains data about the run, as well as a method to stream events.
func (r runner) RunStreamed(ctx context.Context, params RunStreamedParams) (*RunResultStreaming, error) {
	hooks := params.Hooks
	if hooks == nil {
		hooks = NoOpRunHooks{}
	}

	maxTurns := params.MaxTurns
	if maxTurns == 0 {
		maxTurns = DefaultMaxTurns
	}

	if params.StartingAgent == nil {
		return nil, fmt.Errorf("StartingAgent must not be nil")
	}

	outputSchema := params.StartingAgent.OutputSchema
	contextWrapper := runcontext.NewWrapper(params.Context)

	streamedResult := &RunResultStreaming{
		RunResultBase: RunResultBase{
			Input:                  CopyGeneralInput(params.Input),
			NewItems:               nil,
			RawResponses:           nil,
			FinalOutput:            nil,
			InputGuardrailResults:  nil,
			OutputGuardrailResults: nil,
			ContextWrapper:         contextWrapper,
		},
		CurrentAgent:             params.StartingAgent,
		CurrentTurn:              0,
		MaxTurns:                 maxTurns,
		currentAgentOutputSchema: outputSchema,
		IsComplete:               false,
		eventQueue:               asyncqueue.New[StreamEvent](),
		inputGuardrailQueue:      asyncqueue.New[InputGuardrailResult](),
	}

	// Kick off the actual agent loop in the background and return the streamed result object.
	streamedResult.runImplTask = asynctask.CreateTask(ctx, func(ctx context.Context) error {
		return r.runStreamedImpl(
			ctx,
			params.Input,
			streamedResult,
			params.StartingAgent,
			maxTurns,
			hooks,
			contextWrapper,
			params.RunConfig,
			params.PreviousResponseID,
		)
	})

	return streamedResult, nil
}

func (r runner) runInputGuardrailsWithQueue(
	ctx context.Context,
	agent *Agent,
	guardrails []InputGuardrail,
	input Input,
	contextWrapper *runcontext.Wrapper,
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

			result, err := RunImpl().RunSingleInputGuardrail(childCtx, agent, guardrail, input, contextWrapper)
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

	streamedResult.InputGuardrailResults = guardrailResults
	return nil
}

func (r runner) runStreamedImpl(
	ctx context.Context,
	startingInput Input,
	streamedResult *RunResultStreaming,
	startingAgent *Agent,
	maxTurns uint64,
	hooks RunHooks,
	contextWrapper *runcontext.Wrapper,
	runConfig RunConfig,
	previousResponseID string,
) (err error) {
	defer func() {
		if err != nil {
			streamedResult.IsComplete = true
			streamedResult.eventQueue.Put(queueCompleteSentinel{})
		}
	}()

	currentAgent := startingAgent
	currentTurn := uint64(0)
	shouldRunAgentStartHooks := true
	toolUseTracker := NewAgentToolUseTracker()

	streamedResult.eventQueue.Put(AgentUpdatedStreamEvent{
		NewAgent: currentAgent,
		Type:     "agent_updated_stream_event",
	})

	shouldGetAgentTools := true
	var allTools []Tool

	for {
		if streamedResult.IsComplete {
			break
		}

		if shouldGetAgentTools {
			allTools = r.getAllTools(currentAgent)
			shouldGetAgentTools = false
		}

		currentTurn += 1
		streamedResult.CurrentTurn = currentTurn

		if currentTurn > maxTurns {
			streamedResult.eventQueue.Put(queueCompleteSentinel{})
			break
		}

		if currentTurn == 1 {
			// Run the input guardrails in the background and put the results on the queue
			streamedResult.inputGuardrailsTask = asynctask.CreateTask(ctx, func(ctx context.Context) error {
				return r.runInputGuardrailsWithQueue(
					ctx,
					startingAgent,
					slices.Concat(startingAgent.InputGuardrails, runConfig.InputGuardrails),
					InputItems(ItemHelpers().InputToNewInputList(startingInput)),
					contextWrapper,
					streamedResult,
				)
			})
		}

		turnResult, err := r.runSingleTurnStreamed(
			ctx,
			streamedResult,
			currentAgent,
			hooks,
			contextWrapper,
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

		streamedResult.RawResponses = append(streamedResult.RawResponses, turnResult.ModelResponse)
		streamedResult.Input = turnResult.OriginalInput
		streamedResult.NewItems = turnResult.GeneratedItems()

		switch nextStep := turnResult.NextStep.(type) {
		case NextStepFinalOutput:
			streamedResult.outputGuardrailsTask = asynctask.CreateTask(ctx, func(ctx context.Context) outputGuardrailsTaskResult {
				result, err := r.runOutputGuardrails(
					ctx,
					slices.Concat(currentAgent.OutputGuardrails, runConfig.OutputGuardrails),
					currentAgent,
					nextStep.Output,
					contextWrapper,
				)
				return outputGuardrailsTaskResult{Result: result, Err: err}
			})

			taskResult := streamedResult.outputGuardrailsTask.Await()

			var outputGuardrailResults []OutputGuardrailResult
			if taskResult.Canceled {
				return NewCanceledError("output guardrails task has been canceled")
			}
			if taskResult.Result.Err == nil {
				// Errors will be checked in the stream-events loop
				outputGuardrailResults = taskResult.Result.Result
			}

			streamedResult.OutputGuardrailResults = outputGuardrailResults
			streamedResult.FinalOutput = nextStep.Output
			streamedResult.IsComplete = true
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

	streamedResult.IsComplete = true
	return nil
}

func (r runner) runSingleTurnStreamed(
	ctx context.Context,
	streamedResult *RunResultStreaming,
	agent *Agent,
	hooks RunHooks,
	contextWrapper *runcontext.Wrapper,
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
			err := hooks.OnAgentStart(childCtx, contextWrapper, agent)
			if err != nil {
				cancel()
				hooksErrors[0] = fmt.Errorf("RunHooks.OnAgentStart failed: %w", err)
			}
		}()

		if agent.Hooks != nil {
			wg.Add(1)
			go func() {
				defer wg.Done()
				err := agent.Hooks.OnStart(childCtx, contextWrapper, agent)
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

	streamedResult.CurrentAgent = agent
	streamedResult.currentAgentOutputSchema = outputSchema

	systemPrompt, err := agent.GetSystemPrompt(ctx, contextWrapper)
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

	input := ItemHelpers().InputToNewInputList(streamedResult.Input)
	for _, item := range streamedResult.NewItems {
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
			if !reflect.DeepEqual(event.Response.Usage, responses.ResponseUsage{}) {
				u.Requests = 1
				u.InputTokens = uint64(event.Response.Usage.InputTokens)
				u.OutputTokens = uint64(event.Response.Usage.OutputTokens)
				u.TotalTokens = uint64(event.Response.Usage.TotalTokens)
			}
			finalResponse = &ModelResponse{
				Output:     event.Response.Output,
				Usage:      u,
				ResponseID: event.Response.ID,
			}
			contextWrapper.Usage.Add(u)
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
		streamedResult.Input,
		streamedResult.NewItems,
		*finalResponse,
		outputSchema,
		handoffs,
		hooks,
		contextWrapper,
		runConfig,
		toolUseTracker,
	)
	if err != nil {
		return nil, err
	}

	RunImpl().StreamStepResultToQueue(*singleStepResult, streamedResult.eventQueue)
	return singleStepResult, nil
}

func (r runner) runSingleTurn(
	ctx context.Context,
	agent *Agent,
	allTools []Tool,
	originalInput Input,
	generatedItems []RunItem,
	hooks RunHooks,
	contextWrapper *runcontext.Wrapper,
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
			err := hooks.OnAgentStart(childCtx, contextWrapper, agent)
			if err != nil {
				cancel()
				hooksErrors[0] = fmt.Errorf("RunHooks.OnAgentStart failed: %w", err)
			}
		}()

		if agent.Hooks != nil {
			wg.Add(1)
			go func() {
				defer wg.Done()
				err := agent.Hooks.OnStart(childCtx, contextWrapper, agent)
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

	systemPrompt, err := agent.GetSystemPrompt(ctx, contextWrapper)
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
		contextWrapper,
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
		contextWrapper,
		runConfig,
		toolUseTracker,
	)
}

func (runner) getSingleStepResultFromResponse(
	ctx context.Context,
	agent *Agent,
	allTools []Tool,
	originalInput Input,
	preStepItems []RunItem,
	newResponse ModelResponse,
	outputSchema AgentOutputSchemaInterface,
	handoffs []Handoff,
	hooks RunHooks,
	contextWrapper *runcontext.Wrapper,
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
		contextWrapper,
		runConfig,
	)
}

func (runner) runInputGuardrails(
	ctx context.Context,
	agent *Agent,
	guardrails []InputGuardrail,
	input Input,
	contextWrapper *runcontext.Wrapper,
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

			result, err := RunImpl().RunSingleInputGuardrail(childCtx, agent, guardrail, input, contextWrapper)
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

func (runner) runOutputGuardrails(
	ctx context.Context,
	guardrails []OutputGuardrail,
	agent *Agent,
	agentOutput any,
	contextWrapper *runcontext.Wrapper,
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

			result, err := RunImpl().RunSingleOutputGuardrail(childCtx, guardrail, agent, agentOutput, contextWrapper)
			if err != nil {
				cancel()
				guardrailErrors[i] = fmt.Errorf("failed to run output guardrail %s: %w", guardrail.Name, err)
			} else if result.Output.TripwireTriggered {
				cancel() // Cancel all guardrail tasks if a tripwire is triggered.
				err := NewOutputGuardrailTripwireTriggeredError(guardrail.Name, result)
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

func (r runner) getNewResponse(
	ctx context.Context,
	agent *Agent,
	systemPrompt param.Opt[string],
	input []TResponseInputItem,
	outputSchema AgentOutputSchemaInterface,
	allTools []Tool,
	handoffs []Handoff,
	contextWrapper *runcontext.Wrapper,
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

	contextWrapper.Usage.Add(newResponse.Usage)
	return newResponse, err
}

func (runner) getHandoffs(agent *Agent) ([]Handoff, error) {
	handoffs := make([]Handoff, len(agent.Handoffs))
	for i, handoffItem := range agent.Handoffs {
		switch v := handoffItem.(type) {
		case Handoff:
			handoffs[i] = v
		case *Agent:
			h, err := HandoffFromAgent(HandoffFromAgentParams{Agent: v})
			if err != nil {
				return nil, fmt.Errorf("failed to make Handoff from Agent: %w", err)
			}
			handoffs[i] = *h
		default:
			// This would be an unrecoverable implementation bug, so a panic is appropriate.
			panic(fmt.Errorf("unexpected AgentHandoff type %T", v))
		}
	}
	return handoffs, nil
}

func (runner) getAllTools(agent *Agent) []Tool {
	return agent.GetAllTools()
}

func (r runner) getModel(agent *Agent, runConfig RunConfig) (Model, error) {
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
