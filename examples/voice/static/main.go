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

package main

import "time"

/*
This is a simple example that uses a recorded audio buffer.

1. You can record an audio clip in the terminal.
2. The pipeline automatically transcribes the audio.
3. The agent workflow is a simple one that starts at the Assistant agent.
4. The output of the agent is streamed to the audio player.

Try examples like:
- Tell me a joke (will respond with a joke)
- What's the weather in Tokyo? (will call the `get_weather` tool and then speak)
- Hola, como estas? (will handoff to the spanish agent)
*/

func main() {
	err := usingPortaudio(func() error {
		buffer, err := recordAudio()
		if err != nil {
			return err
		}

		//    pipeline = VoicePipeline(
		//        workflow=SingleAgentVoiceWorkflow(agent, callbacks=WorkflowCallbacks())
		//    )
		//
		//    audio_input = AudioInput(buffer=record_audio())
		//
		//    result = await pipeline.run(audio_input)
		//
		//    with AudioPlayer() as player:
		//        async for event in result.stream():
		//            if event.type == "voice_stream_event_audio":
		//                player.add_audio(event.data)
		//                print("Received audio")
		//            elif event.type == "voice_stream_event_lifecycle":
		//                print(f"Received lifecycle event: {event.event}")
		//
		//        # Add 1 second of silence to the end of the stream to avoid cutting off the last audio.
		//        player.add_audio(np.zeros(24000 * 1, dtype=np.int16))

		err = usingAudioPlayer(func(player *AudioPlayer) error {
			return player.AddAudio(buffer)
		})
		if err != nil {
			return err
		}

		time.Sleep(2 * time.Second)

		return nil
	})
	if err != nil {
		panic(err)
	}
}

//@function_tool
//def get_weather(city: str) -> str:
//    """Get the weather for a given city."""
//    print(f"[debug] get_weather called with city: {city}")
//    choices = ["sunny", "cloudy", "rainy", "snowy"]
//    return f"The weather in {city} is {random.choice(choices)}."
//
//
//spanish_agent = Agent(
//    name="Spanish",
//    handoff_description="A spanish speaking agent.",
//    instructions=prompt_with_handoff_instructions(
//        "You're speaking to a human, so be polite and concise. Speak in Spanish.",
//    ),
//    model="gpt-4o-mini",
//)
//
//agent = Agent(
//    name="Assistant",
//    instructions=prompt_with_handoff_instructions(
//        "You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
//    ),
//    model="gpt-4o-mini",
//    handoffs=[spanish_agent],
//    tools=[get_weather],
//)
//
//
//class WorkflowCallbacks(SingleAgentWorkflowCallbacks):
//    def on_run(self, workflow: SingleAgentVoiceWorkflow, transcription: str) -> None:
//        print(f"[debug] on_run called with transcription: {transcription}")
//
//
