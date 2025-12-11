import os
from elevenlabs.client import ElevenLabs
from elevenlabs import ConversationSimulationSpecification, AgentConfig

def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Please set the ELEVENLABS_API_KEY environment variable.")
        return

    client = ElevenLabs(api_key=api_key)

    file_path = "voice_samples/speech_with_fast_speaking.m4a"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Uploading {file_path} for transcription...")
    
    with open(file_path, "rb") as f:
        response = client.speech_to_text.convert(
            model_id="scribe_v1",
            file=f
        )

    transcript = response.text
    print("Transcription result:")
    print(transcript)

    agent_id = "agent_2101kc7b32pze61tsm9gbh818t03"
    print(f"\nSending transcript to agent {agent_id} for feedback...")

    conversation_history = [
        {
            "role": "user",
            "message": f"Here is a transcript of my speech. Please give me feedback on it: {transcript}",
            # "time_in_call_secs": 0 # This might not be needed or might be different
        }
    ]

    # We need a dummy simulated user config because it's required
    # We can try to force the user to say the transcript as the first message
    simulated_user_config = AgentConfig(
        prompt={
            "prompt": f"You are a user. You want to get feedback on your speech. You say exactly this: 'Here is a transcript of my speech. Please give me feedback on it: {transcript}'"
        },
        first_message=f"Here is a transcript of my speech. Please give me feedback on it: {transcript}"
    )

    try:
        # Note: partial_conversation_history structure needs to be correct.
        # Let's try to see if we can just start the conversation with the agent.
        # But simulate_conversation is the only method exposed on 'agents' that looks like it runs a chat.
        
        simulation_response = client.conversational_ai.agents.simulate_conversation(
            agent_id=agent_id,
            simulation_specification=ConversationSimulationSpecification(
                simulated_user_config=simulated_user_config,
                # partial_conversation_history=conversation_history 
            ),
            new_turns_limit=3 # Increase turns to let the conversation flow
        )
        
        print("\nAgent Feedback:")
        for turn in simulation_response.simulated_conversation:
            print(f"{turn.role}: {turn.message}")

    except Exception as e:
        print(f"Error simulating conversation: {e}")

if __name__ == "__main__":
    main()
