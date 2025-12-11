import os
import mutagen
import subprocess
import tempfile
from elevenlabs.client import ElevenLabs
from elevenlabs import ConversationSimulationSpecification, AgentConfig

# Configuration
AUDIO_FILE_PATH = "voice_samples/speech_with_fast_speaking.m4a"
AGENT_ID = "agent_2101kc7b32pze61tsm9gbh818t03"

def analyze_speech(transcript: str, duration_seconds: float) -> dict:
    """Analyze speech for pace and filler words."""
    words = transcript.split()
    word_count = len(words)
    wpm = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0

    return {
        "words_per_minute": round(wpm),
        "word_count": word_count,
        "duration": round(duration_seconds, 1),
    }

def speak(client, text):
    try:
        audio_generator = client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb", # George
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_turbo_v2_5"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in audio_generator:
                f.write(chunk)
            temp_file = f.name
            
        subprocess.run(["afplay", temp_file])
        os.unlink(temp_file)
    except Exception as e:
        print(f"Error generating speech: {e}")

def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Please set the ELEVENLABS_API_KEY environment variable.")
        return

    client = ElevenLabs(api_key=api_key)

    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"File not found: {AUDIO_FILE_PATH}")
        return

    print(f"Uploading {AUDIO_FILE_PATH} for transcription...")
    
    # Get audio duration
    audio = mutagen.File(AUDIO_FILE_PATH)
    duration = audio.info.length

    with open(AUDIO_FILE_PATH, "rb") as f:
        response = client.speech_to_text.convert(
            model_id="scribe_v1",
            file=f
        )

    transcript = response.text
    
    metrics = analyze_speech(transcript, duration)
    
    print("Transcription result:")
    print(transcript)
    print(f"Metrics: {metrics}")

    print(f"\nSending transcript to agent {AGENT_ID} for feedback...")

    # We need a dummy simulated user config because it's required
    # We force the user to say the transcript as the first message
    simulated_user_config = AgentConfig(
        prompt={
            "prompt": f"You are a user. You want to get feedback on your speech. You say exactly this: 'Here is a transcript of my speech. Please give me feedback on it: {transcript}. My speech metrics are: {metrics}'"
        },
        first_message=f"Here is a transcript of my speech. Please give me feedback on it: {transcript}. My speech metrics are: {metrics}"
    )

    try:
        simulation_response = client.conversational_ai.agents.simulate_conversation(
            agent_id=AGENT_ID,
            simulation_specification=ConversationSimulationSpecification(
                simulated_user_config=simulated_user_config,
            ),
            new_turns_limit=1
        )
        
        print("\nAgent Feedback:")
        first_agent_message = True
        for turn in simulation_response.simulated_conversation:
            print(f"{turn.role}: {turn.message}")
            if turn.role == "agent":
                if first_agent_message:
                    first_agent_message = False
                    continue
                speak(client, turn.message)

            # Speak the agent's feedback
            if turn.role == "assistant":
                speak(client, turn.message)

    except Exception as e:
        print(f"Error simulating conversation: {e}")

if __name__ == "__main__":
    main()
