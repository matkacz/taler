#!/usr/bin/env python3
"""
Gandalf Speech Coach - ElevenLabs Conversational AI
Full cloud pipeline: STT → LLM → TTS handled by ElevenLabs
"""

import os
import signal
import sys

def main():
    # Import here to get better error messages
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs.conversational_ai.conversation import Conversation
        from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
    except ImportError:
        print("❌ Missing dependencies. Run:")
        print("   uv pip install 'elevenlabs[pyaudio]'")
        sys.exit(1)

    # Config from environment
    AGENT_ID = os.getenv("AGENT_ID")
    API_KEY = os.getenv("ELEVEN_API_KEY")

    if not AGENT_ID:
        print("❌ AGENT_ID not set!")
        print("   1. Create agent at: https://elevenlabs.io/app/conversational-ai")
        print("   2. Copy the Agent ID")
        print("   3. Run: export AGENT_ID='your-agent-id'")
        sys.exit(1)

    print("=" * 55)
    print("🧙 GANDALF SPEECH COACH")
    print("=" * 55)
    print("Powered by ElevenLabs Conversational AI")
    print("Just speak naturally. Press Ctrl+C to exit.")
    print("=" * 55 + "\n")

    # Initialize client
    client = ElevenLabs(api_key=API_KEY) if API_KEY else ElevenLabs()

    # Track conversation for display
    def on_agent_response(response: str):
        print(f"\n🧙 Gandalf: {response}\n")

    def on_user_transcript(transcript: str):
        print(f"📜 You said: {transcript}")

    def on_agent_thinking():
        print("🧙 (pondering...)")

    # Create conversation
    conversation = Conversation(
        client=client,
        agent_id=AGENT_ID,
        requires_auth=bool(API_KEY),
        audio_interface=DefaultAudioInterface(),

        # Callbacks
        callback_agent_response=on_agent_response,
        callback_user_transcript=on_user_transcript,
    )

    # Clean shutdown on Ctrl+C
    def shutdown(sig, frame):
        print("\n\n🧙 \"Even the smallest voice can echo through the ages. Farewell!\"")
        conversation.end_session()

    signal.signal(signal.SIGINT, shutdown)

    # Start the conversation
    print("🎤 Starting session... speak when ready!\n")
    conversation.start_session()

    # Block until conversation ends
    conversation_id = conversation.wait_for_session_end()

    print(f"\n📋 Session ended. Conversation ID: {conversation_id}")
    print("   View transcript at: https://elevenlabs.io/app/conversational-ai")


if __name__ == "__main__":
    main()
