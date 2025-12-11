# Taler - Real-time Presentation Speech Tutor

Taler is an AI-powered speech coaching application designed to help users improve their public speaking skills. It provides real-time feedback on pacing, filler words, and overall delivery, acting as a personal speech tutor.

## Features

*   **Real-time Speech Analysis**: Analyzes your speech as you talk.
*   **Pace Monitoring**: Calculates Words Per Minute (WPM) to ensure you are speaking at an optimal speed (120-150 WPM).
*   **Filler Word Detection**: Identifies and counts common filler words (e.g., "um", "uh", "like", "you know").
*   **AI Coaching Persona**: Receive feedback from "Gandalf the Grey", offering wisdom and guidance on your speech in a unique and engaging way.
*   **Transcription**: Converts your speech to text using advanced speech-to-text models.
*   **ElevenLabs Integration**: Leverages ElevenLabs for high-quality transcription and agent interactions.
*   **Local Processing**: Utilizes local models (MLX Whisper, Qwen) for privacy and low latency (in the local agent version).

## Project Structure

*   `local_agent/`: Contains the local version of the speech tutor using MLX Whisper and a local LLM.
    *   `local_agent.py`: The main script for the local real-time speech coach.
*   `eleven_test.py`: A script demonstrating integration with the ElevenLabs API for transcription and agent feedback.
*   `voice_samples/`: Directory for storing audio samples for testing.

## Getting Started

### Prerequisites

*   Python 3.12+
*   `uv` (recommended for dependency management) or `pip`
*   ElevenLabs API Key (for `eleven_test.py`)
*   Mac with Apple Silicon (for `local_agent` using MLX)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd taler
    ```

2.  Install dependencies:
    ```bash
    uv sync
    # or
    pip install -r requirements.txt
    ```

### Usage

#### Local Agent (Gandalf)

Run the local real-time coach:

```bash
python local_agent/local_agent.py
```

Speak into your microphone, and the agent will analyze your speech and provide audio feedback.

#### ElevenLabs Integration Test

To test the ElevenLabs integration:

1.  Set your API key:
    ```bash
    export ELEVENLABS_API_KEY="your_api_key_here"
    ```

2.  Run the test script:
    ```bash
    python eleven_test.py
    ```

## Technologies

*   **Python**: Core programming language.
*   **ElevenLabs API**: For cloud-based speech-to-text and conversational agents.
*   **MLX / MLX Whisper**: For efficient local machine learning on Apple Silicon.
*   **SoundDevice**: For audio recording and playback.
*   **Mutagen**: For audio file metadata analysis.

## License

[Add License Information Here]
