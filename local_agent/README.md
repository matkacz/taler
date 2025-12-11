# Gandalf Speech Coach

A real-time speech coaching tool that analyzes your speaking pace and filler words, then provides feedback as Gandalf the Grey.

## Pipeline

```
Audio File → ElevenLabs STT → Local Analysis → Ollama LLM → ElevenLabs TTS
```

## Prerequisites

- macOS with Apple Silicon (for MLX models)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) running locally
- [ffmpeg](https://ffmpeg.org/) for audio conversion
- ElevenLabs API key

## Installation

```bash
# Clone and enter directory
cd local_agent

# Install dependencies
uv pip install elevenlabs sounddevice numpy scipy

# Install Ollama model
ollama pull qwen3:4b

# Install ffmpeg (if not already installed)
brew install ffmpeg
```

## Configuration

Set your ElevenLabs API key:

```bash
export ELEVEN_API_KEY='your-elevenlabs-api-key'
```

Get your API key from: https://elevenlabs.io/app/settings/api-keys

## Usage

### With static audio file

Place your audio file as `filler_words.m4a` in the same directory, then run:

```bash
uv run python gandalf_elevenlabs.py
```

### Output example

```
=======================================================
🧙 GANDALF SPEECH COACH (ElevenLabs Hybrid)
=======================================================
Using static file: filler_words.m4a
Ideal pace: 90-110 WPM

🎵 Loading audio from: filler_words.m4a
   ✓ Loaded 24.1s of audio
📝 Transcribing with ElevenLabs...

   📜 You said: "Okay, so today we are going to talk about..."

   ╭─────────────────────────────────────╮
   │  🏃 WPM: 122  (Too fast  )      │
   │  📊 Words:  49  Duration: 24.1s   │
   │  🚫 Fillers:  7                      │
   │     um(6), yeah(1)                   │
   ╰─────────────────────────────────────╯
🧙 Gandalf is evaluating...

   🧙 Gandalf: You speak as one chased by wargs! Slow down...
🔊 Gandalf speaks...
```

## Speech Metrics

| WPM | Rating |
|-----|--------|
| < 80 | 🐢 Too slow |
| 90-110 | ✅ Good pace |
| > 115 | 🏃 Too fast |

## Filler Words Detected

`um`, `uh`, `er`, `ah`, `like`, `you know`, `basically`, `actually`, `literally`, `i mean`, `sort of`, `kind of`, `right`, `so yeah`, `yeah`, `okay so`, `well`

## Files

- `gandalf_elevenlabs.py` - Main script (ElevenLabs STT + TTS, Ollama LLM)
- `local_agent.py` - Fully local version (MLX Whisper + Kokoro TTS)
- `filler_words.m4a` - Sample audio file for testing

## Troubleshooting

**"ELEVEN_API_KEY not set"**
```bash
export ELEVEN_API_KEY='your-key'
```

**"Audio file not found"**
Place `filler_words.m4a` in the `local_agent` directory.

**Ollama not responding**
```bash
ollama serve  # Start Ollama server
ollama pull qwen3:4b  # Download model
```
