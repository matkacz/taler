#!/usr/bin/env python3
"""
Gandalf Speech Coach - ElevenLabs Hybrid
Pipeline: Local file → ElevenLabs STT → Local analysis → Ollama LLM → ElevenLabs TTS
"""

import os
import sys
import re
import tempfile
import subprocess
import numpy as np
import sounddevice as sd
from pathlib import Path

# === CONFIG ===
SAMPLE_RATE = 16000
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam (male)
ELEVEN_MODEL = "eleven_multilingual_v2"
LLM_MODEL = "qwen3:4b"  # Ollama model for coaching

# === SPEECH ANALYSIS CONFIG ===
WPM_TOO_SLOW = 80
WPM_IDEAL_LOW = 90
WPM_IDEAL_HIGH = 110
WPM_TOO_FAST = 115  # 122 WPM will be "too fast"

FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "actually", "literally", "i mean", "sort of", "kind of",
    "right", "so yeah", "yeah", "okay so", "well"
]

# === GANDALF SPEECH COACH PROMPT ===
SYSTEM_PROMPT = """You are Gandalf the Grey, an ancient wizard mentoring mortals in eloquent speech.

## Your Task
Analyze the user's speech metrics and transcript, then provide brief coaching feedback.

## Feedback Rules

### Pace
- Below 80 WPM: Too slow. They sound hesitant.
- 90-110 WPM: Ideal range. Acknowledge if good.
- Above 115 WPM: Too fast. Rushing like a hobbit fleeing Nazgûl.

### Filler Words
- 0-1 fillers: Excellent, praise them
- 2-3 fillers: Note them gently
- 4+ fillers: List the worst offenders

### Response Style
- Be Gandalf: patient but direct, occasional Middle-earth metaphors
- Keep responses under 50 words
- Give ONE specific, actionable tip
- If they did well, praise briefly

Example for 180 WPM with 5 "like"s:
"You speak as one chased by wargs! Slow down, let your words breathe. And 'like' five times? Filler words are stumbling stones. Pause instead of filling silence."

Example for 135 WPM with 0 fillers:
"Well spoken! Your pace was measured, your words precise. Continue practicing."
"""


def check_api_key():
    """Verify ElevenLabs API key is set."""
    if not ELEVEN_API_KEY:
        print("❌ ELEVEN_API_KEY not set!")
        print("   Run: export ELEVEN_API_KEY='your-key-here'")
        sys.exit(1)


def load_audio_file(file_path: str) -> tuple[np.ndarray, float]:
    """Load audio from a file. Returns audio and duration."""
    print(f"\n🎵 Loading audio from: {file_path}")

    # Convert m4a to wav using ffmpeg, resample to 16kHz mono
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_wav = f.name

    subprocess.run([
        "ffmpeg", "-y", "-i", file_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        temp_wav
    ], capture_output=True)

    import scipy.io.wavfile as wav
    sample_rate, audio = wav.read(temp_wav)
    Path(temp_wav).unlink()

    # Convert to float32 normalized
    audio = audio.astype(np.float32) / 32768.0
    duration = len(audio) / sample_rate

    print(f"   ✓ Loaded {duration:.1f}s of audio")
    return audio, duration


def transcribe_elevenlabs(audio: np.ndarray) -> str:
    """Transcribe audio using ElevenLabs Scribe API."""
    print("📝 Transcribing with ElevenLabs...")

    from elevenlabs.client import ElevenLabs
    import scipy.io.wavfile as wav

    client = ElevenLabs(api_key=ELEVEN_API_KEY)

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))

        with open(f.name, "rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                language_code="en"
            )

        Path(f.name).unlink()

    return result.text.strip() if result.text else ""


def analyze_speech(transcript: str, duration_seconds: float) -> dict:
    """Analyze speech for pace and filler words."""
    words = transcript.split()
    word_count = len(words)
    wpm = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0

    transcript_lower = transcript.lower()
    fillers_found = {}
    for filler in FILLER_WORDS:
        if len(filler.split()) == 1:
            count = len(re.findall(rf'\b{filler}\b', transcript_lower))
        else:
            count = transcript_lower.count(filler)
        if count > 0:
            fillers_found[filler] = count

    filler_count = sum(fillers_found.values())

    return {
        "wpm": round(wpm),
        "word_count": word_count,
        "duration": round(duration_seconds, 1),
        "filler_count": filler_count,
        "fillers_found": fillers_found
    }


def print_metrics(metrics: dict):
    """Display speech metrics visually."""
    wpm = metrics['wpm']

    if wpm < WPM_TOO_SLOW:
        pace_icon, pace_label = "🐢", "Too slow"
    elif wpm > WPM_TOO_FAST:
        pace_icon, pace_label = "🏃", "Too fast"
    else:
        pace_icon, pace_label = "✅", "Good pace"

    print(f"\n   ╭─────────────────────────────────────╮")
    print(f"   │  {pace_icon} WPM: {wpm:3d}  ({pace_label:10s})      │")
    print(f"   │  📊 Words: {metrics['word_count']:3d}  Duration: {metrics['duration']:.1f}s   │")
    print(f"   │  🚫 Fillers: {metrics['filler_count']:2d}                      │")
    if metrics['fillers_found']:
        fillers_str = ', '.join([f"{k}({v})" for k, v in list(metrics['fillers_found'].items())[:3]])
        print(f"   │     {fillers_str[:33]:33s} │")
    print(f"   ╰─────────────────────────────────────╯")


def get_coaching(transcript: str, metrics: dict) -> str:
    """Get Gandalf's coaching feedback via Ollama LLM."""
    print("🧙 Gandalf is evaluating...")

    import ollama

    user_message = f"""[SPEECH METRICS]
- Words per minute: {metrics['wpm']}
- Duration: {metrics['duration']}s
- Word count: {metrics['word_count']}
- Filler words: {metrics['filler_count']}
- Fillers found: {metrics['fillers_found'] if metrics['fillers_found'] else 'None'}

[TRANSCRIPT]
"{transcript}"
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    return response["message"]["content"]


def speak_elevenlabs(text: str):
    """Speak with ElevenLabs TTS."""
    print("🔊 Gandalf speaks...")

    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=ELEVEN_API_KEY)

    # Generate audio
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=ELEVEN_VOICE_ID,
        model_id=ELEVEN_MODEL,
        output_format="pcm_24000"
    )

    # Collect all chunks
    audio_chunks = []
    for chunk in audio:
        if isinstance(chunk, bytes):
            audio_chunks.append(chunk)

    # Convert to numpy and play
    audio_data = b''.join(audio_chunks)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio_array, 24000)
    sd.wait()


def main():
    print("=" * 55)
    print("🧙 GANDALF SPEECH COACH (ElevenLabs Hybrid)")
    print("=" * 55)

    check_api_key()

    # Static audio file
    audio_file = Path(__file__).parent / "filler_words.m4a"
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)

    print(f"Using static file: {audio_file.name}")
    print(f"Ideal pace: {WPM_IDEAL_LOW}-{WPM_IDEAL_HIGH} WPM\n")

    # 1. Load audio from static file
    audio, duration = load_audio_file(str(audio_file))

    # 2. Transcribe with ElevenLabs
    transcript = transcribe_elevenlabs(audio)
    if not transcript or len(transcript) < 5:
        print("   (couldn't transcribe audio)")
        return

    print(f"\n   📜 You said: \"{transcript}\"")

    # 3. Analyze speech locally
    metrics = analyze_speech(transcript, duration)
    print_metrics(metrics)

    # 4. Get coaching from Ollama LLM
    feedback = get_coaching(transcript, metrics)
    print(f"\n   🧙 Gandalf: {feedback}")

    # 5. Speak feedback with ElevenLabs TTS
    speak_elevenlabs(feedback)

    print("\n" + "-" * 55)
    print("🧙 \"Even the smallest voice can echo through the ages. Farewell!\"")


if __name__ == "__main__":
    main()
