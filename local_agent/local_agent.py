#!/usr/bin/env python3
"""Real-time local SPEECH COACH - Gandalf edition"""

import sys
import time
import tempfile
import re
import numpy as np
import sounddevice as sd
import mlx_whisper
from pathlib import Path

# === CONFIG ===
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.2  # Slightly longer for speech practice
MAX_RECORD_SECONDS = 60  # Allow longer speeches
WHISPER_MODEL = "mlx-community/whisper-base-mlx"
LLM_MODEL = "qwen3:4b"
VOICE = "am_michael"

# === SPEECH ANALYSIS CONFIG ===
WPM_TOO_SLOW = 110
WPM_IDEAL_LOW = 120
WPM_IDEAL_HIGH = 150
WPM_TOO_FAST = 160

FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "actually", "literally", "i mean", "sort of", "kind of",
    "right", "so yeah", "yeah", "okay so", "well"
]

# === GANDALF SPEECH COACH PROMPT ===
SYSTEM_PROMPT = """You are Gandalf the Grey, an ancient wizard mentoring mortals in eloquent speech.

## Your Task
Analyze the user's speech metrics and transcript, then provide brief coaching feedback.

## Input Format
You receive:
- transcript: What they said
- wpm: Words per minute
- duration: How long they spoke
- filler_count: Number of filler words detected
- fillers_found: Which specific fillers were used

## Feedback Rules

### Pace
- Below 110 WPM: Too slow. They sound hesitant.
- 120-150 WPM: Ideal range. Acknowledge if good.
- Above 160 WPM: Too fast. Rushing like a hobbit fleeing Nazgûl.

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

# Global TTS pipeline (initialized once)
_tts_pipeline = None


def get_tts_pipeline():
    """Lazy-load TTS pipeline."""
    global _tts_pipeline
    if _tts_pipeline is None:
        print("⏳ Loading TTS model...")
        from mlx_audio.tts.models.kokoro import KokoroPipeline
        from mlx_audio.tts.utils import load_model
        model_id = 'prince-canuma/Kokoro-82M'
        model = load_model(model_id)
        _tts_pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=model_id)
    return _tts_pipeline


def analyze_speech(transcript: str, duration_seconds: float) -> dict:
    """Analyze speech for pace and filler words."""
    words = transcript.split()
    word_count = len(words)
    wpm = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0

    # Find filler words
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


def load_audio_file(file_path: str) -> tuple[np.ndarray, float]:
    """Load audio from a file. Returns audio and duration."""
    print(f"\n🎵 Loading audio from: {file_path}")

    import subprocess

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


def record_until_silence() -> tuple[np.ndarray, float]:
    """Record audio until user stops speaking. Returns audio and duration."""
    print("\n🎤 Speak now... (I'm listening)")

    chunks = []
    silent_chunks = 0
    speaking_started = False
    required_silent = int(SILENCE_DURATION * SAMPLE_RATE / 1024)
    start_time = None

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=1024) as stream:
        while True:
            chunk, _ = stream.read(1024)
            chunks.append(chunk)

            volume = np.abs(chunk).mean()

            # Detect when speaking starts
            if not speaking_started and volume > SILENCE_THRESHOLD:
                speaking_started = True
                start_time = time.time()
                print("   📢 Speaking detected...")

            # Check for silence after speaking started
            if speaking_started:
                if volume < SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= required_silent:
                        print("   ✓ End of speech")
                        break
                else:
                    silent_chunks = 0

            # Safety limit
            if len(chunks) * 1024 / SAMPLE_RATE > MAX_RECORD_SECONDS:
                print("   ⚠ Max duration reached")
                break

    duration = time.time() - start_time if start_time else 0
    return np.concatenate(chunks), duration


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio with MLX Whisper."""
    print("📝 Transcribing...")

    import scipy.io.wavfile as wav

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))

        result = mlx_whisper.transcribe(
            f.name,
            path_or_hf_repo=WHISPER_MODEL,
            language="en"
        )
        Path(f.name).unlink()

    return result["text"].strip()


def get_coaching(transcript: str, metrics: dict, history: list) -> str:
    """Get Gandalf's coaching feedback."""
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

    history.append({"role": "user", "content": user_message})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history
    )

    reply = response["message"]["content"]
    history.append({"role": "assistant", "content": reply})

    return reply


def speak(text: str):
    """Speak with Kokoro TTS."""
    print("🔊 Gandalf speaks...")

    pipeline = get_tts_pipeline()

    for _, _, audio in pipeline(text, voice=VOICE, speed=1.0):
        audio_np = np.array(audio[0])
        sd.play(audio_np, 24000)
        sd.wait()


def main():
    print("=" * 50)
    print("🧙 GANDALF SPEECH COACH")
    print("=" * 50)
    print(f"Ideal pace: {WPM_IDEAL_LOW}-{WPM_IDEAL_HIGH} WPM")
    print("Using static audio file for testing.")
    print("Press Ctrl+C to exit\n")

    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "-q"])

    # Static audio file path
    audio_file = Path(__file__).parent / "filler_words.m4a"

    history = []

    # Warm up Whisper
    print("⏳ Warming up Whisper...")
    import scipy.io.wavfile as wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
        wav.write(f.name, SAMPLE_RATE, silence)
        try:
            mlx_whisper.transcribe(f.name, path_or_hf_repo=WHISPER_MODEL)
        except:
            pass
        Path(f.name).unlink()

    # Pre-load TTS
    get_tts_pipeline()

    print("✅ Ready!\n")

    # 1. Load audio from static file
    audio, duration = load_audio_file(str(audio_file))

    # 2. Hardcoded transcript for testing
    transcript = "Okay, so today, today we're going to talk about 11 laps. And 11 laps is the company that is changing the vote as we know. And I think we will see more and more features coming in the next month. Yeah. Thanks. Let's just have a little sneak."

    print(f"\n   📜 You said: \"{transcript}\"")

    # 3. Analyze speech
    metrics = analyze_speech(transcript, duration)
    print_metrics(metrics)

    # 4. Hardcoded feedback for testing TTS
    feedback = "Your pace was good, but I noticed some filler words. Try to pause instead of saying 'okay' and 'so'. Practice makes perfect!"
    print(f"\n   🧙 Gandalf: {feedback}")

    # 5. Speak feedback
    speak(feedback)

    print("\n" + "-" * 50)
    print("\n🧙 \"Even the smallest voice can echo through the ages. Farewell!\"")


if __name__ == "__main__":
    main()