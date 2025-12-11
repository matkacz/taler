import sys
import time
import tempfile
import re
import numpy as np
import sounddevice as sd
# ...existing code...
# import mlx_whisper # Remove this line
import whisper # Add this line
from pathlib import Path

# === CONFIG ===
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.2
MAX_RECORD_SECONDS = 60
# WHISPER_MODEL = "mlx-community/whisper-base-mlx" # This will be changed
WHISPER_MODEL = "base" # Using a simpler model for OpenAI Whisper
LLM_MODEL = "qwen3:4b"
# VOICE = "am_michael" # This will be changed
TTS_LANG = "en" # Language for gTTS

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

# === TRUMP SPEECH COACH PROMPT ===
SYSTEM_PROMPT = """You are Donald Trump, a legendary speaker and dealmaker. You're coaching someone on their speech delivery.

## Your Task
Analyze the user's speech metrics and transcript, then provide coaching feedback in Trump's distinctive speaking style.

## Input Format
You receive:
- transcript: What they said
- wpm: Words per minute
- duration: How long they spoke
- filler_count: Number of filler words detected
- fillers_found: Which specific fillers were used

## Feedback Rules

### Pace
- Below 110 WPM: Too slow. Weak delivery, no energy.
- 120-150 WPM: Good pace. Strong and powerful.
- Above 160 WPM: Too fast. Tone it down, be strategic.

### Filler Words
- 0-1 fillers: Excellent! Very clean, very professional.
- 2-3 fillers: Some ums and ahs? We can do better, believe me.
- 4+ fillers: Too many fillers! Weak. Use power words instead.

### Response Style
- Use Trump's characteristic phrases: "believe me", "very", "tremendous", "fantastic", "the best", "the worst"
- Be direct and confident
- Keep responses under 50 words
- Give ONE actionable tip
- Reference deals, winners, success
- Slight hyperbole for emphasis

Example for 180 WPM with 5 "like"s:
"You're speaking too fast - too much energy, not enough power. And all those 'likes'? Weak! Winners don't say 'like'. Pause. Be silent. It's stronger. Believe me!"

Example for 135 WPM with 0 fillers:
"Now THAT'S a strong delivery! No fillers, perfect pace. Very professional. This is how winners speak. Tremendous job!"
"""

# Global TTS pipeline (initialized once)
_tts_pipeline = None


def get_tts_pipeline():
    """Lazy-load TTS pipeline."""
    global _tts_pipeline
    if _tts_pipeline is None:
        print("⏳ Loading TTS model...")
        # from mlx_audio.tts.models.kokoro import KokoroPipeline # Remove this
        # from mlx_audio.tts.utils import load_model # Remove this
        # model_id = 'prince-canuma/Kokoro-82M' # Remove this
        # model = load_model(model_id) # Remove this
        # _tts_pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=model_id) # Remove this
        from gtts import gTTS # Add this
        _tts_pipeline = gTTS # gTTS is the constructor itself
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
    """Transcribe audio with OpenAI Whisper."""
    print("📝 Transcribing...")

    import scipy.io.wavfile as wav

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # OpenAI Whisper expects 16-bit PCM WAV
        wav.write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        
        # result = mlx_whisper.transcribe( # Remove this
        #     f.name, # Remove this
        #     path_or_hf_repo=WHISPER_MODEL, # Remove this
        #     language="en" # Remove this
        # ) # Remove this
        model = whisper.load_model(WHISPER_MODEL) # Add this
        result = model.transcribe(f.name, language="en") # Add this
        Path(f.name).unlink()

    return result["text"].strip()


def get_coaching(transcript: str, metrics: dict, history: list) -> str:
    """Get Trump's coaching feedback."""
    print("🏆 Trump is evaluating...")

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
    """Speak with gTTS."""
    print("🔊 Trump speaks...")

    tts_engine = get_tts_pipeline()
    
    # from mlx_audio.tts.models.kokoro import KokoroPipeline # Remove this (if not already removed)
    # for _, _, audio in pipeline(text, voice=VOICE, speed=1.0): # Remove this
    #     audio_np = np.array(audio[0]) # Remove this
    #     sd.play(audio_np, 24000) # Remove this
    #     sd.wait() # Remove this

    tts = tts_engine(text=text, lang=TTS_LANG, slow=False) # Add this
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp: # Add this
        tts.save(fp.name) # Add this
        # Play the audio file # Add this
        import subprocess # Add this
        subprocess.run(["ffplay", "-nodisp", "-autoexit", fp.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Add this
        Path(fp.name).unlink() # Add this



def main():
    print("=" * 50)
    print("🏆 TRUMP SPEECH COACH")
    print("=" * 50)
    print(f"Ideal pace: {WPM_IDEAL_LOW}-{WPM_IDEAL_HIGH} WPM")
    print("Make your speeches great again!")
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

    # 4. Get Trump's feedback
    feedback = "Your pace is strong, but too many 'okay's and 'and's. Cut the weak words! Be decisive. These fillers make you sound unsure - winners don't do that, believe me!"
    print(f"\n   🏆 Trump: {feedback}")

    # 5. Speak feedback
    speak(feedback)

    print("\n" + "-" * 50)
    print("\n🏆 \"That was fantastic! You're going to be a tremendous speaker, believe me!\"")


if __name__ == "__main__":
    main()