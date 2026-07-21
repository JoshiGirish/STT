"""
Transcribes an audio file using the faster-whisper model.

This script initializes a large Whisper model, transcribes a specified audio file,
and prints the duration and text content of each segment.
"""
from faster_whisper import WhisperModel
import sys
import os

# --- Configuration ---
MODEL_NAME = "large-v3"
AUDIO_FILE_PATH = "audio.wav"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# 1. Initialize the Whisper Model
print(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE} device...")
try:
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


# 2. Check for audio file existence
if not os.path.exists(AUDIO_FILE_PATH):
    print(f"Error: Audio file '{AUDIO_FILE_PATH}' not found. Please ensure the audio file is in the current directory.")
    sys.exit(1)


# 3. Transcribe the audio
print(f"Starting transcription of {AUDIO_FILE_PATH}...")
segments, info = model.transcribe(
    AUDIO_FILE_PATH,
    language="hi",
    beam_size=15,
    patience=2,
    temperature=0,
    condition_on_previous_text=False
)

# 4. Output Results
print("\n" + "="*40)
print(f"Transcription Duration: {info.duration:.2f} seconds")
print("="*40 + "\n")

print("--- Transcribed Segments ---")
for i, segment in enumerate(segments):
    print(f"Segment {i+1}:")
    print(f"  Start: {segment.start:.2f}s | End: {segment.end:.2f}s")
    print(f"  Text: {segment.text}")

print("\nDone")

# Flush stdout and exit successfully
sys.stdout.flush()
sys.exit(0)