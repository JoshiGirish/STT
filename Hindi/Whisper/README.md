# Audio Transcription Tool

A Python script that transcribes audio files using the [faster-whisper](https://github.com/guillaumekln/faster-whisper) library with the Whisper large-v3 model.

## Overview

This tool transcribes audio files to text using OpenAI's Whisper model. It supports:
- Transcription with the large-v3 model
- Segmented output with timestamps
- Hindi language detection and transcription
- GPU acceleration with CUDA support

## Prerequisites

### Python
- Python 3.8 or higher
- pip package manager

### Hardware
- **GPU (Recommended)**: NVIDIA GPU with CUDA support for faster processing
- **CPU**: Works on CPU but significantly slower
- **RAM**: Minimum 8GB recommended (model is ~3.5GB)

## Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

Place your audio file in the same directory as `transcribe.py` and run:

```bash
python transcribe.py
```

### Expected Output

```
Loading Whisper model 'large-v3' on cuda device...
Starting transcription of audio.wav...

========================================
Transcription Duration: 18.20 seconds
========================================

--- Transcribed Segments ---
Segment 1:
  Start: 0.00s | End: 17.34s
  Text:  कभी कभी मेरे दिल में खयाल आता है ये जैसे तुझको बनाया गया है मेरे लिए

Done
```

## References

- [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [Whisper Model](https://github.com/openai/whisper)