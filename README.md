# cz-transcribe

Local offline Czech speech-to-text transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with optional speaker diarization via [pyannote](https://github.com/pyannote/pyannote-audio).

Everything runs locally on your GPU. No audio data is ever sent anywhere.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

### 1. Install dependencies

```bash
uv sync
```

### 2. HuggingFace token (required for first run)

The whisper model downloads automatically, but the diarization model (`-d` flag) needs a HuggingFace token.

1. Create a free account at https://huggingface.co if you don't have one
2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-community-1
3. Go to https://huggingface.co/settings/tokens and click **Create new token** -- token type **Read** is enough
4. Save the token to `.env` in the project root:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

### 3. First run (downloads models)

```bash
uv run transcribe.py input/your-file.wav
```

On the first run, models are downloaded from HuggingFace (~3 GB for whisper, ~32 MB for diarization). After that, everything is cached locally and the script runs fully offline.

## Usage

Put your audio/video files in `input/`, transcriptions go to `output/`.

```bash
# Plain text transcription
uv run transcribe.py input/interview.wav

# With speaker diarization (Speaker A / Speaker B labels)
uv run transcribe.py input/interview.wav -d

# SRT subtitles
uv run transcribe.py input/interview.wav -f srt

# WebVTT subtitles with speaker labels
uv run transcribe.py input/interview.wav -d -f vtt

# Print to stdout instead of file
uv run transcribe.py input/interview.wav -o -

# Force CPU (slower, no GPU needed)
uv run transcribe.py input/interview.wav --device cpu
```

Accepts any audio/video format (wav, mp3, m4a, mp4, mkv, ...).

## Output formats

- `txt` (default) -- plain text, one segment per line
- `srt` -- SRT subtitles with timestamps
- `vtt` -- WebVTT subtitles with timestamps

When using `-d`, speaker labels (Speaker A, Speaker B, ...) are prepended to each segment. Both a plain and a diarized output file are written.

## Simple version (no diarization, no token)

If you just need plain text transcription without speaker labels, see the [`simple/`](simple/) folder. It's a single self-contained Python file that works on Windows, macOS, and Linux with no HuggingFace token or GPU required.

## Hardware

- NVIDIA GPU with CUDA 12 recommended (runs in float16, ~6 GB VRAM)
- Falls back to CPU with int8 quantization if no GPU is available
- Tested on RTX 5070 Ti: ~20x realtime transcription speed
