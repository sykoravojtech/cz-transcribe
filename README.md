# cz-transcribe

Local offline Czech speech-to-text transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with speaker diarization via [pyannote](https://github.com/pyannote/pyannote-audio).

Everything runs locally on your GPU. No audio data is ever sent anywhere.

## Setup

```bash
uv sync
```

### Speaker diarization (one-time setup)

The diarization model requires accepting the license on HuggingFace:

1. Accept terms at https://huggingface.co/pyannote/speaker-diarization-community-1
2. Create a token at https://huggingface.co/settings/tokens

Add the token to `.env`:

```bash
echo "HF_TOKEN=hf_your_token" > .env
```

Run once to download the models (~32 MB):

```bash
uv run transcribe.py input/file.wav -d
```

After this first run, models are cached locally and the script runs fully offline.

## Usage

Put audio files in `input/`, transcriptions go to `output/`.

```bash
# Plain text transcription
uv run transcribe.py input/interview.wav

# With speaker diarization (Speaker A / Speaker B labels)
uv run transcribe.py input/interview.wav -d

# SRT subtitles with speaker labels
uv run transcribe.py input/interview.wav -d -f srt

# Print to stdout
uv run transcribe.py input/interview.wav -o -

# Force CPU (slower)
uv run transcribe.py input/interview.wav --device cpu
```

## Output formats

- `txt` (default) -- plain text, with speaker labels when `-d` is used
- `srt` -- SRT subtitles
- `vtt` -- WebVTT subtitles

## Privacy

- **Transcription**: runs entirely on your local GPU via CTranslate2. Zero network requests.
- **Diarization**: runs entirely on your local GPU via PyTorch. Zero network requests.
- **Model download**: one-time download from HuggingFace on first run. After caching, the script sets `HF_HUB_OFFLINE=1` to prevent any network access.

## Hardware

- NVIDIA GPU with CUDA 12 (runs in float16, ~6 GB VRAM)
- Falls back to CPU with int8 quantization if no GPU
- Tested on RTX 5070 Ti: ~20x realtime transcription speed
