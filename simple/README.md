# Czech Transcriber

Offline Czech speech-to-text transcription. Runs locally, no account or API key needed.

Works on **Windows**, **macOS**, and **Linux**.

## Setup

Install [uv](https://docs.astral.sh/uv/) (Python package runner):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

That's it. No other setup needed -- `uv` installs everything automatically on first run.

## Usage

```bash
uv run transcribe.py myfile.wav
```

The first run downloads the whisper model (~3 GB) and installs dependencies. After that it works fully offline.

Output is saved to `output/<filename>.txt` next to the script.

### Options

```bash
uv run transcribe.py myfile.mp3 -o result.txt    # custom output path
uv run transcribe.py myfile.mp3 -o -              # print to terminal
uv run transcribe.py myfile.mp3 -m medium         # smaller/faster model (~1.5 GB)
uv run transcribe.py myfile.mp3 -m small           # even smaller (~500 MB)
uv run transcribe.py myfile.mp3 --language en      # English (default: cs = Czech)
```

Accepts any audio/video format: wav, mp3, m4a, mp4, mkv, etc.

## Performance

- **With NVIDIA GPU**: ~20x realtime (a 1-hour file takes ~3 minutes)
- **CPU only** (Mac/Windows/Linux without GPU): roughly realtime (a 1-hour file takes ~1 hour)

For faster CPU transcription, use a smaller model with `-m medium` or `-m small` (lower accuracy).
