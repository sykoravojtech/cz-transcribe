#!/usr/bin/env python3
"""Local offline Czech speech-to-text transcription using faster-whisper."""

import argparse
import os
import sys
import time
from pathlib import Path


def _preload_cuda_libs():
    """Preload vendored NVIDIA shared libs so ctranslate2 can find them."""
    try:
        import ctypes
        import nvidia.cublas
        import nvidia.cudnn

        for lib_dir in [
            Path(nvidia.cublas.__path__[0]) / "lib",
            Path(nvidia.cudnn.__path__[0]) / "lib",
        ]:
            for so in sorted(lib_dir.glob("*.so*")):
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    except ImportError:
        pass


_preload_cuda_libs()

import av  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "cs"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_audio_duration(path: str) -> float:
    with av.open(path) as container:
        return float(container.duration) / av.time_base


def detect_device() -> str:
    import ctranslate2
    try:
        ctranslate2.get_supported_compute_types("cuda")
        return "cuda"
    except RuntimeError:
        return "cpu"


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_progress(current: float, total: float, elapsed: float):
    pct = current / total * 100 if total > 0 else 0
    speed = current / elapsed if elapsed > 0 else 0
    remaining = (total - current) / speed if speed > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stderr.write(
        f"\r  {bar} {pct:5.1f}%  "
        f"{format_duration(current)}/{format_duration(total)}  "
        f"ETA {format_duration(remaining)}  "
        f"({speed:.1f}x realtime)  "
    )
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Offline detection
# ---------------------------------------------------------------------------

def _check_models_cached(whisper_model: str, diarize: bool) -> bool:
    """Return True if all needed models are already in the local HF cache."""
    from faster_whisper.utils import _MODELS as _whisper_models
    from huggingface_hub import try_to_load_from_cache

    repo_id = _whisper_models.get(whisper_model, whisper_model)
    if not isinstance(try_to_load_from_cache(repo_id, "model.bin"), str):
        return False

    if diarize:
        seg = try_to_load_from_cache("pyannote/segmentation-3.0", "pytorch_model.bin")
        emb = try_to_load_from_cache("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin")
        if not (isinstance(seg, str) and isinstance(emb, str)):
            return False

    return True


def enable_offline_if_cached(whisper_model: str, diarize: bool):
    if _check_models_cached(whisper_model, diarize):
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("  (offline mode -- all models cached)", file=sys.stderr)
    else:
        print("  (online -- downloading missing models)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(
    input_path: str,
    model_name: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    device: str = "auto",
    beam_size: int = 5,
) -> list[dict]:
    if device == "auto":
        device = detect_device()

    compute_type = "float16" if device == "cuda" else "int8"

    print(f"  Model:    {model_name}", file=sys.stderr)
    print(f"  Device:   {device} ({compute_type})", file=sys.stderr)
    print(f"  Language: {language}", file=sys.stderr)

    print(f"\nLoading whisper model...", file=sys.stderr)
    t0 = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print(f"  Loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    duration = get_audio_duration(input_path)
    print(f"\nTranscribing {format_duration(duration)} of audio...", file=sys.stderr)

    t0 = time.time()
    segments_iter, _info = model.transcribe(
        input_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=True,
        hallucination_silence_threshold=2.0,
        no_speech_threshold=0.6,
        repetition_penalty=1.1,
    )

    segments = []
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        print_progress(seg.end, duration, time.time() - t0)

    elapsed = time.time() - t0
    sys.stderr.write("\n")
    print(f"  Done in {format_duration(elapsed)} ({duration / elapsed:.1f}x realtime)", file=sys.stderr)
    print(f"  {len(segments)} segments\n", file=sys.stderr)

    return segments


# ---------------------------------------------------------------------------
# Speaker diarization
# ---------------------------------------------------------------------------

def diarize(input_path: str, device: str = "auto") -> list[dict]:
    """Run pyannote speaker diarization. Returns list of {start, end, speaker}."""
    import torch
    from pyannote.audio import Pipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.environ.get("HF_TOKEN"):
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                    break

    print("Loading diarization model...", file=sys.stderr)
    t0 = time.time()
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
    pipeline.to(torch.device(device))
    print(f"  Loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    print("Identifying speakers...", file=sys.stderr)
    t0 = time.time()
    result = pipeline(input_path)
    print(f"  Done in {format_duration(time.time() - t0)}\n", file=sys.stderr)

    turns = []
    for turn, _track, speaker in result.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Assign a speaker label to each transcription segment based on overlap."""
    for seg in segments:
        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            overlap_start = max(seg["start"], turn["start"])
            overlap_end = min(seg["end"], turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker or "?"
    return segments


def normalize_speaker_labels(segments: list[dict]) -> list[dict]:
    """Rename SPEAKER_00/SPEAKER_01 to Speaker A/Speaker B etc."""
    mapping: dict[str, str] = {}
    for seg in segments:
        raw = seg.get("speaker", "?")
        if raw not in mapping:
            letter = chr(ord("A") + len(mapping))
            mapping[raw] = f"Speaker {letter}"
        seg["speaker"] = mapping[raw]
    return segments


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _speaker_prefix(seg: dict) -> str:
    sp = seg.get("speaker")
    return f"{sp}: " if sp else ""


def format_txt(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"{_speaker_prefix(seg)}{seg['text'].strip()}")
    return "\n".join(lines) + "\n"


def format_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}")
        lines.append(f"{_speaker_prefix(seg)}{seg['text'].strip()}")
        lines.append("")
    return "\n".join(lines)


def format_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{format_timestamp_vtt(seg['start'])} --> {format_timestamp_vtt(seg['end'])}")
        lines.append(f"{_speaker_prefix(seg)}{seg['text'].strip()}")
        lines.append("")
    return "\n".join(lines)


FORMATTERS = {
    "txt": format_txt,
    "srt": format_srt,
    "vtt": format_vtt,
}


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def resolve_output_path(input_path: str, output: str | None, fmt: str) -> Path | None:
    if output == "-":
        return None
    if output:
        return Path(output)
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / f"{Path(input_path).stem}.{fmt}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline Czech speech-to-text transcription",
    )
    parser.add_argument("input", help="Audio/video file to transcribe")
    parser.add_argument("-o", "--output", help="Output file (default: output/<input>.<format>, '-' for stdout)")
    parser.add_argument("-f", "--format", choices=FORMATTERS, default="txt", help="Output format (default: txt)")
    parser.add_argument("-d", "--diarize", action="store_true", help="Enable speaker diarization (requires HF_TOKEN on first run)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Whisper model (default: {DEFAULT_MODEL})")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"Language code (default: {DEFAULT_LANGUAGE})")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device (default: auto)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"  Czech Transcriber", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"  Input:      {input_path}", file=sys.stderr)
    print(f"  Diarize:    {'yes' if args.diarize else 'no'}", file=sys.stderr)

    enable_offline_if_cached(args.model, args.diarize)

    segments = transcribe(
        str(input_path),
        model_name=args.model,
        language=args.language,
        device=args.device,
        beam_size=args.beam_size,
    )

    if args.diarize:
        turns = diarize(str(input_path), device=args.device)
        segments = assign_speakers(segments, turns)
        segments = normalize_speaker_labels(segments)

    formatter = FORMATTERS[args.format]
    result = formatter(segments)

    out_path = resolve_output_path(args.input, args.output, args.format)
    if out_path is None:
        sys.stdout.write(result)
    else:
        out_path.write_text(result, encoding="utf-8")
        print(f"  Output written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
