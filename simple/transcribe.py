#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "faster-whisper",
#     "av",
# ]
# ///
"""Simple offline Czech speech-to-text transcription. No HuggingFace token needed."""

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

        skip = {"libnvblas"}
        for lib_dir in [
            Path(nvidia.cublas.__path__[0]) / "lib",
            Path(nvidia.cudnn.__path__[0]) / "lib",
        ]:
            for so in sorted(lib_dir.glob("*.so*")):
                if any(s in so.name for s in skip):
                    continue
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
OUTPUT_DIR = Path(__file__).parent / "output"


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


def check_model_cached(model_name: str) -> bool:
    from faster_whisper.utils import _MODELS as _whisper_models
    from huggingface_hub import try_to_load_from_cache

    repo_id = _whisper_models.get(model_name, model_name)
    return isinstance(try_to_load_from_cache(repo_id, "model.bin"), str)


def transcribe(
    input_path: str,
    model_name: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    device: str = "auto",
    beam_size: int = 5,
) -> list[str]:
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

    lines = []
    for seg in segments_iter:
        lines.append(seg.text.strip())
        print_progress(seg.end, duration, time.time() - t0)

    elapsed = time.time() - t0
    sys.stderr.write("\n")
    print(f"  Done in {format_duration(elapsed)} ({duration / elapsed:.1f}x realtime)", file=sys.stderr)
    print(f"  {len(lines)} segments\n", file=sys.stderr)

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Simple offline Czech speech-to-text transcription",
    )
    parser.add_argument("input", help="Audio/video file to transcribe")
    parser.add_argument("-o", "--output", help="Output file (default: output/<name>.txt, '-' for stdout)")
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
    print(f"  Input: {input_path}", file=sys.stderr)

    if check_model_cached(args.model):
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("  (offline mode -- model cached)", file=sys.stderr)
    else:
        print("  (online -- downloading model)", file=sys.stderr)

    lines = transcribe(
        str(input_path),
        model_name=args.model,
        language=args.language,
        device=args.device,
        beam_size=args.beam_size,
    )

    text = "\n".join(lines) + "\n"

    if args.output == "-":
        sys.stdout.write(text)
    else:
        if args.output:
            out_path = Path(args.output)
        else:
            OUTPUT_DIR.mkdir(exist_ok=True)
            out_path = OUTPUT_DIR / f"{input_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  Output written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
