#!/usr/bin/env python3
"""
Fully local pipeline: Russian music video -> Vietnamese subtitles (burned-in).
Uses faster-whisper (large-v3, CUDA), Ollama (qwen3:8b), and FFmpeg.
Cost: $0. Requires: FFmpeg, Ollama with qwen3:8b, NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import requests
from faster_whisper import WhisperModel
import ffmpeg


OLLAMA_URL = "http://localhost:11434"
OLLAMA_CHAT = f"{OLLAMA_URL}/api/chat"
OLLAMA_MODEL = "qwen3:8b"
SYSTEM_PROMPT = (
    "You are a professional song translator. Translate the following Russian lyrics "
    "into Vietnamese. Maintain the rhythm, emotion, and meaning. Do not translate literally. "
    "Output only the Vietnamese translation."
)
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"


def check_ollama():
    """Verify Ollama is running and model is available."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        if not any(OLLAMA_MODEL in n for n in models):
            print(f"Warning: Model '{OLLAMA_MODEL}' not found in Ollama. Available: {models}")
        return True
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama. Is it running? Start with: ollama serve")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Error: Ollama request failed: {e}")
        sys.exit(1)


def translate_segment(russian_text: str) -> str:
    """Send one segment to Ollama and return Vietnamese translation."""
    if not russian_text or not russian_text.strip():
        return ""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": russian_text.strip()},
        ],
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_CHAT, json=payload, timeout=120)
        r.raise_for_status()
        out = r.json()
        content = (out.get("message") or {}).get("content", "").strip()
        return content or russian_text
    except requests.exceptions.RequestException as e:
        print(f"Warning: Translation failed for segment: {e}")
        return russian_text


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[tuple[float, float, str]], path: str) -> None:
    """Write segments (start, end, text) to an SRT file (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(segments, 1):
            text = text.strip().replace("\n", " ")
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n")
            f.write(f"{text}\n\n")


def write_transcription_debug(segments: list[tuple[float, float, str]], path: str) -> None:
    """Write raw transcription (start, end, text) to a text file for debugging (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        for start, end, text in segments:
            f.write(f"[{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}]\n")
            f.write(f"{text.strip()}\n\n")


def extract_audio_ffmpeg(video_path: str, wav_path: str) -> None:
    """Extract mono 16kHz WAV from video for Whisper."""
    out, _ = (
        ffmpeg.input(video_path)
        .output(wav_path, acodec="pcm_s16le", ac=1, ar="16000")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def transcribe(audio_path: str, vad_filter: bool = False) -> list[tuple[float, float, str]]:
    """Transcribe audio with faster-whisper large-v3 on CUDA. Returns (start, end, text).
    VAD is off by default so the whole clip is processed (better for music/singing)."""
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    segments, _ = model.transcribe(
        audio_path,
        language="ru",
        vad_filter=vad_filter,
        no_speech_threshold=0.4,  # more permissive for singing / borderline segments
    )
    result = []
    for s in segments:
        result.append((s.start, s.end, (s.text or "").strip()))
    return result


def burn_subtitles_ffmpeg(video_path: str, srt_path: str, output_path: str, font_name: str = "Arial") -> None:
    """Burn SRT into video with Vietnamese-capable font. Use UTF-8 and FontName."""
    # Escape path for FFmpeg filter: Windows needs / and colon escaped as \:
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    # Force UTF-8 and font for Vietnamese
    vf = f"subtitles='{srt_escaped}':charenc=UTF-8:force_style='FontName={font_name},FontSize=24'"
    (
        ffmpeg.input(video_path)
        .output(output_path, vf=vf)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Vietnamese subtitles for Russian music video (local, $0).")
    parser.add_argument("input", type=str, help="Path to input MP4 file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output MP4 path (default: input_with_subs.mp4)")
    parser.add_argument("--keep-srt", action="store_true", help="Keep temporary SRT file")
    parser.add_argument("--save-transcription", action="store_true", help="Save raw Russian transcription to a text file (for debugging)")
    parser.add_argument("--vad", action="store_true", help="Use VAD to filter silence (can cut off music/singing; off by default)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output).resolve() if args.output else input_path.parent / f"{input_path.stem}_with_subs.mp4"
    output_path = output_path.with_suffix(".mp4")

    check_ollama()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wav_path = tmp / "audio.wav"
        srt_path = tmp / "subs.srt"

        print("Step 1/4: Extracting audio...")
        extract_audio_ffmpeg(str(input_path), str(wav_path))

        print("Step 2/4: Transcribing (faster-whisper large-v3, CUDA)...")
        ru_segments = transcribe(str(wav_path), vad_filter=args.vad)
        if not ru_segments:
            print("Error: No speech segments detected.")
            sys.exit(1)
        print(f"  Found {len(ru_segments)} segments.")
        if args.save_transcription:
            transcription_path = output_path.parent / f"{output_path.stem}_transcription_ru.txt"
            write_transcription_debug(ru_segments, str(transcription_path))
            print(f"  Transcription saved: {transcription_path}")

        print("Step 3/4: Translating to Vietnamese (Ollama qwen3:8b)...")
        vi_segments = []
        for i, (start, end, text) in enumerate(ru_segments):
            if text:
                vi_text = translate_segment(text)
                vi_segments.append((start, end, vi_text))
            else:
                vi_segments.append((start, end, ""))
            print(f"  Segment {i + 1}/{len(ru_segments)}")

        write_srt(vi_segments, str(srt_path))
        if args.keep_srt:
            keep_srt = output_path.parent / f"{output_path.stem}.srt"
            write_srt(vi_segments, str(keep_srt))
            print(f"  SRT saved: {keep_srt}")

        print("Step 4/4: Burning subtitles into video (FFmpeg)...")
        burn_subtitles_ffmpeg(str(input_path), str(srt_path), str(output_path))

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
