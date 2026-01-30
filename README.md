# Russian → Vietnamese subtitles (fully local, $0)

Pipeline for generating Vietnamese subtitles for Russian music videos using local AI: **faster-whisper** (transcription) and **Ollama** (translation). Subtitles are burned into the video with FFmpeg.

## Requirements

- **Python 3.9+**
- **FFmpeg** (in PATH)
- **Ollama** with model `qwen3:8b`
- **NVIDIA GPU** (CUDA) — for faster-whisper

## Installation

1. Clone or copy the project, then install dependencies:

```bash
cd f:\Vit-sub
pip install -r requirements.txt
```

2. Install and run **Ollama**, then pull the model:

```bash
ollama serve
ollama pull qwen3:8b
```

3. Ensure **FFmpeg** is installed and available in PATH.

## Usage

**Basic run** — input MP4, output next to the input as `{name}_with_subs.mp4`:

```bash
python ru_to_vi_subtitles.py "path\to\russian_video.mp4"
```

**Specify output file:**

```bash
python ru_to_vi_subtitles.py "path\to\video.mp4" -o "path\to\output_with_subs.mp4"
```

**Keep the generated SRT file** (next to the output video):

```bash
python ru_to_vi_subtitles.py "path\to\video.mp4" --keep-srt
```

**Save raw Russian transcription** (for debugging; creates `{output_stem}_transcription_ru.txt` with timestamps):

```bash
python ru_to_vi_subtitles.py "path\to\video.mp4" --save-transcription
```

## Pipeline steps

1. **Check Ollama** — exits with an error if Ollama is not running.
2. **Extract audio** — FFmpeg extracts mono 16 kHz WAV from the video.
3. **Transcribe** — faster-whisper `large-v3` on CUDA (Russian, with VAD). With `--save-transcription`, the raw Russian text is written to a `.txt` file next to the output video.
4. **Translate** — each segment is sent to Ollama `qwen3:8b` with a song-translator system prompt (Russian → Vietnamese).
5. **SRT** — UTF-8 subtitle file with Vietnamese text and original timestamps.
6. **Burn** — FFmpeg burns subtitles into the video (Arial, UTF-8).

## Troubleshooting

- **"Cannot connect to Ollama"** — start Ollama: `ollama serve`.
- **"Model qwen3:8b not found"** — run: `ollama pull qwen3:8b`.
- **CUDA / GPU errors** — ensure NVIDIA drivers and CUDA are installed; first run will download the `large-v3` model.
- **FFmpeg not found** — install FFmpeg and add it to system PATH.

## Cost

All processing is local (Whisper + Ollama + FFmpeg). **$0.**
