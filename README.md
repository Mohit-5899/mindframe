# MindFrame

**Video analysis platform that extracts transcripts and key visual frames from YouTube videos.**

Turn any YouTube URL or video file into a timestamped transcript paired with intelligently selected key frames — no manual scrubbing needed.

---

## What It Does

```
YouTube URL or video file
  ├─ Transcript Extraction
  │   ├─ YouTube captions (fast path)
  │   └─ Groq Whisper API (fallback)
  │
  └─ Key Frame Extraction
      ├─ SSIM-based change detection
      ├─ Adaptive transition thresholding
      ├─ Edge density filtering (Sobel)
      └─ Gap rule (capture every 60s if no transition)
```

### Transcript Extraction

- Fetches existing YouTube captions first (manual > auto-generated)
- Falls back to audio download + Groq Whisper API when no captions exist
- Supports uploaded video files (mp4, mkv, webm, mov, avi)
- Output: timestamped segments with start/end times

### Intelligent Frame Extraction

Captures frames at moments of **peak visual content** — completed slides, full diagrams, finished whiteboard drawings — not random intervals.

**Algorithm (pure math, no LLM):**

| Step | What | How |
|------|------|-----|
| 1 | Sample frames | 1 fps from video |
| 2 | Difference signal | `d(t) = 1 - SSIM(frame_t, frame_t-1)` |
| 3 | Detect transitions | `d(t) > mean + 2.5 * std` |
| 4 | Capture | Frame 2s **before** transition (completed state) |
| 5 | Filter | Sobel edge density (skip blank frames) |
| 6 | Gap rule | Force capture every 60s if no transition |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on PATH
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) installed and on PATH
- Groq API key ([free at console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone
git clone https://github.com/Mohit-5899/mindframe.git
cd mindframe

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` — paste a YouTube URL or upload a video file.

---

## Project Structure

```
mindframe/
├── src/
│   ├── __init__.py           # Logging setup
│   ├── app.py                # Streamlit UI
│   ├── caption_fetcher.py    # YouTube caption extraction
│   ├── audio_extractor.py    # yt-dlp + ffmpeg audio extraction
│   ├── transcriber.py        # Groq Whisper API transcription
│   ├── formatter.py          # Display formatting utilities
│   ├── frame_analyzer.py     # SSIM-based key frame detection
│   └── frame_extractor.py    # Frame extraction + transcript pairing
├── tests/                    # Test suite (coming soon)
├── output/                   # Downloaded audio + extracted frames (gitignored)
├── pyproject.toml            # Project config
├── requirements.txt          # Python dependencies
└── .env.example              # API key template
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Captions | youtube-transcript-api |
| Audio Download | yt-dlp |
| Audio Processing | ffmpeg |
| Transcription | Groq Whisper API (whisper-large-v3-turbo) |
| Frame Analysis | OpenCV (SSIM, Sobel), scikit-image |

---

## Roadmap

- [x] **Phase 1** — Transcript extraction (YouTube captions + Whisper fallback)
- [x] **Phase 2** — Intelligent frame extraction (SSIM + edge density)
- [ ] **Phase 3** — Channel analysis + database layer (Supabase, pgvector, batch processing)
- [ ] **Phase 4** — MCP server + auth (expose as AI tool)
- [ ] **Phase 5** — LLM-powered insights (hook patterns, script structure, competitor comparison)

---

## License

MIT
