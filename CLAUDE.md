# MindFrame — Content Analysis Platform

## What This Project Is

MindFrame is a video analysis platform that extracts transcripts and key visual frames from YouTube/Instagram videos, analyzes content patterns (hooks, scripting style, visual usage) across entire channels, and exposes everything as an MCP tool for AI-powered workflows.

## How We Work (Critical Rules)

1. **No autonomous coding.** Every file, function, and decision must be discussed and approved before writing code.
2. **Explain before building.** Before creating any file or function, explain what it does, why it exists, and how it fits the architecture.
3. **Step-by-step implementation.** One piece at a time. Get approval, build, move to next.
4. **User is always in the loop.** No surprises. Ask first.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  MCP Clients (Claude Code, Cursor, any MCP-compatible)  │
│  User connects with token: mf_abc123xyz                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  MCP Server (Vercel, serverless)                        │
│                                                         │
│  Tools:                                                 │
│    analyze_video(url)       → transcript + key frames   │
│    analyze_channel(url)     → bulk analysis + patterns  │
│    search_hooks(query)      → similar hooks via vectors │
│    compare_channels(a, b)   → style comparison          │
│    my_videos() / my_channels()                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Supabase (free tier)                                   │
│                                                         │
│  PostgreSQL + pgvector:                                 │
│    users     → id, email, token, created_at             │
│    channels  → user_id, name, platform, url             │
│    videos    → user_id, channel_id, title, duration     │
│    segments  → user_id, video_id, text, embedding       │
│    hooks     → user_id, video_id, text, embedding       │
│    frames    → user_id, video_id, ts, embedding         │
│                                                         │
│  Storage: frames/{user_id}/{video_id}/frame_01.jpg      │
│  Auth: token-based, RLS for per-user data isolation     │
└─────────────────────────────────────────────────────────┘
```

## Phase Roadmap

### Phase 1 — Transcript Extraction ✅ COMPLETE

**Goal**: YouTube URL or video file → timestamped transcript.

**Flow**:
```
Input (YouTube URL or video file)
  ├─ YouTube URL → youtube-transcript-api (captions)
  │                  ├─ Captions found → return transcript
  │                  └─ No captions → yt-dlp download → Groq Whisper API
  └─ Video file → ffmpeg extract audio → Groq Whisper API
```

**Tech**: youtube-transcript-api, yt-dlp, ffmpeg, Groq Whisper API

### Phase 2 — Intelligent Frame Extraction ✅ COMPLETE

**Goal**: Extract frames at moments of peak visual content (completed slides, boards, diagrams).

**Algorithm** (SSIM-based, no LLM needed):
```
1. Sample frames at 1fps
2. d(t) = 1 - SSIM(F(t), F(t-1))          — inter-frame difference
3. Transitions where d(t) > μ(d) + k·σ(d)  — adaptive threshold
4. Capture frame at T - δ (before transition = completed state)
5. Filter by edge density I(t) = |Sobel(F(t))| / pixels
6. Gap rule: force capture if no transition for 60s+
```

**Tech**: OpenCV (SSIM, Sobel), scikit-image, ffmpeg

### Phase 3 — Channel Analysis & Data Layer (NEXT)

**Goal**: Analyze entire YouTube/Instagram channels. Store all data per user.

```
Phase 3A: Supabase schema (pgvector, RLS, storage)
Phase 3B: Channel scraper (list all videos from a channel)
Phase 3C: Batch processor (process N videos, store results)
Phase 3D: Analysis queries (hooks, patterns, style comparison)
```

**Tech**: Supabase (Postgres + pgvector + Storage), Jina AI (embeddings, free tier)

### Phase 4 — MCP Server & Auth

**Goal**: Deploy as MCP tool. Users sign up, get token, connect from any MCP client.

```
Phase 4A: MCP server (expose tools via MCP protocol)
Phase 4B: Auth + token generation
Phase 4C: Web dashboard (signup, token gen, usage stats)
```

**Tech**: Vercel (serverless, free tier), MCP protocol

### Phase 5 — LLM-Powered Insights (Future)

**Goal**: Deep content analysis using LLM on collected data.

- Hook pattern analysis across channels
- Script structure detection (intro/body/CTA)
- Visual style profiling
- Content recommendations
- Competitor comparison reports

**Tech**: Groq (Llama 3, free tier), OpenRouter (optional)

## Implementation Progress

### Phase 1 — ✅ COMPLETE
- [x] Project setup (pyproject.toml, requirements.txt, structure)
- [x] YouTube caption fetcher (youtube-transcript-api v1.2.4, instance API)
- [x] Audio downloader (yt-dlp for YouTube, ffmpeg for local files)
- [x] Transcriber (Groq Whisper API — cloud, fast)
- [x] Output formatter (display formatting)
- [x] Streamlit UI (visual frontend)
- [x] Logging (centralized, all modules)

### Phase 2 — ✅ COMPLETE
- [x] Frame analyzer (SSIM, adaptive transitions, edge density, gap rule)
- [x] Frame extractor (ffmpeg at specific timestamps + transcript pairing)
- [x] Streamlit UI — tabs layout, 3-column frame grid, scrollable transcript

### Phase 3 — NOT STARTED
- [ ] Supabase project setup + schema (pgvector, RLS, storage buckets)
- [ ] Channel scraper (YouTube: yt-dlp, Instagram: yt-dlp + cookies)
- [ ] Batch video processor (run Phase 1+2 on N videos, store results)
- [ ] Embedding pipeline (Jina AI for segments/hooks/frames)
- [ ] Vector search queries (semantic search across stored content)

### Phase 4 — NOT STARTED
- [ ] MCP server skeleton (Vercel serverless, MCP protocol)
- [ ] Auth system (token generation `mf_xxx` + validation)
- [ ] Web dashboard (signup, token display, usage stats)

## Known Gaps (as of 2026-04-01)

1. **No tests** — `tests/` has only `__init__.py`, zero test coverage
2. **No database** — all output is local files in `output/`, no persistence layer
3. **Single-video only** — no batch or channel-level processing
4. **No embeddings** — transcript is plain text, no vector search
5. **No auth / multi-user** — no user isolation, no tokens
6. **`.env` only has `GROQ_API_KEY`** — Phase 3 will need Supabase keys + Jina AI key

## File Structure

```
01-mindframe/
├── CLAUDE.md                # This file
├── ideasv1.md               # Original research & ideation
├── pyproject.toml            # Project config
├── requirements.txt          # Python dependencies (ALWAYS keep updated)
├── .env                      # API keys (never commit)
├── .env.example              # Template for required keys
├── src/
│   ├── __init__.py           # Logging setup
│   ├── app.py                # Streamlit UI
│   ├── caption_fetcher.py    # YouTube caption extraction
│   ├── audio_extractor.py    # yt-dlp + ffmpeg audio extraction
│   ├── transcriber.py        # Groq Whisper API transcription
│   ├── formatter.py          # Display formatting utilities
│   ├── frame_analyzer.py     # SSIM-based key frame detection
│   └── frame_extractor.py    # Frame extraction + transcript pairing
├── output/                   # Downloaded audio + extracted frames
└── tests/
    └── ...
```

## Free-Tier Stack

| Component | Service | Free Limit |
|-----------|---------|------------|
| Database + Vectors | Supabase (pgvector) | 500MB |
| Image Storage | Supabase Storage | 1GB |
| Auth | Supabase Auth | 50K users |
| Transcription | Groq Whisper API | Rate limited |
| Embeddings | Jina AI | 1M tokens |
| LLM Analysis | Groq (Llama 3) | Rate limited |
| MCP Hosting | Vercel | Serverless free |

## Conventions

- Language: Python
- Style: Small focused files (<400 lines), immutable data patterns
- Error handling: Explicit, user-friendly messages
- No hardcoded secrets or API keys
- **Dependencies: Any new library MUST be added to `requirements.txt` immediately when introduced**
- Logging: All modules use `logging.getLogger(__name__)`, configured in `src/__init__.py`
