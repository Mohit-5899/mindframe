# Video-to-Script: Timestamped Transcript + Intelligent Frame Extraction

## Problem

Given a YouTube URL or an uploaded video file:
1. Extract a full script/transcript with timestamps showing exactly what was said and when.
2. Intelligently extract key visual frames (blackboard content, slides, diagrams) guided by the script - not by brute-force frame analysis.

---

## Research Findings

### A. YouTube Caption Extraction (no transcription needed)

YouTube already has captions (auto-generated or manual) on most videos. We can pull these directly.

| Tool | Type | Timestamps | Notes |
|------|------|-----------|-------|
| **youtube-transcript-api** | Python lib, MIT | Segment-level (start + duration) | Lightweight, no API key needed, fetches YT's own captions. Supports SRT/VTT/JSON/CSV output. Latest: v1.2.4. Can be IP-blocked by YouTube on cloud IPs. |
| **yt-dlp** | Python CLI, Unlicense | SRT/VTT/ASS/JSON | Full media downloader. Use `--write-auto-sub --skip-download` for captions only. Heavier but also downloads audio/video when needed. |

**Key insight**: For YouTube videos with existing captions, fetching them is instant, free, and requires zero compute. This should always be the first attempt.

---

### B. Open-Source Speech-to-Text (for uploaded videos or missing captions)

| Tool | Speed | Word-Level Timestamps | Speaker Diarization | Hardware | Best For |
|------|-------|----------------------|---------------------|----------|----------|
| **OpenAI Whisper** | 1x realtime | Yes (`--word_timestamps`) | No | 1-10GB VRAM | Baseline, well-documented |
| **faster-whisper** | 4x faster than Whisper | Yes | No | CPU or CUDA GPU | Best speed/accuracy tradeoff |
| **WhisperX** | 70x realtime (batched) | Yes (precise, via wav2vec2 alignment) | Yes (via pyannote) | <8GB GPU | Best word-level accuracy + speaker labels |
| **whisper-timestamped** | ~1x realtime | Yes + confidence scores | No | Same as Whisper | Word timestamps without extra models |
| **whisper.cpp** | Fast on CPU | Yes | No | Any (C/C++, no Python) | Edge/mobile/embedded, Apple Silicon Metal |
| **insanely-fast-whisper** | 150min audio in ~78s (A100) | Yes | Yes (pyannote) | NVIDIA GPU or Mac MPS | Maximum throughput |

**Model sizes (Whisper family)**:
- tiny (39M, ~1GB VRAM) - fast, lower accuracy
- base (74M) - decent for clear audio
- small (244M) - good balance
- medium (769M) - high accuracy
- large-v3 (1.55B, ~10GB VRAM) - best accuracy
- turbo (809M) - near-large accuracy, much faster

---

### C. Paid / Cloud Transcription APIs

| Service | Price | Word Timestamps | Diarization | Free Tier | Standout Feature |
|---------|-------|----------------|-------------|-----------|-----------------|
| **OpenAI Whisper API** | ~$0.006/min | Yes | No | Pay-as-you-go | Simplest integration, 25MB file limit |
| **Deepgram (Nova-3)** | ~$0.0077/min | Yes | Yes (+$0.002) | $200 credits (no expiry) | Best price/performance, fast |
| **AssemblyAI** | ~$0.0025/min | Yes | Yes (+$0.02/hr) | $50 credits | Rich AI features (sentiment, entities) |
| **Google Cloud STT** | $0.006-0.024/min | Yes | Yes | 60 min/month | GCP-native |
| **Amazon Transcribe** | ~$0.024/min | Yes | Yes | 60 min/month (12mo) | AWS-native, PII redaction |
| **Rev.com** | ~$1.99/min | Yes | N/A | None | 99%+ accuracy (human review) |

---

## Proposed Architecture

### Two-Tier Approach

```
Input (YouTube URL or Video File)
        |
        v
  [1. YouTube URL?] --yes--> [Fetch captions via youtube-transcript-api]
        |                              |
        no                      [Captions exist?]
        |                        /          \
        v                      yes           no
  [2. Extract audio           return          |
     via ffmpeg]             transcript        |
        |                                     |
        v                                     v
  [3. Transcribe via faster-whisper / WhisperX / Cloud API]
        |
        v
  [4. Format output: JSON with timestamps, SRT, plain text]
```

### Why This Design

1. **Caption fetch first** - Free, instant, no compute. YouTube's auto-captions are surprisingly good.
2. **Whisper fallback** - Handles uploaded files and videos without captions.
3. **Cloud API option** - For users without a GPU or wanting managed infrastructure.

---

## Recommended Tech Stack (v1)

| Component | Choice | Why |
|-----------|--------|-----|
| YouTube captions | `youtube-transcript-api` | Lightweight, focused, no API key |
| Audio extraction | `ffmpeg` (via subprocess) | Industry standard, handles all formats |
| Local transcription | `faster-whisper` | 4x faster than Whisper, same accuracy, less VRAM |
| Cloud fallback | Deepgram or AssemblyAI | Generous free tiers, word-level timestamps |
| Output formats | JSON, SRT, plain text | Cover all use cases |
| Language | Python | Best ecosystem for all these tools |

---

## Output Format (Target)

```json
{
  "source": "https://youtube.com/watch?v=...",
  "title": "Video Title",
  "duration": "12:34",
  "method": "youtube_captions",
  "segments": [
    {
      "start": 0.0,
      "end": 3.52,
      "text": "Hey everyone, welcome back to the channel."
    },
    {
      "start": 3.52,
      "end": 7.84,
      "text": "Today we're going to talk about something really interesting."
    }
  ]
}
```

---

## Open Questions

1. **Word-level vs segment-level timestamps?** - Segment-level (2-5 second chunks) is simpler and sufficient for most use cases. Word-level is needed for precise video editing or karaoke-style display.
2. **Speaker diarization needed?** - If videos have multiple speakers, WhisperX or a cloud API with diarization would be valuable.
3. **GPU availability?** - Determines whether we use local Whisper or cloud API as the fallback.
4. **Language support?** - Whisper supports 99+ languages. Do we need multi-language detection?
5. **Rate limiting for YouTube?** - If processing many videos, we may need proxy rotation for youtube-transcript-api.

---

---

## Phase 2: Script-Guided Intelligent Frame Extraction

### The Problem with Naive Frame Extraction

Educational videos are typically 30-60 fps. A 1-hour video at 60 fps = **216,000 frames**. Processing all of them is:
- Computationally expensive
- Wasteful (90%+ of frames are visually redundant in educational content)
- Produces mostly duplicate information

### Why Frame Differencing Alone Doesn't Work

Consider a teacher writing "E = mc²" on a blackboard:

```
Frame 1:  "E"
Frame 2:  "E ="
Frame 3:  "E = m"
Frame 4:  "E = mc"
Frame 5:  "E = mc²"
```

Frame differencing would flag ALL of these as visual changes. But only **Frame 5** (the completed writing) has the full information we need. Every intermediate frame is noise.

In educational videos, the screen changes gradually during writing/drawing, but only the **final state** of that segment matters.

### Script-First Approach (Our Solution)

Use the transcript as the intelligent guide for frame extraction:

1. **Get the timestamped transcript** (Phase 1 output)
2. **LLM analysis of the script** - identify moments where something visual is referenced:
   - Explicit cues: "as you can see", "look at this diagram", "I've written the formula"
   - Topic transitions: when the speaker finishes explaining one concept and moves to the next
   - Pauses or segment boundaries: natural break points where blackboard content is likely complete
3. **Extract ONE frame per marked moment** - at the END of the identified segment (when the visual content is complete, not mid-writing)
4. **Attach frame to transcript segment** - creating a rich script with text + images

### Why Script-First Beats Visual Analysis

| Aspect | Frame Differencing | Script-Guided |
|--------|-------------------|---------------|
| Knows when content is **complete** | No (flags every partial change) | Yes (waits for topic/segment end) |
| Computational cost | High (process all frames) | Low (extract ~10-30 frames per video) |
| Works for blackboard writing | Poorly (every stroke triggers) | Well (grabs final state) |
| Works for still slides | Misses subtle changes | Catches via verbal cues |
| Understands context | No | Yes (knows what was being discussed) |

### Architecture for Phase 2

```
Phase 1 Output (timestamped transcript)
        |
        v
  [1. LLM analyzes script]
        - Mark visual reference points
        - Mark topic transitions
        - Mark segment endings (where visuals are likely complete)
        |
        v
  [2. For each marked timestamp]
        - Extract frame at END of segment (complete visual state)
        - Only 1 frame per marked moment
        |
        v
  [3. Rich script output]
        - Text + timestamps + key frame images
        |
        v
  [4. VLM processing (future)]
        - Feed rich script (text + images) to a VLM
        - Get deeper understanding of both spoken AND visual content
```

### Target Output (Phase 2)

```json
{
  "source": "https://youtube.com/watch?v=...",
  "title": "Calculus Lecture 5 - Integration",
  "segments": [
    {
      "start": 0.0,
      "end": 45.2,
      "text": "Today we'll learn about integration. Let me write the fundamental theorem...",
      "has_visual": true,
      "visual_reason": "Teacher writing fundamental theorem on blackboard",
      "frame_timestamp": 44.8,
      "frame_path": "frames/frame_44.8s.png"
    },
    {
      "start": 45.2,
      "end": 90.0,
      "text": "So as you can see, the integral of f(x) from a to b equals...",
      "has_visual": false
    }
  ]
}
```

### Open Questions (Phase 2)

1. **Which LLM for script analysis?** - Needs to understand educational context and identify visual cues from text alone.
2. **Which VLM for final processing?** - GPT-4o, Claude (vision), Gemini all support image + text. Cost and quality tradeoffs.
3. **Frame extraction precision** - How many seconds before/after the marked timestamp should we look? A small buffer (2-3 seconds before segment end) may help.
4. **Deduplication** - If consecutive segments reference the same visual (e.g., same slide discussed for 5 minutes), we should detect and reuse the same frame.

---

## MVP Scope

### Phase 1 (Core - Build First)
- Accept a YouTube URL or local video file path
- Fetch YouTube captions if available (youtube-transcript-api)
- Fall back to faster-whisper for local transcription (requires audio extraction via yt-dlp/ffmpeg)
- Output as JSON with timestamps + optional SRT export
- CLI interface first, API/UI later

### Phase 2 (Visual Enrichment - Build After Phase 1 Works)
- LLM-based script analysis to mark visual moments
- Intelligent frame extraction at segment boundaries
- Rich output: transcript + key frames
- VLM integration for combined text + visual understanding
