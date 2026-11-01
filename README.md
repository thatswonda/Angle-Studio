# Angle Studio

A personal web app for the Video Angle & Auto-Creation Tool PRD (draft v2). You paste a link to a video that already performs, pick one of 8 to 12 original angles, and take it through a story script, visual style, scenes and images, voice, format, animation and compile to a finished MP4.

**Status:** working. Opened from the server it runs in **live mode** with real APIs; the claude.ai preview is a demo with simulated AI.

## Who does what

| Step | Service | Model (default) |
|---|---|---|
| Video stats and comments | YouTube Data API | - |
| Watch the video: transcript, scene-by-scene description, breakdown, gaps | Google Gemini API | Gemini Pro (or Flash) |
| Angle cards, the story script, hooks, scene prompts, title, description, tags | OpenAI ChatGPT API (swappable to Anthropic Claude) | GPT-5.5 (or GPT-5; Claude Opus 5.5 / Sonnet 5.5 later) |
| Voiceover | Google Gemini API | Gemini TTS Flash (ElevenLabs only for cloning your own voice) |
| Scene images | Google Gemini API | Nano Banana 2 (or Imagen 4) |
| Animated clips | Google Gemini API / Higgsfield API | Veo 3.1 Fast, or Seedance 2.0 / Kling 3.0 via Higgsfield |

The writing model never sees the video itself, only the report Gemini writes about it. Videos run up to 10 minutes. Prices in the app are estimates from public write-ups (October 2026) and should be confirmed on each provider's pricing page. The demo simulates every AI call; live mode calls the real APIs below.

## Layout

```
web/index.html            Clickable prototype. Runs on its own, with an in-browser mock backend.
server/app/main.py        FastAPI routes (projects, stages, checkpoints, jobs, idea bank, templates)
server/app/models.py      Project JSON data model (PRD section 6)
server/app/pipeline.py    Stages: ingest, script, audio, scenes, images, animate, compile
server/app/assembly.py    FFmpeg render plan (zoom and pan, crossfades, captions, ducking, loudnorm)
server/app/jobs.py        Background job queue
server/app/storage.py     Project folders on disk + SQLite for ideas, templates, ingest cache
server/app/providers/     One interface per capability (base.py), mock adapters, model catalog and registry
server/tests/             End-to-end test: link to rendered video, cost limit, cache
```

## Run it (live mode)

1. Install Python 3.11+ and FFmpeg (macOS: `brew install ffmpeg`; Windows: `winget install ffmpeg`).
2. Put your keys in `server/.env` (copy `server/.env.example`). Keys stay on the server and never reach the browser.
3. Install and start:

```bash
cd server && pip install -r requirements.txt
```

```bash
cd server && uvicorn app.main:app
```

4. Open http://localhost:8000. The top bar shows **Live**. Rendered videos are saved in `server/data/media/renders`.

Tests (no API calls, no cost):

```bash
cd server && python -m pytest -q tests
```

## Put it online (Render)

Vercel cannot run this app: it needs a server that stays on, FFmpeg, background jobs and a disk. Render runs it from this repo with the included `Dockerfile` and `render.yaml`.

1. Sign in at render.com with GitHub and allow access to this repository.
2. New > Blueprint, pick this repository. Render reads `render.yaml`.
3. Fill in the secrets it asks for: `APP_PASSWORD` (choose one), `GEMINI_API_KEY`, `OPENAI_API_KEY`, `YOUTUBE_API_KEY`.
4. Deploy. The first build takes a few minutes. Open the `onrender.com` link, type any user name and your `APP_PASSWORD`.

The blueprint uses the Starter plan (always on, about $7 a month). On the free plan the app sleeps when idle and loses rendered videos on restart, so download each video when it is done.

## What is live

| Step | Real service | Notes |
|---|---|---|
| Stats and comments | YouTube Data API | Questions and requests are picked out of the top 100 comments |
| Watching the video | Gemini (Flash or Pro) from the YouTube link | Transcript, scene-by-scene description, breakdown, gaps. Nothing is downloaded |
| Angles and story | ChatGPT (GPT-5.5 / GPT-5), or Gemini Flash / Pro | Writer is chosen in the top bar; Claude can be added later |
| Voice | Gemini TTS | Director's-notes prompt keeps the delivery consistent without reading it aloud |
| Images | Nano Banana 2 / Pro / 2 Lite | Needs billing on the Gemini key (the free tier has no image models) |
| Render | FFmpeg on your machine | Zoom and pan per scene, narration, burned-in captions, loudness normalization |
| Animated clips | Not yet | The Gemini key has no Veo models, so scenes use zoom and pan |
| Music | Not yet | Narration only |

**Gemini free tier:** watching the video (Flash), writing (Flash) and voice work, but voice is limited to a handful of requests a day, Pro models are not included, and images are not included. Turn on billing in Google AI Studio for real use.

## What the prototype covers

| PRD area | Where |
|---|---|
| Ingest, cache, manual transcript (ING-1..6) | Paste link step |
| Breakdown, gaps, 10 angle cards, type filter, sort, merge two, comment source, title and tags, idea bank (ANG-1..9) | Pick angle step |
| Target length, 3 hooks, editable narration or dialogue, style guide, originality check (SCR-1..6) | Script step |
| Style templates and saved looks (SCR-4, PRJ-4) | Visual style step, Templates |
| Voice choice and preview, per-line regenerate, clone consent, characters with voices (AUD-1..4, CHR-1..4) | Voice step |
| Aspect, static or animated, 3 caption styles, music and ducking, cost estimate (ASM-2, AUD-5, PRJ-3) | Format step |
| Per-scene image, prompt edit, versions, reorder, merge, split, upload, character refs (VIS-1..6, CHR-5) | Scenes step |
| Per-scene clips, motion prompts, takes, audio-matched length (ANM-1..6) | Animate step (skipped for static videos) |
| Draft preview, final render, polish pass (ASM-1..6) | Compile step |
| Title, description, hashtags, thumbnail ideas, AI disclosure (ASM-5) | Export step |
| Top-bar model selectors, only configured providers, cost and limits (MDL-1..5) | Top bar, Settings |
| Checkpoints, resume after closing, job queue, cost limit (PRJ-1..3) | Everywhere |

## Switching the writer from ChatGPT to Claude

The writer sits behind one interface (`Writer` in `server/app/providers/base.py`), so switching is a setting, not a code change:

1. Add `ANTHROPIC_API_KEY` to `.env`.
2. Set `ANGLE_STUDIO_WRITER=claude-opus-5-5` (or `claude-sonnet-5-5`) for new projects, or pick Claude in the Writing selector in the top bar for one project.

## Still to do

- Animated clips, once the Gemini key has Veo (or via Higgsfield).
- Background music from a royalty-free library.
- A Claude writer adapter (the Writing selector and `ANGLE_STUDIO_WRITER` are ready for it).
- Hosting with a login, to use the app from a phone.
