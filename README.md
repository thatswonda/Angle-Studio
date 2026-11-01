# Angle Studio

A personal web app for the Video Angle & Auto-Creation Tool PRD (draft v2). You paste a link to a video that already performs, pick one of 8 to 12 original angles, and take it through a story script, visual style, scenes and images, voice, format, animation and compile to a finished MP4.

**Status:** UI and backend skeleton are done.

## Who does what

| Step | Service | Model (default) |
|---|---|---|
| Video stats and comments | YouTube Data API | - |
| Watch the video: transcript, scene-by-scene description, breakdown, gaps | Google Gemini API | Gemini Pro (or Flash) |
| Angle cards, the story script, hooks, scene prompts, title, description, tags | OpenAI ChatGPT API (swappable to Anthropic Claude) | GPT-5.5 (or GPT-5; Claude Opus 5.5 / Sonnet 5.5 later) |
| Voiceover | Google Gemini API | Gemini TTS Flash (ElevenLabs only for cloning your own voice) |
| Scene images | Google Gemini API | Nano Banana 2 (or Imagen 4) |
| Animated clips | Google Gemini API / Higgsfield API | Veo 3.1 Fast, or Seedance 2.0 / Kling 3.0 via Higgsfield |

The writing model never sees the video itself, only the report Gemini writes about it. Videos run up to 10 minutes. Prices in the app are estimates from public write-ups (October 2026) and should be confirmed on each provider's pricing page. Every AI provider is a mock adapter, so the whole flow runs with no API keys or cost. Connecting real APIs comes last.

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

## Run it

```bash
cd server && pip install -r requirements.txt
```

```bash
cd server && uvicorn app.main:app --reload
```

Open http://localhost:8000 for the app or http://localhost:8000/docs for the API.

```bash
cd server && python -m pytest -q tests
```

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

## Connecting real APIs (next)

1. Write one adapter per provider in `server/app/providers/` that implements its interface from `base.py`: `VideoSource` (YouTube Data API), `VideoAnalyst` (Gemini, takes the YouTube URL), `Writer` (ChatGPT now, Claude later), `TextToSpeech`, `ImageGenerator`, `ImageToVideo`.
2. Register it in `registry.REAL`, put the keys in `.env` (see `.env.example`) and set `ANGLE_STUDIO_MOCK=0`.
3. Point the `api` object in `web/index.html` at the server routes with `fetch()`. Its methods already match the route names.
4. Finish the real FFmpeg render in `assembly.render` (the plan is already built).

Open questions from PRD section 13 still apply: which providers you already have accounts with, which languages v1 needs, and whether it runs locally or hosted (hosted needs a login).
