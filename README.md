# Angle Studio

A personal web app for the Video Angle & Auto-Creation Tool PRD (draft v2). You start in one of three ways: paste a link to a video that went viral and pick one of 4 angles that recreate what made it work (fill a gap, same story in a new look, same idea in a new niche, a similar storyline; never a part 2), type your own idea and let the writer draft the story, or paste a story you wrote. Then you take it through a story script, visual style, scenes and images, voice, format, animation and compile to a finished MP4.

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
3. Fill in the secrets it asks for: `APP_PASSWORD` (choose one), `GEMINI_API_KEY`, `OPENAI_API_KEY`, `YOUTUBE_API_KEY`, and `FAL_KEY` for the FLUX image models.
4. Deploy. The first build takes a few minutes. Open the `onrender.com` link, type any user name and your `APP_PASSWORD`.

The blueprint uses the Starter plan (always on, about $7 a month) with a 5 GB persistent disk (about $1.25 a month) mounted at `/app/server/data`. Voice clips, images, videos and the analysis cache live there; without the disk every redeploy deletes them. If files are lost anyway, the Compile step lists them and offers to make only those again. Projects (story, scenes, settings) are saved in the browser and backed up to the server's disk, so a cleared browser or another device gets them back. Renders are kept under 300 MB of memory and run FFmpeg at low priority, so they fit Render's Starter plan (half a CPU) without starving its health checks. Each scene is encoded once, with its captions; finished scenes are kept on disk, so an interrupted render continues where it stopped (the app retries on its own) and unchanged scenes are reused next time. Final renders come in 720p or 1080p, or you can continue with the 480p draft.

## What is live

| Step | Real service | Notes |
|---|---|---|
| Start without a video | The writer | **From my idea**: the writer drafts the story from your idea at your chosen length (one text call). **Paste my story**: used word for word; an optional cheap text call writes an image idea per sentence. The angle step is skipped |
| Stats and comments | YouTube Data API | Questions and requests are picked out of the top 100 comments |
| Watching the video | Gemini (Flash or Pro) from the YouTube link | Transcript, scene-by-scene description, breakdown, gaps. Nothing is downloaded |
| Angles and story | ChatGPT (GPT-5.5 / GPT-5), or Gemini Flash / Pro | Writer is chosen in the top bar; Claude can be added later |
| Voice | Gemini TTS | 30 voices (filter higher or lower, search). Tune tone, pace, energy, accent and notes (director's-notes prompt, never read aloud), plus exact speed and pitch applied with FFmpeg. Edit any sentence's words or how it is said and regenerate just that one; sentences made with older settings are flagged |
| Images | Nano Banana 2 / Pro / 2 Lite, or FLUX schnell / dev on fal.ai (`FAL_KEY`) | 20 visual templates (recipe + avoid list + example images in `web/templates`), custom templates from your own images, and Lock look: scene 1 guides every other scene (Nano Banana only; FLUX uses the template recipe alone, and thumbnails and banners stay on Nano Banana because they print words). Needs Gemini billing or fal credit |
| Your idea on a template | - | Add your own idea to any visual template (added to every image prompt) for one video, or save it as your own template |
| Story edits | The writer | "Add an idea" rewrites the story with your idea and keeps the length; Undo restores the previous version |
| Publish kit | The writer + Nano Banana | After the video: 4 titles, 3 post captions with hashtags, 3 video descriptions and 3 thumbnails with bold text, to choose from, matched to the analyzed video's keywords, title style and thumbnail look so the two get recommended together |
| Channel ideas | The writer + Nano Banana | Optional, on Export: grows from the analyzed video, follows the kind of channel you type, and suggests 3 channels (name, handle, bio, about) to copy; profile photo (1:1) and banner (16:9) are drawn on request in the video's style |
| Render | FFmpeg on the server | Each image stays on screen exactly as long as its own sentences are spoken (measured from the voice clips). Captions are optional (On/Off on the Format and Compile steps; after a render, edits show at once in the preview and "Update the video" re-renders for free), come in 9 styles (Bold pop, One word, Karaoke, Highlight box, Comic, Neon glow, Typewriter, Clean bar, Minimal) and are editable per scene; a .srt captions file is saved with every render |
| Animated clips | Not yet | The Gemini key has no Veo models, so scenes use zoom and pan |
| Music | Not yet | Narration only |

**Gemini free tier:** watching the video (Flash), writing (Flash) and voice work, but voice is limited to a handful of requests a day, Pro models are not included, and images are not included. Turn on billing in Google AI Studio for real use.

## What the prototype covers

| PRD area | Where |
|---|---|
| Ingest, cache, manual transcript (ING-1..6) | Paste link step |
| Breakdown, gaps, the 4 angles with the highest viral potential (each shows what it keeps and what it changes), type filter, sort, merge two, comment source, title and tags, idea bank (ANG-1..9) | Pick angle step |
| Target length, 3 hooks, editable narration or dialogue, style guide, originality check (SCR-1..6) | Script step |
| Style templates and saved looks (SCR-4, PRJ-4) | Visual style step, Templates |
| Voice choice and preview, per-line regenerate, clone consent, characters with voices (AUD-1..4, CHR-1..4) | Voice step |
| Aspect, static or animated, 3 caption styles, music and ducking, cost estimate (ASM-2, AUD-5, PRJ-3) | Format step |
| Per-scene image, prompt edit, versions, reorder, merge, split, upload, character refs (VIS-1..6, CHR-5) | Scenes step |
| Per-scene clips, motion prompts, takes, audio-matched length (ANM-1..6) | Animate step (skipped for static videos) |
| Draft preview, final render, polish pass (ASM-1..6) | Compile step |
| Title, description, hashtags, thumbnail ideas, AI disclosure (ASM-5) | Export step |
| Top-bar model selectors (video analysis, writing, voice, image, video), only configured providers, cost and limits (MDL-1..5) | Top bar, Settings |
| Every rendered video kept with its project, with play and download | Projects page |
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

## Keeping costs down

Scene images are about 90% of the cost (about $0.07 each with Nano Banana 2). The app keeps that number small:

- **Re-planning keeps images:** changing pacing or the story keeps the image of every scene whose sentences did not change.
- **Image reuse:** one image covers two scenes, the second with a different zoom and framing. Half the images.
- **Image quality switch** on the Scenes page: Lite (about $0.03), Standard (about $0.07) or Pro (about $0.14).
- **Thumbnails on request:** ideas are written first; each thumbnail is drawn only when you ask.
- **Analysis cache:** analyzing the same link with the same models again costs nothing, and broken JSON is repaired with a small text-only call instead of re-watching the video.

## Template example images

The 80 images in `web/templates` were made once with Nano Banana 2. To remake missing ones after editing `web/templates/templates.json`:

```bash
cd server && python scripts/make_template_examples.py
```
