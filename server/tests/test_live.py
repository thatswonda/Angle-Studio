"""Live-mode pieces that run without API calls: link parsing, captions and a real FFmpeg render."""
import base64
import math
import shutil
import struct
import wave

import pytest

from app import render
from app.providers import youtube
from app.providers.google import ProviderError


def test_video_links():
    for url in ("https://www.youtube.com/watch?v=V-7zASRCXaQ", "https://youtu.be/V-7zASRCXaQ?t=3",
                "https://www.youtube.com/shorts/V-7zASRCXaQ"):
        assert youtube.video_id(url) == "V-7zASRCXaQ"
    with pytest.raises(ProviderError):
        youtube.video_id("https://example.com/not-a-video")


def test_captions_follow_each_sentence():
    scenes = [{"lines": [0, 1], "caption": "one two three four five six seven", "caption_auto": True},
              {"lines": [2], "caption": "My own edited caption", "caption_auto": False}]
    texts = ["one two three", "four five six seven", "eight nine"]
    times = [(0, 1.0), (1.0, 3.0), (3.0, 4.0)]
    cues = render.caption_cues(scenes, texts, times, [0, 3.0], [3.0, 1.0], per=3)
    # sentence 2 starts exactly when its audio does; the edited caption replaces the story words
    assert cues[0] == (0, 1.0, ["one", "two", "three"]) and cues[1][0] == 1.0
    assert [w for c in cues[-2:] for w in c[2]] == ["My", "own", "edited", "caption"]
    assert "00:00:01,000 --> " in render.captions_srt(cues)


def test_scene_timing_is_exact():
    secs = [1.25, 2.5, 3.0]
    starts, durs = render.scene_timing([{"lines": [0, 1]}, {"lines": [2], "split": {"index": 0, "of": 2}},
                                        {"lines": [2], "split": {"index": 1, "of": 2}}], secs)
    assert durs == [3.75, 1.5, 1.5] and starts == [0, 3.75, 5.25]
    assert render.scene_timing([{"lines": [1]}, {"lines": [0]}], [1, 1]) is None  # out of story order


def _tone(path, seconds):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(n / 20))) for n in range(int(24000 * seconds))))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_render_matches_audio(tmp_path):
    import subprocess
    media = tmp_path / "media"
    (media / "audio").mkdir(parents=True)
    _tone(media / "audio" / "a0.wav", 1.2)
    _tone(media / "audio" / "a1.wav", 0.9)
    png = base64.b64encode(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c4944415408d763f8cfc0000003010100c9fe92ef0000000049454e44ae426082")).decode()
    img = "data:image/png;base64," + png
    for style in ("clean", "none", "typewriter", "word", "neon", "boxed", "comic", "minimal", "karaoke"):
        out = render.render({"final": False, "aspect": "1:1", "caption_style": style,
                             "scenes": [{"image": img, "lines": [0], "caption": "hello there"},
                                        {"image": img, "lines": [1], "caption": "and goodbye"}],
                             "lines": [{"audio": "/media/audio/a0.wav", "text": "hello there"},
                                       {"audio": "/media/audio/a1.wav", "text": "and goodbye"}]}, media, lambda _: None)
        video = media / out["url"].removeprefix("/media/")
        assert video.exists() and out["srt"] and (media / out["srt"].removeprefix("/media/")).exists()
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                                   capture_output=True, text=True).stdout)
        assert abs(dur - 2.1) < 0.1  # video length equals the narration
    with pytest.raises(ProviderError):  # a missing voice clip is reported, not crashed on
        render.render({"scenes": [{"image": img, "lines": [0], "caption": "x"}], "lines": [{"audio": ""}]}, media, lambda _: None)


def test_password_protects_everything(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    assert c.get("/").status_code == 401 and "WWW-Authenticate" in c.get("/").headers
    assert c.get("/api/live/status").status_code == 401
    assert c.get("/healthz").status_code == 200
    assert c.get("/api/live/status", auth=("me", "wrong")).status_code == 401
    assert c.get("/api/live/status", auth=("me", "s3cret")).status_code == 200


def test_broken_json_is_repaired_without_resending(monkeypatch):
    import asyncio
    from app.providers import google
    calls = []

    async def fake_generate(model_id, body, timeout=600.0):
        calls.append((model_id, body["contents"][0]["parts"]))
        text = '{"a": 1,}' if len(calls) == 1 else '{"a": 1}'
        return {"candidates": [{"content": {"parts": [{"text": text}]}}], "usageMetadata": {}}

    monkeypatch.setattr(google, "generate", fake_generate)
    video = google.youtube_part("https://youtu.be/V-7zASRCXaQ")
    data, _ = asyncio.run(google.generate_json("gemini-pro", [video, {"text": "watch"}]))
    assert data == {"a": 1} and len(calls) == 2
    assert all("fileData" not in part for part in calls[1][1])  # the repair call does not carry the video


def test_analysis_is_cached(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import live, storage
    from app.main import app
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    watched = []

    async def fake_fetch(url):
        return {"title": "T", "channel": "C", "duration": 60, "published": "", "views": "", "thumb": "", "comments_scanned": 0, "comments": []}

    async def fake_json(model_id, parts, max_tokens=32768, low_media=False):
        watched.append(model_id)
        return {"summary": "s", "scenes": [], "beats": [], "gaps": []}, 0.01

    prompts = []

    async def fake_write(writer_id, progress, prompt):
        prompts.append(prompt)
        return {"angles": [{"title": f"A{i}", "type": "Same video, new niche" if i % 2 else "Fill a gap", "score": 60 + i * 5,
                            "keeps": "the hook", "changes": "the niche"} for i in range(6)]}, 0.01, "Gemini Flash", None

    monkeypatch.setattr(live.youtube, "fetch", fake_fetch)
    monkeypatch.setattr(live.google, "generate_json", fake_json)
    monkeypatch.setattr(live, "write_json", fake_write)
    c = TestClient(app)

    def run():
        jid = c.post("/api/live/ingest", json={"url": "https://youtu.be/V-7zASRCXaQ", "analyst": "gemini-flash", "writer": "gemini-flash-writer"}).json()["id"]
        import time
        for _ in range(100):
            j = c.get(f"/api/live/jobs/{jid}").json()
            if j["status"] != "running" and j["status"] != "queued":
                return j
            time.sleep(0.02)

    first, second = run(), run()
    assert first["status"] == "done" and second["result"]["cached"] and second["result"]["cost"] == 0
    assert len(watched) == 1  # the video was watched once
    angles = first["result"]["angles"]
    assert [a["title"] for a in angles] == ["A5", "A4", "A3", "A2"]  # only the 4 with the highest viral potential, best first
    assert angles[0]["keeps"] == "the hook" and angles[0]["type"] == "Same video, new niche"
    assert "NOT a continuation" in prompts[0] and "exactly 4" in prompts[0] and "NOT as a story" in prompts[0]


def test_tts_wav_wrapper_is_stripped():
    import io
    from app.providers.google import clean_pcm
    body = struct.pack("<4h", 1000, 2000, -2000, 1000) * 2000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(22050), w.writeframes(body)
    wrapped = buf.getvalue() + b"LIST....c2pa provenance metadata trainedAlgorithmicMedia"
    pcm, rate = clean_pcm(wrapped)
    assert rate == 22050 and len(pcm) == len(body) and b"RIFF" not in pcm and b"trained" not in pcm
    assert pcm[400:600] == body[400:600]  # the middle is untouched; only the 5 ms edges fade
    raw, _ = clean_pcm(body)  # raw samples pass through
    assert len(raw) == len(body)


def _run_job(c, path, body):
    import time
    jid = c.post(path, json=body).json()["id"]
    for _ in range(200):
        j = c.get(f"/api/live/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.02)


def test_publish_kit_and_channel_ideas(monkeypatch):
    from fastapi.testclient import TestClient
    from app import live
    from app.main import app

    async def fake_write(writer_id, progress, prompt):
        if "channels" in prompt:
            return {"story_type": "Practical DIY garden tips, upbeat, for renters", "visual_type": "Warm handmade clay",
                    "channels": [{"name": f"Clay Garden {i}", "handle": f"clay garden {i}!", "tagline": "Tiny gardens, big harvests",
                                  "about": "Weekly clay-animated garden fixes.", "niche": "balcony gardening",
                                  "avatar": "a clay tomato with a smile", "banner": "a clay balcony full of plants"} for i in range(3)]}, 0.01, "Gemini Flash", None
        return {"titles": ["A", "B", "C", "D"], "captions": ["c1 #x", "c2 #y", "c3 #z"], "descriptions": ["d1", "d2", "d3"],
                "thumbnails": [{"concept": "x", "text": "BIG"}], "tags": ["t"]}, 0.01, "Gemini Flash", None

    monkeypatch.setattr(live, "write_json", fake_write)
    c = TestClient(app)
    kit = _run_job(c, "/api/live/publish-kit", {"writer": "gemini-flash-writer", "story": "A story."})
    assert kit["status"] == "done" and kit["result"]["descriptions"] == ["d1", "d2", "d3"]
    ch = _run_job(c, "/api/live/channel-kit", {"writer": "gemini-flash-writer", "story": "A story.", "style_name": "Clay stop-motion"})
    r = ch["result"]
    assert ch["status"] == "done" and len(r["channels"]) == 3 and r["story_type"].startswith("Practical")
    assert r["channels"][0]["handle"] == "@claygarden0"  # handles are cleaned of spaces and symbols


def test_story_from_own_idea_and_visuals_for_pasted_story(monkeypatch):
    from fastapi.testclient import TestClient
    from app import live
    from app.main import app
    seen = []

    async def fake_write(writer_id, progress, prompt):
        seen.append(prompt)
        if "Do not change the story" in prompt:
            return {"visuals": [{"sentence": "A fox found a key.", "visual": "a fox sniffing a brass key in snow"}]}, 0.002, "Gemini Flash", None
        return {"hook_options": ["h1", "h2", "h3"], "story": " ".join(["word"] * 60)}, 0.01, "Gemini Flash", None

    monkeypatch.setattr(live, "write_json", fake_write)
    c = TestClient(app)
    j = _run_job(c, "/api/live/story", {"writer": "gemini-flash-writer", "angles": [{"title": "My idea"}], "target_seconds": 30,
                                         "idea": "A fox finds a key that opens the moon"})
    assert j["status"] == "done", j
    assert "opens the moon" in seen[-1] and "Reference video" not in seen[-1]
    v = _run_job(c, "/api/live/visuals", {"writer": "gemini-flash-writer", "story": "A fox found a key. It glowed."})
    assert v["status"] == "done" and v["result"]["visuals"][0]["visual"].startswith("a fox")


def test_every_caption_style_writes_valid_events():
    cues = [(0.0, 1.5, ["one", "two", "three"]), (1.5, 2.0, ["four"])]
    for style in render.CAPTION_WORDS:
        ass = render.captions_ass(cues, style, 1080, 1920)
        events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
        assert events and ass.count("Style: Default,") == 1, style
    tw = [l for l in render.captions_ass(cues, "typewriter", 1080, 1920).splitlines() if l.startswith("Dialogue:")]
    assert len(tw) == 4 and tw[2].endswith("one two three")  # words appear one at a time


def test_media_check_reports_lost_files():
    from fastapi.testclient import TestClient
    from app import live
    from app.main import app
    (live.MEDIA / "audio").mkdir(parents=True, exist_ok=True)
    kept = live.MEDIA / "audio" / "kept-test.wav"
    kept.write_bytes(b"x")
    try:
        r = TestClient(app).post("/api/live/media-check", json={"paths": [
            "/media/audio/kept-test.wav", "/media/audio/gone.wav", "/media/../app/main.py", "data:image/png;base64,AA"]}).json()
        assert r["missing"] == ["/media/audio/gone.wav", "/media/../app/main.py"]
    finally:
        kept.unlink()


def test_projects_are_backed_up_on_the_server(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import live
    from app.main import app
    monkeypatch.setattr(live, "STATE_FILE", tmp_path / "app_state.json")
    c = TestClient(app)
    assert c.get("/api/live/state").json()["state"] is None
    c.put("/api/live/state", json={"state": {"projects": {"p1": {"title": "One"}}}, "saved_at": 1})
    c.put("/api/live/state", json={"state": {"projects": {"p1": {"title": "Two"}}}, "saved_at": 2})
    got = c.get("/api/live/state").json()
    assert got["saved_at"] == 2 and got["state"]["projects"]["p1"]["title"] == "Two"
    assert (tmp_path / "app_state.prev.json").exists()  # the earlier save is kept


def test_voice_speed_and_pitch_are_exact(tmp_path):
    from app import live
    from app.providers import google
    clip = tmp_path / "v.wav"
    _tone(clip, 2.0)
    assert abs(live.tune_voice(clip, 1.0, 0) - 2.0) < 0.01  # untouched
    assert abs(live.tune_voice(clip, 1.25, 2) - 1.6) < 0.05  # 25% faster, pitch up, length follows speed only
    assert len(google.VOICE_NAMES) == 30 and google.VOICE_NAMES["v-sulafat"] == "Sulafat"


def test_interrupted_render_continues_from_finished_scenes(tmp_path):
    media = tmp_path / "media"
    (media / "audio").mkdir(parents=True)
    for i in range(3):
        _tone(media / "audio" / f"a{i}.wav", 1.0)
    png = base64.b64encode(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c4944415408d763f8cfc0000003010100c9fe92ef0000000049454e44ae426082")).decode()
    img = "data:image/png;base64," + png
    payload = {"final": True, "res": 720, "aspect": "16:9", "caption_style": "karaoke",
               # scenes 2 and 3 share sentence 1 (a split scene), so its caption crosses a cut
               "scenes": [{"image": img, "lines": [0], "caption": "first line here"},
                          {"image": img, "lines": [1], "split": {"index": 0, "of": 2}, "caption": "second line words"},
                          {"image": img, "lines": [1], "split": {"index": 1, "of": 2}, "variant": 1, "caption": "second line words"},
                          {"image": img, "lines": [2], "caption": "third"}],
               "lines": [{"audio": f"/media/audio/a{i}.wav", "text": t} for i, t in enumerate(["first line here", "second line words", "third"])]}
    first = render.render(payload, media, lambda _: None)
    again = render.render(payload, media, lambda _: None)
    assert first["reused"] == 0 and again["reused"] == 4 and first["res"] == 720
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                            str(media / again["url"].removeprefix("/media/"))], capture_output=True, text=True).stdout
    assert probe.splitlines()[0] == "1280,720"


def test_old_cached_analysis_gets_new_angles_without_rewatching(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import live, storage
    from app.main import app
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    url = "https://youtu.be/V-7zASRCXaQ"
    old = {"source": {"title": "T", "channel": "C", "comments": [], "key_frames": [], "transcript": "words"},
           "analysis": {"summary": "s", "gaps": []}, "angles": [{"id": "a1", "type": "Continuation"}] * 10, "written_by": "GPT-5.5"}
    storage.kv_put("live_ingest", f"{url}|gemini-flash|gemini-flash-writer", old)
    watched = []

    async def fake_json(*a, **k):
        watched.append(1)
        return {}, 0.0

    async def fake_write(writer_id, progress, prompt):
        return {"angles": [{"title": f"N{i}", "type": "Fill a gap", "score": 90 - i} for i in range(4)]}, 0.003, "Gemini Flash", None

    monkeypatch.setattr(live.google, "generate_json", fake_json)
    monkeypatch.setattr(live, "write_json", fake_write)
    j = _run_job(TestClient(app), "/api/live/ingest", {"url": url, "analyst": "gemini-flash", "writer": "gemini-flash-writer"})
    assert j["status"] == "done", j
    assert [a["title"] for a in j["result"]["angles"]] == ["N0", "N1", "N2", "N3"] and not watched  # no re-watching


def test_flux_on_fal(monkeypatch, tmp_path):
    import asyncio
    import httpx
    from app.providers import fal
    jpeg = b"\xff\xd8\xff\xe0fakejpeg"
    sent = []

    def handler(req):
        import json as _j
        sent.append((str(req.url), req.headers.get("authorization"), _j.loads(req.content)))
        return httpx.Response(200, json={"images": [{"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}], "has_nsfw_concepts": [False]})

    real = httpx.AsyncClient
    monkeypatch.setattr(fal.httpx, "AsyncClient", lambda **k: real(transport=httpx.MockTransport(handler), **k))
    monkeypatch.setenv("FAL_KEY", "k123")
    prompt = "clay look. Scene: a fox. Avoid: photorealism, text. No words, captions, watermarks or logos in the image."
    path, cost = asyncio.run(fal.draw(prompt, "9:16", "flux-dev", tmp_path / "img"))
    url, auth, body = sent[0]
    assert url == "https://fal.run/fal-ai/flux/dev" and auth == "Key k123" and path.read_bytes() == jpeg and cost == 0.05
    assert "Avoid" not in body["prompt"] and "photorealism" not in body["prompt"]  # FLUX would draw what an avoid list names
    assert 1_800_000 < body["image_size"]["width"] * body["image_size"]["height"] <= 2_000_000 and body["sync_mode"]  # sharp at 1080p, billed as 2 MP
    monkeypatch.setattr(fal.httpx, "AsyncClient", lambda **k: real(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "no"})), **k))
    try:
        asyncio.run(fal.draw(prompt, "1:1", "flux-schnell", tmp_path / "img2"))
        raise AssertionError("expected an error")
    except ProviderError as e:
        assert "refused the API key" in str(e)


def test_silence_between_sentences_is_trimmed(tmp_path):
    """Each clip has about 0.4 s of silence before and 0.5 s after the words; joined, that was a pause at every cut."""
    import array
    import math
    import subprocess
    import wave as _w
    rate = 24000
    media = tmp_path / "media"
    (media / "audio").mkdir(parents=True)
    for i in range(2):
        speech = array.array("h", (int(8000 * math.sin(2 * math.pi * 220 * n / rate)) for n in range(rate)))  # 1.0 s of "speech"
        pcm = array.array("h", [0] * int(0.4 * rate)) + speech + array.array("h", [0] * int(0.5 * rate))
        with _w.open(str(media / "audio" / f"s{i}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    png = base64.b64encode(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c4944415408d763f8cfc0000003010100c9fe92ef0000000049454e44ae426082")).decode()
    img = "data:image/png;base64," + png
    out = render.render({"final": False, "aspect": "1:1", "caption_style": "none",
                         "scenes": [{"image": img, "lines": [0]}, {"image": img, "lines": [1]}],
                         "lines": [{"audio": f"/media/audio/s{i}.wav", "text": "x"} for i in range(2)]}, media, lambda _: None)
    assert all(abs(s - 1.1) < 0.03 for s in out["line_secs"]), out["line_secs"]  # 1.0 s of words + 0.1 s of breath, not 1.9 s
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(media / out["url"].removeprefix("/media/"))], capture_output=True, text=True).stdout)
    assert abs(dur - 2.2) < 0.1  # each image still lasts exactly as long as its own sentence


def test_scene_visuals_carry_the_message(monkeypatch):
    from fastapi.testclient import TestClient
    from app import live
    from app.main import app
    prompts = []

    async def fake_write(writer_id, progress, prompt):
        prompts.append(prompt)
        return {"scenes": [{"n": 1, "visual": "a hand drops a coin into a glass jar", "text": "save money now please", "key": True}]}, 0.004, "Gemini Flash", None

    monkeypatch.setattr(live, "write_json", fake_write)
    j = _run_job(TestClient(app), "/api/live/scene-visuals", {"writer": "gemini-flash-writer", "scenes": [{"n": 1, "text": "Start saving a little every day."}],
                                                               "style_name": "Stick figure", "idea": "every image shows one or two bold words", "words_max": 2})
    assert j["status"] == "done", j
    r = j["result"]["scenes"]["1"]
    assert r["visual"].startswith("a hand") and r["text"] == "save money" and r["key"]  # the idea allows 2 words
    assert "follow it exactly): every image shows one or two bold words" in prompts[0] and "sound off" in prompts[0]


def test_scene_visual_descriptions_never_carry_their_own_words():
    from app.live import _no_words
    assert _no_words("A man holds an empty wallet. Bold words **PAYDAY SPENT** float above.") == "A man holds an empty wallet."


def test_claude_writer(monkeypatch):
    import asyncio
    from types import SimpleNamespace as NS
    from app.providers.anthropic_writer import ClaudeWriter
    sent = {}

    class FakeStream:
        def __init__(self, msg): self.msg = msg
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get_final_message(self): return self.msg

    def client_returning(msg):
        def stream(**kw):
            sent.update(kw)
            return FakeStream(msg)
        return NS(beta=NS(messages=NS(stream=stream)))

    ok = NS(stop_reason="end_turn", content=[NS(type="thinking", thinking=""), NS(type="text", text='{"story": "Once"}')],
            usage=NS(input_tokens=1000, output_tokens=2000))
    data, cost = asyncio.run(ClaudeWriter("claude-opus-5-5", client_returning(ok)).complete_json("sys", "write it"))
    assert data == {"story": "Once"} and abs(cost - (1000 * 4 + 2000 * 20) / 1e6) < 1e-9
    assert sent["model"] == "claude-opus-5-5" and sent["output_config"] == {"effort": "medium"} and sent["fallbacks"] == "default"
    assert "thinking" not in sent and "temperature" not in sent  # adaptive thinking is the default; sampling params are rejected
    refused = NS(stop_reason="refusal", content=[], usage=NS(input_tokens=1, output_tokens=0))
    try:
        asyncio.run(ClaudeWriter("claude-sonnet-5-5", client_returning(refused)).complete_json("sys", "x"))
        raise AssertionError("expected an error")
    except ProviderError as e:
        assert "declined" in str(e)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        asyncio.run(ClaudeWriter("claude-opus-5-5").complete_json("sys", "x"))
        raise AssertionError("expected an error")
    except ProviderError as e:
        assert "ANTHROPIC_API_KEY" in str(e)


def test_claude_is_the_brain_when_its_key_is_set(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("ANGLE_STUDIO_WRITER", raising=False)
    s = TestClient(app).get("/api/live/status").json()
    assert s["providers"]["anthropic"] and s["writer"] == "claude-opus-5-5"


def test_script_type_reaches_the_writer(monkeypatch):
    from fastapi.testclient import TestClient
    from app import live
    from app.main import app
    seen = []

    async def fake_write(writer_id, progress, prompt):
        seen.append(prompt)
        return {"hook_options": ["a", "b", "c"], "story": " ".join(["word"] * 60)}, 0.01, "Claude Opus 5.5", None

    monkeypatch.setattr(live, "write_json", fake_write)
    j = _run_job(TestClient(app), "/api/live/story", {"writer": "claude-opus-5-5", "angles": [{"title": "Everyone is gone"}],
                                                       "target_seconds": 30, "script_type": "narration"})
    assert j["status"] == "done" and "NARRATION in second person" in seen[-1] and "original script" in seen[-1]


def test_background_music_under_the_voice(tmp_path):
    """The track is looped to the video's length and mixed under the narration; the video still ends with the voice."""
    import subprocess
    media = tmp_path / "media"
    (media / "audio").mkdir(parents=True)
    (media / "music").mkdir()
    _tone(media / "audio" / "v.wav", 3.0)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=1.2", "-ac", "2",
                    str(media / "music" / "m.mp3")], check=True)  # shorter than the video: must loop
    png = base64.b64encode(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c4944415408d763f8cfc0000003010100c9fe92ef0000000049454e44ae426082")).decode()
    for duck in (True, False):
        out = render.render({"final": True, "res": 480, "aspect": "1:1", "caption_style": "none",
                             "scenes": [{"image": "data:image/png;base64," + png, "lines": [0]}],
                             "lines": [{"audio": "/media/audio/v.wav", "text": "x"}],
                             "music": {"url": "/media/music/m.mp3", "volume": 0.3, "duck": duck}}, media, lambda _: None)
        video = media / out["url"].removeprefix("/media/")
        info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,channels:format=duration", "-of", "json", str(video)],
                              capture_output=True, text=True).stdout
        import json as _j
        d = _j.loads(info)
        assert abs(float(d["format"]["duration"]) - 3.0) < 0.15
        assert any(st["codec_type"] == "audio" and st.get("channels") == 2 for st in d["streams"])  # stereo mix


def test_music_upload(monkeypatch, tmp_path):
    import subprocess
    from fastapi.testclient import TestClient
    from app import live
    from app.main import app
    monkeypatch.setattr(live, "MEDIA", tmp_path / "media")
    track = tmp_path / "t.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(track)], check=True)
    c = TestClient(app)
    r = c.post("/api/live/music?name=Calm%20piano", content=track.read_bytes()).json()
    assert r["url"].startswith("/media/music/") and abs(r["duration"] - 2.0) < 0.2 and r["name"] == "Calm piano"
    assert c.post("/api/live/music", content=b"not music at all").status_code == 422


def test_template_examples_can_be_style_references():
    from app import live
    data, mime = live.load_ref("templates/stickman-1.jpg")
    assert data[:2] == b"\xff\xd8" and mime == "image/jpeg"
    import pytest as _p
    from fastapi import HTTPException as _H
    with _p.raises(_H):
        live.load_ref("templates/../../server/.env")  # nothing outside the templates folder


def test_flux_kontext_gets_the_template_example(monkeypatch, tmp_path):
    import asyncio
    import httpx
    from app.providers import fal
    sent = []

    def handler(req):
        import json as _j
        sent.append((str(req.url), _j.loads(req.content)))
        return httpx.Response(200, json={"images": [{"url": "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8x").decode()}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(fal.httpx, "AsyncClient", lambda **k: real(transport=httpx.MockTransport(handler), **k))
    monkeypatch.setenv("FAL_KEY", "k")
    from app import live
    ref = live.load_ref("templates/stickman-1.jpg")
    path, cost = asyncio.run(fal.draw("stick figure. Scene: a jar. Avoid: shading. No watermarks or signatures.", "9:16", "flux-kontext", tmp_path / "k", ref))
    url, body = sent[0]
    assert url.endswith("/fal-ai/flux-pro/kontext") and body["image_url"].startswith("data:image/jpeg;base64,/9j/") and body["aspect_ratio"] == "9:16"
    assert body["prompt"].startswith("Create a completely new image in exactly the same visual style") and cost == 0.04
    try:
        asyncio.run(fal.draw("x", "9:16", "flux-kontext", tmp_path / "k2", None))
        raise AssertionError("expected an error")
    except ProviderError as e:
        assert "example pictures" in str(e)


def test_kontext_uses_the_style_swatch():
    from app.live import kontext_ref
    assert kontext_ref("templates/stickman-1.jpg") == "templates/stickman-style.jpg"
    assert kontext_ref("/media/images/abc.png") == "/media/images/abc.png"  # a scene image or upload stays as it is


def test_text_cards(tmp_path, monkeypatch):
    """Cards are drawn for free in the template's look; the writer's cards are kept off scene 1 and never back to back."""
    import subprocess
    from fastapi.testclient import TestClient
    from app import live
    from app.main import app
    from app.cards import make_card
    out = make_card({"text": "Save *first*", "small": "Number 3", "layout": "number"}, {"id": "stickman", "kind": "flat", "palette": ["#1A1A1A", "#9AA0A6", "#FAF8F3", "#E4572E"]}, 360, 640, tmp_path / "c.png")
    assert out.read_bytes()[:4] == b"\x89PNG"
    c = TestClient(app)
    r = c.get("/api/live/card", params={"text": "What if?", "layout": "question", "style": "pixel", "palette": "#1F1B3A,#3F6EA8,#F3E7C9,#E9724C"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"

    async def fake_write(writer_id, progress, prompt):
        return {"scenes": [{"n": n, "visual": "v", "text": "w", "card": True, "card_text": f"Point {n}", "card_layout": "big"} for n in range(1, 7)]}, 0.003, "Claude Opus 5.5", None
    monkeypatch.setattr(live, "write_json", fake_write)
    j = _run_job(c, "/api/live/scene-visuals", {"writer": "claude-opus-5-5", "scenes": [{"n": n, "text": "x"} for n in range(1, 7)], "cards_share": 0.3})
    cards = [n for n, v in sorted(j["result"]["scenes"].items(), key=lambda kv: int(kv[0])) if v["card"]]
    assert "1" not in cards and all(int(b) - int(a) > 1 for a, b in zip(cards, cards[1:])) and len(cards) <= 3

    media = tmp_path / "media"; (media / "audio").mkdir(parents=True)
    _tone(media / "audio" / "a.wav", 1.0); _tone(media / "audio" / "b.wav", 1.0)
    png = base64.b64encode(bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c4944415408d763f8cfc0000003010100c9fe92ef0000000049454e44ae426082")).decode()
    res = render.render({"final": False, "aspect": "9:16", "caption_style": "none", "card_style": {"id": "cinematic", "palette": ["#0E1A24", "#1F4E5A", "#E7B15A", "#C2563A"]},
                         "scenes": [{"image": "data:image/png;base64," + png, "lines": [0]}, {"image": "", "lines": [1], "card": {"text": "Nobody came *back*", "layout": "big"}}],
                         "lines": [{"audio": "/media/audio/a.wav", "text": "x"}, {"audio": "/media/audio/b.wav", "text": "y"}]}, media, lambda _: None)
    assert (media / res["url"].removeprefix("/media/")).exists()
