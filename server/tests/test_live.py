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

    async def fake_write(writer_id, progress, prompt):
        return {"angles": [{"title": f"A{i}", "type": "Deep dive", "score": 80} for i in range(4)]}, 0.01, "Gemini Flash", None

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
