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
    for style in ("clean", "none"):
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
