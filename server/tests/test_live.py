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


def test_captions_cover_every_word():
    ass = render.captions([{"text": "one two three four five"}, {"text": "six seven"}], [0, 2], [2, 1], "bold", 1080, 1920)
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert len(lines) == 3 and "ONE TWO THREE" in lines[0] and lines[-1].endswith("SIX SEVEN")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_render_makes_a_video(tmp_path):
    media = tmp_path / "media"
    (media / "audio").mkdir(parents=True)
    for i in range(2):
        with wave.open(str(media / "audio" / f"a{i}.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000)
            w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(n / 20))) for n in range(24000)))
    png = base64.b64encode(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c4944415408d763f8cfc0000003010100c9fe92ef0000000049454e44ae426082")).decode()
    url = render.render({"final": False, "aspect": "1:1", "caption_style": "clean",
                         "scenes": [{"image": "data:image/png;base64," + png, "duration": 1, "text": "hello there"},
                                    {"image": "data:image/png;base64," + png, "duration": 1, "text": "and goodbye"}],
                         "lines": [{"audio": "/media/audio/a0.wav"}, {"audio": "/media/audio/a1.wav"}]}, media, lambda _: None)
    out = media / url.removeprefix("/media/")
    assert out.exists() and out.stat().st_size > 1000
    with pytest.raises(ProviderError):  # a missing voice clip is reported, not crashed on
        render.render({"scenes": [{"image": "data:image/png;base64," + png, "duration": 1, "text": "x"}], "lines": [{"audio": ""}]}, media, lambda _: None)
