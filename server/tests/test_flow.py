"""Walks the whole pipeline against the mock adapters: link to rendered video."""
import time

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANGLE_STUDIO_DATA", str(tmp_path))
    monkeypatch.setenv("ANGLE_STUDIO_MOCK", "1")
    from app import storage
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def wait(c, job):
    for _ in range(200):
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed"):
            assert j["status"] == "done", j["error"]
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def approve(c, pid, *names):
    for name in names:
        r = c.post(f"/api/projects/{pid}/checkpoints/{name}")
        assert r.status_code == 200, (name, r.json())


def to_story(c, pid, angle="a1", seconds=60):
    wait(c, c.post(f"/api/projects/{pid}/ingest", json={"url": "https://www.youtube.com/watch?v=x"}).json())
    # Length and frame are chosen with the angle, before the story is written.
    c.patch(f"/api/projects/{pid}", json={"format": {"target_seconds": seconds, "aspect_ratio": "9:16"}, "selected_angle_ids": [angle],
                                          "style": {"template_id": "botanical", "image_prefix": "soft window light"}})
    approve(c, pid, "angles")
    wait(c, c.post(f"/api/projects/{pid}/script").json())


def test_link_to_rendered_video(client):
    c = client
    pid = c.post("/api/projects", json={"title": "Balcony follow-up"}).json()["id"]

    wait(c, c.post(f"/api/projects/{pid}/ingest", json={"url": "https://www.youtube.com/watch?v=x"}).json())
    p = c.get(f"/api/projects/{pid}").json()
    assert 8 <= len(p["angles"]) <= 12
    assert all(a["relatedness_reason"] for a in p["angles"])
    # Gemini watched the video; Claude wrote the angles from its report.
    assert p["analysis"]["analyzed_by"] == "Gemini Pro" and p["angles_written_by"] == "Claude Opus 5.5"
    assert p["source"]["key_frames"] and p["source"]["transcript"]

    # Downstream stages refuse to run before their checkpoint (PRJ-2).
    assert c.post(f"/api/projects/{pid}/script").status_code == 409

    to_story(c, pid, seconds=90)
    p = c.get(f"/api/projects/{pid}").json()
    sc = p["script"]
    assert len(sc["hook_options"]) == 3 and sc["written_by"] == "Claude Opus 5.5"
    assert sc["story"].startswith(sc["hook_options"][0]) and "\n\n" in sc["story"] and sc["lines"] == []
    assert c.get(f"/api/projects/{pid}/originality").json()["too_close"] is False

    # Approving the story splits it into sentences.
    approve(c, pid, "script")
    p = c.get(f"/api/projects/{pid}").json()
    assert len(p["script"]["lines"]) > 8 and all(l["text"].rstrip()[-1] in ".!?" for l in p["script"]["lines"])

    # Scenes and images come after the style, before the voice.
    assert c.post(f"/api/projects/{pid}/scenes/plan", json={"mode": "moderate"}).status_code == 409
    approve(c, pid, "style")
    opts = c.get(f"/api/projects/{pid}/pacing").json()
    counts = [o["images"] for o in opts]
    assert counts[0] < counts[1] < counts[2]  # simple < moderate < compact (2 s)
    p = c.post(f"/api/projects/{pid}/scenes/plan", json={"mode": "compact", "compact_every": 2}).json()
    assert len(p["scenes"]) == counts[2] and all(s["image_status"] == "pending" for s in p["scenes"])
    assert any(s["split_of"] for s in p["scenes"])
    p = c.post(f"/api/projects/{pid}/scenes/plan", json={"mode": "simple"}).json()
    assert len(p["scenes"]) == counts[0] and any(len(s["line_ids"]) > 1 for s in p["scenes"])
    p = c.post(f"/api/projects/{pid}/scenes/plan", json={"mode": "moderate"}).json()
    assert len(p["scenes"]) == counts[1]
    assert c.post(f"/api/projects/{pid}/checkpoints/scenes").status_code == 409  # no images yet

    wait(c, c.post(f"/api/projects/{pid}/scenes/images", json={}).json())
    p = c.get(f"/api/projects/{pid}").json()
    assert all(s["image_status"] == "ready" for s in p["scenes"])
    sid = p["scenes"][0]["id"]
    wait(c, c.post(f"/api/projects/{pid}/scenes/{sid}/image", json={}).json())  # redo keeps the earlier version (VIS-3)
    assert len(c.get(f"/api/projects/{pid}").json()["scenes"][0]["image_versions"]) == 2
    approve(c, pid, "scenes")
    assert c.post(f"/api/projects/{pid}/scenes/plan", json={"mode": "simple"}).status_code == 409  # locked once approved

    wait(c, c.post(f"/api/projects/{pid}/audio", json={}).json())
    p = c.get(f"/api/projects/{pid}").json()
    assert all(l["audio_status"] == "ready" and l["duration"] for l in p["script"]["lines"])

    approve(c, pid, "voice", "format")
    from app import assembly, storage
    assert assembly.plan(storage.load(pid), 1080, polish=True)["size"] == [1080, 1920]  # vertical 1080p
    wait(c, c.post(f"/api/projects/{pid}/render", json={"final": False}).json())
    wait(c, c.post(f"/api/projects/{pid}/render", json={"final": True}).json())
    p = c.get(f"/api/projects/{pid}").json()
    assert p["status"] == "rendered" and p["render"]["final_path"]
    assert p["render"]["cost_actual"] > 0


def test_character_story_assigns_speakers(client):
    c = client
    pid = c.post("/api/projects", json={}).json()["id"]
    wait(c, c.post(f"/api/projects/{pid}/ingest", json={"url": "https://example.com/c"}).json())
    c.patch(f"/api/projects/{pid}", json={"video_type": "character"})
    to_story(c, pid, angle="a4")
    p = c.get(f"/api/projects/{pid}").json()
    assert p["script"]["story"].startswith("Tomi: ")
    approve(c, pid, "script")
    p = c.get(f"/api/projects/{pid}").json()
    speakers = {l["speaker"] for l in p["script"]["lines"]}
    assert speakers == {c_["id"] for c_ in p["characters"]}


def test_cost_limit_needs_confirmation(client, monkeypatch):
    c = client
    monkeypatch.setenv("ANGLE_STUDIO_COST_LIMIT", "0.01")
    pid = c.post("/api/projects", json={}).json()["id"]
    to_story(c, pid, angle="a2")
    approve(c, pid, "script", "style")
    c.post(f"/api/projects/{pid}/scenes/plan", json={"mode": "moderate"})
    r = c.post(f"/api/projects/{pid}/scenes/images", json={})
    assert r.status_code == 402 and r.json()["error"] == "cost_limit"
    assert c.post(f"/api/projects/{pid}/scenes/images", json={"confirm_over_limit": True}).status_code == 202


def test_models_and_cache(client):
    c = client
    m = c.get("/api/models").json()
    assert set(m) == {"analyst", "writer", "voice", "image", "video"} and all(x["configured"] for x in m["voice"])
    assert {x["provider"] for x in m["video"]} == {"google", "higgsfield"}
    a = c.post("/api/projects", json={}).json()["id"]
    b = c.post("/api/projects", json={}).json()["id"]
    wait(c, c.post(f"/api/projects/{a}/ingest", json={"url": "https://example.com/same"}).json())
    j = wait(c, c.post(f"/api/projects/{b}/ingest", json={"url": "https://example.com/same"}).json())
    assert j["stage"] == "Loaded from cache"  # ING-6


def test_frame_sizes(client):
    from app import assembly
    from app.models import Project
    p = Project()
    for ratio, size in {"9:16": [1080, 1920], "16:9": [1920, 1080], "1:1": [1080, 1080], "4:5": [1080, 1350]}.items():
        p.format.aspect_ratio = ratio
        assert assembly.plan(p, 1080, polish=False)["size"] == size
