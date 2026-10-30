"""Walks the whole pipeline against the mock adapters: link to rendered video."""
import os
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


def test_link_to_rendered_video(client):
    c = client
    pid = c.post("/api/projects", json={"title": "Balcony follow-up"}).json()["id"]

    wait(c, c.post(f"/api/projects/{pid}/ingest", json={"url": "https://www.youtube.com/watch?v=x"}).json())
    p = c.get(f"/api/projects/{pid}").json()
    assert 8 <= len(p["angles"]) <= 12
    assert all(a["relatedness_reason"] for a in p["angles"])

    # Downstream stages refuse to run before their checkpoint (PRJ-2).
    assert c.post(f"/api/projects/{pid}/script").status_code == 409

    c.patch(f"/api/projects/{pid}", json={"selected_angle_ids": ["a1"], "style": {"template_id": "botanical", "image_prefix": "soft window light"}})
    c.post(f"/api/projects/{pid}/checkpoints/angles")
    wait(c, c.post(f"/api/projects/{pid}/script").json())
    p = c.get(f"/api/projects/{pid}").json()
    assert len(p["script"]["hook_options"]) == 3
    assert c.get(f"/api/projects/{pid}/originality").json()["too_close"] is False

    for cp in ("script", "style"):
        assert c.post(f"/api/projects/{pid}/checkpoints/{cp}").status_code == 200
    wait(c, c.post(f"/api/projects/{pid}/audio", json={}).json())
    p = c.get(f"/api/projects/{pid}").json()
    assert all(l["audio_status"] == "ready" and l["duration"] for l in p["script"]["lines"])

    for cp in ("voice", "format"):
        c.post(f"/api/projects/{pid}/checkpoints/{cp}")
    wait(c, c.post(f"/api/projects/{pid}/scenes", json={}).json())
    p = c.get(f"/api/projects/{pid}").json()
    assert p["scenes"] and all(s["image_status"] == "ready" for s in p["scenes"])

    # Regenerating one scene keeps the earlier version (VIS-3).
    sid = p["scenes"][0]["id"]
    wait(c, c.post(f"/api/projects/{pid}/scenes/{sid}/image", json={}).json())
    assert len(c.get(f"/api/projects/{pid}").json()["scenes"][0]["image_versions"]) == 2

    c.post(f"/api/projects/{pid}/checkpoints/scenes")
    wait(c, c.post(f"/api/projects/{pid}/render", json={"final": False}).json())
    wait(c, c.post(f"/api/projects/{pid}/render", json={"final": True}).json())
    p = c.get(f"/api/projects/{pid}").json()
    assert p["status"] == "rendered" and p["render"]["final_path"]
    assert p["render"]["cost_actual"] > 0


def test_cost_limit_needs_confirmation(client, monkeypatch):
    c = client
    monkeypatch.setenv("ANGLE_STUDIO_COST_LIMIT", "0.01")
    pid = c.post("/api/projects", json={}).json()["id"]
    wait(c, c.post(f"/api/projects/{pid}/ingest", json={"url": "https://example.com/v"}).json())
    c.patch(f"/api/projects/{pid}", json={"selected_angle_ids": ["a2"], "style": {"template_id": "infographic", "image_prefix": "flat vector"}})
    for cp in ("angles",):
        c.post(f"/api/projects/{pid}/checkpoints/{cp}")
    wait(c, c.post(f"/api/projects/{pid}/script").json())
    for cp in ("script", "style"):
        c.post(f"/api/projects/{pid}/checkpoints/{cp}")
    r = c.post(f"/api/projects/{pid}/audio", json={})
    assert r.status_code == 402 and r.json()["error"] == "cost_limit"
    assert c.post(f"/api/projects/{pid}/audio", json={"confirm_over_limit": True}).status_code == 202


def test_models_and_cache(client):
    c = client
    m = c.get("/api/models").json()
    assert set(m) == {"voice", "image", "video"} and all(x["configured"] for x in m["voice"])
    a = c.post("/api/projects", json={}).json()["id"]
    b = c.post("/api/projects", json={}).json()["id"]
    wait(c, c.post(f"/api/projects/{a}/ingest", json={"url": "https://example.com/same"}).json())
    j = wait(c, c.post(f"/api/projects/{b}/ingest", json={"url": "https://example.com/same"}).json())
    assert j["stage"] == "Loaded from cache"  # ING-6
