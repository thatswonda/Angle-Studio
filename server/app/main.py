"""Angle Studio API. Run with: uvicorn app.main:app --reload (from server/)."""
from __future__ import annotations

import base64
import hmac
import os

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import config  # noqa: F401  (loads server/.env before anything reads a key)
from . import jobs, live, pipeline, storage
from .models import Pacing, Project, new_id
from .providers import registry, sample

app = FastAPI(title="Angle Studio API", version="0.2.0")


@app.middleware("http")
async def require_password(request: Request, call_next):
    """When APP_PASSWORD is set (always, once hosted), every page, API call and media file needs it.
    The browser asks once and remembers it; any user name works."""
    password = os.getenv("APP_PASSWORD")
    if password and request.url.path != "/healthz":
        ok = False
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                ok = hmac.compare_digest(base64.b64decode(auth[6:]).decode().split(":", 1)[1], password)
            except Exception:
                ok = False
        if not ok:
            return Response("Angle Studio needs its password.", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Angle Studio", charset="UTF-8"'})
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.include_router(live.router)
live.MEDIA.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=live.MEDIA), name="media")
app.mount("/templates", StaticFiles(directory=Path(__file__).resolve().parents[2] / "web" / "templates"), name="templates")
WEB = Path(__file__).resolve().parents[2] / "web" / "index.html"
CHECKPOINT_ORDER = ["angles", "script", "style", "scenes", "voice", "format", "clips"]


@app.exception_handler(pipeline.CheckpointError)
async def _checkpoint(_: Request, e: pipeline.CheckpointError):
    return JSONResponse(status_code=409, content={"error": "checkpoint", "message": str(e)})


@app.exception_handler(pipeline.CostLimitError)
async def _cost(_: Request, e: pipeline.CostLimitError):
    return JSONResponse(status_code=402, content={"error": "cost_limit", "message": str(e), "estimate": e.estimate, "spent": e.spent, "limit": e.limit})


def _project(pid: str) -> Project:
    try:
        return storage.load(pid)
    except FileNotFoundError:
        raise HTTPException(404, f"No project {pid}")


def _deep_merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        base[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


# ---------- catalog

@app.get("/api/models")
def models():
    return registry.available()


@app.get("/api/voices")
def voices():
    return sample.VOICES


@app.get("/api/styles")
def styles():
    return sample.STYLES


# ---------- projects

class NewProject(BaseModel):
    title: str = "Untitled project"
    template_id: Optional[str] = None


@app.get("/api/projects")
def list_projects():
    return [{"id": p.id, "title": p.title, "status": p.status, "video_type": p.video_type, "spent": p.render.cost_actual}
            for p in sorted(storage.list_projects(), key=lambda p: -p.created_at)]


@app.post("/api/projects", status_code=201)
def create_project(body: NewProject):
    p = Project(title=body.title, template_id=body.template_id)
    storage.save(p)
    return p


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _project(pid)


@app.patch("/api/projects/{pid}")
def patch_project(pid: str, patch: dict[str, Any]):
    """Edits from the UI: script lines, scene prompts, chosen models (MDL-2), format and so on."""
    p = _project(pid)
    try:
        merged = Project.model_validate(_deep_merge(p.model_dump(), patch))
    except ValidationError as e:
        raise HTTPException(422, [{"field": ".".join(map(str, err["loc"])), "message": err["msg"]} for err in e.errors()])
    storage.save(merged)
    return merged


@app.delete("/api/projects/{pid}", status_code=204)
def delete_project(pid: str):
    storage.delete(pid)


@app.get("/api/projects/{pid}/estimate")
def get_estimate(pid: str):
    return pipeline.estimate(_project(pid))


@app.get("/api/projects/{pid}/originality")
def originality(pid: str):
    p = _project(pid)
    if not (p.script and p.source):
        raise HTTPException(409, "Write a script first.")
    score = pipeline.overlap(p.script.full_text, p.source.transcript)
    return {"overlap": round(score, 3), "too_close": score > 0.25}


# ---------- checkpoints (PRJ-2)

@app.post("/api/projects/{pid}/checkpoints/{name}")
def approve(pid: str, name: str):
    if name not in CHECKPOINT_ORDER:
        raise HTTPException(404, f"Unknown checkpoint {name}")
    p = _project(pid)
    earlier = CHECKPOINT_ORDER[:CHECKPOINT_ORDER.index(name)]
    if name == "clips" and not any(s.animate for s in p.scenes):
        raise HTTPException(409, "No scenes are animated, so there are no clips to approve.")
    pipeline.require(p, *[c for c in earlier if not (c == "clips")])
    if name == "script":
        pipeline.approve_story(p)  # the story becomes sentences for scenes and narration
    if name == "scenes" and not (p.scenes and all(s.image_status == "ready" for s in p.scenes)):
        raise HTTPException(409, "Every scene needs an image before the scenes can be approved.")
    setattr(p.checkpoints, name, True)
    storage.save(p)
    return p.checkpoints


@app.delete("/api/projects/{pid}/checkpoints/{name}")
def reopen(pid: str, name: str):
    """Reopening a step clears it and every later checkpoint; their work stays for re-approval."""
    p = _project(pid)
    for c in CHECKPOINT_ORDER[CHECKPOINT_ORDER.index(name):]:
        setattr(p.checkpoints, c, False)
    p.render.draft_path = p.render.final_path = None
    p.status = "draft"
    storage.save(p)
    return p.checkpoints


# ---------- stages (each returns a job)

class IngestBody(BaseModel):
    url: str
    transcript: Optional[str] = None


class Paid(BaseModel):
    confirm_over_limit: bool = False


class AudioBody(Paid):
    line_ids: Optional[list[str]] = None


class AnimateBody(Paid):
    scene_ids: Optional[list[str]] = None


class RenderBody(BaseModel):
    final: bool = False


@app.post("/api/projects/{pid}/ingest", status_code=202)
async def ingest(pid: str, body: IngestBody):
    _project(pid)
    return jobs.submit(pid, "Analyzing reference video", lambda prog: pipeline.ingest(pid, body.url, body.transcript, prog))


@app.post("/api/projects/{pid}/script", status_code=202)
async def script(pid: str):
    pipeline.require(_project(pid), "angles")
    return jobs.submit(pid, "Writing script", lambda prog: pipeline.write_script(pid, prog))


@app.post("/api/projects/{pid}/audio", status_code=202)
async def audio(pid: str, body: AudioBody):
    p = _project(pid)
    pipeline.require(p, "scenes")
    pipeline.guard_cost(p, pipeline.estimate(p)["voice"], body.confirm_over_limit)
    return jobs.submit(pid, "Generating narration", lambda prog: pipeline.generate_audio(pid, body.line_ids, prog))


@app.get("/api/projects/{pid}/pacing")
def pacing(pid: str):
    """Scene and image counts for Simple, Moderate and Compact at the script's current length."""
    return pipeline.pacing_options(_project(pid))


@app.post("/api/projects/{pid}/scenes/plan")
def plan_scenes(pid: str, body: Optional[Pacing] = None):
    """Free, right after the script: splits it into scenes at the chosen pacing. Re-planning discards scene images."""
    p = _project(pid)
    pipeline.require(p, "script", "style")
    if p.checkpoints.scenes:
        raise HTTPException(409, "The scenes are approved. Reopen them to change the pacing.")
    return pipeline.plan_scenes(pid, body)


@app.post("/api/projects/{pid}/scenes/images", status_code=202)
async def scene_images(pid: str, body: Paid):
    p = _project(pid)
    pipeline.require(p, "script", "style")
    if not p.scenes:
        raise HTTPException(409, "Plan the scenes first.")
    pipeline.guard_cost(p, pipeline.estimate(p)["images"], body.confirm_over_limit)
    ids = [s.id for s in p.scenes if s.image_status in ("pending", "failed")]
    return jobs.submit(pid, f"Generating {len(ids)} images", lambda prog: pipeline.generate_images(pid, ids, prog))


@app.post("/api/projects/{pid}/scenes/{sid}/image", status_code=202)
async def regenerate_image(pid: str, sid: str, body: Paid):
    p = _project(pid)
    pipeline.guard_cost(p, registry.model("image", p.models.image)["cost"], body.confirm_over_limit)
    return jobs.submit(pid, "Regenerating one image", lambda prog: pipeline.generate_image(pid, sid, prog))


@app.post("/api/projects/{pid}/animate", status_code=202)
async def animate(pid: str, body: AnimateBody):
    p = _project(pid)
    pipeline.require(p, "scenes")
    ids = body.scene_ids or [s.id for s in p.scenes if s.animate]
    pipeline.guard_cost(p, pipeline.estimate(p)["video"], body.confirm_over_limit)
    return jobs.submit(pid, "Animating scenes", lambda prog: pipeline.animate(pid, ids, prog))


@app.post("/api/projects/{pid}/render", status_code=202)
async def render(pid: str, body: RenderBody):
    pipeline.require(_project(pid), "scenes")
    return jobs.submit(pid, "Rendering final MP4" if body.final else "Building 480p draft", lambda prog: pipeline.compile_video(pid, body.final, prog))


@app.get("/api/projects/{pid}/jobs")
def project_jobs(pid: str):
    return jobs.for_project(pid)


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "No such job")
    return j


# ---------- library: idea bank (ANG-9, PRJ-5) and templates (PRJ-4)

@app.get("/api/ideas")
def ideas():
    return storage.kv_list("ideas")


@app.post("/api/ideas", status_code=201)
def save_idea(body: dict[str, Any]):
    idea = {"id": new_id("idea"), **body}
    storage.kv_put("ideas", idea["id"], idea)
    return idea


@app.delete("/api/ideas/{iid}", status_code=204)
def delete_idea(iid: str):
    storage.kv_delete("ideas", iid)


@app.get("/api/templates")
def templates():
    return storage.kv_list("templates")


@app.post("/api/templates", status_code=201)
def save_template(body: dict[str, Any]):
    t = {"id": new_id("tpl"), **body}
    storage.kv_put("templates", t["id"], t)
    return t


# ---------- web app

@app.get("/")
def index():
    """Serves the web app. The page file has no document skeleton (claude.ai adds one when publishing), so add it here."""
    page = WEB.read_text()
    if not page.lstrip().lower().startswith("<!doctype"):
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'
                + page + "</body></html>")
    return HTMLResponse(page)
