"""Project folders on disk: data/<project_id>/project.json plus media (PRJ-1).
A small SQLite file holds the idea bank, templates and the ingest cache."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .models import Project

ROOT = Path(os.getenv("ANGLE_STUDIO_DATA", "./data"))


def project_dir(pid: str) -> Path:
    d = ROOT / pid
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(p: Project) -> None:
    path = project_dir(p.id) / "project.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(p.model_dump_json(indent=1))
    tmp.replace(path)  # atomic, so a crash never leaves a half-written project


def load(pid: str) -> Project:
    return Project.model_validate_json((ROOT / pid / "project.json").read_text())


def list_projects() -> list[Project]:
    if not ROOT.exists():
        return []
    return [load(d.name) for d in ROOT.iterdir() if (d / "project.json").exists()]


def delete(pid: str) -> None:
    import shutil
    shutil.rmtree(ROOT / pid, ignore_errors=True)


def _db() -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ROOT / "library.db")
    con.execute("create table if not exists kv (bucket text, key text, value text, primary key (bucket, key))")
    return con


def kv_put(bucket: str, key: str, value: dict) -> None:
    with _db() as con:
        con.execute("insert or replace into kv values (?, ?, ?)", (bucket, key, json.dumps(value)))


def kv_get(bucket: str, key: str) -> dict | None:
    with _db() as con:
        row = con.execute("select value from kv where bucket = ? and key = ?", (bucket, key)).fetchone()
    return json.loads(row[0]) if row else None


def kv_list(bucket: str) -> list[dict]:
    with _db() as con:
        return [json.loads(r[0]) for r in con.execute("select value from kv where bucket = ?", (bucket,))]


def kv_delete(bucket: str, key: str) -> None:
    with _db() as con:
        con.execute("delete from kv where bucket = ? and key = ?", (bucket, key))
