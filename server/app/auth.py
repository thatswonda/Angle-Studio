"""Accounts: anyone can register with a username, email and password; the owner approves them, gives each a credit
budget in dollars, the models they may use at each step and the longest video they may make. Every paid call is
checked and charged here on the server, so limits hold whatever the browser sends.

The owner is the account registered with ANGLE_STUDIO_ADMIN_EMAIL, or, when that is not set, the first account.
ANGLE_STUDIO_OPEN=1 turns accounts off (local use and tests): everyone is the owner.
"""
from __future__ import annotations

import contextvars
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from typing import Optional

from fastapi import HTTPException

from . import storage

COOKIE = "as_session"
SESSION_DAYS = 30
STEPS = ("analyst", "writer", "voice", "image", "video")
STEP_NAMES = {"analyst": "Video analysis", "writer": "Writing", "voice": "Voiceover", "image": "Images", "video": "Video clips"}
# What a newly approved account may use until the owner changes it: the cheaper models at each step.
DEFAULT_MODELS = {"analyst": ["gemini-flash"], "writer": ["claude-sonnet-5-5", "gemini-flash-writer"], "voice": ["gemini-tts-flash"],
                  "image": ["seedream", "nano-banana-2-lite", "flux-schnell"], "video": []}
DEFAULT_CREDIT = 5.0
DEFAULT_MAX_SECONDS = 180
USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_lock = threading.RLock()
CURRENT: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("current_user", default=None)


def open_mode() -> bool:
    return os.getenv("ANGLE_STUDIO_OPEN") == "1"


OWNER_OPEN = {"username": "owner", "email": "", "role": "admin", "status": "approved", "owner": True}


# ---------- passwords and sessions

def hash_pw(pw: str) -> str:
    salt = os.urandom(16)
    return salt.hex() + "$" + hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32).hex()


def check_pw(pw: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$", 1)
        got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1, dklen=32).hex()
        return hmac.compare_digest(got, h)
    except (ValueError, TypeError):
        return False


def new_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    storage.kv_put("sessions", hashlib.sha256(token.encode()).hexdigest(), {"username": username, "exp": time.time() + SESSION_DAYS * 86400})
    return token


def end_session(token: str) -> None:
    if token:
        storage.kv_delete("sessions", hashlib.sha256(token.encode()).hexdigest())


def user_for_token(token: str | None) -> Optional[dict]:
    if not token:
        return None
    s = storage.kv_get("sessions", hashlib.sha256(token.encode()).hexdigest())
    if not s or s.get("exp", 0) < time.time():
        return None
    return get_user(s["username"])


# ---------- users

def admin_email() -> str:
    return (os.getenv("ANGLE_STUDIO_ADMIN_EMAIL") or "").strip().lower()


def _as_owner(u: dict) -> dict:
    """When ANGLE_STUDIO_ADMIN_EMAIL is set, the account with that email is always the owner (approved, no limits),
    and no other account is, whatever happened before (for example someone else signing up first)."""
    em = admin_email()
    if not u or not em:
        return u
    if u.get("email", "").lower() == em:
        u.update(role="admin", owner=True, status="approved", credit=None, models=None, max_seconds=None)
    elif u.get("owner"):
        u.update(owner=False, role="user", status="pending",
                 credit=DEFAULT_CREDIT, models={k: list(v) for k, v in DEFAULT_MODELS.items()}, max_seconds=DEFAULT_MAX_SECONDS)
    return u


def get_user(username: str) -> Optional[dict]:
    return _as_owner(storage.kv_get("users", (username or "").lower()))


def recover_owner() -> Optional[str]:
    """Owner recovery, run when the server starts: with ANGLE_STUDIO_ADMIN_EMAIL and ANGLE_STUDIO_ADMIN_PASSWORD set,
    the account with that email gets that password (and is created if it does not exist yet), so the owner can always
    sign in again. Remove ANGLE_STUDIO_ADMIN_PASSWORD afterwards and change the password from the account menu."""
    em, pw = admin_email(), os.getenv("ANGLE_STUDIO_ADMIN_PASSWORD") or ""
    if not em or len(pw) < 8:
        return None
    with _lock:
        u = next((x for x in all_users() if x.get("email", "").lower() == em), None)
        if not u:
            base = re.sub(r"[^A-Za-z0-9_.-]", "", em.split("@")[0])[:28] or "owner"
            name = base if len(base) >= 3 else base + "123"
            while get_user(name):
                name = base + secrets.token_hex(2)
            u = {"username": name, "email": em, "created_at": time.time(), "last_active": 0, "spent": 0.0, "usage": {}, "note": ""}
        u["pw"] = hash_pw(pw)
        save_user(_as_owner(u))
        return u["username"]


def save_user(u: dict) -> None:
    storage.kv_put("users", u["username"].lower(), u)


def all_users() -> list[dict]:
    return sorted((_as_owner(u) for u in storage.kv_list("users")), key=lambda u: u.get("created_at", 0))


def register(username: str, email: str, password: str) -> dict:
    username, email = username.strip(), email.strip().lower()
    if not USERNAME.match(username):
        raise HTTPException(422, "Username: 3 to 32 letters, numbers, dots, dashes or underscores.")
    if not EMAIL.match(email):
        raise HTTPException(422, "Enter a valid email address.")
    if len(password) < 8:
        raise HTTPException(422, "Password: at least 8 characters.")
    with _lock:
        if get_user(username):
            raise HTTPException(409, "That username is taken.")
        users = all_users()
        if any(u.get("email") == email for u in users):
            raise HTTPException(409, "An account with that email already exists.")
        owner = (email == admin_email()) if admin_email() else not users
        u = {"username": username, "email": email, "pw": hash_pw(password), "created_at": time.time(), "last_active": 0,
             "role": "admin" if owner else "user", "owner": owner, "status": "approved" if owner else "pending",
             "credit": None if owner else DEFAULT_CREDIT, "spent": 0.0, "models": None if owner else {k: list(v) for k, v in DEFAULT_MODELS.items()},
             "max_seconds": None if owner else DEFAULT_MAX_SECONDS, "usage": {}, "note": ""}
        save_user(u)
    return u


def public(u: dict) -> dict:
    """What the browser may see about an account (never the password hash)."""
    credit = u.get("credit")
    return {"username": u["username"], "email": u.get("email", ""), "role": u.get("role", "user"), "status": u.get("status"),
            "owner": bool(u.get("owner")), "credit": credit, "spent": round(u.get("spent", 0.0), 4),
            "remaining": None if credit is None else round(max(0.0, credit - u.get("spent", 0.0)), 4),
            "models": u.get("models"), "max_seconds": u.get("max_seconds"), "usage": u.get("usage", {}),
            "created_at": u.get("created_at", 0), "last_active": u.get("last_active", 0), "note": u.get("note", ""),
            "projects": u.get("projects", 0), "videos": u.get("videos", 0)}


# ---------- the signed-in user for this request, and the checks every paid call goes through

def current() -> dict:
    u = CURRENT.get()
    if u is None:
        if open_mode():
            return OWNER_OPEN
        raise HTTPException(401, "Sign in to use Angle Studio.")
    return u


def is_admin(u: dict) -> bool:
    return u.get("role") == "admin"


def remaining(u: dict) -> Optional[float]:
    if u.get("credit") is None:
        return None
    return max(0.0, u["credit"] - u.get("spent", 0.0))


def gate(**models: str) -> dict:
    """Call at the start of every paid step: the account must have credit left and be allowed each model it names,
    e.g. gate(writer="claude-opus-5-5", analyst="gemini-flash")."""
    u = current()
    if u.get("status") != "approved":
        raise HTTPException(403, "Your account is waiting for approval.")
    left = remaining(u)
    if left is not None and left <= 0.0001:
        raise HTTPException(402, "Your credit is used up. Ask the owner for more credit to keep creating.")
    allowed = u.get("models")
    if allowed is not None:
        for step, mid in models.items():
            if mid and mid not in (allowed.get(step) or []):
                raise HTTPException(403, f"{STEP_NAMES.get(step, step)} model “{mid}” is not turned on for your account. "
                                         "Pick another model at the top of the app.")
    return u


def check_seconds(seconds: float, label: str = "This video") -> None:
    u = current()
    cap = u.get("max_seconds")
    if cap and seconds > cap * 1.15 + 5:  # a little room: spoken length is only estimated until the voice is made
        m, s = divmod(int(cap), 60)
        raise HTTPException(403, f"{label} is longer than your account allows ({m}:{s:02d}). Shorten the story or ask the owner.")


def charge(cost: float, what: str = "", user: dict | None = None, count: int = 1) -> None:
    """Adds what a step really cost to the account (and counts the step), read fresh so parallel steps add up."""
    u = user or CURRENT.get()
    if not u or u is OWNER_OPEN:
        return
    with _lock:
        fresh = get_user(u["username"])
        if not fresh:
            return
        fresh["spent"] = round(fresh.get("spent", 0.0) + max(0.0, float(cost or 0)), 6)
        if what:
            usage = fresh.setdefault("usage", {})
            usage[what] = usage.get(what, 0) + count
        fresh["last_active"] = time.time()
        save_user(fresh)


def touch(u: dict, **fields) -> None:
    if not u or u is OWNER_OPEN:
        return
    with _lock:
        fresh = get_user(u["username"])
        if fresh:
            fresh.update(fields)
            fresh["last_active"] = time.time()
            save_user(fresh)


# ---------- failed sign-ins: a short pause after several wrong passwords

_fails: dict[str, list[float]] = {}


def too_many(key: str) -> bool:
    now = time.time()
    _fails[key] = [t for t in _fails.get(key, []) if now - t < 600]
    return len(_fails[key]) >= 8


def failed(key: str) -> None:
    _fails.setdefault(key, []).append(time.time())
