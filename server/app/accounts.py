"""Sign-in endpoints (/api/auth) and the owner's account controls (/api/admin)."""
from __future__ import annotations

import secrets
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from . import auth, storage

router = APIRouter()


def _cookie(resp: Response, request: Request, token: str) -> None:
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    resp.set_cookie(auth.COOKIE, token, max_age=auth.SESSION_DAYS * 86400, httponly=True, samesite="lax", secure=secure, path="/")


class RegisterBody(BaseModel):
    username: str
    email: str
    password: str


@router.post("/api/auth/register")
def register(body: RegisterBody, request: Request, response: Response):
    u = auth.register(body.username, body.email, body.password)
    _cookie(response, request, auth.new_session(u["username"]))
    return {"user": auth.public(u)}


class LoginBody(BaseModel):
    login: str  # username or email
    password: str


@router.post("/api/auth/login")
def login(body: LoginBody, request: Request, response: Response):
    key = body.login.strip().lower()
    if auth.too_many(key):
        raise HTTPException(429, "Too many tries. Wait 10 minutes and try again.")
    u = auth.get_user(key) or next((x for x in auth.all_users() if x.get("email") == key), None)
    if not u or not auth.check_pw(body.password, u.get("pw", "")):
        auth.failed(key)
        raise HTTPException(401, "Wrong username, email or password.")
    if u.get("status") == "disabled":
        raise HTTPException(403, "This account is turned off. Contact the owner.")
    _cookie(response, request, auth.new_session(u["username"]))
    auth.touch(u)
    return {"user": auth.public(u)}


@router.post("/api/auth/logout")
def logout(request: Request, response: Response):
    auth.end_session(request.cookies.get(auth.COOKIE, ""))
    response.delete_cookie(auth.COOKIE, path="/")
    return {"ok": True}


@router.get("/api/auth/me")
def me():
    u = auth.CURRENT.get()
    if u is None and auth.open_mode():
        u = auth.OWNER_OPEN
    return {"user": auth.public(u) if u else None, "open": auth.open_mode()}


class PasswordBody(BaseModel):
    old: str
    new: str


@router.post("/api/auth/password")
def change_password(body: PasswordBody):
    u = auth.current()
    if u is auth.OWNER_OPEN:
        raise HTTPException(400, "Accounts are off on this server.")
    if not auth.check_pw(body.old, u.get("pw", "")):
        raise HTTPException(401, "Your current password is not right.")
    if len(body.new) < 8:
        raise HTTPException(422, "Password: at least 8 characters.")
    auth.touch(u, pw=auth.hash_pw(body.new))
    return {"ok": True}


# ---------- the owner's controls

def _admin() -> dict:
    u = auth.current()
    if not auth.is_admin(u):
        raise HTTPException(403, "Only the owner can do this.")
    return u


@router.get("/api/admin/users")
def users():
    _admin()
    return {"users": [auth.public(u) for u in auth.all_users()], "steps": auth.STEP_NAMES, "defaults": auth.DEFAULT_MODELS}


class UserChange(BaseModel):
    status: Optional[str] = None          # approved | pending | disabled
    credit: Optional[float] = None        # the account's total budget in dollars
    add_credit: Optional[float] = None    # add to the budget
    models: Optional[dict] = None         # {"image": ["seedream", ...], ...}
    max_seconds: Optional[int] = None
    note: Optional[str] = None
    role: Optional[str] = None            # user | admin
    reset_spent: Optional[bool] = None


@router.patch("/api/admin/users/{username}")
def change_user(username: str, body: UserChange):
    me_ = _admin()
    with auth._lock:
        u = auth.get_user(username)
        if not u:
            raise HTTPException(404, "No such account.")
        if u.get("owner") and (body.status in ("pending", "disabled") or body.role == "user"):
            raise HTTPException(400, "The owner account cannot be turned off.")
        if body.status in ("approved", "pending", "disabled"):
            u["status"] = body.status
            if body.status == "approved" and not u.get("approved_at"):
                u["approved_at"] = time.time()
        if body.credit is not None:
            u["credit"] = max(0.0, round(float(body.credit), 2))
        if body.add_credit:
            u["credit"] = max(0.0, round((u.get("credit") or 0.0) + float(body.add_credit), 2))
        if body.reset_spent:
            u["spent"] = 0.0
        if body.models is not None:
            u["models"] = {s: [str(m) for m in (body.models.get(s) or [])][:40] for s in auth.STEPS}
        if body.max_seconds is not None:
            u["max_seconds"] = max(15, min(3600, int(body.max_seconds)))
        if body.note is not None:
            u["note"] = body.note[:500]
        if body.role in ("user", "admin") and not u.get("owner"):
            u["role"] = body.role  # another admin can manage accounts; their own limits still apply
        auth.save_user(u)
    return {"user": auth.public(u), "by": me_["username"]}


@router.post("/api/admin/users/{username}/reset-password")
def reset_password(username: str):
    _admin()
    u = auth.get_user(username)
    if not u:
        raise HTTPException(404, "No such account.")
    temp = secrets.token_urlsafe(7)
    auth.touch(u, pw=auth.hash_pw(temp))
    return {"password": temp}


@router.delete("/api/admin/users/{username}")
def delete_user(username: str):
    _admin()
    u = auth.get_user(username)
    if not u:
        raise HTTPException(404, "No such account.")
    if u.get("owner"):
        raise HTTPException(400, "The owner account cannot be deleted.")
    storage.kv_delete("users", username.lower())
    return {"ok": True}
