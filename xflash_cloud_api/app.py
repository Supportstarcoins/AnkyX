import time
from typing import List, Optional

import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
from auth import require_api_key
from config import settings
from limits import concurrency_limiter, rate_limiter

app = FastAPI(title="X-FLASH Cloud API", version="1.0.0")

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.6
    meta: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    ok: bool
    reply: str
    credits_spent: int
    remaining_credits: int
    server_time_ms: int


@app.on_event("startup")
async def startup_event() -> None:
    db.init_db()


def _validate_payload(payload: ChatRequest) -> None:
    if payload.meta is not None and len(payload.meta) > settings.max_meta_items:
        raise HTTPException(status_code=413, detail="payload_too_large")

    total_bytes = sum(len(msg.content.encode("utf-8")) for msg in payload.messages)
    if total_bytes > settings.max_payload_bytes:
        raise HTTPException(status_code=413, detail="payload_too_large")


def _count_chars(messages: List[Message]) -> int:
    return sum(len(msg.content) for msg in messages)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest, request: Request, user=Depends(require_api_key)
):
    _validate_payload(payload)

    auth_header = request.headers.get("authorization", "")
    api_key_value = auth_header.replace("Bearer ", "", 1).strip()
    if not rate_limiter.allow(api_key_value):
        raise HTTPException(status_code=429, detail="rate_limited")

    user_id = int(user["id"])
    plan = str(user["plan"]).lower()
    limit, since = db.get_plan_window(plan)
    used = db.get_usage_count(user_id, since)
    if used >= limit:
        raise HTTPException(status_code=429, detail="rate_limited")

    cost = 2 if plan == "pro" else 5
    ok, remaining = db.reserve_credits(user_id, cost)
    if not ok:
        raise HTTPException(status_code=402, detail="not_enough_credits")

    if not concurrency_limiter.try_acquire(user_id):
        db.update_credits(user_id, cost)
        raise HTTPException(status_code=429, detail="server_busy", headers={"Retry-After": "2"})

    start_time = time.monotonic()
    request_chars = _count_chars(payload.messages)
    response_text = ""
    status = "ok"
    try:
        model_name = payload.model or settings.default_model
        max_tokens = min(payload.max_tokens or settings.max_tokens, settings.max_tokens)
        response = requests.post(
            f"{settings.ollama_url.rstrip('/')}/api/chat",
            json={
                "model": model_name,
                "messages": [msg.dict() for msg in payload.messages],
                "stream": False,
                "options": {
                    "temperature": payload.temperature or 0.6,
                    "num_predict": max_tokens,
                },
            },
            timeout=settings.ollama_timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        response_text = data.get("message", {}).get("content", "")
        if not response_text:
            response_text = data.get("response", "")
        if response_text is None:
            response_text = ""
    except requests.RequestException:
        status = "ollama_offline"
        db.update_credits(user_id, cost)
        db.log_usage(
            user_id=user_id,
            credits_spent=0,
            request_chars=request_chars,
            response_chars=0,
            status=status,
        )
        raise HTTPException(status_code=503, detail="ollama_offline")
    finally:
        concurrency_limiter.release(user_id)

    response_chars = len(response_text)
    db.log_usage(
        user_id=user_id,
        credits_spent=cost,
        request_chars=request_chars,
        response_chars=response_chars,
        status=status,
    )

    server_time_ms = int((time.monotonic() - start_time) * 1000)
    return ChatResponse(
        ok=True,
        reply=response_text,
        credits_spent=cost,
        remaining_credits=remaining if remaining is not None else 0,
        server_time_ms=server_time_ms,
    )
