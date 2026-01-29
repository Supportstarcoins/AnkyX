from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    text: str
    attachments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatSession:
    id: int
    title: str
    created_at: int
    messages: list[Message] = field(default_factory=list)


@dataclass
class DraftCard:
    front: str
    back: str
    tags: list[str] = field(default_factory=list)
    media: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftBatch:
    draft_id: str
    deck_id: int
    cards: list[DraftCard]
    total_credits: int
    created_at: int
