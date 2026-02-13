from __future__ import annotations

from typing import Any

import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.1:8b") -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        return str(message.get("content") or "").strip()
