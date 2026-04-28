from __future__ import annotations

from urllib.parse import urlparse, urlunparse
from typing import Any

import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.1:8b") -> None:
        self.base_url = self._normalize_base_url(base_url)
        self.model = model

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        raw = (base_url or "").strip()
        if not raw:
            return ""
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
        path = (parsed.path or "").rstrip("/")
        for suffix in ("/api/chat", "/api"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        normalized = urlunparse((parsed.scheme or "http", parsed.netloc, path.rstrip("/"), "", "", ""))
        return normalized.rstrip("/")

    def _chat_endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=30)
        response.raise_for_status()
        data = response.json() if response.content else {}
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        names: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
        return names

    def is_model_available(self, model: str) -> bool:
        try:
            models = self.list_models()
        except requests.RequestException:
            return True
        return model in models

    def chat(self, messages: list[dict[str, str]], options: dict[str, Any] | None = None) -> str:
        if not self.is_model_available(self.model):
            available_models = self.list_models()
            models_preview = ", ".join(available_models[:20]) if available_models else "нет данных"
            raise RuntimeError(f"Модель '{self.model}' не найдена в Ollama. Доступные модели: {models_preview}")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options or {"temperature": 0.4},
        }
        endpoint = self._chat_endpoint()
        response = requests.post(endpoint, json=payload, timeout=180)
        if response.status_code == 404:
            body_preview = (response.text or "").strip()[:300]
            raise RuntimeError(
                "Ollama HTTP 404: проверьте имя модели и Ollama URL. URL должен быть "
                "http://127.0.0.1:11434, модель должна существовать в ollama list. "
                f"base_url={self.base_url}; model={self.model}; endpoint={endpoint}; response={body_preview}"
            )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        return str(message.get("content") or "").strip()
