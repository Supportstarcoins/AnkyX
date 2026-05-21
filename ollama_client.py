from __future__ import annotations

from urllib.parse import urlparse, urlunparse
from typing import Any

import requests


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "",
        auto_fallback: bool = True,
    ) -> None:
        self.base_url = self._normalize_base_url(base_url)
        self.model = model
        self.auto_fallback = bool(auto_fallback)

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
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.append(name)
        return names

    def is_model_available(self, model: str) -> bool:
        try:
            models = self.list_models()
        except requests.RequestException:
            return True
        return model in models

    def _resolve_model(self, requested_model: str) -> str:
        available_models = self.list_models()
        if requested_model in available_models:
            return requested_model
        if not available_models:
            raise RuntimeError("В Ollama нет установленных моделей. Выполните: ollama pull llama3.1:8b")
        models_preview = ", ".join(available_models[:20])
        if self.auto_fallback and available_models:
            return available_models[0]
        raise RuntimeError(f"Модель '{requested_model}' не найдена в Ollama. Доступные модели: {models_preview}")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        requested_model = str(model or self.model or "").strip()
        if not requested_model:
            raise RuntimeError("Не задана Ollama model. Откройте AI настройки и выберите модель.")
        target_model = self._resolve_model(requested_model)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": options or {"temperature": 0.4},
        }
        endpoint = self._chat_endpoint()
        timeout = int((options or {}).get("timeout") or 180)
        response = requests.post(endpoint, json=payload, timeout=max(30, timeout))
        if response.status_code == 404:
            response_excerpt = (response.text or "")[:300]
            available_models = self.list_models()
            models_preview = ", ".join(available_models[:20]) if available_models else "нет данных"
            if self.auto_fallback and available_models:
                payload["model"] = available_models[0]
                response = requests.post(endpoint, json=payload, timeout=max(30, timeout))
            if response.status_code == 404:
                raise RuntimeError(
                    "Ollama HTTP 404: проверьте имя модели и Ollama URL. URL должен быть "
                    "http://127.0.0.1:11434, модель должна существовать в ollama list. "
                    f"endpoint={endpoint} model={target_model} base_url={self.base_url} "
                    f"response={response_excerpt} available={models_preview}"
                )
        response.raise_for_status()
        return response.json() if response.content else {}
