from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from schema_validator import SchemaValidationError, load_schema, validate_schema

LOGGER = logging.getLogger(__name__)


CARD_QUALITY_SYSTEM_PROMPT = (
    "Ты опытный преподаватель и эксперт по SRS. "
    "Одна карточка = один факт. Не объединяй несвязанные факты. "
    "Вопросы должны быть однозначными, ответы — выводимыми из материала. "
    "Сложные концепции дроби на простые карточки. "
    "Для терминов: front=термин/чёткий вопрос/cloze, back=определение + короткий пример. "
    "Минимизируй когнитивную нагрузку и максимизируй запоминание. "
    "Верни строго валидный JSON по предоставленной схеме."
)


class ILLMProvider(Protocol):
    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str: ...

    def generate_json(
        self,
        schema_name: str,
        user_input: str,
        settings: dict[str, Any],
        system_prompt: str | None = None,
    ) -> dict[str, Any]: ...

    def generate_image(self, prompt: str, settings: dict[str, Any]) -> dict[str, str]: ...


@dataclass
class ProviderCallError(RuntimeError):
    provider: str
    message: str


class OpenAIProvider:
    name = "openai"

    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
        key = settings.get("api_key") or ""
        if not key:
            raise ProviderCallError(self.name, "OpenAI API key is empty")
        model = settings.get("model") or "gpt-4o-mini"
        base_url = (settings.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": settings.get("temperature", 0.2)},
            timeout=settings.get("timeout", 60),
        )
        if not resp.ok:
            raise ProviderCallError(self.name, f"OpenAI HTTP {resp.status_code}: {resp.text[:240]}")
        data = resp.json()
        return str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))

    def generate_json(self, schema_name: str, user_input: str, settings: dict[str, Any], system_prompt: str | None = None) -> dict[str, Any]:
        schema = load_schema(schema_name)
        prompt = system_prompt or CARD_QUALITY_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"SCHEMA={json.dumps(schema, ensure_ascii=False)}\nINPUT={user_input}"},
        ]
        retries = 2
        last_error = ""
        for i in range(retries + 1):
            raw = self.chat(messages, settings)
            try:
                payload = json.loads(raw)
                validate_schema(schema_name, payload)
                return payload
            except Exception as exc:
                last_error = str(exc)
                if i < retries:
                    messages.append({"role": "user", "content": "Повтори ответ ТОЛЬКО валидным JSON по схеме. Никакого текста."})
                continue
        raise SchemaValidationError(f"invalid json after retries: {last_error}")

    def generate_image(self, prompt: str, settings: dict[str, Any]) -> dict[str, str]:
        key = settings.get("api_key") or ""
        if not key:
            raise ProviderCallError(self.name, "OpenAI API key is empty")
        model = settings.get("image_model") or "gpt-image-1"
        base_url = (settings.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        resp = requests.post(
            f"{base_url}/images/generations",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "prompt": prompt, "size": "1024x1024"},
            timeout=settings.get("timeout", 90),
        )
        if not resp.ok:
            raise ProviderCallError(self.name, f"OpenAI image HTTP {resp.status_code}: {resp.text[:240]}")
        data = resp.json()
        b64 = data.get("data", [{}])[0].get("b64_json")
        if not b64:
            raise ProviderCallError(self.name, "OpenAI image payload is empty")
        content = base64.b64decode(b64)
        out_dir = settings.get("image_dir") or os.path.join("data", "media", "generated")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"img_{int(time.time() * 1000)}.png")
        with open(path, "wb") as fh:
            fh.write(content)
        return {"path": path, "mime": "image/png"}


class OllamaProvider:
    name = "ollama"

    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
        model = settings.get("model") or "llama3.1"
        base_url = (settings.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        resp = requests.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=settings.get("timeout", 60),
        )
        if not resp.ok:
            raise ProviderCallError(self.name, f"Ollama HTTP {resp.status_code}: {resp.text[:240]}")
        data = resp.json() or {}
        return str((data.get("message") or {}).get("content") or "")

    def generate_json(self, schema_name: str, user_input: str, settings: dict[str, Any], system_prompt: str | None = None) -> dict[str, Any]:
        schema = load_schema(schema_name)
        prompt = system_prompt or CARD_QUALITY_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"SCHEMA={json.dumps(schema, ensure_ascii=False)}\nINPUT={user_input}"},
        ]
        retries = 2
        err = ""
        for i in range(retries + 1):
            raw = self.chat(messages, settings)
            try:
                payload = json.loads(raw)
                validate_schema(schema_name, payload)
                return payload
            except Exception as exc:
                err = str(exc)
                if i < retries:
                    messages.append({"role": "user", "content": "Повтори ответ ТОЛЬКО валидным JSON по схеме. Никакого текста."})
        raise SchemaValidationError(f"invalid json after retries: {err}")

    def generate_image(self, prompt: str, settings: dict[str, Any]) -> dict[str, str]:
        raise ProviderCallError(self.name, "Image generation is not supported by Ollama provider")


class LLMRouter:
    def __init__(self, primary: ILLMProvider, fallback: ILLMProvider | None = None):
        self.primary = primary
        self.fallback = fallback

    def _run_with_fallback(self, method: str, *args, **kwargs):
        providers = [self.primary] + ([self.fallback] if self.fallback else [])
        last_exc: Exception | None = None
        for provider in providers:
            if provider is None:
                continue
            try:
                return getattr(provider, method)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                LOGGER.warning("provider call failed", extra={"provider": getattr(provider, "name", "unknown"), "method": method, "error": str(exc)})
        if last_exc:
            raise last_exc
        raise RuntimeError("no providers configured")

    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
        return self._run_with_fallback("chat", messages, settings)

    def generate_json(self, schema_name: str, user_input: str, settings: dict[str, Any], system_prompt: str | None = None) -> dict[str, Any]:
        return self._run_with_fallback("generate_json", schema_name, user_input, settings, system_prompt)

    def generate_image(self, prompt: str, settings: dict[str, Any]) -> dict[str, str]:
        return self._run_with_fallback("generate_image", prompt, settings)
