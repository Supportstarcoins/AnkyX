from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import request


@dataclass
class CloudProviderError(Exception):
    message: str
    status_code: int | None = None
    retry_after: str | None = None

    def __str__(self) -> str:
        return self.message


class XFlashCloudProvider:
    def __init__(self, base_url: str, api_key: str, timeout: int = 180) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        chat_id: int | str | None,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if not self.base_url:
            raise CloudProviderError("Cloud URL не задан.")
        if not self.api_key:
            raise CloudProviderError("API ключ не задан.")
        url = f"{self.base_url}/v1/chat"
        payload = {
            "messages": messages,
            "chat_id": chat_id,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                status = resp.getcode()
        except urlerror.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            message = self._parse_error_body(exc.read())
            raise CloudProviderError(
                message or f"HTTP {exc.code}",
                status_code=exc.code,
                retry_after=retry_after,
            ) from exc
        except urlerror.URLError as exc:
            raise CloudProviderError(f"Ошибка соединения: {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise CloudProviderError(f"Ошибка соединения: {exc}") from exc
        if status != 200:
            raise CloudProviderError(f"HTTP {status}", status_code=status)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise CloudProviderError("Некорректный ответ сервера.") from exc
        return {
            "reply": data.get("reply", ""),
            "credits_spent": data.get("credits_spent", 0),
            "remaining_credits": data.get("remaining_credits", 0),
        }

    def _parse_error_body(self, raw: bytes) -> str:
        if not raw:
            return ""
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return raw.decode("utf-8", errors="ignore")
        if isinstance(data, dict):
            detail = data.get("detail") or data.get("message") or ""
            if isinstance(detail, str):
                return detail
        return ""
