from __future__ import annotations

import importlib
import importlib.util
import os
import threading
from typing import Any

import requests


class LLMEngineBase:
    def is_available(self) -> bool:
        return False

    def get_status(self) -> str:
        return "недоступна"

    def chat(self, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaModelNotFoundError(RuntimeError):
    pass


class OllamaEngine(LLMEngineBase):
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model

    def _resolve_url(self, path: str) -> str:
        base = self.base_url or "http://127.0.0.1:11434"
        return f"{base}{path}"

    def is_available(self) -> bool:
        try:
            response = requests.get(self._resolve_url("/api/version"), timeout=1.5)
            return response.ok
        except requests.RequestException:
            return False

    def get_status(self) -> str:
        return "OK" if self.is_available() else "offline"

    def chat(self, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int) -> str:
        if not self.is_available():
            raise OllamaUnavailableError("Ollama недоступна")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = requests.post(
                self._resolve_url("/api/chat"),
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise OllamaUnavailableError("Ollama недоступна") from exc
        if response.status_code == 404:
            raise OllamaModelNotFoundError("Модель не найдена")
        if not response.ok:
            raise OllamaUnavailableError(f"Ollama error: {response.status_code}")
        data = response.json() or {}
        message = data.get("message") or {}
        return str(message.get("content") or "")


class MockEngine(LLMEngineBase):
    def is_available(self) -> bool:
        return True

    def get_status(self) -> str:
        return "заглушка"

    def chat(self, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int) -> str:
        user_message = ""
        for item in reversed(messages or []):
            if item.get("role") == "user":
                user_message = item.get("content", "")
                break
        snippet = user_message.strip()
        if len(snippet) > 200:
            snippet = f"{snippet[:200]}..."
        return f"Это заглушка ответа. Ваш запрос: {snippet}"


class LlamaCppEngine(LLMEngineBase):
    def __init__(
        self,
        model_path: str | None,
        *,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, (os.cpu_count() or 2) // 2)
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._llm = None
        self._lock = threading.Lock()
        self._loading = False
        self._error: str | None = None

    @staticmethod
    def is_llama_cpp_available() -> bool:
        return importlib.util.find_spec("llama_cpp") is not None

    def is_available(self) -> bool:
        return bool(self.model_path and os.path.isfile(self.model_path) and self.is_llama_cpp_available())

    def is_loading(self) -> bool:
        return self._loading

    def is_loaded(self) -> bool:
        return self._llm is not None

    def get_status(self) -> str:
        if self._error:
            return f"ошибка: {self._error}"
        if self._loading:
            return "загрузка..."
        if self._llm:
            return "готово"
        if not self.is_available():
            return "недоступна"
        return "ожидание"

    def _load(self) -> None:
        with self._lock:
            if self._llm is not None or self._loading:
                return
            self._loading = True
        llama_cpp = importlib.import_module("llama_cpp")
        try:
            self._llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )
            self._error = None
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            raise
        finally:
            self._loading = False

    def chat(self, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int) -> str:
        if not self.is_available():
            raise RuntimeError("LLM недоступна. Проверьте наличие модели и llama-cpp-python.")
        if self._llm is None:
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                raise
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            raise
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content") or ""
