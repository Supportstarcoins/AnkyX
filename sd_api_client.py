from __future__ import annotations

import base64
import binascii
import json
import os
import time
from typing import Any

import requests


class SDAPIError(Exception):
    pass


class SDAPIClient:
    def __init__(self, base_url: str):
        self.base_url = (base_url or "").rstrip("/")

    def _require_url(self) -> None:
        if not self.base_url:
            raise SDAPIError("SD API URL не задан.")

    def health(self) -> bool:
        self._require_url()
        endpoints = ("/sdapi/v1/samplers", "/sdapi/v1/options")
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.ok:
                    return True
            except requests.RequestException:
                continue
            try:
                response = requests.options(f"{self.base_url}{endpoint}", timeout=10)
                if response.ok:
                    return True
            except requests.RequestException:
                continue
        return False

    def set_checkpoint(self, checkpoint_name: str) -> None:
        self._require_url()
        checkpoint = (checkpoint_name or "").strip()
        if not checkpoint:
            return
        try:
            response = requests.post(
                f"{self.base_url}/sdapi/v1/options",
                json={"sd_model_checkpoint": checkpoint},
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SDAPIError(f"Не удалось установить SD checkpoint: {checkpoint}") from exc

    def txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        cfg: float = 6.5,
        sampler: str = "DPM++ 2M Karras",
        seed: int = -1,
    ) -> bytes:
        self._require_url()
        if not (prompt or "").strip():
            raise SDAPIError("Промпт для генерации изображения пуст.")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "cfg_scale": float(cfg),
            "sampler_name": sampler,
            "seed": int(seed),
        }
        try:
            response = requests.post(f"{self.base_url}/sdapi/v1/txt2img", json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise SDAPIError("Stable Diffusion API недоступен. Проверьте URL и запуск WebUI с --api.") from exc
        except json.JSONDecodeError as exc:
            raise SDAPIError("Stable Diffusion API вернул некорректный JSON.") from exc

        images = data.get("images") or []
        if not images:
            raise SDAPIError("Stable Diffusion API не вернул изображение.")

        image_b64 = str(images[0])
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        try:
            return base64.b64decode(image_b64)
        except (binascii.Error, ValueError) as exc:
            raise SDAPIError("Не удалось декодировать изображение из ответа SD API.") from exc

    def save_image_bytes(self, png_bytes: bytes) -> str:
        if not png_bytes:
            raise SDAPIError("Пустые данные изображения для сохранения.")
        out_dir = r"C:\X-FLASH\media\generated"
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        path = os.path.join(out_dir, f"sd_{ts}_{int(time.time_ns() % 1_000_000)}.png")
        try:
            with open(path, "wb") as handle:
                handle.write(png_bytes)
        except OSError as exc:
            raise SDAPIError("Не удалось сохранить изображение на диск.") from exc
        return path
