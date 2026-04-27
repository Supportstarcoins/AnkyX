from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import requests


class SDXLProviderError(Exception):
    pass


class SDXLProvider:
    def __init__(self, api_url: str) -> None:
        self.api_url = (api_url or "").rstrip("/")

    def ensure_model(self, checkpoint_name: str = "sd_xl_base_1.0.safetensors") -> None:
        if not self.api_url:
            raise SDXLProviderError("SD API URL не задан.")
        try:
            response = requests.post(
                f"{self.api_url}/sdapi/v1/options",
                json={"sd_model_checkpoint": checkpoint_name},
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SDXLProviderError("Не удалось переключить SDXL модель через WebUI API.") from exc

    def txt2img(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler: str,
        seed: int | None = None,
        batch_size: int = 1,
        batch_count: int = 1,
        timeout: float = 90,
    ) -> str:
        if not self.api_url:
            raise SDXLProviderError("SD API URL не задан.")
        if not prompt.strip():
            raise SDXLProviderError("Промпт пуст.")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "cfg_scale": float(cfg),
            "sampler_name": sampler,
            "seed": int(seed) if seed is not None else -1,
            "batch_size": max(1, int(batch_size)),
            "n_iter": max(1, int(batch_count)),
        }
        try:
            response = requests.post(f"{self.api_url}/sdapi/v1/txt2img", json=payload, timeout=float(timeout))
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise SDXLProviderError("Stable Diffusion WebUI API не доступно. Запустите WebUI с --api.") from exc
        except json.JSONDecodeError as exc:
            raise SDXLProviderError("Некорректный ответ SD API.") from exc

        images = data.get("images") or []
        if not images:
            raise SDXLProviderError("SD API не вернул изображение.")

        info_raw = data.get("info")
        resolved_seed = seed
        if isinstance(info_raw, str):
            try:
                resolved_seed = json.loads(info_raw).get("seed", resolved_seed)
            except Exception:
                pass

        out_dir = r"C:\X-FLASH\media\generated"
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        seed_label = resolved_seed if resolved_seed not in (None, -1, "") else "na"
        output_path = os.path.join(out_dir, f"sdxl_{ts}_{seed_label}.png")

        image_b64 = images[0]
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        with open(output_path, "wb") as fh:
            fh.write(base64.b64decode(image_b64))
        return output_path
