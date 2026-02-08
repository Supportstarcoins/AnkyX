from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import requests


class SDProviderError(Exception):
    """Raised when Stable Diffusion API interaction fails."""


class SD15Provider:
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url.rstrip("/")

    def txt2img(self, prompt: str, out_dir: str, **opts: Any) -> str:
        if not self.api_url:
            raise SDProviderError("Stable Diffusion API URL не задан.")
        if not prompt:
            raise SDProviderError("Промпт пуст.")

        payload = {
            "prompt": prompt,
            "negative_prompt": opts.get(
                "negative_prompt",
                "lowres, blurry, bad anatomy, extra fingers, text, watermark",
            ),
            "width": int(opts.get("width", 512)),
            "height": int(opts.get("height", 512)),
            "steps": int(opts.get("steps", 20)),
            "sampler_name": opts.get("sampler_name", "Euler a"),
            "cfg_scale": float(opts.get("cfg_scale", 7)),
            "seed": int(opts.get("seed", -1)),
        }
        timeout = float(opts.get("timeout", 60))
        url = f"{self.api_url}/sdapi/v1/txt2img"

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise SDProviderError("Stable Diffusion API не доступно.") from exc
        except json.JSONDecodeError as exc:
            raise SDProviderError("Некорректный ответ SD API.") from exc

        images = data.get("images") or []
        if not images:
            raise SDProviderError("SD API не вернул изображение.")

        info_raw = data.get("info")
        seed = None
        if isinstance(info_raw, str):
            try:
                info = json.loads(info_raw)
                seed = info.get("seed")
            except json.JSONDecodeError:
                seed = None
        elif isinstance(info_raw, dict):
            seed = info_raw.get("seed")
        if seed in (None, "", -1):
            seed = payload["seed"] if payload["seed"] not in (-1, None) else "na"

        os.makedirs(out_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f"sd15_{timestamp}_{seed}.png"
        output_path = os.path.join(out_dir, filename)

        image_b64 = images[0]
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        with open(output_path, "wb") as file:
            file.write(image_bytes)

        return output_path
