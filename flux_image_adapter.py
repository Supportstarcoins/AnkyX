from __future__ import annotations

import json
import os
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests


class FluxImageAdapter:
    def __init__(self, api_url: str = "http://127.0.0.1:8188", model_path: str = "", output_dir: str = "media/generated"):
        self.api_url = self._normalize_api_url(api_url or "http://127.0.0.1:8188")
        self.model_path = model_path or "models/flux/flux-2-klein-4b-fp8.safetensors"
        self.output_dir = output_dir or "media/generated"

    @staticmethod
    def _normalize_api_url(api_url: str) -> str:
        parsed = urlparse(api_url if "://" in api_url else f"http://{api_url}")
        path = (parsed.path or "").rstrip("/")
        for suffix in ("/api", "/prompt", "/system_stats", "/view"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
        return urlunparse((parsed.scheme or "http", parsed.netloc, path.rstrip("/"), "", "", "")).rstrip("/")

    def is_available(self) -> tuple[bool, str]:
        model_file = Path(self.model_path)
        if not model_file.exists():
            return False, f"FLUX model file not found: {self.model_path}"
        try:
            r = requests.get(f"{self.api_url}/system_stats", timeout=8)
            if r.ok:
                return True, "FLUX: OK"
        except requests.RequestException:
            pass
        return False, f"FLUX/ComfyUI недоступен: проверьте {self.api_url}"

    def _workflow(self, prompt: str, negative_prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        seed = int(options.get("seed", int(time.time()) % 2_147_483_647))
        width = int(options.get("width", 768))
        height = int(options.get("height", 768))
        steps = int(options.get("steps", 24))
        cfg = float(options.get("cfg", 4.0))
        ckpt_name = options.get("comfy_model_name") or os.path.basename(self.model_path)
        return {
            "3": {"inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"},
            "4": {"inputs": {"ckpt_name": ckpt_name}, "class_type": "CheckpointLoaderSimple"},
            "5": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "6": {"inputs": {"text": prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "7": {"inputs": {"text": negative_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
            "9": {"inputs": {"filename_prefix": "xflash_flux", "images": ["8", 0]}, "class_type": "SaveImage"},
        }

    def generate_image(self, prompt: str, negative_prompt: str = "", options: dict | None = None) -> dict:
        options = dict(options or {})
        result = {
            "ok": False,
            "image_path": "",
            "provider": "flux_comfyui",
            "model_path": self.model_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "error": "",
        }
        ok, status = self.is_available()
        if not ok:
            result["error"] = status
            return result
        try:
            payload = {"prompt": self._workflow(prompt, negative_prompt, options), "client_id": str(uuid.uuid4())}
            p = requests.post(f"{self.api_url}/prompt", json=payload, timeout=20)
            p.raise_for_status()
            prompt_id = (p.json() or {}).get("prompt_id")
            if not prompt_id:
                raise RuntimeError("ComfyUI did not return prompt_id")
            history = None
            for _ in range(60):
                h = requests.get(f"{self.api_url}/history/{prompt_id}", timeout=10)
                if h.ok:
                    history = h.json() or {}
                    if prompt_id in history:
                        break
                time.sleep(1)
            if not history or prompt_id not in history:
                raise RuntimeError("ComfyUI generation timeout")
            outputs = (((history.get(prompt_id) or {}).get("outputs") or {}))
            images = []
            for node in outputs.values():
                images.extend((node or {}).get("images") or [])
            if not images:
                raise RuntimeError(
                    "FLUX model should be available to ComfyUI. Copy or symlink flux-2-klein-4b-fp8.safetensors to ComfyUI model folder or set correct workflow model name."
                )
            img = images[0]
            params = urllib.parse.urlencode({"filename": img.get("filename", ""), "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")})
            data = requests.get(f"{self.api_url}/view?{params}", timeout=30)
            data.raise_for_status()
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"flux_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
            out_path.write_bytes(data.content)
            result["ok"] = True
            result["image_path"] = str(out_path)
            return result
        except Exception as exc:
            result["error"] = str(exc)
            return result
