from __future__ import annotations

import logging
import os
from typing import Any

from flux_image_adapter import FluxImageAdapter


class StableDiffusionAdapter:
    def __init__(self, app: Any = None, api_url: str | None = None, checkpoint: str | None = None) -> None:
        self.app = app
        self.api_url = api_url
        self.checkpoint = checkpoint

    def generate_image(self, image_prompt: str, negative_prompt: str | None = None) -> tuple[str | None, str]:
        if not (image_prompt or "").strip():
            return None, "Промт изображения пустой"
        try:
            from sdxl_provider import SDXLProvider

            provider = SDXLProvider(self.api_url) if self.api_url else SDXLProvider()
            if self.checkpoint:
                try:
                    provider.ensure_model(self.checkpoint)
                except Exception:
                    pass
            path = provider.generate_image(image_prompt, negative_prompt=negative_prompt or "")
            return path, ("Изображение сгенерировано" if path else "Генерация завершена без файла")
        except Exception as exc:
            logging.exception("Image generation failed")
            return None, f"Legacy SD недоступен: {exc}"


class ImageGenerationAdapter:
    def __init__(self, settings: dict | None = None, app: Any = None) -> None:
        s = dict(settings or {})
        self.app = app
        self.image_provider = os.getenv("XFLASH_IMAGE_PROVIDER", s.get("image_provider", "flux_comfyui"))
        self.flux_model_path = os.getenv("XFLASH_FLUX_MODEL_PATH", s.get("flux_model_path", "models/flux/flux-2-klein-4b-fp8.safetensors"))
        self.flux_api_url = os.getenv("XFLASH_FLUX_API_URL", s.get("flux_api_url", "http://127.0.0.1:8188"))
        self.flux_output_dir = os.getenv("XFLASH_FLUX_OUTPUT_DIR", s.get("flux_output_dir", "media/generated"))
        self.legacy_sd_api_url = s.get("legacy_sd_api_url", s.get("sd_api_url", "http://127.0.0.1:7860"))
        self.legacy_sd_checkpoint = s.get("legacy_sd_checkpoint", s.get("sd_checkpoint", "sd_xl_base_1.0.safetensors"))
        self.enable_legacy_fallback = bool(s.get("enable_legacy_sd_fallback", s.get("enable_legacy_fallback", True)))

    def generate_card_image(self, card: dict, options: dict | None = None) -> dict:
        c = dict(card or {})
        if c.get("front_image_path") and not (options or {}).get("force_generate"):
            c["image_status"] = "source image kept"
            return c
        prompt = (c.get("image_prompt") or c.get("back") or c.get("front") or "").strip()
        negative = c.get("negative_prompt") or "blurry, low quality, chaotic composition, watermark, logo, unreadable text, random letters, distorted anatomy, extra limbs, messy background, duplicated subjects"
        result = {"ok": False, "error": ""}
        if self.image_provider == "flux_comfyui":
            result = FluxImageAdapter(self.flux_api_url, self.flux_model_path, self.flux_output_dir).generate_image(prompt, negative, options)
        elif self.image_provider == "flux_diffusers":
            result = {"ok": False, "provider": "flux_diffusers", "error": "Для локального flux_diffusers установите torch/diffusers, либо используйте ComfyUI provider."}
        else:
            path, status = StableDiffusionAdapter(app=self.app, api_url=self.legacy_sd_api_url, checkpoint=self.legacy_sd_checkpoint).generate_image(prompt, negative)
            result = {"ok": bool(path), "image_path": path or "", "provider": "legacy_sd_auto1111", "error": "" if path else status}

        if not result.get("ok") and self.image_provider != "legacy_sd_auto1111" and self.enable_legacy_fallback:
            path, status = StableDiffusionAdapter(app=self.app, api_url=self.legacy_sd_api_url, checkpoint=self.legacy_sd_checkpoint).generate_image(prompt, negative)
            if path:
                c["metadata"] = dict(c.get("metadata") or {})
                c["metadata"]["image_status"] = f"FLUX failed, used legacy SD fallback: {result.get('error') or 'unknown error'}"
                result = {"ok": True, "image_path": path, "provider": "legacy_sd_auto1111", "error": ""}
            else:
                result["error"] = result.get("error") or status
        elif not result.get("ok") and not self.enable_legacy_fallback:
            result["error"] = result.get("error") or "FLUX error: fallback в legacy SD отключен"

        c["image_prompt"] = prompt
        c["image_provider"] = result.get("provider") or self.image_provider
        if result.get("ok") and result.get("image_path"):
            c["image_path"] = result["image_path"]
            c["answer_image_path"] = result["image_path"]
            c["front_image_path"] = result["image_path"]
            c["front_image_origin"] = "generated_flux" if str(c["image_provider"]).startswith("flux") else "generated"
            c["image_status"] = "generated"
        else:
            c["image_status"] = result.get("error") or "generation failed"
        meta = dict(c.get("metadata") or {})
        meta["image_status"] = c["image_status"]
        meta["image_provider"] = c["image_provider"]
        meta["flux_model_path"] = self.flux_model_path
        c["metadata"] = meta
        return c


def generate_card_image(card: dict, app: Any = None) -> tuple[str | None, str]:
    updated = ImageGenerationAdapter(app=app).generate_card_image(card)
    return updated.get("front_image_path"), updated.get("image_status") or ""
