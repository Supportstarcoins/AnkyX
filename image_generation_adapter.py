from __future__ import annotations

import logging
from typing import Any


class StableDiffusionAdapter:
    def __init__(self, app: Any = None) -> None:
        self.app = app

    def generate_image(self, image_prompt: str, negative_prompt: str | None = None) -> tuple[str | None, str]:
        if not (image_prompt or "").strip():
            return None, "Промт изображения пустой"

        provider = getattr(self.app, "sdxl_provider", None) if self.app is not None else None
        if provider is None:
            try:
                from sdxl_provider import SDXLProvider

                provider = SDXLProvider()
            except Exception:
                return None, "Stable Diffusion недоступен"

        try:
            if hasattr(provider, "generate"):
                path = provider.generate(image_prompt)
            elif hasattr(provider, "generate_image"):
                path = provider.generate_image(image_prompt, negative_prompt=negative_prompt)
            else:
                return None, "Stable Diffusion провайдер не поддерживает generate/generate_image"
            return path, ("Изображение сгенерировано" if path else "Генерация завершена без файла")
        except Exception as exc:
            logging.exception("Image generation failed")
            return None, f"Stable Diffusion недоступен: {exc}"


def generate_card_image(card: dict, app: Any = None) -> tuple[str | None, str]:
    adapter = StableDiffusionAdapter(app=app)
    return adapter.generate_image((card or {}).get("image_prompt", ""), (card or {}).get("negative_prompt"))
