from __future__ import annotations

import logging
from typing import Any


def generate_card_image(card: dict, app: Any = None) -> tuple[str | None, str]:
    prompt = (card or {}).get("image_prompt", "")
    if not prompt:
        return None, "Промт изображения пустой"

    provider = None
    if app is not None:
        provider = getattr(app, "sdxl_provider", None)

    if provider is None:
        try:
            from sdxl_provider import SDXLProvider

            provider = SDXLProvider()
        except Exception:
            return None, "Stable Diffusion недоступен"

    try:
        if hasattr(provider, "generate"):
            path = provider.generate(prompt)
        elif hasattr(provider, "generate_image"):
            path = provider.generate_image(prompt)
        else:
            return None, "Stable Diffusion провайдер не поддерживает generate/generate_image"
        return path, "Изображение сгенерировано" if path else "Генерация завершена без файла"
    except Exception as exc:
        logging.exception("Image generation failed")
        return None, f"Stable Diffusion недоступен: {exc}"
