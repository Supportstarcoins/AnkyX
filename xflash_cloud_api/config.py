import os
from dataclasses import dataclass
from typing import List

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _resolve_default_model() -> str:
    env_model = (os.getenv("XFLASH_DEFAULT_MODEL") or "").strip()
    if env_model:
        return env_model
    app_model = (os.getenv("XFLASH_OLLAMA_MODEL") or "").strip()
    if app_model:
        return app_model
    return "llama3.1:8b"


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("XFLASH_DB_PATH", "./data/xflash.db")
    bind_host: str = os.getenv("XFLASH_BIND_HOST", "127.0.0.1")
    bind_port: int = int(os.getenv("XFLASH_BIND_PORT", "8000"))
    ollama_url: str = os.getenv("XFLASH_OLLAMA_URL", "http://127.0.0.1:11434")
    default_model: str = _resolve_default_model()
    max_concurrency_global: int = int(os.getenv("XFLASH_MAX_CONCURRENCY_GLOBAL", "2"))
    max_concurrency_per_user: int = int(os.getenv("XFLASH_MAX_CONCURRENCY_PER_USER", "1"))
    rate_limit_per_min: int = int(os.getenv("XFLASH_RATE_LIMIT_PER_MIN", "10"))
    max_payload_bytes: int = int(os.getenv("XFLASH_MAX_PAYLOAD_BYTES", str(50 * 1024)))
    max_meta_items: int = int(os.getenv("XFLASH_MAX_META_ITEMS", "10"))
    max_tokens: int = int(os.getenv("XFLASH_MAX_TOKENS", "700"))
    ollama_timeout_s: int = int(os.getenv("XFLASH_OLLAMA_TIMEOUT_S", "180"))
    cors_origins: List[str] = [
        origin.strip()
        for origin in os.getenv("XFLASH_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]


settings = Settings()
