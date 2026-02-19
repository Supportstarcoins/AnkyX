from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    pass


SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


@lru_cache(maxsize=16)
def load_schema(schema_name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / schema_name
    if not path.exists():
        raise FileNotFoundError(f"schema not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _simple_validate(schema: dict[str, Any], payload: Any) -> None:
    if not isinstance(payload, dict):
        raise SchemaValidationError("payload must be object")
    if "cards" not in payload or not isinstance(payload["cards"], list) or not payload["cards"]:
        raise SchemaValidationError("cards must be non-empty list")


def validate_schema(schema_name: str, payload: Any) -> None:
    schema = load_schema(schema_name)
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(payload, schema)
    except ModuleNotFoundError:
        _simple_validate(schema, payload)
    except Exception as exc:  # noqa: BLE001
        raise SchemaValidationError(str(exc)) from exc
