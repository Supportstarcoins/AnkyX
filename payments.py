import hmac
import os
import time
from hashlib import sha256
from typing import Dict, Tuple
from urllib.parse import urlencode

PACKAGES: Dict[str, Dict] = {
    "pack_500": {"credits": 500, "price": 4.99, "currency": "USD", "title": "500 кредитов"},
    "pack_1000": {"credits": 1000, "price": 7.99, "currency": "USD", "title": "1000 кредитов"},
    "pack_2000": {"credits": 2000, "price": 14.99, "currency": "USD", "title": "2000 кредитов"},
}


def _sign_payload(payload: Dict[str, str], secret: str) -> str:
    serialized = "&".join(f"{k}={payload[k]}" for k in sorted(payload))
    digest = hmac.new(secret.encode("utf-8"), serialized.encode("utf-8"), sha256).hexdigest()
    return digest


def build_payment_url(user_id: str, package_id: str) -> str:
    if package_id not in PACKAGES:
        raise ValueError(f"Unknown package_id: {package_id}")
    package = PACKAGES[package_id]
    timestamp = int(time.time())
    payload = {
        "user_id": user_id,
        "package_id": package_id,
        "credits": str(package["credits"]),
        "price": str(package["price"]),
        "currency": package["currency"],
        "ts": str(timestamp),
    }
    secret = os.getenv("XFLASH_PAY_SECRET", "")
    if secret:
        payload["signature"] = _sign_payload(payload, secret)
    else:
        payload["signature"] = "demo"
    base_url = os.getenv("XFLASH_PAY_URL", "https://pay.x-flash.app/checkout")
    return f"{base_url}?{urlencode(payload)}"


def verify_payment(user_id: str, package_id: str, payment_token_or_id: str) -> Tuple[bool, Dict]:
    """Заглушка для проверки платежей. Можно заменить реальным API."""
    if not payment_token_or_id:
        return False, {"error": "payment_id_missing"}
    details = {
        "user_id": user_id,
        "package_id": package_id,
        "payment_id": payment_token_or_id,
        "status": "confirmed",
        "mode": "manual_stub",
    }
    return True, details
