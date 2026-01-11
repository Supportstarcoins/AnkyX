import time
from uuid import uuid4


class SyncClient:
    def __init__(self, api_base_url: str, timeout: int) -> None:
        self.api_base_url = api_base_url
        self.timeout = timeout

    def login(self, email: str, password: str) -> str | None:
        """Mock login. TODO: заменить на requests.post к API."""
        time.sleep(0.5)
        if email and "@" in email and password and len(password) >= 4:
            return f"mock-token-{uuid4().hex[:8]}"
        return None

    def push_deck(self, token: str | None, deck_payload: dict) -> bool:
        """Mock push. TODO: заменить на requests.post к API."""
        time.sleep(0.5)
        if token and token.startswith("mock-token"):
            return True
        return False
