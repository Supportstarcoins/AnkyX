import threading
import time
from collections import deque
from typing import Deque, Dict

from config import settings


class RateLimiter:
    def __init__(self, limit_per_min: int) -> None:
        self.limit = limit_per_min
        self.window_s = 60
        self.lock = threading.Lock()
        self.buckets: Dict[str, Deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            bucket = self.buckets.setdefault(key, deque())
            while bucket and now - bucket[0] > self.window_s:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


class ConcurrencyLimiter:
    def __init__(self, max_global: int, max_per_user: int) -> None:
        self.global_sem = threading.BoundedSemaphore(max_global)
        self.max_per_user = max_per_user
        self.user_lock = threading.Lock()
        self.user_sems: Dict[int, threading.BoundedSemaphore] = {}

    def _get_user_sem(self, user_id: int) -> threading.BoundedSemaphore:
        with self.user_lock:
            sem = self.user_sems.get(user_id)
            if sem is None:
                sem = threading.BoundedSemaphore(self.max_per_user)
                self.user_sems[user_id] = sem
            return sem

    def try_acquire(self, user_id: int) -> bool:
        user_sem = self._get_user_sem(user_id)
        if not self.global_sem.acquire(blocking=False):
            return False
        if not user_sem.acquire(blocking=False):
            self.global_sem.release()
            return False
        return True

    def release(self, user_id: int) -> None:
        user_sem = self._get_user_sem(user_id)
        try:
            user_sem.release()
        finally:
            self.global_sem.release()


rate_limiter = RateLimiter(settings.rate_limit_per_min)
concurrency_limiter = ConcurrencyLimiter(
    settings.max_concurrency_global, settings.max_concurrency_per_user
)
