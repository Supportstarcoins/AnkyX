import threading
import queue
from typing import Any, Callable


class BackgroundTask:
    def __init__(self, target: Callable[..., Any], *args, **kwargs):
        self.queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_wrapper, args=(target, args, kwargs), daemon=True
        )

    def _run_wrapper(self, target: Callable[..., Any], args: tuple, kwargs: dict):
        try:
            result = target(self, *args, **kwargs)
            self.queue.put(("done", result))
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("error", str(exc)))

    def start(self):
        self._thread.start()

    def cancel(self):
        self._cancel_event.set()

    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def start_background_task(target: Callable[..., Any], *args, **kwargs) -> BackgroundTask:
    task = BackgroundTask(target, *args, **kwargs)
    task.start()
    return task
