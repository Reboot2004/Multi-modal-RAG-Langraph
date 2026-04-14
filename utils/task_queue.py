from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

from config.settings import ENABLE_ASYNC_INGESTION_WORKERS, INGESTION_WORKER_MAX_THREADS


class TaskQueue:
    def __init__(self):
        self.enabled = bool(ENABLE_ASYNC_INGESTION_WORKERS)
        self.max_workers = max(1, int(INGESTION_WORKER_MAX_THREADS))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers) if self.enabled else None

    def run(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        if not self.enabled or self._executor is None:
            return fn(*args, **kwargs)
        future = self._executor.submit(fn, *args, **kwargs)
        return future.result()

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
