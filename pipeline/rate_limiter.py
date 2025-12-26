"""
Async rate limiter utility.

Provides a RateLimiter that enforces a maximum concurrent "in-flight" count
and sliding-window rate limits (requests per minute and requests per second).
This is intentionally lightweight and designed to be awaited around an
awaitable (e.g. an `asyncio.to_thread(...)` call or any coroutine).
"""
from collections import deque
import time
import asyncio
from typing import Deque


class RateLimiter:
    def __init__(
        self,
        max_concurrency: int = 50,
        max_rpm: int = 600,
        max_rps: int = 8,
        window_seconds: float = 60.0,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rpm_lock = asyncio.Lock()
        self._request_times_60s: Deque[float] = deque()
        self._request_times_1s: Deque[float] = deque()
        self.max_rpm = max_rpm
        self.max_rps = max_rps
        self.window_seconds = window_seconds

    async def _acquire_rate_window(self) -> None:
        """Wait until both 60s-window and 1s-window allow another request."""
        while True:
            now = time.time()
            # drop old timestamps
            while self._request_times_60s and now - self._request_times_60s[0] > self.window_seconds:
                self._request_times_60s.popleft()
            while self._request_times_1s and now - self._request_times_1s[0] > 1.0:
                self._request_times_1s.popleft()

            if len(self._request_times_60s) < self.max_rpm and len(self._request_times_1s) < self.max_rps:
                self._request_times_60s.append(now)
                self._request_times_1s.append(now)
                return

            # slightly back off and retry
            await asyncio.sleep(0.05)

    async def run(self, awaitable):
        """
        Run the given awaitable under concurrency + rate limits.
        `awaitable` must be an awaitable object (coroutine or Task).
        """
        async with self._semaphore:
            async with self._rpm_lock:
                await self._acquire_rate_window()
            # after acquiring windows, actually await the provided awaitable
            return await awaitable


# default/global rate limiter instance used by pipeline modules
default_rate_limiter = RateLimiter()


