import asyncio
import random
import time
from dataclasses import dataclass

@dataclass
class RateLimitState:
    last_429_utc: str | None = None
    last_backoff_seconds: float | None = None
    backoff_count: int = 0
    recent_errors: int = 0

class AsyncTokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = max(0.1, float(rate))
        self.capacity = max(1.0, float(capacity))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        tokens = float(tokens)
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    self.updated = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                need = tokens - self.tokens
                wait_s = need / self.rate
            await asyncio.sleep(max(0.0, wait_s))

def jitter_sleep(base: float, factor: float = 0.25) -> float:
    j = base * factor
    return max(0.0, base + random.uniform(-j, j))
