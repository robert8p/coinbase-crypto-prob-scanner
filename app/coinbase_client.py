import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

from .rate_limiter import AsyncTokenBucket, RateLimitState, jitter_sleep

@dataclass
class CoinbaseResult:
    ok: bool
    message: str
    last_request_utc: str | None = None
    last_error: str | None = None

class CoinbaseClient:
    def __init__(self, base_url: str, max_rps: float, max_inflight: int, demo_mode: bool = False):
        self.base_url = base_url.rstrip("/")
        self.demo_mode = demo_mode
        self.bucket = AsyncTokenBucket(rate=max_rps, capacity=max(2.0, max_rps * 2.0))
        self.sema = asyncio.Semaphore(max(1, int(max_inflight)))
        self.state = RateLimitState()
        self._client: httpx.AsyncClient | None = None
        self._last_request_utc: str | None = None
        self._last_error: str | None = None

    async def __aenter__(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def status(self) -> CoinbaseResult:
        msg = "DEMO_MODE" if self.demo_mode else ("OK" if self._last_error is None else "WARN")
        return CoinbaseResult(
            ok=(not self.demo_mode) and (self._last_error is None),
            message=msg,
            last_request_utc=self._last_request_utc,
            last_error=self._last_error,
        )

    async def _request(self, method: str, path: str, params: dict | None = None) -> Any:
        if self.demo_mode:
            self._last_request_utc = dt.datetime.now(dt.timezone.utc).isoformat()
            return None

        assert self._client is not None
        url = f"{self.base_url}{path}"
        headers = {"User-Agent": "coinbase-crypto-prob-scanner/1.0"}

        retries = 5
        backoff = 1.0
        for _ in range(retries):
            await self.bucket.acquire(1.0)
            async with self.sema:
                self._last_request_utc = dt.datetime.now(dt.timezone.utc).isoformat()
                try:
                    resp = await self._client.request(method, url, params=params, headers=headers)
                except Exception:
                    self._last_error = "request_error"
                    self.state.recent_errors += 1
                    sleep_s = jitter_sleep(backoff)
                    self.state.last_backoff_seconds = sleep_s
                    self.state.backoff_count += 1
                    await asyncio.sleep(sleep_s)
                    backoff = min(30.0, backoff*2.0)
                    continue

            if resp.status_code == 200:
                self._last_error = None
                self.state.recent_errors = 0
                return resp.json()

            if resp.status_code in (429, 500, 502, 503, 504):
                self._last_error = f"http_{resp.status_code}"
                if resp.status_code == 429:
                    self.state.last_429_utc = dt.datetime.now(dt.timezone.utc).isoformat()
                sleep_s = jitter_sleep(backoff)
                self.state.last_backoff_seconds = sleep_s
                self.state.backoff_count += 1
                await asyncio.sleep(sleep_s)
                backoff = min(30.0, backoff*2.0)
                continue

            self._last_error = f"http_{resp.status_code}"
            raise RuntimeError(self._last_error)

        raise RuntimeError(self._last_error or "coinbase_request_failed")

    async def list_products(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/products")
        return data if isinstance(data, list) else []

    async def get_candles(self, product_id: str, granularity_sec: int, start: dt.datetime, end: dt.datetime):
        params = {
            "granularity": int(granularity_sec),
            "start": start.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z"),
            "end": end.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z"),
        }
        data = await self._request("GET", f"/products/{product_id}/candles", params=params)
        return data if isinstance(data, list) else []
