from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import AppConfig
from .demo_data import demo_candles, demo_currencies, demo_products, demo_stats_map

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DataSourceHealth:
    ok: bool
    message: str
    base_url: str
    last_request_utc: str | None
    last_bar_timestamp: str | None
    pagination_warnings: List[str]
    rate_limit_warn: str | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "message": self.message,
            "base_url": self.base_url,
            "last_request_utc": self.last_request_utc,
            "last_bar_timestamp": self.last_bar_timestamp,
            "pagination_warnings": list(self.pagination_warnings),
            "rate_limit_warn": self.rate_limit_warn,
        }


class CoinbaseClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self._state_lock = threading.Lock()
        self.last_request_utc: str | None = None
        self.last_bar_timestamp: str | None = None
        self.pagination_warnings: List[str] = []
        self.rate_limit_warn: str | None = None
        self.session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.4,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8))
        self.session.headers.update({"User-Agent": "coinbase-crypto-prob-scanner/4.6.3", "Accept": "application/json"})

    def health(self) -> DataSourceHealth:
        if self.config.demo_mode:
            return DataSourceHealth(
                ok=True,
                message="demo_mode",
                base_url=self.config.coinbase_exchange_base_url,
                last_request_utc=self.last_request_utc,
                last_bar_timestamp=self.last_bar_timestamp,
                pagination_warnings=list(self.pagination_warnings),
                rate_limit_warn=self.rate_limit_warn,
            )
        try:
            self._get_json("/time")
            return DataSourceHealth(
                ok=True,
                message="ok",
                base_url=self.config.coinbase_exchange_base_url,
                last_request_utc=self.last_request_utc,
                last_bar_timestamp=self.last_bar_timestamp,
                pagination_warnings=list(self.pagination_warnings),
                rate_limit_warn=self.rate_limit_warn,
            )
        except Exception as exc:
            return DataSourceHealth(
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
                base_url=self.config.coinbase_exchange_base_url,
                last_request_utc=self.last_request_utc,
                last_bar_timestamp=self.last_bar_timestamp,
                pagination_warnings=list(self.pagination_warnings),
                rate_limit_warn=self.rate_limit_warn,
            )

    def list_products(self) -> List[dict]:
        if self.config.demo_mode:
            return demo_products()
        products = self._get_json("/products")
        tradability = self._get_advanced_tradability_status()
        if tradability:
            merged = []
            for item in products:
                pid = item.get("id") or item.get("product_id")
                extra = tradability.get(pid, {})
                if extra:
                    merged.append({**item, **extra})
                else:
                    merged.append(item)
            return merged
        return products

    def list_currencies(self) -> List[dict]:
        if self.config.demo_mode:
            return demo_currencies()
        return self._get_json("/currencies")

    def _get_advanced_tradability_status(self) -> Dict[str, dict]:
        if self.config.demo_mode:
            return {}
        params = {
            "product_type": "SPOT",
            "get_tradability_status": "true",
            "limit": 250,
        }
        out: Dict[str, dict] = {}
        base_candidates = [self.config.coinbase_advanced_base_url.rstrip("/")]
        if base_candidates[0].endswith("/market"):
            base_candidates.append(base_candidates[0][:-7])
        last_exc: Exception | None = None
        for base in [b for b in base_candidates if b]:
            out.clear()
            cursor = None
            page = 0
            while True:
                page += 1
                req_params = dict(params)
                if cursor:
                    req_params["cursor"] = cursor
                try:
                    payload = self._get_json_advanced(base, "/products", params=req_params)
                except Exception as exc:
                    last_exc = exc
                    out.clear()
                    break
                products = payload.get("products") if isinstance(payload, dict) else None
                if not isinstance(products, list):
                    out.clear()
                    break
                for item in products:
                    pid = item.get("product_id") or item.get("id")
                    if not pid:
                        continue
                    out[pid] = {"view_only": bool(item.get("view_only", False))}
                pagination = payload.get("pagination") if isinstance(payload, dict) else None
                has_next = bool((pagination or {}).get("has_next"))
                cursor = (pagination or {}).get("next_cursor")
                if not has_next or not cursor:
                    return out
                if page > 20:
                    self._append_warning("advanced_tradability_pagination_limit")
                    return out
                time.sleep(self.config.request_pause_seconds)
        if last_exc is not None:
            self._append_warning(f"advanced_tradability_unavailable={type(last_exc).__name__}")
        return {}

    def _get_json_advanced(self, base_url: str, path: str, params: Optional[dict] = None):
        url = base_url.rstrip("/") + path
        headers = {"cache-control": "no-cache"}
        resp = self.session.get(url, params=params, timeout=self.config.http_timeout_seconds, headers=headers)
        with self._state_lock:
            self.last_request_utc = datetime.now(timezone.utc).isoformat()
            if resp.status_code == 429:
                self.rate_limit_warn = f"HTTP 429 on advanced {path}"
        resp.raise_for_status()
        return resp.json()

    def get_volume_summary(self) -> Dict[str, dict]:
        if self.config.demo_mode:
            return demo_stats_map([p["id"] for p in demo_products()])
        try:
            payload = self._get_json("/products/volume-summary")
            flattened: Dict[str, dict] = {}
            for group in payload:
                if isinstance(group, list):
                    for item in group:
                        pid = item.get("id") or item.get("product_id")
                        if pid:
                            flattened[pid] = item
                elif isinstance(group, dict):
                    pid = group.get("id") or group.get("product_id")
                    if pid:
                        flattened[pid] = group
            return flattened
        except Exception as exc:
            self._append_warning(f"volume_summary_unavailable={exc}")
            return {}

    def _align_5m_floor(self, value: datetime | str) -> datetime:
        ts = pd.to_datetime(value, utc=True).to_pydatetime().replace(second=0, microsecond=0)
        floored_minute = ts.minute - (ts.minute % 5)
        return ts.replace(minute=floored_minute)

    def get_candles(self, product_id: str, lookback_bars: int) -> pd.DataFrame:
        return self.get_candles_until(product_id, lookback_bars, datetime.now(timezone.utc))

    def get_candles_until(self, product_id: str, lookback_bars: int, end_time: datetime | str) -> pd.DataFrame:
        if self.config.demo_mode:
            if product_id in self.config.demo_fail_symbols:
                raise RuntimeError(f"demo failure injected for {product_id}")
            df = demo_candles(product_id, lookback_bars=max(lookback_bars, 600))
            df = self._regularize_candles(df.tail(lookback_bars).copy(), lookback_bars=lookback_bars)
            self._update_last_bar(df["ts"].iloc[-1] if not df.empty else None)
            return df

        end = self._align_5m_floor(end_time)
        start = end - timedelta(minutes=5 * max(int(lookback_bars) - 1, 0))
        df = self.get_candles_range(product_id, start, end)
        df = self._regularize_candles(df, lookback_bars=max(int(lookback_bars), 1), end_ts=end)
        if not df.empty:
            self._update_last_bar(df["ts"].iloc[-1])
        return df

    def get_candles_range(self, product_id: str, start_time: datetime | str, end_time: datetime | str) -> pd.DataFrame:
        start = self._align_5m_floor(start_time)
        end = self._align_5m_floor(end_time)
        if start > end:
            raise ValueError("start_time must be <= end_time")
        if self.config.demo_mode:
            lookback_bars = max(1, int(((end - start).total_seconds() // 300) + 1))
            raw = demo_candles(product_id, lookback_bars=max(lookback_bars + 10, 600)).copy()
            if not raw.empty:
                raw = raw.sort_values("ts").reset_index(drop=True)
                full_index = pd.date_range(end=end, periods=len(raw), freq="5min", tz="UTC")
                raw["ts"] = full_index
                raw = raw[(pd.to_datetime(raw["ts"], utc=True) >= pd.Timestamp(start)) & (pd.to_datetime(raw["ts"], utc=True) <= pd.Timestamp(end))].copy()
            return self._regularize_candles(raw, lookback_bars=lookback_bars, end_ts=end)

        granularity = 300
        current_end = end
        chunks: List[pd.DataFrame] = []
        while current_end >= start:
            chunk_bars = min(300, max(1, int(((current_end - start).total_seconds() // 300) + 1)))
            chunk_start = max(start, current_end - timedelta(minutes=5 * max(chunk_bars - 1, 0)))
            params = {
                "start": chunk_start.isoformat(),
                "end": current_end.isoformat(),
                "granularity": granularity,
            }
            candles = self._get_json(f"/products/{product_id}/candles", params=params)
            if not isinstance(candles, list):
                raise RuntimeError(f"unexpected candle payload for {product_id}")
            if len(candles) >= 300 and chunk_start > start:
                self._append_warning(f"chunked_candles={product_id}")
            frame = self._candles_to_frame(candles)
            chunks.append(frame)
            if chunk_start <= start:
                break
            current_end = chunk_start - timedelta(minutes=5)
            time.sleep(self.config.request_pause_seconds)
        raw = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["ts"]).sort_values("ts") if chunks else pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        lookback_bars = max(1, int(((end - start).total_seconds() // 300) + 1))
        df = self._regularize_candles(raw, lookback_bars=lookback_bars, end_ts=end)
        df = df[(pd.to_datetime(df["ts"], utc=True) >= pd.Timestamp(start)) & (pd.to_datetime(df["ts"], utc=True) <= pd.Timestamp(end))].copy().reset_index(drop=True)
        if not df.empty:
            self._update_last_bar(df["ts"].iloc[-1])
        return df

    def _candles_to_frame(self, rows: list) -> pd.DataFrame:
        values = []
        for item in rows:
            if len(item) < 6:
                continue
            values.append(
                {
                    "ts": pd.to_datetime(item[0], unit="s", utc=True),
                    "low": float(item[1]),
                    "high": float(item[2]),
                    "open": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        return pd.DataFrame(values)

    def _regularize_candles(self, df: pd.DataFrame, lookback_bars: int, end_ts=None) -> pd.DataFrame:
        if df is None or df.empty:
            empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            empty.attrs["observed_bars"] = 0
            empty.attrs["filled_bars"] = 0
            empty.attrs["raw_bars"] = 0
            return empty

        df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
        observed_ts = set(pd.to_datetime(df["ts"], utc=True))
        end_ts = pd.to_datetime(end_ts, utc=True) if end_ts is not None else pd.to_datetime(df["ts"].iloc[-1], utc=True)
        start_ts = end_ts - pd.Timedelta(minutes=5 * (max(int(lookback_bars), 1) - 1))
        full_index = pd.date_range(start=start_ts, end=end_ts, freq="5min", tz="UTC")

        work = df.set_index(pd.to_datetime(df["ts"], utc=True))[["open", "high", "low", "close", "volume"]].sort_index()
        work = work[~work.index.duplicated(keep="last")]
        work = work.reindex(full_index)

        first_valid = work["close"].first_valid_index()
        if first_valid is None:
            empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            empty.attrs["observed_bars"] = 0
            empty.attrs["filled_bars"] = 0
            empty.attrs["raw_bars"] = 0
            return empty

        work = work.loc[first_valid:].copy()
        for col in ["open", "high", "low", "close"]:
            work[col] = work[col].ffill()
        work["volume"] = work["volume"].fillna(0.0)
        # if an interval had no trade, freeze OHLC at the last close to produce a regular 5m bar
        frozen = work["open"].isna() | work["high"].isna() | work["low"].isna() | work["close"].isna()
        if frozen.any():
            prior_close = work["close"].ffill()
            work.loc[frozen, "open"] = prior_close.loc[frozen]
            work.loc[frozen, "high"] = prior_close.loc[frozen]
            work.loc[frozen, "low"] = prior_close.loc[frozen]
            work.loc[frozen, "close"] = prior_close.loc[frozen]

        work = work.reset_index().rename(columns={"index": "ts"})
        observed_in_window = sum(1 for ts in work["ts"] if ts in observed_ts)
        work.attrs["observed_bars"] = int(observed_in_window)
        work.attrs["filled_bars"] = int(len(work) - observed_in_window)
        work.attrs["raw_bars"] = int(len(df))
        return work.tail(lookback_bars).reset_index(drop=True)

    def _get_json(self, path: str, params: Optional[dict] = None):
        url = self.config.coinbase_exchange_base_url.rstrip("/") + path
        headers = {"cache-control": "no-cache"}
        resp = self.session.get(url, params=params, timeout=self.config.http_timeout_seconds, headers=headers)
        with self._state_lock:
            self.last_request_utc = datetime.now(timezone.utc).isoformat()
            if resp.status_code == 429:
                self.rate_limit_warn = f"HTTP 429 on {path}"
        resp.raise_for_status()
        return resp.json()

    def _append_warning(self, warning: str) -> None:
        with self._state_lock:
            if warning not in self.pagination_warnings:
                self.pagination_warnings.append(warning)

    def _update_last_bar(self, ts) -> None:
        if ts is None:
            return
        ts_iso = pd.to_datetime(ts, utc=True).isoformat()
        with self._state_lock:
            if self.last_bar_timestamp is None or ts_iso > self.last_bar_timestamp:
                self.last_bar_timestamp = ts_iso
