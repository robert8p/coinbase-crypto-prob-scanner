"""
Lightweight Binance public candle client for cross-exchange features.

Only used for a small set of reference symbols to detect price-discovery
lag between Binance (typically the leading venue) and Coinbase.

No authentication required. Public REST only.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"

# Map Coinbase symbols to Binance symbols
# Coinbase uses "-" separator (BTC-USD), Binance uses no separator (BTCUSDT)
COINBASE_TO_BINANCE = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT",
    "ADA-USD": "ADAUSDT",
    "AVAX-USD": "AVAXUSDT",
    "LINK-USD": "LINKUSDT",
    "DOT-USD": "DOTUSDT",
    "SUI-USD": "SUIUSDT",
    "APT-USD": "APTUSDT",
    "NEAR-USD": "NEARUSDT",
    "ARB-USD": "ARBUSDT",
    "OP-USD": "OPUSDT",
    "FET-USD": "FETUSDT",
    "INJ-USD": "INJUSDT",
    "RENDER-USD": "RENDERUSDT",
    "PEPE-USD": "PEPEUSDT",
    "SHIB-USD": "SHIBUSDT",
    "BONK-USD": "BONKUSDT",
    "FIL-USD": "FILUSDT",
    "ATOM-USD": "ATOMUSDT",
    "UNI-USD": "UNIUSDT",
    "AAVE-USD": "AAVEUSDT",
    "LTC-USD": "LTCUSDT",
    "BCH-USD": "BCHUSDT",
    "HBAR-USD": "HBARUSDT",
    "SEI-USD": "SEIUSDT",
    "TIA-USD": "TIAUSDT",
    "TAO-USD": "TAOUSDT",
    "ONDO-USD": "ONDOUSDT",
    "WIF-USD": "WIFUSDT",
    "COMP-USD": "COMPUSDT",
    "ICP-USD": "ICPUSDT",
}


class BinanceClient:
    """Minimal Binance public REST client for cross-exchange signals."""

    def __init__(self, timeout: float = 8.0, pause: float = 0.05):
        self.timeout = timeout
        self.pause = pause
        self._session = requests.Session()
        retries = Retry(
            total=1,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503],
            allowed_methods=frozenset(["GET"]),
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4))
        self._session.headers.update({"User-Agent": "crypto-prob-scanner/4.1.1"})
        self._available = None  # None = not checked, True/False after first call

    def is_available(self) -> bool:
        """Check if Binance API is reachable. Cached after first call."""
        if self._available is not None:
            return self._available
        try:
            resp = self._session.get(
                f"{BINANCE_BASE_URL}/api/v3/ping",
                timeout=self.timeout,
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        logger.info("binance_availability=%s", self._available)
        return self._available

    def get_recent_candles(self, binance_symbol: str, interval: str = "5m", limit: int = 24) -> pd.DataFrame:
        """Fetch recent candles from Binance public API.

        Returns DataFrame with columns: ts, open, high, low, close, volume
        """
        try:
            resp = self._session.get(
                f"{BINANCE_BASE_URL}/api/v3/klines",
                params={"symbol": binance_symbol, "interval": interval, "limit": limit},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw = resp.json()
            if not raw or not isinstance(raw, list):
                return pd.DataFrame()

            rows = []
            for k in raw:
                rows.append({
                    "ts": pd.to_datetime(k[0], unit="ms", utc=True),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.debug("binance_candle_fetch_failed symbol=%s error=%s", binance_symbol, exc)
            return pd.DataFrame()

    def get_cross_exchange_signals(self, coinbase_symbols: List[str]) -> Dict[str, dict]:
        """Fetch Binance candles for symbols that have a Binance mapping,
        and compute cross-exchange features.

        Returns {coinbase_symbol: {binance_ret_15m, binance_ret_1h, binance_price, price_gap_pct}}
        """
        if not self.is_available():
            return {}

        signals: Dict[str, dict] = {}
        for cb_sym in coinbase_symbols:
            bn_sym = COINBASE_TO_BINANCE.get(cb_sym)
            if not bn_sym:
                continue
            df = self.get_recent_candles(bn_sym, interval="5m", limit=18)
            if df.empty or len(df) < 4:
                continue

            close = df["close"]
            bn_price = float(close.iloc[-1])
            bn_ret_15m = float(close.iloc[-1] / close.iloc[-4] - 1) if len(close) >= 4 else 0.0
            bn_ret_1h = float(close.iloc[-1] / close.iloc[-13] - 1) if len(close) >= 13 else 0.0

            signals[cb_sym] = {
                "binance_price": bn_price,
                "binance_ret_15m": bn_ret_15m,
                "binance_ret_1h": bn_ret_1h,
            }
            time.sleep(self.pause)

        return signals
