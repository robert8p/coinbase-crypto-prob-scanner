from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

STABLES = {"USDT", "USDC", "DAI", "PYUSD", "USDP", "EURC", "USDS", "GUSD", "FDUSD", "TUSD", "RLUSD", "USD1"}


@dataclass(slots=True)
class DemoProduct:
    id: str
    base_currency: str
    quote_currency: str
    display_name: str
    status: str = "online"
    cancel_only: bool = False
    limit_only: bool = False
    post_only: bool = False
    auction_mode: bool = False
    trading_disabled: bool = False
    product_type: str = "SPOT"
    new: bool = False
    created_at: str | None = None
    view_only: bool = False


DEFAULT_DEMO_PRODUCTS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD",
    "UNI-USD", "AAVE-USD", "BCH-USD", "LTC-USD", "APT-USD", "ATOM-USD", "NEAR-USD", "ARB-USD",
    "OP-USD", "INJ-USD", "PEPE-USD", "BONK-USD", "SHIB-USD", "SUI-USD", "TIA-USD", "SEI-USD",
    "FET-USD", "RNDR-USD", "ONDO-USD", "WIF-USD", "JUP-USD", "PYTH-USD", "MATIC-USD", "ETC-USD",
    "DOT-USD", "FIL-USD", "ICP-USD", "HBAR-USD", "TRX-USD", "USDT-USD", "USDC-USD", "cbETH-USD", "OGN-USD",
]


def demo_currencies() -> List[dict]:
    return [
        {"id": "USD", "status": "online", "details": {"type": "fiat", "display_name": "US Dollar"}},
        {"id": "USDC", "status": "online", "details": {"type": "crypto", "display_name": "USD Coin"}},
        {"id": "USDT", "status": "online", "details": {"type": "crypto", "display_name": "Tether USD"}},
    ]


def demo_products() -> List[dict]:
    out: List[dict] = []
    now = datetime.now(timezone.utc)
    for idx, product_id in enumerate(DEFAULT_DEMO_PRODUCTS):
        base, quote = product_id.split("-")
        created_at = (now - timedelta(days=25 + idx * 7)).isoformat()
        out.append(asdict(DemoProduct(
            id=product_id,
            base_currency=base,
            quote_currency=quote,
            display_name=product_id,
            created_at=created_at,
            new=idx < 2,
            view_only=(product_id == "OGN-USD"),
        )))
    return out


def _seed_for(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8], 16)


def demo_stats_map(products: Iterable[str]) -> Dict[str, dict]:
    now = datetime.now(timezone.utc)
    stats: Dict[str, dict] = {}
    for i, pid in enumerate(products):
        rs = np.random.default_rng(_seed_for(pid))
        base_price = float(np.exp(rs.normal(np.log(10 + i * 2 + 1), 0.35)))
        price = round(base_price * (1 + rs.normal(0, 0.03)), 6)
        volume_24h = max(5_000, abs(rs.normal(2_000_000 + i * 150_000, 1_500_000)))
        volume_30d = volume_24h * (25 + rs.uniform(5, 12))
        stats[pid] = {
            "last": str(price),
            "open": str(round(price * (1 + rs.normal(0, 0.04)), 6)),
            "high": str(round(price * (1 + abs(rs.normal(0.04, 0.03))), 6)),
            "low": str(round(max(0.0001, price * (1 - abs(rs.normal(0.04, 0.03)))), 6)),
            "volume": str(round(volume_24h / max(price, 0.1), 6)),
            "volume_30day": str(round(volume_30d / max(price, 0.1), 6)),
            "asof": now.isoformat(),
        }
    return stats


def demo_candles(product_id: str, lookback_bars: int = 2200) -> pd.DataFrame:
    rs = np.random.default_rng(_seed_for(product_id))
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    times = pd.date_range(end=now, periods=lookback_bars, freq="5min", tz="UTC")

    symbol_bias = 0.0004 if product_id.startswith(("BTC", "ETH", "SOL", "LINK", "AAVE", "FET", "RNDR")) else 0.0001
    shock = rs.normal(0, 0.012, size=lookback_bars)
    trend = np.sin(np.linspace(0, 6.5, lookback_bars)) * 0.0008
    regime = np.where(np.arange(lookback_bars) > lookback_bars * 0.75, 0.0006, 0.0)
    returns = symbol_bias + trend + regime + shock
    close = 1.0 + np.cumsum(returns)
    close = np.exp(close)
    level = float(np.exp(rs.normal(np.log(5 + (abs(_seed_for(product_id)) % 200)), 0.8)))
    close = close / close[0] * level
    open_ = np.r_[close[0], close[:-1]]
    wick_scale = np.abs(rs.normal(0.005, 0.004, size=lookback_bars))
    high = np.maximum(open_, close) * (1 + wick_scale)
    low = np.minimum(open_, close) * (1 - wick_scale * (0.7 + rs.random(lookback_bars) * 0.6))
    volume = np.maximum(1.0, np.exp(rs.normal(9.0, 0.6, size=lookback_bars)))
    if product_id.startswith(("PEPE", "BONK", "WIF", "SHIB")):
        high *= (1 + np.abs(rs.normal(0.0, 0.012, size=lookback_bars)))
        low *= (1 - np.abs(rs.normal(0.0, 0.010, size=lookback_bars)))
    df = pd.DataFrame({
        "ts": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df
