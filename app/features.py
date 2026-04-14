from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# v2.6.0 feature set
# Removed: drawdown_24h (dup of dist_24h_high), drawdown_7d (dup of dist_7d_high), weekend_flag (redundant w/ dow_*)
# Added v2.6.0: btc_corr_24h, momentum_persistence_1h, rv_ratio_1h_24h, up_volume_ratio_1h, time_since_impulse
# Added v2.6.1: btc_recovery_from_trough, btc_trough_depth, move_vs_atr_ratio,
#               volume_concentration, volume_acceleration, spread_cost_proxy, session_bucket
FEATURE_COLUMNS = [
    "ret_5m", "ret_15m", "ret_30m", "ret_60m", "ret_6h", "ret_24h", "ret_3d", "ret_7d",
    "impulse_60m", "accel_30_60", "ema_fast_gap", "ema_slow_gap", "adx_proxy", "atr_pct",
    "rv_1h", "rv_6h", "rv_24h", "range_pct", "bb_width", "rvol_1h", "obv_slope",
    "dist_24h_high", "dist_7d_high", "ma_gap_24h",
    "path_smoothness", "reversal_rate", "downside_impulse", "downside_accel", "wickiness",
    "candle_efficiency", "jumpiness", "failed_breakout", "history_bars_ratio_24h", "history_bars_ratio_7d",
    "observed_bar_density_24h", "observed_bar_density_7d", "nonzero_volume_rate_24h", "dollar_vol_24h_log",
    "btc_ret_1h", "btc_ret_24h", "eth_ret_1h", "eth_ret_24h", "asset_vs_btc_1h", "asset_vs_eth_1h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # v2.6.0 features
    "btc_corr_24h", "momentum_persistence_1h", "rv_ratio_1h_24h",
    "up_volume_ratio_1h", "time_since_impulse",
    # v2.6.1 blindspot features
    "btc_recovery_from_trough", "btc_trough_depth",
    "move_vs_atr_ratio", "volume_concentration", "volume_acceleration",
    "spread_cost_proxy", "session_bucket",
    # NOTE: Binance cross-exchange signals (binance_lead_15m, binance_lead_1h,
    # binance_price_gap) are NOT in FEATURE_COLUMNS. They are computed by
    # compute_live_features and stored in the feature row, but applied as
    # post-model adjustments in the scanner — not learned by the model.
    # This is because historical training data has no Binance backfill,
    # so including them would create a train/live distribution mismatch.
]


@dataclass(slots=True)
class FeatureResult:
    feature_row: dict
    diagnostics: dict
    block_reason: str | None


def _ret(series: pd.Series, bars: int) -> float:
    if len(series) <= bars:
        return 0.0
    prev = float(series.iloc[-bars - 1])
    cur = float(series.iloc[-1])
    return (cur / prev) - 1.0 if prev > 0 else 0.0


def _efficiency(close: pd.Series, window: int) -> float:
    s = close.tail(window)
    if len(s) < 3:
        return 0.0
    net = abs(float(s.iloc[-1] - s.iloc[0]))
    gross = float(np.abs(np.diff(s.values)).sum()) + 1e-9
    return net / gross


def _adx_proxy(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> float:
    if len(close) <= window + 2:
        return 0.0
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window).mean() / (atr + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return float(dx.rolling(window).mean().iloc[-1]) if len(dx.dropna()) else 0.0


def _momentum_persistence(returns_1: pd.Series, window: int = 12) -> float:
    """Autocorrelation of 5m returns over the last `window` bars. High = trending."""
    tail = returns_1.tail(window + 1)
    if len(tail) < 4:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ac = float(tail.autocorr(lag=1))
        return ac if not np.isnan(ac) else 0.0
    except Exception:
        return 0.0


def _up_volume_ratio(close: pd.Series, volume: pd.Series, window: int = 12) -> float:
    """Fraction of volume on up-bars in the last `window` bars."""
    if len(close) < window + 1:
        return 0.5
    tail_ret = close.pct_change().tail(window)
    tail_vol = volume.tail(window)
    total_vol = float(tail_vol.sum())
    if total_vol <= 0:
        return 0.5
    up_vol = float(tail_vol[tail_ret > 0].sum())
    return up_vol / total_vol


def _time_since_impulse(returns_1: pd.Series, threshold_sigma: float = 2.0) -> float:
    """Bars since the last |return| > threshold_sigma * std(returns). Capped at 100."""
    if len(returns_1) < 10:
        return 100.0
    sigma = float(returns_1.std())
    if sigma < 1e-9:
        return 100.0
    threshold = threshold_sigma * sigma
    impulse_mask = returns_1.abs() > threshold
    if not impulse_mask.any():
        return 100.0
    last_impulse_idx = int(impulse_mask.values[::-1].argmax())
    return float(min(last_impulse_idx, 100))


def _btc_correlation(asset_returns: pd.Series, btc_returns: pd.Series, window: int = 288) -> float:
    """Rolling correlation between asset and BTC 5m returns over `window` bars."""
    if len(asset_returns) < 20 or len(btc_returns) < 20:
        return 0.0
    a = asset_returns.tail(window)
    b = btc_returns.tail(window)
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a.tail(min_len)
        b = b.tail(min_len)
    if len(a) < 10:
        return 0.0
    # Zero-variance assets (flat price / no trades) produce NaN correlation
    # with a numpy divide-by-zero warning. Suppress and return 0.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = float(a.corr(b))
        return corr if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


# ── v2.6.1 blindspot features ─────────────────────────────────────────────

def _btc_recovery_from_trough(btc_df: pd.DataFrame | None) -> tuple[float, float]:
    """Bars since BTC hit its 24h low, and depth of the trough.

    Returns (bars_since_trough, trough_depth).
    bars_since_trough: 0-288, low = fresh trough (dead cat bounce risk)
    trough_depth: 24h high-to-low range as fraction (0.0 = flat, 0.08 = 8% drop)
    """
    if btc_df is None or btc_df.empty or len(btc_df) < 12:
        return 288.0, 0.0
    low = btc_df["low"].astype(float).tail(288)
    high = btc_df["high"].astype(float).tail(288)
    if len(low) < 2:
        return 288.0, 0.0
    trough_idx = int(low.values.argmin())
    bars_since = len(low) - 1 - trough_idx
    high_24h = float(high.max())
    low_24h = float(low.min())
    trough_depth = (high_24h - low_24h) / (high_24h + 1e-9)
    return float(min(bars_since, 288)), float(trough_depth)


def _volume_concentration(volume: pd.Series, window: int = 12) -> float:
    """Ratio of max single-candle volume to mean volume in the window.

    High values (>3) suggest a single large actor; low values (~1) suggest
    distributed organic flow.
    """
    tail = volume.tail(window)
    if len(tail) < 2:
        return 1.0
    mean_vol = float(tail.mean())
    if mean_vol <= 0:
        return 1.0
    return float(tail.max() / mean_vol)


def _volume_acceleration(volume: pd.Series, half_window: int = 6) -> float:
    """Ratio of recent volume to prior volume.

    > 1 = volume still increasing (follow-through).
    < 1 = volume peaked and declining (potential exhaustion).
    """
    if len(volume) < half_window * 2:
        return 1.0
    recent = float(volume.tail(half_window).mean())
    prior = float(volume.iloc[-(half_window * 2):-half_window].mean())
    if prior <= 0:
        return 1.0
    return float(recent / (prior + 1e-9))


def _spread_cost_proxy(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 12) -> float:
    """Execution cost proxy: average (high-low)/close normalized by volume.

    High values = thin orderbook, expensive to trade.
    Low values = tight spread, cheap execution.
    """
    if len(close) < window:
        return 0.0
    h = high.tail(window)
    l = low.tail(window)
    c = close.tail(window)
    v = volume.tail(window)
    # range per dollar of volume
    spread = ((h - l) / (c + 1e-9))
    vol_norm = v / (v.mean() + 1e-9)
    # spread is worst when range is wide and volume is low
    cost = float((spread / (vol_norm + 1e-9)).mean())
    return float(min(cost, 1.0))  # cap at 1.0


def _session_bucket(hour: float) -> float:
    """Map UTC hour to trading session.
    0 = Asia (00:00-08:00 UTC) — Tokyo/Singapore
    1 = Europe (08:00-13:00 UTC) — London
    2 = US (13:00-21:00 UTC) — New York
    3 = Overnight (21:00-00:00 UTC) — thin liquidity
    """
    if hour < 8:
        return 0.0
    elif hour < 13:
        return 1.0
    elif hour < 21:
        return 2.0
    else:
        return 3.0


def compute_live_features(
    symbol: str,
    df: pd.DataFrame,
    btc_ctx: dict | None = None,
    eth_ctx: dict | None = None,
    btc_df: pd.DataFrame | None = None,
    cross_exchange: dict | None = None,
) -> FeatureResult:
    df = df.sort_values("ts").reset_index(drop=True).copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    observed_bars = int(df.attrs.get("observed_bars", len(df)))
    bar_count = len(df)

    logret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns_1 = close.pct_change().fillna(0.0)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=48, adjust=False).mean()
    ma_24h = close.rolling(288).mean()
    atr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
    tr_pct = float(atr / close.iloc[-1]) if close.iloc[-1] else 0.0
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std().fillna(0.0)
    bb_width = float((4 * bb_std.iloc[-1]) / (bb_mid.iloc[-1] + 1e-9)) if len(bb_mid.dropna()) else 0.0
    obv = (np.sign(close.diff().fillna(0.0)) * volume).cumsum()
    obv_slope = float((obv.iloc[-1] - obv.iloc[-12]) / (abs(obv.iloc[-12]) + 1e-9)) if len(obv) > 12 else 0.0
    highs_24h = float(high.tail(288).max()) if len(high) >= 1 else float(high.iloc[-1])
    highs_7d = float(high.tail(2016).max()) if len(high) >= 1 else float(high.iloc[-1])
    reversals = (np.sign(returns_1).diff().abs() > 0).rolling(24).mean().iloc[-1] if len(returns_1) > 25 else 0.0
    wickiness = float(((high - low) / (close + 1e-9)).tail(12).mean())
    jumpiness = float(np.abs(returns_1.tail(24)).quantile(0.95)) if len(returns_1) >= 5 else 0.0
    candle_eff = _efficiency(close, 24)
    failed_breakout = float((high.tail(12).max() > high.tail(48).max() * 0.997) and (close.iloc[-1] < high.tail(12).max() * 0.985)) if len(high) >= 48 else 0.0
    downside_impulse = float(min(0.0, _ret(close, 12)))
    downside_accel = float(min(0.0, _ret(close, 6) - _ret(close, 12)))
    rvol_1h = float(volume.tail(12).mean() / (volume.tail(288).mean() + 1e-9)) if len(volume) >= 12 else 1.0
    observed_24h = int(min(observed_bars, 288))
    observed_7d = int(min(observed_bars, 2016))
    available_24h = max(1, min(bar_count, 288))
    available_7d = max(1, min(bar_count, 2016))
    history_ratio_24h = min(bar_count / 288.0, 1.0)
    history_ratio_7d = min(bar_count / 2016.0, 1.0)
    observed_density_24h = observed_24h / available_24h
    observed_density_7d = observed_7d / available_7d
    nonzero_volume_rate_24h = float((volume.tail(288) > 0).mean()) if len(volume) >= 1 else 0.0
    dollar_vol_24h_log = float(np.log1p((close.tail(288) * volume.tail(288)).sum()))

    # v2.6.0 new features
    rv_1h = float(logret.tail(12).std() * np.sqrt(12))
    rv_24h = float(logret.tail(288).std() * np.sqrt(288))
    rv_ratio_1h_24h = rv_1h / (rv_24h + 1e-9) if rv_24h > 1e-9 else 1.0

    momentum_persist = _momentum_persistence(returns_1, window=12)
    up_vol_ratio = _up_volume_ratio(close, volume, window=12)
    time_since_imp = _time_since_impulse(returns_1, threshold_sigma=2.0)

    # BTC correlation
    btc_corr = 0.0
    if btc_df is not None and not btc_df.empty:
        btc_close = btc_df["close"].astype(float)
        btc_rets = btc_close.pct_change().fillna(0.0)
        btc_corr = _btc_correlation(returns_1, btc_rets, window=288)

    # v2.6.1 blindspot features
    btc_recovery, btc_trough = _btc_recovery_from_trough(btc_df)
    move_vs_atr = abs(_ret(close, 288)) / (tr_pct + 1e-9) if tr_pct > 1e-9 else 0.0
    vol_concentration = _volume_concentration(volume, window=12)
    vol_acceleration = _volume_acceleration(volume, half_window=6)
    spread_cost = _spread_cost_proxy(high, low, close, volume, window=12)

    ts = pd.to_datetime(df["ts"].iloc[-1], utc=True)
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    session = _session_bucket(hour)

    # v2.6.1 cross-exchange features
    # binance_lead_15m: Binance 15m return minus Coinbase 15m return.
    #   Positive = Binance moved MORE than Coinbase (Coinbase is lagging, will catch up or Binance will reverse)
    #   Negative = Coinbase moved more (less common, potentially more organic)
    # binance_lead_1h: same for 1h
    # binance_price_gap: (binance_price - coinbase_price) / coinbase_price
    #   Positive gap = Binance is higher (Coinbase may rise to converge)
    #   Negative gap = Binance is lower (Coinbase momentum may be about to reverse)
    cx = cross_exchange or {}
    coinbase_price = float(close.iloc[-1])
    coinbase_ret_15m = _ret(close, 3)
    coinbase_ret_1h = _ret(close, 12)
    bn_ret_15m = float(cx.get("binance_ret_15m", 0.0))
    bn_ret_1h = float(cx.get("binance_ret_1h", 0.0))
    bn_price = float(cx.get("binance_price", 0.0))

    binance_lead_15m = bn_ret_15m - coinbase_ret_15m if bn_ret_15m != 0.0 else 0.0
    binance_lead_1h = bn_ret_1h - coinbase_ret_1h if bn_ret_1h != 0.0 else 0.0
    binance_price_gap = (bn_price - coinbase_price) / (coinbase_price + 1e-9) if bn_price > 0 else 0.0

    row = {
        "ret_5m": _ret(close, 1),
        "ret_15m": _ret(close, 3),
        "ret_30m": _ret(close, 6),
        "ret_60m": _ret(close, 12),
        "ret_6h": _ret(close, 72),
        "ret_24h": _ret(close, 288),
        "ret_3d": _ret(close, 864),
        "ret_7d": _ret(close, 2016),
        "impulse_60m": float(returns_1.tail(12).clip(lower=0).sum()),
        "accel_30_60": _ret(close, 6) - _ret(close, 12),
        "ema_fast_gap": float(close.iloc[-1] / (ema_fast.iloc[-1] + 1e-9) - 1),
        "ema_slow_gap": float(close.iloc[-1] / (ema_slow.iloc[-1] + 1e-9) - 1),
        "adx_proxy": _adx_proxy(close, high, low),
        "atr_pct": tr_pct,
        "rv_1h": rv_1h,
        "rv_6h": float(logret.tail(72).std() * np.sqrt(72)),
        "rv_24h": rv_24h,
        "range_pct": float((high.tail(12).max() - low.tail(12).min()) / (close.iloc[-1] + 1e-9)),
        "bb_width": bb_width,
        "rvol_1h": rvol_1h,
        "obv_slope": obv_slope,
        "dist_24h_high": float(close.iloc[-1] / (highs_24h + 1e-9) - 1),
        "dist_7d_high": float(close.iloc[-1] / (highs_7d + 1e-9) - 1),
        "ma_gap_24h": float(close.iloc[-1] / (ma_24h.iloc[-1] + 1e-9) - 1) if len(ma_24h.dropna()) else 0.0,
        "path_smoothness": _efficiency(close, 48),
        "reversal_rate": float(reversals),
        "downside_impulse": downside_impulse,
        "downside_accel": downside_accel,
        "wickiness": wickiness,
        "candle_efficiency": candle_eff,
        "jumpiness": jumpiness,
        "failed_breakout": failed_breakout,
        "history_bars_ratio_24h": history_ratio_24h,
        "history_bars_ratio_7d": history_ratio_7d,
        "observed_bar_density_24h": observed_density_24h,
        "observed_bar_density_7d": observed_density_7d,
        "nonzero_volume_rate_24h": nonzero_volume_rate_24h,
        "dollar_vol_24h_log": dollar_vol_24h_log,
        "btc_ret_1h": (btc_ctx or {}).get("ret_1h", 0.0),
        "btc_ret_24h": (btc_ctx or {}).get("ret_24h", 0.0),
        "eth_ret_1h": (eth_ctx or {}).get("ret_1h", 0.0),
        "eth_ret_24h": (eth_ctx or {}).get("ret_24h", 0.0),
        "asset_vs_btc_1h": _ret(close, 12) - (btc_ctx or {}).get("ret_1h", 0.0),
        "asset_vs_eth_1h": _ret(close, 12) - (eth_ctx or {}).get("ret_1h", 0.0),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
        # v2.6.0 new features
        "btc_corr_24h": btc_corr,
        "momentum_persistence_1h": momentum_persist,
        "rv_ratio_1h_24h": rv_ratio_1h_24h,
        "up_volume_ratio_1h": up_vol_ratio,
        "time_since_impulse": time_since_imp,
        # v2.6.1 blindspot features
        "btc_recovery_from_trough": btc_recovery,
        "btc_trough_depth": btc_trough,
        "move_vs_atr_ratio": float(min(move_vs_atr, 20.0)),  # cap at 20x
        "volume_concentration": float(min(vol_concentration, 15.0)),  # cap outliers
        "volume_acceleration": float(min(vol_acceleration, 10.0)),  # cap outliers
        "spread_cost_proxy": spread_cost,
        "session_bucket": session,
        # v2.6.1 cross-exchange features
        "binance_lead_15m": float(binance_lead_15m),
        "binance_lead_1h": float(binance_lead_1h),
        "binance_price_gap": float(np.clip(binance_price_gap, -0.05, 0.05)),  # cap at ±5%
    }

    activity_rate = nonzero_volume_rate_24h
    illiquidity_proxy = float(((high - low) / (close + 1e-9)).tail(24).median()) if len(close) >= 2 else 0.0
    diag = {
        "ret_1h": row["ret_60m"],
        "ret_24h": row["ret_24h"],
        "activity_rate": activity_rate,
        "illiquidity_proxy": illiquidity_proxy,
        "latest_price": float(close.iloc[-1]),
        "ts": ts.isoformat(),
        "history_bars": int(bar_count),
        "observed_bars": int(observed_bars),
        "observed_bar_density_24h": observed_density_24h,
        "observed_bar_density_7d": observed_density_7d,
        "history_bars_ratio_24h": history_ratio_24h,
        "history_bars_ratio_7d": history_ratio_7d,
        "dollar_vol_24h": float((close.tail(288) * volume.tail(288)).sum()),
    }

    block_reason = None
    if row["jumpiness"] > 0.08 and row["wickiness"] > 0.055 and row["observed_bar_density_24h"] < 0.55:
        block_reason = "one_candle_pump_or_thin_market"
    elif row["downside_impulse"] < -0.05 and row["ema_slow_gap"] < -0.03:
        block_reason = "falling_knife"
    elif row["ret_24h"] < -0.12 and row["path_smoothness"] < 0.15:
        block_reason = "structural_weakness"
    # v2.6.1: btc_led_panic downgraded from hard block to soft block.
    # Only hard-block the most extreme cases: BTC down >5% AND asset actively crashing.
    # Normal BTC panic (-2.5% to -5%) is handled by EVENT_RISK + regime gating.
    elif row["btc_ret_1h"] < -0.05 and row["ret_15m"] < -0.03 and row["asset_vs_btc_1h"] < -0.02:
        block_reason = "btc_led_crash"
    return FeatureResult(feature_row=row, diagnostics=diag, block_reason=block_reason)


def _stage1_primary_score(row: dict, guard: dict, *, btc_regime: str) -> float:
    is_panic = btc_regime == "BTC panic"
    is_weak = btc_regime == "BTC weak"
    liquidity_penalty = float(guard.get("liquidity_penalty", 0.0) or 0.0)
    history_bonus = 0.08 * float(row.get("history_bars_ratio_24h", 0.0) or 0.0) + 0.05 * float(row.get("history_bars_ratio_7d", 0.0) or 0.0)
    observed_bonus = 0.08 * float(row.get("observed_bar_density_24h", 0.0) or 0.0) + 0.03 * float(row.get("nonzero_volume_rate_24h", 0.0) or 0.0)
    score = (
        0.28 * float(row.get("ret_60m", 0.0) or 0.0)
        + 0.18 * float(row.get("ret_24h", 0.0) or 0.0)
        + 0.08 * float(row.get("ret_6h", 0.0) or 0.0)
        + 0.10 * float(row.get("asset_vs_btc_1h", 0.0) or 0.0)
        + 0.06 * float(row.get("adx_proxy", 0.0) or 0.0) / 100.0
        + 0.08 * float(row.get("path_smoothness", 0.0) or 0.0)
        + 0.06 * float(row.get("candle_efficiency", 0.0) or 0.0)
        + 0.08 * min(float(row.get("rvol_1h", 0.0) or 0.0), 4.0) / 4.0
        + history_bonus
        + observed_bonus
        + 0.04 * min(float(row.get("dollar_vol_24h_log", 0.0) or 0.0), 18.0) / 18.0
        - 0.10 * float(row.get("wickiness", 0.0) or 0.0)
        - 0.09 * max(0.0, -float(row.get("downside_impulse", 0.0) or 0.0))
        - 0.10 * float(guard.get("uncertainty", 0.0) or 0.0)
        - 0.12 * liquidity_penalty
    )
    if (is_panic or is_weak) and float(row.get("asset_vs_btc_1h", 0.0) or 0.0) < 0:
        score -= 0.06
    if is_panic:
        score -= 0.03
    if float(row.get("dollar_vol_24h_log", 0.0) or 0.0) < np.log1p(250_000):
        score -= 0.05
    return float(score)


def _stage1_recall_score(row: dict, guard: dict, *, btc_regime: str) -> float:
    """Alternative stage1 score biased toward opportunity recall rather than shortlist neatness."""
    is_panic = btc_regime == "BTC panic"
    is_weak = btc_regime == "BTC weak"
    liquidity_penalty = float(guard.get("liquidity_penalty", 0.0) or 0.0)
    uncertainty = float(guard.get("uncertainty", 0.0) or 0.0)
    history_bonus = 0.06 * float(row.get("history_bars_ratio_24h", 0.0) or 0.0) + 0.04 * float(row.get("history_bars_ratio_7d", 0.0) or 0.0)
    observed_bonus = 0.06 * float(row.get("observed_bar_density_24h", 0.0) or 0.0) + 0.03 * float(row.get("nonzero_volume_rate_24h", 0.0) or 0.0)
    score = (
        0.20 * float(row.get("ret_15m", 0.0) or 0.0)
        + 0.18 * float(row.get("ret_60m", 0.0) or 0.0)
        + 0.10 * float(row.get("ret_6h", 0.0) or 0.0)
        + 0.08 * float(row.get("ret_24h", 0.0) or 0.0)
        + 0.12 * float(row.get("asset_vs_btc_1h", 0.0) or 0.0)
        + 0.07 * float(row.get("impulse_60m", 0.0) or 0.0)
        + 0.06 * float(row.get("momentum_persistence_1h", 0.0) or 0.0)
        + 0.06 * float(row.get("up_volume_ratio_1h", 0.0) or 0.0)
        + 0.06 * float(row.get("move_vs_atr_ratio", 0.0) or 0.0)
        + 0.05 * float(row.get("volume_acceleration", 0.0) or 0.0)
        + 0.05 * float(row.get("path_smoothness", 0.0) or 0.0)
        + 0.04 * float(row.get("candle_efficiency", 0.0) or 0.0)
        + 0.05 * min(float(row.get("rvol_1h", 0.0) or 0.0), 4.0) / 4.0
        + history_bonus
        + observed_bonus
        + 0.03 * min(float(row.get("dollar_vol_24h_log", 0.0) or 0.0), 18.0) / 18.0
        - 0.07 * float(row.get("wickiness", 0.0) or 0.0)
        - 0.05 * max(0.0, -float(row.get("downside_impulse", 0.0) or 0.0))
        - 0.06 * uncertainty
        - 0.06 * liquidity_penalty
    )
    if (is_panic or is_weak) and float(row.get("asset_vs_btc_1h", 0.0) or 0.0) < 0:
        score -= 0.04
    if is_panic:
        score -= 0.02
    return float(score)


def stage1_select(
    feature_rows: Dict[str, dict],
    guardrails: Dict[str, dict],
    max_candidates: int,
    btc_regime: str = "BTC mixed",
    *,
    selection_mode: str = "primary_only",
    recall_reserve_frac: float = 0.25,
    recall_reserve_min: int = 6,
    recall_reserve_max: int = 12,
    promotion_overflow_window: int = 20,
    opportunity_model_scores: Dict[str, float] | None = None,
) -> tuple[List[str], dict]:
    """Stage1 selection with explicit modes.

    Default is primary_only because replay evidence has shown that widening stage1 without strong evidence can
    easily hurt visible-vs-hidden separation. Hybrid modes remain available for replay audits and controlled
    experiments.
    """
    is_panic = btc_regime == "BTC panic"
    primary_scored: List[Tuple[str, float]] = []
    recall_scored: List[Tuple[str, float]] = []
    for symbol, row in feature_rows.items():
        guard = guardrails[symbol]
        if str(guard.get("block_code") or "") == "BLOCKED":
            continue
        primary_scored.append((symbol, _stage1_primary_score(row, guard, btc_regime=btc_regime)))
        recall_scored.append((symbol, _stage1_recall_score(row, guard, btc_regime=btc_regime)))
    primary_scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    recall_scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

    effective_max = max(1, int(max_candidates))
    if is_panic:
        effective_max = min(effective_max, max(1, effective_max // 2))

    mode = str(selection_mode or "primary_only").strip().lower()
    if mode not in {"primary_only", "hybrid_primary_plus_recall_reserve", "primary_plus_near_miss_recall_promotion", "stage1_opportunity_model", "primary_plus_opportunity_reserve"}:
        mode = "primary_only"

    reserve_n = int(round(effective_max * max(0.0, float(recall_reserve_frac))))
    reserve_n = max(0, reserve_n)
    reserve_n = max(reserve_n, max(0, int(recall_reserve_min))) if effective_max > 8 else min(reserve_n, effective_max)
    reserve_n = min(reserve_n, max(0, int(recall_reserve_max)))
    reserve_n = min(reserve_n, max(0, effective_max // 2))
    if is_panic:
        reserve_n = min(reserve_n, max(2, reserve_n // 2))
    primary_slots = max(1, effective_max - reserve_n)

    selected: List[str] = []
    selected_sources: Dict[str, str] = {}
    primary_ranks = {symbol: idx for idx, (symbol, _) in enumerate(primary_scored, start=1)}
    recall_ranks = {symbol: idx for idx, (symbol, _) in enumerate(recall_scored, start=1)}
    primary_scores = {symbol: round(float(score), 6) for symbol, score in primary_scored}
    recall_scores = {symbol: round(float(score), 6) for symbol, score in recall_scored}
    opportunity_scores = {str(symbol): round(float(score), 6) for symbol, score in (opportunity_model_scores or {}).items()}
    opportunity_ranked = sorted([(symbol, score) for symbol, score in opportunity_scores.items() if symbol in primary_ranks], key=lambda x: (x[1], x[0]), reverse=True)
    opportunity_ranks = {symbol: idx for idx, (symbol, _) in enumerate(opportunity_ranked, start=1)}

    if mode == "primary_only":
        for symbol, _ in primary_scored[:effective_max]:
            selected.append(symbol)
            selected_sources[symbol] = "primary"
        primary_slots = effective_max
        reserve_n = 0
    elif mode == "stage1_opportunity_model":
        ranked = opportunity_ranked if opportunity_ranked else primary_scored
        for symbol, _ in ranked[:effective_max]:
            if symbol not in selected:
                selected.append(symbol)
                selected_sources[symbol] = "opportunity_model"
        primary_slots = effective_max
        reserve_n = 0
    elif mode == "primary_plus_opportunity_reserve":
        for symbol, _ in primary_scored[:primary_slots]:
            if symbol not in selected:
                selected.append(symbol)
                selected_sources[symbol] = "primary"
        ranked = opportunity_ranked if opportunity_ranked else recall_scored
        added = 0
        for symbol, _ in ranked:
            if len(selected) >= effective_max or added >= reserve_n:
                break
            if symbol in selected:
                continue
            selected.append(symbol)
            selected_sources[symbol] = "opportunity_reserve"
            added += 1
        if len(selected) < effective_max:
            for symbol, _ in primary_scored:
                if len(selected) >= effective_max:
                    break
                if symbol in selected:
                    continue
                selected.append(symbol)
                selected_sources[symbol] = "primary_backfill"
    elif mode == "primary_plus_near_miss_recall_promotion":
        overflow_window = max(1, int(promotion_overflow_window))
        for symbol, _ in primary_scored[:primary_slots]:
            if symbol not in selected:
                selected.append(symbol)
                selected_sources[symbol] = "primary"
        promotion_pool = [
            (symbol, recall_scores[symbol])
            for symbol, _ in recall_scored
            if symbol not in selected and primary_ranks.get(symbol, 999999) <= (effective_max + overflow_window)
        ]
        promotion_pool.sort(key=lambda x: (x[1], x[0]), reverse=True)
        promoted = 0
        for symbol, _ in promotion_pool:
            if len(selected) >= effective_max or promoted >= reserve_n:
                break
            selected.append(symbol)
            selected_sources[symbol] = "recall_promotion"
            promoted += 1
        if len(selected) < effective_max:
            for symbol, _ in primary_scored:
                if len(selected) >= effective_max:
                    break
                if symbol in selected:
                    continue
                selected.append(symbol)
                selected_sources[symbol] = "primary_backfill"
    else:
        for symbol, _ in primary_scored[:primary_slots]:
            if symbol not in selected:
                selected.append(symbol)
                selected_sources[symbol] = "primary"

        recall_added = 0
        for symbol, _ in recall_scored:
            if len(selected) >= effective_max or recall_added >= reserve_n:
                break
            if symbol in selected:
                continue
            selected.append(symbol)
            selected_sources[symbol] = "recall_reserve"
            recall_added += 1

        if len(selected) < effective_max:
            for symbol, _ in primary_scored:
                if len(selected) >= effective_max:
                    break
                if symbol in selected:
                    continue
                selected.append(symbol)
                selected_sources[symbol] = "primary_backfill"

    meta = {
        "selection_mode": mode,
        "effective_max_candidates": effective_max,
        "primary_slots": primary_slots,
        "recall_reserve_slots": reserve_n,
        "selected_sources": selected_sources,
        "primary_ranks": primary_ranks,
        "recall_ranks": recall_ranks,
        "opportunity_ranks": opportunity_ranks,
        "primary_scores": primary_scores,
        "recall_scores": recall_scores,
        "opportunity_scores": opportunity_scores,
        "selected_primary_count": sum(1 for source in selected_sources.values() if source == "primary"),
        "selected_recall_reserve_count": sum(1 for source in selected_sources.values() if source == "recall_reserve"),
        "selected_recall_promotion_count": sum(1 for source in selected_sources.values() if source == "recall_promotion"),
        "selected_opportunity_model_count": sum(1 for source in selected_sources.values() if source == "opportunity_model"),
        "selected_opportunity_reserve_count": sum(1 for source in selected_sources.values() if source == "opportunity_reserve"),
        "selected_primary_backfill_count": sum(1 for source in selected_sources.values() if source == "primary_backfill"),
    }
    return selected, meta


def stage1_rank(
    feature_rows: Dict[str, dict],
    guardrails: Dict[str, dict],
    max_candidates: int,
    btc_regime: str = "BTC mixed",
) -> List[str]:
    selected, _ = stage1_select(feature_rows, guardrails, max_candidates, btc_regime=btc_regime, selection_mode="primary_only")
    return selected


def compute_guardrails(symbol: str, row: dict, diag: dict, block_reason: str | None, training_profile: dict | None, cfg) -> dict:
    downside = min(
        1.0,
        max(
            0.0,
            0.32 * max(0.0, -row["downside_impulse"]) / 0.05
            + 0.18 * max(0.0, -row["downside_accel"]) / 0.03
            + 0.15 * row["wickiness"] / 0.05
            + 0.09 * row["jumpiness"] / 0.05
            + 0.10 * max(0.0, -row["ema_slow_gap"]) / 0.04,
        ),
    )
    downside_reasons: List[str] = []
    if row["downside_impulse"] < -0.03:
        downside_reasons.append("recent downside impulse")
    if row["downside_accel"] < -0.015:
        downside_reasons.append("downside acceleration")
    if row["wickiness"] > 0.04:
        downside_reasons.append("high wickiness")

    uncertainty = 0.10
    uncertainty_reasons: List[str] = []
    if training_profile and training_profile.get("feature_mean"):
        means = training_profile["feature_mean"]
        stds = training_profile["feature_std"]
        z_count = 0
        abs_z_sum = 0.0
        tracked = 0
        for key in FEATURE_COLUMNS:
            mu = means.get(key)
            sd = stds.get(key)
            if mu is None or sd is None:
                continue
            sd = max(float(sd), 1e-6)
            z = abs((row[key] - float(mu)) / sd)
            if z > 3.0:
                z_count += 1
            abs_z_sum += min(z, 6.0)
            tracked += 1
        if tracked:
            uncertainty = min(1.0, 0.07 + 0.05 * z_count + 0.020 * (abs_z_sum / tracked))
        if z_count >= 3:
            uncertainty_reasons.append("ood feature mix")
        if abs(row["btc_ret_24h"] - training_profile.get("btc_ret_24h_median", 0.0)) > 0.08:
            uncertainty = min(1.0, uncertainty + 0.10)
            uncertainty_reasons.append("btc regime mismatch")
    else:
        if row["jumpiness"] > 0.06:
            uncertainty += 0.16
            uncertainty_reasons.append("jumpiness")
        if row["wickiness"] > 0.05:
            uncertainty += 0.10
            uncertainty_reasons.append("thin-market proxy")

    activity_rate = float(diag.get("activity_rate", 1.0))
    illiq = float(diag.get("illiquidity_proxy", 0.0))
    rolling_dollar_volume = float(diag.get("rolling_dollar_volume", 0.0) or 0.0)
    dollar_vol_24h = float(diag.get("dollar_vol_24h", 0.0) or 0.0)
    history_bars = int(diag.get("history_bars", 0))
    observed_bars = int(diag.get("observed_bars", 0))
    liquidity_penalty = 0.0

    if history_bars < cfg.stage2_min_history_5m_bars:
        add = min(0.28, (cfg.stage2_min_history_5m_bars - history_bars) / max(cfg.stage2_min_history_5m_bars, 1) * 0.30)
        uncertainty = min(1.0, uncertainty + add)
        liquidity_penalty += add * 0.5
        uncertainty_reasons.append("limited history")
    if observed_bars < cfg.stage2_min_observed_5m_bars:
        add = min(0.24, (cfg.stage2_min_observed_5m_bars - observed_bars) / max(cfg.stage2_min_observed_5m_bars, 1) * 0.28)
        uncertainty = min(1.0, uncertainty + add)
        downside = min(1.0, downside + add * 0.4)
        liquidity_penalty += add
        uncertainty_reasons.append("sparse prints")
    if activity_rate < cfg.universe_min_activity_rate:
        add = min(0.22, (cfg.universe_min_activity_rate - activity_rate) / max(cfg.universe_min_activity_rate, 1e-6) * 0.25)
        uncertainty = min(1.0, uncertainty + add)
        downside = min(1.0, downside + add * 0.4)
        liquidity_penalty += add
        uncertainty_reasons.append("low activity")
    if illiq > cfg.universe_max_illiquidity_proxy:
        stretch = min(0.35, (illiq - cfg.universe_max_illiquidity_proxy) / max(cfg.universe_max_illiquidity_proxy, 1e-6) * 0.18)
        uncertainty = min(1.0, uncertainty + stretch)
        downside = min(1.0, downside + stretch)
        liquidity_penalty += stretch
        downside_reasons.append("illiquidity proxy")
    if rolling_dollar_volume and rolling_dollar_volume < cfg.universe_min_24h_dollar_volume_usd:
        uncertainty = min(1.0, uncertainty + 0.04)
        uncertainty_reasons.append("below volume floor")
        liquidity_penalty += 0.04

    # v2.6.0: Stage 2 dollar volume penalty
    effective_dollar_vol = max(dollar_vol_24h, rolling_dollar_volume)
    if effective_dollar_vol < cfg.stage2_min_dollar_volume_soft and effective_dollar_vol >= cfg.stage2_min_dollar_volume_hard:
        uncertainty = min(1.0, uncertainty + 0.10)
        uncertainty_reasons.append("low dollar volume")
        liquidity_penalty += 0.10

    risk = min(1.0, 0.40 * downside + 0.34 * uncertainty + 0.14 * max(0.0, -row["ret_60m"]) / 0.04 + 0.12 * liquidity_penalty)
    risk_reasons: List[str] = []
    if row["ret_60m"] < -0.02:
        risk_reasons.append("negative 1h momentum")
    if row["asset_vs_btc_1h"] < -0.015:
        risk_reasons.append("lagging BTC")
    risk_reasons.extend(downside_reasons[:1])
    risk_reasons.extend(uncertainty_reasons[:1])

    block_code = "OK"
    if block_reason:
        block_code = "BLOCKED"
    elif observed_bars < max(6, cfg.stage1_min_observed_5m_bars // 2) and activity_rate < 0.10 and illiq > cfg.universe_max_illiquidity_proxy * 2.5:
        block_code = "BLOCKED"
        block_reason = "extreme_thin_liquidity"
    # v2.6.0: hard dollar volume block
    elif effective_dollar_vol > 0 and effective_dollar_vol < cfg.stage2_min_dollar_volume_hard:
        block_code = "BLOCKED"
        block_reason = "below_dollar_volume_hard_floor"
    elif row["btc_ret_1h"] < cfg.btc_panic_threshold and (row["ret_15m"] < -0.01 or row["asset_vs_btc_1h"] < -0.01):
        block_code = "EVENT_RISK"
        risk_reasons.append("BTC shock regime")

    capped = False
    if downside > cfg.downside_cap or uncertainty > cfg.uncertainty_cap:
        capped = True
    if block_code == "EVENT_RISK" and risk > cfg.event_risk_cap:
        capped = True

    return {
        "risk": round(float(risk), 4),
        "risk_reasons": risk_reasons or ["normal"],
        "downside_risk": round(float(downside), 4),
        "downside_reasons": downside_reasons or ["contained"],
        "uncertainty": round(float(min(1.0, uncertainty)), 4),
        "uncertainty_reasons": uncertainty_reasons or ["within training envelope"],
        "block_code": block_code,
        "block_reason": block_reason,
        "capped": capped,
        "liquidity_penalty": round(float(liquidity_penalty), 4),
    }


def heuristic_probability(row: dict, guardrail: dict, *, guardrail_cap: float = 0.65) -> float:
    raw = (
        0.16
        + 0.88 * max(0.0, row["ret_60m"])
        + 0.44 * max(0.0, row["ret_24h"])
        + 0.18 * max(0.0, row["asset_vs_btc_1h"])
        + 0.06 * min(row["adx_proxy"], 50.0) / 50.0
        + 0.06 * min(row["rvol_1h"], 3.0) / 3.0
        + 0.05 * row["path_smoothness"]
        + 0.04 * row["history_bars_ratio_24h"]
        + 0.03 * row["observed_bar_density_24h"]
        + 0.03 * max(0.0, row.get("momentum_persistence_1h", 0.0))
        + 0.03 * max(0.0, row.get("up_volume_ratio_1h", 0.5) - 0.5) * 2
        - 0.32 * guardrail["downside_risk"]
        - 0.28 * guardrail["uncertainty"]
        - 0.14 * row["wickiness"]
        - 0.12 * guardrail.get("liquidity_penalty", 0.0)
        - 0.06 * max(0.0, row.get("btc_corr_24h", 0.0))
        # v2.6.1 blindspot penalties
        - 0.08 * max(0.0, (row.get("move_vs_atr_ratio", 0.0) - 4.0)) / 10.0  # penalize extended moves >4x ATR
        - 0.05 * max(0.0, (row.get("volume_concentration", 1.0) - 4.0)) / 6.0  # penalize whale-driven volume
        - 0.04 * max(0.0, row.get("spread_cost_proxy", 0.0) - 0.03) / 0.10  # penalize expensive execution
        - 0.06 * max(0.0, row.get("btc_trough_depth", 0.0) - 0.03) / 0.05 * max(0.0, 1.0 - row.get("btc_recovery_from_trough", 288.0) / 36.0)  # dead cat bounce: trough depth × recency
        # NOTE: Binance cross-exchange penalties are applied post-model in the scanner,
        # not here, to keep heuristic and trained model on the same feature basis.
    )
    prob = 1.0 / (1.0 + np.exp(-4.0 * (raw - 0.24)))
    if guardrail["block_code"] == "BLOCKED":
        return 0.0
    if guardrail["capped"]:
        prob = min(prob, float(guardrail_cap or 0.65))
    return float(np.clip(prob, 0.01, 0.95))


def build_training_frame(
    symbol: str,
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None,
    eth_df: pd.DataFrame | None,
    sample_every: int,
    horizon_bars: int,
    target_move_pct: float,
    warmup_bars: int,
    quality_max_mae: float = -0.020,
    quality_min_end_ret: float = -0.008,
) -> pd.DataFrame:
    """v2.7.0: quality-conditioned label with configurable thresholds."""
    df = df.sort_values("ts").reset_index(drop=True)
    btc_ctx_series = _context_lookup(btc_df)
    eth_ctx_series = _context_lookup(eth_df)
    rows = []
    warmup = max(96, warmup_bars)
    end_idx = len(df) - horizon_bars - 1
    for i in range(warmup, max(warmup, end_idx), sample_every):
        window = df.iloc[: i + 1].copy()
        window.attrs["observed_bars"] = int(window["volume"].gt(0).sum())
        ts = pd.to_datetime(window["ts"].iloc[-1], utc=True)
        btc_ctx = btc_ctx_series.get(ts, {"ret_1h": 0.0, "ret_24h": 0.0})
        eth_ctx = eth_ctx_series.get(ts, {"ret_1h": 0.0, "ret_24h": 0.0})
        # pass btc_df slice for btc_corr feature
        btc_slice = None
        if btc_df is not None and not btc_df.empty:
            btc_ts = pd.to_datetime(btc_df["ts"], utc=True)
            btc_slice = btc_df[btc_ts <= ts].copy()
        feat = compute_live_features(symbol, window, btc_ctx=btc_ctx, eth_ctx=eth_ctx, btc_df=btc_slice)
        future_high = float(df["high"].iloc[i + 1 : i + 1 + horizon_bars].max())
        p0 = float(window["close"].iloc[-1])
        touched = int(future_high >= (1.0 + target_move_pct) * p0)
        future_low = float(df["low"].iloc[i + 1 : i + 1 + horizon_bars].min())
        mae = (future_low / p0) - 1.0
        future_end = float(df["close"].iloc[i + horizon_bars])
        end_ret = (future_end / p0) - 1.0

        # v2.6.0: quality-conditioned label is the primary training target
        y_quality = int(touched and mae > quality_max_mae and end_ret > quality_min_end_ret)

        # legacy diagnostics preserved — use actual quality thresholds
        touch_quality = touched and mae > quality_max_mae and end_ret > quality_min_end_ret
        path_ugliness = feat.feature_row["wickiness"] + max(0.0, -mae)

        # v2.6.0: aggressive sample weighting based on path quality
        sample_weight = 1.0
        if touched and mae < -0.08:
            sample_weight = 0.0  # exclude extremely ugly touches
        elif touched and mae < -0.05:
            sample_weight = 0.30
        elif touched and mae < -0.03:
            sample_weight = 0.50
        elif y_quality:
            sample_weight = 1.10  # bonus for quality touches

        rows.append(
            {
                **feat.feature_row,
                "symbol": symbol,
                "ts": ts,
                "y": y_quality,  # v2.6.0: primary target is quality-conditioned
                "y_raw_touch": touched,  # raw touch preserved as diagnostic
                "mae": mae,
                "end_ret": end_ret,
                "touch_quality": int(touch_quality),
                "touched_before_major_adverse": int(touched and mae > -0.03),
                "path_ugliness": path_ugliness,
                "sample_weight": sample_weight,
                "block_reason": feat.block_reason,
            }
        )
    return pd.DataFrame(rows)


def _context_lookup(df: pd.DataFrame | None) -> Dict[pd.Timestamp, dict]:
    if df is None or df.empty:
        return {}
    df = df.sort_values("ts").reset_index(drop=True)
    out: Dict[pd.Timestamp, dict] = {}
    close = df["close"].astype(float)
    for i in range(len(df)):
        ts = pd.to_datetime(df["ts"].iloc[i], utc=True)
        out[ts] = {
            "ret_1h": _ret(close.iloc[: i + 1], 12),
            "ret_24h": _ret(close.iloc[: i + 1], 288),
        }
    return out
