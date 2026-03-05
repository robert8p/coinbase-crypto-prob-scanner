import datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import Settings
from .storage import ensure_dir, atomic_write_json, read_json, safe_filename

SCHEMA_VERSION = "coinbase-crypto-v1"

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-12)
    dx = (100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)).fillna(0.0)
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0.0)

def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    arr = y.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x-x_mean)**2).sum()
    for i in range(window-1, n):
        yy = arr[i-window+1:i+1]
        if np.any(np.isnan(yy)):
            continue
        y_mean = yy.mean()
        num = ((x-x_mean)*(yy-y_mean)).sum()
        out[i] = num / (denom + 1e-12)
    return pd.Series(out, index=y.index)

def _tod_frac(ts_end: pd.Timestamp) -> float:
    mins = ts_end.hour*60 + ts_end.minute
    return mins/1440.0

def _slot_5m(ts_end: pd.Timestamp) -> int:
    mins = ts_end.hour*60 + ts_end.minute
    return int(mins//5)

def _volume_profile_path(cfg: Settings, product_id: str) -> Path:
    return Path(cfg.model_dir) / "volume_profiles" / f"{safe_filename(product_id)}.json"

def ensure_volume_profile_runtime(cfg: Settings, product_id: str, df5m_with_end: pd.DataFrame):
    prof_dir = Path(cfg.model_dir) / "volume_profiles"
    ensure_dir(prof_dir)
    path = _volume_profile_path(cfg, product_id)

    now = dt.datetime.now(dt.timezone.utc)
    today = now.date()

    if df5m_with_end.empty:
        return {"slot_median_volume": None, "updated_at_utc": None}, "no_candles"

    df = df5m_with_end.copy()
    df["day"] = df["ts_end"].dt.date
    df_prior = df[df["day"] < today].copy()
    if df_prior.empty:
        return {"slot_median_volume": None, "updated_at_utc": None}, "insufficient_prior_days"

    days = sorted(df_prior["day"].unique())[-cfg.tod_rvol_lookback_days:]
    df_prior = df_prior[df_prior["day"].isin(days)]
    n_days = len(set(days))
    if n_days < cfg.tod_rvol_min_days:
        return {"slot_median_volume": None, "updated_at_utc": None}, f"insufficient_days({n_days})"

    df_prior["slot"] = df_prior["ts_end"].apply(_slot_5m)
    med = df_prior.groupby("slot")["volume"].median()
    slots = [float(med.get(i, np.nan)) for i in range(288)]
    obj = {"product_id": product_id, "slot_median_volume": slots, "n_days": n_days, "updated_at_utc": now.isoformat()}
    atomic_write_json(path, obj)
    return obj, "ok"

def load_volume_profile(cfg: Settings, product_id: str):
    return read_json(_volume_profile_path(cfg, product_id))

def compute_features_5m(cfg: Settings, df5m_raw: pd.DataFrame, product_id: str, benchmark_ret_30m: float = 0.0, for_training: bool = False):
    info = {"schema_version": SCHEMA_VERSION}
    if df5m_raw is None or df5m_raw.empty:
        return pd.DataFrame(), {**info, "error":"no_candles"}

    df = df5m_raw.copy()
    df["ts_start"] = pd.to_datetime(df["ts_start"], utc=True)
    df = df.sort_values("ts_start").drop_duplicates(subset=["ts_start"], keep="last").reset_index(drop=True)
    df["ts_end"] = df["ts_start"] + pd.Timedelta(minutes=5)

    now = dt.datetime.now(dt.timezone.utc)
    df = df[df["ts_end"] <= pd.Timestamp(now)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), {**info, "error":"no_completed_candles"}

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    ret_5m = close.pct_change().fillna(0.0)
    ret_30m = close.pct_change(6).fillna(0.0)

    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    ema_diff_pct = ((ema_fast - ema_slow) / (close + 1e-12)).fillna(0.0)
    adx = _adx(df, 14)

    atr = _atr(df, 14).fillna(method="bfill").fillna(0.0)
    atr_pct = (atr / (close + 1e-12)).fillna(0.0)
    logret = np.log((close + 1e-12) / (close.shift(1) + 1e-12)).replace([np.inf,-np.inf],0.0).fillna(0.0)
    rv = logret.rolling(24, min_periods=6).std().fillna(0.0)

    pv = (close*vol).rolling(24, min_periods=6).sum()
    vv = vol.rolling(24, min_periods=6).sum()
    vwap = (pv / (vv + 1e-12)).fillna(method="bfill").fillna(close)
    vwap_loc = ((close - vwap) / (atr + 1e-12)).fillna(0.0)

    recent_high = high.rolling(72, min_periods=6).max()
    donch_dist = ((recent_high - close) / (atr + 1e-12)).fillna(0.0)

    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction*vol).cumsum()
    obv_slope = _rolling_slope(obv, 24).fillna(0.0)

    roll_bars = max(3, int(cfg.liq_rolling_bars))
    dvol = (close*vol).fillna(0.0)
    med_dvol = dvol.rolling(roll_bars, min_periods=3).median().fillna(0.0)
    med_range_pct = ((high-low)/(close+1e-12)).rolling(roll_bars, min_periods=3).median().fillna(0.0)
    cur_range_pct = ((high-low)/(close+1e-12)).fillna(0.0)

    upper_wick = (high - np.maximum(open_, close)).fillna(0.0)
    lower_wick = (np.minimum(open_, close) - low).fillna(0.0)
    wickiness = ((upper_wick+lower_wick)/(atr+1e-12)).fillna(0.0)
    med_wick = wickiness.rolling(roll_bars, min_periods=3).median().fillna(0.0)

    tod_frac = df["ts_end"].apply(_tod_frac).astype(float)
    slot = df["ts_end"].apply(_slot_5m).astype(int)

    if for_training:
        base = vol.shift(1).rolling(roll_bars, min_periods=3).median()
        rvol_tod = (vol/(base+1e-12)).replace([np.inf,-np.inf], np.nan).fillna(1.0)
        profile_note = "training_fallback_rolling"
    else:
        prof = load_volume_profile(cfg, product_id)
        if prof and isinstance(prof.get("slot_median_volume"), list) and len(prof["slot_median_volume"])==288:
            base = np.array(prof["slot_median_volume"], dtype=float)
            cur_base = np.take(base, slot.to_numpy())
            rvol_tod = pd.Series(vol.to_numpy()/ (cur_base+1e-12), index=df.index).replace([np.inf,-np.inf], np.nan).fillna(1.0)
            profile_note = "profile:loaded"
        else:
            built, note = ensure_volume_profile_runtime(cfg, product_id, df)
            if built and isinstance(built.get("slot_median_volume"), list) and len(built["slot_median_volume"])==288:
                base = np.array(built["slot_median_volume"], dtype=float)
                cur_base = np.take(base, slot.to_numpy())
                rvol_tod = pd.Series(vol.to_numpy()/ (cur_base+1e-12), index=df.index).replace([np.inf,-np.inf], np.nan).fillna(1.0)
                profile_note = f"profile:built:{note}"
            else:
                base = vol.shift(1).rolling(roll_bars, min_periods=3).median()
                rvol_tod = (vol/(base+1e-12)).replace([np.inf,-np.inf], np.nan).fillna(1.0)
                profile_note = f"profile:fallback:{note}"

    if cfg.horizon_mode == "UTC_DAY_END":
        ts_end = df["ts_end"]
        day_end = (ts_end.dt.floor("D") + pd.Timedelta(hours=23, minutes=59))
        mins_rem = ((day_end - ts_end).dt.total_seconds()/60.0).clip(lower=0.0)
        mins_rem = np.minimum(mins_rem, float(cfg.horizon_minutes))
        time_remaining_minutes = mins_rem
        time_remaining_frac = (mins_rem/float(cfg.horizon_minutes)).clip(0.0,1.0)
    else:
        time_remaining_minutes = pd.Series(float(cfg.horizon_minutes), index=df.index)
        time_remaining_frac = pd.Series(1.0, index=df.index)

    log_minutes_remaining = np.log1p(time_remaining_minutes).replace([np.inf,-np.inf],0.0).fillna(0.0)

    i1 = ret_30m * (rvol_tod - 1.0)
    i2 = ret_30m * vwap_loc
    i3 = ema_diff_pct * adx
    i4 = ret_30m * time_remaining_frac

    feat = pd.DataFrame({
        "ts_end": df["ts_end"],
        "price": close,
        "vwap": vwap,
        "ret_5m": ret_5m,
        "ret_30m": ret_30m,
        "bench_ret_30m": float(benchmark_ret_30m),
        "ema_diff_pct": ema_diff_pct,
        "adx": adx,
        "atr_pct": atr_pct,
        "rv": rv,
        "rvol_tod": rvol_tod,
        "obv_slope": obv_slope,
        "vwap_loc": vwap_loc,
        "donch_dist": donch_dist,
        "time_remaining_frac": time_remaining_frac,
        "log_minutes_remaining": log_minutes_remaining,
        "tod_frac": tod_frac,
        "log1p_med_dvol": np.log1p(med_dvol),
        "med_range_pct": med_range_pct,
        "cur_range_pct": cur_range_pct,
        "med_wick_atr": med_wick,
        "cur_wick_atr": wickiness,
        "i_ret30_rvol": i1,
        "i_ret30_vwaploc": i2,
        "i_ema_adx": i3,
        "i_ret30_timefrac": i4,
    }).replace([np.inf,-np.inf],0.0).fillna(0.0)

    return feat, {**info, "rows": int(len(feat)), "profile_note": profile_note}

def liquidity_risk(cfg: Settings, row: pd.Series) -> Tuple[str, str]:
    reasons = []
    try:
        dvol = float(np.expm1(row.get("log1p_med_dvol",0.0)))
        if dvol < float(cfg.liq_dvol_min_usd):
            reasons.append("LOW_LIQ")
    except Exception:
        pass
    try:
        if float(row.get("cur_range_pct",0.0)) > float(cfg.liq_range_pct_max):
            reasons.append("WIDE_RANGE")
    except Exception:
        pass
    try:
        if float(row.get("cur_wick_atr",0.0)) > float(cfg.liq_wick_atr_max):
            reasons.append("WICKY")
    except Exception:
        pass
    if len(reasons) >= 2:
        return "HIGH", ",".join(reasons)
    if len(reasons) == 1:
        return "CAUTION", ",".join(reasons)
    return "OK", ""
