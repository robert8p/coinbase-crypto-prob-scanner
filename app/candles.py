import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd

from .storage import ensure_dir, read_pickle, write_pickle, safe_filename
from .coinbase_client import CoinbaseClient

def _parse_candles(data):
    rows = []
    for row in data:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        t, low, high, open_, close, vol = row[:6]
        try:
            ts = dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc)
            rows.append((ts, float(open_), float(high), float(low), float(close), float(vol)))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["ts_start","open","high","low","close","volume"]) if rows else pd.DataFrame(columns=["ts_start","open","high","low","close","volume"])
    if df.empty:
        return df
    df["ts_start"] = pd.to_datetime(df["ts_start"], utc=True)
    return df.sort_values("ts_start").drop_duplicates(subset=["ts_start"], keep="last").reset_index(drop=True)

def cache_path(model_dir: str, product_id: str, granularity_sec: int) -> Path:
    return Path(model_dir) / "cache" / "candles" / f"g{granularity_sec}" / f"{safe_filename(product_id)}.pkl"

def load_cached(model_dir: str, product_id: str, granularity_sec: int) -> Optional[pd.DataFrame]:
    obj = read_pickle(cache_path(model_dir, product_id, granularity_sec))
    return obj if isinstance(obj, pd.DataFrame) else None

def save_cached(model_dir: str, product_id: str, granularity_sec: int, df: pd.DataFrame) -> None:
    p = cache_path(model_dir, product_id, granularity_sec)
    ensure_dir(p.parent)
    write_pickle(p, df)

def _chunk_ranges(start: dt.datetime, end: dt.datetime, granularity_sec: int, max_points: int = 300):
    chunk = dt.timedelta(seconds=granularity_sec * max_points)
    cur = start
    while cur < end:
        nxt = min(end, cur + chunk)
        yield cur, nxt
        cur = nxt

async def get_candles_incremental(cb: CoinbaseClient, model_dir: str, product_id: str, granularity_sec: int,
                                 start: dt.datetime, end: dt.datetime, overlap_points: int = 2) -> pd.DataFrame:
    start = start.replace(tzinfo=dt.timezone.utc)
    end = end.replace(tzinfo=dt.timezone.utc)
    cached = load_cached(model_dir, product_id, granularity_sec)

    if cached is not None and not cached.empty:
        last_ts = pd.to_datetime(cached["ts_start"], utc=True).max().to_pydatetime()
        fetch_start = max(start, last_ts - dt.timedelta(seconds=granularity_sec*overlap_points))
        frames = [cached]
    else:
        fetch_start = start
        frames = []

    for cstart, cend in _chunk_ranges(fetch_start, end, granularity_sec):
        data = await cb.get_candles(product_id, granularity_sec, cstart, cend)
        df = _parse_candles(data)
        if not df.empty:
            frames.append(df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts_start","open","high","low","close","volume"])
    if not out.empty:
        out["ts_start"] = pd.to_datetime(out["ts_start"], utc=True)
        out = out.sort_values("ts_start").drop_duplicates(subset=["ts_start"], keep="last").reset_index(drop=True)
    save_cached(model_dir, product_id, granularity_sec, out)
    return out
