import asyncio
import datetime as dt
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .config import Settings
from .coinbase_client import CoinbaseClient
from .universe import UniverseManager
from .candles import get_candles_incremental
from .features import compute_features_5m, liquidity_risk
from .heuristic import score_heuristic
from .model import load_model, predict_proba
from .storage import ensure_dir, atomic_write_json

PAUSE_FILE = "pause_scans.flag"

try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None

def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _next_aligned(now: dt.datetime, interval_minutes: int) -> dt.datetime:
    interval_minutes = max(1, int(interval_minutes))
    base = now.replace(second=0, microsecond=0)
    mins = base.hour*60 + base.minute
    next_mins = ((mins//interval_minutes)+1)*interval_minutes
    return base + dt.timedelta(minutes=(next_mins - mins))

def _acquire_lock(lock_path: Path) -> Tuple[bool, object | None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open("a+")
    if fcntl is None:
        return True, f
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True, f
    except Exception:
        try:
            f.close()
        except Exception:
            pass
        return False, None

class ScanState:
    def __init__(self):
        self.scan_running: bool = False
        self.last_run_utc: str | None = None
        self.last_error: str | None = None
        self.last_candle_timestamp: str | None = None
        self.rows: List[dict] = []
        self.coverage: dict = {}
        self.model_notes: dict = {}
        self._skipped: List[dict] = []
        self._lock_handle = None

    def debug_skipped(self, limit: int = 200) -> List[dict]:
        return list(self._skipped[:limit])

async def scan_once(cfg: Settings, cb: CoinbaseClient, universe_mgr: UniverseManager, state: ScanState) -> None:
    if state.scan_running:
        return
    if (Path(cfg.model_dir) / PAUSE_FILE).exists():
        return

    # Cross-worker scan lock (prevents multiple workers doing the same heavy scan)
    lock_ok, scan_handle = _acquire_lock(Path(cfg.model_dir) / "scan.lock")
    if not lock_ok:
        return

    state.scan_running = True
    try:
        scan_time = _now().replace(second=0, microsecond=0)
        prods, uni_meta = await universe_mgr.resolve_universe(cb)

        bench_ret_30m = 0.0
        try:
            bench5 = await get_candles_incremental(cb, cfg.model_dir, cfg.benchmark_symbol, 300, scan_time - dt.timedelta(hours=12), scan_time)
            bench_feat, _ = compute_features_5m(cfg, bench5, cfg.benchmark_symbol, benchmark_ret_30m=0.0, for_training=False)
            if not bench_feat.empty:
                bench_ret_30m = float(bench_feat.iloc[-1]["ret_30m"])
        except Exception:
            bench_ret_30m = 0.0

        pt1 = load_model(cfg.model_dir, "pt1")
        pt2 = load_model(cfg.model_dir, "pt2")
        use_models = pt1.ok and pt2.ok

        rows = []
        skipped = []
        skip_counts: dict[str,int] = {}
        returned = 0
        sufficient = 0
        scored = 0

        for p in prods:
            pid = p["id"]
            try:
                df5 = await get_candles_incremental(cb, cfg.model_dir, pid, 300, scan_time - dt.timedelta(hours=48), scan_time)
                if df5.empty:
                    skip_counts["no_candles"] = skip_counts.get("no_candles",0)+1
                    skipped.append({"product":pid,"reason":"no_candles","last_candle":None})
                    continue
                returned += 1

                feat, _ = compute_features_5m(cfg, df5, pid, benchmark_ret_30m=bench_ret_30m, for_training=False)
                if feat.empty or len(feat) < int(cfg.min_bars_5m):
                    skip_counts["insufficient_candles"] = skip_counts.get("insufficient_candles",0)+1
                    last_ts = str(pd.to_datetime(feat.iloc[-1]["ts_end"], utc=True)) if not feat.empty else None
                    skipped.append({"product":pid,"reason":"insufficient_candles","last_candle":last_ts})
                    continue
                sufficient += 1

                last = feat.iloc[-1]
                last_ts = pd.to_datetime(last["ts_end"], utc=True).to_pydatetime()
                staleness_min = (scan_time - last_ts).total_seconds()/60.0
                if staleness_min > float(cfg.max_candle_staleness_minutes):
                    skip_counts["stale_candles"] = skip_counts.get("stale_candles",0)+1
                    skipped.append({"product":pid,"reason":"stale_candles","last_candle":last_ts.isoformat()})
                    rows.append({
                        "product": pid, "display_symbol": pid.replace("-","/"),
                        "price": float(last.get("price",0.0)), "vwap": float(last.get("vwap",0.0)),
                        "risk": "N/A", "risk_reasons": "stale_candles",
                        "prob_1": None, "prob_2": None, "prob_1_source":"skipped", "prob_2_source":"skipped",
                        "quote": p.get("quote",""), "category": p.get("status","spot"),
                        "reasons":"stale_candles", "last_candle_time": last_ts.isoformat(),
                        "included": False, "skip_reason":"stale_candles",
                    })
                    continue

                price = float(last.get("price",0.0))
                vwap = float(last.get("vwap",0.0))
                if price <= 0 or vwap <= 0:
                    skip_counts["missing_price_or_vwap"] = skip_counts.get("missing_price_or_vwap",0)+1
                    skipped.append({"product":pid,"reason":"missing_price_or_vwap","last_candle":last_ts.isoformat()})
                    continue

                risk, risk_reasons = liquidity_risk(cfg, last)

                src = "heuristic"
                if use_models:
                    try:
                        X = feat.tail(1).copy()
                        p1,_ = predict_proba(pt1.bundle, X)  # type: ignore
                        p2,_ = predict_proba(pt2.bundle, X)  # type: ignore
                        prob_1, prob_2 = float(p1[0]), float(p2[0])
                        src = "model"
                    except Exception:
                        skip_counts["model_schema_incompatible"] = skip_counts.get("model_schema_incompatible",0)+1
                        h = score_heuristic(last)
                        prob_1, prob_2 = float(h["prob_1"]), float(h["prob_2"])
                        src = "heuristic(schema_mismatch)"
                else:
                    h = score_heuristic(last)
                    prob_1, prob_2 = float(h["prob_1"]), float(h["prob_2"])

                scored += 1
                rows.append({
                    "product": pid, "display_symbol": pid.replace("-","/"),
                    "price": price, "vwap": vwap,
                    "risk": risk, "risk_reasons": risk_reasons,
                    "prob_1": prob_1, "prob_2": prob_2,
                    "prob_1_source": src, "prob_2_source": src,
                    "quote": p.get("quote",""), "category": p.get("status","spot"),
                    "reasons": risk_reasons, "last_candle_time": last_ts.isoformat(),
                    "included": True, "skip_reason": None,
                })
            except Exception as e:
                msg = str(e)
                key = "rate_limited" if "http_429" in msg else "other_errors"
                skip_counts[key] = skip_counts.get(key,0)+1
                skipped.append({"product":pid,"reason":f"{key}:{type(e).__name__}","last_candle":None})
                continue

        rows_sorted = sorted(rows, key=lambda r: (-(r["prob_2"] if isinstance(r.get("prob_2"), (int,float)) else -1), -(r["prob_1"] if isinstance(r.get("prob_1"), (int,float)) else -1), r.get("product","")))
        last_candle = None
        for r in rows_sorted:
            if r.get("last_candle_time"):
                last_candle = r["last_candle_time"]
                break

        state.last_run_utc = scan_time.isoformat()
        state.last_error = None
        state.last_candle_timestamp = last_candle
        state.rows = rows_sorted
        state._skipped = skipped[:]
        state.coverage = {
            "universe_count": len(prods),
            "products_requested_count": len(prods),
            "products_returned_with_candles_count": returned,
            "products_with_sufficient_candles_count": sufficient,
            "products_scored_count": scored,
            "top_skip_reasons": dict(sorted(skip_counts.items(), key=lambda kv: -kv[1])),
            "last_run_utc": state.last_run_utc,
            "last_candle_timestamp": last_candle,
            "universe_meta": uni_meta,
        }
        state.model_notes = {
            "using": "model" if use_models else "heuristic",
            "pt1": {"ok": pt1.ok, "reason": pt1.reason, "target": "3%"},
            "pt2": {"ok": pt2.ok, "reason": pt2.reason, "target": "5%"},
            "schema_warning": None if use_models else "Model missing or incompatible; heuristic fallback in use.",
        }

        try:
            ensure_dir(Path(cfg.model_dir))
            atomic_write_json(Path(cfg.model_dir) / "last_scores_meta.json", {"last_run_utc": state.last_run_utc, "coverage": state.coverage, "model": state.model_notes, "rows_count": len(state.rows)})
            # Persist rows so other workers can serve /api/scores reliably
            atomic_write_json(Path(cfg.model_dir) / "last_scores.json", {"last_run_utc": state.last_run_utc, "rows": state.rows})
        except Exception:
            pass
    finally:
        try:
            if scan_handle is not None:
                scan_handle.close()
        except Exception:
            pass
        state.scan_running = False

async def scheduler_loop(cfg: Settings, cb: CoinbaseClient, universe_mgr: UniverseManager, state: ScanState) -> None:
    while True:
        now = _now()
        nxt = _next_aligned(now, int(cfg.scan_interval_minutes))
        await asyncio.sleep(max(0.0, (nxt-now).total_seconds()))
        try:
            await scan_once(cfg, cb, universe_mgr, state)
        except Exception as e:
            state.last_error = f"{type(e).__name__}: {e}"

def try_start_scheduler(cfg: Settings, cb: CoinbaseClient, universe_mgr: UniverseManager, state: ScanState) -> bool:
    if cfg.disable_scheduler:
        return False
    lock_path = Path(cfg.model_dir) / "scheduler.lock"
    ok, handle = _acquire_lock(lock_path)
    if not ok:
        return False
    state._lock_handle = handle
    asyncio.create_task(scheduler_loop(cfg, cb, universe_mgr, state))
    return True
