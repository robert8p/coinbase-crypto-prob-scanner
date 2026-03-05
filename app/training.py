import asyncio
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression

from .config import Settings
from .coinbase_client import CoinbaseClient
from .candles import get_candles_incremental
from .features import compute_features_5m, ensure_volume_profile_runtime, SCHEMA_VERSION
from .storage import ensure_dir, atomic_write_json

try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None


PAUSE_FILE = "pause_scans.flag"

def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p-y)**2))

def _bucket_id(frac: float) -> int:
    x = float(np.clip(frac, 0.0, 1.0))
    b = int(np.floor(x*4.0))
    return int(min(3, max(0, b if b < 4 else 3)))

def _future_window_max(highs: np.ndarray, window: int) -> np.ndarray:
    from collections import deque
    n = len(highs)
    out = np.full(n, np.nan, dtype=float)
    dq = deque()
    for i in range(n):
        while dq and highs[dq[-1]] <= highs[i]:
            dq.pop()
        dq.append(i)
        while dq and dq[0] <= i-window:
            dq.popleft()
        start = i-window+1
        if start >= 0:
            out[start] = highs[dq[0]]
    last_max = -np.inf
    for i in range(n-1, -1, -1):
        last_max = max(last_max, highs[i])
        if np.isnan(out[i]):
            out[i] = last_max
    return out

def build_labels(df1m: pd.DataFrame, ts_end_5m: np.ndarray, horizon_minutes: int, p0: np.ndarray, pct: float) -> np.ndarray:
    ts1 = pd.to_datetime(df1m["ts_start"], utc=True).to_numpy(dtype="datetime64[ns]")
    highs = df1m["high"].to_numpy(dtype=float)
    max_from = _future_window_max(highs, window=int(horizon_minutes))
    idx = np.searchsorted(ts1, ts_end_5m)
    idx = np.clip(idx, 0, len(ts1)-1)
    h_future = max_from[idx]
    return (h_future >= (1.0+pct)*p0).astype(int)

def fit_platt(raw_p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    X = np.log(np.clip(raw_p,1e-6,1-1e-6)/np.clip(1-raw_p,1e-6,1-1e-6)).reshape(-1,1)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X, y)
    return lr

def fit_isotonic(raw_p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_p, y)
    return iso

def make_pipeline(feature_names: List[str]) -> Pipeline:
    spline_cols = [c for c in ["time_remaining_frac","rvol_tod","vwap_loc","donch_dist"] if c in feature_names]
    other_cols = [c for c in feature_names if c not in spline_cols]
    pre = ColumnTransformer(
        transformers=[
            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False), spline_cols),
            ("num", "passthrough", other_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    clf = LogisticRegression(solver="saga", penalty="elasticnet", max_iter=4000, n_jobs=1)
    return Pipeline([("pre", pre), ("scaler", StandardScaler(with_mean=False)), ("clf", clf)])

def time_splits(n: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    idx = np.arange(n)
    n_train = int(n*0.6)
    n_cal = int(n*0.2)
    return idx[:n_train], idx[n_train:n_train+n_cal], idx[n_train+n_cal:]

def _persist_status(model_dir: str, status: dict) -> None:
    try:
        ensure_dir(Path(model_dir))
        atomic_write_json(Path(model_dir) / "training_status.json", status)
    except Exception:
        pass


class TrainingManager:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._task: asyncio.Task | None = None
        self._override_max_products: Optional[int] = None
        self._override_lookback_days: Optional[int] = None
        self.status: dict = {"running": False, "started_at_utc": None, "finished_at_utc": None, "last_result": None, "last_error": None, "progress": None}

    def is_running(self) -> bool:
        return bool(self.status.get("running"))

    def start(self, cb: CoinbaseClient, products: List[Dict[str,str]], *, max_products: int | None = None, lookback_days: int | None = None) -> None:
        if self.is_running():
            return
        self._override_max_products = max_products
        self._override_lookback_days = lookback_days
        self.status.update({"running": True, "started_at_utc": _now_utc(), "finished_at_utc": None, "last_result": None, "last_error": None, "progress":"starting"})
        _persist_status(self.cfg.model_dir, self.status)
        self._task = asyncio.create_task(self._run(cb, products))

    async def _run(self, cb: CoinbaseClient, products: List[Dict[str,str]]) -> None:
        pause_path = Path(self.cfg.model_dir) / PAUSE_FILE
        lock_path = Path(self.cfg.model_dir) / "training.lock"
        lock_handle = None
        try:
            ensure_dir(Path(self.cfg.model_dir))
            # Cross-worker training lock (prevents duplicate training when using multiple uvicorn workers)
            lock_handle = lock_path.open("a+")
            if fcntl is not None:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except Exception:
                    try:
                        lock_handle.close()
                    except Exception:
                        pass
                    self.status.update({"running": False, "finished_at_utc": _now_utc(), "last_error": "training_lock_not_acquired", "progress":"error"})
                    _persist_status(self.cfg.model_dir, self.status)
                    return
            pause_path.write_text("training", encoding="utf-8")
            _persist_status(self.cfg.model_dir, self.status)

            cfg = self.cfg
            max_products = self._override_max_products if self._override_max_products is not None else cfg.train_max_products
            lookback_days = self._override_lookback_days if self._override_lookback_days is not None else cfg.train_lookback_days

            prods = products
            # TRAIN_MAX_PRODUCTS / AUTO_TRAIN_MAX_PRODUCTS semantics:
            # - <=0 means "no cap" (train all products provided)
            # - >0 caps to that many products
            if max_products and int(max_products) > 0:
                prods = prods[: int(max_products)]
            else:
                max_products = 0
            if not prods:
                raise RuntimeError("no_products_to_train")

            end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
            start = end - dt.timedelta(days=int(lookback_days)+2)

            bench5 = await get_candles_incremental(cb, cfg.model_dir, cfg.benchmark_symbol, 300, start, end)
            bench_feat, _ = await asyncio.to_thread(compute_features_5m, cfg, bench5, cfg.benchmark_symbol, 0.0, True)
            bench_ret = bench_feat[["ts_end","ret_30m"]].rename(columns={"ret_30m":"bench_ret_30m"}) if not bench_feat.empty else None

            all_rows = []
            per_prod = {}
            for i, p in enumerate(prods):
                pid = p["id"]
                self.status["progress"] = f"download {i+1}/{len(prods)} {pid}"
                _persist_status(self.cfg.model_dir, self.status)

                df5 = await get_candles_incremental(cb, cfg.model_dir, pid, 300, start, end)
                df1 = await get_candles_incremental(cb, cfg.model_dir, pid, 60, start, end)
                per_prod[pid] = {"candles5_rows": int(len(df5)), "candles1_rows": int(len(df1))}
                if df5.empty or df1.empty:
                    continue

                feat, _ = await asyncio.to_thread(compute_features_5m, cfg, df5, pid, 0.0, True)
                if feat.empty or len(feat) < max(30, int(cfg.min_bars_5m)):
                    continue

                try:
                    df5tmp = df5.copy()
                    df5tmp["ts_start"] = pd.to_datetime(df5tmp["ts_start"], utc=True)
                    df5tmp["ts_end"] = df5tmp["ts_start"] + pd.Timedelta(minutes=5)
                    ensure_volume_profile_runtime(cfg, pid, df5tmp)
                except Exception:
                    pass

                if bench_ret is not None:
                    feat = feat.merge(bench_ret, on="ts_end", how="left", suffixes=("", "_bench"))
                    if "bench_ret_30m_bench" in feat.columns:
                        feat["bench_ret_30m"] = feat["bench_ret_30m_bench"].fillna(feat.get("bench_ret_30m", 0.0)).fillna(0.0)
                        feat = feat.drop(columns=["bench_ret_30m_bench"], errors="ignore")
                    else:
                        feat["bench_ret_30m"] = feat.get("bench_ret_30m", 0.0)

                df1 = df1.copy()
                df1["ts_start"] = pd.to_datetime(df1["ts_start"], utc=True)
                df1 = df1.sort_values("ts_start").drop_duplicates(subset=["ts_start"], keep="last").reset_index(drop=True)

                ts_end = pd.to_datetime(feat["ts_end"], utc=True).to_numpy(dtype="datetime64[ns]")
                p0 = feat["price"].to_numpy(dtype=float)
                y1 = await asyncio.to_thread(build_labels, df1, ts_end, int(cfg.horizon_minutes), p0, 0.03)
                y2 = await asyncio.to_thread(build_labels, df1, ts_end, int(cfg.horizon_minutes), p0, 0.05)

                feat["product_id"] = pid
                feat["y1"] = y1
                feat["y2"] = y2
                feat = feat.iloc[max(30, int(cfg.min_bars_5m)):]

                all_rows.append(feat)

            if not all_rows:
                raise RuntimeError("no_training_rows")

            data = pd.concat(all_rows, ignore_index=True).sort_values("ts_end").reset_index(drop=True)
            data = data.replace([np.inf,-np.inf],0.0).fillna(0.0)

            drop_cols = {"ts_end","product_id","y1","y2"}
            feature_names = [c for c in data.columns if c not in drop_cols]

            n = len(data)
            train_idx, cal_idx, val_idx = time_splits(n)

            
            # CPU-heavy model fitting/calibration runs in a worker thread so /health stays responsive on Render.
            def _train_models_cpu(cfg, data, feature_names, train_idx, cal_idx, val_idx):
                import numpy as np
                from sklearn.metrics import roc_auc_score

                def calibrate(raw_cal, y_cal, raw_val, y_val, bucket_cal, bucket_val):
                    cals = {}
                    for b in range(4):
                        mc = bucket_cal==b
                        mv = bucket_val==b
                        if mc.sum()<20 or mv.sum()<20:
                            cals[str(b)] = {"type":"identity","obj":None,"note":"few_samples"}
                            continue
                        if len(np.unique(y_cal[mc]))<2 or len(np.unique(y_val[mv]))<2:
                            cals[str(b)] = {"type":"identity","obj":None,"note":"single_class"}
                            continue
                        platt = fit_platt(raw_cal[mc], y_cal[mc])
                        pv = platt.predict_proba(np.log(np.clip(raw_val[mv],1e-6,1-1e-6)/np.clip(1-raw_val[mv],1e-6,1-1e-6)).reshape(-1,1))[:,1]
                        best_type, best_obj, best_brier = "platt", platt, _brier(y_val[mv], pv)
                        if mc.sum() >= int(cfg.calib_min_bucket_samples):
                            iso = fit_isotonic(raw_cal[mc], y_cal[mc])
                            pv2 = iso.predict(raw_val[mv])
                            b2 = _brier(y_val[mv], pv2)
                            if b2 + 1e-9 < best_brier:
                                best_type, best_obj, best_brier = "isotonic", iso, b2
                        cals[str(b)] = {"type":best_type,"obj":best_obj,"val_brier":float(best_brier),"n_cal":int(mc.sum()),"n_val":int(mv.sum())}
                    return cals

                def apply_cal(cals, raw, buckets):
                    out = np.copy(raw)
                    for b in range(4):
                        m = buckets==b
                        if not np.any(m):
                            continue
                        entry = cals.get(str(b))
                        if not entry:
                            continue
                        typ, obj = entry.get("type"), entry.get("obj")
                        if typ=="platt" and obj is not None:
                            out[m] = obj.predict_proba(np.log(np.clip(raw[m],1e-6,1-1e-6)/np.clip(1-raw[m],1e-6,1-1e-6)).reshape(-1,1))[:,1]
                        elif typ=="isotonic" and obj is not None:
                            out[m] = obj.predict(raw[m])
                    return np.clip(out,0.0,1.0)

                def select_alpha(alphas, p_cal, y, priors):
                    best_a, best_b = float(alphas[0]) if alphas else 0.8, 1e9
                    for a in alphas:
                        b = _brier(y, a*p_cal + (1.0-a)*priors)
                        if b < best_b:
                            best_b, best_a = b, float(a)
                    return best_a, float(best_b)

                def train_one(target: str, label: str) -> dict:
                    y = data[target].to_numpy(dtype=int)
                    X = data[feature_names].copy()
                    best_bundle = None
                    best_brier = 1e9
                    best_params = None

                    for C in cfg.enet_c_values:
                        for l1r in cfg.enet_l1_values:
                            pipe = make_pipeline(feature_names)
                            pipe.named_steps["clf"].set_params(C=float(C), l1_ratio=float(l1r))
                            pipe.fit(X.iloc[train_idx], y[train_idx])

                            raw_cal = pipe.predict_proba(X.iloc[cal_idx])[:,1]
                            raw_val = pipe.predict_proba(X.iloc[val_idx])[:,1]

                            bcal = np.array([_bucket_id(x) for x in X.iloc[cal_idx]["time_remaining_frac"].to_numpy(dtype=float)], dtype=int)
                            bval = np.array([_bucket_id(x) for x in X.iloc[val_idx]["time_remaining_frac"].to_numpy(dtype=float)], dtype=int)

                            cals = calibrate(raw_cal, y[cal_idx], raw_val, y[val_idx], bcal, bval)
                            p_cal = apply_cal(cals, raw_val, bval)

                            pri = {}
                            m = np.concatenate([train_idx, cal_idx])
                            for b in range(4):
                                mb = np.array([_bucket_id(x) for x in X.iloc[m]["time_remaining_frac"].to_numpy(dtype=float)])==b
                                pri[str(b)] = float(np.mean(y[m][mb])) if mb.sum()>0 else 0.5
                            p_prior = np.array([pri[str(int(b))] for b in bval], dtype=float)

                            alpha, brier = select_alpha(cfg.prior_alpha_values, p_cal, y[val_idx], p_prior)

                            if brier < best_brier:
                                best_brier = brier
                                best_params = {"C":float(C),"l1_ratio":float(l1r),"alpha":float(alpha)}
                                best_bundle = {
                                    "schema_version": SCHEMA_VERSION,
                                    "feature_names": feature_names,
                                    "pipeline": pipe,
                                    "calibrators": cals,
                                    "bucket_priors": pri,
                                    "prior_blend_alpha": float(alpha),
                                }

                    assert best_bundle is not None and best_params is not None

                    Xv = data[feature_names].iloc[val_idx]
                    raw = best_bundle["pipeline"].predict_proba(Xv)[:,1]
                    bval = np.array([_bucket_id(x) for x in data.iloc[val_idx]["time_remaining_frac"].to_numpy(dtype=float)], dtype=int)
                    p_cal = apply_cal(best_bundle["calibrators"], raw, bval)
                    p_prior = np.array([best_bundle["bucket_priors"][str(int(b))] for b in bval], dtype=float)
                    p_final = best_bundle["prior_blend_alpha"]*p_cal + (1.0-best_bundle["prior_blend_alpha"])*p_prior
                    p_final = np.clip(p_final,0.0,1.0)

                    auc = None
                    try:
                        auc = float(roc_auc_score(y[val_idx], p_final))
                    except Exception:
                        auc = None
                    best_bundle["metrics"] = {"auc_val": auc, "brier_val": float(_brier(y[val_idx], p_final)), "best_params": best_params, "target": label}
                    return best_bundle

                b1 = train_one("y1", "3%")
                b2 = train_one("y2", "5%")
                return b1, b2

            self.status["progress"] = "training pt1+pt2 (CPU thread)"
            _persist_status(self.cfg.model_dir, self.status)
            b1, b2 = await asyncio.to_thread(_train_models_cpu, cfg, data, feature_names, train_idx, cal_idx, val_idx)

            for pt, bundle in [("pt1", b1), ("pt2", b2)]:
                out_dir = ensure_dir(Path(cfg.model_dir) / pt)
                joblib.dump(bundle, out_dir / "bundle.joblib")

            result = {
                "trained_products": len(prods),
                "rows": int(len(data)),
                "pt1_metrics": b1.get("metrics"),
                "pt2_metrics": b2.get("metrics"),
                "schema_version": SCHEMA_VERSION,
                "lookback_days": int(lookback_days),
                "max_products": int(max_products),
            }
            atomic_write_json(Path(cfg.model_dir) / "last_training_result.json", {"finished_at_utc": _now_utc(), "result": result, "per_product": per_prod})

            self.status.update({"running": False, "finished_at_utc": _now_utc(), "last_result": result, "last_error": None, "progress":"done"})
            _persist_status(self.cfg.model_dir, self.status)
        except Exception as e:
            self.status.update({"running": False, "finished_at_utc": _now_utc(), "last_error": f"{type(e).__name__}: {e}", "progress":"error"})
            _persist_status(self.cfg.model_dir, self.status)
        finally:
            try:
                if pause_path.exists():
                    pause_path.unlink()
            except Exception:
                pass
