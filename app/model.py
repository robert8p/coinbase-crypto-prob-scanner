import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from .features import SCHEMA_VERSION

@dataclass
class LoadedModel:
    ok: bool
    reason: str
    bundle: dict | None = None

def model_path(model_dir: str, pt: str) -> Path:
    return Path(model_dir) / pt / "bundle.joblib"

def load_model(model_dir: str, pt: str) -> LoadedModel:
    p = model_path(model_dir, pt)
    if not p.exists():
        return LoadedModel(False, "missing")
    try:
        bundle = joblib.load(p)
    except Exception as e:
        return LoadedModel(False, f"load_error:{type(e).__name__}")
    if not isinstance(bundle, dict):
        return LoadedModel(False, "invalid_bundle")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        return LoadedModel(False, "schema_version_mismatch", bundle=bundle)
    if not isinstance(bundle.get("feature_names"), list):
        return LoadedModel(False, "missing_feature_names", bundle=bundle)
    return LoadedModel(True, "ok", bundle=bundle)

def _bucket_id(frac: float) -> int:
    x = float(np.clip(frac, 0.0, 1.0))
    b = int(math.floor(x*4.0))
    return min(3, max(0, b if b < 4 else 3))

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def _apply_calibrator(bundle: dict, raw_p: np.ndarray, buckets: np.ndarray) -> np.ndarray:
    out = np.copy(raw_p)
    cals = bundle.get("calibrators", {}) or {}
    for b in range(4):
        m = buckets == b
        if not np.any(m):
            continue
        entry = cals.get(str(b)) or cals.get(b)
        if not entry:
            continue
        typ = entry.get("type")
        obj = entry.get("obj")
        if typ == "platt" and obj is not None:
            out[m] = obj.predict_proba(_logit(raw_p[m]).reshape(-1,1))[:,1]
        elif typ == "isotonic" and obj is not None:
            out[m] = obj.predict(raw_p[m])
        else:
            out[m] = raw_p[m]
    return np.clip(out, 0.0, 1.0)

def predict_proba(bundle: dict, X: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    feat_names = bundle["feature_names"]
    missing = [c for c in feat_names if c not in X.columns]
    if missing:
        raise ValueError(f"missing_features:{missing[:5]}")
    Xin = X[feat_names].copy()
    pipe = bundle["pipeline"]
    raw = pipe.predict_proba(Xin)[:,1]
    buckets = np.array([_bucket_id(x) for x in X["time_remaining_frac"].to_numpy(dtype=float)], dtype=int)
    p_cal = _apply_calibrator(bundle, raw, buckets)
    priors = bundle.get("bucket_priors", {}) or {}
    p_prior = np.array([float(priors.get(str(int(b)), 0.5)) for b in buckets], dtype=float)
    alpha = float(bundle.get("prior_blend_alpha", 0.8))
    p_final = alpha * p_cal + (1.0-alpha) * p_prior
    return np.clip(p_final, 0.0, 1.0), {"alpha": alpha}
