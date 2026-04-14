from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .demo_data import STABLES
from .features import FEATURE_COLUMNS
from .persist import ensure_dir
from .live_scoring import simulate_live_post_model_adjustments
from .version import APP_VERSION

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

EMBARGO_BUFFER_MINUTES = 30  # added on top of horizon for embargo gap


def _embargo_minutes(horizon_minutes: int = 240) -> int:
    """Derive embargo from the prediction horizon + safety buffer."""
    return horizon_minutes + EMBARGO_BUFFER_MINUTES


# ── Wilson confidence bound ───────────────────────────────────────────────

def _wilson_lower(hits: int, total: int, z: float = 1.645) -> float:
    """Wilson score interval lower bound (90% confidence by default).

    With 20 samples at 75% precision, the lower bound is ~0.56.
    This prevents model selection from over-trusting small lucky tails.
    """
    if total == 0:
        return 0.0
    p = hits / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - spread) / denom)


# ── ModelBundle ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class ModelBundle:
    pipeline: object
    calibrator: IsotonicRegression | None
    metadata: dict
    model_type: str

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = sanitize_feature_frame(df)
        base = self.pipeline.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            return np.clip(self.calibrator.predict(base), 0.0, 1.0)
        return np.clip(base, 0.0, 1.0)

    def save(self, path: str) -> None:
        ensure_dir(Path(path).parent)
        joblib.dump({
            "pipeline": self.pipeline,
            "calibrator": self.calibrator,
            "metadata": self.metadata,
            "model_type": self.model_type,
        }, path)

    @staticmethod
    def load(path: str) -> "ModelBundle | None":
        p = Path(path)
        if not p.exists():
            return None
        payload = joblib.load(path)
        return ModelBundle(
            pipeline=payload["pipeline"],
            calibrator=payload.get("calibrator"),
            metadata=payload["metadata"],
            model_type=payload.get("model_type", "logistic"),
        )


def _validate_calibrator(calibrator: IsotonicRegression, raw_preds: np.ndarray, min_spread: float = 0.04) -> IsotonicRegression | None:
    calibrated = np.clip(calibrator.predict(raw_preds), 0.0, 1.0)
    raw_std = float(np.std(raw_preds))
    cal_std = float(np.std(calibrated))
    if cal_std < min_spread and raw_std > cal_std * 1.5:
        logger.info("calibrator_skipped reason=spread_collapsed raw_std=%.4f cal_std=%.4f", raw_std, cal_std)
        return None
    return calibrator


def sanitize_feature_frame(df: pd.DataFrame, feature_columns: list[str] | None = None, *, dtype: str = "float32") -> pd.DataFrame:
    """Return a clean feature frame with stable names/order and lower memory usage.

    Preserves column names for estimators fitted with named features, reindexes to
    the expected training order, replaces inf/nan, and downcasts to float32 to
    reduce memory pressure without changing the model logic.
    """
    cols = feature_columns or FEATURE_COLUMNS
    X = (
        df.reindex(columns=cols)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if dtype:
        try:
            X = X.astype(dtype, copy=False)
        except Exception:
            pass
    return X


# ── Purged split ──────────────────────────────────────────────────────────

def _purged_time_split(training_df: pd.DataFrame, embargo_minutes: int | None = None):
    if embargo_minutes is None:
        embargo_minutes = _embargo_minutes()
    ts = pd.to_datetime(training_df["ts"], utc=True)
    embargo = pd.Timedelta(minutes=embargo_minutes)
    q_train_end = ts.quantile(0.65)
    q_val_end = ts.quantile(0.82)
    train_mask = ts <= q_train_end
    val_mask = (ts > q_train_end + embargo) & (ts <= q_val_end)
    test_mask = ts > q_val_end + embargo
    df_train = training_df.loc[train_mask].copy()
    df_val = training_df.loc[val_mask].copy()
    df_test = training_df.loc[test_mask].copy()
    embargo_dropped = len(training_df) - len(df_train) - len(df_val) - len(df_test)
    if min(len(df_train), len(df_val), len(df_test)) < 80:
        n = len(training_df)
        gap = max(1, embargo_minutes // 5)
        i1 = int(n * 0.65)
        i2s = min(i1 + gap, n - 100)
        i2e = int(n * 0.82)
        i3s = min(i2e + gap, n - 50)
        df_train = training_df.iloc[:i1].copy()
        df_val = training_df.iloc[i2s:i2e].copy()
        df_test = training_df.iloc[i3s:].copy()
        embargo_dropped = n - len(df_train) - len(df_val) - len(df_test)
    return df_train, df_val, df_test, embargo_dropped


# ── Adjusted-score simulation ─────────────────────────────────────────────

def _simulate_adjusted_scores(pred_model: np.ndarray, df: pd.DataFrame, cfg_dict: dict) -> tuple[np.ndarray, dict]:
    """Simulate the scanner's post-model adjustments on historical data.

    Uses the same panic/Binance penalty formulas as the live scanner where the
    historical frame contains the necessary columns. Guardrail capping remains a
    best-effort replay unless explicit capped flags are present, and sector
    penalties remain approximate/offline-limited.
    """
    return simulate_live_post_model_adjustments(pred_model, df, cfg_dict or {})


# ── Temporal stability ────────────────────────────────────────────────────

def _temporal_stability(df_test: pd.DataFrame, pred_test: np.ndarray, n_windows: int = 3) -> list:
    """Split test set into time windows, evaluate each.

    Temporal support is evaluated at the 0.60 reference band only. We expose
    window-level counts, precision, and Wilson lower bounds so any downstream
    semantics can remain honest about what is validated versus what is merely
    supportive/advisory.
    """
    n = len(df_test)
    if n < n_windows * 20:
        return []

    window_size = n // n_windows
    windows = []
    y = df_test["y"].astype(int).values

    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size if i < n_windows - 1 else n
        w_y = y[start:end]
        w_pred = pred_test[start:end]

        if len(np.unique(w_y)) < 2:
            w_auc = 0.5
        else:
            w_auc = float(roc_auc_score(w_y, w_pred))

        mask_60 = w_pred >= 0.60
        c60 = int(mask_60.sum())
        hits_60 = int(w_y[mask_60].sum()) if c60 > 0 else 0
        p60 = hits_60 / c60 if c60 > 0 else None
        wl60 = _wilson_lower(hits_60, c60) if c60 > 0 else 0.0

        ts_col = pd.to_datetime(df_test.iloc[start:end]["ts"], utc=True)
        windows.append({
            "window": i + 1,
            "start": ts_col.iloc[0].isoformat() if len(ts_col) > 0 else None,
            "end": ts_col.iloc[-1].isoformat() if len(ts_col) > 0 else None,
            "samples": end - start,
            "event_rate": round(float(w_y.mean()), 4),
            "auc": round(w_auc, 4),
            "reference_threshold": 0.60,
            "hits_at_0_60": hits_60,
            "precision_at_0_60": round(p60, 4) if p60 is not None else None,
            "count_at_0_60": c60,
            "wilson_lower_0_60": round(float(wl60), 4),
        })

    aucs = [w["auc"] for w in windows]
    p60s = [w["precision_at_0_60"] for w in windows if w["precision_at_0_60"] is not None]
    return {
        "windows": windows,
        "reference_threshold": 0.60,
        "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        "worst_auc": round(float(np.min(aucs)), 4) if aucs else None,
        "mean_precision_at_0_60": round(float(np.mean(p60s)), 4) if p60s else None,
        "worst_precision_at_0_60": round(float(np.min(p60s)), 4) if p60s else None,
    }


def _contract_metric(metrics: dict, name: str, default=0.0, *, prefer_adjusted: bool = True):
    if prefer_adjusted and f"adjusted_{name}" in metrics:
        value = metrics.get(f"adjusted_{name}", default)
    else:
        value = metrics.get(name, default)
    return default if value is None else value


THRESHOLD_SUFFIXES = {
    0.80: "0_80",
    0.75: "0_75",
    0.70: "0_70",
    0.60: "0_60",
    0.55: "0_55",
    0.50: "0_50",
    0.45: "0_45",
}


def _threshold_metric_present(metrics: dict, suffix: str, *, prefer_adjusted: bool) -> bool:
    prefixes = ["adjusted_", ""] if prefer_adjusted else [""]
    for prefix in prefixes:
        if any(f"{prefix}{stem}_{suffix}" in metrics for stem in ("count_at", "precision_at", "wilson_lower")):
            return True
    return False


def threshold_metric_snapshot(metrics: dict, *, prefer_adjusted: bool = True) -> list[dict]:
    rows: list[dict] = []
    for th, suffix in THRESHOLD_SUFFIXES.items():
        if not _threshold_metric_present(metrics, suffix, prefer_adjusted=prefer_adjusted):
            continue
        rows.append({
            "threshold": th,
            "count": int(_contract_metric(metrics, f"count_at_{suffix}", 0, prefer_adjusted=prefer_adjusted) or 0),
            "precision": round(float(_contract_metric(metrics, f"precision_at_{suffix}", 0.0, prefer_adjusted=prefer_adjusted) or 0.0), 4),
            "wilson_lower": round(float(_contract_metric(metrics, f"wilson_lower_{suffix}", 0.0, prefer_adjusted=prefer_adjusted) or 0.0), 4),
        })
    return rows


def contract_matches_metrics(contract: dict | None, metrics: dict, *, prefer_adjusted: bool = True) -> bool:
    if not isinstance(contract, dict):
        return False
    stats = contract.get("threshold_stats") or []
    if not isinstance(stats, list) or not stats:
        return False
    live_stats = threshold_metric_snapshot(metrics, prefer_adjusted=prefer_adjusted)
    if len(stats) != len(live_stats):
        return False
    by_threshold = {round(float(row.get("threshold", -1.0)), 2): row for row in stats}
    for row in live_stats:
        key = round(float(row["threshold"]), 2)
        existing = by_threshold.get(key)
        if existing is None:
            return False
        if int(existing.get("count", 0) or 0) != int(row["count"]):
            return False
        if abs(float(existing.get("precision", 0.0) or 0.0) - float(row["precision"])) > 1e-4:
            return False
        if abs(float(existing.get("wilson_lower", 0.0) or 0.0) - float(row["wilson_lower"])) > 1e-4:
            return False
    return True


def _derive_temporal_tail_support(
    metrics: dict,
    *,
    prefer_adjusted: bool,
    precision_floor: float,
    min_count: int,
    highest_validated: float | None,
) -> dict:
    stability = _contract_metric(
        metrics,
        "temporal_stability",
        metrics.get("temporal_stability_adjusted" if prefer_adjusted else "temporal_stability_model", {}),
        prefer_adjusted=prefer_adjusted,
    ) or {}
    windows = list(stability.get("windows") or [])
    reference_threshold = float(stability.get("reference_threshold", 0.60) or 0.60)
    supported_windows = 0
    observed_windows = 0
    sparse_windows = 0
    for window in windows:
        count = int(window.get("count_at_0_60") or 0)
        precision = window.get("precision_at_0_60")
        hits = int(window.get("hits_at_0_60") or 0)
        wilson = float(window.get("wilson_lower_0_60") or (_wilson_lower(hits, count) if count > 0 else 0.0))
        if count >= int(min_count):
            observed_windows += 1
        if count >= int(min_count) and precision is not None and float(wilson) >= float(precision_floor):
            supported_windows += 1
        elif count > 0:
            sparse_windows += 1

    windows_total = len(windows)
    if not windows:
        state = "temporal_support_unknown"
        semantics = "advisory_only"
        evidence = "temporal_window_support_unavailable"
        note = "Temporal tail support unavailable; treat any actionability overlay as advisory only."
    elif highest_validated is not None and float(highest_validated) > reference_threshold:
        state = "temporal_support_heuristic_only"
        semantics = "advisory_only"
        evidence = "reference_band_only"
        note = (
            f"Temporal support is only measured at the {reference_threshold:.2f} reference band, "
            f"while the validated tail extends to {float(highest_validated):.2f}; use temporal support as advisory only."
        )
    elif supported_windows >= 2:
        state = "validated_tail_temporally_supported"
        semantics = "reference_band_supported"
        evidence = "window_level_validated_reference_band"
        note = f"Reference-band temporal support cleared the validation floor in {supported_windows} of {windows_total} windows."
    elif supported_windows == 1:
        state = "validated_but_temporally_sparse"
        semantics = "reference_band_sparse"
        evidence = "window_level_sparse_reference_band"
        note = "Reference-band temporal support exists, but it is concentrated in one window."
    elif observed_windows > 0:
        state = "validated_but_temporally_mixed"
        semantics = "reference_band_mixed"
        evidence = "window_level_mixed_reference_band"
        note = "Validated tail exists overall, but window-level reference-band support is inconsistent."
    else:
        state = "validated_but_temporally_unobserved"
        semantics = "reference_band_unobserved"
        evidence = "window_level_unobserved_reference_band"
        note = "Validated tail exists overall, but no temporal window had enough reference-band evidence to support it."

    return {
        "temporal_tail_state": state,
        "temporal_tail_semantics": semantics,
        "temporal_support_basis": "window_level_reference_band",
        "temporal_support_threshold": round(reference_threshold, 2),
        "temporal_support_validation_floor": round(float(precision_floor), 4),
        "temporal_support_min_count": int(min_count),
        "temporal_support_evidence": evidence,
        "temporal_supported_windows": supported_windows,
        "temporal_observed_windows": observed_windows,
        "temporal_windows_total": windows_total,
        "temporal_sparse_windows": sparse_windows,
        "temporal_note": note,
    }


def derive_tail_trust_profile(
    metrics: dict,
    *,
    min_count: int = 25,
    min_wilson_lift: float = 1.10,
    min_precision_floor: float = 0.18,
    unvalidated_tail_cap: float = 0.65,
    prefer_adjusted: bool = True,
    family_label: str | None = None,
) -> dict:
    """Summarise whether the model has a statistically defensible upper tail."""
    event_rate = float(
        _contract_metric(
            metrics,
            "quality_event_rate_holdout",
            metrics.get("event_rate_test", 0.0),
            prefer_adjusted=prefer_adjusted,
        )
        or 0.0
    )
    precision_floor = max(float(min_precision_floor), float(event_rate) * float(min_wilson_lift))

    threshold_stats = []
    validated_thresholds = []
    for row in threshold_metric_snapshot(metrics, prefer_adjusted=prefer_adjusted):
        validated = bool(int(row["count"]) >= int(min_count) and float(row["wilson_lower"]) >= precision_floor)
        enriched = {**row, "validated": validated}
        threshold_stats.append(enriched)
        if validated:
            validated_thresholds.append(float(row["threshold"]))

    highest_validated = max(validated_thresholds) if validated_thresholds else None
    if highest_validated is None:
        tail_state = "no_validated_tail"
        semantics = "ranking_only"
        notes = [
            "No statistically defensible >=0.60 tail was validated on holdout data.",
            "Treat upper-tail scores as ranking hints, not trustworthy probabilities.",
        ]
        temporal_support = {
            "temporal_tail_state": "no_validated_tail",
            "temporal_tail_semantics": "ranking_only",
            "temporal_supported_windows": 0,
            "temporal_observed_windows": 0,
            "temporal_windows_total": len(((_contract_metric(metrics, "temporal_stability", {}, prefer_adjusted=prefer_adjusted) or {}).get("windows") or [])),
            "temporal_sparse_windows": 0,
            "temporal_note": "Temporal support not relevant because no validated upper tail exists.",
        }
    else:
        tail_state = f"validated_tail_to_{highest_validated:.2f}"
        semantics = "validated_tail_probability"
        notes = [f"Validated holdout tail available at threshold {highest_validated:.2f} and below."]
        temporal_support = _derive_temporal_tail_support(
            metrics,
            prefer_adjusted=prefer_adjusted,
            precision_floor=precision_floor,
            min_count=min_count,
            highest_validated=highest_validated,
        )
        notes.append(str(temporal_support["temporal_note"]))

    quantiles = _contract_metric(metrics, "score_quantiles", metrics.get("score_quantiles", {}), prefer_adjusted=prefer_adjusted) or {}
    top_bucket_quality_rate = _contract_metric(metrics, "top_bucket_quality_rate", metrics.get("top_bucket_quality_rate", {}), prefer_adjusted=prefer_adjusted) or {}
    top_bucket_lift = _contract_metric(metrics, "top_bucket_lift", metrics.get("top_bucket_lift", {}), prefer_adjusted=prefer_adjusted) or {}
    dead_upper_tail = bool(_contract_metric(metrics, "dead_upper_tail", False, prefer_adjusted=prefer_adjusted))

    return {
        "score_family": family_label or ("adjusted_live" if prefer_adjusted else "raw_model"),
        "tail_validation_state": tail_state,
        "probability_semantics_default": semantics,
        "validated_thresholds": validated_thresholds,
        "highest_validated_threshold": highest_validated,
        "event_rate_reference": round(event_rate, 4),
        "validation_count_floor": int(min_count),
        "validation_precision_floor": round(precision_floor, 4),
        "validation_wilson_lift_min": float(min_wilson_lift),
        "unvalidated_tail_cap": float(unvalidated_tail_cap),
        "threshold_stats": threshold_stats,
        "score_quantiles": quantiles,
        "top_bucket_quality_rate": top_bucket_quality_rate,
        "top_bucket_lift": top_bucket_lift,
        "dead_upper_tail": dead_upper_tail,
        "notes": notes,
        **temporal_support,
    }


def derive_score_contracts(
    metrics: dict,
    *,
    min_count: int = 25,
    min_wilson_lift: float = 1.10,
    min_precision_floor: float = 0.18,
    unvalidated_tail_cap: float = 0.65,
) -> tuple[dict, dict, dict]:
    raw_contract = derive_tail_trust_profile(
        metrics,
        min_count=min_count,
        min_wilson_lift=min_wilson_lift,
        min_precision_floor=min_precision_floor,
        unvalidated_tail_cap=unvalidated_tail_cap,
        prefer_adjusted=False,
        family_label="raw_model",
    )
    live_contract = derive_tail_trust_profile(
        metrics,
        min_count=min_count,
        min_wilson_lift=min_wilson_lift,
        min_precision_floor=min_precision_floor,
        unvalidated_tail_cap=unvalidated_tail_cap,
        prefer_adjusted=True,
        family_label="adjusted_live",
    )
    reconciliation = {
        "raw_model": {
            "score_family": "raw_model",
            "highest_validated_threshold": raw_contract.get("highest_validated_threshold"),
            "validated_thresholds": list(raw_contract.get("validated_thresholds") or []),
            "score_quantiles": raw_contract.get("score_quantiles", {}),
            "dead_upper_tail": bool(raw_contract.get("dead_upper_tail", False)),
        },
        "adjusted_live": {
            "score_family": "adjusted_live",
            "highest_validated_threshold": live_contract.get("highest_validated_threshold"),
            "validated_thresholds": list(live_contract.get("validated_thresholds") or []),
            "score_quantiles": live_contract.get("score_quantiles", {}),
            "dead_upper_tail": bool(live_contract.get("dead_upper_tail", False)),
        },
        "live_contract_family": "adjusted_live",
        "legacy_metric_family": "raw_model",
        "mismatch_detected": bool(
            (raw_contract.get("validated_thresholds") or []) != (live_contract.get("validated_thresholds") or [])
            or (raw_contract.get("score_quantiles", {}).get("q99") != live_contract.get("score_quantiles", {}).get("q99"))
        ),
        "notes": [
            "Legacy precision_at_* metrics are raw-model holdout metrics.",
            "Live scanner suppression and semantics use adjusted_live metrics.",
        ],
    }
    if reconciliation["mismatch_detected"]:
        reconciliation["notes"].append("Raw-model and adjusted-live score families diverge; compare like with like.")
    return raw_contract, live_contract, reconciliation


def _is_stablecoin_pair(symbol: str) -> bool:
    base = str(symbol).split("-", 1)[0].upper()
    return base in STABLES


def _repair_cohort_metadata(meta: dict) -> tuple[dict, list[str]]:
    repaired = dict(meta or {})
    removed: list[str] = []
    keys = ("trained_cohort_symbols", "training_symbols_requested", "training_symbols_used")
    for key in keys:
        values = repaired.get(key)
        if not isinstance(values, list):
            continue
        filtered: list[str] = []
        seen: set[str] = set()
        for value in values:
            symbol = str(value)
            if _is_stablecoin_pair(symbol):
                removed.append(symbol)
                continue
            if symbol in seen:
                continue
            seen.add(symbol)
            filtered.append(symbol)
        repaired[key] = filtered
    cohort = repaired.get("trained_cohort_symbols")
    if isinstance(cohort, list):
        repaired["trained_cohort_size"] = len(cohort)
        repaired["trained_cohort_hash"] = hashlib.sha256("|".join(cohort).encode("utf-8")).hexdigest()[:16] if cohort else "none"
    return repaired, sorted(set(removed))


def _metric_family_summary(metrics: dict, *, prefer_adjusted: bool) -> dict:
    return {
        "event_rate_reference": round(float(_contract_metric(metrics, "quality_event_rate_holdout", metrics.get("event_rate_test", 0.0), prefer_adjusted=prefer_adjusted) or 0.0), 4),
        "count_at_0_60": int(_contract_metric(metrics, "count_at_0_60", 0, prefer_adjusted=prefer_adjusted) or 0),
        "precision_at_0_60": round(float(_contract_metric(metrics, "precision_at_0_60", 0.0, prefer_adjusted=prefer_adjusted) or 0.0), 4),
        "wilson_lower_0_60": round(float(_contract_metric(metrics, "wilson_lower_0_60", 0.0, prefer_adjusted=prefer_adjusted) or 0.0), 4),
        "validated_thresholds_from_metrics": [
            row["threshold"]
            for row in threshold_metric_snapshot(metrics, prefer_adjusted=prefer_adjusted)
            if int(row["count"]) > 0
        ],
    }


def build_model_status_summary(meta: dict) -> dict:
    pt2 = dict(meta or {})
    summary_keys = [
        "trained", "path", "model_type", "selection_method", "trained_at_utc", "model_fingerprint",
        "rows_all", "rows_train", "rows_validation", "rows_test", "embargo_minutes", "embargo_dropped",
        "event_rate_all", "event_rate_test", "raw_touch_rate_all", "raw_touch_rate_holdout", "quality_event_rate_holdout",
        "auc_holdout", "adjusted_auc_holdout", "brier_holdout", "adjusted_brier_holdout",
        "val_auc_holdout", "val_adjusted_auc_holdout", "val_brier_holdout", "val_adjusted_brier_holdout",
        "touch_quality_rate", "adjusted_touch_quality_rate", "mean_mae", "adjusted_mean_mae", "mean_end_ret", "adjusted_mean_end_ret",
        "training_symbol_selection_method", "training_candidate_pool_size", "live_universe_mode",
        "trained_cohort_size", "trained_cohort_hash",
    ]
    compact = {k: pt2.get(k) for k in summary_keys if k in pt2}
    compact["score_distribution_adjusted"] = {
        "score_quantiles": (pt2.get("score_distribution_adjusted") or {}).get("score_quantiles") or pt2.get("adjusted_score_quantiles", {}),
        "top_bucket_lift": (pt2.get("score_distribution_adjusted") or {}).get("top_bucket_lift") or pt2.get("adjusted_top_bucket_lift", {}),
        "dead_upper_tail": bool((pt2.get("score_distribution_adjusted") or {}).get("dead_upper_tail", pt2.get("adjusted_dead_upper_tail", False))),
    }
    compact["adjusted_score_parity_audit"] = {
        "guardrail_cap_exact": bool((pt2.get("adjusted_score_parity_audit") or {}).get("guardrail_cap_exact", False)),
        "panic_penalty_exact": bool((pt2.get("adjusted_score_parity_audit") or {}).get("panic_penalty_exact", False)),
        "binance_penalty_exact": bool((pt2.get("adjusted_score_parity_audit") or {}).get("binance_penalty_exact", False)),
        "sector_penalty_exact": bool((pt2.get("adjusted_score_parity_audit") or {}).get("sector_penalty_exact", False)),
        "guardrail_cap_rows": int((pt2.get("adjusted_score_parity_audit") or {}).get("guardrail_cap_rows", 0) or 0),
        "binance_penalty_rows": int((pt2.get("adjusted_score_parity_audit") or {}).get("binance_penalty_rows", 0) or 0),
        "limitations": list((pt2.get("adjusted_score_parity_audit") or {}).get("limitations") or []),
        "parity_summary": (pt2.get("adjusted_score_parity_audit") or {}).get("parity_summary"),
    }
    compact["score_distribution_model"] = {
        "score_quantiles": (pt2.get("score_distribution_model") or {}).get("score_quantiles") or pt2.get("score_quantiles", {}),
        "top_bucket_lift": (pt2.get("score_distribution_model") or {}).get("top_bucket_lift") or pt2.get("top_bucket_lift", {}),
        "dead_upper_tail": bool((pt2.get("score_distribution_model") or {}).get("dead_upper_tail", pt2.get("dead_upper_tail", False))),
    }
    return compact


def compact_score_contract(contract: dict | None) -> dict:
    src = dict(contract or {})
    if not src:
        return {}
    keys = [
        "score_family", "tail_validation_state", "probability_semantics_default", "validated_thresholds",
        "highest_validated_threshold", "event_rate_reference", "validation_count_floor",
        "validation_precision_floor", "validation_wilson_lift_min", "unvalidated_tail_cap",
        "dead_upper_tail", "notes", "contract_metric_source",
        "temporal_tail_state", "temporal_tail_semantics", "temporal_support_basis",
        "temporal_support_threshold", "temporal_support_validation_floor", "temporal_support_min_count",
        "temporal_support_evidence", "temporal_supported_windows", "temporal_observed_windows",
        "temporal_windows_total", "temporal_sparse_windows", "temporal_note",
    ]
    compact = {k: src.get(k) for k in keys if k in src}
    compact["threshold_stats"] = [
        {
            "threshold": float(row.get("threshold", 0.0)),
            "count": int(row.get("count", 0) or 0),
            "precision": float(row.get("precision", 0.0) or 0.0) if row.get("precision") is not None else None,
            "wilson_lower": float(row.get("wilson_lower", 0.0) or 0.0),
            "validated": bool(row.get("validated", False)),
        }
        for row in (src.get("threshold_stats") or [])
    ]
    return compact


def compact_score_reconciliation(reconciliation: dict | None) -> dict:
    src = dict(reconciliation or {})
    if not src:
        return {}
    return {
        "live_contract_family": src.get("live_contract_family"),
        "legacy_metric_family": src.get("legacy_metric_family"),
        "mismatch_detected": bool(src.get("mismatch_detected", False)),
        "notes": list(src.get("notes") or []),
        "ran_on_startup": bool(src.get("ran_on_startup", False)),
        "ran_on_model_load": bool(src.get("ran_on_model_load", False)),
        "stale_contract_detected": bool(src.get("stale_contract_detected", False)),
        "stale_contract_overridden": bool(src.get("stale_contract_overridden", False)),
        "saved_artifact_contract_stale": src.get("saved_artifact_contract_stale") or {"raw": False, "live": False},
        "persisted_status_contract_stale": src.get("persisted_status_contract_stale") or {"raw": False, "live": False},
        "scanner_contract_source": src.get("scanner_contract_source"),
        "threshold_suppression_contract_source": src.get("threshold_suppression_contract_source"),
        "raw_metric_family_summary": src.get("raw_metric_family_summary") or {},
        "adjusted_metric_family_summary": src.get("adjusted_metric_family_summary") or {},
        "raw_threshold_stats": list(src.get("raw_threshold_stats") or []),
        "live_threshold_stats": list(src.get("live_threshold_stats") or []),
        "repaired_cohort_symbol_count": int(src.get("repaired_cohort_symbol_count", 0) or 0),
        "removed_stablecoin_symbols": list(src.get("removed_stablecoin_symbols") or []),
        "reconciliation_warning": src.get("reconciliation_warning"),
        "reconciliation_error": src.get("reconciliation_error"),
    }


def reconcile_runtime_metadata(
    meta: dict,
    *,
    existing_status: dict | None = None,
    min_count: int = 25,
    min_wilson_lift: float = 1.10,
    min_precision_floor: float = 0.18,
    unvalidated_tail_cap: float = 0.65,
    ran_on_startup: bool = False,
    ran_on_model_load: bool = False,
    scanner_contract_source: str | None = None,
    threshold_suppression_contract_source: str | None = None,
) -> tuple[dict, dict]:
    repaired, removed_symbols = _repair_cohort_metadata(meta)
    saved_raw = (repaired.get("score_contract_raw") or repaired.get("tail_validation") or {}) if isinstance(repaired, dict) else {}
    saved_live = (repaired.get("score_contract_live") or repaired.get("adjusted_tail_validation") or {}) if isinstance(repaired, dict) else {}
    status_raw = ((existing_status or {}).get("score_contract_raw") or {}) if isinstance(existing_status, dict) else {}
    status_live = ((existing_status or {}).get("score_contract_live") or {}) if isinstance(existing_status, dict) else {}
    prior_rec = ((existing_status or {}).get("score_reconciliation") or {}) if isinstance(existing_status, dict) else {}
    ran_on_startup = bool(ran_on_startup or prior_rec.get("ran_on_startup"))
    ran_on_model_load = bool(ran_on_model_load or prior_rec.get("ran_on_model_load"))

    raw_contract, live_contract, reconciliation = derive_score_contracts(
        repaired,
        min_count=min_count,
        min_wilson_lift=min_wilson_lift,
        min_precision_floor=min_precision_floor,
        unvalidated_tail_cap=unvalidated_tail_cap,
    )
    stale_saved_raw = bool(saved_raw) and not contract_matches_metrics(saved_raw, repaired, prefer_adjusted=False)
    stale_saved_live = bool(saved_live) and not contract_matches_metrics(saved_live, repaired, prefer_adjusted=True)
    stale_status_raw = bool(status_raw) and not contract_matches_metrics(status_raw, repaired, prefer_adjusted=False)
    stale_status_live = bool(status_live) and not contract_matches_metrics(status_live, repaired, prefer_adjusted=True)
    stale_detected = any((stale_saved_raw, stale_saved_live, stale_status_raw, stale_status_live))

    live_stats = threshold_metric_snapshot(repaired, prefer_adjusted=True)
    raw_stats = threshold_metric_snapshot(repaired, prefer_adjusted=False)
    live_060 = next((row for row in live_stats if round(float(row["threshold"]), 2) == 0.60), {"count": 0})
    reconciliation_warning = None
    if int(_contract_metric(repaired, "count_at_0_60", 0, prefer_adjusted=True) or 0) > 0 and int(live_060.get("count", 0) or 0) == 0:
        reconciliation_warning = "adjusted_metrics_present_but_live_threshold_stats_zero"
        logger.warning("runtime_reconciliation_warning %s", reconciliation_warning)

    reconciliation = {
        **dict(reconciliation),
        "ran_on_startup": bool(ran_on_startup),
        "ran_on_model_load": bool(ran_on_model_load),
        "stale_contract_detected": bool(stale_detected or reconciliation_warning),
        "stale_contract_overridden": bool(stale_detected or reconciliation_warning),
        "saved_artifact_contract_stale": {"raw": bool(stale_saved_raw), "live": bool(stale_saved_live)},
        "persisted_status_contract_stale": {"raw": bool(stale_status_raw), "live": bool(stale_status_live)},
        "scanner_contract_source": scanner_contract_source or "recomputed_runtime_adjusted",
        "threshold_suppression_contract_source": threshold_suppression_contract_source or "recomputed_runtime_adjusted",
        "raw_metric_family_summary": _metric_family_summary(repaired, prefer_adjusted=False),
        "adjusted_metric_family_summary": _metric_family_summary(repaired, prefer_adjusted=True),
        "raw_threshold_stats": raw_stats,
        "live_threshold_stats": live_stats,
        "repaired_cohort_symbol_count": int(repaired.get("trained_cohort_size", 0) or 0),
        "removed_stablecoin_symbols": removed_symbols,
        "reconciliation_warning": reconciliation_warning,
        "reconciliation_error": None,
    }
    if removed_symbols:
        reconciliation.setdefault("notes", []).append(
            f"Removed stablecoin-base symbols from cohort metadata: {', '.join(removed_symbols[:5])}"
        )

    raw_contract = {**dict(raw_contract), "contract_metric_source": "numeric_metrics_runtime"}
    live_contract = {**dict(live_contract), "contract_metric_source": "numeric_metrics_runtime"}
    live_contract["raw_model_contract"] = dict(raw_contract)
    live_contract["live_contract"] = dict(live_contract)
    live_contract["score_reconciliation"] = dict(reconciliation)
    live_contract["scanner_contract_source"] = reconciliation["scanner_contract_source"]
    live_contract["threshold_suppression_contract_source"] = reconciliation["threshold_suppression_contract_source"]

    repaired["tail_validation"] = raw_contract
    repaired["adjusted_tail_validation"] = live_contract
    repaired["tail_validation_raw"] = raw_contract
    repaired["tail_validation_adjusted"] = live_contract
    repaired["score_contract_raw"] = raw_contract
    repaired["score_contract_live"] = live_contract
    repaired["score_reconciliation"] = reconciliation
    repaired["probability_semantics_default"] = live_contract.get("probability_semantics_default")
    repaired["tail_validation_state"] = live_contract.get("tail_validation_state")
    repaired["highest_validated_threshold"] = live_contract.get("highest_validated_threshold")
    repaired["trained_cohort_size"] = int(repaired.get("trained_cohort_size", 0) or 0)

    bundle = {
        "score_contract": live_contract,
        "score_contract_live": live_contract,
        "score_contract_raw": raw_contract,
        "score_reconciliation": reconciliation,
    }
    return repaired, bundle


# ── Training ──────────────────────────────────────────────────────────────

def train_pt2(training_df: pd.DataFrame, cfg_dict: dict | None = None) -> ModelBundle:
    """v2.7.0: Wilson bounds, temporal stability, adjusted-score evaluation."""
    training_df = training_df.sort_values("ts").reset_index(drop=True)
    if len(training_df) < 300:
        raise ValueError(f"not enough samples for training: {len(training_df)}")

    if "sample_weight" in training_df.columns:
        training_df = training_df[training_df["sample_weight"] > 0].reset_index(drop=True)

    # Derive embargo from configured horizon
    horizon_minutes = (cfg_dict or {}).get("target_horizon_minutes", 240)
    embargo = _embargo_minutes(horizon_minutes)
    df_train, df_val, df_test, embargo_dropped = _purged_time_split(training_df, embargo_minutes=embargo)

    logger.info(
        "training_splits train=%d val=%d test=%d embargo_dropped=%d event_rate=%.3f",
        len(df_train), len(df_val), len(df_test), embargo_dropped, training_df["y"].mean(),
    )

    train_weights = _build_train_weights(df_train)

    best_score: tuple | None = None
    best_bundle: ModelBundle | None = None
    best_val_metrics: dict | None = None

    # --- LightGBM candidates ---
    if LIGHTGBM_AVAILABLE:
        lgbm_candidates = [
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 20},
            {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 20},
            {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 0.5, "min_child_samples": 15},
            {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 25},
            {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 2.0, "min_child_samples": 30},
            {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.05, "reg_lambda": 0.5, "min_child_samples": 15},
            {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 20},
            {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.04, "subsample": 0.7, "colsample_bytree": 0.8, "reg_alpha": 0.15, "reg_lambda": 1.5, "min_child_samples": 25},
        ]

        X_train = sanitize_feature_frame(df_train)
        y_train = df_train["y"].astype(np.int8).values
        X_val = sanitize_feature_frame(df_val)
        y_val = df_val["y"].astype(np.int8).values

        for hp_idx, hp in enumerate(lgbm_candidates):
            try:
                model = lgb.LGBMClassifier(
                    objective="binary", metric="binary_logloss",
                    verbosity=-1, random_state=42, is_unbalance=True, **hp,
                )
                model.fit(
                    X_train, y_train, sample_weight=train_weights,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                )
                raw_val = model.predict_proba(X_val)[:, 1]
                calibrator = None
                calibrator_status = "none"
                if len(np.unique(y_val)) > 1 and len(df_val) >= 50:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(raw_val, y_val)
                    calibrator = _validate_calibrator(iso, raw_val)
                    calibrator_status = "isotonic" if calibrator is not None else "skipped_low_spread"

                pred_val = np.clip(calibrator.predict(raw_val), 0.0, 1.0) if calibrator else np.clip(raw_val, 0.0, 1.0)
                val_metrics = evaluate_predictions(df_val, pred_val, shortlist_threshold=float((cfg_dict or {}).get("live_raw_threshold", 0.35) or 0.35))
                adjusted_val, _ = _simulate_adjusted_scores(pred_val, df_val, cfg_dict or {})
                adjusted_val_metrics = evaluate_predictions(df_val, adjusted_val, shortlist_threshold=float((cfg_dict or {}).get("live_raw_threshold", 0.35) or 0.35))
                for k, v in adjusted_val_metrics.items():
                    val_metrics[f"adjusted_{k}"] = v
                score = model_selection_score(val_metrics)

                logger.info(
                    "lgbm_candidate %d/%d val_auc=%.4f pred_std=%.4f calibrator=%s",
                    hp_idx + 1, len(lgbm_candidates),
                    val_metrics["auc_holdout"], float(np.std(pred_val)), calibrator_status,
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_bundle = ModelBundle(model, calibrator, metadata={}, model_type="lgbm")
                    best_val_metrics = val_metrics
                    best_bundle.metadata = {"_hp": hp, "_calibrator_status": calibrator_status}
                else:
                    del model
                gc.collect()
            except Exception as exc:
                logger.warning("lgbm_candidate_failed hp=%s error=%s", hp, exc)

    # --- Logistic regression candidates ---
    for hp in [
        {"C": 0.35, "l1_ratio": 0.35, "class_weight": None},
        {"C": 0.75, "l1_ratio": 0.65, "class_weight": "balanced"},
    ]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=1800,
                C=hp["C"], l1_ratio=hp["l1_ratio"],
                class_weight=hp["class_weight"], random_state=42,
            )),
        ])
        X_train_lr = sanitize_feature_frame(df_train, dtype="float32")
        X_val_lr = sanitize_feature_frame(df_val, dtype="float32")
        pipe.fit(X_train_lr, df_train["y"], lr__sample_weight=train_weights)
        raw_val = pipe.predict_proba(X_val_lr)[:, 1]
        calibrator = None
        calibrator_status = "none"
        if len(np.unique(df_val["y"])) > 1 and len(df_val) >= 50:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_val, df_val["y"])
            calibrator = _validate_calibrator(iso, raw_val)
            calibrator_status = "isotonic" if calibrator is not None else "skipped_low_spread"

        pred_val = np.clip(calibrator.predict(raw_val), 0.0, 1.0) if calibrator else np.clip(raw_val, 0.0, 1.0)
        val_metrics = evaluate_predictions(df_val, pred_val, shortlist_threshold=float((cfg_dict or {}).get("live_raw_threshold", 0.35) or 0.35))
        adjusted_val, _ = _simulate_adjusted_scores(pred_val, df_val, cfg_dict or {})
        adjusted_val_metrics = evaluate_predictions(df_val, adjusted_val, shortlist_threshold=float((cfg_dict or {}).get("live_raw_threshold", 0.35) or 0.35))
        for k, v in adjusted_val_metrics.items():
            val_metrics[f"adjusted_{k}"] = v
        score = model_selection_score(val_metrics)

        if best_score is None or score > best_score:
            best_score = score
            best_bundle = ModelBundle(pipe, calibrator, metadata={}, model_type="logistic")
            best_val_metrics = val_metrics
            best_bundle.metadata = {"_hp": hp, "_calibrator_status": calibrator_status}

    if best_bundle is None:
        raise RuntimeError("all model candidates failed during training")

    # ── Final evaluation on UNTOUCHED test set ────────────────────────────
    if best_bundle.model_type == "lgbm":
        X_test = sanitize_feature_frame(df_test)
        raw_test = best_bundle.pipeline.predict_proba(X_test)[:, 1]
    else:
        raw_test = best_bundle.pipeline.predict_proba(sanitize_feature_frame(df_test))[:, 1]

    pred_test = np.clip(best_bundle.calibrator.predict(raw_test), 0.0, 1.0) if best_bundle.calibrator else np.clip(raw_test, 0.0, 1.0)
    test_metrics = evaluate_predictions(df_test, pred_test, shortlist_threshold=float((cfg_dict or {}).get("live_raw_threshold", 0.35) or 0.35))

    # ── v2.7.0: Adjusted-score offline evaluation ─────────────────────────
    # Simulate the scanner's post-model overlays on the test set
    adjusted_test, adjusted_parity_audit = _simulate_adjusted_scores(pred_test, df_test, cfg_dict or {})
    adjusted_metrics = evaluate_predictions(df_test, adjusted_test, shortlist_threshold=float((cfg_dict or {}).get("live_raw_threshold", 0.35) or 0.35))

    # ── v2.7.0: Temporal stability ────────────────────────────────────────
    stability = _temporal_stability(df_test, pred_test, n_windows=3)
    adjusted_stability = _temporal_stability(df_test, adjusted_test, n_windows=3)

    # ── v4.1.1: score-distribution + tail trust diagnostics ───────────────
    model_score_distribution = _score_distribution(df_test["y"].astype(int).values, pred_test)
    adjusted_score_distribution = _score_distribution(df_test["y"].astype(int).values, adjusted_test)
    tail_trust = derive_tail_trust_profile(test_metrics)
    adjusted_tail_trust = derive_tail_trust_profile(adjusted_metrics)

    # ── Build metadata ────────────────────────────────────────────────────
    pred_std = float(np.std(pred_test))
    pred_range = float(np.ptp(pred_test))
    raw_std = float(np.std(raw_test))
    calibrator_status = best_bundle.metadata.get("_calibrator_status", "unknown")
    hp_info = best_bundle.metadata.get("_hp", {})

    logger.info(
        "model_selected type=%s test_auc=%.4f val_auc=%.4f adjusted_test_auc=%.4f pred_std=%.4f",
        best_bundle.model_type,
        test_metrics["auc_holdout"], best_val_metrics["auc_holdout"],
        adjusted_metrics["auc_holdout"], pred_std,
    )

    trained_at_utc = datetime.now(timezone.utc).isoformat()
    fingerprint_payload = {
        "trained_at_utc": trained_at_utc,
        "model_type": best_bundle.model_type,
        "hp": hp_info,
        "feature_cols": FEATURE_COLUMNS,
        "rows_train": int(len(df_train)),
        "rows_validation": int(len(df_val)),
        "rows_test": int(len(df_test)),
        "target_move_pct": (cfg_dict or {}).get("target_move_pct"),
        "target_horizon_minutes": (cfg_dict or {}).get("target_horizon_minutes"),
        "quality_max_mae": (cfg_dict or {}).get("quality_max_mae"),
        "quality_min_end_ret": (cfg_dict or {}).get("quality_min_end_ret"),
        "app_version": (cfg_dict or {}).get("app_version", APP_VERSION),
    }
    model_fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    base_meta = {
        "model_type": best_bundle.model_type,
        "feature_cols": FEATURE_COLUMNS,
        "rows_all": int(len(training_df)),
        "rows_train": int(len(df_train)),
        "rows_validation": int(len(df_val)),
        "rows_test": int(len(df_test)),
        "embargo_minutes": embargo,
        "embargo_dropped": embargo_dropped,
        "event_rate_all": float(training_df["y"].mean()),
        "event_rate_test": float(df_test["y"].mean()),
        "raw_touch_rate_all": float(training_df.get("y_raw_touch", training_df["y"]).mean()),
        "calibrator": calibrator_status,
        "pred_spread_std": pred_std,
        "pred_spread_range": pred_range,
        "raw_spread_std": raw_std,
        "feature_mean": {k: float(training_df[k].mean()) for k in FEATURE_COLUMNS},
        "feature_std": {k: float(training_df[k].std() or 1.0) for k in FEATURE_COLUMNS},
        "btc_ret_24h_median": float(training_df["btc_ret_24h"].median()),
        "split_design": "purged_embargo",
        "selection_method": "adjusted_scan_shortlist_utility_then_tail_confidence",
        "trained_at_utc": trained_at_utc,
        "model_fingerprint": model_fingerprint,
    }

    if best_bundle.model_type == "lgbm":
        base_meta.update({f"hp_{k}": v for k, v in hp_info.items()})
    else:
        base_meta.update({
            "best_C": hp_info.get("C"),
            "best_l1_ratio": hp_info.get("l1_ratio"),
            "class_weight": hp_info.get("class_weight") or "none",
        })

    # Raw model test metrics (official)
    base_meta.update(test_metrics)

    # Validation metrics (prefixed)
    for k, v in best_val_metrics.items():
        base_meta[f"val_{k}"] = v

    # v2.7.0: Adjusted-score test metrics (prefixed)
    for k, v in adjusted_metrics.items():
        base_meta[f"adjusted_{k}"] = v

    # v2.7.0: Temporal stability
    base_meta["temporal_stability_model"] = stability
    base_meta["temporal_stability_adjusted"] = adjusted_stability

    # v4.3.9: derive contracts from the merged artifact metrics so saved metadata
    # and live/runtime derivations read from the exact same score family.
    tail_trust_raw, tail_trust_adjusted, score_reconciliation = derive_score_contracts(
        base_meta,
        min_count=25,
        min_wilson_lift=1.10,
        min_precision_floor=0.18,
        unvalidated_tail_cap=0.65,
    )

    base_meta["score_distribution_model"] = model_score_distribution
    base_meta["score_distribution_adjusted"] = adjusted_score_distribution
    base_meta["adjusted_score_parity_audit"] = adjusted_parity_audit
    base_meta["threshold_research_candidates"] = [0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80]
    base_meta["tail_validation"] = tail_trust_raw
    base_meta["adjusted_tail_validation"] = tail_trust_adjusted
    base_meta["tail_validation_raw"] = tail_trust_raw
    base_meta["tail_validation_adjusted"] = tail_trust_adjusted
    base_meta["score_contract_raw"] = tail_trust_raw
    base_meta["score_contract_live"] = tail_trust_adjusted
    base_meta["score_reconciliation"] = score_reconciliation
    base_meta["probability_semantics_default"] = tail_trust_adjusted.get("probability_semantics_default")
    base_meta["tail_validation_state"] = tail_trust_adjusted.get("tail_validation_state")
    base_meta["highest_validated_threshold"] = tail_trust_adjusted.get("highest_validated_threshold")

    best_bundle.metadata = base_meta
    return best_bundle


# ── Train weights ─────────────────────────────────────────────────────────

def _build_train_weights(df_train: pd.DataFrame) -> np.ndarray:
    n = len(df_train)
    recency = np.linspace(0.85, 1.20, n)
    if "sample_weight" in df_train.columns:
        explicit = df_train["sample_weight"].astype(float).values
        weights = recency * explicit
    else:
        touch_quality_bonus = np.where(df_train.get("touch_quality", 0).astype(int).values == 1, 1.10, 1.00)
        ugly_penalty = np.where(df_train.get("path_ugliness", 0.0).astype(float).values > 0.10, 0.90, 1.00)
        weights = recency * touch_quality_bonus * ugly_penalty
    return weights.astype(float)


# ── Model selection (v2.7.0: Wilson lower bounds) ─────────────────────────

def model_selection_score(metrics: dict) -> tuple:
    """Score candidates using scan-level shortlist utility before generic model metrics.

    The selector still requires the raw and adjusted model to clear a minimal AUC
    gate, but it now prefers candidates whose *adjusted* holdout scores produce
    better scan-level visible-vs-hidden separation and stronger top-of-scan
    usefulness at the live shortlist boundary.
    """
    auc = float(metrics["auc_holdout"])
    adjusted_auc = float(metrics.get("adjusted_auc_holdout", auc))
    auc_gate = 1.0 if min(auc, adjusted_auc) >= 0.58 else 0.0

    def _metric(name: str, default: float = 0.0) -> float:
        return float(metrics.get(f"adjusted_{name}", metrics.get(name, default)))

    def _count(name: str) -> int:
        return int(round(_metric(name, 0.0)))

    return (
        auc_gate,
        _metric("scan_shortlist_utility_score", -1.0),
        _metric("scan_shortlist_mean_gap", -1.0),
        _metric("scan_shortlist_pairwise_win_rate", 0.0),
        _metric("scan_shortlist_top1_mean_quality", 0.0),
        _metric("scan_shortlist_top3_mean_quality", 0.0),
        -_metric("scan_shortlist_avg_visible_rows_per_scan", 999.0),
        _metric("wilson_lower_0_80"),
        _count("count_at_0_80"),
        _metric("wilson_lower_0_75"),
        _count("count_at_0_75"),
        _metric("wilson_lower_0_70"),
        _count("count_at_0_70"),
        _metric("wilson_lower_0_60"),
        _metric("acceptable_risk_precision"),
        -_metric("brier_holdout", metrics.get("brier_holdout", 1.0)),
        adjusted_auc,
        auc,
    )


# ── Evaluation ────────────────────────────────────────────────────────────

def _score_distribution(y: np.ndarray, pred: np.ndarray) -> dict:
    if len(pred) == 0:
        return {
            "score_quantiles": {},
            "top_bucket_quality_rate": {},
            "top_bucket_lift": {},
            "dead_upper_tail": True,
        }

    quantiles = {
        "q50": round(float(np.quantile(pred, 0.50)), 4),
        "q75": round(float(np.quantile(pred, 0.75)), 4),
        "q90": round(float(np.quantile(pred, 0.90)), 4),
        "q95": round(float(np.quantile(pred, 0.95)), 4),
        "q99": round(float(np.quantile(pred, 0.99)), 4),
        "max": round(float(np.max(pred)), 4),
    }

    base_rate = float(np.mean(y)) if len(y) else 0.0
    top_rates = {}
    top_lift = {}
    for frac, label in ((0.01, "top_1pct"), (0.05, "top_5pct"), (0.10, "top_10pct")):
        count = max(1, int(np.ceil(len(pred) * frac)))
        idx = np.argsort(pred)[-count:]
        rate = float(np.mean(y[idx])) if len(idx) else 0.0
        top_rates[label] = {"count": int(count), "quality_rate": round(rate, 4)}
        top_lift[label] = round(rate / base_rate, 4) if base_rate > 0 else None

    dead_upper_tail = bool(quantiles["q99"] < 0.60 and quantiles["max"] < 0.70)
    return {
        "score_quantiles": quantiles,
        "top_bucket_quality_rate": top_rates,
        "top_bucket_lift": top_lift,
        "dead_upper_tail": dead_upper_tail,
    }


def _scan_shortlist_utility(df_hold: pd.DataFrame, pred_hold: np.ndarray, *, threshold: float = 0.35) -> dict:
    if len(pred_hold) == 0 or df_hold.empty or "ts" not in df_hold.columns:
        return {
            "scan_shortlist_threshold": float(threshold),
            "scan_shortlist_scans": 0,
            "scan_shortlist_scans_with_visible": 0,
            "scan_shortlist_scan_capture_rate": None,
            "scan_shortlist_avg_visible_rows_per_scan": None,
            "scan_shortlist_visible_quality_rate_mean": None,
            "scan_shortlist_hidden_quality_rate_mean": None,
            "scan_shortlist_mean_gap": None,
            "scan_shortlist_pairwise_win_rate": None,
            "scan_shortlist_pairwise_comparable_scans": 0,
            "scan_shortlist_top1_mean_quality": None,
            "scan_shortlist_top3_mean_quality": None,
            "scan_shortlist_top5_mean_quality": None,
            "scan_shortlist_overwide_penalty": None,
            "scan_shortlist_utility_score": None,
        }

    frame = pd.DataFrame({
        "ts": pd.to_datetime(df_hold["ts"], utc=True, errors="coerce"),
        "score": np.asarray(pred_hold, dtype=float),
        "y": df_hold["y"].astype(int).values,
    }).dropna(subset=["ts"]).sort_values(["ts", "score"], ascending=[True, False]).reset_index(drop=True)
    if frame.empty:
        return {
            "scan_shortlist_threshold": float(threshold),
            "scan_shortlist_scans": 0,
            "scan_shortlist_scans_with_visible": 0,
            "scan_shortlist_scan_capture_rate": None,
            "scan_shortlist_avg_visible_rows_per_scan": None,
            "scan_shortlist_visible_quality_rate_mean": None,
            "scan_shortlist_hidden_quality_rate_mean": None,
            "scan_shortlist_mean_gap": None,
            "scan_shortlist_pairwise_win_rate": None,
            "scan_shortlist_pairwise_comparable_scans": 0,
            "scan_shortlist_top1_mean_quality": None,
            "scan_shortlist_top3_mean_quality": None,
            "scan_shortlist_top5_mean_quality": None,
            "scan_shortlist_overwide_penalty": None,
            "scan_shortlist_utility_score": None,
        }

    base_event_rate = float(frame["y"].mean()) if len(frame) else 0.0
    scan_count = int(frame["ts"].nunique())
    scans_with_visible = 0
    pairwise_wins = 0.0
    pairwise_comparable = 0
    visible_counts: list[int] = []
    visible_rates: list[float] = []
    hidden_rates: list[float] = []
    gaps: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    top5: list[float] = []

    for _, scan in frame.groupby("ts", sort=False):
        scan = scan.sort_values("score", ascending=False)
        visible = scan[scan["score"] >= float(threshold)]
        hidden = scan[scan["score"] < float(threshold)]
        visible_counts.append(int(len(visible)))
        top1.append(float(scan.iloc[:1]["y"].mean()))
        top3.append(float(scan.iloc[: min(3, len(scan))]["y"].mean()))
        top5.append(float(scan.iloc[: min(5, len(scan))]["y"].mean()))
        if not visible.empty:
            scans_with_visible += 1
            visible_rate = float(visible["y"].mean())
            visible_rates.append(visible_rate)
            if not hidden.empty:
                hidden_rate = float(hidden["y"].mean())
                hidden_rates.append(hidden_rate)
                gap = visible_rate - hidden_rate
                gaps.append(gap)
                pairwise_comparable += 1
                if gap > 0:
                    pairwise_wins += 1.0
                elif abs(gap) <= 1e-12:
                    pairwise_wins += 0.5

    scan_capture_rate = float(scans_with_visible) / float(scan_count) if scan_count else None
    avg_visible_rows = float(np.mean(visible_counts)) if visible_counts else None
    visible_quality_mean = float(np.mean(visible_rates)) if visible_rates else None
    hidden_quality_mean = float(np.mean(hidden_rates)) if hidden_rates else None
    mean_gap = float(np.mean(gaps)) if gaps else None
    pairwise_win_rate = float(pairwise_wins) / float(pairwise_comparable) if pairwise_comparable else None
    top1_mean = float(np.mean(top1)) if top1 else None
    top3_mean = float(np.mean(top3)) if top3 else None
    top5_mean = float(np.mean(top5)) if top5 else None
    overwide_penalty = max(0.0, (avg_visible_rows or 0.0) - 5.0) / 5.0 if avg_visible_rows is not None else None

    utility_score = None
    if mean_gap is not None:
        utility_score = (
            float(mean_gap)
            + 0.25 * (((pairwise_win_rate if pairwise_win_rate is not None else 0.5) - 0.5))
            + 0.10 * (((top1_mean if top1_mean is not None else base_event_rate) - base_event_rate))
            + 0.05 * (((top3_mean if top3_mean is not None else base_event_rate) - base_event_rate))
            - 0.02 * (overwide_penalty or 0.0)
        )

    return {
        "scan_shortlist_threshold": float(threshold),
        "scan_shortlist_scans": scan_count,
        "scan_shortlist_scans_with_visible": scans_with_visible,
        "scan_shortlist_scan_capture_rate": round(scan_capture_rate, 6) if scan_capture_rate is not None else None,
        "scan_shortlist_avg_visible_rows_per_scan": round(avg_visible_rows, 6) if avg_visible_rows is not None else None,
        "scan_shortlist_visible_quality_rate_mean": round(visible_quality_mean, 6) if visible_quality_mean is not None else None,
        "scan_shortlist_hidden_quality_rate_mean": round(hidden_quality_mean, 6) if hidden_quality_mean is not None else None,
        "scan_shortlist_mean_gap": round(mean_gap, 6) if mean_gap is not None else None,
        "scan_shortlist_pairwise_win_rate": round(pairwise_win_rate, 6) if pairwise_win_rate is not None else None,
        "scan_shortlist_pairwise_comparable_scans": int(pairwise_comparable),
        "scan_shortlist_top1_mean_quality": round(top1_mean, 6) if top1_mean is not None else None,
        "scan_shortlist_top3_mean_quality": round(top3_mean, 6) if top3_mean is not None else None,
        "scan_shortlist_top5_mean_quality": round(top5_mean, 6) if top5_mean is not None else None,
        "scan_shortlist_overwide_penalty": round(overwide_penalty, 6) if overwide_penalty is not None else None,
        "scan_shortlist_utility_score": round(utility_score, 6) if utility_score is not None else None,
    }


def evaluate_predictions(df_hold: pd.DataFrame, pred_hold: np.ndarray, *, shortlist_threshold: float = 0.35) -> dict:
    y = df_hold["y"].astype(int).values
    auc = float(roc_auc_score(y, pred_hold)) if len(np.unique(y)) > 1 else 0.5
    brier = float(brier_score_loss(y, pred_hold))
    metrics: dict[str, float | int] = {"auc_holdout": auc, "brier_holdout": brier}

    def _precision(th: float) -> tuple[float, int]:
        mask = pred_hold >= th
        count = int(mask.sum())
        if count == 0:
            return 0.0, 0
        return float(precision_score(y[mask], np.ones(count), zero_division=0)), count

    for th, suffix in THRESHOLD_SUFFIXES.items():
        precision, count = _precision(float(th))
        wl = _wilson_lower(int(round(precision * count)), count)
        metrics[f"precision_at_{suffix}"] = precision
        metrics[f"count_at_{suffix}"] = count
        metrics[f"wilson_lower_{suffix}"] = wl

    acceptable_mask = (
        (pred_hold >= 0.60)
        & (df_hold.get("path_ugliness", 0.0).astype(float).values < 0.08)
        & (df_hold.get("mae", 0.0).astype(float).values > -0.04)
    )
    acceptable_precision = float(precision_score(y[acceptable_mask], np.ones(int(acceptable_mask.sum())), zero_division=0)) if acceptable_mask.sum() > 0 else 0.0
    challenge_mask = (
        (df_hold["wickiness"] > df_hold["wickiness"].quantile(0.75))
        | (df_hold["jumpiness"] > df_hold["jumpiness"].quantile(0.75))
        | (df_hold["downside_impulse"] < df_hold["downside_impulse"].quantile(0.15))
    )
    challenge_rate = float(df_hold.loc[challenge_mask, "y"].mean()) if challenge_mask.sum() else 0.0
    challenge_precision = float(df_hold.loc[challenge_mask & (pred_hold >= 0.6), "y"].mean()) if (challenge_mask & (pred_hold >= 0.6)).sum() else 0.0
    btc_panic_mask = df_hold["btc_ret_1h"] < -0.025
    btc_panic_precision = 0.0
    btc_panic_count = 0
    if btc_panic_mask.any():
        btc_panic_above_60 = btc_panic_mask & (pred_hold >= 0.6)
        btc_panic_count = int(btc_panic_above_60.sum())
        btc_panic_precision = float(df_hold.loc[btc_panic_above_60, "y"].mean()) if btc_panic_above_60.sum() else 0.0
    low_activity_mask = df_hold["observed_bar_density_24h"] < 0.5
    low_activity_precision = 0.0
    low_activity_count = 0
    if low_activity_mask.any():
        low_activity_above_60 = low_activity_mask & (pred_hold >= 0.6)
        low_activity_count = int(low_activity_above_60.sum())
        low_activity_precision = float(df_hold.loc[low_activity_above_60, "y"].mean()) if low_activity_above_60.sum() else 0.0
    raw_touch_rate = float(df_hold.get("y_raw_touch", df_hold["y"]).mean()) if "y_raw_touch" in df_hold else float(y.mean())
    score_diag = _score_distribution(y, pred_hold)
    scan_shortlist = _scan_shortlist_utility(df_hold, pred_hold, threshold=shortlist_threshold)
    metrics.update({
        "acceptable_risk_precision": acceptable_precision,
        "challenge_set_event_rate": challenge_rate,
        "challenge_set_precision_at_0_60": challenge_precision,
        "btc_panic_challenge_precision": btc_panic_precision,
        "btc_panic_challenge_count": btc_panic_count,
        "low_activity_challenge_precision": low_activity_precision,
        "low_activity_challenge_count": low_activity_count,
        "touch_quality_rate": float(df_hold.get("touch_quality", pd.Series(dtype=float)).mean() if "touch_quality" in df_hold else 0.0),
        "touched_before_major_adverse_rate": float(df_hold.get("touched_before_major_adverse", pd.Series(dtype=float)).mean() if "touched_before_major_adverse" in df_hold else 0.0),
        "mean_mae": float(df_hold.get("mae", pd.Series(dtype=float)).mean() if "mae" in df_hold else 0.0),
        "mean_end_ret": float(df_hold.get("end_ret", pd.Series(dtype=float)).mean() if "end_ret" in df_hold else 0.0),
        "raw_touch_rate_holdout": raw_touch_rate,
        "quality_event_rate_holdout": float(y.mean()),
        **score_diag,
        **scan_shortlist,
    })
    return metrics
