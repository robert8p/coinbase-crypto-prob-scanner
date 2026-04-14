from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def compute_binance_gap_penalty(binance_gap: float) -> float:
    gap = float(binance_gap or 0.0)
    return float(min(0.08, abs(gap) * 4.0)) if gap < -0.005 else 0.0

def compute_binance_lead_penalty(binance_lead_1h: float) -> float:
    lead = float(binance_lead_1h or 0.0)
    return float(min(0.06, abs(lead) * 2.0)) if lead < -0.02 else 0.0

def apply_live_post_model_adjustments(prob_model: float, row: dict[str, Any], guard: dict[str, Any], *, is_panic: bool=False, threshold_boost: float=0.0, sector_penalty: float=0.0, guardrail_cap: float=0.65) -> tuple[float, dict[str, float|bool]]:
    prob = float(prob_model)
    capped = bool(guard.get("capped", False))
    guardrail_cap = float(guardrail_cap or 0.65)
    if capped:
        prob = min(prob, guardrail_cap)
    panic_applied = bool(is_panic and float(threshold_boost or 0.0) > 0.0)
    if panic_applied:
        prob = max(0.01, prob - float(threshold_boost or 0.0))
    sector_penalty = max(0.0, float(sector_penalty or 0.0))
    if sector_penalty > 0.0:
        prob = max(0.01, prob - sector_penalty)
    gap_penalty = compute_binance_gap_penalty(row.get("binance_price_gap", 0.0))
    if gap_penalty > 0.0:
        prob = max(0.01, prob - gap_penalty)
    lead_penalty = compute_binance_lead_penalty(row.get("binance_lead_1h", 0.0))
    if lead_penalty > 0.0:
        prob = max(0.01, prob - lead_penalty)
    return float(np.clip(prob, 0.01, 0.95)), {
        "guardrail_capped": capped,
        "panic_penalty": float(threshold_boost or 0.0) if panic_applied else 0.0,
        "sector_penalty": sector_penalty,
        "binance_gap_penalty": gap_penalty,
        "binance_lead_penalty": lead_penalty,
        "guardrail_cap": guardrail_cap,
        "total_penalty": float((float(threshold_boost or 0.0) if panic_applied else 0.0) + sector_penalty + gap_penalty + lead_penalty),
    }

def compute_vector_binance_gap_penalty(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.zeros(len(vals), dtype=float)
    mask = vals < -0.005
    out[mask] = np.minimum(0.08, np.abs(vals[mask]) * 4.0)
    return out

def compute_vector_binance_lead_penalty(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.zeros(len(vals), dtype=float)
    mask = vals < -0.02
    out[mask] = np.minimum(0.06, np.abs(vals[mask]) * 2.0)
    return out

def simulate_live_post_model_adjustments(pred_model: np.ndarray, df: pd.DataFrame, cfg_dict: dict|None=None) -> tuple[np.ndarray, dict[str, Any]]:
    cfg_dict = dict(cfg_dict or {})
    pred = np.asarray(pred_model, dtype=float).copy()
    details: dict[str, Any] = {"guardrail_cap_exact": False, "panic_penalty_exact": True, "binance_penalty_exact": True, "sector_penalty_exact": False, "limitations": []}
    if "guard_capped" in df.columns:
        cap_mask = df["guard_capped"].astype(bool).values
        details["guardrail_cap_exact"] = True
    elif "guardrail_capped" in df.columns:
        cap_mask = df["guardrail_capped"].astype(bool).values
        details["guardrail_cap_exact"] = True
    else:
        downside_cap = float(cfg_dict.get("downside_cap", 0.78) or 0.78)
        uncertainty_cap = float(cfg_dict.get("uncertainty_cap", 0.72) or 0.72)
        downside_proxy = np.zeros(len(df), dtype=float)
        if "downside_impulse" in df.columns and "wickiness" in df.columns:
            downside_proxy = 0.32*np.maximum(0.0, -df["downside_impulse"].astype(float).values)/0.05 + 0.15*df["wickiness"].astype(float).values/0.05
        uncertainty_terms = []
        if "observed_bar_density_24h" in df.columns:
            uncertainty_terms.append(np.maximum(0.0, 0.80 - df["observed_bar_density_24h"].astype(float).values))
        if "nonzero_volume_rate_24h" in df.columns:
            uncertainty_terms.append(np.maximum(0.0, 0.70 - df["nonzero_volume_rate_24h"].astype(float).values))
        if "rv_ratio_1h_24h" in df.columns:
            uncertainty_terms.append(np.maximum(0.0, df["rv_ratio_1h_24h"].astype(float).values - 1.8)/3.0)
        uncertainty_proxy = np.clip(np.sum(uncertainty_terms, axis=0), 0.0, 1.5) if uncertainty_terms else np.zeros(len(df), dtype=float)
        cap_mask = (downside_proxy > downside_cap) | (uncertainty_proxy > uncertainty_cap)
        details["limitations"].append("guardrail cap approximated from feature proxies")
    guardrail_cap = float(cfg_dict.get("tail_unvalidated_cap", 0.65) or 0.65)
    pred = np.where(cap_mask, np.minimum(pred, guardrail_cap), pred)
    details["guardrail_cap"] = guardrail_cap
    details["guardrail_cap_rows"] = int(cap_mask.sum())
    panic_threshold = float(cfg_dict.get("btc_panic_threshold", -0.025) or -0.025)
    panic_boost = float(cfg_dict.get("panic_threshold_boost", 0.10) or 0.10)
    panic_mask = (df["btc_ret_1h"].astype(float).values < panic_threshold) if "btc_ret_1h" in df.columns else np.zeros(len(df), dtype=bool)
    pred = np.where(panic_mask, np.maximum(0.01, pred - panic_boost), pred)
    details["panic_penalty_rows"] = int(panic_mask.sum())
    gap_penalties = compute_vector_binance_gap_penalty(df["binance_price_gap"].astype(float).values) if "binance_price_gap" in df.columns else np.zeros(len(df), dtype=float)
    lead_penalties = compute_vector_binance_lead_penalty(df["binance_lead_1h"].astype(float).values) if "binance_lead_1h" in df.columns else np.zeros(len(df), dtype=float)
    if "binance_price_gap" not in df.columns or "binance_lead_1h" not in df.columns:
        details["limitations"].append("binance penalties unavailable on some historical rows")
    pred = np.maximum(0.01, pred - gap_penalties)
    pred = np.maximum(0.01, pred - lead_penalties)
    details["binance_gap_penalty_rows"] = int((gap_penalties > 0).sum())
    details["binance_lead_penalty_rows"] = int((lead_penalties > 0).sum())
    details["binance_penalty_rows"] = int(((gap_penalties > 0) | (lead_penalties > 0)).sum())
    details["avg_binance_penalty"] = round(float((gap_penalties + lead_penalties).mean()), 6)
    sector_penalties = np.zeros(len(df), dtype=float)
    if "sector_penalty_proxy" in df.columns:
        sector_penalties = np.clip(df["sector_penalty_proxy"].astype(float).values, 0.0, 0.08)
        details["limitations"].append("sector penalty proxied, not replayed from live sector leaders")
    else:
        details["limitations"].append("sector penalty not simulated on holdout data")
    pred = np.maximum(0.01, pred - sector_penalties)
    details["sector_penalty_rows"] = int((sector_penalties > 0).sum())
    details["parity_summary"] = "exact for panic and Binance penalties; guardrail cap exact only when capped flags exist; sector penalty remains offline-limited"
    return np.clip(pred, 0.01, 0.95), details
