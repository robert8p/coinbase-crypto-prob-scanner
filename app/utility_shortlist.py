from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

from .config import AppConfig
from .persist import read_json
from .runtime_scope import current_runtime_scope, scope_key
from .version import APP_VERSION

TRUST_MULTIPLIER = {
    "validated_tail_probability": 1.00,
    "calibrated_below_tail": 0.92,
    "tail_caution": 0.78,
    "ranking_only": 0.55,
}



def _config_to_dict(config: object) -> dict:
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    data = getattr(config, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    slots = getattr(type(config), "__slots__", None) or getattr(config, "__slots__", None) or []
    if isinstance(slots, str):
        slots = [slots]
    slot_payload = {name: getattr(config, name) for name in slots if isinstance(name, str) and hasattr(config, name)}
    if slot_payload:
        return slot_payload
    payload = {}
    for name in dir(config):
        if name.startswith('_'):
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        payload[name] = value
    return payload


LIQUIDITY_PENALTY = {
    "tier1": 0.00,
    "tier2": 0.03,
    "tier3": 0.08,
}

SCORE_BAND_BONUS = {
    "validated": 0.04,
    "near_validated": 0.02,
    "exploratory": 0.00,
}

ACTIONABILITY_BONUS = {
    "action_ready": 0.03,
    "selective": 0.00,
    "watchlist": -0.02,
}



def _parse_iso(value: object) -> datetime | None:
    try:
        if value in (None, ''):
            return None
        text = str(value).replace('Z', '+00:00')
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _override_matches_current_scope(model_dir: str | Path, raw: dict, *, app_version: str = APP_VERSION) -> bool:
    current_scope = current_runtime_scope(model_dir, app_version=app_version)
    current_scope_key = current_scope.get('state_scope_key')
    override_scope_key = raw.get('state_scope_key') or scope_key(raw.get('app_version'), raw.get('deployed_since_utc'))
    if current_scope_key and override_scope_key and current_scope_key != override_scope_key:
        return False
    return True


def load_active_utility_tuning_proof_override(model_dir: str | Path, *, app_version: str = APP_VERSION) -> dict:
    path = Path(model_dir) / 'runtime_live_overrides.json'
    raw = read_json(path, {})
    if not isinstance(raw, dict) or raw.get('source') not in {'utility_tuning_proof', 'utility_model_proof'}:
        return {}
    if not bool(raw.get('proof_window_active')):
        return {}
    if not _override_matches_current_scope(model_dir, raw, app_version=app_version):
        return {}
    expires_at = _parse_iso(raw.get('expires_at_utc'))
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        return {}
    return raw


def load_active_utility_tuning_override(model_dir: str | Path, *, app_version: str = APP_VERSION) -> dict:
    path = Path(model_dir) / 'runtime_live_overrides.json'
    raw = read_json(path, {})
    if not isinstance(raw, dict):
        return {}
    source = str(raw.get('source') or '')
    if source in {'utility_tuning_proof', 'utility_model_proof'}:
        return load_active_utility_tuning_proof_override(model_dir, app_version=app_version)
    if source not in {'utility_tuning_adoption', 'utility_model_adoption'}:
        return {}
    adopted_flag = bool(raw.get('adopted_tuned_bundle_active')) or bool(raw.get('adopted_utility_model_active'))
    if not adopted_flag:
        return {}
    if not _override_matches_current_scope(model_dir, raw, app_version=app_version):
        return {}
    return raw


def utility_config_with_runtime_override(config: AppConfig, override: dict | None = None) -> AppConfig | SimpleNamespace:
    override = dict(override or {})
    if not override:
        return config
    merged = _config_to_dict(config)
    merged.update({
        'utility_expected_edge_weight': float(override.get('utility_expected_edge_weight') or getattr(config, 'utility_expected_edge_weight', 0.52)),
        'utility_confidence_weight': float(override.get('utility_confidence_weight') or getattr(config, 'utility_confidence_weight', 0.30)),
        'utility_probability_weight': float(override.get('utility_probability_weight') or getattr(config, 'utility_probability_weight', 0.18)),
        'utility_scan_readiness_floor': float(override.get('utility_scan_readiness_floor') or getattr(config, 'utility_scan_readiness_floor', 0.57)),
        'utility_pairwise_margin_floor': float(override.get('utility_pairwise_margin_floor') or getattr(config, 'utility_pairwise_margin_floor', 0.055)),
        'utility_pairwise_margin_soft_floor': float(override.get('utility_pairwise_margin_soft_floor') or getattr(config, 'utility_pairwise_margin_soft_floor', 0.03)),
        'utility_multi_name_relaxation': float(override.get('utility_multi_name_relaxation') or getattr(config, 'utility_multi_name_relaxation', 0.06)),
        'utility_strong_support_count_min': int(override.get('utility_strong_support_count_min') or getattr(config, 'utility_strong_support_count_min', 2)),
        'utility_moderate_support_count_min': int(override.get('utility_moderate_support_count_min') or getattr(config, 'utility_moderate_support_count_min', 2)),
        'utility_strong_top_live_floor': float(override.get('utility_strong_top_live_floor') or getattr(config, 'utility_strong_top_live_floor', 0.42)),
        'utility_moderate_top_live_floor': float(override.get('utility_moderate_top_live_floor') or getattr(config, 'utility_moderate_top_live_floor', 0.34)),
        'utility_weak_top_live_floor': float(override.get('utility_weak_top_live_floor') or getattr(config, 'utility_weak_top_live_floor', 0.28)),
        'utility_shortlist_target_max_names': int(override.get('utility_shortlist_target_max_names') or getattr(config, 'utility_shortlist_target_max_names', 8)),
        'utility_shortlist_score_floor': float(override.get('utility_shortlist_score_floor') or getattr(config, 'utility_shortlist_score_floor', 0.52)),
        'utility_shortlist_score_dropoff': float(override.get('utility_shortlist_score_dropoff') or getattr(config, 'utility_shortlist_score_dropoff', 0.16)),
        'utility_confidence_floor': float(override.get('utility_confidence_floor') or getattr(config, 'utility_confidence_floor', 0.35)),
        'utility_tier3_max_frac': float(override.get('utility_tier3_max_frac') or getattr(config, 'utility_tier3_max_frac', 0.25)),
        'utility_selection_engine_label': str(override.get('utility_selection_engine_label') or 'utility_constrained_v7'),
    })
    return SimpleNamespace(**merged)


@dataclass(slots=True)
class UtilityShortlistResult:
    visible_rows: List[dict]
    trimmed_rows: List[dict]
    meta: dict


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value or 0.0)))


def _percentile_desc(values: Iterable[float]) -> list[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    if len(vals) == 1:
        return [1.0]
    ranked = sorted(enumerate(vals), key=lambda item: item[1], reverse=True)
    out = [0.0] * len(vals)
    denom = max(1, len(vals) - 1)
    for pos, (idx, _val) in enumerate(ranked):
        out[idx] = 1.0 - (pos / denom)
    return out


def annotate_rows_for_utility(rows: List[dict], config: AppConfig) -> List[dict]:
    rows = list(rows or [])
    if not rows:
        return rows

    raw_edge_values: list[float] = []
    confidence_values: list[float] = []
    model_values: list[float] = []

    for row in rows:
        live_score = float(row.get("live_score", row.get("prob_2") or 0.0) or 0.0)
        downside = float(row.get("downside_risk", row.get("risk") or 0.0) or 0.0)
        uncertainty = float(row.get("uncertainty", 0.0) or 0.0)
        risk = float(row.get("risk", 0.0) or 0.0)
        liquidity_tier = str(row.get("liquidity_tier") or "tier3").lower()
        semantics = str(row.get("probability_semantics") or row.get("contract_truth_semantics") or "ranking_only")
        score_band = str(row.get("score_band") or "exploratory")
        actionability_tier = str(row.get("actionability_tier") or "watchlist")

        risk_load = (0.55 * downside) + (0.30 * uncertainty) + (0.15 * risk)
        expected_edge = live_score - risk_load
        trust_multiplier = TRUST_MULTIPLIER.get(semantics, 0.70)
        confidence = _clamp((1.0 - risk_load - LIQUIDITY_PENALTY.get(liquidity_tier, 0.08)) * trust_multiplier)
        row["utility_expected_edge"] = round(expected_edge, 6)
        row["utility_confidence"] = round(confidence, 6)
        row["utility_reward_term"] = round(live_score * float(config.target_move_pct), 6)
        row["utility_risk_term"] = round(risk_load, 6)
        row["utility_band_bonus"] = SCORE_BAND_BONUS.get(score_band, 0.0)
        row["utility_actionability_bonus"] = ACTIONABILITY_BONUS.get(actionability_tier, 0.0)
        raw_edge_values.append(expected_edge)
        confidence_values.append(confidence)
        model_values.append(live_score)

    edge_pct = _percentile_desc(raw_edge_values)
    conf_pct = _percentile_desc(confidence_values)
    model_pct = _percentile_desc(model_values)

    for idx, row in enumerate(rows):
        band_bonus = float(row.get("utility_band_bonus") or 0.0)
        action_bonus = float(row.get("utility_actionability_bonus") or 0.0)
        decision_score = (
            float(config.utility_expected_edge_weight) * edge_pct[idx]
            + float(config.utility_confidence_weight) * conf_pct[idx]
            + float(config.utility_probability_weight) * model_pct[idx]
            + band_bonus
            + action_bonus
        )
        readiness_score = (
            0.50 * float(decision_score)
            + 0.30 * conf_pct[idx]
            + 0.20 * edge_pct[idx]
        )
        row["utility_edge_percentile"] = round(edge_pct[idx], 6)
        row["utility_confidence_percentile"] = round(conf_pct[idx], 6)
        row["utility_probability_percentile"] = round(model_pct[idx], 6)
        row["utility_scan_readiness_score"] = round(_clamp(readiness_score), 6)
        row["utility_decision_score"] = round(_clamp(decision_score), 6)
        row["selection_engine"] = str(getattr(config, "utility_selection_engine_label", "utility_constrained_v7") or "utility_constrained_v7")

    rows.sort(
        key=lambda row: (
            float(row.get("utility_decision_score") or 0.0),
            float(row.get("utility_confidence") or 0.0),
            float(row.get("utility_expected_edge") or -999.0),
            float(row.get("live_score") or 0.0),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["utility_rank"] = idx
    return rows


def optimize_visible_shortlist(
    rows: List[dict],
    *,
    effective_max: int,
    config: AppConfig,
    tracked_priority_symbols: List[str] | None = None,
) -> UtilityShortlistResult:
    rows = annotate_rows_for_utility(rows, config)
    if not rows:
        return UtilityShortlistResult([], [], {"selection_engine": str(getattr(config, "utility_selection_engine_label", "utility_constrained_v7") or "utility_constrained_v7"), "visible_cap": 0})

    tracked_set = {str(s) for s in (tracked_priority_symbols or []) if str(s)}
    visible_cap = max(0, min(int(effective_max or 0), int(config.utility_shortlist_target_max_names or 0)))
    if visible_cap == 0:
        visible_cap = max(0, int(effective_max or 0))

    top_score = float(rows[0].get("utility_decision_score") or 0.0)
    top_conf = float(rows[0].get("utility_confidence") or 0.0)
    top_edge = float(rows[0].get("utility_expected_edge") or 0.0)
    top_live = float(rows[0].get("live_score") or rows[0].get("prob_2") or 0.0)
    top2_score = float(rows[1].get("utility_decision_score") or 0.0) if len(rows) > 1 else 0.0
    top3_score = float(rows[2].get("utility_decision_score") or 0.0) if len(rows) > 2 else 0.0
    top2_live = float(rows[1].get("live_score") or rows[1].get("prob_2") or 0.0) if len(rows) > 1 else 0.0
    top3_live = float(rows[2].get("live_score") or rows[2].get("prob_2") or 0.0) if len(rows) > 2 else top2_live
    top2_edge = float(rows[1].get("utility_expected_edge") or 0.0) if len(rows) > 1 else 0.0
    lead_margin = top_live - top2_live if len(rows) > 1 else top_live
    third_margin = top_live - top3_live if len(rows) > 2 else lead_margin
    edge_margin = top_edge - top2_edge if len(rows) > 1 else top_edge
    dominance_margin = (0.45 * lead_margin) + (0.25 * third_margin) + (0.30 * edge_margin)
    scan_readiness = (0.50 * top_score) + (0.30 * top_conf) + (0.20 * _clamp(top_edge + 0.20, 0.0, 1.0))

    dynamic_floor = max(float(config.utility_shortlist_score_floor), top_score - float(config.utility_shortlist_score_dropoff))
    confidence_floor = float(config.utility_confidence_floor)
    multi_name_relaxation = float(getattr(config, 'utility_multi_name_relaxation', 0.06) or 0.06)
    scan_readiness_floor = float(getattr(config, 'utility_scan_readiness_floor', 0.57) or 0.57)
    margin_floor = float(getattr(config, 'utility_pairwise_margin_floor', 0.055) or 0.055)
    margin_soft_floor = float(getattr(config, 'utility_pairwise_margin_soft_floor', 0.03) or 0.03)

    support_floor = max(float(config.utility_shortlist_score_floor) - 0.08, 0.20)
    support_conf_floor = max(confidence_floor - 0.12, 0.16)
    support_count = sum(
        1
        for row in rows[: min(len(rows), 8)]
        if float(row.get("utility_decision_score") or 0.0) >= support_floor
        and float(row.get("utility_confidence") or 0.0) >= support_conf_floor
        and float(row.get("live_score") or row.get("prob_2") or 0.0) >= 0.30
    )

    min_visible_target = 0
    floor_relaxation = 0.0
    scan_mode = "blocked"
    pairwise_ready = False
    strong_support_min = max(1, int(getattr(config, 'utility_strong_support_count_min', 2) or 2))
    moderate_support_min = max(1, int(getattr(config, 'utility_moderate_support_count_min', 2) or 2))
    strong_top_live_floor = float(getattr(config, 'utility_strong_top_live_floor', 0.42) or 0.42)
    moderate_top_live_floor = float(getattr(config, 'utility_moderate_top_live_floor', 0.34) or 0.34)
    weak_top_live_floor = float(getattr(config, 'utility_weak_top_live_floor', 0.28) or 0.28)

    if top_score >= 0.56 and top_conf >= max(confidence_floor - 0.04, 0.24) and top_live >= strong_top_live_floor and support_count >= strong_support_min:
        scan_mode = "strong"
        pairwise_ready = True
        min_visible_target = 1
        if len(rows) > 2 and dominance_margin <= margin_soft_floor:
            min_visible_target = min(3, visible_cap or 3)
            floor_relaxation = multi_name_relaxation
        elif len(rows) > 1 and dominance_margin <= margin_floor:
            min_visible_target = min(2, visible_cap or 2)
            floor_relaxation = multi_name_relaxation * 0.5
    elif top_score >= 0.47 and top_conf >= max(confidence_floor - 0.10, 0.20) and top_live >= moderate_top_live_floor:
        scan_mode = "moderate"
        pairwise_ready = True
        min_visible_target = 1
        floor_relaxation = multi_name_relaxation * 0.5
        if len(rows) > 1 and support_count >= moderate_support_min and dominance_margin <= max(margin_floor, 0.07):
            min_visible_target = min(2, visible_cap or 2)
    elif top_score >= 0.40 and top_conf >= 0.18 and top_live >= weak_top_live_floor and top_edge >= -0.01:
        scan_mode = "weak"
        min_visible_target = 1
        floor_relaxation = multi_name_relaxation * 0.35

    if scan_mode == "strong":
        dynamic_floor = max(0.0, dynamic_floor - floor_relaxation)
        confidence_floor = max(0.0, confidence_floor - 0.04)
    elif scan_mode == "moderate":
        dynamic_floor = max(0.0, min(dynamic_floor - floor_relaxation, top_score - 0.12, 0.42))
        confidence_floor = max(0.0, confidence_floor - 0.08)
    elif scan_mode == "weak":
        dynamic_floor = max(0.0, min(dynamic_floor - floor_relaxation, top_score - 0.06, 0.34))
        confidence_floor = max(0.0, confidence_floor - 0.12)

    tier3_cap = max(0, ceil(max(1, visible_cap) * float(config.utility_tier3_max_frac))) if visible_cap else 0
    pinned_visible_cap = max(0, int(config.utility_pinned_visible_cap or 0))

    visible: list[dict] = []
    trimmed: list[dict] = []
    tier3_used = 0
    pinned_used = 0

    def _annotate_trim(row: dict, reason: str, detail: str) -> dict:
        trimmed_row = dict(row)
        trimmed_row["suppression_reason"] = reason
        trimmed_row["suppression_reason_detail"] = detail
        trimmed_row["informational_only"] = True
        trimmed_row["is_actionable_now"] = False
        return trimmed_row

    for row in rows:
        symbol = str(row.get("symbol") or "")
        decision_score = float(row.get("utility_decision_score") or 0.0)
        confidence = float(row.get("utility_confidence") or 0.0)
        liquidity_tier = str(row.get("liquidity_tier") or "tier3").lower()
        is_tracked = symbol in tracked_set
        relaxed_floor = dynamic_floor - float(config.utility_tracked_symbol_floor_relaxation) if is_tracked else dynamic_floor
        relaxed_confidence = confidence_floor - float(config.utility_tracked_symbol_confidence_relaxation) if is_tracked else confidence_floor
        if len(visible) < min_visible_target:
            relaxed_floor = max(0.0, relaxed_floor - multi_name_relaxation)
            relaxed_confidence = max(0.0, relaxed_confidence - 0.05)

        if scan_mode == "blocked" and not is_tracked:
            trimmed.append(_annotate_trim(row, "utility_scan_readiness", "scan-level readiness gate did not support surfacing a trustworthy shortlist"))
            continue
        if decision_score < relaxed_floor:
            trimmed.append(_annotate_trim(row, "utility_floor", f"decision score {decision_score:.3f} below dynamic floor {relaxed_floor:.3f}"))
            continue
        if confidence < relaxed_confidence:
            trimmed.append(_annotate_trim(row, "utility_confidence", f"confidence {confidence:.3f} below floor {relaxed_confidence:.3f}"))
            continue
        if len(visible) >= visible_cap:
            trimmed.append(_annotate_trim(row, "utility_display_trim", "trimmed to preserve a small utility-ranked visible shortlist"))
            continue
        if liquidity_tier == "tier3" and tier3_used >= tier3_cap:
            trimmed.append(_annotate_trim(row, "utility_tier3_cap", "trimmed because the tier3 quota for the visible shortlist was reached"))
            continue
        if is_tracked and pinned_visible_cap and pinned_used >= pinned_visible_cap:
            trimmed.append(_annotate_trim(row, "tracked_pin_cap", "tracked follow-up pin quota already used"))
            continue

        kept = dict(row)
        kept["visibility_reason"] = f"decision {decision_score:.3f} / confidence {confidence:.3f} cleared utility shortlist gate"
        visible.append(kept)
        if liquidity_tier == "tier3":
            tier3_used += 1
        if is_tracked:
            pinned_used += 1

    if not visible and rows and scan_mode in {"strong", "moderate", "weak"} and top_score >= max(dynamic_floor, 0.30):
        fallback = rows[0]
        fallback_score = float(fallback.get("utility_decision_score") or 0.0)
        fallback_conf = float(fallback.get("utility_confidence") or 0.0)
        fallback_live = float(fallback.get("live_score") or fallback.get("prob_2") or 0.0)
        fallback_conf_floor = max(0.0, confidence_floor - (0.03 if scan_mode == "strong" else 0.06 if scan_mode == "moderate" else 0.09))
        if fallback_score >= dynamic_floor and fallback_conf >= fallback_conf_floor and fallback_live >= 0.28:
            kept = dict(fallback)
            kept["visibility_reason"] = f"forced singleton shortlist because the top candidate cleared the {scan_mode} scan gate"
            visible = [kept]
            trimmed = [r for r in trimmed if str(r.get("symbol") or "") != str(fallback.get("symbol") or "")]

    meta = {
        "selection_engine": str(getattr(config, "utility_selection_engine_label", "utility_constrained_v7") or "utility_constrained_v7"),
        "visible_cap": visible_cap,
        "dynamic_score_floor": round(dynamic_floor, 6),
        "confidence_floor": round(confidence_floor, 6),
        "scan_readiness": round(scan_readiness, 6),
        "dominance_margin": round(dominance_margin, 6),
        "min_visible_target": int(min_visible_target),
        "pairwise_ready": bool(pairwise_ready),
        "scan_mode": scan_mode,
        "support_count": int(support_count),
        "tier3_cap": tier3_cap,
        "visible_count": len(visible),
        "trimmed_count": len(trimmed),
        "tracked_visible_symbols": [r.get("symbol") for r in visible if str(r.get("symbol") or "") in tracked_set],
        "tracked_visible_promoted": sum(1 for r in visible if str(r.get("symbol") or "") in tracked_set),
        "top_visible_symbols": [r.get("symbol") for r in visible[: min(5, len(visible))]],
        "empty_shortlist": len(visible) == 0,
    }
    return UtilityShortlistResult(visible, trimmed, meta)


def legacy_visible_shortlist(
    rows: List[dict],
    *,
    effective_max: int,
    config: AppConfig,
    tracked_priority_symbols: List[str] | None = None,
) -> UtilityShortlistResult:
    rows = [dict(r) for r in list(rows or [])]
    tracked_set = {str(s) for s in (tracked_priority_symbols or []) if str(s)}
    pin_cap = max(0, int(getattr(config, "cooldown_followup_visible_pin_count", 5) or 5))

    def _pin(rows_in: List[dict]) -> List[dict]:
        if not tracked_set:
            return list(rows_in)
        pinned = [r for r in rows_in if str(r.get("symbol") or "") in tracked_set][:pin_cap]
        pinned_symbols = {str(r.get("symbol") or "") for r in pinned}
        others = [r for r in rows_in if str(r.get("symbol") or "") not in pinned_symbols]
        return pinned + others

    def _legacy_sort_key(row: dict) -> tuple:
        action_order = {"action_ready": 3, "selective": 2, "watchlist": 1}
        return (
            action_order.get(str(row.get("actionability_tier") or "watchlist"), 1),
            float(row.get("prob_2_rank", row.get("prob_2") or 0.0) or 0.0),
            float(row.get("opportunity_score", 0.0) or 0.0),
            float(row.get("prob_2", 0.0) or 0.0),
            -float(row.get("risk", 0.0) or 0.0),
        )

    rows.sort(key=_legacy_sort_key, reverse=True)
    action_selective = [r for r in rows if str(r.get("actionability_tier") or "") in {"action_ready", "selective"}]
    watchlist = [r for r in rows if str(r.get("actionability_tier") or "") == "watchlist"]
    base_watchlist_cap = max(0, int(getattr(config, "stage2_watchlist_max_names", 12) or 12))
    watchlist_only_cap = max(0, int(getattr(config, "stage2_watchlist_only_max_names", base_watchlist_cap) or base_watchlist_cap))
    exploratory_only_cap = max(0, int(getattr(config, "stage2_watchlist_only_exploratory_max_names", 5) or 5))
    watchlist_cap = watchlist_only_cap if not action_selective else base_watchlist_cap
    kept_action = _pin(action_selective)[:effective_max] if effective_max else []
    remaining_slots = max(0, effective_max - len(kept_action))
    if action_selective:
        kept_watchlist = _pin(watchlist)[: min(remaining_slots, watchlist_cap)]
    else:
        near_watchlist = _pin([r for r in watchlist if str(r.get("score_band") or "") == "near_validated"])
        exploratory_watchlist = _pin([r for r in watchlist if str(r.get("score_band") or "") != "near_validated"])
        near_cap = min(remaining_slots, watchlist_cap)
        kept_near = near_watchlist[:near_cap]
        remaining_after_near = max(0, near_cap - len(kept_near))
        kept_exploratory = exploratory_watchlist[: min(remaining_after_near, exploratory_only_cap)]
        kept_watchlist = kept_near + kept_exploratory
    pre_cap_visible = kept_action + kept_watchlist
    max_symbol_share = 0.25
    absolute_symbol_cap = 2
    visible_symbol_counts: dict[str, int] = {}
    visible: List[dict] = []
    trimmed: List[dict] = []
    for idx, row in enumerate(pre_cap_visible, start=1):
        symbol = str(row.get("symbol") or "")
        already = int(visible_symbol_counts.get(symbol, 0))
        share_cap = max(1, int(max(1, len(pre_cap_visible)) * max_symbol_share))
        effective_symbol_cap = max(1, min(absolute_symbol_cap, share_cap))
        top5_cap = 2 if idx <= 5 else effective_symbol_cap
        allowed = min(effective_symbol_cap, top5_cap)
        if already >= allowed:
            trimmed_row = dict(row)
            trimmed_row["suppression_reason"] = "symbol_concentration"
            trimmed_row["suppression_reason_detail"] = "trimmed to prevent one symbol from dominating the visible shortlist"
            trimmed_row["informational_only"] = True
            trimmed_row["is_actionable_now"] = False
            trimmed.append(trimmed_row)
            continue
        visible_symbol_counts[symbol] = already + 1
        visible.append(dict(row))
    visible_symbols = {str(r.get("symbol") or "") for r in visible}
    for row in rows:
        symbol = str(row.get("symbol") or "")
        if symbol in visible_symbols:
            continue
        trimmed_row = dict(row)
        trimmed_row["suppression_reason"] = trimmed_row.get("suppression_reason") or "legacy_trim"
        trimmed_row["suppression_reason_detail"] = trimmed_row.get("suppression_reason_detail") or "trimmed by legacy ranked-cap shortlist"
        trimmed_row["informational_only"] = True
        trimmed_row["is_actionable_now"] = False
        trimmed.append(trimmed_row)
    meta = {
        "selection_engine": "legacy_ranked_cap_v1",
        "visible_cap": effective_max,
        "visible_count": len(visible),
        "trimmed_count": len(trimmed),
        "tracked_visible_symbols": [r.get("symbol") for r in visible if str(r.get("symbol") or "") in tracked_set],
    }
    return UtilityShortlistResult(visible, trimmed, meta)
