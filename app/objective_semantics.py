from __future__ import annotations

from pathlib import Path
from typing import Any

from .persist import read_json


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _bucket_rate(bucket: dict | None) -> float | None:
    if not isinstance(bucket, dict):
        return None
    return _f(bucket.get("quality_rate"))


def _bucket_floor(bucket: dict | None) -> float | None:
    if not isinstance(bucket, dict):
        return None
    return _f(bucket.get("score_min"))


def load_objective_semantics_contract(
    model_dir: str | Path,
    *,
    live_threshold: float,
    stage1_selection_mode: str | None = None,
) -> dict:
    model_dir = Path(model_dir)
    baseline = read_json(model_dir / "raw_score_baseline" / "latest_raw_score_baseline_summary.json", {})
    if not isinstance(baseline, dict) or not baseline.get("available"):
        return {}

    diagnosis = dict(baseline.get("diagnosis") or {})
    ranking_strength = str(diagnosis.get("ranking_strength") or "")
    if str(diagnosis.get("primary_blocker") or "") != "calibration_semantics":
        return {}
    if ranking_strength not in {"strong", "moderate"}:
        return {}
    if bool(diagnosis.get("compression_significant", False)):
        return {}

    raw_dist = dict(baseline.get("raw_model_score_distribution") or {})
    top_buckets = dict(raw_dist.get("top_bucket_quality_rate") or {})
    if not top_buckets:
        return {}

    source_replay = dict(baseline.get("source_replay") or {})
    replay_threshold = _f(source_replay.get("raw_threshold"))
    threshold_matches = replay_threshold is not None and abs(float(replay_threshold) - float(live_threshold)) <= 0.03

    checkpoint = read_json(model_dir / "decision_checkpoint_summary.json", {})
    checkpoint_confirmed = False
    checkpoint_quality_rate = None
    checkpoint_visible_rows = None
    if isinstance(checkpoint, dict):
        checkpoint_outcome = str(checkpoint.get("decision_checkpoint_outcome") or checkpoint.get("current_outcome") or "")
        checkpoint_threshold = _f(checkpoint.get("live_raw_threshold"))
        checkpoint_mode = str(checkpoint.get("stage1_selection_mode") or "")
        mode_matches = not stage1_selection_mode or not checkpoint_mode or checkpoint_mode == str(stage1_selection_mode)
        threshold_ok = checkpoint_threshold is not None and abs(float(checkpoint_threshold) - float(live_threshold)) <= 0.03
        checkpoint_confirmed = checkpoint_outcome == "confirmed" and threshold_ok and mode_matches
        checkpoint_quality_rate = _f(checkpoint.get("current_visible_quality_hit_rate")) if checkpoint_confirmed else None
        checkpoint_visible_rows = int(checkpoint.get("resolved_visible_rows") or 0) if checkpoint_confirmed else None

    confirmed_shortlist_floor = round(float(live_threshold), 4) if threshold_matches else None
    strong_edge_floor = _bucket_floor(top_buckets.get("top_10pct"))
    priority_edge_floor = _bucket_floor(top_buckets.get("top_5pct"))
    elite_edge_floor = _bucket_floor(top_buckets.get("top_1pct"))

    if strong_edge_floor is not None:
        strong_edge_floor = max(round(float(strong_edge_floor), 4), round(float(live_threshold), 4))
    if priority_edge_floor is not None:
        priority_edge_floor = max(round(float(priority_edge_floor), 4), round(float(live_threshold), 4))
    if elite_edge_floor is not None:
        elite_edge_floor = max(round(float(elite_edge_floor), 4), round(float(live_threshold), 4))

    if strong_edge_floor is not None and confirmed_shortlist_floor is not None and strong_edge_floor <= confirmed_shortlist_floor + 0.005:
        strong_edge_floor = None
    if priority_edge_floor is not None and strong_edge_floor is not None and priority_edge_floor <= strong_edge_floor + 0.005:
        priority_edge_floor = None
    if elite_edge_floor is not None and priority_edge_floor is not None and elite_edge_floor <= priority_edge_floor + 0.005:
        elite_edge_floor = None

    return {
        "available": True,
        "source": "raw_score_baseline",
        "generated_at_utc": baseline.get("generated_at_utc"),
        "headline": diagnosis.get("headline"),
        "primary_blocker": diagnosis.get("primary_blocker"),
        "ranking_strength": ranking_strength,
        "tail_state": diagnosis.get("tail_state"),
        "base_quality_rate": _f(baseline.get("base_quality_rate")),
        "confirmed_shortlist_floor": confirmed_shortlist_floor,
        "confirmed_shortlist_quality_reference": checkpoint_quality_rate,
        "confirmed_shortlist_quality_source": "decision_checkpoint_confirmed" if checkpoint_quality_rate is not None else None,
        "confirmed_shortlist_visible_rows": checkpoint_visible_rows,
        "strong_edge_floor": strong_edge_floor,
        "strong_edge_quality_reference": _bucket_rate(top_buckets.get("top_10pct")),
        "strong_edge_quality_source": "raw_score_baseline_top_10pct",
        "priority_edge_floor": priority_edge_floor,
        "priority_edge_quality_reference": _bucket_rate(top_buckets.get("top_5pct")),
        "priority_edge_quality_source": "raw_score_baseline_top_5pct",
        "elite_edge_floor": elite_edge_floor,
        "elite_edge_quality_reference": _bucket_rate(top_buckets.get("top_1pct")),
        "elite_edge_quality_source": "raw_score_baseline_top_1pct",
        "scan_topk_quality": dict(baseline.get("scan_topk_quality") or {}),
        "recommended_next_tranche": diagnosis.get("recommended_next_tranche"),
    }


def score_objective_band(*, live_score: float, contract: dict | None, near_gap: float = 0.08) -> dict:
    contract = dict(contract or {})
    score = float(live_score or 0.0)
    confirmed_floor = _f(contract.get("confirmed_shortlist_floor"))
    strong_floor = _f(contract.get("strong_edge_floor"))
    priority_floor = _f(contract.get("priority_edge_floor"))
    elite_floor = _f(contract.get("elite_edge_floor"))

    band = None
    label = None
    reference_rate = None
    reference_source = None
    floor = confirmed_floor
    monitor_priority = "below_confirmed_shortlist"

    if elite_floor is not None and score >= elite_floor:
        band = "elite_edge"
        label = "Elite edge band"
        floor = elite_floor
        reference_rate = _f(contract.get("elite_edge_quality_reference"))
        reference_source = contract.get("elite_edge_quality_source")
        monitor_priority = "elite_edge"
    elif priority_floor is not None and score >= priority_floor:
        band = "priority_edge"
        label = "Priority edge band"
        floor = priority_floor
        reference_rate = _f(contract.get("priority_edge_quality_reference"))
        reference_source = contract.get("priority_edge_quality_source")
        monitor_priority = "priority_edge"
    elif strong_floor is not None and score >= strong_floor:
        band = "strong_edge"
        label = "Strong edge band"
        floor = strong_floor
        reference_rate = _f(contract.get("strong_edge_quality_reference"))
        reference_source = contract.get("strong_edge_quality_source")
        monitor_priority = "strong_edge"
    elif confirmed_floor is not None and score >= confirmed_floor:
        band = "confirmed_shortlist"
        label = "Confirmed shortlist band"
        floor = confirmed_floor
        reference_rate = _f(contract.get("confirmed_shortlist_quality_reference"))
        reference_source = contract.get("confirmed_shortlist_quality_source")
        monitor_priority = "confirmed_shortlist"
    elif confirmed_floor is not None and score >= max(0.0, confirmed_floor - float(near_gap or 0.08)):
        band = "near_confirmed_shortlist"
        label = "Near shortlist threshold"
        floor = confirmed_floor
        reference_rate = _f(contract.get("confirmed_shortlist_quality_reference"))
        reference_source = contract.get("confirmed_shortlist_quality_source")
        monitor_priority = "near_shortlist"
    else:
        band = "below_confirmed_shortlist"
        label = "Below confirmed shortlist"
        monitor_priority = "below_confirmed_shortlist"

    gap = max(0.0, float(floor) - score) if floor is not None else None
    return {
        "objective_score_band": band,
        "objective_score_band_label": label,
        "objective_monitor_priority": monitor_priority,
        "objective_quality_reference_rate": round(float(reference_rate), 6) if reference_rate is not None else None,
        "objective_quality_reference_source": reference_source,
        "objective_distance_to_confirmed_shortlist": round(float(gap), 4) if gap is not None else None,
        "objective_distance_to_confirmed_shortlist_pct_points": round(float(gap) * 100.0, 2) if gap is not None else None,
        "objective_confirmed_shortlist_floor": round(float(confirmed_floor), 4) if confirmed_floor is not None else None,
        "objective_strong_edge_floor": round(float(strong_floor), 4) if strong_floor is not None else None,
        "objective_priority_edge_floor": round(float(priority_floor), 4) if priority_floor is not None else None,
        "objective_elite_edge_floor": round(float(elite_floor), 4) if elite_floor is not None else None,
    }
