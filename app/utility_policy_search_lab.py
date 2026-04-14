from __future__ import annotations

import json
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .replay import HistoricalReplayService
from .review_runs import ReviewPackService
from .utility_shortlist import (
    annotate_rows_for_utility,
    legacy_visible_shortlist,
    optimize_visible_shortlist,
    utility_config_with_runtime_override,
)
from .version import APP_VERSION


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


class UtilityPolicySearchLabService:
    def __init__(self, config: AppConfig, replay: HistoricalReplayService, review_packs: ReviewPackService):
        self.config = config
        self.replay = replay
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / "utility_policy_search_lab")
        self.summary_path = self.root_dir / "latest_utility_policy_search_lab_summary.json"
        self.pack_path = self.root_dir / "latest_utility_policy_search_lab_pack.zip"
        self.status_path = self.root_dir / "latest_utility_policy_search_lab_status.json"
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def latest_status(self) -> dict:
        payload = read_json(self.status_path, {})
        summary = self.latest_summary()
        if payload.get("active"):
            return payload
        ready = bool(summary.get("available"))
        if ready:
            verdict = str(summary.get("verdict") or "")
            winner = dict(summary.get("winner") or {})
            winner_name = str(
                winner.get("policy_name")
                or winner.get("engine_label")
                or winner.get("engine")
                or ""
            ).strip()
            status = dict(payload or {})
            status.update({
                "available": True,
                "active": False,
                "status": "completed",
                "headline": "Utility Policy Search completed",
                "summary": (
                    f"Completed. Winner: {winner_name}."
                    if verdict == "supported_policy_found" and winner_name
                    else "Completed. No supported policy found."
                ),
                "phase": "completed",
                "result_ready": True,
                "pack_ready": self.latest_pack() is not None,
                "summary_ready": True,
                "last_error": None,
                "updated_at_utc": _utc_now_iso(),
                "last_completed_at_utc": summary.get("generated_at_utc"),
                "progress_pct": 100,
                "current_step": status.get("total_steps") or status.get("current_step") or 0,
                "total_steps": status.get("total_steps") or status.get("current_step") or 0,
                "current_policy": None,
                "verdict": verdict or None,
                "winner_policy_name": winner_name or None,
            })
            return status
        return {
            "available": True,
            "active": False,
            "status": "idle",
            "headline": "Utility Policy Search idle",
            "summary": "No policy-search run has started yet.",
            "phase": "idle",
            "result_ready": False,
            "pack_ready": False,
            "summary_ready": False,
            "last_error": None,
            "updated_at_utc": _utc_now_iso(),
            "last_completed_at_utc": None,
            "progress_pct": 0,
            "current_step": None,
            "total_steps": None,
            "current_policy": None,
        }

    def start_run(self, *, hours: int = 168, step_minutes: int = 120, max_scans: int = 84, max_symbols: int = 100) -> dict:
        with self._lock:
            status = self.latest_status()
            if status.get("active"):
                return status
            run_id = f"utility-policy-search-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            total_steps = len(self._candidate_policies()) + 4
            payload = {
                "available": True,
                "active": True,
                "status": "running",
                "headline": "Utility Policy Search is running",
                "summary": "Run started. Progress will update automatically.",
                "phase": "starting",
                "run_id": run_id,
                "started_at_utc": _utc_now_iso(),
                "updated_at_utc": _utc_now_iso(),
                "last_completed_at_utc": status.get("last_completed_at_utc"),
                "result_ready": False,
                "pack_ready": False,
                "summary_ready": False,
                "last_error": None,
                "current_step": 0,
                "total_steps": total_steps,
                "progress_pct": 0,
                "current_policy": None,
                "inputs": {
                    "hours": int(hours),
                    "step_minutes": int(step_minutes),
                    "max_scans": int(max_scans),
                    "max_symbols": int(max_symbols),
                },
            }
            self._write_status(payload)
            self._worker_thread = threading.Thread(
                target=self._run_background,
                kwargs={
                    "run_id": run_id,
                    "hours": int(hours),
                    "step_minutes": int(step_minutes),
                    "max_scans": int(max_scans),
                    "max_symbols": int(max_symbols),
                },
                daemon=True,
                name="utility-policy-search-lab",
            )
            self._worker_thread.start()
            return payload

    def run(self, *, hours: int = 168, step_minutes: int = 120, max_scans: int = 84, max_symbols: int = 100) -> dict:
        try:
            current_version = self.review_packs.get_current_version_summary() or {}
        except FileNotFoundError:
            current_version = {}
        live_threshold = self._current_live_threshold(current_version)
        policies = self._candidate_policies()
        total_policy_count = max(len(policies), 1)
        total_steps = total_policy_count + 4
        self._status_update(phase="running_replay", summary="Building replay frame for policy search.", current_step=1, total_steps=total_steps, progress_pct=5)
        replay_result = self.replay.run(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            pipeline_mode="raw_threshold",
            raw_threshold=live_threshold,
        )
        replay_summary = dict(replay_result.get("summary") or {})
        replay_rows = list(replay_summary.get("replay_rows") or [])
        scans = self._build_scan_groups(replay_rows)

        self._status_update(phase="evaluating_legacy", summary="Scoring legacy shortlist baseline.", current_step=2, total_steps=total_steps, progress_pct=12)
        legacy_rows: list[dict] = []
        for as_of, scan_rows_all in scans.items():
            candidate_pool = self._candidate_pool(scan_rows_all)
            regime_state = str((scan_rows_all[0] if scan_rows_all else {}).get("market_regime_state") or "green").lower()
            effective_max = self._effective_max_for_regime(regime_state)
            legacy_result = legacy_visible_shortlist(candidate_pool, effective_max=effective_max, config=self.config, tracked_priority_symbols=[])
            legacy_eval = self._evaluate_engine(as_of=as_of, visible_rows=legacy_result.visible_rows, pool_rows=candidate_pool)
            legacy_rows.extend(legacy_eval["visible_rows"] + legacy_eval["hidden_rows"])
        legacy_summary = self._engine_summary("legacy_ranked_cap_v1", legacy_rows)

        ranked_policies: list[dict] = []
        policy_rows: dict[str, list[dict]] = {}
        for idx, policy in enumerate(policies, start=1):
            self._status_update(
                phase="evaluating_policy",
                summary=f"Evaluating policy {idx} of {total_policy_count}: {policy.get('policy_name') or policy.get('policy_id')}",
                current_step=2 + idx,
                total_steps=total_steps,
                progress_pct=min(12 + int((idx / total_policy_count) * 72), 86),
                current_policy={"policy_id": policy.get("policy_id"), "policy_name": policy.get("policy_name")},
            )
            override = dict(policy.get("override") or {})
            override["utility_selection_engine_label"] = policy["policy_id"]
            policy_config = utility_config_with_runtime_override(self.config, override)
            rows: list[dict] = []
            for as_of, scan_rows_all in scans.items():
                candidate_pool = self._policy_candidate_pool(scan_rows_all, policy)
                regime_state = str((scan_rows_all[0] if scan_rows_all else {}).get("market_regime_state") or "green").lower()
                effective_max = self._policy_effective_max_for_regime(policy, regime_state)
                result = optimize_visible_shortlist(candidate_pool, effective_max=effective_max, config=policy_config, tracked_priority_symbols=[])
                evaluation = self._evaluate_engine(as_of=as_of, visible_rows=result.visible_rows, pool_rows=candidate_pool)
                rows.extend(evaluation["visible_rows"] + evaluation["hidden_rows"])
            summary = self._engine_summary(policy["policy_id"], rows)
            summary["policy_name"] = policy["policy_name"]
            summary["policy_notes"] = policy["notes"]
            summary["override"] = override
            summary["utility_score_delta_vs_legacy"] = self._delta(summary.get("scan_shortlist_utility_score"), legacy_summary.get("scan_shortlist_utility_score"))
            summary["pairwise_delta_vs_legacy"] = self._delta(summary.get("scan_shortlist_pairwise_win_rate"), legacy_summary.get("scan_shortlist_pairwise_win_rate"))
            summary["mean_gap_delta_vs_legacy"] = self._delta(summary.get("scan_shortlist_mean_gap"), legacy_summary.get("scan_shortlist_mean_gap"))
            summary["avg_visible_rows_delta_vs_legacy"] = self._delta(summary.get("scan_shortlist_avg_visible_rows_per_scan"), legacy_summary.get("scan_shortlist_avg_visible_rows_per_scan"))
            summary["support_level"] = self._support_level(summary, legacy_summary)
            ranked_policies.append(summary)
            policy_rows[policy["policy_id"]] = rows

        ranked_policies.sort(
            key=lambda item: (
                _f(item.get("scan_shortlist_utility_score"), -9.0) or -9.0,
                _f(item.get("scan_shortlist_pairwise_win_rate"), -9.0) or -9.0,
                _f(item.get("scan_shortlist_mean_gap"), -9.0) or -9.0,
                -abs((_f(item.get("scan_shortlist_avg_visible_rows_per_scan"), 0.0) or 0.0) - 2.0),
            ),
            reverse=True,
        )
        supported = [item for item in ranked_policies if str(item.get("support_level") or "") == "supported_offline"]
        winner = supported[0] if supported else (ranked_policies[0] if ranked_policies else {})
        family_results, supported_families = self._family_rankings(ranked_policies)

        self._status_update(phase="assembling_results", summary="Assembling ranked results.", current_step=total_policy_count + 3, total_steps=total_steps, progress_pct=92, current_policy=None)
        summary = self._build_summary(
            replay_summary=replay_summary,
            legacy_summary=legacy_summary,
            ranked_policies=ranked_policies,
            family_results=family_results,
            winner=winner,
            supported=supported,
            supported_families=supported_families,
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
        )
        atomic_write_json(self.summary_path, summary)
        self._status_update(phase="building_pack", summary="Building downloadable pack.", current_step=total_policy_count + 4, total_steps=total_steps, progress_pct=97, current_policy=None)
        self._build_pack(summary, replay_summary, legacy_rows, policy_rows)
        return summary

    def _write_status(self, payload: dict) -> None:
        atomic_write_json(self.status_path, payload)

    def _status_update(self, **updates: Any) -> dict:
        payload = self.latest_status()
        payload.update(updates)
        payload["updated_at_utc"] = _utc_now_iso()
        self._write_status(payload)
        return payload

    def _run_background(self, *, run_id: str, hours: int, step_minutes: int, max_scans: int, max_symbols: int) -> None:
        try:
            self._status_update(
                run_id=run_id,
                active=True,
                status="running",
                headline="Utility Policy Search is running",
                summary="Starting replay frame for policy search.",
                phase="starting_replay",
                current_step=1,
                total_steps=len(self._candidate_policies()) + 4,
                progress_pct=5,
                current_policy=None,
                result_ready=False,
                pack_ready=False,
                summary_ready=False,
                last_error=None,
            )
            summary = self.run(hours=hours, step_minutes=step_minutes, max_scans=max_scans, max_symbols=max_symbols)
            winner = dict(summary.get("winner") or {})
            self._status_update(
                run_id=run_id,
                active=False,
                status="completed",
                headline="Utility Policy Search completed",
                summary=f"Completed. Winner: {winner.get('policy_name') or winner.get('engine_label') or 'n/a'}.",
                phase="completed",
                current_step=self.latest_status().get("total_steps") or 100,
                total_steps=self.latest_status().get("total_steps") or 100,
                progress_pct=100,
                current_policy=None,
                result_ready=True,
                pack_ready=self.latest_pack() is not None,
                summary_ready=True,
                last_error=None,
                last_completed_at_utc=summary.get("generated_at_utc"),
            )
        except Exception as exc:
            self._status_update(
                run_id=run_id,
                active=False,
                status="error",
                headline="Utility Policy Search failed",
                summary="The run stopped before completion. Review the error below.",
                phase="error",
                last_error=str(exc),
                result_ready=False,
                pack_ready=self.latest_pack() is not None,
                summary_ready=bool(self.latest_summary().get("available")),
            )

    def _candidate_policies(self) -> list[dict]:
        seeds = [
            {
                "family": "edge_conf",
                "family_name": "Edge-confidence family",
                "family_notes": "Confidence-and-edge weighted shortlist family with density variants.",
                "policy_name": "Edge-confidence frontier",
                "notes": "Searches around the current winner by reweighting edge and confidence while loosening scan gating.",
                "override": {
                    "utility_expected_edge_weight": 0.58,
                    "utility_confidence_weight": 0.30,
                    "utility_probability_weight": 0.12,
                    "utility_scan_readiness_floor": 0.53,
                    "utility_pairwise_margin_floor": 0.05,
                    "utility_pairwise_margin_soft_floor": 0.025,
                    "utility_multi_name_relaxation": 0.065,
                    "utility_shortlist_target_max_names": 3,
                    "utility_shortlist_score_floor": 0.49,
                    "utility_shortlist_score_dropoff": 0.16,
                    "utility_confidence_floor": 0.31,
                    "utility_strong_support_count_min": 2,
                    "utility_moderate_support_count_min": 2,
                    "utility_strong_top_live_floor": 0.42,
                    "utility_moderate_top_live_floor": 0.34,
                    "utility_weak_top_live_floor": 0.28,
                },
                "filters": {},
                "regime_scales": {"green": 1.0, "amber": 1.0, "red": 1.0},
                "variants": [
                    {"suffix": "dense_a", "policy_name": "Edge-confidence dense A", "utility_scan_readiness_floor": 0.50, "utility_shortlist_target_max_names": 4, "utility_shortlist_score_floor": 0.46, "utility_confidence_floor": 0.28, "utility_strong_support_count_min": 1, "utility_moderate_support_count_min": 1, "utility_moderate_top_live_floor": 0.30, "utility_weak_top_live_floor": 0.25, "utility_multi_name_relaxation": 0.10},
                    {"suffix": "dense_b", "policy_name": "Edge-confidence dense B", "utility_scan_readiness_floor": 0.48, "utility_shortlist_target_max_names": 5, "utility_shortlist_score_floor": 0.44, "utility_confidence_floor": 0.26, "utility_strong_support_count_min": 1, "utility_moderate_support_count_min": 1, "utility_strong_top_live_floor": 0.38, "utility_moderate_top_live_floor": 0.29, "utility_weak_top_live_floor": 0.24, "utility_multi_name_relaxation": 0.12},
                    {"suffix": "quality_a", "policy_name": "Edge-confidence quality A", "utility_scan_readiness_floor": 0.55, "utility_shortlist_target_max_names": 3, "utility_shortlist_score_floor": 0.50, "utility_confidence_floor": 0.32, "utility_multi_name_relaxation": 0.07},
                ],
            },
            {
                "family": "balanced",
                "family_name": "Balanced family",
                "family_notes": "Balances density and trustworthiness with broader shortlist caps.",
                "policy_name": "Balanced frontier",
                "notes": "Balances density and trustworthiness with broader shortlist caps.",
                "override": {
                    "utility_expected_edge_weight": 0.50,
                    "utility_confidence_weight": 0.28,
                    "utility_probability_weight": 0.22,
                    "utility_scan_readiness_floor": 0.52,
                    "utility_pairwise_margin_floor": 0.045,
                    "utility_pairwise_margin_soft_floor": 0.022,
                    "utility_multi_name_relaxation": 0.08,
                    "utility_shortlist_target_max_names": 4,
                    "utility_shortlist_score_floor": 0.47,
                    "utility_shortlist_score_dropoff": 0.18,
                    "utility_confidence_floor": 0.28,
                    "utility_strong_support_count_min": 1,
                    "utility_moderate_support_count_min": 1,
                    "utility_strong_top_live_floor": 0.39,
                    "utility_moderate_top_live_floor": 0.31,
                    "utility_weak_top_live_floor": 0.25,
                },
                "filters": {},
                "regime_scales": {"green": 1.0, "amber": 0.9, "red": 0.8},
                "variants": [
                    {"suffix": "dense_a", "policy_name": "Balanced dense A", "utility_shortlist_target_max_names": 5, "utility_shortlist_score_floor": 0.43, "utility_confidence_floor": 0.24, "utility_multi_name_relaxation": 0.12},
                    {"suffix": "quality_a", "policy_name": "Balanced quality A", "utility_shortlist_target_max_names": 3, "utility_shortlist_score_floor": 0.49, "utility_confidence_floor": 0.30, "utility_scan_readiness_floor": 0.54},
                ],
            },
            {
                "family": "pairwise",
                "family_name": "Pairwise family",
                "family_notes": "Tests whether a broader compact cluster improves scan-level pairwise wins.",
                "policy_name": "Pairwise frontier",
                "notes": "Tests whether a broader compact cluster improves scan-level pairwise wins.",
                "override": {
                    "utility_expected_edge_weight": 0.46,
                    "utility_confidence_weight": 0.24,
                    "utility_probability_weight": 0.30,
                    "utility_scan_readiness_floor": 0.50,
                    "utility_pairwise_margin_floor": 0.035,
                    "utility_pairwise_margin_soft_floor": 0.018,
                    "utility_multi_name_relaxation": 0.10,
                    "utility_shortlist_target_max_names": 5,
                    "utility_shortlist_score_floor": 0.45,
                    "utility_shortlist_score_dropoff": 0.20,
                    "utility_confidence_floor": 0.24,
                    "utility_strong_support_count_min": 1,
                    "utility_moderate_support_count_min": 1,
                    "utility_strong_top_live_floor": 0.36,
                    "utility_moderate_top_live_floor": 0.28,
                    "utility_weak_top_live_floor": 0.23,
                },
                "filters": {},
                "regime_scales": {"green": 1.0, "amber": 1.0, "red": 0.9},
                "variants": [
                    {"suffix": "dense_a", "policy_name": "Pairwise dense A", "utility_shortlist_target_max_names": 6, "utility_shortlist_score_floor": 0.42, "utility_confidence_floor": 0.22, "utility_multi_name_relaxation": 0.14},
                    {"suffix": "quality_a", "policy_name": "Pairwise quality A", "utility_shortlist_target_max_names": 4, "utility_shortlist_score_floor": 0.47, "utility_confidence_floor": 0.27, "utility_scan_readiness_floor": 0.52},
                ],
            },
            {
                "family": "validated_tail",
                "family_name": "Validated-tail family",
                "family_notes": "Restricts the pool to validated or near-validated candidates before shortlist optimization.",
                "policy_name": "Validated-tail frontier",
                "notes": "Tests whether legacy is beatable by prefiltering to validated/near-validated candidates.",
                "override": {
                    "utility_expected_edge_weight": 0.54,
                    "utility_confidence_weight": 0.26,
                    "utility_probability_weight": 0.20,
                    "utility_scan_readiness_floor": 0.55,
                    "utility_pairwise_margin_floor": 0.04,
                    "utility_pairwise_margin_soft_floor": 0.02,
                    "utility_multi_name_relaxation": 0.06,
                    "utility_shortlist_target_max_names": 3,
                    "utility_shortlist_score_floor": 0.50,
                    "utility_shortlist_score_dropoff": 0.14,
                    "utility_confidence_floor": 0.30,
                    "utility_strong_support_count_min": 1,
                    "utility_moderate_support_count_min": 1,
                    "utility_strong_top_live_floor": 0.40,
                    "utility_moderate_top_live_floor": 0.32,
                    "utility_weak_top_live_floor": 0.0,
                },
                "filters": {
                    "candidate_allowed_score_bands": ["validated", "near_validated"],
                    "candidate_allowed_actionability": ["action_ready", "selective"],
                    "candidate_max_risk": 0.18,
                },
                "regime_scales": {"green": 1.0, "amber": 0.85, "red": 0.6},
                "variants": [
                    {"suffix": "dense_a", "policy_name": "Validated-tail dense A", "utility_shortlist_target_max_names": 4, "utility_shortlist_score_floor": 0.47, "candidate_max_risk": 0.20},
                    {"suffix": "quality_a", "policy_name": "Validated-tail quality A", "utility_shortlist_target_max_names": 2, "utility_shortlist_score_floor": 0.53, "candidate_max_risk": 0.16},
                ],
            },
            {
                "family": "action_ready",
                "family_name": "Action-ready family",
                "family_notes": "Restricts the pool to action-ready names with explicit live-score and risk floors.",
                "policy_name": "Action-ready frontier",
                "notes": "Tests whether legacy is beatable by focusing on action-ready candidates only.",
                "override": {
                    "utility_expected_edge_weight": 0.52,
                    "utility_confidence_weight": 0.30,
                    "utility_probability_weight": 0.18,
                    "utility_scan_readiness_floor": 0.56,
                    "utility_pairwise_margin_floor": 0.045,
                    "utility_pairwise_margin_soft_floor": 0.022,
                    "utility_multi_name_relaxation": 0.05,
                    "utility_shortlist_target_max_names": 3,
                    "utility_shortlist_score_floor": 0.50,
                    "utility_shortlist_score_dropoff": 0.13,
                    "utility_confidence_floor": 0.32,
                    "utility_strong_support_count_min": 1,
                    "utility_moderate_support_count_min": 1,
                    "utility_strong_top_live_floor": 0.41,
                    "utility_moderate_top_live_floor": 0.33,
                    "utility_weak_top_live_floor": 0.0,
                },
                "filters": {
                    "candidate_allowed_actionability": ["action_ready"],
                    "candidate_min_live_score": 0.32,
                    "candidate_max_risk": 0.16,
                    "candidate_top_n_prefilter": 12,
                },
                "regime_scales": {"green": 1.0, "amber": 0.8, "red": 0.5},
                "variants": [
                    {"suffix": "dense_a", "policy_name": "Action-ready dense A", "utility_shortlist_target_max_names": 4, "candidate_top_n_prefilter": 16, "candidate_max_risk": 0.18},
                    {"suffix": "quality_a", "policy_name": "Action-ready quality A", "utility_shortlist_target_max_names": 2, "candidate_top_n_prefilter": 8, "candidate_min_live_score": 0.36},
                ],
            },
            {
                "family": "regime_responsive",
                "family_name": "Regime-responsive family",
                "family_notes": "Uses the same utility basis but materially different regime cap behavior and pool trimming.",
                "policy_name": "Regime-responsive frontier",
                "notes": "Tests whether regime-aware cap scaling changes shortlist usefulness enough to beat legacy.",
                "override": {
                    "utility_expected_edge_weight": 0.50,
                    "utility_confidence_weight": 0.28,
                    "utility_probability_weight": 0.22,
                    "utility_scan_readiness_floor": 0.51,
                    "utility_pairwise_margin_floor": 0.04,
                    "utility_pairwise_margin_soft_floor": 0.02,
                    "utility_multi_name_relaxation": 0.08,
                    "utility_shortlist_target_max_names": 4,
                    "utility_shortlist_score_floor": 0.47,
                    "utility_shortlist_score_dropoff": 0.17,
                    "utility_confidence_floor": 0.28,
                    "utility_strong_support_count_min": 1,
                    "utility_moderate_support_count_min": 1,
                    "utility_strong_top_live_floor": 0.38,
                    "utility_moderate_top_live_floor": 0.30,
                    "utility_weak_top_live_floor": 0.24,
                },
                "filters": {
                    "candidate_top_n_prefilter": 20,
                },
                "regime_scales": {"green": 1.0, "amber": 0.6, "red": 0.3},
                "variants": [
                    {"suffix": "dense_a", "policy_name": "Regime-responsive dense A", "utility_shortlist_target_max_names": 5, "candidate_top_n_prefilter": 24, "regime_cap_scale_amber": 0.7, "regime_cap_scale_red": 0.4},
                    {"suffix": "quality_a", "policy_name": "Regime-responsive quality A", "utility_shortlist_target_max_names": 3, "utility_shortlist_score_floor": 0.50, "regime_cap_scale_amber": 0.5, "regime_cap_scale_red": 0.25},
                ],
            },
        ]
        policies: list[dict] = []
        for seed in seeds:
            base_override = dict(seed["override"])
            base_id = f"utility_{seed['family']}_base_v3"
            base_override["utility_selection_engine_label"] = base_id
            filters = dict(seed.get("filters") or {})
            regime_scales = dict(seed.get("regime_scales") or {})
            policies.append({
                "policy_id": base_id,
                "policy_name": seed["policy_name"],
                "notes": seed["notes"],
                "family_id": seed["family"],
                "family_name": seed["family_name"],
                "family_notes": seed["family_notes"],
                "override": base_override,
                "filters": filters,
                "regime_scales": regime_scales,
            })
            for variant in seed.get("variants", []):
                override = dict(base_override)
                variant_filters = dict(filters)
                variant_scales = dict(regime_scales)
                suffix = str(variant.get("suffix") or "variant")
                policy_id = f"utility_{seed['family']}_{suffix}_v3"
                for k, v in variant.items():
                    if k in {"suffix", "policy_name"}:
                        continue
                    if k.startswith("candidate_"):
                        variant_filters[k] = v
                    elif k.startswith("regime_cap_scale_"):
                        state = k.replace("regime_cap_scale_", "")
                        variant_scales[state] = float(v)
                    else:
                        override[k] = v
                override["utility_selection_engine_label"] = policy_id
                policies.append({
                    "policy_id": policy_id,
                    "policy_name": str(variant.get("policy_name") or policy_id.replace('_', ' ')),
                    "notes": seed["notes"],
                    "family_id": seed["family"],
                    "family_name": seed["family_name"],
                    "family_notes": seed["family_notes"],
                    "override": override,
                    "filters": variant_filters,
                    "regime_scales": variant_scales,
                })
        return policies

    def _policy_candidate_pool(self, rows: list[dict], policy: dict) -> list[dict]:
        pool = self._candidate_pool(rows)
        if not pool:
            return []
        annotated = annotate_rows_for_utility([dict(r) for r in pool], self.config)
        filters = dict(policy.get("filters") or {})
        allowed_actionability = {str(v) for v in filters.get("candidate_allowed_actionability") or []}
        allowed_score_bands = {str(v) for v in filters.get("candidate_allowed_score_bands") or []}
        min_live_score = _f(filters.get("candidate_min_live_score"))
        max_risk = _f(filters.get("candidate_max_risk"))
        min_confidence = _f(filters.get("candidate_min_confidence"))
        filtered: list[dict] = []
        for row in annotated:
            if allowed_actionability and str(row.get("actionability_tier") or "") not in allowed_actionability:
                continue
            if allowed_score_bands and str(row.get("score_band") or "") not in allowed_score_bands:
                continue
            if min_live_score is not None and _f(row.get("live_score", row.get("prob_2")), 0.0) < min_live_score:
                continue
            if max_risk is not None and _f(row.get("risk"), 0.0) > max_risk:
                continue
            if min_confidence is not None and _f(row.get("utility_confidence"), 0.0) < min_confidence:
                continue
            filtered.append(row)
        if not filtered:
            filtered = annotated
        top_n = int(filters.get("candidate_top_n_prefilter") or 0)
        if top_n > 0:
            filtered = filtered[:top_n]
        return filtered

    def _policy_effective_max_for_regime(self, policy: dict, regime_state: str) -> int:
        base = self._effective_max_for_regime(regime_state)
        scales = dict(policy.get("regime_scales") or {})
        scale = _f(scales.get(str(regime_state).lower()), 1.0) or 1.0
        adjusted = int(round(float(base) * float(scale)))
        target_cap = int((policy.get("override") or {}).get("utility_shortlist_target_max_names") or base or 0)
        if target_cap > 0:
            adjusted = min(adjusted, target_cap)
        return max(0, adjusted)

    def _family_rankings(self, ranked_policies: list[dict]) -> tuple[list[dict], list[dict]]:
        grouped: dict[str, list[dict]] = {}
        for item in ranked_policies:
            grouped.setdefault(str(item.get("family_id") or "unknown"), []).append(dict(item))
        families: list[dict] = []
        for family_id, items in grouped.items():
            items.sort(
                key=lambda item: (
                    _f(item.get("scan_shortlist_utility_score"), -9.0) or -9.0,
                    _f(item.get("scan_shortlist_pairwise_win_rate"), -9.0) or -9.0,
                    _f(item.get("scan_shortlist_mean_gap"), -9.0) or -9.0,
                ),
                reverse=True,
            )
            winner = dict(items[0]) if items else {}
            families.append({
                "family_id": family_id,
                "family_name": winner.get("family_name") or family_id.replace("_", " ").title(),
                "family_notes": winner.get("family_notes") or winner.get("notes") or "",
                "policy_count": len(items),
                "family_support_level": winner.get("support_level") or "insufficient_evidence",
                "family_winner": winner,
                "top_policies": items[:3],
            })
        families.sort(
            key=lambda item: (
                _f(((item.get("family_winner") or {}).get("scan_shortlist_utility_score")), -9.0) or -9.0,
                _f(((item.get("family_winner") or {}).get("scan_shortlist_pairwise_win_rate")), -9.0) or -9.0,
            ),
            reverse=True,
        )
        supported = [item for item in families if str(item.get("family_support_level") or "") == "supported_offline"]
        return families, supported

    def _support_level(self, candidate: dict, legacy: dict) -> str:
        candidate_score = _f(candidate.get("scan_shortlist_utility_score"))
        legacy_score = _f(legacy.get("scan_shortlist_utility_score"))
        candidate_gap = _f(candidate.get("scan_shortlist_mean_gap"))
        legacy_gap = _f(legacy.get("scan_shortlist_mean_gap"))
        candidate_pairwise = _f(candidate.get("scan_shortlist_pairwise_win_rate"))
        legacy_pairwise = _f(legacy.get("scan_shortlist_pairwise_win_rate"))
        if None in {candidate_score, legacy_score, candidate_gap, legacy_gap, candidate_pairwise, legacy_pairwise}:
            return "insufficient_evidence"
        if (candidate_score - legacy_score) >= 0.02 and (candidate_gap - legacy_gap) >= 0.01 and (candidate_pairwise - legacy_pairwise) >= 0.03:
            return "supported_offline"
        if candidate_score >= legacy_score and candidate_pairwise >= (legacy_pairwise - 0.02):
            return "close_but_not_supported"
        return "not_supported"

    def _delta(self, value: Any, baseline: Any) -> float | None:
        if _f(value) is None or _f(baseline) is None:
            return None
        return round((_f(value, 0.0) or 0.0) - (_f(baseline, 0.0) or 0.0), 6)

    def _current_live_threshold(self, current_version: dict) -> float:
        checkpoint = current_version.get("decision_checkpoint") or current_version.get("decision_rule_checkpoint") or {}
        return _f(
            checkpoint.get("live_raw_threshold")
            or checkpoint.get("effective_live_raw_threshold")
            or (current_version.get("decision_branch_automation") or {}).get("effective_live_raw_threshold")
            or self.config.live_raw_threshold,
            0.35,
        ) or 0.35

    def _build_scan_groups(self, replay_rows: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for row in replay_rows:
            as_of = str(row.get("as_of_utc") or "")
            if not as_of:
                continue
            out.setdefault(as_of, []).append(dict(row))
        return out

    def _candidate_pool(self, rows: list[dict]) -> list[dict]:
        blocked_reasons = {"threshold", "regime", "cooldown"}
        pool = []
        for row in rows:
            reason = str(row.get("suppression_reason") or "")
            if reason in blocked_reasons:
                continue
            candidate = dict(row)
            candidate["row_type"] = "candidate_pool"
            pool.append(candidate)
        pool.sort(
            key=lambda r: (
                {"action_ready": 3, "selective": 2, "watchlist": 1}.get(str(r.get("actionability_tier") or "watchlist"), 1),
                float(r.get("prob_2_rank", r.get("prob_2") or 0.0) or 0.0),
                float(r.get("opportunity_score", 0.0) or 0.0),
                float(r.get("prob_2", 0.0) or 0.0),
                -float(r.get("risk", 0.0) or 0.0),
                str(r.get("symbol") or ""),
            ),
            reverse=True,
        )
        return pool

    def _effective_max_for_regime(self, regime_state: str) -> int:
        effective_max = int(self.config.stage2_max_names)
        if regime_state == "amber":
            effective_max = min(effective_max, max(6, int(self.config.stage2_max_names * 0.65)))
        elif regime_state == "red":
            effective_max = min(effective_max, max(2, int(self.config.stage2_max_names * 0.20)))
        return max(0, effective_max)

    def _evaluate_engine(self, *, as_of: str, visible_rows: list[dict], pool_rows: list[dict]) -> dict:
        visible_symbols = {str(r.get("symbol") or "") for r in visible_rows}
        visible = []
        hidden = []
        for row in pool_rows:
            item = dict(row)
            item["as_of_utc"] = as_of
            if str(item.get("symbol") or "") in visible_symbols:
                item["row_type"] = "visible"
                visible.append(item)
            else:
                item["row_type"] = "hidden"
                hidden.append(item)
        visible_summary = self.review_packs._bucket_summary(visible) if visible else {}
        hidden_summary = self.review_packs._bucket_summary(hidden) if hidden else {}
        return {
            "visible_rows": visible,
            "hidden_rows": hidden,
            "visible_quality_hit_rate": visible_summary.get("quality_hit_rate"),
            "hidden_quality_hit_rate": hidden_summary.get("quality_hit_rate"),
            "visible_hidden_gap": None
            if _f(visible_summary.get("quality_hit_rate")) is None or _f(hidden_summary.get("quality_hit_rate")) is None
            else round((_f(visible_summary.get("quality_hit_rate"), 0.0) or 0.0) - (_f(hidden_summary.get("quality_hit_rate"), 0.0) or 0.0), 6),
        }

    def _scan_shortlist_utility(self, rows: list[dict]) -> dict:
        empty = {
            "scan_shortlist_scans": 0,
            "scan_shortlist_scans_with_visible": 0,
            "scan_shortlist_avg_visible_rows_per_scan": None,
            "scan_shortlist_visible_quality_rate_mean": None,
            "scan_shortlist_hidden_quality_rate_mean": None,
            "scan_shortlist_mean_gap": None,
            "scan_shortlist_pairwise_win_rate": None,
            "scan_shortlist_pairwise_comparable_scans": 0,
            "scan_shortlist_top1_visible_quality": None,
            "scan_shortlist_top3_visible_quality": None,
            "scan_shortlist_overwide_penalty": None,
            "scan_shortlist_utility_score": None,
        }
        if not rows:
            return empty
        frame = pd.DataFrame([
            {
                "scan_id": r.get("as_of_utc"),
                "row_type": str(r.get("row_type") or ""),
                "score": _f(r.get("utility_decision_score", r.get("prob_2_rank") or r.get("prob_2") or 0.0), 0.0) or 0.0,
                "y": int(r.get("quality_touched") or 0),
            }
            for r in rows
        ]).dropna(subset=["scan_id"])
        if frame.empty:
            return empty
        frame = frame.sort_values(["scan_id", "score"], ascending=[True, False]).reset_index(drop=True)
        base_event_rate = float(frame["y"].mean()) if len(frame) else 0.0
        scan_count = int(frame["scan_id"].nunique())
        scans_with_visible = 0
        pairwise_wins = 0.0
        pairwise_comparable = 0
        visible_counts, visible_rates, hidden_rates, gaps, top1_visible, top3_visible = [], [], [], [], [], []
        for _, scan in frame.groupby("scan_id", sort=False):
            visible = scan[scan["row_type"] == "visible"]
            hidden = scan[scan["row_type"] != "visible"]
            visible_counts.append(int(len(visible)))
            if not visible.empty:
                scans_with_visible += 1
                visible_rate = float(visible["y"].mean())
                visible_rates.append(visible_rate)
                top1_visible.append(float(visible.iloc[:1]["y"].mean()))
                top3_visible.append(float(visible.iloc[: min(3, len(visible))]["y"].mean()))
                if not hidden.empty:
                    hidden_rate = float(hidden["y"].mean())
                    hidden_rates.append(hidden_rate)
                    gap = visible_rate - hidden_rate
                    gaps.append(gap)
                    pairwise_comparable += 1
                    if gap > 0:
                        pairwise_wins += 1.0
                    elif gap == 0:
                        pairwise_wins += 0.5
        avg_visible_rows = float(pd.Series(visible_counts).mean()) if visible_counts else None
        visible_quality_mean = float(pd.Series(visible_rates).mean()) if visible_rates else None
        hidden_quality_mean = float(pd.Series(hidden_rates).mean()) if hidden_rates else None
        mean_gap = float(pd.Series(gaps).mean()) if gaps else None
        pairwise_win_rate = float(pairwise_wins) / float(pairwise_comparable) if pairwise_comparable else None
        top1_mean = float(pd.Series(top1_visible).mean()) if top1_visible else None
        top3_mean = float(pd.Series(top3_visible).mean()) if top3_visible else None
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
            "scan_shortlist_scans": scan_count,
            "scan_shortlist_scans_with_visible": scans_with_visible,
            "scan_shortlist_avg_visible_rows_per_scan": round(avg_visible_rows, 6) if avg_visible_rows is not None else None,
            "scan_shortlist_visible_quality_rate_mean": round(visible_quality_mean, 6) if visible_quality_mean is not None else None,
            "scan_shortlist_hidden_quality_rate_mean": round(hidden_quality_mean, 6) if hidden_quality_mean is not None else None,
            "scan_shortlist_mean_gap": round(mean_gap, 6) if mean_gap is not None else None,
            "scan_shortlist_pairwise_win_rate": round(pairwise_win_rate, 6) if pairwise_win_rate is not None else None,
            "scan_shortlist_pairwise_comparable_scans": int(pairwise_comparable),
            "scan_shortlist_top1_visible_quality": round(top1_mean, 6) if top1_mean is not None else None,
            "scan_shortlist_top3_visible_quality": round(top3_mean, 6) if top3_mean is not None else None,
            "scan_shortlist_overwide_penalty": round(overwide_penalty, 6) if overwide_penalty is not None else None,
            "scan_shortlist_utility_score": round(utility_score, 6) if utility_score is not None else None,
        }

    def _engine_summary(self, label: str, rows: list[dict]) -> dict:
        utility = self._scan_shortlist_utility(rows)
        visible = [r for r in rows if str(r.get("row_type") or "") == "visible"]
        hidden = [r for r in rows if str(r.get("row_type") or "") != "visible"]
        visible_summary = self.review_packs._bucket_summary(visible) if visible else {}
        hidden_summary = self.review_packs._bucket_summary(hidden) if hidden else {}
        return {
            "engine_label": label,
            "visible_row_count": len(visible),
            "hidden_row_count": len(hidden),
            "visible_quality_hit_rate": visible_summary.get("quality_hit_rate"),
            "hidden_quality_hit_rate": hidden_summary.get("quality_hit_rate"),
            "visible_avg_end_ret": visible_summary.get("avg_end_ret"),
            "hidden_avg_end_ret": hidden_summary.get("avg_end_ret"),
            "visible_avg_mae": visible_summary.get("avg_mae"),
            "hidden_avg_mae": hidden_summary.get("avg_mae"),
            **utility,
        }

    def _build_summary(
        self,
        *,
        replay_summary: dict,
        legacy_summary: dict,
        ranked_policies: list[dict],
        family_results: list[dict],
        winner: dict,
        supported: list[dict],
        supported_families: list[dict],
        hours: int,
        step_minutes: int,
        max_scans: int,
        max_symbols: int,
    ) -> dict:
        headline = "No challenger family beat legacy offline"
        verdict = "no_supported_policy_found"
        recommended_action = "expand_or_replace_the_challenger_family_search_before_any_live_shadow_testing"
        if supported:
            headline = "At least one challenger policy beat legacy offline"
            verdict = "supported_policy_found"
            recommended_action = "promote_the_supported_offline_policy_into_a_controlled_shadow_candidate"
        elif supported_families:
            headline = "At least one challenger family produced an offline winner"
            verdict = "supported_policy_found"
            recommended_action = "promote_the_supported_family_winner_into_a_controlled_shadow_candidate"
        summary = "This offline lab searches multiple challenger families and variants on the same replay frame so unsupported families never proceed to live shadow."
        decision_memo_markdown = (
            "# Offline challenger family search lab\n\n"
            f"- **Headline:** {headline}\n"
            f"- **Verdict:** {verdict}\n"
            f"- **Recommended action:** {recommended_action}\n\n"
            "## Why this exists\n"
            "- Unsupported offline challengers should not proceed to live or shadow testing.\n"
            "- This lab evaluates materially different challenger families and variants on the same replay frame.\n"
            "- The goal is to find a genuine offline winner before any future live/shadow testing.\n"
        )
        return {
            "available": True,
            "generated_at_utc": _utc_now_iso(),
            "app_version": APP_VERSION,
            "headline": headline,
            "summary": summary,
            "verdict": verdict,
            "recommended_action": recommended_action,
            "lab_inputs": {
                "hours": int(hours),
                "step_minutes": int(step_minutes),
                "max_scans": int(max_scans),
                "max_symbols": int(max_symbols),
                "scan_count": int((replay_summary.get("window") or {}).get("scan_count") or 0),
                "policy_count": len(ranked_policies),
                "family_count": len(family_results),
            },
            "legacy_engine": legacy_summary,
            "winner": winner,
            "supported_candidates": supported,
            "supported_families": supported_families,
            "family_results": family_results,
            "ranked_policies": ranked_policies,
            "decision_memo_markdown": decision_memo_markdown,
        }

    def _build_pack(self, summary: dict, replay_summary: dict, legacy_rows: list[dict], policy_rows: dict[str, list[dict]]) -> None:
        with zipfile.ZipFile(self.pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("utility_policy_search_summary.json", json.dumps(summary, indent=2, default=str))
            zf.writestr("utility_policy_search_decision_memo.md", str(summary.get("decision_memo_markdown") or ""))
            zf.writestr(
                "replay_summary_snapshot.json",
                json.dumps({k: v for k, v in replay_summary.items() if k not in {"replay_rows", "counterfactual_rows", "scan_summaries"}}, indent=2, default=str),
            )
            zf.writestr("legacy_engine_rows.json", json.dumps(legacy_rows, indent=2, default=str))
            for policy_id, rows in policy_rows.items():
                zf.writestr(f"{policy_id}_rows.json", json.dumps(rows, indent=2, default=str))
