from __future__ import annotations

import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .regime import pending_market_regime


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppState:
    def _merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    def __init__(self, config: AppConfig):
        self.config = config
        ensure_dir(config.model_dir)
        self._lock = threading.RLock()
        self.snapshot_path = Path(config.model_dir) / "latest_state.json"
        self.coverage_path = Path(config.model_dir) / "latest_coverage.json"
        persisted = read_json(self.snapshot_path, {})
        self.status: Dict[str, Any] = self._merge_dict(self._default_status(), persisted.get("status") or {})
        self.scores: List[dict] = persisted.get("scores") or []
        self.informational_scores: List[dict] = persisted.get("informational_scores") or []
        self.coverage: Dict[str, Any] = read_json(self.coverage_path, self._default_coverage())
        self.training: Dict[str, Any] = self._merge_dict(self._default_training(), persisted.get("training") or {})
        self.model_metadata: Dict[str, Any] = persisted.get("model_metadata") or {"pt2": {"trained": False, "path": config.model_path_pt2}}
        self.scan_lock = threading.Lock()
        self._recover_interrupted_runtime_state()



    def _recover_interrupted_runtime_state(self) -> None:
        """Reset stale in-memory runtime flags after an unclean restart.

        Background scan/training threads do not survive process restarts. If the
        last persisted snapshot says they were still running, mark them as
        interrupted so the UI does not appear stuck forever.
        """
        now = utc_now()
        if self.training.get("running"):
            self.training["running"] = False
            self.training["phase"] = "interrupted"
            self.training["message"] = "training interrupted by restart; previous run did not complete"
            self.training["finished_at_utc"] = now
            self.training["heartbeat_utc"] = now
            last_error = self.training.get("last_error")
            if not last_error:
                self.training["last_error"] = "interrupted_by_restart"
        if (self.status.get("scan") or {}).get("running"):
            self.status["scan"]["running"] = False
            self.status["scan"]["phase"] = "interrupted"
            self.status["scan"]["message"] = "scan interrupted by restart; previous run did not complete"
            self.status["scan"]["finished_at_utc"] = now
            self.status["scan"]["heartbeat_utc"] = now
            self.status["updated_at_utc"] = now

    def _default_training(self) -> Dict[str, Any]:
        return {
            "running": False,
            "phase": "idle",
            "message": "ready",
            "started_at_utc": None,
            "finished_at_utc": None,
            "heartbeat_utc": None,
            "symbols_total": 0,
            "symbols_done": 0,
            "skipped_symbols_count": 0,
            "rows_accumulated": 0,
            "last_result": None,
            "last_error": None,
        }

    def _default_coverage(self) -> Dict[str, Any]:
        return {
            "universe_count": 0,
            "cohort_mode": "unresolved",
            "trained_cohort_requested_count": 0,
            "trained_cohort_available_count": 0,
            "trained_cohort_missing_count": 0,
            "symbols_requested_count": 0,
            "symbols_returned_with_bars_count": 0,
            "symbols_with_sufficient_bars_count": 0,
            "symbols_scored_count": 0,
            "symbols_previewed_count": 0,
            "symbols_deep_confirmed_count": 0,
            "symbols_stage1_preview_count": 0,
            "symbols_stage2_partial_count": 0,
            "symbols_stage2_final_count": 0,
            "stage1_feature_ready_count": 0,
            "stage2_fetch_requested_count": 0,
            "stage2_fetch_returned_count": 0,
            "stage2_feature_ready_count": 0,
            "dropped_stage1_insufficient_history": 0,
            "dropped_stage1_fetch_failed": 0,
            "dropped_stage1_insufficient_observed": 0,
            "dropped_stage1_blocked": 0,
            "dropped_stage1_by_rank": 0,
            "dropped_stage2_insufficient_history": 0,
            "dropped_stage2_fetch_failed": 0,
            "dropped_stage2_insufficient_observed": 0,
            "dropped_stage2_blocked": 0,
            "dropped_stage2_regime_suppressed": 0,
            "dropped_stage2_threshold_suppressed": 0,
            "dropped_stage2_cooldown_suppressed": 0,
            "dropped_stage2_display_trimmed": 0,
            "dropped_stage2_output_cap": 0,
            "top_skip_reasons": [],
        }

    def _default_guardrails(self) -> Dict[str, Any]:
        return {
            "blocked": 0,
            "blocked_stage1": 0,
            "blocked_stage2": 0,
            "event_risk": 0,
            "probability_capped": 0,
            "capped": 0,
            "suppressed_regime": 0,
            "suppressed_threshold": 0,
            "suppressed_cooldown": 0,
        }

    def _default_stage_counts(self) -> Dict[str, Any]:
        return {
            "stage1_candidates": 0,
            "visible_rows": 0,
            "informational_rows": 0,
            "informational_regime_rows": 0,
            "informational_cooldown_rows": 0,
            "informational_threshold_rows": 0,
            "informational_display_trim_rows": 0,
            "informational_overflow_rows": 0,
            "preview_rows": 0,
            "deep_confirmed_rows": 0,
            "stage1_preview_rows": 0,
            "stage2_partial_rows": 0,
            "stage2_final_rows": 0,
            "stage2_scored": 0,
            "action_ready_rows": 0,
            "selective_rows": 0,
            "watchlist_rows": 0,
        }

    def _default_tail_counts(self) -> Dict[str, Any]:
        return {"above_0_60": 0, "above_0_70": 0, "above_0_75": 0, "above_0_80": 0}

    def _default_actionability_summary(self) -> Dict[str, Any]:
        return {
            "action_ready_rows": 0,
            "selective_rows": 0,
            "watchlist_rows": 0,
            "actionability_type": "advisory_heuristic",
            "tail_validation_state": None,
            "temporal_tail_state": None,
            "temporal_tail_semantics": None,
            "temporal_support_basis": None,
            "market_regime_actionability": None,
        }

    def _default_suppression_summary(self) -> Dict[str, Any]:
        return {
            "threshold_suppressed_rows": 0,
            "regime_suppressed_rows": 0,
            "cooldown_suppressed_rows": 0,
            "display_trimmed_rows": 0,
            "visible_rows": 0,
            "informational_rows": 0,
            "informational_regime_rows": 0,
            "informational_cooldown_rows": 0,
            "informational_threshold_rows": 0,
            "informational_display_trim_rows": 0,
            "informational_overflow_rows": 0,
            "action_ready_rows": 0,
            "selective_rows": 0,
            "watchlist_rows": 0,
        }

    def _default_last_completed_scan_result(self) -> Dict[str, Any]:
        return {
            "available": False,
            "scan_finished_at_utc": None,
            "scan_result_generated_at_utc": None,
            "scan_result_scope": None,
            "decision_summary": {},
            "score_diagnostics": {},
            "candidate_quality": {},
            "stage1_omission_audit": {},
            "stage1_selection_repair_review": {},
            "threshold_experiment_review": {},
            "suppression_summary": self._default_suppression_summary(),
            "actionability_summary": self._default_actionability_summary(),
            "stage_counts": self._default_stage_counts(),
            "coverage": self._default_coverage(),
        }

    def _capture_last_completed_scan_result(self) -> Dict[str, Any]:
        status = self.status or {}
        scan = status.get("scan") or {}
        if (not scan.get("finished_at_utc")) and not status.get("scan_result_generated_at_utc"):
            existing = status.get("last_completed_scan_result") or {}
            return deepcopy(existing) if existing else self._default_last_completed_scan_result()
        return {
            "available": True,
            "scan_finished_at_utc": scan.get("finished_at_utc"),
            "scan_result_generated_at_utc": status.get("scan_result_generated_at_utc"),
            "scan_result_scope": status.get("scan_result_scope"),
            "decision_summary": deepcopy(status.get("decision_summary") or {}),
            "score_diagnostics": deepcopy(status.get("score_diagnostics") or {}),
            "candidate_quality": deepcopy(status.get("candidate_quality") or {}),
            "stage1_omission_audit": deepcopy(status.get("stage1_omission_audit") or {}),
            "stage1_selection_repair_review": deepcopy(status.get("stage1_selection_repair_review") or {}),
            "threshold_experiment_review": deepcopy(status.get("threshold_experiment_review") or {}),
            "suppression_summary": deepcopy(status.get("suppression_summary") or self._default_suppression_summary()),
            "actionability_summary": deepcopy(status.get("actionability_summary") or self._default_actionability_summary()),
            "stage_counts": deepcopy(status.get("stage_counts") or self._default_stage_counts()),
            "coverage": deepcopy(status.get("coverage") or self._default_coverage()),
        }

    def _default_status(self) -> Dict[str, Any]:
        return {
            "data_source": {
                "ok": False,
                "message": "not checked",
                "base_url": self.config.coinbase_exchange_base_url,
                "last_request_utc": None,
                "last_bar_timestamp": None,
                "pagination_warnings": [],
                "rate_limit_warn": None,
            },
            "universe": {
                "policy": self.config.universe_policy,
                "summary": "not scanned yet",
                "eligible_count": 0,
                "viable_count": 0,
                "selected_for_fetch_count": 0,
                "viability_signals": {},
            },
            "coverage": self._default_coverage(),
            "profiles": {"training_profile_available": False, "activity_profile_available": False},
            "guardrails": self._default_guardrails(),
            "stage_counts": self._default_stage_counts(),
            "tail_counts": self._default_tail_counts(),
            "model": {"pt2": {"trained": False, "path": self.config.model_path_pt2}},
            "scan": {
                "running": False,
                "phase": "idle",
                "message": "ready",
                "started_at_utc": None,
                "finished_at_utc": None,
                "heartbeat_utc": None,
                "symbols_total": 0,
                "symbols_done": 0,
                "failed_symbols_count": 0,
                "skipped_symbols_count": 0,
            },
            "regime_context": "unknown",
            "market_regime": pending_market_regime(reason="regime not computed yet").as_dict(),
            "score_contract": {},
            "actionability_summary": {
                "action_ready_rows": 0,
                "selective_rows": 0,
                "watchlist_rows": 0,
                "actionability_type": "advisory_heuristic",
                "tail_validation_state": None,
                "temporal_tail_state": None,
                "temporal_tail_semantics": None,
                "temporal_support_basis": None,
                "market_regime_actionability": None,
            },
            "suppression_summary": {
                "threshold_suppressed_rows": 0,
                "regime_suppressed_rows": 0,
                "cooldown_suppressed_rows": 0,
                "display_trimmed_rows": 0,
                "visible_rows": 0,
                "informational_rows": 0,
                "informational_regime_rows": 0,
                "informational_cooldown_rows": 0,
                "informational_threshold_rows": 0,
                "informational_display_trim_rows": 0,
                "informational_overflow_rows": 0,
                "action_ready_rows": 0,
                "selective_rows": 0,
                "watchlist_rows": 0,
            },
            "scan_result_scope": "final",
            "scan_result_generated_at_utc": None,
            "follow_up_scan": {
                "scheduled": False,
                "reason": None,
                "trigger": None,
                "run_after_utc": None,
                "tracked_symbols": [],
                "tracked_count": 0,
                "source_scan_finished_utc": None,
            },
            "blocked_monitoring_context": {
                "context_active": False,
                "tracked_symbols": [],
                "tracked_rows": [],
                "tracked_count": 0,
                "source_run_finished_utc": None,
                "market_regime_state": None,
                "cooldown_until_utc": None,
            },
            "cooldown_campaign": {
                "active": False,
                "cooldown_until_utc": None,
                "run_after_utc": None,
                "tracked_symbols": [],
                "tracked_rows": [],
                "tracked_count": 0,
                "merged_from_runs": 0,
                "source_runs": [],
                "latest_source_run_finished_utc": None,
                "reason": None,
            },
            "followup_comparison": {
                "available": False,
                "tracked_count": 0,
                "visible_now_count": 0,
                "still_blocked_count": 0,
                "missing_count": 0,
                "top_changes": [],
            },
            "live_universe_mode_requested": self.config.live_universe_mode,
            "live_universe_mode_effective": "unresolved",
            "decision_summary": {},
            "score_diagnostics": {},
            "candidate_quality": {},
            "stage1_omission_audit": {},
            "stage1_selection_repair_review": {},
            "threshold_experiment_review": {},
            "threshold_policy": {},
            "informational_rankings_summary": {},
            "last_completed_scan_result": self._default_last_completed_scan_result(),
            "updated_at_utc": utc_now(),
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self.status)

    def get_scores(self) -> List[dict]:
        with self._lock:
            return deepcopy(self.scores)

    def get_informational_scores(self) -> List[dict]:
        with self._lock:
            return deepcopy(self.informational_scores)

    def get_training(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self.training)

    def get_coverage(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self.coverage)

    def update_status(self, **kwargs) -> None:
        with self._lock:
            coverage_updated = False
            for key, value in kwargs.items():
                self.status[key] = value
                if key == "coverage":
                    self.coverage = deepcopy(value)
                    coverage_updated = True
            self.status["updated_at_utc"] = utc_now()
            if coverage_updated:
                atomic_write_json(self.coverage_path, self.coverage)
            self._persist()

    def set_scores(self, scores: List[dict]) -> None:
        self.set_score_views(scores, self.informational_scores)

    def set_score_views(self, scores: List[dict], informational_scores: List[dict] | None = None) -> None:
        with self._lock:
            self.scores = list(scores or [])
            self.informational_scores = list(informational_scores or [])
            now = utc_now()
            self.status["scan_result_generated_at_utc"] = now
            self.status["updated_at_utc"] = now
            self._persist()

    def set_coverage(self, coverage: Dict[str, Any]) -> None:
        with self._lock:
            self.coverage = coverage
            self.status["coverage"] = deepcopy(coverage)
            now = utc_now()
            self.status["scan_result_generated_at_utc"] = now
            self.status["updated_at_utc"] = now
            atomic_write_json(self.coverage_path, coverage)
            self._persist()

    def set_model_metadata(self, pt2_meta: dict) -> None:
        with self._lock:
            self.model_metadata = {"pt2": pt2_meta}
            self.status["model"] = {"pt2": pt2_meta}
            self.status["profiles"] = {
                "training_profile_available": bool(pt2_meta.get("feature_mean")),
                "activity_profile_available": True,
            }
            self.status["updated_at_utc"] = utc_now()
            self._persist()

    def scan_started(self, message: str, symbols_total: int = 0) -> None:
        with self._lock:
            now = utc_now()
            self.status["last_completed_scan_result"] = self._capture_last_completed_scan_result()
            empty_coverage = self._default_coverage()
            empty_coverage["symbols_requested_count"] = int(symbols_total or 0)
            self.scores = []
            self.informational_scores = []
            self.coverage = deepcopy(empty_coverage)
            self.status["coverage"] = deepcopy(empty_coverage)
            self.status["guardrails"] = self._default_guardrails()
            self.status["stage_counts"] = self._default_stage_counts()
            self.status["tail_counts"] = self._default_tail_counts()
            self.status["actionability_summary"] = self._default_actionability_summary()
            self.status["suppression_summary"] = self._default_suppression_summary()
            self.status["decision_summary"] = {}
            self.status["score_diagnostics"] = {}
            self.status["candidate_quality"] = {}
            self.status["stage1_omission_audit"] = {}
            self.status["stage1_selection_repair_review"] = {}
            self.status["threshold_experiment_review"] = {}
            self.status["threshold_policy"] = {}
            self.status["informational_rankings_summary"] = {}
            self.status["scan_result_scope"] = "partial"
            self.status["scan_result_generated_at_utc"] = now
            self.status["live_universe_mode_requested"] = self.config.live_universe_mode
            self.status["live_universe_mode_effective"] = "unresolved"
            self.status["scan"] = {
                "running": True,
                "phase": "starting",
                "message": message,
                "started_at_utc": now,
                "finished_at_utc": None,
                "heartbeat_utc": now,
                "symbols_total": symbols_total,
                "symbols_done": 0,
                "failed_symbols_count": 0,
                "skipped_symbols_count": 0,
            }
            self.status["market_regime"] = pending_market_regime(
                previous=self.status.get("market_regime") or {},
                reason="scan running; regime evaluation pending",
            ).as_dict()
            self.status["updated_at_utc"] = now
            atomic_write_json(self.coverage_path, self.coverage)
            self._persist()

    def scan_progress(self, phase: str, message: str, *, heartbeat: bool = True, inc_done: bool = False, inc_failed: bool = False, inc_skipped: bool = False, symbols_total: int | None = None) -> None:
        with self._lock:
            now = utc_now()
            scan = self.status["scan"]
            scan["phase"] = phase
            scan["message"] = message
            if heartbeat:
                scan["heartbeat_utc"] = now
            if inc_done:
                scan["symbols_done"] += 1
                total = int(scan.get("symbols_total") or 0)
                if total > 0:
                    scan["symbols_done"] = min(int(scan["symbols_done"]), total)
            if inc_failed:
                scan["failed_symbols_count"] += 1
            if inc_skipped:
                scan["skipped_symbols_count"] += 1
            if symbols_total is not None:
                scan["symbols_total"] = symbols_total
            self.status["updated_at_utc"] = now
            self._persist()

    def scan_finished(self, message: str, phase: str = "complete") -> None:
        with self._lock:
            now = utc_now()
            scan = self.status["scan"]
            scan["running"] = False
            scan["phase"] = phase
            scan["message"] = message
            scan["finished_at_utc"] = now
            scan["heartbeat_utc"] = now
            self.status["scan_result_scope"] = "final" if phase == "complete" else self.status.get("scan_result_scope", "partial")
            self.status["last_completed_scan_result"] = self._capture_last_completed_scan_result()
            self.status["updated_at_utc"] = now
            self._persist()

    def training_started(self, message: str = "training started", symbols_total: int = 0) -> None:
        with self._lock:
            self.training = {
                "running": True,
                "phase": "starting",
                "message": message,
                "started_at_utc": utc_now(),
                "finished_at_utc": None,
                "heartbeat_utc": utc_now(),
                "symbols_total": symbols_total,
                "symbols_done": 0,
                "skipped_symbols_count": 0,
                "rows_accumulated": 0,
                "last_result": None,
                "last_error": None,
            }
            self._persist()

    def training_progress(self, phase: str, message: str, *, symbols_total: int | None = None, inc_done: bool = False, inc_skipped: bool = False, rows_accumulated: int | None = None, heartbeat: bool = True) -> None:
        with self._lock:
            self.training["phase"] = phase
            self.training["message"] = message
            if heartbeat:
                self.training["heartbeat_utc"] = utc_now()
            if symbols_total is not None:
                self.training["symbols_total"] = symbols_total
            if inc_done:
                self.training["symbols_done"] += 1
            if inc_skipped:
                self.training["skipped_symbols_count"] += 1
            if rows_accumulated is not None:
                self.training["rows_accumulated"] = int(rows_accumulated)
            self._persist()

    def training_finished(self, result: dict | None = None, error: str | None = None) -> None:
        with self._lock:
            self.training["running"] = False
            self.training["phase"] = "failed" if error else "complete"
            self.training["message"] = error or "training complete"
            self.training["finished_at_utc"] = utc_now()
            self.training["heartbeat_utc"] = utc_now()
            self.training["last_result"] = result
            self.training["last_error"] = error
            self._persist()

    def _persist(self) -> None:
        atomic_write_json(
            self.snapshot_path,
            {
                "status": self.status,
                "scores": self.scores,
                "informational_scores": self.informational_scores,
                "training": self.training,
                "model_metadata": self.model_metadata,
            },
        )
