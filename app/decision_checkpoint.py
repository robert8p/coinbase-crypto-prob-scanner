from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig
from .persist import atomic_write_json, read_json
from .version import APP_VERSION
from .runtime_scope import current_runtime_scope, scope_key


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



class DecisionCheckpointService:
    def __init__(self, config: AppConfig, review_packs):
        self.config = config
        self.review_packs = review_packs
        self.summary_path = Path(config.model_dir) / "decision_checkpoint_summary.json"

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def _classify_outcome(self, visible_rows: int, visible_q, non_visible_q) -> str:
        if visible_rows < 30:
            return "waiting_for_more_resolved_visible_rows"
        if visible_q is None:
            return "inconclusive"
        visible_q = float(visible_q)
        non_visible_q = float(non_visible_q) if non_visible_q is not None else None
        # V2-consistent interpretation: confirmation must clear the legacy 15% floor
        # and beat the hidden remainder in the same resolved deployment window.
        if visible_q >= 0.15 and (non_visible_q is None or visible_q > non_visible_q):
            return "confirmed"
        if non_visible_q is not None and visible_q < non_visible_q:
            return "falsified"
        return "inconclusive"

    def build_summary(self, *, summary: dict | None = None) -> dict:
        source_summary = dict(summary or {})
        if not source_summary:
            source_summary = self.review_packs.get_current_version_summary()
        evidence = dict(source_summary.get("evidence") or {})
        visible_rows = int(evidence.get("visible_rows") or 0)
        visible_q = evidence.get("visible_quality_hit_rate")
        non_visible_q = evidence.get("non_visible_quality_hit_rate")
        target_rows = 30
        current_outcome = self._classify_outcome(visible_rows, visible_q, non_visible_q)

        runtime_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        current_app_version = runtime_scope.get("app_version") or source_summary.get("app_version") or APP_VERSION
        deployed_since_utc = runtime_scope.get("deployed_since_utc")
        current_scope_key = runtime_scope.get("state_scope_key") or scope_key(current_app_version, deployed_since_utc)

        previous = read_json(self.summary_path, {})
        previous_scope_key = previous.get("state_scope_key") or scope_key(
            previous.get("app_version"),
            previous.get("deployed_since_utc"),
        )
        previous_is_same_scope = previous_scope_key == current_scope_key

        triggered_at = previous.get("decision_checkpoint_triggered_at_utc") if previous_is_same_scope else None
        previous_outcome = previous.get("decision_checkpoint_outcome") if previous_is_same_scope else None
        outcome_changed = previous_is_same_scope and previous_outcome not in (None, current_outcome)
        acknowledged_for_outcome = previous.get("acknowledged_for_outcome") if previous_is_same_scope else None
        acknowledged = False
        if previous_is_same_scope and bool(previous.get("decision_checkpoint_acknowledged", False)):
            if acknowledged_for_outcome:
                acknowledged = acknowledged_for_outcome == current_outcome
            else:
                acknowledged = previous_outcome == current_outcome

        if visible_rows >= target_rows and not triggered_at:
            triggered_at = _utc_now_iso()

        current_stage1_mode = str(getattr(self.config, "stage1_selection_mode", "primary_only") or "primary_only")
        current_live_threshold = float(getattr(self.config, "live_raw_threshold", 0.35) or 0.35)
        result = {
            "app_version": current_app_version,
            "generated_at_utc": _utc_now_iso(),
            "source": "current_version",
            "deployed_since_utc": deployed_since_utc,
            "state_scope_key": current_scope_key,
            "hypothesis": f"{current_stage1_mode} + LIVE_RAW_THRESHOLD={current_live_threshold:.2f} produces a visible shortlist that beats non-visible on quality hit rate.",
            "confirmation_rule": "visible quality hit rate >= 15% across 30+ resolved visible rows",
            "falsification_rule": "visible quality hit rate < non-visible quality hit rate after 30+ resolved visible rows",
            "action_on_confirmation": "If confirmed, keep the current live path unchanged and continue accumulating resolved evidence before testing any lower threshold.",
            "action_on_falsification": "If falsified, revert the live Stage 1 switch and reassess Stage 1 selection before changing the Stage 2 threshold.",
            "action_on_inconclusive": "Keep the live path unchanged and continue accumulating resolved visible rows until the next review checkpoint is justified.",
            "stage1_selection_mode": current_stage1_mode,
            "live_raw_threshold": current_live_threshold,
            "decision_target_visible_rows": target_rows,
            "resolved_visible_rows": visible_rows,
            "rows_remaining_until_decision": max(0, target_rows - visible_rows),
            "current_visible_quality_hit_rate": visible_q,
            "current_non_visible_quality_hit_rate": non_visible_q,
            "decision_ready": visible_rows >= target_rows,
            "status": "decision_ready" if visible_rows >= target_rows else "waiting_for_more_resolved_visible_rows",
            "current_outcome": current_outcome,
            "triggered": bool(triggered_at),
            "decision_checkpoint_triggered_at_utc": triggered_at,
            "decision_checkpoint_outcome": current_outcome if visible_rows >= target_rows else None,
            "one_time_notification_pending": bool(triggered_at and not acknowledged),
            "decision_checkpoint_acknowledged": acknowledged,
            "decision_checkpoint_outcome_changed": outcome_changed,
            "automation_note": "Future decision points and follow-on actions should default to in-app automation rather than manual monitoring wherever practical.",
            "headline": {
                "waiting_for_more_resolved_visible_rows": "Decision checkpoint pending — collecting resolved visible rows",
                "confirmed": "Decision checkpoint confirmed — visible shortlist is clearing the 15% bar",
                "falsified": "Decision checkpoint falsified — visible shortlist is underperforming the hidden remainder",
                "inconclusive": "Decision checkpoint reached but outcome is inconclusive",
            }[current_outcome],
            "summary": {
                "waiting_for_more_resolved_visible_rows": "The app will keep tracking current-version resolved evidence until 30 visible rows are available.",
                "confirmed": "The 30-row checkpoint has been reached and the visible shortlist is clearing the confirmation bar.",
                "falsified": "The 30-row checkpoint has been reached and the visible shortlist is underperforming the hidden remainder, so the current hypothesis failed.",
                "inconclusive": "The 30-row checkpoint has been reached but the result is neither confirmed nor falsified under the declared rule.",
            }[current_outcome],
            "future_action_automation_note": "Automate future decision checkpoints and triggered follow-on actions before adding new diagnostic surfaces.",
        }
        atomic_write_json(self.summary_path, result)
        return result

    def acknowledge(self) -> dict:
        current = self.latest_summary()
        if not current:
            current = self.build_summary()
        current["decision_checkpoint_acknowledged"] = True
        current["acknowledged_for_outcome"] = current.get("decision_checkpoint_outcome") or current.get("current_outcome")
        current["one_time_notification_pending"] = False
        atomic_write_json(self.summary_path, current)
        return current
