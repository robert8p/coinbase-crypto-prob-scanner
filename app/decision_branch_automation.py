from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig
from .persist import atomic_write_json, read_json
from .version import APP_VERSION
from .runtime_scope import current_runtime_scope, scope_key


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_explicit_scope(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("state_scope_key"):
        return True
    return bool(data.get("app_version") or data.get("deployed_since_utc"))


def load_runtime_live_overrides(model_dir: str | Path) -> dict:
    path = Path(model_dir) / "runtime_live_overrides.json"
    data = read_json(path, {})
    return data if isinstance(data, dict) else {}


def _canonical_checkpoint_outcome(checkpoint: dict) -> tuple[str | None, bool]:
    decision_outcome = checkpoint.get("decision_checkpoint_outcome")
    current_outcome = checkpoint.get("current_outcome")
    truth_conflict = bool(
        checkpoint.get("triggered")
        and decision_outcome in {"confirmed", "falsified", "inconclusive"}
        and current_outcome in {"confirmed", "falsified", "inconclusive"}
        and decision_outcome != current_outcome
    )
    if checkpoint.get("triggered") and current_outcome in {"confirmed", "falsified", "inconclusive"}:
        return current_outcome, truth_conflict
    return decision_outcome or current_outcome, truth_conflict


def effective_live_raw_threshold(config: AppConfig) -> float:
    default_threshold = max(0.0, min(1.0, float(getattr(config, "live_raw_threshold", 0.30) or 0.30)))
    overrides = load_runtime_live_overrides(getattr(config, "model_dir", "/var/data/model"))
    override = overrides.get("live_raw_threshold_override")
    if override is None:
        return default_threshold
    current_scope = current_runtime_scope(getattr(config, "model_dir", "/var/data/model"), app_version=APP_VERSION)
    current_scope_key = current_scope.get("state_scope_key")
    override_scope_key = overrides.get("state_scope_key") or scope_key(
        overrides.get("app_version"),
        overrides.get("deployed_since_utc"),
    )
    if overrides.get("source") in {"decision_branch_automation", "live_candidate_proof", "live_candidate_adoption"}:
        if not _has_explicit_scope(overrides):
            return default_threshold
        if current_scope_key and override_scope_key != current_scope_key:
            return default_threshold
    try:
        return max(0.0, min(1.0, float(override)))
    except Exception:
        return default_threshold


class DecisionBranchAutomationService:
    def __init__(self, config: AppConfig, decision_checkpoint, fresh_retrain_audit_service=None):
        self.config = config
        self.decision_checkpoint = decision_checkpoint
        self.fresh_retrain_audit_service = fresh_retrain_audit_service
        self.summary_path = Path(config.model_dir) / "decision_branch_automation_summary.json"
        self.overrides_path = Path(config.model_dir) / "runtime_live_overrides.json"
        self.operator_state_path = Path(config.model_dir) / "decision_branch_operator_state.json"

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_overrides(self) -> dict:
        return load_runtime_live_overrides(self.config.model_dir)

    def latest_operator_state(self) -> dict:
        data = read_json(self.operator_state_path, {})
        return data if isinstance(data, dict) else {}

    def _write_operator_state(self, payload: dict) -> dict:
        atomic_write_json(self.operator_state_path, payload)
        return payload

    def _write_overrides(self, payload: dict) -> dict:
        atomic_write_json(self.overrides_path, payload)
        return payload

    def _current_scope(self) -> dict:
        return current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)

    def _checkpoint_scope_key(self, checkpoint: dict) -> str:
        return checkpoint.get("state_scope_key") or self._current_scope().get("state_scope_key") or scope_key(
            checkpoint.get("app_version"),
            checkpoint.get("deployed_since_utc"),
        )

    def _is_override_in_scope(self, overrides: dict, current_scope_key: str) -> bool:
        override_scope_key = overrides.get("state_scope_key") or scope_key(
            overrides.get("app_version"),
            overrides.get("deployed_since_utc"),
        )
        if overrides.get("source") != "decision_branch_automation":
            return True
        if not _has_explicit_scope(overrides):
            return False
        if current_scope_key and not override_scope_key:
            return False
        if not current_scope_key:
            return True
        return current_scope_key == override_scope_key

    def _apply_threshold_experiment(self, checkpoint: dict) -> dict:
        current_scope = self._current_scope()
        payload = {
            "live_raw_threshold_override": 0.28,
            "source": "decision_branch_automation",
            "reason": "checkpoint_confirmed_threshold_experiment",
            "applied_at_utc": _utc_now_iso(),
            "note": "Applied automatically from the decision branch automation after checkpoint confirmation.",
            "app_version": current_scope.get("app_version") or APP_VERSION,
            "deployed_since_utc": current_scope.get("deployed_since_utc"),
            "state_scope_key": current_scope.get("state_scope_key") or self._checkpoint_scope_key(checkpoint),
            "checkpoint_triggered_at_utc": checkpoint.get("decision_checkpoint_triggered_at_utc"),
            "checkpoint_outcome": checkpoint.get("decision_checkpoint_outcome") or checkpoint.get("current_outcome"),
        }
        return self._write_overrides(payload)

    def clear_active_override(self) -> dict:
        current_scope = self._current_scope()
        payload = {
            "live_raw_threshold_override": None,
            "source": "decision_branch_automation",
            "reason": "override_cleared",
            "applied_at_utc": _utc_now_iso(),
            "note": "Threshold override cleared by operator action.",
            "app_version": current_scope.get("app_version") or APP_VERSION,
            "deployed_since_utc": current_scope.get("deployed_since_utc"),
            "state_scope_key": current_scope.get("state_scope_key"),
            "cleared_by_operator": True,
        }
        self._write_overrides(payload)
        current = self.latest_summary()
        ts = _utc_now_iso()
        current["auto_execute_supported_actions_enabled"] = False
        current["last_execution_action"] = "clear_threshold_override"
        current["last_execution_result"] = "override_cleared_auto_execute_disabled"
        current["last_execution_at_utc"] = ts
        atomic_write_json(self.summary_path, current)
        state = self.latest_operator_state()
        state.update({
            "last_execution_action": "clear_threshold_override",
            "last_execution_result": "override_cleared_auto_execute_disabled",
            "last_execution_at_utc": ts,
            "last_execution_scope_key": current.get("state_scope_key"),
        })
        self._write_operator_state(state)
        return payload

    def set_auto_execute_enabled(self, enabled: bool) -> dict:
        state = self.latest_operator_state()
        state["auto_execute_supported_actions_enabled"] = bool(enabled)
        state["auto_execute_updated_at_utc"] = _utc_now_iso()
        self._write_operator_state(state)
        current = self.latest_summary()
        current["auto_execute_supported_actions_enabled"] = bool(enabled)
        current["updated_at_utc"] = _utc_now_iso()
        atomic_write_json(self.summary_path, current)
        return self.build_summary(checkpoint_summary=None)

    def acknowledge(self) -> dict:
        current = self.latest_summary()
        if not current:
            current = self.build_summary()
        ts = _utc_now_iso()
        checkpoint_triggered_at_utc = current.get("checkpoint_triggered_at_utc")
        checkpoint_outcome = current.get("checkpoint_outcome")
        state_scope_key = current.get("state_scope_key")
        state = self.latest_operator_state()
        state.update({
            "decision_branch_acknowledged": True,
            "acknowledged_for_checkpoint_triggered_at_utc": checkpoint_triggered_at_utc,
            "acknowledged_for_outcome": checkpoint_outcome,
            "acknowledged_for_scope_key": state_scope_key,
            "acknowledged_at_utc": ts,
        })
        self._write_operator_state(state)
        current["decision_branch_acknowledged"] = True
        current["branch_notification_pending"] = False
        current["updated_at_utc"] = ts
        atomic_write_json(self.summary_path, current)
        return current

    def execute_now(self) -> dict:
        checkpoint = self.decision_checkpoint.latest_summary() or self.decision_checkpoint.build_summary()
        outcome, _ = _canonical_checkpoint_outcome(checkpoint)
        if not checkpoint.get("triggered"):
            return self.build_summary(checkpoint_summary=checkpoint)
        if outcome == "confirmed":
            ts = _utc_now_iso()
            state = self.latest_operator_state()
            state.update({
                "last_execution_action": "keep_live_path_unchanged",
                "last_execution_result": "no_change_required",
                "last_execution_at_utc": ts,
                "last_execution_checkpoint_triggered_at_utc": checkpoint.get("decision_checkpoint_triggered_at_utc"),
                "last_execution_outcome": outcome,
                "last_execution_scope_key": self._checkpoint_scope_key(checkpoint),
            })
            self._write_operator_state(state)
            current = self.latest_summary()
            current["last_execution_action"] = "keep_live_path_unchanged"
            current["last_execution_result"] = "no_change_required"
            current["last_execution_at_utc"] = ts
            atomic_write_json(self.summary_path, current)
            return self.build_summary(checkpoint_summary=checkpoint)
        ts = _utc_now_iso()
        state = self.latest_operator_state()
        if outcome == "falsified" and self.fresh_retrain_audit_service is not None:
            audit = self.fresh_retrain_audit_service.start_run()
            running = bool((audit or {}).get("running"))
            result_code = "fresh_retrain_audit_started" if running else (audit or {}).get("status") or "fresh_retrain_audit_queued"
            state.update({
                "last_execution_action": "fresh_retrain_and_symbol_concentration_audit",
                "last_execution_result": result_code,
                "last_execution_at_utc": ts,
                "last_execution_checkpoint_triggered_at_utc": checkpoint.get("decision_checkpoint_triggered_at_utc"),
                "last_execution_outcome": outcome,
                "last_execution_scope_key": self._checkpoint_scope_key(checkpoint),
            })
            self._write_operator_state(state)
            current = self.latest_summary()
            current["last_execution_action"] = "fresh_retrain_and_symbol_concentration_audit"
            current["last_execution_result"] = result_code
            current["last_execution_at_utc"] = ts
            atomic_write_json(self.summary_path, current)
            return self.build_summary(checkpoint_summary=checkpoint)
        state.update({
            "last_execution_action": f"manual_branch_{outcome}",
            "last_execution_result": "manual_required_branch_not_auto_executable",
            "last_execution_at_utc": ts,
            "last_execution_checkpoint_triggered_at_utc": checkpoint.get("decision_checkpoint_triggered_at_utc"),
            "last_execution_outcome": outcome,
            "last_execution_scope_key": self._checkpoint_scope_key(checkpoint),
        })
        self._write_operator_state(state)
        current = self.latest_summary()
        current["last_execution_action"] = f"manual_branch_{outcome}"
        current["last_execution_result"] = "manual_required_branch_not_auto_executable"
        current["last_execution_at_utc"] = ts
        atomic_write_json(self.summary_path, current)
        return self.build_summary(checkpoint_summary=checkpoint)

    def build_summary(self, *, checkpoint_summary: dict | None = None) -> dict:
        checkpoint = dict(checkpoint_summary or self.decision_checkpoint.latest_summary() or self.decision_checkpoint.build_summary())
        previous = self.latest_summary()
        operator_state = self.latest_operator_state()
        current_scope = self._current_scope()
        current_scope_key = current_scope.get("state_scope_key") or self._checkpoint_scope_key(checkpoint)
        previous_scope_key = previous.get("state_scope_key")
        previous_same_scope = previous_scope_key == current_scope_key
        auto_execute_enabled = bool(
            operator_state.get(
                "auto_execute_supported_actions_enabled",
                previous.get("auto_execute_supported_actions_enabled", True) if previous_same_scope else True,
            )
        )

        raw_overrides = self.latest_overrides()
        override_in_scope = self._is_override_in_scope(raw_overrides, current_scope_key)
        active_threshold_override = raw_overrides.get("live_raw_threshold_override") if override_in_scope else None
        threshold_experiment_active = active_threshold_override is not None and float(active_threshold_override) == 0.28
        stale_runtime_override_detected = (
            raw_overrides.get("live_raw_threshold_override") is not None
            and raw_overrides.get("source") == "decision_branch_automation"
            and not override_in_scope
        )

        checkpoint_outcome, truth_conflict_detected = _canonical_checkpoint_outcome(checkpoint)
        triggered = bool(checkpoint.get("triggered"))
        checkpoint_triggered_at_utc = checkpoint.get("decision_checkpoint_triggered_at_utc")
        acknowledged = bool(
            triggered
            and operator_state.get("decision_branch_acknowledged", False)
            and operator_state.get("acknowledged_for_checkpoint_triggered_at_utc") == checkpoint_triggered_at_utc
            and operator_state.get("acknowledged_for_outcome") == checkpoint_outcome
            and operator_state.get("acknowledged_for_scope_key") == current_scope_key
        )

        operator_state_scope_matches = operator_state.get("last_execution_scope_key") == current_scope_key
        last_execution_action = (operator_state.get("last_execution_action") if operator_state_scope_matches else None) or (
            previous.get("last_execution_action") if previous_same_scope else None
        )
        last_execution_result = (operator_state.get("last_execution_result") if operator_state_scope_matches else None) or (
            previous.get("last_execution_result") if previous_same_scope else None
        )
        last_execution_at_utc = (operator_state.get("last_execution_at_utc") if operator_state_scope_matches else None) or (
            previous.get("last_execution_at_utc") if previous_same_scope else None
        )

        branch_status = "waiting_for_checkpoint"
        branch_headline = "Decision branch waiting for checkpoint"
        branch_summary = "The app will not branch until the decision checkpoint is triggered."
        next_action_id = None
        next_action_label = None
        auto_executable = False
        manual_required = False

        if stale_runtime_override_detected and not triggered:
            branch_status = "stale_override_ignored"
            branch_headline = "Stale threshold override ignored for current deployment"
            branch_summary = "A prior decision-branch override exists on disk, but it does not belong to the current deployment window so it is ignored for current experiment semantics."

        if triggered and truth_conflict_detected:
            branch_status = "truth_conflict_resolved_to_current_evidence"
            branch_headline = "Checkpoint truth conflict resolved to current evidence"
            branch_summary = "A stale checkpoint outcome on disk conflicted with the current deployment-window evidence. The branch used the current evidence outcome as the canonical verdict for this scope."

        if triggered:
            if checkpoint_outcome == "confirmed":
                next_action_id = "keep_live_path_unchanged"
                next_action_label = "Keep live path unchanged"
                auto_executable = False
                if threshold_experiment_active:
                    branch_status = "clear_override_recommended"
                    branch_headline = "Confirmed branch says clear lower-threshold override"
                    branch_summary = "The checkpoint confirmed the current setup. A current-scope 0.28 threshold override conflicts with that confirmed branch and should be cleared before continuing evidence accrual."
                else:
                    branch_status = "no_change_branch"
                    branch_headline = "Confirmed branch says keep live path unchanged"
                    branch_summary = checkpoint.get("action_on_confirmation") or "The checkpoint confirmed the current setup, so the live path should stay unchanged while resolved evidence continues to accrue."
            elif checkpoint_outcome == "falsified":
                next_action_id = "fresh_retrain_and_symbol_concentration_audit"
                next_action_label = "Fresh retrain + symbol concentration audit"
                auto_executable = self.fresh_retrain_audit_service is not None
                manual_required = not auto_executable
                audit_summary = self.fresh_retrain_audit_service.latest_summary() if self.fresh_retrain_audit_service is not None else {}
                audit_same_scope = bool((audit_summary or {}).get("state_scope_key") == current_scope_key)
                audit_running = audit_same_scope and bool((audit_summary or {}).get("running"))
                audit_completed = audit_same_scope and str((audit_summary or {}).get("status")) == "completed"
                audit_failed = audit_same_scope and str((audit_summary or {}).get("status")) == "failed"
                if auto_executable and audit_running:
                    branch_status = "supported_action_running"
                    branch_headline = "Fresh retrain audit is running"
                    branch_summary = "A non-promoting fresh retrain + symbol concentration audit is currently building. Live scoring remains unchanged while the shadow artifacts are prepared."
                elif auto_executable and audit_completed:
                    branch_status = "supported_action_completed"
                    branch_headline = "Fresh retrain audit artifacts are ready"
                    branch_summary = "The falsified branch has produced a non-promoting fresh retrain + symbol concentration audit pack for review. Live promotion remains blocked."
                elif auto_executable and audit_failed:
                    branch_status = "supported_action_failed"
                    branch_headline = "Fresh retrain audit failed"
                    branch_summary = (audit_summary or {}).get("summary") or "The last fresh retrain audit failed. Re-run the supported action after checking the error."
                elif auto_executable:
                    branch_status = "supported_action_pending"
                    branch_headline = "Supported shadow retrain branch ready after falsification"
                    branch_summary = "The current checkpoint failed. The next safe in-app action is to build a non-promoting fresh retrain + symbol concentration audit pack."
                else:
                    branch_status = "manual_required"
                    branch_headline = "Manual branch required after falsification"
                    branch_summary = "The current checkpoint failed. The declared next move requires a manual retrain-and-audit branch because that full path is not yet safely self-executable in-app."
            elif checkpoint_outcome == "inconclusive":
                next_action_id = "keep_live_path_unchanged"
                next_action_label = "Keep live path unchanged"
                branch_status = "no_change_branch"
                branch_headline = "No live-path change branch selected"
                branch_summary = "The checkpoint reached the inconclusive branch, so the app will keep the live path unchanged until a later review checkpoint is justified."

        if (
            threshold_experiment_active
            and not last_execution_result
            and raw_overrides.get("source") == "decision_branch_automation"
            and checkpoint_outcome == "confirmed"
            and override_in_scope
        ):
            last_execution_action = "apply_threshold_experiment_0_28"
            last_execution_result = "override_applied"
            last_execution_at_utc = raw_overrides.get("applied_at_utc")
            operator_state.update({
                "last_execution_action": last_execution_action,
                "last_execution_result": last_execution_result,
                "last_execution_at_utc": last_execution_at_utc,
                "last_execution_checkpoint_triggered_at_utc": checkpoint_triggered_at_utc,
                "last_execution_outcome": checkpoint_outcome,
                "last_execution_scope_key": current_scope_key,
            })
            self._write_operator_state(operator_state)

        notification_pending = bool(triggered and not acknowledged)
        result = {
            "app_version": current_scope.get("app_version") or checkpoint.get("app_version") or APP_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "source": "decision_checkpoint_branch",
            "deployed_since_utc": current_scope.get("deployed_since_utc"),
            "state_scope_key": current_scope_key,
            "checkpoint_triggered": triggered,
            "checkpoint_outcome": checkpoint_outcome,
            "truth_conflict_detected": truth_conflict_detected,
            "checkpoint_triggered_at_utc": checkpoint_triggered_at_utc,
            "auto_execute_supported_actions_enabled": auto_execute_enabled,
            "decision_branch_acknowledged": acknowledged,
            "branch_notification_pending": notification_pending,
            "stage1_selection_mode": checkpoint.get("stage1_selection_mode"),
            "configured_live_raw_threshold": checkpoint.get("live_raw_threshold"),
            "effective_live_raw_threshold": effective_live_raw_threshold(self.config),
            "runtime_overrides": {
                "live_raw_threshold_override": active_threshold_override,
                "override_source": raw_overrides.get("source") if override_in_scope else None,
                "override_reason": raw_overrides.get("reason") if override_in_scope else None,
                "override_applied_at_utc": raw_overrides.get("applied_at_utc") if override_in_scope else None,
                "threshold_experiment_active": threshold_experiment_active,
                "stale_override_ignored_for_current_scope": stale_runtime_override_detected,
                "raw_override_scope_key": (raw_overrides.get("state_scope_key") or scope_key(
                    raw_overrides.get("app_version"),
                    raw_overrides.get("deployed_since_utc"),
                )) if _has_explicit_scope(raw_overrides) else None,
            },
            "branch_action": {
                "status": branch_status,
                "headline": branch_headline,
                "summary": branch_summary,
                "next_action_id": next_action_id,
                "next_action_label": next_action_label,
                "auto_executable": auto_executable,
                "manual_required": manual_required,
                "can_execute_now": bool(triggered and auto_executable and not threshold_experiment_active and branch_status != "supported_action_running"),
                "can_clear_override": bool(threshold_experiment_active),
                "can_acknowledge": bool(triggered),
            },
            "buttons": {
                "toggle_auto_execute": True,
                "execute_now": bool(triggered and auto_executable and not threshold_experiment_active and branch_status != "supported_action_running"),
                "clear_override": bool(threshold_experiment_active),
                "acknowledge": bool(triggered),
            },
            "manual_follow_up_note": (
                "Run the supported fresh retrain + symbol concentration audit branch to build non-promoting shadow artifacts, then review that pack before any live model change."
                if checkpoint_outcome == "falsified" and self.fresh_retrain_audit_service is not None else (
                    "Falsification currently requires a manual retrain + symbol concentration audit branch. This app will surface that requirement automatically, but it will not self-retrain or self-redeploy that branch yet."
                    if manual_required else None
                )
            ),
            "last_execution_action": last_execution_action,
            "last_execution_result": last_execution_result,
            "last_execution_at_utc": last_execution_at_utc,
            "future_action_automation_note": "Future branch actions should be automated end-to-end where they are safely executable in-app; unsupported branches should still surface explicit operator actions and state.",
        }
        atomic_write_json(self.summary_path, result)
        return result
