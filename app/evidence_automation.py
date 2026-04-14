from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .persist import atomic_write_json, ensure_dir, read_json
from .version import APP_VERSION

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: str | Path, content: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_name = tmp.name
    Path(tmp_name).replace(path)
    return path


class EvidenceAutomationService:
    """Control-plane automation only.

    This service deliberately avoids changing live scoring semantics. It automates:
    - checkpoint / branch refresh and safe no-op acknowledgement
    - post-maturity bundled review artifact generation
    - automated diagnostic verdict generation
    - training submission/status orchestration artifacts (no auto-promotion)
    """

    def __init__(
        self,
        config,
        state,
        review_packs,
        decision_checkpoint,
        decision_branch_automation,
        misranking_diagnostic,
        threshold_boundary_review,
        cooldown_shortlist_review,
        stage2_retrain_review,
        trainer,
        model_output_distribution,
    ):
        self.config = config
        self.state = state
        self.review_packs = review_packs
        self.decision_checkpoint = decision_checkpoint
        self.decision_branch_automation = decision_branch_automation
        self.misranking_diagnostic = misranking_diagnostic
        self.threshold_boundary_review = threshold_boundary_review
        self.cooldown_shortlist_review = cooldown_shortlist_review
        self.stage2_retrain_review = stage2_retrain_review
        self.trainer = trainer
        self.model_output_distribution = model_output_distribution

        self.root_dir = ensure_dir(Path(config.model_dir) / "automation")
        self.review_dir = ensure_dir(self.root_dir / "review")
        self.training_dir = ensure_dir(self.root_dir / "training")
        self.status_path = self.root_dir / "automation_status.json"
        self.bundle_path = self.review_dir / "latest_post_maturity_review_bundle.zip"
        self.bundle_manifest_path = self.review_dir / "latest_post_maturity_review_bundle_manifest.json"
        self.diag_json_path = self.review_dir / "latest_diagnostic_battery.json"
        self.diag_txt_path = self.review_dir / "latest_diagnostic_battery.txt"
        self.training_json_path = self.training_dir / "latest_training_orchestration.json"
        self.training_txt_path = self.training_dir / "latest_training_orchestration.txt"
        self.training_zip_path = self.training_dir / "latest_training_orchestration_bundle.zip"

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_training_fingerprint: str | None = None

    def latest_status(self) -> dict:
        data = read_json(self.status_path, {})
        return data if isinstance(data, dict) else {}

    def latest_diagnostic_battery(self) -> dict:
        data = read_json(self.diag_json_path, {})
        return data if isinstance(data, dict) else {}

    def latest_training_orchestration(self) -> dict:
        data = read_json(self.training_json_path, {})
        return data if isinstance(data, dict) else {}

    def start_background_threads(self) -> None:
        if not bool(getattr(self.config, "automation_enabled", True)):
            return
        self.refresh(reason="startup", event_type="startup")
        self.refresh_training_only(reason="startup")
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="evidence-automation")
        self._thread.start()

    def stop_background_threads(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        interval = max(10, int(getattr(self.config, "automation_training_poll_seconds", 30) or 30))
        while not self._stop.wait(interval):
            try:
                self.refresh_training_only(reason="poll")
            except Exception as exc:  # pragma: no cover
                logger.warning("evidence_automation_training_poll_failed error=%s", exc)

    def _safe_json(self, payload: Any) -> str:
        return json.dumps(payload, indent=2, sort_keys=False, default=str)

    def _safe_build(self, fn: Callable[..., Any], *args, **kwargs) -> dict:
        try:
            result = fn(*args, **kwargs)
            return result if isinstance(result, dict) else {"available": False, "error": "unexpected_result_type"}
        except FileNotFoundError as exc:
            return {"available": False, "error": str(exc)}
        except Exception as exc:  # pragma: no cover
            logger.warning("evidence_automation_build_failed fn=%s error=%s", getattr(fn, "__name__", "unknown"), exc)
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def _build_training_orchestration_payload(self, *, reason: str = "api", submission_result: str | None = None) -> dict:
        training = self.state.get_training() or {}
        last_result = training.get("last_result") if isinstance(training.get("last_result"), dict) else {}
        model_artifact_path = str(getattr(self.config, "model_path_pt2", ""))
        payload = {
            "app_version": APP_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "reason": reason,
            "submission_result": submission_result,
            "training_submission_supported": True,
            "training_status_polling_supported": True,
            "training_artifact_generation_supported": True,
            "operator_readable_status_reporting_supported": True,
            "auto_live_promotion_supported": False,
            "auto_live_promotion_blocked": True,
            "manual_training_submission_mutates_live_model_bundle": True,
            "live_model_swap_behavior": "No automatic live promotion or live-model swap is performed by this tranche. Existing /train behavior still writes the pt2 bundle path directly when the operator starts training.",
            "training": {
                "running": bool(training.get("running")),
                "phase": training.get("phase"),
                "message": training.get("message"),
                "started_at_utc": training.get("started_at_utc"),
                "finished_at_utc": training.get("finished_at_utc"),
                "heartbeat_utc": training.get("heartbeat_utc"),
                "symbols_total": int(training.get("symbols_total") or 0),
                "symbols_done": int(training.get("symbols_done") or 0),
                "skipped_symbols_count": int(training.get("skipped_symbols_count") or 0),
                "rows_accumulated": int(training.get("rows_accumulated") or 0),
                "last_error": training.get("last_error"),
            },
            "latest_training_result": {
                "available": bool(last_result),
                "model_type": last_result.get("model_type"),
                "trained_at_utc": last_result.get("trained_at_utc") or last_result.get("trained_at"),
                "training_rows": last_result.get("training_rows") or last_result.get("rows_all"),
                "quality_event_rate": last_result.get("quality_event_rate") or last_result.get("event_rate_all"),
                "recommended_live_threshold": last_result.get("recommended_live_threshold"),
                "high_confidence_ready": last_result.get("high_confidence_ready"),
            },
            "model_artifact": {
                "path": model_artifact_path,
                "exists": bool(model_artifact_path and Path(model_artifact_path).exists()),
            },
            "unsafe_actions_blocked": [
                "automatic live promotion",
                "automatic live-model swap",
                "automatic Stage 1 mode switch",
                "automatic threshold change",
                "automatic Stage 2 semantic change",
            ],
        }
        return payload

    def _write_training_artifacts(self, payload: dict) -> dict:
        atomic_write_json(self.training_json_path, payload)
        _atomic_write_text(self.training_txt_path, self._safe_json(payload))
        with zipfile.ZipFile(self.training_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("training_orchestration.json", self._safe_json(payload))
            zf.writestr("training_orchestration.txt", self._safe_json(payload))
        return payload

    def refresh_training_only(self, *, reason: str = "poll", submission_result: str | None = None) -> dict:
        payload = self._build_training_orchestration_payload(reason=reason, submission_result=submission_result)
        fingerprint = json.dumps(payload.get("training") or {}, sort_keys=True, default=str) + json.dumps(payload.get("latest_training_result") or {}, sort_keys=True, default=str)
        if fingerprint != self._last_training_fingerprint or submission_result is not None:
            self._write_training_artifacts(payload)
            self._last_training_fingerprint = fingerprint
            status = self.latest_status()
            if status:
                status["latest_training_orchestration_generated_at_utc"] = payload.get("generated_at_utc")
                status["latest_training_orchestration_path"] = str(self.training_json_path)
                atomic_write_json(self.status_path, status)
        return payload

    def start_training(self) -> dict:
        started = self.trainer.start_training()
        payload = self.refresh_training_only(
            reason="training_submission",
            submission_result="training_started" if started else "training_already_running",
        )
        return {
            "started": started,
            "message": "training started" if started else "training already running",
            "orchestration": payload,
        }

    def _safe_branch_auto_handle(self, *, checkpoint: dict, branch: dict) -> dict:
        action = dict((branch or {}).get("branch_action") or {})
        performed = False
        details = {
            "performed": False,
            "action": None,
            "reason": None,
            "checkpoint_acknowledged": False,
            "branch_acknowledged": False,
        }
        if not bool(getattr(self.config, "automation_safe_branch_ack_enabled", True)):
            details["reason"] = "safe_branch_ack_disabled"
            return details
        if not checkpoint.get("triggered"):
            details["reason"] = "checkpoint_not_triggered"
            return details
        if (checkpoint.get("decision_checkpoint_outcome") or checkpoint.get("current_outcome")) != "confirmed":
            details["reason"] = "checkpoint_not_confirmed"
            return details
        if action.get("next_action_id") != "keep_live_path_unchanged":
            details["reason"] = "branch_not_safe_noop"
            return details
        if bool(action.get("manual_required")):
            details["reason"] = "manual_follow_up_required"
            return details
        if checkpoint.get("one_time_notification_pending"):
            self.decision_checkpoint.acknowledge()
            performed = True
            details["checkpoint_acknowledged"] = True
        if branch.get("branch_notification_pending"):
            self.decision_branch_automation.acknowledge()
            performed = True
            details["branch_acknowledged"] = True
        details["performed"] = performed
        details["action"] = "auto_acknowledge_no_change_branch" if performed else "no_action_required"
        details["reason"] = "safe_no_change_branch" if performed else "already_acknowledged"
        return details

    def _build_diagnostic_battery(self, *, status: dict, current_version_summary: dict, checkpoint: dict, branch: dict, training: dict, reason: str, event_type: str | None, latest_evaluated_run_id: str | None) -> dict:
        misranking = self._safe_build(self.misranking_diagnostic.build_summary)
        threshold_boundary = self._safe_build(self.threshold_boundary_review.build_summary)
        cooldown_shortlist = self._safe_build(self.cooldown_shortlist_review.build_summary)
        stage2_retrain = self._safe_build(self.stage2_retrain_review.build_summary)
        model_output = self._safe_build(self.model_output_distribution.latest_summary)
        evidence = dict((current_version_summary or {}).get("evidence") or {})
        branch_action = dict((branch or {}).get("branch_action") or {})
        checkpoint_outcome = checkpoint.get("decision_checkpoint_outcome") or checkpoint.get("current_outcome")
        v2_node = "confirmed_checkpoint_keep_live_path_unchanged" if checkpoint_outcome == "confirmed" else "non_confirmed_review_required"
        verdict = {
            "app_version": APP_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "reason": reason,
            "event_type": event_type,
            "latest_evaluated_run_id": latest_evaluated_run_id,
            "v2_node": v2_node,
            "checkpoint_outcome": checkpoint_outcome,
            "causal_layer_change_justified": False if checkpoint_outcome == "confirmed" else None,
            "headline": "Confirmed checkpoint — preserve live path, automate evidence handling" if checkpoint_outcome == "confirmed" else "Automation verdict available for review",
            "summary": (
                "The live path is confirmed under V2. This diagnostic battery is control-plane only and does not authorize a causal-layer tranche."
                if checkpoint_outcome == "confirmed"
                else "This diagnostic battery does not itself change live scoring logic."
            ),
            "allowed_actions": [
                "checkpoint refresh",
                "branch refresh",
                "safe no-op acknowledgement for confirmed keep-path-unchanged branches",
                "post-maturity review bundle generation",
                "diagnostic verdict generation",
                "training submission/status orchestration without auto-promotion",
            ],
            "disallowed_actions": [
                "automatic threshold changes",
                "automatic Stage 1 switches",
                "automatic Stage 2 changes",
                "automatic penalty/cap changes",
                "automatic retrain promotion",
                "automatic live-model swaps",
            ],
            "evidence_snapshot": {
                "resolved_visible_rows": checkpoint.get("resolved_visible_rows"),
                "visible_quality_hit_rate": checkpoint.get("current_visible_quality_hit_rate") or evidence.get("visible_quality_hit_rate"),
                "non_visible_quality_hit_rate": checkpoint.get("current_non_visible_quality_hit_rate") or evidence.get("non_visible_quality_hit_rate"),
                "threshold_bands": evidence.get("threshold_bands") or [],
                "score_range": evidence.get("score_range") or {},
            },
            "watch_items": {
                "stage1_omission_audit_verdict": ((current_version_summary.get("stage1_omission_audit_latest") or {}).get("verdict") if isinstance(current_version_summary, dict) else None),
                "stage1_selection_repair_headline": ((current_version_summary.get("stage1_selection_repair_review_latest") or {}).get("headline") if isinstance(current_version_summary, dict) else None),
                "threshold_boundary_headline": threshold_boundary.get("headline"),
                "misranking_headline": misranking.get("headline"),
                "cooldown_shortlist_headline": cooldown_shortlist.get("headline"),
                "stage2_retrain_headline": stage2_retrain.get("headline"),
                "model_output_distribution_headline": model_output.get("headline") if isinstance(model_output, dict) else None,
                "model_output_avg_ge_0_45": ((model_output.get("average_upper_tail_counts_per_scan") or {}).get("ge_0.45") if isinstance(model_output, dict) else None),
                "model_output_fraction_zero_ge_0_45": model_output.get("fraction_of_scans_with_zero_ge_0.45_rows") if isinstance(model_output, dict) else None,
                "model_output_max_score_seen": model_output.get("max_score_seen_in_window") if isinstance(model_output, dict) else None,
            },
            "branch_action": branch_action,
            "training_orchestration": {
                "auto_live_promotion_blocked": bool(training.get("auto_live_promotion_blocked")),
                "manual_training_submission_mutates_live_model_bundle": bool(training.get("manual_training_submission_mutates_live_model_bundle")),
            },
            "diagnostics": {
                "misranking": misranking,
                "threshold_boundary": threshold_boundary,
                "cooldown_shortlist": cooldown_shortlist,
                "stage2_retrain_review": stage2_retrain,
                "model_output_distribution": model_output,
            },
        }
        return verdict

    def _diagnostic_text(self, diagnostic: dict) -> str:
        evidence = diagnostic.get("evidence_snapshot") or {}
        watch_items = diagnostic.get("watch_items") or {}
        allowed = diagnostic.get("allowed_actions") or []
        disallowed = diagnostic.get("disallowed_actions") or []
        lines = [
            f"Coinbase Crypto Prob Scanner automation verdict {diagnostic.get('app_version')}",
            f"Generated UTC: {diagnostic.get('generated_at_utc')}",
            f"V2 node: {diagnostic.get('v2_node')}",
            f"Checkpoint outcome: {diagnostic.get('checkpoint_outcome')}",
            f"Headline: {diagnostic.get('headline')}",
            f"Summary: {diagnostic.get('summary')}",
            "",
            "Evidence snapshot:",
            f"- Resolved visible rows: {evidence.get('resolved_visible_rows')}",
            f"- Visible quality hit rate: {evidence.get('visible_quality_hit_rate')}",
            f"- Non-visible quality hit rate: {evidence.get('non_visible_quality_hit_rate')}",
            f"- Score range: {json.dumps(evidence.get('score_range') or {}, default=str)}",
            "",
            "Allowed actions:",
        ]
        lines.extend(f"- {item}" for item in allowed)
        lines.extend(["", "Disallowed actions:"])
        lines.extend(f"- {item}" for item in disallowed)
        lines.extend(["", "Watch items:"])
        for key, value in watch_items.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _write_diagnostic_artifacts(self, diagnostic: dict) -> None:
        atomic_write_json(self.diag_json_path, diagnostic)
        _atomic_write_text(self.diag_txt_path, self._diagnostic_text(diagnostic))

    def _latest_evaluated_pack(self, explicit_run_id: str | None = None, explicit_pack_path: str | Path | None = None) -> dict:
        if explicit_pack_path and Path(explicit_pack_path).exists():
            return {
                "available": True,
                "run_id": explicit_run_id,
                "pack_path": str(Path(explicit_pack_path)),
                "filename": Path(explicit_pack_path).name,
            }
        pack = self.review_packs.latest_eval_link if getattr(self.review_packs, "latest_eval_link", None) and self.review_packs.latest_eval_link.exists() else None
        latest_run_id = explicit_run_id
        if latest_run_id is None:
            for row in self.review_packs.get_runs(limit=50):
                if row.get("evaluation_complete"):
                    latest_run_id = str(row.get("run_id"))
                    break
        return {
            "available": bool(pack),
            "run_id": latest_run_id,
            "pack_path": str(pack) if pack else None,
            "filename": pack.name if pack else None,
        }

    def _build_post_maturity_review_bundle(self, *, status: dict, current_version_summary: dict, checkpoint: dict, branch: dict, diagnostic: dict, training: dict, latest_evaluated: dict, event_type: str | None) -> dict:
        manifest = {
            "app_version": APP_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "event_type": event_type,
            "latest_evaluated_run_id": latest_evaluated.get("run_id"),
            "latest_evaluated_pack_path": latest_evaluated.get("pack_path"),
            "includes_latest_evaluated_pack": bool(latest_evaluated.get("available")),
            "stage1_selection_mode": checkpoint.get("stage1_selection_mode"),
            "live_raw_threshold": checkpoint.get("live_raw_threshold"),
            "live_logic_unchanged": True,
        }
        with zipfile.ZipFile(self.bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", self._safe_json(manifest))
            zf.writestr("status.json", self._safe_json(status))
            zf.writestr("status.txt", self._safe_json(status))
            zf.writestr("current_version_summary.json", self._safe_json(current_version_summary))
            zf.writestr("current_version_summary.txt", self._safe_json(current_version_summary))
            zf.writestr("decision_checkpoint.json", self._safe_json(checkpoint))
            zf.writestr("decision_branch_automation.json", self._safe_json(branch))
            zf.writestr("diagnostic_battery.json", self._safe_json(diagnostic))
            zf.writestr("diagnostic_battery.txt", self._diagnostic_text(diagnostic))
            zf.writestr("training_orchestration.json", self._safe_json(training))
            zf.writestr("training_orchestration.txt", self._safe_json(training))
            zf.writestr("model_output_distribution_summary.json", self._safe_json(self.model_output_distribution.latest_summary()))
            zf.writestr("model_output_distribution_summary.txt", self._safe_json(self.model_output_distribution.latest_summary()))
            if latest_evaluated.get("available") and latest_evaluated.get("pack_path") and Path(latest_evaluated["pack_path"]).exists():
                zf.write(str(latest_evaluated["pack_path"]), arcname=f"embedded/{Path(latest_evaluated['pack_path']).name}")
        atomic_write_json(self.bundle_manifest_path, manifest)
        return manifest

    def refresh(self, *, reason: str = "manual", event_type: str | None = None, latest_evaluated_run_id: str | None = None, latest_evaluated_pack_path: str | Path | None = None) -> dict:
        status = self.state.get_status() or {}
        current_version_summary = self._safe_build(self.review_packs.get_current_version_summary)
        checkpoint = self.decision_checkpoint.build_summary(summary=current_version_summary)
        branch = self.decision_branch_automation.build_summary(checkpoint_summary=checkpoint)
        safe_branch = self._safe_branch_auto_handle(checkpoint=checkpoint, branch=branch)
        if safe_branch.get("performed"):
            checkpoint = self.decision_checkpoint.latest_summary() or self.decision_checkpoint.build_summary(summary=current_version_summary)
            branch = self.decision_branch_automation.build_summary(checkpoint_summary=checkpoint)
        training = self.refresh_training_only(reason=reason)
        latest_evaluated = self._latest_evaluated_pack(explicit_run_id=latest_evaluated_run_id, explicit_pack_path=latest_evaluated_pack_path)
        diagnostic = self._build_diagnostic_battery(
            status=status,
            current_version_summary=current_version_summary,
            checkpoint=checkpoint,
            branch=branch,
            training=training,
            reason=reason,
            event_type=event_type,
            latest_evaluated_run_id=latest_evaluated.get("run_id"),
        )
        self._write_diagnostic_artifacts(diagnostic)
        bundle_manifest = None
        if bool(getattr(self.config, "automation_review_bundle_enabled", True)):
            bundle_manifest = self._build_post_maturity_review_bundle(
                status=status,
                current_version_summary=current_version_summary,
                checkpoint=checkpoint,
                branch=branch,
                diagnostic=diagnostic,
                training=training,
                latest_evaluated=latest_evaluated,
                event_type=event_type,
            )
        automation_status = {
            "app_version": APP_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "reason": reason,
            "event_type": event_type,
            "state_scope_key": branch.get("state_scope_key") or checkpoint.get("state_scope_key"),
            "checkpoint_outcome": checkpoint.get("decision_checkpoint_outcome") or checkpoint.get("current_outcome"),
            "branch_status": ((branch.get("branch_action") or {}).get("status")),
            "safe_branch_auto_handle": safe_branch,
            "latest_evaluated_run_id": latest_evaluated.get("run_id"),
            "latest_evaluated_pack_path": latest_evaluated.get("pack_path"),
            "latest_post_maturity_review_bundle_path": str(self.bundle_path) if self.bundle_path.exists() else None,
            "latest_post_maturity_review_bundle_generated_at_utc": (bundle_manifest or {}).get("generated_at_utc"),
            "latest_diagnostic_battery_path": str(self.diag_json_path),
            "latest_diagnostic_battery_generated_at_utc": diagnostic.get("generated_at_utc"),
            "latest_training_orchestration_path": str(self.training_json_path),
            "latest_training_orchestration_generated_at_utc": training.get("generated_at_utc"),
            "unsafe_auto_actions_blocked": [
                "threshold changes",
                "Stage 1 switches",
                "Stage 2 changes",
                "penalty/cap changes",
                "retrain promotion",
                "live-model swap",
            ],
        }
        atomic_write_json(self.status_path, automation_status)
        return automation_status

    def handle_post_maturity(self, run_id: str, pack_path: str | Path) -> dict:
        return self.refresh(
            reason="post_maturity",
            event_type="post_maturity_evidence_ready",
            latest_evaluated_run_id=str(run_id),
            latest_evaluated_pack_path=str(pack_path),
        )
