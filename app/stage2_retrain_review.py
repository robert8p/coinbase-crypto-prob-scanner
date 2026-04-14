from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import AppConfig
from .modeling import ModelBundle
from .persist import atomic_write_json, ensure_dir


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


class Stage2RetrainReviewService:
    def __init__(self, config: AppConfig, review_packs, state):
        self.config = config
        self.review_packs = review_packs
        self.state = state
        self.summary_path = Path(config.model_dir) / "stage2_retrain_review_summary.json"
        self.pack_path = Path(config.model_dir) / "stage2_retrain_review_pack.zip"

    def latest_summary(self) -> dict:
        if self.summary_path.exists():
            try:
                return json.loads(self.summary_path.read_text())
            except Exception:
                return {}
        return {}

    def _load_model_metadata(self) -> dict:
        meta = ((self.state.model_metadata or {}).get("pt2") or {}) if getattr(self.state, "model_metadata", None) else {}
        if meta:
            return dict(meta)
        bundle = ModelBundle.load(self.config.model_path_pt2)
        if bundle is not None:
            return dict(bundle.metadata or {})
        return {}

    def build_summary(self) -> dict:
        try:
            current_version_summary = self.review_packs.get_current_version_summary()
        except Exception:
            current_version_summary = {}
        status = self.state.get_status() or {}
        model_meta = self._load_model_metadata()
        summary = self._build_from_inputs(status=status, current_version_summary=current_version_summary, model_meta=model_meta)
        ensure_dir(self.summary_path.parent)
        atomic_write_json(self.summary_path, summary)
        return summary

    def _build_from_inputs(self, *, status: dict, current_version_summary: dict, model_meta: dict) -> dict:
        status = dict(status or {})
        current_version_summary = dict(current_version_summary or {})
        model_meta = dict(model_meta or {})

        candidate_quality = status.get("candidate_quality") or {}
        score_diag = status.get("score_diagnostics") or {}
        omission = status.get("stage1_omission_audit") or current_version_summary.get("stage1_omission_audit_latest") or {}
        selection_repair = status.get("stage1_selection_repair_review") or current_version_summary.get("stage1_selection_repair_review_latest") or {}
        threshold_review = status.get("threshold_experiment_review") or current_version_summary.get("threshold_experiment_review_latest") or {}
        scan = status.get("scan") or {}
        regime = status.get("market_regime") or {}
        evidence = current_version_summary.get("evidence") or {}
        score_dist = ((model_meta.get("score_distribution_adjusted") or {}).get("score_quantiles") or {})
        top_bucket_lift = ((model_meta.get("score_distribution_adjusted") or {}).get("top_bucket_lift") or {})

        current_stage1_mode = str(candidate_quality.get("stage1_selection_mode") or status.get("stage1_selection_mode") or self.config.stage1_selection_mode or "unknown")
        live_threshold = _f(status.get("effective_live_raw_threshold"), _f(self.config.live_raw_threshold, 0.35)) or 0.35
        max_live = _f(((score_diag.get("live_score") or {}).get("max")), 0.0) or 0.0
        count_ge_035 = 0
        count_ge_045 = 0
        for row in (score_diag.get("counts_above_thresholds") or []):
            threshold = _f(row.get("threshold"))
            if threshold == 0.35:
                count_ge_035 = int(row.get("live_count") or 0)
            elif threshold == 0.45:
                count_ge_045 = int(row.get("live_count") or 0)

        q95 = _f(score_dist.get("q95"))
        q99 = _f(score_dist.get("q99"))
        historical_max = _f(score_dist.get("max"))
        upper_tail_gap_vs_q99 = round((q99 - max_live), 4) if q99 is not None else None
        upper_tail_gap_vs_threshold = round((live_threshold - max_live), 4)
        threshold_added = int(threshold_review.get("additional_visible_count") or 0)
        threshold_added_ge_045 = int(((threshold_review.get("added_band_counts") or {}).get("count_ge_0_45")) or 0)

        candidate_spec = {
            "label": "shadow_recency_retrain_90d_dense120",
            "train_lookback_days": 90,
            "train_max_symbols": 120,
            "train_sample_every_n_bars": 2,
            "recency_weight_start": 0.70,
            "recency_weight_end": 1.40,
            "keep_stage1_selection_mode": current_stage1_mode,
            "keep_live_raw_threshold": round(float(live_threshold), 4),
            "switch_live_model_automatically": False,
            "purpose": "test whether a fresher, denser Stage 2 training slice can create a stronger live upper tail before any live model switch",
        }

        rationale: list[str] = []
        if current_stage1_mode == "stage1_opportunity_model":
            rationale.append("Stage 1 already runs in stage1_opportunity_model, so the main shortlist lever has been exercised.")
        if str(omission.get("verdict") or "") == "stage2_score_compression_likely":
            rationale.append("The Stage 1 omission audit points to Stage 2 score compression rather than missed Stage 1 winners.")
        if str(selection_repair.get("verdict") or "") == "no_clear_stage1_repair_mode":
            rationale.append("No alternative Stage 1 mode clearly beats the current live mode on this scan.")
        if max_live < live_threshold:
            rationale.append(f"The current scan max live score ({max_live:.4f}) still sits below the live threshold ({live_threshold:.2f}).")
        if q99 is not None:
            rationale.append(f"Current live max ({max_live:.4f}) remains below the model's historical adjusted q99 ({q99:.4f}).")
        if threshold_added > 0 and threshold_added_ge_045 == 0:
            rationale.append(f"Lowering threshold to 0.28 would add breadth ({threshold_added} rows) but no >=0.45 candidates in this scan.")
        if (regime.get("state") or "") == "green":
            rationale.append("This scan ran in a green / normal regime, so the weak upper tail is less likely to be explained by regime suppression.")

        supported = (
            current_stage1_mode == "stage1_opportunity_model"
            and str(omission.get("verdict") or "") == "stage2_score_compression_likely"
            and str(selection_repair.get("verdict") or "") in {"no_clear_stage1_repair_mode", "mixed_or_inconclusive", ""}
            and max_live < live_threshold
            and count_ge_035 == 0
            and (regime.get("state") or "") == "green"
        )

        if supported:
            verdict = "stage2_shadow_retrain_supported"
            headline = "Stage 2 shadow retrain is the next justified lever"
            summary = (
                "The live app is already on stage1_opportunity_model at 0.35, yet the current green-regime scan still produced no names above threshold. "
                "With Stage 1 omission no longer leading and threshold-lowering only adding exploratory breadth, the next evidence-driven move is a shadow recency retrain review for Stage 2."
            )
            recommended_action = "build_and_run_shadow_retrain_candidate"
            recommended_action_reason = "The current live path appears Stage 2-limited, not Stage 1-limited."
        else:
            verdict = "keep_current_live_path_collect_more_or_recheck"
            headline = "Stage 2 shadow retrain is not yet cleanly justified"
            summary = (
                "The evidence is not yet clean enough to escalate to a Stage 2 shadow retrain recommendation. Keep the current live path stable and re-check once the Stage 1 and threshold diagnostics are unambiguous."
            )
            recommended_action = "keep_live_path_unchanged"
            recommended_action_reason = "The current scan and review evidence do not yet isolate Stage 2 strongly enough."

        decision_memo_markdown = (
            "# Stage 2 retrain review\n\n"
            f"- **Headline:** {headline}\n"
            f"- **Verdict:** {verdict}\n"
            f"- **Summary:** {summary}\n\n"
            "## Why this verdict\n"
            + "\n".join(f"- {item}" for item in rationale)
            + "\n\n## Candidate shadow retrain spec\n"
            + "\n".join(f"- **{k}**: {v}" for k, v in candidate_spec.items())
        )

        return {
            "available": True,
            "generated_at_utc": _utc_now_iso(),
            "app_version": status.get("app_version") or current_version_summary.get("app_version"),
            "headline": headline,
            "summary": summary,
            "verdict": verdict,
            "recommended_action": recommended_action,
            "recommended_action_reason": recommended_action_reason,
            "current_scan": {
                "scan_finished_at_utc": scan.get("finished_at_utc"),
                "market_regime_state": regime.get("state"),
                "market_regime_actionability": regime.get("actionability_state") or regime.get("effective_actionability_state"),
                "stage1_selection_mode": current_stage1_mode,
                "effective_live_raw_threshold": round(float(live_threshold), 4),
                "visible_rows": int((status.get("stage_counts") or {}).get("visible_rows") or 0),
                "stage2_scored": int((status.get("stage_counts") or {}).get("stage2_scored") or 0),
                "max_live_score": round(float(max_live), 4),
                "count_ge_0_35": count_ge_035,
                "count_ge_0_45": count_ge_045,
            },
            "model_tail_reference": {
                "trained_at_utc": model_meta.get("trained_at_utc"),
                "model_fingerprint": model_meta.get("model_fingerprint"),
                "q95_adjusted": round(float(q95), 4) if q95 is not None else None,
                "q99_adjusted": round(float(q99), 4) if q99 is not None else None,
                "historical_max_adjusted": round(float(historical_max), 4) if historical_max is not None else None,
                "top_5pct_lift": _f(top_bucket_lift.get("top_5pct")),
                "top_10pct_lift": _f(top_bucket_lift.get("top_10pct")),
                "upper_tail_gap_vs_q99": upper_tail_gap_vs_q99,
                "upper_tail_gap_vs_live_threshold": upper_tail_gap_vs_threshold,
            },
            "decision_inputs": {
                "stage1_omission_verdict": omission.get("verdict"),
                "stage1_selection_repair_verdict": selection_repair.get("verdict"),
                "threshold_experiment_verdict": threshold_review.get("verdict"),
                "threshold_experiment_additional_visible_count": threshold_added,
                "threshold_experiment_added_ge_0_45": threshold_added_ge_045,
                "resolved_rows_current_version": int(evidence.get("resolved_rows") or 0),
                "visible_quality_hit_rate_current_version": evidence.get("visible_quality_hit_rate"),
                "non_visible_quality_hit_rate_current_version": evidence.get("non_visible_quality_hit_rate"),
            },
            "candidate_shadow_retrain_spec": candidate_spec,
            "rationale": rationale,
            "decision_memo_markdown": decision_memo_markdown,
            "notes": [
                "This is a diagnostic recommendation only. It does not switch the live model.",
                "Use this to justify or reject a shadow retrain run before any live model replacement.",
            ],
        }

    def build_pack(self) -> Path:
        summary = self.build_summary()
        ensure_dir(self.pack_path.parent)
        status = self.state.get_status() or {}
        try:
            current_version_summary = self.review_packs.get_current_version_summary()
        except Exception:
            current_version_summary = {}
        model_meta = self._load_model_metadata()
        with zipfile.ZipFile(self.pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage2_retrain_review_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr("current_status_excerpt.json", json.dumps({
                "app_version": status.get("app_version"),
                "effective_live_raw_threshold": status.get("effective_live_raw_threshold"),
                "candidate_quality": status.get("candidate_quality") or {},
                "score_diagnostics": status.get("score_diagnostics") or {},
                "stage1_omission_audit": status.get("stage1_omission_audit") or {},
                "stage1_selection_repair_review": status.get("stage1_selection_repair_review") or {},
                "threshold_experiment_review": status.get("threshold_experiment_review") or {},
            }, indent=2, sort_keys=True))
            zf.writestr("current_version_summary_excerpt.json", json.dumps({
                "app_version": current_version_summary.get("app_version"),
                "generated_at_utc": current_version_summary.get("generated_at_utc"),
                "deployed_since_utc": current_version_summary.get("deployed_since_utc"),
                "evidence": current_version_summary.get("evidence") or {},
            }, indent=2, sort_keys=True))
            zf.writestr("model_metadata_excerpt.json", json.dumps({
                "trained_at_utc": model_meta.get("trained_at_utc"),
                "model_fingerprint": model_meta.get("model_fingerprint"),
                "score_distribution_adjusted": model_meta.get("score_distribution_adjusted") or {},
                "auc_holdout": model_meta.get("auc_holdout"),
                "adjusted_auc_holdout": model_meta.get("adjusted_auc_holdout"),
                "brier_holdout": model_meta.get("brier_holdout"),
                "adjusted_brier_holdout": model_meta.get("adjusted_brier_holdout"),
            }, indent=2, sort_keys=True))
            zf.writestr("decision_memo.md", summary.get("decision_memo_markdown") or "")
        return self.pack_path
