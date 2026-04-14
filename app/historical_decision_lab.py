from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from .benchmark import BenchmarkLabService
from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .replay import HistoricalReplayService
from .review_runs import ReviewPackService
from .stage2_retrain_review import Stage2RetrainReviewService


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


class HistoricalDecisionLabService:
    """Operator-friendly offline research suite.

    This intentionally stays off the causal live path. It reuses existing historical
    replay and benchmark machinery so the operator can prepare future decisions
    faster without touching live scoring logic.
    """

    def __init__(
        self,
        config: AppConfig,
        replay: HistoricalReplayService,
        benchmark_lab: BenchmarkLabService,
        review_packs: ReviewPackService,
        stage2_retrain_review: Stage2RetrainReviewService,
    ):
        self.config = config
        self.replay = replay
        self.benchmark_lab = benchmark_lab
        self.review_packs = review_packs
        self.stage2_retrain_review = stage2_retrain_review
        self.root_dir = ensure_dir(Path(config.model_dir) / "historical_decision_lab")
        self.summary_path = self.root_dir / "latest_historical_decision_lab_summary.json"
        self.pack_path = self.root_dir / "latest_historical_decision_lab_pack.zip"

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def run(
        self,
        *,
        hours: int = 168,
        step_minutes: int = 120,
        max_scans: int = 84,
        max_symbols: int = 100,
        thresholds: str | Iterable[float] | None = None,
    ) -> dict:
        current_version = self.review_packs.get_current_version_summary() or {}
        live_threshold = self._current_live_threshold(current_version)
        checkpoint = current_version.get("decision_checkpoint") or current_version.get("decision_rule_checkpoint") or {}
        replay_result = self.replay.run(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            pipeline_mode="raw_threshold",
            raw_threshold=live_threshold,
        )
        replay_summary = dict(replay_result.get("summary") or {})
        benchmark_summary = self.benchmark_lab.run_threshold_sweep(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            thresholds=thresholds,
        )
        benchmark_pack = self.benchmark_lab.build_benchmark_pack()
        classification_pack = self.benchmark_lab.build_symbol_classification_pack()
        stage2_summary = self.stage2_retrain_review.build_summary()
        replay_pack = self.replay.latest_pack()
        summary = self._build_summary(
            current_version=current_version,
            checkpoint=checkpoint,
            replay_summary=replay_summary,
            benchmark_summary=benchmark_summary,
            stage2_summary=stage2_summary,
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            thresholds=thresholds,
        )
        atomic_write_json(self.summary_path, summary)
        self._build_pack(
            summary=summary,
            current_version=current_version,
            replay_summary=replay_summary,
            benchmark_summary=benchmark_summary,
            stage2_summary=stage2_summary,
            replay_pack=replay_pack,
            benchmark_pack=benchmark_pack,
            classification_pack=classification_pack,
        )
        return summary

    def _current_live_threshold(self, current_version: dict) -> float:
        checkpoint = current_version.get("decision_checkpoint") or current_version.get("decision_rule_checkpoint") or {}
        return _f(
            checkpoint.get("live_raw_threshold")
            or checkpoint.get("effective_live_raw_threshold")
            or (current_version.get("decision_branch_automation") or {}).get("effective_live_raw_threshold")
            or self.config.live_raw_threshold,
            0.35,
        ) or 0.35

    def _build_summary(
        self,
        *,
        current_version: dict,
        checkpoint: dict,
        replay_summary: dict,
        benchmark_summary: dict,
        stage2_summary: dict,
        hours: int,
        step_minutes: int,
        max_scans: int,
        max_symbols: int,
        thresholds: str | Iterable[float] | None,
    ) -> dict:
        evidence = current_version.get("evidence") or {}
        outlier = current_version.get("outlier_concentration") or {}
        model_output = current_version.get("model_output_distribution") or {}
        recommendation = benchmark_summary.get("recommendation") or {}
        live_045 = ((outlier.get("thresholds") or {}).get("0.45") or {})
        live_060 = ((outlier.get("thresholds") or {}).get("0.60") or {})
        benchmark_rows = list(benchmark_summary.get("rows") or [])
        benchmark_best = None
        rec_threshold = recommendation.get("recommended_threshold")
        for row in benchmark_rows:
            if row.get("threshold") == rec_threshold:
                benchmark_best = row
                break
        if benchmark_best is None and benchmark_rows:
            benchmark_best = benchmark_rows[0]

        visible_q = _f(evidence.get("visible_quality_hit_rate"), 0.0) or 0.0
        hidden_q = _f(evidence.get("non_visible_quality_hit_rate"), 0.0) or 0.0
        outcome = checkpoint.get("current_outcome") or checkpoint.get("decision_checkpoint_outcome") or checkpoint.get("status")
        live_path_confirmed = str(outcome) == "confirmed"
        top_symbol_share_045 = _f(live_045.get("top_symbol_share"), 0.0) or 0.0
        top_symbol_share_060 = _f(live_060.get("top_symbol_share"), 0.0) or 0.0
        concentration_flag = top_symbol_share_045 >= 0.60 or top_symbol_share_060 >= 0.90
        threshold_delta = None
        if rec_threshold is not None:
            threshold_delta = round(float(rec_threshold) - self._current_live_threshold(current_version), 4)

        if live_path_confirmed:
            headline = "Live path is confirmed; offline historical suite is for future-causal preparation only"
            live_action = "keep_live_path_unchanged"
            live_action_reason = (
                f"Visible quality-hit rate ({visible_q:.2%}) still beats hidden ({hidden_q:.2%}) over the current deployment window."
            )
        else:
            headline = "Live path is not yet settled; offline historical suite should be used to narrow the next causal candidate"
            live_action = "hold_until_live_decision_boundary"
            live_action_reason = "The current live window is not yet at a clean no-change state."

        retrain_readiness = {
            "future_retrain_spec_should_include_symbol_concentration_controls": concentration_flag,
            "reason": (
                "The strongest upper-tail rows remain concentrated in a small number of symbols, so any future retrain spec should explicitly constrain symbol concentration."
                if concentration_flag
                else "Upper-tail concentration is not yet severe enough to require a hard concentration-control flag in a future retrain spec."
            ),
        }

        notes = [
            "This suite is offline-only. It must not change live Stage 1, threshold, Stage 2 semantics, or model promotion state.",
            "Use the sweep to reject weak next-causal ideas cheaply; use fresh live evidence later for final acceptance.",
        ]
        if benchmark_best is not None:
            notes.append(
                f"Offline sweep currently prefers threshold {benchmark_best.get('threshold'):.2f}, but that is not a live change instruction while the current live path remains confirmed."
            )
        if concentration_flag:
            notes.append(
                f"Upper-tail concentration is concrete: top >=0.45 symbol share is {top_symbol_share_045:.2%}; top >=0.60 symbol share is {top_symbol_share_060:.2%}."
            )

        decision_memo_markdown = (
            "# Historical decision lab\n\n"
            f"- **Headline:** {headline}\n"
            f"- **Live action:** {live_action}\n"
            f"- **Why:** {live_action_reason}\n\n"
            "## Current live state\n"
            f"- Confirmed checkpoint outcome: {outcome}\n"
            f"- Visible quality-hit rate: {visible_q:.2%}\n"
            f"- Hidden quality-hit rate: {hidden_q:.2%}\n"
            f"- Resolved visible rows: {int(checkpoint.get('resolved_visible_rows') or evidence.get('visible_rows') or 0)}\n\n"
            "## Offline sweep\n"
            f"- Hours: {int(hours)}\n"
            f"- Step minutes: {int(step_minutes)}\n"
            f"- Max scans: {int(max_scans)}\n"
            f"- Max symbols: {int(max_symbols)}\n"
            f"- Thresholds tested: {thresholds if thresholds is not None else 'default benchmark set'}\n"
            f"- Benchmark recommendation: {recommendation.get('recommended_threshold')}\n"
            f"- Recommendation reason: {recommendation.get('reason')}\n\n"
            "## Upper-tail concentration\n"
            f"- >=0.45 top symbol: {live_045.get('top_symbol')} ({top_symbol_share_045:.2%} share)\n"
            f"- >=0.60 top symbol: {live_060.get('top_symbol')} ({top_symbol_share_060:.2%} share)\n"
            f"- Future retrain spec should include concentration controls: {'yes' if concentration_flag else 'not yet required'}\n\n"
            "## How to use this\n"
            "- Reject weak candidate next moves offline first.\n"
            "- Keep live unchanged while the confirmed path remains confirmed.\n"
            "- Only let an offline winner become a live change after fresh live proof.\n"
        )

        return {
            "available": True,
            "generated_at_utc": _utc_now_iso(),
            "app_version": current_version.get("app_version"),
            "headline": headline,
            "summary": live_action_reason,
            "live_action": live_action,
            "live_action_reason": live_action_reason,
            "live_path_confirmed": live_path_confirmed,
            "live_current_threshold": self._current_live_threshold(current_version),
            "benchmark_recommendation": recommendation,
            "benchmark_threshold_delta_vs_live": threshold_delta,
            "historical_suite_inputs": {
                "hours": int(hours),
                "step_minutes": int(step_minutes),
                "max_scans": int(max_scans),
                "max_symbols": int(max_symbols),
                "thresholds": list(benchmark_summary.get("thresholds") or []),
            },
            "current_live_evidence": {
                "resolved_visible_rows": int(checkpoint.get("resolved_visible_rows") or 0),
                "visible_quality_hit_rate": evidence.get("visible_quality_hit_rate"),
                "non_visible_quality_hit_rate": evidence.get("non_visible_quality_hit_rate"),
                "visible_rows": evidence.get("visible_rows"),
                "non_visible_rows": evidence.get("non_visible_rows"),
                "threshold_0_45_quality_hit_rate": self._threshold_rate(evidence, 0.45),
            },
            "upper_tail_concentration": {
                "threshold_0_45": {
                    "row_count": int(live_045.get("row_count") or 0),
                    "top_symbol": live_045.get("top_symbol"),
                    "top_symbol_share": live_045.get("top_symbol_share"),
                    "unique_symbols": int(live_045.get("unique_symbols") or 0),
                },
                "threshold_0_60": {
                    "row_count": int(live_060.get("row_count") or 0),
                    "top_symbol": live_060.get("top_symbol"),
                    "top_symbol_share": live_060.get("top_symbol_share"),
                    "unique_symbols": int(live_060.get("unique_symbols") or 0),
                },
                "future_retrain_spec": retrain_readiness,
            },
            "model_output_distribution": {
                "headline": model_output.get("headline"),
                "scans_in_window": model_output.get("scans_in_window"),
                "average_upper_tail_counts_per_scan": dict(model_output.get("average_upper_tail_counts_per_scan") or {}),
                "fraction_of_scans_with_zero_ge_0.45_rows": model_output.get("fraction_of_scans_with_zero_ge_0.45_rows"),
                "fraction_of_scans_with_zero_ge_0.50_rows": model_output.get("fraction_of_scans_with_zero_ge_0.50_rows"),
                "max_score_seen_in_window": model_output.get("max_score_seen_in_window"),
            },
            "stage2_retrain_review": {
                "headline": stage2_summary.get("headline"),
                "verdict": stage2_summary.get("verdict"),
                "recommended_action": stage2_summary.get("recommended_action"),
            },
            "notes": notes,
            "decision_memo_markdown": decision_memo_markdown,
        }

    def _threshold_rate(self, evidence: dict, threshold: float) -> float | None:
        for row in (evidence.get("threshold_bands") or []):
            if _f(row.get("threshold")) == float(threshold):
                return _f(row.get("quality_hit_rate"))
        return None

    def _build_pack(
        self,
        *,
        summary: dict,
        current_version: dict,
        replay_summary: dict,
        benchmark_summary: dict,
        stage2_summary: dict,
        replay_pack: Path | None,
        benchmark_pack: Path | None,
        classification_pack: Path | None,
    ) -> Path:
        ensure_dir(self.pack_path.parent)
        with zipfile.ZipFile(self.pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("historical_decision_lab_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr("decision_memo.md", summary.get("decision_memo_markdown") or "")
            zf.writestr("current_version_summary.json", json.dumps(current_version, indent=2, sort_keys=True))
            zf.writestr("replay_summary_current_threshold.json", json.dumps(replay_summary, indent=2, sort_keys=True))
            zf.writestr("benchmark_summary.json", json.dumps(benchmark_summary, indent=2, sort_keys=True))
            zf.writestr("stage2_retrain_review_summary.json", json.dumps(stage2_summary, indent=2, sort_keys=True))
            if replay_pack and replay_pack.exists():
                zf.writestr(f"embedded/{replay_pack.name}", replay_pack.read_bytes())
            if benchmark_pack and benchmark_pack.exists():
                zf.writestr(f"embedded/{benchmark_pack.name}", benchmark_pack.read_bytes())
            if classification_pack and classification_pack.exists():
                zf.writestr(f"embedded/{classification_pack.name}", classification_pack.read_bytes())
        return self.pack_path
