from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from .config import AppConfig
from .persist import atomic_write_json, read_json
from .version import APP_VERSION
from .decision_branch_automation import effective_live_raw_threshold


GAP_BUCKETS = [0.02, 0.035, 0.05, 0.075]
SCENARIO_THRESHOLDS = [0.34, 0.33, 0.32, 0.30]
REPEATED_NEAR_THRESHOLD_MIN_SCANS = 3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pct(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "-"


def _csv_text(rows: List[dict], fieldnames: List[str]) -> str:
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    return sio.getvalue()


def _atomic_zip_write(path: Path, write_fn) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            write_fn(zf)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return path


class ThresholdBoundaryReviewService:
    def __init__(self, config: AppConfig, review_packs):
        self.config = config
        self.review_packs = review_packs
        self.summary_path = Path(config.model_dir) / "threshold_boundary_review_summary.json"
        self.pack_path = Path(config.model_dir) / "threshold_boundary_review_pack.zip"

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def _resolve_latest_evaluated_run(self, requested_version: str) -> dict:
        get_runs_for_app_version = getattr(self.review_packs, "get_runs_for_app_version", None)
        if callable(get_runs_for_app_version):
            for run in list(get_runs_for_app_version(requested_version, limit=250) or []):
                if bool(run.get("evaluation_complete")):
                    return {
                        "requested_version": requested_version,
                        "source_mode": "current_version_latest_evaluated_run",
                        "source_version": requested_version,
                        "run": run,
                        "fallback_used": False,
                        "fallback_reason": None,
                    }

        get_runs = getattr(self.review_packs, "get_runs", None)
        if callable(get_runs):
            for run in list(get_runs(limit=250) or []):
                run_version = str(run.get("app_version") or "")
                if run_version == requested_version:
                    continue
                if bool(run.get("evaluation_complete")):
                    return {
                        "requested_version": requested_version,
                        "source_mode": "fallback_recent_evaluated_run",
                        "source_version": run_version,
                        "run": run,
                        "fallback_used": True,
                        "fallback_reason": (
                            f"No evaluated runs are available yet for deployed version {requested_version}; "
                            f"using the most recent evaluated run from app version {run_version}."
                        ),
                    }
        raise FileNotFoundError(f"no evaluated review runs available for requested version {requested_version}")

    def _current_summary(self, app_version: str) -> dict:
        getter = getattr(self.review_packs, "get_current_version_summary", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter(app_version=app_version) or {})
        except FileNotFoundError:
            return {}

    def _resolved_threshold_rows(self, rows: List[dict]) -> List[dict]:
        selected = []
        for row in list(rows or []):
            if not bool(row.get("resolved")):
                continue
            if str(row.get("suppression_reason") or "") != "threshold":
                continue
            selected.append(row)
        selected.sort(key=lambda row: ((_f(row.get("live_score")) or -1.0), -(_f(row.get("end_ret")) or -9.0)), reverse=True)
        return selected

    def _metrics(self, rows: List[dict]) -> dict:
        count = len(rows)
        if count <= 0:
            return {
                "count": 0,
                "quality_hit_rate": None,
                "raw_hit_rate": None,
                "avg_end_ret": None,
                "avg_mae": None,
                "avg_mfe": None,
                "avg_time_to_touch_minutes": None,
            }
        quality_hits = sum(1 for row in rows if bool(row.get("quality_touched")))
        raw_hits = sum(1 for row in rows if bool(row.get("raw_touched")))
        end_rets = [_f(row.get("end_ret")) for row in rows if _f(row.get("end_ret")) is not None]
        maes = [_f(row.get("mae")) for row in rows if _f(row.get("mae")) is not None]
        mfes = [_f(row.get("mfe")) for row in rows if _f(row.get("mfe")) is not None]
        ttts = [_f(row.get("time_to_touch_minutes")) for row in rows if _f(row.get("time_to_touch_minutes")) is not None]
        return {
            "count": count,
            "quality_hit_rate": round(quality_hits / count, 4),
            "raw_hit_rate": round(raw_hits / count, 4),
            "avg_end_ret": round(sum(end_rets) / len(end_rets), 6) if end_rets else None,
            "avg_mae": round(sum(maes) / len(maes), 6) if maes else None,
            "avg_mfe": round(sum(mfes) / len(mfes), 6) if mfes else None,
            "avg_time_to_touch_minutes": round(sum(ttts) / len(ttts), 3) if ttts else None,
        }

    def _gap_bucket_rows(self, rows: List[dict], *, live_threshold: float) -> List[dict]:
        bucket_rows = []
        for gap in GAP_BUCKETS:
            bucket = [
                row for row in rows
                if _f(row.get("distance_to_live_threshold")) is not None and _f(row.get("distance_to_live_threshold")) <= gap
            ]
            metrics = self._metrics(bucket)
            bucket_rows.append({
                "gap_max": gap,
                "live_threshold": live_threshold,
                **metrics,
            })
        return bucket_rows

    def _threshold_scenarios(self, rows: List[dict], *, live_threshold: float) -> List[dict]:
        scenario_rows = []
        for scenario_threshold in SCENARIO_THRESHOLDS:
            promoted = [row for row in rows if (_f(row.get("live_score")) or -1.0) >= scenario_threshold]
            metrics = self._metrics(promoted)
            scenario_rows.append({
                "scenario_threshold": scenario_threshold,
                "current_live_threshold": live_threshold,
                "promoted_rows": metrics.pop("count"),
                **metrics,
            })
        return scenario_rows

    def _top_false_suppressions(self, rows: List[dict]) -> List[dict]:
        flagged = [
            {
                "symbol": str(row.get("symbol") or ""),
                "live_score": _f(row.get("live_score")),
                "pre_policy_score": _f(row.get("pre_policy_score")),
                "distance_to_live_threshold": _f(row.get("distance_to_live_threshold")),
                "quality_touched": bool(row.get("quality_touched")),
                "raw_touched": bool(row.get("raw_touched")),
                "end_ret": _f(row.get("end_ret")),
                "mae": _f(row.get("mae")),
                "mfe": _f(row.get("mfe")),
                "time_to_touch_minutes": _f(row.get("time_to_touch_minutes")),
                "liquidity_tier": row.get("liquidity_tier"),
                "market_regime_state": row.get("market_regime_state"),
            }
            for row in rows
            if bool(row.get("quality_touched")) or ((_f(row.get("end_ret")) or -9.0) > 0.0)
        ]
        flagged.sort(
            key=lambda row: (
                not bool(row.get("quality_touched")),
                row.get("distance_to_live_threshold") if row.get("distance_to_live_threshold") is not None else 999.0,
                -(row.get("end_ret") or -9.0),
            )
        )
        return flagged[:25]

    def _repeated_near_threshold_symbols(self, summary: dict, *, live_threshold: float) -> List[dict]:
        rows = list(((summary or {}).get("cohort_symbols") or {}).get("rows") or [])
        selected = []
        for row in rows:
            selected_scans = int(row.get("selected_scans") or 0)
            max_live_score = _f(row.get("max_live_score"))
            hidden_scans = int(row.get("hidden_scans") or 0)
            visible_scans = int(row.get("visible_scans") or 0)
            if selected_scans < REPEATED_NEAR_THRESHOLD_MIN_SCANS or max_live_score is None:
                continue
            gap = round(live_threshold - max_live_score, 6)
            if gap < 0 or gap > 0.05:
                continue
            if hidden_scans <= visible_scans:
                continue
            selected.append({
                "symbol": str(row.get("symbol") or ""),
                "liquidity_tier": row.get("liquidity_tier"),
                "selected_scans": selected_scans,
                "visible_scans": visible_scans,
                "hidden_scans": hidden_scans,
                "max_live_score": max_live_score,
                "gap_to_threshold": gap,
                "count_ge_0_30": int(row.get("count_ge_0_30") or 0),
                "count_ge_0_35": int(row.get("count_ge_0_35") or 0),
            })
        selected.sort(key=lambda row: (row.get("gap_to_threshold") or 999.0, -(row.get("selected_scans") or 0), row.get("symbol") or ""))
        return selected[:25]

    def _build_no_evidence_summary(self, *, requested_version: str, source_mode: str, fallback_used: bool, fallback_reason: str | None) -> dict:
        headline = "No evaluated run yet for threshold-boundary review"
        summary_text = fallback_reason or (
            f"Deployed version {requested_version} does not yet have an evaluated run, so threshold-boundary review should wait for resolved evidence."
        )
        memo_lines = [
            f"# Threshold-boundary review — {requested_version}",
            "",
            f"Headline: {headline}",
            "",
            "## What is working",
            "- The app is scanning, but there is not yet an evaluated run for threshold-boundary diagnosis.",
            "",
            "## What is failing",
            "- No threshold-boundary judgment is trustworthy yet because resolved evaluated rows are not available.",
            "",
            "## Dominant bottleneck",
            "- no_evaluated_run_yet: preserve comparability until an evaluated run exists.",
            "",
            "## Recommended next move",
            "- keep_live_path_unchanged_wait_for_evaluated_run: do not move the threshold boundary until at least one evaluated run exists.",
        ]
        summary = {
            "app_version": requested_version,
            "generated_at_utc": _utc_now_iso(),
            "source": source_mode,
            "evidence_source_app_version": requested_version,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "headline": headline,
            "summary": summary_text,
            "run_snapshot": {"available": False},
            "threshold_boundary": {
                "available": False,
                "gap_buckets": [],
                "scenario_thresholds": [],
                "top_false_suppressions": [],
                "repeated_near_threshold_symbols": [],
            },
            "verdict": {
                "threshold_boundary_problem_detected": False,
                "dominant_bottleneck": "no_evaluated_run_yet",
                "dominant_bottleneck_reason": "No evaluated run is available yet for threshold-boundary review.",
                "recommended_action": "keep_live_path_unchanged_wait_for_evaluated_run",
                "recommended_action_reason": "Do not change the threshold boundary before at least one run resolves.",
            },
            "decision_memo_markdown": "\n".join(memo_lines),
            "notes": [
                "This tranche exists to diagnose threshold-boundary overblocking without contaminating the live path.",
                "Rerun the review once an evaluated run exists.",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        return summary

    def build_summary(self, *, app_version: str | None = None) -> dict:
        version = str(app_version or APP_VERSION)
        try:
            source = self._resolve_latest_evaluated_run(version)
        except FileNotFoundError:
            return self._build_no_evidence_summary(
                requested_version=version,
                source_mode="current_version_no_evaluated_run",
                fallback_used=False,
                fallback_reason=f"No evaluated review runs are available yet for deployed version {version}.",
            )

        run = dict(source.get("run") or {})
        source_version = str(source.get("source_version") or version)
        source_mode = str(source.get("source_mode") or "current_version_latest_evaluated_run")
        fallback_used = bool(source.get("fallback_used"))
        fallback_reason = source.get("fallback_reason")
        get_run = getattr(self.review_packs, "get_run", None)
        if not callable(get_run):
            raise FileNotFoundError("review pack service does not support run-level threshold-boundary review")
        run_data = dict(get_run(str(run.get("run_id") or "")) or {})
        policy_audit = dict(run_data.get("policy_audit") or {})
        visible_rows = [row for row in list(run_data.get("visible_rows") or []) if bool(row.get("resolved"))]
        suppressed_rows = self._resolved_threshold_rows(list(run_data.get("suppressed_rows") or []))
        if int(policy_audit.get("evaluated_rows") or 0) <= 0 and not suppressed_rows and not visible_rows:
            return self._build_no_evidence_summary(
                requested_version=version,
                source_mode=source_mode,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )

        live_threshold = float(effective_live_raw_threshold(self.config))
        for row in suppressed_rows + visible_rows:
            if _f(row.get("live_threshold")) is not None:
                live_threshold = _f(row.get("live_threshold")) or 0.35
                break

        gap_buckets = self._gap_bucket_rows(suppressed_rows, live_threshold=live_threshold)
        scenario_thresholds = self._threshold_scenarios(suppressed_rows, live_threshold=live_threshold)
        top_false_suppressions = self._top_false_suppressions(suppressed_rows)
        summary_for_requested = self._current_summary(version)
        repeated_near_threshold_symbols = self._repeated_near_threshold_symbols(summary_for_requested, live_threshold=live_threshold)
        best_bucket = next((row for row in gap_buckets if int(row.get("count") or 0) > 0 and (row.get("quality_hit_rate") or 0.0) >= 0.25 and (row.get("avg_end_ret") or 0.0) > 0.0), None)
        best_scenario = next((row for row in scenario_thresholds if int(row.get("promoted_rows") or 0) > 0 and (row.get("quality_hit_rate") or 0.0) >= 0.25 and (row.get("avg_end_ret") or 0.0) > 0.0), None)
        false_quality = int(policy_audit.get("false_suppressions_quality_count") or 0)
        false_raw = int(policy_audit.get("false_suppressions_raw_count") or 0)
        threshold_boundary_problem = bool(false_quality > 0 and suppressed_rows and ((policy_audit.get("suppressed") or {}).get("avg_end_ret") or 0.0) > 0.0)
        if threshold_boundary_problem:
            headline = "Threshold-boundary overblocking is now the leading suspect"
            summary_text = (
                f"The latest evaluated run for app version {source_version} resolved with {len(visible_rows)} visible rows and {len(suppressed_rows)} threshold-suppressed resolved rows. "
                f"Suppressed rows posted {_pct(((policy_audit.get('suppressed') or {}).get('quality_hit_rate')))} quality-hit rate and {_pct(((policy_audit.get('suppressed') or {}).get('avg_end_ret')))} average end return, "
                f"so the shortlist boundary looks too harsh in at least some runs."
            )
        else:
            headline = "Threshold boundary not yet proven to be the bottleneck"
            summary_text = "The latest evaluated run does not yet show enough near-threshold strength to blame the shortlist boundary confidently."
        if fallback_used:
            summary_text += f" Using fallback evaluated evidence from app version {source_version} because deployed version {version} has not resolved an evaluated run yet."

        if threshold_boundary_problem:
            recommendation = "keep_live_threshold_0_35_collect_more_evaluated_runs_then_test_boundary_rule"
            recommendation_reason = (
                "The latest evaluated run shows too many good suppressed names, but one run is still not enough to move the live threshold immediately. "
                "Collect several more evaluated runs, then test a narrow threshold-boundary rule rather than broad model churn."
            )
            bottleneck = "threshold_boundary_overblocking"
            bottleneck_reason = "Threshold-suppressed rows are resolving well enough that the shortlist boundary is likely blocking too many good names."
        else:
            recommendation = "keep_live_path_unchanged_extend_sample"
            recommendation_reason = "The threshold boundary is not yet proven guilty enough to justify a live change."
            bottleneck = "not_proven_yet"
            bottleneck_reason = "The latest evaluated evidence is not yet strong enough to assign the main bottleneck to the threshold boundary."

        memo_lines = [
            f"# Threshold-boundary review — {version}",
            *( [f"Using evidence source app version: {source_version}", ""] if source_version != version else [] ),
            "",
            f"Headline: {headline}",
            "",
            "## What is working",
            f"- Latest evaluated run resolved with {int(policy_audit.get('evaluated_rows') or 0)} rows; the evaluator is working.",
            f"- Latest run visible/suppressed counts: {len(visible_rows)} visible vs {len(suppressed_rows)} threshold-suppressed resolved rows.",
            "",
            "## What is failing",
            f"- False suppressions (quality/raw): {false_quality}/{false_raw}.",
            f"- Best near-threshold bucket: <= {_pct(best_bucket.get('gap_max')) if best_bucket else '-'} gap, quality-hit {_pct(best_bucket.get('quality_hit_rate')) if best_bucket else '-'}, avg end ret {_pct(best_bucket.get('avg_end_ret')) if best_bucket else '-' }.",
            "",
            "## Dominant bottleneck",
            f"- {bottleneck}: {bottleneck_reason}",
            "",
            "## Recommended next move",
            f"- {recommendation}: {recommendation_reason}",
        ]

        summary = {
            "app_version": version,
            "generated_at_utc": _utc_now_iso(),
            "source": source_mode,
            "evidence_source_app_version": source_version,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "headline": headline,
            "summary": summary_text,
            "run_snapshot": {
                "available": True,
                "run_id": run.get("run_id"),
                "scan_finished_utc": run.get("scan_finished_utc"),
                "evaluation_complete": bool(run.get("evaluation_complete")),
                "market_regime_state": run.get("market_regime_state"),
                "market_regime_actionability": run.get("market_regime_actionability"),
                "evaluated_rows": int(policy_audit.get("evaluated_rows") or 0),
                "visible_rows": len(visible_rows),
                "threshold_suppressed_rows": len(suppressed_rows),
                "visible": dict(policy_audit.get("visible") or {}),
                "suppressed": dict(policy_audit.get("suppressed") or {}),
                "false_suppressions_quality_count": false_quality,
                "false_suppressions_raw_count": false_raw,
            },
            "threshold_boundary": {
                "available": True,
                "live_threshold": live_threshold,
                "gap_buckets": gap_buckets,
                "scenario_thresholds": scenario_thresholds,
                "top_false_suppressions": top_false_suppressions,
                "repeated_near_threshold_symbols": repeated_near_threshold_symbols,
            },
            "verdict": {
                "threshold_boundary_problem_detected": threshold_boundary_problem,
                "dominant_bottleneck": bottleneck,
                "dominant_bottleneck_reason": bottleneck_reason,
                "recommended_action": recommendation,
                "recommended_action_reason": recommendation_reason,
            },
            "decision_memo_markdown": "\n".join(memo_lines),
            "notes": [
                "This tranche diagnoses whether the shortlist boundary is blocking too many good names without changing the live path.",
                "Use scenario_thresholds as decision support only; do not lower the live threshold on one evaluated run alone.",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        return summary

    def build_pack(self, *, app_version: str | None = None) -> Path:
        summary = self.build_summary(app_version=app_version)
        boundary = dict(summary.get("threshold_boundary") or {})

        def write(zf: zipfile.ZipFile):
            zf.writestr("threshold_boundary_review_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr("threshold_boundary_review_manifest.json", json.dumps({
                "app_version": summary.get("app_version"),
                "source": summary.get("source"),
                "evidence_source_app_version": summary.get("evidence_source_app_version"),
                "fallback_used": summary.get("fallback_used"),
                "fallback_reason": summary.get("fallback_reason"),
                "generated_at_utc": summary.get("generated_at_utc"),
            }, indent=2, sort_keys=True))
            zf.writestr("decision_memo.md", str(summary.get("decision_memo_markdown") or ""))
            zf.writestr(
                "gap_buckets.csv",
                _csv_text(list(boundary.get("gap_buckets") or []), [
                    "gap_max", "live_threshold", "count", "quality_hit_rate", "raw_hit_rate", "avg_end_ret", "avg_mae", "avg_mfe", "avg_time_to_touch_minutes",
                ]),
            )
            zf.writestr(
                "scenario_thresholds.csv",
                _csv_text(list(boundary.get("scenario_thresholds") or []), [
                    "scenario_threshold", "current_live_threshold", "promoted_rows", "quality_hit_rate", "raw_hit_rate", "avg_end_ret", "avg_mae", "avg_mfe", "avg_time_to_touch_minutes",
                ]),
            )
            zf.writestr(
                "top_false_suppressions.csv",
                _csv_text(list(boundary.get("top_false_suppressions") or []), [
                    "symbol", "live_score", "pre_policy_score", "distance_to_live_threshold", "quality_touched", "raw_touched", "end_ret", "mae", "mfe", "time_to_touch_minutes", "liquidity_tier", "market_regime_state",
                ]),
            )
            zf.writestr(
                "repeated_near_threshold_symbols.csv",
                _csv_text(list(boundary.get("repeated_near_threshold_symbols") or []), [
                    "symbol", "liquidity_tier", "selected_scans", "visible_scans", "hidden_scans", "max_live_score", "gap_to_threshold", "count_ge_0_30", "count_ge_0_35",
                ]),
            )

        return _atomic_zip_write(self.pack_path, write)
