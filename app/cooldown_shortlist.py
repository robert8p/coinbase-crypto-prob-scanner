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


RECENT_COOLDOWN_RUN_LIMIT = 40
MIN_REPEAT_VISIBLE_RUNS = 2
NEAR_THRESHOLD_GAP_MAX = 0.03


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


def _non_visible_rows(rows: list[dict]) -> list[dict]:
    return [row for row in list(rows or []) if bool(row.get("resolved")) and str(row.get("row_type") or "") in ("suppressed", "informational", "overflow")]


def _visible_rows(rows: list[dict]) -> list[dict]:
    return [row for row in list(rows or []) if bool(row.get("resolved")) and str(row.get("row_type") or "") == "visible"]


def _is_cooldown_restricted_run(run: dict) -> bool:
    actionability = str(run.get("market_regime_actionability") or "").lower()
    cooldown_active = bool(run.get("cooldown_active"))
    return "cooldown" in actionability or cooldown_active


class CooldownShortlistReviewService:
    def __init__(self, config: AppConfig, review_packs):
        self.config = config
        self.review_packs = review_packs
        self.summary_path = Path(config.model_dir) / "cooldown_shortlist_review_summary.json"
        self.pack_path = Path(config.model_dir) / "cooldown_shortlist_review_pack.zip"

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def _metrics(self, rows: list[dict]) -> dict:
        count = len(rows)
        if count <= 0:
            return {
                "count": 0,
                "quality_hit_rate": None,
                "raw_hit_rate": None,
                "avg_end_ret": None,
                "avg_mae": None,
                "avg_mfe": None,
            }
        quality_hits = sum(1 for row in rows if bool(row.get("quality_touched")))
        raw_hits = sum(1 for row in rows if bool(row.get("raw_touched")))
        end_rets = [_f(row.get("end_ret")) for row in rows if _f(row.get("end_ret")) is not None]
        maes = [_f(row.get("mae")) for row in rows if _f(row.get("mae")) is not None]
        mfes = [_f(row.get("mfe")) for row in rows if _f(row.get("mfe")) is not None]
        return {
            "count": count,
            "quality_hit_rate": round(quality_hits / count, 4),
            "raw_hit_rate": round(raw_hits / count, 4),
            "avg_end_ret": round(sum(end_rets) / len(end_rets), 6) if end_rets else None,
            "avg_mae": round(sum(maes) / len(maes), 6) if maes else None,
            "avg_mfe": round(sum(mfes) / len(mfes), 6) if mfes else None,
        }

    def _resolve_latest_cooldown_run(self, requested_version: str) -> dict:
        get_runs_for_app_version = getattr(self.review_packs, "get_runs_for_app_version", None)
        if callable(get_runs_for_app_version):
            for run in list(get_runs_for_app_version(requested_version, limit=250) or []):
                if bool(run.get("evaluation_complete")) and _is_cooldown_restricted_run(run):
                    return {
                        "requested_version": requested_version,
                        "source_mode": "current_version_latest_cooldown_run",
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
                if bool(run.get("evaluation_complete")) and _is_cooldown_restricted_run(run):
                    return {
                        "requested_version": requested_version,
                        "source_mode": "fallback_recent_cooldown_run",
                        "source_version": run_version,
                        "run": run,
                        "fallback_used": True,
                        "fallback_reason": (
                            f"No cooldown-restricted evaluated runs are available yet for deployed version {requested_version}; "
                            f"using the most recent cooldown-restricted evaluated run from app version {run_version}."
                        ),
                    }
        raise FileNotFoundError(f"no cooldown-restricted evaluated review runs available for requested version {requested_version}")

    def _iter_recent_cooldown_runs(self, requested_version: str, source_version: str) -> list[dict]:
        get_runs = getattr(self.review_packs, "get_runs", None)
        if not callable(get_runs):
            return []
        selected = []
        for run in list(get_runs(limit=250) or []):
            run_version = str(run.get("app_version") or "")
            if run_version != source_version:
                continue
            if not bool(run.get("evaluation_complete")):
                continue
            if not _is_cooldown_restricted_run(run):
                continue
            selected.append(run)
            if len(selected) >= RECENT_COOLDOWN_RUN_LIMIT:
                break
        return selected

    def _near_threshold_hidden_rows(self, rows: list[dict], *, live_threshold: float) -> list[dict]:
        selected = []
        for row in rows:
            gap = _f(row.get("distance_to_live_threshold"))
            if gap is None or gap > NEAR_THRESHOLD_GAP_MAX:
                continue
            quality = bool(row.get("quality_touched"))
            raw = bool(row.get("raw_touched"))
            end_ret = _f(row.get("end_ret")) or 0.0
            if not (quality or raw or end_ret > 0.0):
                continue
            selected.append({
                "symbol": str(row.get("symbol") or ""),
                "live_score": _f(row.get("live_score")),
                "pre_policy_score": _f(row.get("pre_policy_score")),
                "distance_to_live_threshold": gap,
                "quality_touched": quality,
                "raw_touched": raw,
                "end_ret": _f(row.get("end_ret")),
                "mae": _f(row.get("mae")),
                "mfe": _f(row.get("mfe")),
                "liquidity_tier": row.get("liquidity_tier"),
                "market_regime_state": row.get("market_regime_state"),
            })
        selected.sort(key=lambda row: (not bool(row.get("quality_touched")), row.get("distance_to_live_threshold") if row.get("distance_to_live_threshold") is not None else 999.0, -(row.get("end_ret") or -9.0)))
        return selected[:25]

    def _recent_symbol_reviews(self, runs: list[dict]) -> tuple[list[dict], list[dict], dict]:
        get_run = getattr(self.review_packs, "get_run", None)
        if not callable(get_run):
            return [], [], {}
        agg: dict[str, dict[str, Any]] = {}
        visible_rows_all = []
        hidden_rows_all = []
        for run in runs:
            run_data = dict(get_run(str(run.get("run_id") or "")) or {})
            vis_rows = _visible_rows(list(run_data.get("visible_rows") or []))
            hid_rows = _non_visible_rows(list(run_data.get("suppressed_rows") or [])) + _non_visible_rows(list(run_data.get("overflow_rows") or []))
            visible_rows_all.extend(vis_rows)
            hidden_rows_all.extend(hid_rows)
            seen_visible_symbols = set()
            for row in vis_rows:
                symbol = str(row.get("symbol") or "")
                item = agg.setdefault(symbol, {
                    "symbol": symbol,
                    "visible_runs": 0,
                    "visible_rows": 0,
                    "quality_hits": 0,
                    "raw_hits": 0,
                    "end_rets": [],
                    "maes": [],
                    "max_live_score": None,
                })
                item["visible_rows"] += 1
                if symbol not in seen_visible_symbols:
                    item["visible_runs"] += 1
                    seen_visible_symbols.add(symbol)
                if bool(row.get("quality_touched")):
                    item["quality_hits"] += 1
                if bool(row.get("raw_touched")):
                    item["raw_hits"] += 1
                if _f(row.get("end_ret")) is not None:
                    item["end_rets"].append(_f(row.get("end_ret")))
                if _f(row.get("mae")) is not None:
                    item["maes"].append(_f(row.get("mae")))
                score = _f(row.get("live_score"))
                if score is not None:
                    item["max_live_score"] = max(score, item["max_live_score"] if item["max_live_score"] is not None else score)
        weak, strong = [], []
        for symbol, item in agg.items():
            rows = int(item["visible_rows"] or 0)
            runs_ct = int(item["visible_runs"] or 0)
            qhr = round(item["quality_hits"] / rows, 4) if rows else None
            rawhr = round(item["raw_hits"] / rows, 4) if rows else None
            avg_end = round(sum(item["end_rets"]) / len(item["end_rets"]), 6) if item["end_rets"] else None
            avg_mae = round(sum(item["maes"]) / len(item["maes"]), 6) if item["maes"] else None
            row = {
                "symbol": symbol,
                "visible_runs": runs_ct,
                "visible_rows": rows,
                "quality_hit_rate": qhr,
                "raw_hit_rate": rawhr,
                "avg_end_ret": avg_end,
                "avg_mae": avg_mae,
                "max_live_score": item["max_live_score"],
            }
            if runs_ct >= MIN_REPEAT_VISIBLE_RUNS and (qhr or 0.0) <= 0.15 and (avg_end or 0.0) <= 0.0:
                weak.append(row)
            if runs_ct >= MIN_REPEAT_VISIBLE_RUNS and (qhr or 0.0) >= 0.40 and (avg_end or 0.0) > 0.0:
                strong.append(row)
        weak.sort(key=lambda row: (row.get("quality_hit_rate") or 1.0, row.get("avg_end_ret") or 0.0, -int(row.get("visible_runs") or 0)))
        strong.sort(key=lambda row: (-(row.get("quality_hit_rate") or 0.0), -(row.get("avg_end_ret") or 0.0), -int(row.get("visible_runs") or 0)))
        rolling = {
            "run_count": len(runs),
            "visible": self._metrics(visible_rows_all),
            "hidden": self._metrics(hidden_rows_all),
        }
        return weak[:20], strong[:20], rolling

    def _build_no_evidence_summary(self, *, requested_version: str, source_mode: str, fallback_used: bool, fallback_reason: str | None) -> dict:
        headline = "No cooldown-restricted evaluated evidence yet"
        summary_text = "Wait for a cooldown-restricted evaluated run before judging whether amber/cooldown shortlist quality needs a stricter path."
        memo_lines = [
            f"# Cooldown-restricted shortlist review — {requested_version}",
            "",
            f"Headline: {headline}",
            "",
            "## What is working",
            "- The live path remains unchanged; this tranche should not contaminate the experiment prematurely.",
            "",
            "## What is failing",
            "- No cooldown-restricted evaluated run exists yet, so there is nothing trustworthy to diagnose.",
            "",
            "## Dominant bottleneck",
            "- no_cooldown_restricted_evidence_yet: more resolved evidence is required before making a cooldown-specific judgment.",
            "",
            "## Recommended next move",
            "- keep_live_path_unchanged_wait_for_cooldown_evidence: do not change threshold or cooldown behavior until evaluated evidence exists.",
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
            "cooldown_shortlist": {
                "available": False,
                "near_threshold_hidden_rows": [],
                "repeated_surfaced_weak_symbols": [],
                "repeated_surfaced_strong_symbols": [],
                "rolling_recent_cooldown": {"run_count": 0, "visible": self._metrics([]), "hidden": self._metrics([])},
            },
            "verdict": {
                "cooldown_shortlist_problem_detected": False,
                "dominant_bottleneck": "no_cooldown_restricted_evidence_yet",
                "dominant_bottleneck_reason": "No cooldown-restricted evaluated run is available yet.",
                "recommended_action": "keep_live_path_unchanged_wait_for_cooldown_evidence",
                "recommended_action_reason": "Do not change cooldown shortlist behavior before at least one cooldown-restricted run resolves.",
            },
            "decision_memo_markdown": "\n".join(memo_lines),
            "notes": [
                "This tranche exists to diagnose cooldown-restricted shortlist quality without changing the live path.",
                "Run the review again once at least one cooldown-restricted evaluated pack exists.",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        return summary

    def build_summary(self, *, app_version: str | None = None) -> dict:
        version = str(app_version or APP_VERSION)
        try:
            source = self._resolve_latest_cooldown_run(version)
        except FileNotFoundError:
            return self._build_no_evidence_summary(
                requested_version=version,
                source_mode="current_version_no_cooldown_evaluated_run",
                fallback_used=False,
                fallback_reason=f"No cooldown-restricted evaluated review runs are available yet for deployed version {version}.",
            )

        run = dict(source.get("run") or {})
        source_version = str(source.get("source_version") or version)
        source_mode = str(source.get("source_mode") or "current_version_latest_cooldown_run")
        fallback_used = bool(source.get("fallback_used"))
        fallback_reason = source.get("fallback_reason")

        get_run = getattr(self.review_packs, "get_run", None)
        if not callable(get_run):
            raise FileNotFoundError("review pack service does not support cooldown shortlist review")
        run_data = dict(get_run(str(run.get("run_id") or "")) or {})
        visible_rows = _visible_rows(list(run_data.get("visible_rows") or []))
        hidden_rows = _non_visible_rows(list(run_data.get("suppressed_rows") or [])) + _non_visible_rows(list(run_data.get("overflow_rows") or []))
        live_threshold = float(effective_live_raw_threshold(self.config))
        all_rows = visible_rows + hidden_rows
        for row in all_rows:
            if _f(row.get("live_threshold")) is not None:
                live_threshold = _f(row.get("live_threshold")) or 0.35
                break

        if not visible_rows and not hidden_rows:
            return self._build_no_evidence_summary(
                requested_version=version,
                source_mode=source_mode,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )

        visible_metrics = self._metrics(visible_rows)
        hidden_metrics = self._metrics(hidden_rows)
        near_threshold_hidden = self._near_threshold_hidden_rows(hidden_rows, live_threshold=live_threshold)

        recent_runs = self._iter_recent_cooldown_runs(version, source_version)
        repeated_weak, repeated_strong, rolling_recent = self._recent_symbol_reviews(recent_runs)

        visible_q = visible_metrics.get("quality_hit_rate")
        hidden_q = hidden_metrics.get("quality_hit_rate")
        visible_ret = visible_metrics.get("avg_end_ret")
        hidden_ret = hidden_metrics.get("avg_end_ret")
        visible_underperforming = bool(
            visible_metrics.get("count", 0) > 0 and hidden_metrics.get("count", 0) > 0 and
            (visible_q is not None and hidden_q is not None and visible_q < hidden_q) and
            (visible_ret is not None and hidden_ret is not None and visible_ret < hidden_ret)
        )

        headline = "Cooldown-restricted shortlist quality needs review" if visible_underperforming else "Cooldown-restricted shortlist quality not yet proven broken"
        summary_text = (
            f"The latest cooldown-restricted evaluated run for app version {source_version} resolved with "
            f"{visible_metrics.get('count', 0)} visible rows and {hidden_metrics.get('count', 0)} non-visible rows. "
            f"Visible quality-hit rate was {_pct(visible_q)} versus {_pct(hidden_q)} for non-visible rows, with "
            f"visible average end return {_pct(visible_ret)} versus {_pct(hidden_ret)}."
        )
        if fallback_used:
            summary_text += f" Using fallback evaluated evidence from app version {source_version} because deployed version {version} has not yet produced a cooldown-restricted evaluated run."
        if visible_underperforming:
            summary_text += " In this evidence slice, surfaced cooldown-restricted names are underperforming the hidden remainder, so the next bottleneck looks like cooldown shortlist quality rather than simple threshold looseness."

        if visible_underperforming:
            bottleneck = "cooldown_restricted_visible_underperforming"
            bottleneck_reason = "In cooldown-restricted runs, surfaced rows are resolving worse than the hidden remainder."
            recommendation = "keep_live_path_unchanged_collect_more_cooldown_runs_then_review_cooldown_visibility_policy"
            recommendation_reason = "Do not loosen the threshold blindly. First confirm whether cooldown-restricted surfaced rows are consistently weak across several evaluated runs."
        elif repeated_weak:
            bottleneck = "cooldown_restricted_repeat_surface_weak_names"
            bottleneck_reason = "Cooldown-restricted surfaced rows are not broadly proven worse in this run, but some symbols are recurring weak surfaces in recent cooldown evidence."
            recommendation = "keep_live_path_unchanged_review_repeat_cooldown_symbols"
            recommendation_reason = "Use more cooldown-restricted evaluated evidence to decide whether symbol-level handling or a stricter cooldown shortlist is needed."
        else:
            bottleneck = "not_proven_yet"
            bottleneck_reason = "Cooldown-restricted shortlist quality is not yet proven to be the main bottleneck."
            recommendation = "keep_live_path_unchanged_extend_sample"
            recommendation_reason = "The evidence is still too thin for a cooldown-specific live change."

        memo_lines = [
            f"# Cooldown-restricted shortlist review — {version}",
            *( [f"Using evidence source app version: {source_version}", ""] if source_version != version else [] ),
            "",
            f"Headline: {headline}",
            "",
            "## What is working",
            f"- Latest cooldown-restricted evaluated run resolved with {visible_metrics.get('count', 0) + hidden_metrics.get('count', 0)} evidence rows; evaluator and review persistence are working.",
            f"- Latest run visible/non-visible counts: {visible_metrics.get('count', 0)} visible vs {hidden_metrics.get('count', 0)} non-visible.",
            "",
            "## What is failing",
            f"- Visible cooldown quality-hit vs hidden: {_pct(visible_q)} vs {_pct(hidden_q)}.",
            f"- Visible cooldown avg end ret vs hidden: {_pct(visible_ret)} vs {_pct(hidden_ret)}.",
            f"- Near-threshold better hidden rows in latest run: {len(near_threshold_hidden)}.",
            f"- Repeated surfaced weak cooldown symbols in recent evidence: {len(repeated_weak)}.",
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
                "cooldown_active": bool(run.get("cooldown_active")),
                "live_threshold": live_threshold,
                "visible": visible_metrics,
                "non_visible": hidden_metrics,
            },
            "cooldown_shortlist": {
                "available": True,
                "near_threshold_hidden_rows": near_threshold_hidden,
                "repeated_surfaced_weak_symbols": repeated_weak,
                "repeated_surfaced_strong_symbols": repeated_strong,
                "rolling_recent_cooldown": rolling_recent,
            },
            "verdict": {
                "cooldown_shortlist_problem_detected": visible_underperforming or bool(repeated_weak),
                "dominant_bottleneck": bottleneck,
                "dominant_bottleneck_reason": bottleneck_reason,
                "recommended_action": recommendation,
                "recommended_action_reason": recommendation_reason,
            },
            "decision_memo_markdown": "\n".join(memo_lines),
            "notes": [
                "This tranche exists to diagnose cooldown-restricted shortlist quality without changing the live path.",
                "Do not loosen the threshold based on one cooldown-restricted run when the visible shortlist itself may be the problem.",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        return summary

    def build_pack(self, *, app_version: str | None = None) -> Path:
        summary = self.build_summary(app_version=app_version)
        review = dict(summary.get("cooldown_shortlist") or {})

        def write(zf: zipfile.ZipFile):
            zf.writestr("cooldown_shortlist_review_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr("cooldown_shortlist_review_manifest.json", json.dumps({
                "app_version": summary.get("app_version"),
                "source": summary.get("source"),
                "evidence_source_app_version": summary.get("evidence_source_app_version"),
                "fallback_used": summary.get("fallback_used"),
                "fallback_reason": summary.get("fallback_reason"),
                "generated_at_utc": summary.get("generated_at_utc"),
            }, indent=2, sort_keys=True))
            zf.writestr("decision_memo.md", str(summary.get("decision_memo_markdown") or ""))
            zf.writestr(
                "near_threshold_hidden_rows.csv",
                _csv_text(list(review.get("near_threshold_hidden_rows") or []), [
                    "symbol", "live_score", "pre_policy_score", "distance_to_live_threshold", "quality_touched", "raw_touched", "end_ret", "mae", "mfe", "liquidity_tier", "market_regime_state",
                ]),
            )
            zf.writestr(
                "repeated_surfaced_weak_symbols.csv",
                _csv_text(list(review.get("repeated_surfaced_weak_symbols") or []), [
                    "symbol", "visible_runs", "visible_rows", "quality_hit_rate", "raw_hit_rate", "avg_end_ret", "avg_mae", "max_live_score",
                ]),
            )
            zf.writestr(
                "repeated_surfaced_strong_symbols.csv",
                _csv_text(list(review.get("repeated_surfaced_strong_symbols") or []), [
                    "symbol", "visible_runs", "visible_rows", "quality_hit_rate", "raw_hit_rate", "avg_end_ret", "avg_mae", "max_live_score",
                ]),
            )
            zf.writestr(
                "rolling_recent_cooldown.json",
                json.dumps(review.get("rolling_recent_cooldown") or {}, indent=2, sort_keys=True),
            )

        return _atomic_zip_write(self.pack_path, write)
