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


MIN_RESOLVED_TO_JUDGE = 4
MIN_HIDDEN_ROWS = 4
MIN_VISIBLE_ROWS = 3


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


class MisrankingDiagnosticService:
    def __init__(self, config: AppConfig, review_packs):
        self.config = config
        self.review_packs = review_packs
        self.summary_path = Path(config.model_dir) / "misranking_diagnostic_summary.json"
        self.pack_path = Path(config.model_dir) / "misranking_diagnostic_pack.zip"

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def _classify_symbol(self, row: dict, *, live_threshold: float) -> dict:
        symbol = str(row.get("symbol") or "")
        resolved_rows = int(row.get("resolved_rows") or 0)
        visible_rows = int(row.get("visible_rows") or 0)
        hidden_rows = int(row.get("non_visible_rows") or 0)
        quality_hit_rate = _f(row.get("quality_hit_rate"))
        visible_quality_hit_rate = _f(row.get("visible_quality_hit_rate"))
        hidden_quality_hit_rate = _f(row.get("non_visible_quality_hit_rate"))
        avg_end_ret = _f(row.get("avg_end_ret"))
        avg_mae = _f(row.get("avg_mae"))
        max_live_score = _f(row.get("max_live_score"))
        gap_to_threshold = None if max_live_score is None else round(live_threshold - max_live_score, 6)

        classification = "mixed_signal"
        reasoning = "Signal is mixed: not cleanly strong enough to call a hidden winner, and not weak enough to call a surfaced disappointment."

        hidden_q = hidden_quality_hit_rate if hidden_quality_hit_rate is not None else quality_hit_rate
        visible_q = visible_quality_hit_rate if visible_quality_hit_rate is not None else quality_hit_rate

        if resolved_rows < MIN_RESOLVED_TO_JUDGE:
            classification = "too_sparse_to_judge"
            reasoning = "Resolved evidence is still too thin to judge this symbol reliably."
        elif hidden_rows >= MIN_HIDDEN_ROWS and (hidden_q or 0.0) >= 0.40 and (avg_end_ret or 0.0) >= 0.0 and (
            visible_rows == 0 or ((visible_q is not None) and (hidden_q is not None) and hidden_q > visible_q + 0.15)
        ):
            classification = "hidden_winner"
            reasoning = "Resolved evidence says the symbol is strong while remaining mostly or entirely hidden."
        elif visible_rows >= MIN_VISIBLE_ROWS and (visible_q or 0.0) <= 0.10 and (avg_end_ret or 0.0) <= 0.0:
            classification = "surfaced_disappointment"
            reasoning = "The symbol has been surfaced repeatedly but resolved outcomes are weak."
        elif visible_rows >= MIN_VISIBLE_ROWS and (visible_q or 0.0) >= 0.40 and (avg_end_ret or 0.0) >= 0.0:
            classification = "correctly_surfaced_strong"
            reasoning = "The symbol has been surfaced repeatedly and the resolved outcomes support that choice."
        elif visible_rows == 0 and hidden_rows >= 6 and (hidden_q or 0.0) <= 0.10 and (avg_end_ret or 0.0) <= 0.0:
            classification = "correctly_hidden_weak"
            reasoning = "The symbol stayed hidden and the resolved evidence says that was appropriate."

        threshold_relevance = "not_applicable"
        if classification == "hidden_winner":
            if max_live_score is None:
                threshold_relevance = "unknown"
            elif max_live_score < live_threshold:
                if gap_to_threshold is not None and gap_to_threshold <= 0.03:
                    threshold_relevance = "near_threshold_hidden_winner"
                else:
                    threshold_relevance = "well_below_threshold_hidden_winner"
            else:
                threshold_relevance = "would_have_cleared_threshold"

        return {
            "symbol": symbol,
            "classification": classification,
            "reasoning": reasoning,
            "resolved_rows": resolved_rows,
            "visible_rows": visible_rows,
            "non_visible_rows": hidden_rows,
            "quality_hit_rate": quality_hit_rate,
            "visible_quality_hit_rate": visible_quality_hit_rate,
            "non_visible_quality_hit_rate": hidden_quality_hit_rate,
            "avg_end_ret": avg_end_ret,
            "avg_mae": avg_mae,
            "max_live_score": max_live_score,
            "live_threshold": live_threshold,
            "gap_to_threshold": gap_to_threshold,
            "threshold_relevance": threshold_relevance,
        }

    def _regime_rows(self, regime_rows: List[dict]) -> List[dict]:
        rows = []
        for row in list(regime_rows or []):
            state = str(row.get("market_regime_state") or "unknown")
            vis_q = _f(row.get("visible_quality_hit_rate"))
            non_q = _f(row.get("non_visible_quality_hit_rate"))
            vis_ret = _f(row.get("visible_avg_end_ret"))
            non_ret = _f(row.get("non_visible_avg_end_ret"))
            issue = bool(vis_q is not None and non_q is not None and vis_q < non_q)
            rows.append({
                "market_regime_state": state,
                "market_regime_actionability": row.get("market_regime_actionability"),
                "resolved_rows": int(row.get("resolved_rows") or 0),
                "visible_rows": int(row.get("visible_rows") or 0),
                "non_visible_rows": int(row.get("non_visible_rows") or 0),
                "visible_quality_hit_rate": vis_q,
                "non_visible_quality_hit_rate": non_q,
                "visible_avg_end_ret": vis_ret,
                "non_visible_avg_end_ret": non_ret,
                "diagnosis": "visible_underperforming_hidden" if issue else "visible_beating_hidden_or_inconclusive",
            })
        return rows

    def _dominant_bottleneck(self, *, hidden_winners: List[dict], surfaced_disappointments: List[dict], regime_rows: List[dict]) -> tuple[str, str]:
        green_row = next((row for row in regime_rows if str(row.get("market_regime_state") or "") == "green"), None)
        green_failure = bool(green_row and str(green_row.get("diagnosis") or "") == "visible_underperforming_hidden")
        near_threshold_hidden = sum(1 for row in hidden_winners if row.get("threshold_relevance") == "near_threshold_hidden_winner")
        if green_failure and len(surfaced_disappointments) >= max(2, len(hidden_winners)):
            return (
                "green_regime_shortlist_mismatch",
                "Green-regime surfaced rows are underperforming while there are enough surfaced disappointments to suggest a regime-specific shortlist-quality problem.",
            )
        if len(hidden_winners) >= len(surfaced_disappointments) + 1:
            if near_threshold_hidden >= max(1, len(hidden_winners) // 2):
                return (
                    "threshold_boundary_hidden_winners",
                    "Too many strong names are staying just below the live threshold, so the shortlist boundary itself looks like the dominant bottleneck.",
                )
            return (
                "under_scored_hidden_winners",
                "Strong resolved names are staying hidden even when there is enough evidence to call them winners, which points to under-ranking of real opportunities.",
            )
        if len(surfaced_disappointments) >= max(2, len(hidden_winners)):
            return (
                "over_ranked_surfaced_disappointments",
                "Too many surfaced names are resolving badly, which points to over-ranking weak opportunities rather than merely missing hidden ones.",
            )
        return (
            "mixed_shortlist_boundary",
            "The visible slice is directionally useful overall, but the remaining errors are mixed across hidden winners, surfaced disappointments, and regime-specific behavior.",
        )

    def _resolved_rows_count(self, summary: dict) -> int:
        evidence = dict((summary or {}).get("evidence") or {})
        return int(evidence.get("resolved_rows") or 0)

    def _resolve_evidence_source(self, requested_version: str) -> dict:
        requested_summary = self.review_packs.get_current_version_summary(app_version=requested_version)
        requested_resolved_rows = self._resolved_rows_count(requested_summary)
        if requested_resolved_rows > 0:
            return {
                "requested_version": requested_version,
                "requested_summary": requested_summary,
                "requested_resolved_rows": requested_resolved_rows,
                "source_summary": requested_summary,
                "source_version": requested_version,
                "source_mode": "current_version",
                "fallback_used": False,
                "fallback_reason": None,
            }

        recent_versions: list[str] = []
        get_runs = getattr(self.review_packs, "get_runs", None)
        if callable(get_runs):
            for run in list(get_runs(limit=250) or []):
                run_version = str(run.get("app_version") or "").strip()
                if not run_version or run_version == requested_version or run_version in recent_versions:
                    continue
                recent_versions.append(run_version)

        for candidate_version in recent_versions:
            try:
                candidate_summary = self.review_packs.get_current_version_summary(app_version=candidate_version)
            except FileNotFoundError:
                continue
            candidate_resolved_rows = self._resolved_rows_count(candidate_summary)
            if candidate_resolved_rows > 0:
                return {
                    "requested_version": requested_version,
                    "requested_summary": requested_summary,
                    "requested_resolved_rows": requested_resolved_rows,
                    "source_summary": candidate_summary,
                    "source_version": candidate_version,
                    "source_mode": "fallback_recent_mature_version",
                    "fallback_used": True,
                    "fallback_reason": (
                        f"No resolved rows are available yet for deployed version {requested_version}; "
                        f"using the most recent mature evidence from app version {candidate_version}."
                    ),
                }

        return {
            "requested_version": requested_version,
            "requested_summary": requested_summary,
            "requested_resolved_rows": requested_resolved_rows,
            "source_summary": requested_summary,
            "source_version": requested_version,
            "source_mode": "current_version_no_resolved_rows",
            "fallback_used": False,
            "fallback_reason": (
                f"No resolved rows are available yet for deployed version {requested_version}, and no earlier mature versioned evidence was found."
            ),
        }

    def _build_no_evidence_summary(self, *, requested_version: str, source_mode: str, fallback_used: bool, fallback_reason: str | None) -> dict:
        headline = "No resolved evidence yet for misranking diagnosis"
        summary_text = (
            fallback_reason
            or f"Deployed version {requested_version} does not yet have resolved rows, so shortlist-boundary diagnosis should wait for evaluated evidence."
        )
        memo_lines = [
            f"# Misranking diagnostic — {requested_version}",
            "",
            f"Headline: {headline}",
            "",
            "## What is working",
            "- The app is scanning, but the misranking endpoint does not yet have resolved evidence for this deployed version.",
            "",
            "## What is failing",
            "- No hidden-winner or surfaced-disappointment judgment is trustworthy yet because resolved rows are not available.",
            "",
            "## Dominant bottleneck",
            "- no_resolved_evidence_yet: the diagnostic cannot judge shortlist-boundary quality until evaluated rows exist.",
            "",
            "## Recommended next move",
            "- keep_live_path_unchanged_wait_for_resolved_rows: preserve comparability and rerun the diagnostic once evaluated evidence exists.",
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
            "evidence_snapshot": {
                "resolved_rows": 0,
                "visible_rows": 0,
                "non_visible_rows": 0,
                "visible_quality_hit_rate": None,
                "non_visible_quality_hit_rate": None,
                "visible_avg_end_ret": None,
                "non_visible_avg_end_ret": None,
                "visible_avg_mae": None,
                "non_visible_avg_mae": None,
            },
            "verdict": {
                "visible_beating_hidden": False,
                "green_regime_issue_detected": False,
                "hidden_winner_count": 0,
                "surfaced_disappointment_count": 0,
                "dominant_bottleneck": "no_resolved_evidence_yet",
                "dominant_bottleneck_reason": "No resolved rows are available yet for trustworthy shortlist-boundary diagnosis.",
                "recommended_action": "keep_live_path_unchanged_wait_for_resolved_rows",
                "recommended_action_reason": "Do not change the live path based on empty evidence; rerun the diagnostic after evaluated rows exist.",
            },
            "regime_diagnostics": {"available": False, "rows": []},
            "symbol_diagnostics": {
                "available": False,
                "rows": [],
                "hidden_winners": [],
                "surfaced_disappointments": [],
                "correctly_surfaced_strong": [],
                "correctly_hidden_weak": [],
                "too_sparse_to_judge": [],
            },
            "decision_memo_markdown": "\n".join(memo_lines),
            "notes": [
                "This diagnostic tranche should not manufacture shortlist judgments when resolved evidence is absent.",
                "Once evaluated rows exist, rerun the diagnostic to classify hidden winners and surfaced disappointments.",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        return summary

    def build_summary(self, *, app_version: str | None = None) -> dict:
        version = str(app_version or APP_VERSION)
        source = self._resolve_evidence_source(version)
        current = dict(source.get("source_summary") or {})
        source_version = str(source.get("source_version") or version)
        source_mode = str(source.get("source_mode") or "current_version")
        fallback_used = bool(source.get("fallback_used"))
        fallback_reason = source.get("fallback_reason")
        if self._resolved_rows_count(current) <= 0:
            return self._build_no_evidence_summary(
                requested_version=version,
                source_mode=source_mode,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
        evidence = dict(current.get("evidence") or {})
        symbol_rows = list(((current.get("symbol_repeatability") or {}).get("rows") or []))
        regime_rows = self._regime_rows(list(((current.get("regime_evidence") or {}).get("rows") or [])))
        live_threshold = float(effective_live_raw_threshold(self.config))
        threshold_bands = list(evidence.get("threshold_bands") or [])
        threshold_035 = next((row for row in threshold_bands if round(float(row.get("threshold") or 0.0), 2) == 0.35), None)
        if threshold_035 is not None:
            live_threshold = float(threshold_035.get("threshold") or 0.35)
        symbol_diagnostics = [self._classify_symbol(row, live_threshold=live_threshold) for row in symbol_rows]
        hidden_winners = [row for row in symbol_diagnostics if row.get("classification") == "hidden_winner"]
        surfaced_disappointments = [row for row in symbol_diagnostics if row.get("classification") == "surfaced_disappointment"]
        correctly_surfaced_strong = [row for row in symbol_diagnostics if row.get("classification") == "correctly_surfaced_strong"]
        correctly_hidden_weak = [row for row in symbol_diagnostics if row.get("classification") == "correctly_hidden_weak"]
        too_sparse = [row for row in symbol_diagnostics if row.get("classification") == "too_sparse_to_judge"]

        hidden_winners.sort(key=lambda row: ((row.get("non_visible_quality_hit_rate") or 0.0), (row.get("avg_end_ret") or -9.0), int(row.get("resolved_rows") or 0)), reverse=True)
        surfaced_disappointments.sort(key=lambda row: (-(row.get("avg_end_ret") or 9.0), -(row.get("visible_quality_hit_rate") or 9.0), -int(row.get("visible_rows") or 0)))
        correctly_surfaced_strong.sort(key=lambda row: ((row.get("visible_quality_hit_rate") or 0.0), (row.get("avg_end_ret") or -9.0), int(row.get("visible_rows") or 0)), reverse=True)
        correctly_hidden_weak.sort(key=lambda row: (-(row.get("avg_end_ret") or 9.0), -(row.get("non_visible_quality_hit_rate") or 9.0), -int(row.get("resolved_rows") or 0)))

        bottleneck, bottleneck_reason = self._dominant_bottleneck(hidden_winners=hidden_winners, surfaced_disappointments=surfaced_disappointments, regime_rows=regime_rows)
        green_row = next((row for row in regime_rows if str(row.get("market_regime_state") or "") == "green"), None)
        green_failure = bool(green_row and str(green_row.get("diagnosis") or "") == "visible_underperforming_hidden")
        visible_q = _f(evidence.get("visible_quality_hit_rate"))
        hidden_q = _f(evidence.get("non_visible_quality_hit_rate"))
        visible_ret = _f(evidence.get("visible_avg_end_ret"))
        hidden_ret = _f(evidence.get("non_visible_avg_end_ret"))
        visible_beating_hidden = bool(
            visible_q is not None and hidden_q is not None and visible_ret is not None and hidden_ret is not None and visible_q > hidden_q and visible_ret > hidden_ret
        )

        if visible_beating_hidden:
            headline = "Visible slice is working overall, but shortlist-boundary mistakes remain"
            top_summary = "Deployment-window evidence says visible rows are beating non-visible rows overall, so the next work should diagnose misranked symbols rather than reset the live path."
        else:
            headline = "Visible slice is not clearly beating hidden rows"
            top_summary = "The diagnostic focus should remain on shortlist quality because the visible slice is not yet decisively outperforming the hidden remainder."

        if fallback_used:
            top_summary += f" Using mature fallback evidence from app version {source_version} because deployed version {version} has not resolved enough rows yet."

        if bottleneck == "threshold_boundary_hidden_winners":
            recommendation = "keep_live_path_unchanged_collect_more_evidence_then_review_threshold_boundary"
            recommendation_reason = "Some strong hidden names appear to be staying just below the live threshold, but the sample is still better suited to diagnosis than immediate threshold movement."
        elif bottleneck == "under_scored_hidden_winners":
            recommendation = "keep_live_path_unchanged_then_review_score_semantics_for_hidden_winners"
            recommendation_reason = "The shortlist is directionally useful, but real winners are still being hidden often enough to justify a narrow score-semantics review after more evidence accrues."
        elif bottleneck == "over_ranked_surfaced_disappointments":
            recommendation = "keep_live_path_unchanged_then_review_surface_overranking"
            recommendation_reason = "The current shortlist still contains repeated disappointments, so the next live-path change should focus on over-ranked surfaced names rather than broad architecture."
        elif bottleneck == "green_regime_shortlist_mismatch":
            recommendation = "keep_live_path_unchanged_then_review_green_regime_shortlist_logic"
            recommendation_reason = "Green-regime visible underperformance is the clearest regime-specific risk and should be diagnosed before any global threshold change."
        else:
            recommendation = "keep_live_path_unchanged_extend_sample"
            recommendation_reason = "The visible slice is good overall and the remaining errors are mixed, so the right next move is still a narrow diagnostic tranche plus more resolved evidence."

        memo_lines = [
            f"# Misranking diagnostic — {version}",
            *( [f"Using evidence source app version: {source_version}", ""] if source_version != version else [] ),
            "",
            f"Headline: {headline}",
            "",
            "## What is working",
            f"- Visible vs hidden overall: visible quality-hit {_pct(visible_q)} vs hidden {_pct(hidden_q)}; visible avg end return {_pct(visible_ret)} vs hidden {_pct(hidden_ret)}.",
            "",
            "## What is failing",
            f"- Green-regime visible failure detected: {'yes' if green_failure else 'no'}.",
            f"- Hidden winners identified: {len(hidden_winners)}.",
            f"- Surfaced disappointments identified: {len(surfaced_disappointments)}.",
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
            "summary": top_summary,
            "evidence_snapshot": {
                "resolved_rows": int(evidence.get("resolved_rows") or 0),
                "visible_rows": int(evidence.get("visible_rows") or 0),
                "non_visible_rows": int(evidence.get("non_visible_rows") or 0),
                "visible_quality_hit_rate": visible_q,
                "non_visible_quality_hit_rate": hidden_q,
                "visible_avg_end_ret": visible_ret,
                "non_visible_avg_end_ret": hidden_ret,
                "visible_avg_mae": _f(evidence.get("visible_avg_mae")),
                "non_visible_avg_mae": _f(evidence.get("non_visible_avg_mae")),
            },
            "verdict": {
                "visible_beating_hidden": visible_beating_hidden,
                "green_regime_issue_detected": green_failure,
                "hidden_winner_count": len(hidden_winners),
                "surfaced_disappointment_count": len(surfaced_disappointments),
                "dominant_bottleneck": bottleneck,
                "dominant_bottleneck_reason": bottleneck_reason,
                "recommended_action": recommendation,
                "recommended_action_reason": recommendation_reason,
            },
            "regime_diagnostics": {
                "available": bool(regime_rows),
                "rows": regime_rows,
            },
            "symbol_diagnostics": {
                "available": bool(symbol_diagnostics),
                "rows": symbol_diagnostics,
                "hidden_winners": hidden_winners[:15],
                "surfaced_disappointments": surfaced_disappointments[:15],
                "correctly_surfaced_strong": correctly_surfaced_strong[:15],
                "correctly_hidden_weak": correctly_hidden_weak[:15],
                "too_sparse_to_judge": too_sparse[:15],
            },
            "decision_memo_markdown": "\n".join(memo_lines),
            "notes": [
                "This diagnostic tranche is meant to explain shortlist-boundary errors without contaminating the current live experiment.",
                "Use the dominant bottleneck field to decide whether the next code change should target threshold placement, score semantics, surfaced-name overranking, or regime-specific handling.",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        return summary

    def build_pack(self, *, app_version: str | None = None) -> Path:
        summary = self.build_summary(app_version=app_version)
        symbol_diag = dict(summary.get("symbol_diagnostics") or {})
        regime_diag = dict(summary.get("regime_diagnostics") or {})

        def write(zf: zipfile.ZipFile):
            zf.writestr("misranking_diagnostic_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr("misranking_diagnostic_manifest.json", json.dumps({
                "app_version": summary.get("app_version"),
                "source": summary.get("source"),
                "evidence_source_app_version": summary.get("evidence_source_app_version"),
                "fallback_used": summary.get("fallback_used"),
                "fallback_reason": summary.get("fallback_reason"),
                "generated_at_utc": summary.get("generated_at_utc"),
            }, indent=2, sort_keys=True))
            zf.writestr("decision_memo.md", str(summary.get("decision_memo_markdown") or ""))
            zf.writestr(
                "symbol_diagnostics.csv",
                _csv_text(list(symbol_diag.get("rows") or []), [
                    "symbol", "classification", "reasoning", "resolved_rows", "visible_rows", "non_visible_rows",
                    "quality_hit_rate", "visible_quality_hit_rate", "non_visible_quality_hit_rate", "avg_end_ret", "avg_mae",
                    "max_live_score", "live_threshold", "gap_to_threshold", "threshold_relevance",
                ]),
            )
            zf.writestr(
                "hidden_winners.csv",
                _csv_text(list(symbol_diag.get("hidden_winners") or []), [
                    "symbol", "resolved_rows", "visible_rows", "non_visible_rows", "non_visible_quality_hit_rate", "avg_end_ret",
                    "max_live_score", "live_threshold", "gap_to_threshold", "threshold_relevance", "reasoning",
                ]),
            )
            zf.writestr(
                "surfaced_disappointments.csv",
                _csv_text(list(symbol_diag.get("surfaced_disappointments") or []), [
                    "symbol", "resolved_rows", "visible_rows", "visible_quality_hit_rate", "avg_end_ret", "avg_mae", "max_live_score", "reasoning",
                ]),
            )
            zf.writestr(
                "correctly_surfaced_strong.csv",
                _csv_text(list(symbol_diag.get("correctly_surfaced_strong") or []), [
                    "symbol", "resolved_rows", "visible_rows", "visible_quality_hit_rate", "avg_end_ret", "avg_mae", "max_live_score", "reasoning",
                ]),
            )
            zf.writestr(
                "regime_diagnostics.csv",
                _csv_text(list(regime_diag.get("rows") or []), [
                    "market_regime_state", "market_regime_actionability", "resolved_rows", "visible_rows", "non_visible_rows",
                    "visible_quality_hit_rate", "non_visible_quality_hit_rate", "visible_avg_end_ret", "non_visible_avg_end_ret", "diagnosis",
                ]),
            )
        return _atomic_zip_write(self.pack_path, write)
