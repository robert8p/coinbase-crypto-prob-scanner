from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .utility_shortlist import optimize_visible_shortlist, utility_config_with_runtime_override
from .version import APP_VERSION


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _slug(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return s or "challenger"


class ShadowSelectionComparisonService:
    """Record live-vs-shadow shortlist differences for a bounded concurrent challenger set.

    Live remains legacy. Up to N near-miss challengers from the latest policy search are
    evaluated in shadow on each completed scan so we can compare them side-by-side against
    the same incumbent without risking live quality.
    """

    def __init__(self, config: AppConfig, review_packs: Any | None = None):
        self.config = config
        self.review_packs = review_packs
        self.max_active_challengers = 3
        self.root_dir = ensure_dir(Path(config.model_dir) / "shadow_selection_comparison")
        self.summary_path = self.root_dir / "latest_shadow_selection_comparison_summary.json"
        self.pack_path = self.root_dir / "latest_shadow_selection_comparison_pack.zip"
        self.history_path = self.root_dir / "comparison_history.jsonl"
        self.policy_search_summary_path = Path(config.model_dir) / "utility_policy_search_lab" / "latest_utility_policy_search_lab_summary.json"
        self.outcome_review_summary_path = Path(config.model_dir) / "shadow_selection_outcome_review" / "latest_shadow_selection_outcome_review_summary.json"

    def latest_summary(self) -> dict:
        summary = read_json(self.summary_path, {})
        if not summary:
            return {
                "available": False,
                "app_version": APP_VERSION,
                "headline": "No shadow comparison summary available yet",
                "summary": "The first comparison will appear after a completed scan when legacy is live and challenger policies are available.",
                "status": "waiting",
                "pack_available": False,
            }
        summary.setdefault("available", True)
        summary.setdefault("app_version", APP_VERSION)
        status = str(summary.get("status") or "recorded")
        summary["pack_available"] = self.pack_path.exists() if status == "recorded" else False
        summary.setdefault("status", status)
        return summary

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _resolve_live_selection_state(self, status: dict[str, Any] | None) -> dict[str, str]:
        payload = dict(status or {})
        configured = str(
            payload.get("configured_live_selection_mode")
            or getattr(self.config, "live_selection_mode", "legacy")
            or "legacy"
        ).strip().lower() or "legacy"
        effective_mode = str(
            payload.get("effective_live_selection_mode")
            or payload.get("effective_live_selection_engine")
            or payload.get("selection_engine")
            or configured
        ).strip().lower() or configured
        effective_engine = str(
            payload.get("effective_live_selection_engine")
            or payload.get("selection_engine")
            or effective_mode
        ).strip().lower() or effective_mode
        return {
            "configured_live_selection_mode": configured,
            "effective_live_selection_mode": effective_mode,
            "effective_live_selection_engine": effective_engine,
        }

    def _retired_challengers(self) -> set[str]:
        summary = read_json(self.outcome_review_summary_path, {})
        retired = set()
        for item in summary.get("retired_challengers") or []:
            name = str(item.get("policy_name") or item.get("engine") or item or "").strip()
            if name:
                retired.add(name)
        return retired

    def _select_shadow_policies(self, policy_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        ranked = [dict(item) for item in (policy_summary.get("ranked_policies") or []) if isinstance(item, dict)]
        winner = dict(policy_summary.get("winner") or {})
        retired = self._retired_challengers()
        supported = [
            dict(item)
            for item in ranked
            if str(item.get("support_level") or "") == "supported_offline"
            and isinstance(item.get("override"), dict)
            and bool(item.get("override"))
        ]
        supported = [
            item for item in supported
            if str(item.get("policy_name") or item.get("engine_label") or "").strip() not in retired
        ]
        winner_name = str(winner.get("policy_name") or winner.get("engine_label") or "").strip()
        winner_supported = (
            winner_name
            and winner_name not in retired
            and str(winner.get("support_level") or "") == "supported_offline"
            and isinstance(winner.get("override"), dict)
            and bool(winner.get("override"))
        )
        if winner_supported:
            return [winner], "supported_winner"
        if supported:
            supported.sort(
                key=lambda item: (
                    _safe_float(item.get("scan_shortlist_utility_score")) if _safe_float(item.get("scan_shortlist_utility_score")) is not None else -999.0,
                    _safe_float(item.get("scan_shortlist_pairwise_win_rate")) if _safe_float(item.get("scan_shortlist_pairwise_win_rate")) is not None else -999.0,
                    _safe_float(item.get("scan_shortlist_mean_gap")) if _safe_float(item.get("scan_shortlist_mean_gap")) is not None else -999.0,
                ),
                reverse=True,
            )
            return [supported[0]], "best_supported_offline"
        return [], "no_supported_offline_candidate"

    def _offline_gate_state(self, policy_summary: dict[str, Any], selected_policies: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
        verdict = str(policy_summary.get("verdict") or "")
        ranked = [dict(item) for item in (policy_summary.get("ranked_policies") or []) if isinstance(item, dict)]
        supported = [item for item in ranked if str(item.get("support_level") or "") == "supported_offline"]
        if verdict == "supported_policy_found" and supported:
            return True, {
                "offline_gate_status": "passed",
                "supported_policy_count": len(supported),
                "supported_policy_names": [str(item.get("policy_name") or item.get("engine_label") or "") for item in supported],
            }
        selected_names = [str(item.get("policy_name") or item.get("engine_label") or "") for item in selected_policies]
        return False, {
            "offline_gate_status": "blocked",
            "policy_search_verdict": verdict or "unknown",
            "supported_policy_count": 0,
            "supported_policy_names": [],
            "selected_policy_names": [name for name in selected_names if name],
        }

    def _match_run(self, *, app_version: str, generated_at_utc: str) -> dict | None:
        if self.review_packs is None:
            return None
        generated = _parse_utc(generated_at_utc)
        if generated is None:
            return None
        try:
            runs = self.review_packs.get_runs_for_app_version(str(app_version or APP_VERSION), limit=30)
        except Exception:
            return None
        best = None
        best_delta = None
        for run in runs:
            finished = _parse_utc(run.get("scan_finished_utc"))
            if finished is None:
                continue
            delta = abs((generated - finished).total_seconds())
            if delta > 900:
                continue
            if best is None or delta < (best_delta or 10**18):
                best = run
                best_delta = delta
        return best

    def _candidate_pool(self, live_rows: list[dict], trimmed_rows: list[dict]) -> list[dict]:
        dedup: dict[str, dict] = {}
        for row in list(live_rows or []) + list(trimmed_rows or []):
            symbol = str(row.get("symbol") or "")
            if not symbol:
                continue
            existing = dedup.get(symbol)
            current_rank = int(row.get("candidate_rank_all") or row.get("pre_policy_rank") or row.get("score_rank") or 999999)
            existing_rank = int((existing or {}).get("candidate_rank_all") or (existing or {}).get("pre_policy_rank") or (existing or {}).get("score_rank") or 999999)
            if existing is None or current_rank < existing_rank:
                dedup[symbol] = dict(row)
        rows = list(dedup.values())
        rows.sort(
            key=lambda r: (
                int(r.get("candidate_rank_all") or r.get("pre_policy_rank") or r.get("score_rank") or 999999),
                -float(r.get("live_score", r.get("prob_2") or 0.0) or 0.0),
                str(r.get("symbol") or ""),
            )
        )
        return rows

    def _avg(self, rows: list[dict], key: str) -> float | None:
        values = [float(r.get(key) or 0.0) for r in rows if r.get(key) is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    def _build_record_summary(
        self,
        *,
        generated_at: str,
        status: dict[str, Any] | None,
        incumbent_rows: list[dict],
        challenger_rows: list[dict],
        candidate_pool: list[dict],
        current_policy_search_verdict: str,
        policy: dict[str, Any],
        trigger_source: str,
        source_run_id: str | None,
        source_scan_finished_utc: str | None,
        selected_policy_source: str,
    ) -> dict[str, Any]:
        incumbent_symbols = [str(r.get("symbol") or "") for r in incumbent_rows]
        challenger_symbols = [str(r.get("symbol") or "") for r in challenger_rows]
        incumbent_set = {s for s in incumbent_symbols if s}
        challenger_set = {s for s in challenger_symbols if s}
        overlap = sorted(incumbent_set & challenger_set)
        incumbent_only = sorted(incumbent_set - challenger_set)
        challenger_only = sorted(challenger_set - incumbent_set)
        density_delta = round(len(challenger_rows) - len(incumbent_rows), 6)
        if challenger_only and incumbent_only:
            headline = "Shadow challenger diverged from the live legacy shortlist"
            detail = "The challenger selected a materially different shortlist than the live legacy incumbent on this scan."
        elif challenger_only:
            headline = "Shadow challenger was broader than the live legacy shortlist"
            detail = "The challenger added names beyond the live legacy incumbent on this scan."
        elif incumbent_only:
            headline = "Shadow challenger was stricter than the live legacy shortlist"
            detail = "The challenger selected fewer names than the live legacy incumbent on this scan."
        else:
            headline = "Legacy and shadow challenger matched on this scan"
            detail = "The challenger produced the same shortlist as the live legacy incumbent on this scan."
        policy_name = str(policy.get("policy_name") or policy.get("engine_label") or "-")
        policy_id = str(policy.get("engine_label") or policy_name)
        return {
            "available": True,
            "generated_at_utc": generated_at,
            "app_version": APP_VERSION,
            "status": "recorded",
            "headline": headline,
            "summary": detail,
            "trigger_source": trigger_source,
            "selected_policy_source": selected_policy_source,
            "source_run_id": source_run_id,
            "source_scan_finished_utc": source_scan_finished_utc,
            **self._resolve_live_selection_state(status),
            "candidate_pool_count": len(candidate_pool),
            "comparison_scope": "same_scan_live_shadow",
            "current_policy_search_verdict": current_policy_search_verdict,
            "promotion_ready": False,
            "promotion_ready_reason": "Shadow comparison is evidence-only. Live promotion remains disallowed until a challenger beats legacy offline and then survives a controlled proof window.",
            "challenger_policy": {
                "policy_name": policy_name,
                "policy_id": policy_id,
                "support_level": str(policy.get("support_level") or ""),
                "winner_override": dict(policy.get("override") or {}),
            },
            "incumbent": {
                "engine": "legacy",
                "visible_count": len(incumbent_rows),
                "symbols": incumbent_symbols,
                "top_symbols": incumbent_symbols[:5],
                "avg_live_score": self._avg(incumbent_rows, "live_score"),
            },
            "challenger": {
                "engine": policy_id,
                "visible_count": len(challenger_rows),
                "symbols": challenger_symbols,
                "top_symbols": challenger_symbols[:5],
                "avg_live_score": self._avg(challenger_rows, "live_score"),
                "avg_utility_decision_score": self._avg(challenger_rows, "utility_decision_score"),
            },
            "comparison": {
                "overlap_count": len(overlap),
                "overlap_symbols": overlap,
                "incumbent_only_count": len(incumbent_only),
                "incumbent_only_symbols": incumbent_only[:10],
                "challenger_only_count": len(challenger_only),
                "challenger_only_symbols": challenger_only[:10],
                "overlap_ratio_vs_incumbent": round(len(overlap) / len(incumbent_rows), 6) if incumbent_rows else None,
                "overlap_ratio_vs_challenger": round(len(overlap) / len(challenger_rows), 6) if challenger_rows else None,
                "density_delta": density_delta,
            },
        }

    def _append_history(self, record: dict[str, Any]) -> None:
        ensure_dir(self.root_dir)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _read_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        out: list[dict] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def _trailing_summary(self, *, window_hours: int = 24) -> dict[str, Any]:
        items = self._read_history()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(window_hours or 24))
        recent: list[dict] = []
        for item in items:
            generated = _parse_utc(item.get("generated_at_utc"))
            if generated is not None and generated >= cutoff:
                recent.append(item)
        if not recent:
            return {
                "window_hours": int(window_hours or 24),
                "comparisons": 0,
                "avg_overlap_ratio_vs_incumbent": None,
                "avg_overlap_ratio_vs_challenger": None,
                "avg_density_delta": None,
                "challenger_counts": [],
            }
        overlap_inc = [_safe_float((item.get("comparison") or {}).get("overlap_ratio_vs_incumbent")) for item in recent]
        overlap_inc = [v for v in overlap_inc if v is not None]
        overlap_ch = [_safe_float((item.get("comparison") or {}).get("overlap_ratio_vs_challenger")) for item in recent]
        overlap_ch = [v for v in overlap_ch if v is not None]
        density = [_safe_float((item.get("comparison") or {}).get("density_delta")) for item in recent]
        density = [v for v in density if v is not None]
        counts: dict[str, int] = {}
        for item in recent:
            name = str((item.get("challenger_policy") or {}).get("policy_name") or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
        challenger_counts = [{"policy_name": k, "comparisons": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        return {
            "window_hours": int(window_hours or 24),
            "comparisons": len(recent),
            "avg_overlap_ratio_vs_incumbent": round(sum(overlap_inc) / len(overlap_inc), 6) if overlap_inc else None,
            "avg_overlap_ratio_vs_challenger": round(sum(overlap_ch) / len(overlap_ch), 6) if overlap_ch else None,
            "avg_density_delta": round(sum(density) / len(density), 6) if density else None,
            "challenger_counts": challenger_counts,
        }

    def _build_group_summary(
        self,
        *,
        records: list[dict[str, Any]],
        live_state: dict[str, str],
        current_policy_search_verdict: str,
        trigger_source: str,
        candidate_pool_count: int,
        selected_policy_source: str,
    ) -> dict[str, Any]:
        if not records:
            return {
                "available": True,
                "app_version": APP_VERSION,
                "generated_at_utc": _utc_now_iso(),
                "headline": "No supported shadow candidate was recorded on this scan",
                "summary": "No supported offline winner with a usable override payload was available to run in controlled shadow on this scan.",
                "status": "skipped",
                "skip_reason": "no_supported_shadow_candidate",
                **live_state,
                "current_policy_search_verdict": current_policy_search_verdict,
                "candidate_pool_count": candidate_pool_count,
                "trigger_source": trigger_source,
                "selected_policy_source": selected_policy_source,
                "live_path_unchanged": True,
                "live_path_statement": "Legacy remains the effective live selection engine; shadow work is evidence-only.",
                "shadow_candidate_mode": "single_supported_offline_winner",
                "primary_shadow_candidate": None,
                "pack_available": False,
            }
        primary = records[0]
        generated_at = primary.get("generated_at_utc") or _utc_now_iso()
        count = len(records)
        if count == 1:
            headline = primary.get("headline") or "Recorded the supported shadow candidate against the live legacy shortlist"
            detail = primary.get("summary") or "The supported offline winner was scored in controlled shadow on the same scan while the live legacy path remained unchanged."
        else:
            headline = f"Recorded {count} concurrent shadow challengers against the live legacy shortlist"
            detail = "Multiple challengers were scored on the same live scan so they can be compared side-by-side against legacy."
        challenger_records = []
        for rec in records:
            challenger_records.append({
                "policy_name": ((rec.get("challenger_policy") or {}).get("policy_name") or "-"),
                "policy_id": ((rec.get("challenger_policy") or {}).get("policy_id") or "-"),
                "visible_count": ((rec.get("challenger") or {}).get("visible_count") or 0),
                "top_symbols": ((rec.get("challenger") or {}).get("top_symbols") or []),
                "overlap_count": ((rec.get("comparison") or {}).get("overlap_count") or 0),
                "incumbent_only_count": ((rec.get("comparison") or {}).get("incumbent_only_count") or 0),
                "challenger_only_count": ((rec.get("comparison") or {}).get("challenger_only_count") or 0),
                "density_delta": ((rec.get("comparison") or {}).get("density_delta")),
            })
        primary_policy = dict(primary.get("challenger_policy") or {})
        summary = {
            "available": True,
            "generated_at_utc": generated_at,
            "app_version": APP_VERSION,
            "headline": headline,
            "summary": detail,
            "status": "recorded",
            "trigger_source": trigger_source,
            "selected_policy_source": selected_policy_source,
            **live_state,
            "current_policy_search_verdict": current_policy_search_verdict,
            "candidate_pool_count": candidate_pool_count,
            "active_challenger_count": count,
            "challenger_records": challenger_records,
            # backward-compatible primary challenger fields
            "challenger_policy": primary_policy,
            "challenger": primary.get("challenger") or {},
            "incumbent": primary.get("incumbent") or {},
            "comparison": primary.get("comparison") or {},
            "live_path_unchanged": True,
            "live_path_statement": "Legacy remains the effective live selection engine; the supported winner is being measured in shadow only.",
            "shadow_candidate_mode": "single_supported_offline_winner" if count == 1 else "multi_challenger_context",
            "primary_shadow_candidate": {
                "policy_name": primary_policy.get("policy_name"),
                "policy_id": primary_policy.get("policy_id"),
                "support_level": primary_policy.get("support_level"),
                "winner_override": dict(primary_policy.get("winner_override") or {}),
            } if primary_policy else None,
            "promotion_ready": False,
            "promotion_ready_reason": "Shadow comparison is evidence-only. Live promotion remains disallowed until a challenger beats legacy offline and then survives a controlled proof window.",
            "pack_available": False,
        }
        return summary

    def _build_pack(
        self,
        summary: dict[str, Any],
        candidate_pool: list[dict],
        incumbent_rows: list[dict],
        trimmed_rows: list[dict],
        challenger_row_map: dict[str, list[dict]],
        policy_summary: dict[str, Any],
    ) -> None:
        recent_history = self._read_history()[-100:]
        with zipfile.ZipFile(self.pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("latest_shadow_selection_comparison_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr("trailing_24h_summary.json", json.dumps(summary.get("trailing_24h") or {}, indent=2, sort_keys=True))
            zf.writestr("candidate_pool.json", json.dumps(candidate_pool, indent=2, sort_keys=True))
            zf.writestr("incumbent_visible_rows.json", json.dumps(incumbent_rows, indent=2, sort_keys=True))
            zf.writestr("incumbent_trimmed_rows.json", json.dumps(trimmed_rows, indent=2, sort_keys=True))
            for name, rows in challenger_row_map.items():
                zf.writestr(f"challenger_rows/{_slug(name)}.json", json.dumps(rows, indent=2, sort_keys=True))
            zf.writestr("policy_search_summary_snapshot.json", json.dumps(policy_summary, indent=2, sort_keys=True))
            zf.writestr("comparison_history_recent.json", json.dumps(recent_history, indent=2, sort_keys=True))
            zf.writestr("README.txt", self._pack_readme(summary))

    def _pack_readme(self, summary: dict[str, Any]) -> str:
        counts = ", ".join(
            f"{item.get('policy_name')}: {item.get('visible_count')}"
            for item in (summary.get("challenger_records") or [])
        ) or "-"
        return (
            "Shadow Selection Comparison Pack\n\n"
            "This pack records the unchanged live legacy shortlist against the supported offline winner in controlled shadow on the same scan.\n\n"
            f"Generated: {summary.get('generated_at_utc')}\n"
            f"Live engine: {summary.get('effective_live_selection_engine')}\n"
            f"Live path unchanged: {summary.get('live_path_unchanged')}\n"
            f"Active challengers: {summary.get('active_challenger_count')}\n"
            f"Challenger visible counts: {counts}\n"
        )

    def record_scan(
        self,
        *,
        status: dict,
        live_rows: list[dict],
        trimmed_visible_rows: list[dict],
        effective_max: int,
        tracked_priority_symbols: list[str] | None = None,
        trigger_source: str = "manual",
    ) -> dict:
        generated_at = _utc_now_iso()
        matched_run = self._match_run(app_version=str((status or {}).get("version") or APP_VERSION), generated_at_utc=generated_at)
        source_run_id = str((matched_run or {}).get("run_id") or "") or None
        source_scan_finished_utc = str((matched_run or {}).get("scan_finished_utc") or generated_at)
        live_state = self._resolve_live_selection_state(status)
        live_mode = str(live_state.get("effective_live_selection_mode") or live_state.get("configured_live_selection_mode") or "legacy").lower()
        policy_summary = read_json(self.policy_search_summary_path, {})
        selected_policies, selected_policy_source = self._select_shadow_policies(policy_summary)
        current_policy_search_verdict = str(policy_summary.get("verdict") or "")
        offline_gate_ok, offline_gate_meta = self._offline_gate_state(policy_summary, selected_policies)

        actionable_trimmed = [dict(r) for r in (trimmed_visible_rows or []) if str(r.get("suppression_reason") or "") in {"", "display_trim"}]
        incumbent_rows = [dict(r) for r in (live_rows or [])]
        candidate_pool = self._candidate_pool(incumbent_rows, actionable_trimmed)

        if live_mode != "legacy":
            summary = {
                "available": True,
                "generated_at_utc": generated_at,
                "app_version": APP_VERSION,
                "headline": "Shadow comparison skipped because legacy is not the live incumbent",
                "summary": "This automatic shadow comparison only runs when legacy is the effective live selection engine.",
                "status": "skipped",
                "skip_reason": "live_engine_not_legacy",
                **live_state,
                "current_policy_search_verdict": current_policy_search_verdict,
                "candidate_pool_count": len(candidate_pool),
                "trigger_source": trigger_source,
                "selected_policy_source": selected_policy_source,
                "live_path_unchanged": False,
                "live_path_statement": "Legacy is not currently the effective live engine, so the legacy-vs-shadow comparison is intentionally disabled.",
                "shadow_candidate_mode": "single_supported_offline_winner",
                "primary_shadow_candidate": None,
                "pack_available": False,
            }
            if self.pack_path.exists():
                try:
                    self.pack_path.unlink()
                except Exception:
                    pass
            atomic_write_json(self.summary_path, summary)
            return summary

        if not offline_gate_ok:
            summary = {
                "available": True,
                "generated_at_utc": generated_at,
                "app_version": APP_VERSION,
                "headline": "Shadow comparison blocked by the offline gate",
                "summary": "Live shadow testing is disabled until at least one challenger beats legacy on historic/offline data.",
                "status": "blocked_offline_gate",
                "skip_reason": "offline_gate_not_met",
                **live_state,
                **offline_gate_meta,
                "current_policy_search_verdict": current_policy_search_verdict,
                "candidate_pool_count": len(candidate_pool),
                "trigger_source": trigger_source,
                "selected_policy_source": selected_policy_source,
                "live_path_unchanged": True,
                "live_path_statement": "Legacy remains the live path; no shadow candidate is allowed until the offline gate passes.",
                "shadow_candidate_mode": "single_supported_offline_winner",
                "primary_shadow_candidate": None,
                "pack_available": False,
            }
            if self.pack_path.exists():
                try:
                    self.pack_path.unlink()
                except Exception:
                    pass
            atomic_write_json(self.summary_path, summary)
            return summary

        if not selected_policies:
            summary = {
                "available": True,
                "generated_at_utc": generated_at,
                "app_version": APP_VERSION,
                "headline": "Shadow comparison skipped because no supported shadow candidate is available",
                "summary": "The offline gate may have passed, but no supported winner with a clean override payload is currently available for controlled shadow.",
                "status": "skipped",
                "skip_reason": "no_supported_shadow_candidate",
                **live_state,
                "current_policy_search_verdict": current_policy_search_verdict,
                "candidate_pool_count": len(candidate_pool),
                "trigger_source": trigger_source,
                "selected_policy_source": selected_policy_source,
                "live_path_unchanged": True,
                "live_path_statement": "Legacy remains the live path while the app waits for one clean supported shadow candidate.",
                "shadow_candidate_mode": "single_supported_offline_winner",
                "primary_shadow_candidate": None,
                "pack_available": False,
            }
            if self.pack_path.exists():
                try:
                    self.pack_path.unlink()
                except Exception:
                    pass
            atomic_write_json(self.summary_path, summary)
            return summary

        records: list[dict[str, Any]] = []
        challenger_row_map: dict[str, list[dict]] = {}
        for policy in selected_policies:
            override = dict(policy.get("override") or {})
            policy_name = str(policy.get("policy_name") or policy.get("engine_label") or "").strip()
            if not override or not policy_name:
                continue
            policy_config = utility_config_with_runtime_override(self.config, override)
            shadow_result = optimize_visible_shortlist(
                candidate_pool,
                effective_max=max(0, int(effective_max or 0)),
                config=policy_config,
                tracked_priority_symbols=list(tracked_priority_symbols or []),
            )
            challenger_rows = [dict(r) for r in (shadow_result.visible_rows or [])]
            challenger_row_map[policy_name] = challenger_rows
            record = self._build_record_summary(
                generated_at=generated_at,
                status=status,
                incumbent_rows=incumbent_rows,
                challenger_rows=challenger_rows,
                candidate_pool=candidate_pool,
                current_policy_search_verdict=current_policy_search_verdict,
                policy=policy,
                trigger_source=trigger_source,
                source_run_id=source_run_id,
                source_scan_finished_utc=source_scan_finished_utc,
                selected_policy_source=selected_policy_source,
            )
            records.append(record)
            self._append_history(record)

        summary = self._build_group_summary(
            records=records,
            live_state=live_state,
            current_policy_search_verdict=current_policy_search_verdict,
            trigger_source=trigger_source,
            candidate_pool_count=len(candidate_pool),
            selected_policy_source=selected_policy_source,
        )
        summary["trailing_24h"] = self._trailing_summary(window_hours=24)
        atomic_write_json(self.summary_path, summary)
        self._build_pack(summary, candidate_pool, incumbent_rows, actionable_trimmed, challenger_row_map, policy_summary)
        summary["pack_available"] = self.pack_path.exists()
        atomic_write_json(self.summary_path, summary)
        return summary
