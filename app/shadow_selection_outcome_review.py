from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .version import APP_VERSION


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_utc(value: str | None) -> datetime | None:
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


def _avg(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / float(len(clean))


def _csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    fieldnames = sorted({k for row in rows for k in row.keys()})
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


class ShadowSelectionOutcomeReviewService:
    def __init__(self, config: AppConfig, review_packs: Any, shadow_comparison_service: Any, utility_policy_search_lab: Any | None = None):
        self.config = config
        self.review_packs = review_packs
        self.shadow_comparison_service = shadow_comparison_service
        self.utility_policy_search_lab = utility_policy_search_lab
        self.root_dir = ensure_dir(Path(config.model_dir) / "shadow_selection_outcome_review")
        self.summary_path = self.root_dir / "latest_shadow_selection_outcome_review_summary.json"
        self.pack_path = self.root_dir / "latest_shadow_selection_outcome_review_pack.zip"
        self.history_path = Path(config.model_dir) / "shadow_selection_comparison" / "comparison_history.jsonl"
        self.shadow_summary_path = Path(config.model_dir) / "shadow_selection_comparison" / "latest_shadow_selection_comparison_summary.json"
        self.policy_search_summary_path = Path(config.model_dir) / "utility_policy_search_lab" / "latest_utility_policy_search_lab_summary.json"
        self.policy_search_status_path = Path(config.model_dir) / "utility_policy_search_lab" / "latest_utility_policy_search_lab_status.json"
        self.automation_state_path = self.root_dir / "retired_shadow_candidate_automation_state.json"

    def latest_summary(self) -> dict:
        summary = self._build_summary()
        atomic_write_json(self.summary_path, summary)
        if summary.get("pack_available"):
            self._build_pack(summary)
        return summary

    def latest_pack(self) -> Path | None:
        summary = self.latest_summary()
        return self.pack_path if summary.get("pack_available") and self.pack_path.exists() else None

    def _history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        rows = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _match_run(self, item: dict[str, Any]) -> dict | None:
        existing = str(item.get("source_run_id") or "")
        if existing:
            try:
                run = self.review_packs.get_run(existing)
            except Exception:
                run = None
            if run:
                return run
        app_version = str(item.get("app_version") or APP_VERSION)
        generated_at = _parse_utc(item.get("generated_at_utc"))
        source_finished = _parse_utc(item.get("source_scan_finished_utc"))
        anchor = source_finished or generated_at
        try:
            runs = self.review_packs.get_runs_for_app_version(app_version, limit=500)
        except Exception:
            return None
        best = None
        best_delta = None
        for run in runs:
            finished = _parse_utc(run.get("scan_finished_utc"))
            if finished is None or anchor is None:
                continue
            delta = abs((anchor - finished).total_seconds())
            if delta > 1800:
                continue
            if best is None or delta < (best_delta or 10**18):
                best = run
                best_delta = delta
        return best

    def _engine_rows(self, rows: list[dict], symbols: set[str], scan_id: str) -> list[dict]:
        out = []
        for row in rows:
            symbol = str(row.get("symbol") or "")
            out.append({
                "as_of_utc": scan_id,
                "row_type": "visible" if symbol in symbols else "hidden",
                "quality_touched": int(bool(row.get("quality_touched"))),
                "end_ret": _safe_float(row.get("end_ret")),
                "mae": _safe_float(row.get("mae")),
                "symbol": symbol,
            })
        return out

    def _offline_gate_blocked(self) -> tuple[bool, dict[str, Any]]:
        policy_summary = self._latest_policy_search_summary()
        verdict = str(policy_summary.get("verdict") or "")
        ranked = [dict(item) for item in (policy_summary.get("ranked_policies") or []) if isinstance(item, dict)]
        supported = [item for item in ranked if str(item.get("support_level") or "") == "supported_offline"]
        if verdict == "supported_policy_found" and supported:
            return False, {
                "offline_gate_status": "passed",
                "supported_policy_names": [str(item.get("policy_name") or item.get("engine_label") or "") for item in supported],
            }
        shadow_summary = read_json(self.shadow_summary_path, {})
        return True, {
            "offline_gate_status": "blocked",
            "policy_search_verdict": verdict or "unknown",
            "supported_policy_names": [],
            "selected_policy_source": shadow_summary.get("selected_policy_source") or policy_summary.get("selected_policy_source"),
            "active_challengers": [str(item.get("policy_name") or "") for item in (shadow_summary.get("challenger_records") or []) if str(item.get("policy_name") or "")],
        }

    def _active_challenger_names(self) -> list[str]:
        summary = read_json(self.shadow_summary_path, {})
        retired = self._retired_challenger_names()
        primary = dict(summary.get("primary_shadow_candidate") or {})
        primary_name = str(primary.get("policy_name") or "").strip()
        if primary_name and primary_name not in retired:
            return [primary_name]
        names = []
        for item in (summary.get("challenger_records") or []):
            name = str(item.get("policy_name") or "").strip()
            if name and name not in retired:
                names.append(name)
        return names

    def _retired_challenger_names(self) -> set[str]:
        latest = read_json(self.summary_path, {})
        retired = set()
        for item in latest.get("retired_challengers") or []:
            name = str(item.get("policy_name") or item.get("engine") or item or "").strip()
            if name:
                retired.add(name)
        return retired

    def _latest_policy_search_summary(self) -> dict[str, Any]:
        if self.utility_policy_search_lab is not None:
            try:
                return dict(self.utility_policy_search_lab.latest_summary() or {})
            except Exception:
                pass
        return read_json(self.policy_search_summary_path, {})

    def _latest_policy_search_status(self) -> dict[str, Any]:
        if self.utility_policy_search_lab is not None:
            try:
                return dict(self.utility_policy_search_lab.latest_status() or {})
            except Exception:
                pass
        return read_json(self.policy_search_status_path, {})

    def _maybe_trigger_blocked_gate_policy_search(self, *, active_challengers: list[str], policy_search_verdict: str) -> dict[str, Any] | None:
        if active_challengers:
            return None
        latest_status = self._latest_policy_search_status()
        state = read_json(self.automation_state_path, {})
        signature = f"blocked_offline_gate|{APP_VERSION}|{policy_search_verdict}"
        if bool(latest_status.get("active")):
            return {
                "triggered": False,
                "reason": "already_active",
                "status": latest_status.get("status"),
                "run_id": latest_status.get("run_id"),
                "active": True,
            }
        if str(state.get("last_blocked_gate_trigger_signature") or "") == signature:
            return {
                "triggered": False,
                "reason": "already_triggered_for_scope",
                "status": latest_status.get("status"),
                "run_id": latest_status.get("run_id"),
                "active": bool(latest_status.get("active")),
                "last_triggered_at_utc": state.get("last_blocked_gate_triggered_at_utc"),
            }
        if self.utility_policy_search_lab is None:
            return {
                "triggered": False,
                "reason": "service_unavailable",
                "status": latest_status.get("status"),
                "run_id": latest_status.get("run_id"),
                "active": False,
            }
        try:
            trigger_result = dict(self.utility_policy_search_lab.start_run() or {})
            latest_status = self._latest_policy_search_status()
            state.update({
                "last_blocked_gate_trigger_signature": signature,
                "last_blocked_gate_triggered_at_utc": _utc_now_iso(),
                "last_blocked_gate_trigger_reason": "offline_gate_not_met_after_retirement_cycle",
                "last_blocked_gate_trigger_result": {
                    "run_id": trigger_result.get("run_id"),
                    "status": trigger_result.get("status"),
                    "active": trigger_result.get("active"),
                },
            })
            atomic_write_json(self.automation_state_path, state)
            return {
                "triggered": True,
                "reason": "started_fresh_offline_family_search",
                "status": latest_status.get("status"),
                "run_id": latest_status.get("run_id") or trigger_result.get("run_id"),
                "active": bool(latest_status.get("active") or trigger_result.get("active")),
            }
        except Exception as exc:
            state.update({
                "last_blocked_gate_trigger_signature": signature,
                "last_blocked_gate_triggered_at_utc": _utc_now_iso(),
                "last_blocked_gate_trigger_reason": "offline_gate_not_met_after_retirement_cycle",
                "last_blocked_gate_trigger_error": str(exc),
            })
            atomic_write_json(self.automation_state_path, state)
            return {
                "triggered": False,
                "reason": "trigger_failed",
                "error": str(exc),
                "status": latest_status.get("status"),
                "run_id": latest_status.get("run_id"),
                "active": False,
            }

    def _maybe_trigger_next_policy_search(self, *, retired_policy_name: str, retirement_anchor_utc: str, matured_comparisons: int) -> dict[str, Any]:
        status = read_json(self.policy_search_status_path, {})
        summary = read_json(self.policy_search_summary_path, {})
        latest_status = {}
        if self.utility_policy_search_lab is not None:
            try:
                latest_status = dict(self.utility_policy_search_lab.latest_status() or {})
            except Exception:
                latest_status = dict(status or {})
        else:
            latest_status = dict(status or {})
        state = read_json(self.automation_state_path, {})
        signature = f"{retired_policy_name}|{matured_comparisons}|{retirement_anchor_utc}"
        already_triggered = str(state.get("last_trigger_signature") or "") == signature
        active = bool(latest_status.get("active"))
        policy_generated = _parse_utc(summary.get("generated_at_utc"))
        retired_generated = _parse_utc(retirement_anchor_utc)
        post_retirement_summary_exists = bool(policy_generated and retired_generated and policy_generated >= retired_generated)
        triggered = False
        trigger_result = None
        if (
            self.utility_policy_search_lab is not None
            and not already_triggered
            and not active
            and not post_retirement_summary_exists
        ):
            try:
                trigger_result = dict(self.utility_policy_search_lab.start_run() or {})
                latest_status = dict(self.utility_policy_search_lab.latest_status() or trigger_result)
                triggered = True
                state.update({
                    "last_trigger_signature": signature,
                    "last_triggered_at_utc": _utc_now_iso(),
                    "last_retired_policy_name": retired_policy_name,
                    "last_trigger_reason": "retired_supported_shadow_candidate",
                    "last_trigger_result": {
                        "run_id": trigger_result.get("run_id"),
                        "status": trigger_result.get("status"),
                        "active": trigger_result.get("active"),
                    },
                })
                atomic_write_json(self.automation_state_path, state)
            except Exception as exc:
                state.update({
                    "last_trigger_signature": signature,
                    "last_triggered_at_utc": _utc_now_iso(),
                    "last_retired_policy_name": retired_policy_name,
                    "last_trigger_reason": "retired_supported_shadow_candidate",
                    "last_trigger_error": str(exc),
                })
                atomic_write_json(self.automation_state_path, state)
        return {
            "retired_policy_name": retired_policy_name,
            "trigger_signature": signature,
            "triggered_new_run": triggered,
            "already_triggered_for_this_retirement": already_triggered,
            "post_retirement_summary_exists": post_retirement_summary_exists,
            "latest_policy_search_status": latest_status,
            "latest_policy_search_summary_generated_at_utc": summary.get("generated_at_utc"),
        }

    def _scan_utility(self, rows: list[dict]) -> dict[str, Any]:
        empty = {
            "scan_shortlist_scans": 0,
            "scan_shortlist_scans_with_visible": 0,
            "scan_shortlist_avg_visible_rows_per_scan": None,
            "scan_shortlist_visible_quality_rate_mean": None,
            "scan_shortlist_hidden_quality_rate_mean": None,
            "scan_shortlist_mean_gap": None,
            "scan_shortlist_pairwise_win_rate": None,
            "scan_shortlist_pairwise_comparable_scans": 0,
            "scan_shortlist_top1_visible_quality": None,
            "scan_shortlist_top3_visible_quality": None,
            "scan_shortlist_overwide_penalty": None,
            "scan_shortlist_utility_score": None,
            "resolved_rows": 0,
            "visible_row_count": 0,
            "hidden_row_count": 0,
            "visible_quality_hit_rate": None,
            "hidden_quality_hit_rate": None,
            "visible_avg_end_ret": None,
            "hidden_avg_end_ret": None,
            "visible_avg_mae": None,
            "hidden_avg_mae": None,
        }
        if not rows:
            return dict(empty)
        by_scan: dict[str, list[dict]] = {}
        for row in rows:
            by_scan.setdefault(str(row.get("as_of_utc") or ""), []).append(row)
        scan_count = 0
        scans_with_visible = 0
        pairwise_wins = 0.0
        pairwise_comparable = 0
        visible_counts = []
        visible_rates = []
        hidden_rates = []
        gaps = []
        top1_rates = []
        top3_rates = []
        visible_rows_all = [r for r in rows if str(r.get("row_type")) == "visible"]
        hidden_rows_all = [r for r in rows if str(r.get("row_type")) != "visible"]
        for scan_id, scan_rows in by_scan.items():
            if not scan_id:
                continue
            scan_count += 1
            visible = [r for r in scan_rows if str(r.get("row_type")) == "visible"]
            hidden = [r for r in scan_rows if str(r.get("row_type")) != "visible"]
            visible_counts.append(len(visible))
            if visible:
                scans_with_visible += 1
                vr = sum(int(bool(r.get("quality_touched"))) for r in visible) / float(len(visible))
                visible_rates.append(vr)
                top1_rates.append(int(bool(visible[0].get("quality_touched"))))
                top3 = visible[: min(3, len(visible))]
                top3_rates.append(sum(int(bool(r.get("quality_touched"))) for r in top3) / float(len(top3)))
                if hidden:
                    hr = sum(int(bool(r.get("quality_touched"))) for r in hidden) / float(len(hidden))
                    hidden_rates.append(hr)
                    gap = vr - hr
                    gaps.append(gap)
                    pairwise_comparable += 1
                    if gap > 0:
                        pairwise_wins += 1.0
                    elif gap == 0:
                        pairwise_wins += 0.5
        avg_visible_rows = _avg([float(v) for v in visible_counts])
        visible_quality_mean = _avg(visible_rates)
        hidden_quality_mean = _avg(hidden_rates)
        mean_gap = _avg(gaps)
        pairwise_win_rate = (pairwise_wins / float(pairwise_comparable)) if pairwise_comparable else None
        top1_mean = _avg(top1_rates)
        top3_mean = _avg(top3_rates)
        overwide_penalty = max(0.0, (avg_visible_rows or 0.0) - 5.0) / 5.0 if avg_visible_rows is not None else None
        base_event_rate = _avg([int(bool(r.get("quality_touched"))) for r in rows]) or 0.0
        utility_score = None
        if mean_gap is not None:
            utility_score = float(mean_gap) + 0.25 * (((pairwise_win_rate if pairwise_win_rate is not None else 0.5) - 0.5)) + 0.10 * (((top1_mean if top1_mean is not None else base_event_rate) - base_event_rate)) + 0.05 * (((top3_mean if top3_mean is not None else base_event_rate) - base_event_rate)) - 0.02 * (overwide_penalty or 0.0)
        return {
            "resolved_rows": len(rows),
            "visible_row_count": len(visible_rows_all),
            "hidden_row_count": len(hidden_rows_all),
            "visible_quality_hit_rate": round(_avg([int(bool(r.get("quality_touched"))) for r in visible_rows_all]), 6) if visible_rows_all else None,
            "hidden_quality_hit_rate": round(_avg([int(bool(r.get("quality_touched"))) for r in hidden_rows_all]), 6) if hidden_rows_all else None,
            "visible_avg_end_ret": round(_avg([_safe_float(r.get("end_ret")) for r in visible_rows_all]), 6) if visible_rows_all else None,
            "hidden_avg_end_ret": round(_avg([_safe_float(r.get("end_ret")) for r in hidden_rows_all]), 6) if hidden_rows_all else None,
            "visible_avg_mae": round(_avg([_safe_float(r.get("mae")) for r in visible_rows_all]), 6) if visible_rows_all else None,
            "hidden_avg_mae": round(_avg([_safe_float(r.get("mae")) for r in hidden_rows_all]), 6) if hidden_rows_all else None,
            "scan_shortlist_scans": scan_count,
            "scan_shortlist_scans_with_visible": scans_with_visible,
            "scan_shortlist_avg_visible_rows_per_scan": round(avg_visible_rows, 6) if avg_visible_rows is not None else None,
            "scan_shortlist_visible_quality_rate_mean": round(visible_quality_mean, 6) if visible_quality_mean is not None else None,
            "scan_shortlist_hidden_quality_rate_mean": round(hidden_quality_mean, 6) if hidden_quality_mean is not None else None,
            "scan_shortlist_mean_gap": round(mean_gap, 6) if mean_gap is not None else None,
            "scan_shortlist_pairwise_win_rate": round(pairwise_win_rate, 6) if pairwise_win_rate is not None else None,
            "scan_shortlist_pairwise_comparable_scans": pairwise_comparable,
            "scan_shortlist_top1_visible_quality": round(top1_mean, 6) if top1_mean is not None else None,
            "scan_shortlist_top3_visible_quality": round(top3_mean, 6) if top3_mean is not None else None,
            "scan_shortlist_overwide_penalty": round(overwide_penalty, 6) if overwide_penalty is not None else None,
            "scan_shortlist_utility_score": round(utility_score, 6) if utility_score is not None else None,
        }



    def _resolved_rows_from_evaluated_pack(self, pack_path: str | None) -> list[dict[str, Any]]:
        path = Path(str(pack_path or "").strip()) if pack_path else None
        if path is None or not path.exists():
            return []
        try:
            with zipfile.ZipFile(path) as zf:
                if "outcomes.csv" not in zf.namelist():
                    return []
                raw = zf.read("outcomes.csv").decode("utf-8", errors="replace")
        except Exception:
            return []
        try:
            rows = list(csv.DictReader(io.StringIO(raw)))
        except Exception:
            return []
        return [dict(r) for r in rows if isinstance(r, dict)]

    def _load_resolved_rows_for_run(self, run_id: str, run_detail: dict[str, Any] | None) -> tuple[list[dict[str, Any]], str | None]:
        rows = self.review_packs._load_rows_for_run_ids([run_id], resolved_only=True)
        if rows:
            return rows, "review_db"
        pack_path = str((run_detail or {}).get("latest_evaluated_pack_path") or "").strip()
        if pack_path:
            pack_rows = self._resolved_rows_from_evaluated_pack(pack_path)
            if pack_rows:
                return pack_rows, "evaluated_pack"
        return [], None

    def _pending_reason(self, *, run_detail: dict[str, Any] | None, resolved_rows: list[dict[str, Any]], resolved_source: str | None, now: datetime, due: datetime | None) -> str:
        if resolved_rows:
            return "resolved"
        if run_detail is None:
            return "missing_run"
        evaluation_complete = bool((run_detail or {}).get("evaluation_complete"))
        if due is not None and now < due:
            return "waiting_for_due_time"
        if not evaluation_complete:
            return "evaluation_overdue" if due is not None and now >= due else "evaluation_pending"
        pack_path = str((run_detail or {}).get("latest_evaluated_pack_path") or "").strip()
        if pack_path:
            return "evaluated_pack_missing_rows"
        return "evaluated_rows_missing"

    def _matured_records(self) -> tuple[list[dict], int, int, dict[str, int], dict[str, int], dict[str, Any]]:
        horizon = int(getattr(self.config, "target_horizon_minutes", 240) or 240)
        threshold = _utc_now() - timedelta(minutes=horizon)
        now = _utc_now()
        matured: list[dict] = []
        waiting = 0
        pending_resolution = 0
        waiting_by_policy: dict[str, int] = {}
        pending_by_policy: dict[str, int] = {}
        pending_reason_counts: Counter[str] = Counter()
        pending_by_policy_reason: dict[str, Counter[str]] = {}
        oldest_pending_ts: datetime | None = None
        oldest_pending_policy: str | None = None
        resolved_source_counts: Counter[str] = Counter()
        for item in self._history():
            ts = _parse_utc(item.get("generated_at_utc"))
            if ts is None:
                continue
            policy_name = str(((item.get("challenger_policy") or {}).get("policy_name") or (item.get("challenger") or {}).get("engine") or "-")).strip() or "-"
            if str(item.get("status") or "") != "recorded":
                continue
            run = self._match_run(item)
            run_id = str((run or {}).get("run_id") or item.get("source_run_id") or "")
            run_detail = None
            if run_id:
                try:
                    run_detail = self.review_packs._load_run(run_id)
                except Exception:
                    run_detail = None
                if run_detail is None and run is not None:
                    run_detail = dict(run)
            due = _parse_utc((run_detail or {}).get("evaluation_due_utc"))
            if run_detail is not None and due is not None and now < due:
                waiting += 1
                waiting_by_policy[policy_name] = waiting_by_policy.get(policy_name, 0) + 1
                continue
            if run_detail is None and ts > threshold:
                waiting += 1
                waiting_by_policy[policy_name] = waiting_by_policy.get(policy_name, 0) + 1
                continue
            if not run_id:
                pending_resolution += 1
                pending_by_policy[policy_name] = pending_by_policy.get(policy_name, 0) + 1
                pending_reason_counts["missing_run"] += 1
                pending_by_policy_reason.setdefault(policy_name, Counter())["missing_run"] += 1
                if oldest_pending_ts is None or ts < oldest_pending_ts:
                    oldest_pending_ts = ts
                    oldest_pending_policy = policy_name
                continue
            rows, resolved_source = self._load_resolved_rows_for_run(run_id, run_detail)
            if not rows:
                reason = self._pending_reason(run_detail=run_detail, resolved_rows=rows, resolved_source=resolved_source, now=now, due=due)
                if reason == "waiting_for_due_time":
                    waiting += 1
                    waiting_by_policy[policy_name] = waiting_by_policy.get(policy_name, 0) + 1
                    continue
                pending_resolution += 1
                pending_by_policy[policy_name] = pending_by_policy.get(policy_name, 0) + 1
                pending_reason_counts[reason] += 1
                pending_by_policy_reason.setdefault(policy_name, Counter())[reason] += 1
                if oldest_pending_ts is None or ts < oldest_pending_ts:
                    oldest_pending_ts = ts
                    oldest_pending_policy = policy_name
                continue
            resolved_source_counts[resolved_source or "unknown"] += 1
            incumbent_symbols = {str(x) for x in ((item.get("incumbent") or {}).get("symbols") or []) if x}
            challenger_symbols = {str(x) for x in ((item.get("challenger") or {}).get("symbols") or []) if x}
            scan_id = str(item.get("generated_at_utc") or (run_detail or {}).get("scan_finished_utc") or run_id)
            matured.append({
                "summary": item,
                "run": run_detail or run or {},
                "run_id": run_id,
                "scan_id": scan_id,
                "resolved_source": resolved_source or "unknown",
                "incumbent_rows": self._engine_rows(rows, incumbent_symbols, scan_id),
                "challenger_rows": self._engine_rows(rows, challenger_symbols, scan_id),
            })
        diagnostics = {
            "pending_reason_counts": dict(sorted(pending_reason_counts.items())),
            "pending_by_policy_reason": {k: dict(sorted(v.items())) for k, v in sorted(pending_by_policy_reason.items())},
            "resolved_source_counts": dict(sorted(resolved_source_counts.items())),
            "oldest_pending_generated_at_utc": oldest_pending_ts.isoformat() if oldest_pending_ts is not None else None,
            "oldest_pending_policy_name": oldest_pending_policy,
            "oldest_pending_age_hours": round((now - oldest_pending_ts).total_seconds() / 3600.0, 3) if oldest_pending_ts is not None else None,
        }
        return matured, waiting, pending_resolution, waiting_by_policy, pending_by_policy, diagnostics

    def _is_clearly_worse(self, result: dict[str, Any]) -> bool:
        matured = int(result.get("matured_comparisons") or 0)
        if matured < 3:
            return False
        utility_delta = _safe_float(result.get("utility_score_delta_vs_legacy"))
        pairwise_delta = _safe_float(result.get("pairwise_delta_vs_legacy"))
        incumbent = dict(result.get("incumbent") or {})
        challenger = dict(result.get("challenger") or {})
        incumbent_quality = _safe_float(incumbent.get("visible_quality_hit_rate"))
        challenger_quality = _safe_float(challenger.get("visible_quality_hit_rate"))
        incumbent_end_ret = _safe_float(incumbent.get("visible_avg_end_ret"))
        challenger_end_ret = _safe_float(challenger.get("visible_avg_end_ret"))
        if utility_delta is not None and utility_delta <= -0.05:
            return True
        if pairwise_delta is not None and pairwise_delta <= -0.20:
            return True
        # After a meaningful proof window, retire challengers that remain tighter but are still not
        # outperforming legacy on the combination of pairwise wins, visible quality, and visible end return.
        if matured >= 40:
            quality_not_better = (
                incumbent_quality is not None and challenger_quality is not None and challenger_quality <= incumbent_quality
            )
            end_ret_materially_worse = (
                incumbent_end_ret is not None and challenger_end_ret is not None and challenger_end_ret < (incumbent_end_ret - 0.002)
            )
            pairwise_losing = pairwise_delta is not None and pairwise_delta < 0.0
            # A challenger that still loses pairwise after a meaningful proof window and produces
            # non-positive visible average end return is not trustworthy for the app's true objective,
            # even if it looks slightly tighter on utility or visible hit rate.
            end_ret_non_positive = challenger_end_ret is not None and challenger_end_ret <= 0.0
            incumbent_end_ret_positive = incumbent_end_ret is not None and incumbent_end_ret > 0.0
            if pairwise_losing and quality_not_better and end_ret_materially_worse:
                return True
            if pairwise_losing and incumbent_end_ret_positive and end_ret_non_positive:
                return True
        return False

    def _build_summary(self) -> dict[str, Any]:
        matured, waiting, pending_resolution, waiting_by_policy, pending_by_policy, pending_diagnostics = self._matured_records()
        generated_at = _utc_now_iso()
        active_challengers = self._active_challenger_names()
        offline_gate_blocked, offline_gate_meta = self._offline_gate_blocked()
        if offline_gate_blocked:
            next_policy_search = self._maybe_trigger_blocked_gate_policy_search(
                active_challengers=active_challengers,
                policy_search_verdict=str(offline_gate_meta.get("policy_search_verdict") or "unknown"),
            )
            return {
                "available": True,
                "app_version": APP_VERSION,
                "generated_at_utc": generated_at,
                "headline": "Shadow outcome review blocked by the offline gate",
                "summary": "Live shadow outcome review is disabled because no challenger has beaten legacy on historic/offline data.",
                "status": "blocked_offline_gate",
                "verdict": "offline_gate_not_met",
                "recommended_action": "run_offline_family_search_before_any_live_shadow_testing",
                "pack_available": False,
                "matured_comparisons": 0,
                "waiting_for_maturity": 0,
                "pending_resolution": 0,
                "waiting_by_policy": {},
                "pending_resolution_by_policy": {},
                "pending_resolution_diagnostics": {"pending_reason_counts": {}, "pending_by_policy_reason": {}, "resolved_source_counts": {}},
                **offline_gate_meta,
                "active_challengers": active_challengers,
                "challenger_results": [],
                "retired_challengers": [],
                "recent_comparisons": [],
                "next_policy_search": next_policy_search,
            }
        if not matured:
            return {
                "available": True,
                "app_version": APP_VERSION,
                "generated_at_utc": generated_at,
                "headline": "Waiting for matured supported-shadow evidence",
                "summary": "The supported offline winner is being recorded in controlled shadow, but none of those comparisons have matured past the 240-minute target horizon yet.",
                "status": "waiting",
                "pack_available": False,
                "matured_comparisons": 0,
                "waiting_for_maturity": waiting,
                "pending_resolution": pending_resolution,
                "waiting_by_policy": waiting_by_policy,
                "pending_resolution_by_policy": pending_by_policy,
                "pending_resolution_diagnostics": pending_diagnostics,
                "active_challengers": active_challengers,
                "live_path_unchanged": True,
                "shadow_candidate_mode": "single_supported_offline_winner",
                "challenger_results": [],
                "retired_challengers": [],
            }

        by_challenger: dict[str, list[dict]] = {}
        comparison_rows: list[dict] = []
        for rec in matured:
            name = str(((rec["summary"].get("challenger_policy") or {}).get("policy_name") or (rec["summary"].get("challenger") or {}).get("engine") or "-")).strip() or "-"
            by_challenger.setdefault(name, []).append(rec)
            comparison_rows.append({
                "generated_at_utc": rec["summary"].get("generated_at_utc"),
                "run_id": rec["run_id"],
                "challenger_policy": name,
                "incumbent_visible_count": len([r for r in rec["incumbent_rows"] if r["row_type"] == "visible"]),
                "challenger_visible_count": len([r for r in rec["challenger_rows"] if r["row_type"] == "visible"]),
            })

        challenger_results: list[dict[str, Any]] = []
        for name, items in by_challenger.items():
            incumbent_rows = []
            challenger_rows = []
            support_level = None
            engine_id = None
            for rec in items:
                incumbent_rows.extend(rec["incumbent_rows"])
                challenger_rows.extend(rec["challenger_rows"])
                policy = rec["summary"].get("challenger_policy") or {}
                support_level = support_level or str(policy.get("support_level") or "")
                engine_id = engine_id or str(policy.get("policy_id") or rec["summary"].get("challenger", {}).get("engine") or name)
            incumbent = self._scan_utility(incumbent_rows)
            challenger = self._scan_utility(challenger_rows)
            inc_u = _safe_float(incumbent.get("scan_shortlist_utility_score"))
            ch_u = _safe_float(challenger.get("scan_shortlist_utility_score"))
            inc_pw = _safe_float(incumbent.get("scan_shortlist_pairwise_win_rate"))
            ch_pw = _safe_float(challenger.get("scan_shortlist_pairwise_win_rate"))
            inc_gap = _safe_float(incumbent.get("scan_shortlist_mean_gap"))
            ch_gap = _safe_float(challenger.get("scan_shortlist_mean_gap"))
            result = {
                "policy_name": name,
                "engine": engine_id or name,
                "support_level": support_level or "",
                "matured_comparisons": len(items),
                "utility_score_delta_vs_legacy": round((ch_u - inc_u), 6) if ch_u is not None and inc_u is not None else None,
                "pairwise_delta_vs_legacy": round((ch_pw - inc_pw), 6) if ch_pw is not None and inc_pw is not None else None,
                "mean_gap_delta_vs_legacy": round((ch_gap - inc_gap), 6) if ch_gap is not None and inc_gap is not None else None,
                "incumbent": {"engine": "legacy", **incumbent},
                "challenger": {"engine": name, **challenger},
                "recent_comparisons": [
                    {
                        "generated_at_utc": rec["summary"].get("generated_at_utc"),
                        "run_id": rec["run_id"],
                        "incumbent_visible_count": len([r for r in rec["incumbent_rows"] if r["row_type"] == "visible"]),
                        "challenger_visible_count": len([r for r in rec["challenger_rows"] if r["row_type"] == "visible"]),
                    }
                    for rec in items[-20:]
                ],
            }
            result["retire_recommended"] = self._is_clearly_worse(result)
            challenger_results.append(result)

        challenger_results.sort(
            key=lambda r: (
                -(_safe_float(r.get("utility_score_delta_vs_legacy")) if _safe_float(r.get("utility_score_delta_vs_legacy")) is not None else -999.0),
                -(_safe_float(r.get("pairwise_delta_vs_legacy")) if _safe_float(r.get("pairwise_delta_vs_legacy")) is not None else -999.0),
                str(r.get("policy_name") or ""),
            )
        )
        retired = [
            {
                "policy_name": r.get("policy_name"),
                "engine": r.get("engine"),
                "support_level": r.get("support_level"),
                "reason": "underperforming_after_meaningful_proof_window" if int(r.get("matured_comparisons") or 0) >= 40 else "clearly_worse_after_3_matured_comparisons",
            }
            for r in challenger_results if r.get("retire_recommended")
        ]
        active_result_names = {str(r.get("policy_name") or "") for r in challenger_results if str(r.get("policy_name") or "") in set(active_challengers)}
        if active_challengers and not active_result_names:
            return {
                "available": True,
                "app_version": APP_VERSION,
                "generated_at_utc": generated_at,
                "headline": "The supported shadow candidate is still awaiting matured live outcomes",
                "summary": "The controlled shadow candidate is being recorded, but it has not yet produced matured resolved evidence.",
                "status": "waiting_active_maturity",
                "verdict": "waiting_for_active_challenger_maturity",
                "recommended_action": "continue_shadow_collection_until_the_supported_candidate_matures",
                "pack_available": bool(challenger_results),
                "matured_comparisons": len(matured),
                "waiting_for_maturity": waiting,
                "pending_resolution": pending_resolution,
                "waiting_by_policy": waiting_by_policy,
                "pending_resolution_by_policy": pending_by_policy,
                "pending_resolution_diagnostics": pending_diagnostics,
                "active_challengers": active_challengers,
                "live_path_unchanged": True,
                "shadow_candidate_mode": "single_supported_offline_winner",
                "incumbent": {},
                "challenger": {},
                "challenger_results": challenger_results,
                "retired_challengers": retired,
                "recent_comparisons": comparison_rows[-50:],
            }
        active_name_set = {str(name or "") for name in active_challengers}
        top = next((r for r in challenger_results if str(r.get("policy_name") or "") in active_name_set), None)
        if top is None:
            top = next((r for r in challenger_results if str(r.get("support_level") or "") == "supported_offline"), challenger_results[0])
            if not active_challengers and top:
                active_challengers = [str(top.get("policy_name") or "")]
        top_incumbent = top.get("incumbent") or {}
        top_challenger = top.get("challenger") or {}
        top_u = _safe_float(top.get("utility_score_delta_vs_legacy"))
        top_pw = _safe_float(top.get("pairwise_delta_vs_legacy"))
        top_retire = bool(top.get("retire_recommended"))
        matured_top = int(top.get("matured_comparisons") or 0)
        next_policy_search = None
        if top_u is not None and top_u > 0 and (top_pw is None or top_pw >= 0):
            headline = "The supported shadow candidate is beating legacy in matured live evidence"
            verdict = "challenger_beating_legacy_live_shadow"
            reco = "continue_shadow_collection_until_sample_is_sufficient"
        elif top_retire and matured_top >= 40:
            headline = "The supported shadow candidate has failed its controlled proof window"
            verdict = "supported_challenger_failed_live_shadow_proof"
            reco = "keep_legacy_live_and_retire_current_supported_shadow_candidate"
            retired_name = str(top.get("policy_name") or top.get("engine") or "").strip()
            if retired_name:
                active_challengers = [name for name in active_challengers if str(name or "").strip() != retired_name]
                next_policy_search = self._maybe_trigger_next_policy_search(
                    retired_policy_name=retired_name,
                    retirement_anchor_utc=(str(((top.get("recent_comparisons") or [{}])[-1].get("generated_at_utc") or generated_at))),
                    matured_comparisons=matured_top,
                )
        else:
            headline = "Legacy is still beating the supported shadow candidate in matured live evidence"
            verdict = "legacy_still_better_in_live_shadow"
            reco = "keep_legacy_live_and_continue_shadow_collection"
        return {
            "available": True,
            "app_version": APP_VERSION,
            "generated_at_utc": generated_at,
            "headline": headline,
            "summary": "This review compares the unchanged live legacy path against the supported shadow candidate using matured rows from the same recorded scan comparisons.",
            "status": "reviewed",
            "verdict": verdict,
            "recommended_action": reco,
            "pack_available": True,
            "matured_comparisons": len(matured),
            "waiting_for_maturity": waiting,
            "pending_resolution": pending_resolution,
            "waiting_by_policy": waiting_by_policy,
            "pending_resolution_by_policy": pending_by_policy,
            "pending_resolution_diagnostics": pending_diagnostics,
            "active_challengers": active_challengers,
            "live_path_unchanged": True,
            "shadow_candidate_mode": "single_supported_offline_winner",
            # backward-compatible top result
            "incumbent": top_incumbent,
            "challenger": top_challenger,
            "challenger_results": challenger_results,
            "retired_challengers": retired,
            "retired_supported_shadow_candidate": (str(top.get("policy_name") or top.get("engine") or "") if top_retire and matured_top >= 40 else None),
            "next_policy_search": next_policy_search,
            "recent_comparisons": comparison_rows[-50:],
            "active_challenger_retire_recommended": top_retire,
            "active_challenger_retire_reason": (
                "underperforming_after_meaningful_proof_window" if top_retire and matured_top >= 40 else None
            ),
            "proof_window_matured_comparisons_threshold": 40,
        }

    def _build_pack(self, summary: dict[str, Any]) -> None:
        matured, _, _, _, _, _ = self._matured_records()
        ensure_dir(self.pack_path.parent)
        with zipfile.ZipFile(self.pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("shadow_selection_outcome_review_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr("shadow_selection_outcome_review_summary.txt", self._summary_text(summary))
            zf.writestr("recent_comparisons.csv", _csv_bytes(summary.get("recent_comparisons") or []))
            for result in summary.get("challenger_results") or []:
                name = str(result.get("policy_name") or result.get("engine") or "challenger")
                zf.writestr(f"challenger_results/{name.replace('/', '_')}.json", json.dumps(result, indent=2, sort_keys=True))
            inc_rows = []
            ch_rows = []
            for rec in matured:
                inc_rows.extend(rec["incumbent_rows"])
                ch_rows.extend(rec["challenger_rows"])
            zf.writestr("incumbent_resolved_rows.csv", _csv_bytes(inc_rows))
            zf.writestr("challenger_resolved_rows.csv", _csv_bytes(ch_rows))

    def _summary_text(self, summary: dict[str, Any]) -> str:
        lines = [
            f"Headline: {summary.get('headline') or '-'}",
            f"Summary: {summary.get('summary') or '-'}",
            f"Status: {summary.get('status') or '-'}",
            f"Verdict: {summary.get('verdict') or '-'}",
            f"Recommended action: {summary.get('recommended_action') or '-'}",
            f"Matured comparisons: {summary.get('matured_comparisons') or 0}",
            f"Waiting for maturity: {summary.get('waiting_for_maturity') or 0}",
            "",
            "Challenger results:",
        ]
        for result in summary.get("challenger_results") or []:
            ch = result.get("challenger") or {}
            lines.extend([
                f"- {result.get('policy_name')}",
                f"  Matured comparisons: {result.get('matured_comparisons')}",
                f"  Utility delta vs legacy: {result.get('utility_score_delta_vs_legacy')}",
                f"  Pairwise delta vs legacy: {result.get('pairwise_delta_vs_legacy')}",
                f"  Mean-gap delta vs legacy: {result.get('mean_gap_delta_vs_legacy')}",
                f"  Challenger utility score: {ch.get('scan_shortlist_utility_score')}",
                f"  Retire recommended: {result.get('retire_recommended')}",
                "",
            ])
        return "\n".join(lines).rstrip()
