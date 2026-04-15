from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .persist import atomic_write_json, ensure_dir, read_json
from .objective_semantics import load_objective_semantics_contract
from .version import APP_VERSION


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelOutputDistributionService:
    """Read-only raw model output distribution diagnostic.

    This service records per-scan distribution snapshots after Stage 2 scoring,
    persists an append-only log, and maintains a rolling summary over the latest
    scans. It is intentionally observational only and must not change any live
    scoring or policy behavior.
    """

    def __init__(self, config):
        self.config = config
        self.root_dir = ensure_dir(Path(config.model_dir) / "diagnostics")
        self.snapshot_dir = ensure_dir(self.root_dir / "model_output_distribution")
        self.summary_path = self.root_dir / "model_output_distribution_summary.json"
        self.log_path = self.root_dir / "model_output_distribution_log.jsonl"
        self.max_log_lines = 200
        self.summary_window = 50

    def latest_summary(self) -> dict:
        data = read_json(self.summary_path, {})
        return data if isinstance(data, dict) else {}

    def _percentile(self, values: list[float], q: float) -> float | None:
        if not values:
            return None
        if len(values) == 1:
            return float(values[0])
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ordered[lo])
        weight = pos - lo
        return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    def _clip_prob(self, value: Any) -> float | None:
        num = self._safe_float(value)
        if num is None:
            return None
        return max(0.0, min(1.0, float(num)))

    def _row_model_output(self, row: dict) -> float | None:
        return (
            self._clip_prob(row.get("prob_2_model"))
            if self._clip_prob(row.get("prob_2_model")) is not None
            else self._clip_prob(row.get("live_score"))
            if self._clip_prob(row.get("live_score")) is not None
            else self._clip_prob(row.get("pre_policy_score"))
        )

    def _distribution_stats(self, values: list[float]) -> dict:
        if not values:
            return {
                "min": None,
                "p10": None,
                "p25": None,
                "median": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return {
            "min": round(min(values), 6),
            "p10": round(self._percentile(values, 0.10) or 0.0, 6),
            "p25": round(self._percentile(values, 0.25) or 0.0, 6),
            "median": round(self._percentile(values, 0.50) or 0.0, 6),
            "p75": round(self._percentile(values, 0.75) or 0.0, 6),
            "p90": round(self._percentile(values, 0.90) or 0.0, 6),
            "p95": round(self._percentile(values, 0.95) or 0.0, 6),
            "max": round(max(values), 6),
            "mean": round(mean, 6),
            "std": round(math.sqrt(variance), 6),
        }

    def _tail_count(self, values: Iterable[float], threshold: float) -> int:
        return sum(1 for value in values if value >= threshold)

    def _tail_symbols(self, rows: list[dict], threshold: float) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()
        ordered = sorted(
            [row for row in rows if (self._row_model_output(row) or -1.0) >= threshold],
            key=lambda row: (self._row_model_output(row) or -1.0),
            reverse=True,
        )
        for row in ordered:
            symbol = str(row.get("symbol") or "").strip()
            if symbol and symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
        return symbols

    def _append_log(self, snapshot: dict) -> None:
        ensure_dir(self.log_path.parent)
        lines: list[str] = []
        if self.log_path.exists():
            try:
                lines = [line for line in self.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except Exception:
                lines = []
        lines.append(json.dumps(snapshot, sort_keys=False, default=str))
        if len(lines) > self.max_log_lines:
            lines = lines[-self.max_log_lines :]
        self.log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _load_recent_snapshots(self, limit: int | None = None) -> list[dict]:
        if not self.log_path.exists():
            return []
        try:
            lines = [line for line in self.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            return []
        rows: list[dict] = []
        for line in lines[-(limit or self.summary_window) :]:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _headline(self, avg_ge_045: float, *, objective_contract: dict | None = None) -> str:
        objective_contract = dict(objective_contract or {})
        if objective_contract.get("available"):
            if avg_ge_045 >= 2.0:
                return "Model is producing useful upper-tail density"
            if avg_ge_045 >= 0.5:
                return "Upper tail still appears intermittently in live scans, but replay-backed baseline says ranking is usable and semantics remain the leading blocker"
            return "Live upper tail is sparse in this window, but replay-backed baseline says ranking is usable; semantics remain the leading blocker"
        if avg_ge_045 >= 2.0:
            return "Model is producing useful upper-tail density"
        if avg_ge_045 >= 0.5:
            return "Model reaches upper tail occasionally but not reliably"
        return "Model is upper-tail starved — Stage 2 compression is the leading hypothesis"

    def _rolling_summary(self, snapshots: list[dict]) -> dict:
        if not snapshots:
            return {
                "available": False,
                "app_version": APP_VERSION,
                "generated_at_utc": _utc_now_iso(),
                "scans_in_window": 0,
                "headline": "No model output distribution scans captured yet",
            }
        candidate_counts = [int(s.get("stage2_candidate_count") or 0) for s in snapshots]
        medians = [self._safe_float((s.get("distribution") or {}).get("median")) for s in snapshots]
        medians = [v for v in medians if v is not None]
        p90s = [self._safe_float((s.get("distribution") or {}).get("p90")) for s in snapshots]
        p90s = [v for v in p90s if v is not None]
        maxes = [self._safe_float((s.get("distribution") or {}).get("max")) for s in snapshots]
        maxes = [v for v in maxes if v is not None]
        avg_ge_035 = sum(int((s.get("upper_tail_counts") or {}).get("ge_0.35") or 0) for s in snapshots) / len(snapshots)
        avg_ge_040 = sum(int((s.get("upper_tail_counts") or {}).get("ge_0.40") or 0) for s in snapshots) / len(snapshots)
        avg_ge_045 = sum(int((s.get("upper_tail_counts") or {}).get("ge_0.45") or 0) for s in snapshots) / len(snapshots)
        avg_ge_050 = sum(int((s.get("upper_tail_counts") or {}).get("ge_0.50") or 0) for s in snapshots) / len(snapshots)
        avg_ge_055 = sum(int((s.get("upper_tail_counts") or {}).get("ge_0.55") or 0) for s in snapshots) / len(snapshots)
        avg_ge_060 = sum(int((s.get("upper_tail_counts") or {}).get("ge_0.60") or 0) for s in snapshots) / len(snapshots)
        zero_ge_045 = sum(1 for s in snapshots if int((s.get("upper_tail_counts") or {}).get("ge_0.45") or 0) == 0) / len(snapshots)
        regime_distribution: dict[str, int] = {}
        for snapshot in snapshots:
            regime = str(snapshot.get("market_regime_state") or "unknown")
            regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
        objective_contract = load_objective_semantics_contract(
            self.config.model_dir,
            live_threshold=float(getattr(self.config, "live_raw_threshold", 0.0) or 0.0),
            stage1_selection_mode=str(getattr(self.config, "stage1_selection_mode", "") or ""),
        )
        summary = {
            "available": True,
            "app_version": APP_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "scans_in_window": len(snapshots),
            "average_stage2_candidate_count": round(sum(candidate_counts) / len(candidate_counts), 2),
            "median_stage2_candidate_count": self._percentile([float(v) for v in candidate_counts], 0.50),
            "average_distribution_median": round(sum(medians) / len(medians), 6) if medians else None,
            "median_distribution_median": round(self._percentile(medians, 0.50) or 0.0, 6) if medians else None,
            "average_distribution_p90": round(sum(p90s) / len(p90s), 6) if p90s else None,
            "median_distribution_p90": round(self._percentile(p90s, 0.50) or 0.0, 6) if p90s else None,
            "average_distribution_max": round(sum(maxes) / len(maxes), 6) if maxes else None,
            "median_distribution_max": round(self._percentile(maxes, 0.50) or 0.0, 6) if maxes else None,
            "average_upper_tail_counts_per_scan": {
                "ge_0.35": round(avg_ge_035, 3),
                "ge_0.40": round(avg_ge_040, 3),
                "ge_0.45": round(avg_ge_045, 3),
                "ge_0.50": round(avg_ge_050, 3),
                "ge_0.55": round(avg_ge_055, 3),
                "ge_0.60": round(avg_ge_060, 3),
            },
            "fraction_of_scans_with_zero_ge_0.45_rows": round(zero_ge_045, 4),
            "fraction_of_scans_with_zero_ge_0.50_rows": round(
                sum(1 for s in snapshots if int((s.get("upper_tail_counts") or {}).get("ge_0.50") or 0) == 0) / len(snapshots),
                4,
            ),
            "max_score_seen_in_window": round(max(maxes), 6) if maxes else None,
            "regime_distribution": regime_distribution,
            "headline": self._headline(avg_ge_045, objective_contract=objective_contract),
            "objective_semantics_contract": objective_contract if isinstance(objective_contract, dict) else {},
            "latest_snapshot": snapshots[-1],
        }
        return summary

    def record_scan(
        self,
        *,
        status: dict,
        visible_rows: list[dict],
        suppressed_rows: list[dict],
        informational_rows: list[dict],
        overflow_rows: list[dict],
        trigger_source: str,
        review_run_id: str | None = None,
    ) -> dict:
        visible_rows = list(visible_rows or [])
        hidden_rows = list(suppressed_rows or []) + list(informational_rows or []) + list(overflow_rows or [])
        all_rows = visible_rows + hidden_rows
        values = [self._row_model_output(row) for row in all_rows]
        values = [value for value in values if value is not None]
        stage2_candidate_count = len(values)
        scan = status.get("scan") or {}
        market_regime = status.get("market_regime") or {}
        model_pt2 = ((status.get("model") or {}).get("pt2") or {})
        state_scope_key = str((status.get("decision_branch_automation") or {}).get("state_scope_key") or (status.get("decision_checkpoint") or {}).get("state_scope_key") or "")
        generated_at = str(status.get("scan_result_generated_at_utc") or scan.get("finished_at_utc") or _utc_now_iso())
        scan_id = review_run_id or str(scan.get("finished_at_utc") or generated_at)
        snapshot = {
            "scan_id": scan_id,
            "run_id": review_run_id,
            "generated_at_utc": generated_at,
            "app_version": APP_VERSION,
            "state_scope_key": state_scope_key,
            "trigger_source": trigger_source,
            "stage2_candidate_count": stage2_candidate_count,
            "model_source": "trained" if bool(model_pt2.get("trained")) else "heuristic",
            "distribution": self._distribution_stats(values),
            "upper_tail_counts": {
                "ge_0.35": self._tail_count(values, 0.35),
                "ge_0.40": self._tail_count(values, 0.40),
                "ge_0.45": self._tail_count(values, 0.45),
                "ge_0.50": self._tail_count(values, 0.50),
                "ge_0.55": self._tail_count(values, 0.55),
                "ge_0.60": self._tail_count(values, 0.60),
            },
            "upper_tail_symbols": {
                "ge_0.45": self._tail_symbols(all_rows, 0.45),
                "ge_0.50": self._tail_symbols(all_rows, 0.50),
            },
            "visible_count": len(visible_rows),
            "hidden_count": len(hidden_rows),
            "live_raw_threshold": self._safe_float((status.get("decision_checkpoint") or {}).get("live_raw_threshold")) or self._safe_float((status.get("decision_branch_automation") or {}).get("effective_live_raw_threshold")) or self._safe_float(status.get("effective_live_raw_threshold")) or self._safe_float(getattr(self.config, "live_raw_threshold", 0.35)) or 0.35,
            "market_regime_state": str(market_regime.get("state") or "unknown"),
            "btc_regime": status.get("regime_context"),
        }
        atomic_write_json(self.snapshot_dir / f"{str(scan_id).replace(':', '-')}.json", snapshot)
        self._append_log(snapshot)
        summary = self._rolling_summary(self._load_recent_snapshots(limit=self.summary_window))
        atomic_write_json(self.summary_path, summary)
        return summary
