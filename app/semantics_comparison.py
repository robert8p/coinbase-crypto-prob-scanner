from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import AppConfig
from .modeling import ModelBundle
from .objective_semantics import load_objective_semantics_contract
from .persist import atomic_write_json, ensure_dir, read_json
from .review_runs import ReviewPackService
from .replay import HistoricalReplayService
from .universe import UniverseBuilder
from .version import APP_VERSION


VISIBLE_OBJECTIVE_BANDS = {"confirmed_shortlist", "strong_edge", "priority_edge", "elite_edge"}
STRONGER_OBJECTIVE_BANDS = {"strong_edge", "priority_edge", "elite_edge"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _i(value: Any, default: int | None = None) -> int | None:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except Exception:
        return default


def _rate(rows: Iterable[dict], field: str) -> float | None:
    rows = list(rows or [])
    if not rows:
        return None
    vals = [int(r.get(field) or 0) for r in rows]
    return round(sum(vals) / len(vals), 6)


def _avg(rows: Iterable[dict], field: str) -> float | None:
    vals: list[float] = []
    for row in list(rows or []):
        value = _f(row.get(field))
        if value is not None:
            vals.append(float(value))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _bucket_summary(rows: Iterable[dict]) -> dict:
    rows = list(rows or [])
    return {
        "count": len(rows),
        "quality_hit_rate": _rate(rows, "quality_touched"),
        "raw_hit_rate": _rate(rows, "raw_touched"),
        "avg_end_ret": _avg(rows, "end_ret"),
        "avg_mae": _avg(rows, "mae"),
        "avg_mfe": _avg(rows, "mfe"),
        "avg_time_to_touch_minutes": _avg(rows, "time_to_touch_minutes"),
    }


class SemanticsComparisonService:
    def __init__(self, config: AppConfig, replay: HistoricalReplayService, review_packs: ReviewPackService):
        self.config = config
        self.replay = replay
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / "semantics_comparison")
        self.summary_path = self.root_dir / "latest_semantics_comparison_summary.json"
        self.pack_path = self.root_dir / "latest_semantics_comparison_pack.zip"

    def latest_summary(self) -> dict:
        payload = read_json(self.summary_path, {})
        if isinstance(payload, dict) and payload:
            return payload
        return {
            "available": False,
            "app_version": APP_VERSION,
            "headline": "No semantics comparison has been run yet.",
            "summary": "Run the offline semantics comparison to compare the current 0.35 path, the contract-aligned recalibration path, and the 0.28 widening reference path on one shared replay frame.",
        }

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def run(
        self,
        *,
        hours: int = 168,
        step_minutes: int = 120,
        max_scans: int = 84,
        max_symbols: int = 100,
    ) -> dict:
        shared = self._build_shared_frame(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
        )
        current_threshold = round(float(getattr(self.config, "live_raw_threshold", 0.35) or 0.35), 4)
        widening_threshold = 0.28
        contract = load_objective_semantics_contract(
            self.config.model_dir,
            live_threshold=current_threshold,
            stage1_selection_mode=getattr(self.config, "stage1_selection_mode", None),
        ) or {}

        current_window = self.replay._execute_replay_window(  # noqa: SLF001 - deliberate internal reuse for like-for-like frame
            timestamps=shared["timestamps"],
            selected_for_fetch=shared["selected_for_fetch"],
            universe=shared["universe"],
            bundle=shared["bundle"],
            histories=shared["histories"],
            pipeline_mode="raw_threshold",
            raw_threshold=current_threshold,
            stage1_selection_mode_override=None,
            stage1_max_candidates_override=None,
            capture_full_rankable_rows=False,
        )

        widening_window = self.replay._execute_replay_window(  # noqa: SLF001 - deliberate internal reuse for like-for-like frame
            timestamps=shared["timestamps"],
            selected_for_fetch=shared["selected_for_fetch"],
            universe=shared["universe"],
            bundle=shared["bundle"],
            histories=shared["histories"],
            pipeline_mode="raw_threshold",
            raw_threshold=widening_threshold,
            stage1_selection_mode_override=None,
            stage1_max_candidates_override=None,
            capture_full_rankable_rows=False,
        )

        current_rows = list((current_window or {}).get("replay_rows") or [])
        widening_rows = list((widening_window or {}).get("replay_rows") or [])

        current_path_rows = self._normalize_replay_rows(current_rows, path_name="current_035_path")
        widening_path_rows = self._normalize_replay_rows(widening_rows, path_name="widening_028_reference_path")
        contract_path_rows = self._apply_contract_path(
            widening_rows=widening_rows,
            path_name="recalibrated_contract_path",
            contract=contract,
        )

        base_rows = widening_path_rows
        base_quality_rate = _rate(base_rows, "quality_touched")
        current_summary = self._build_path_summary(
            path_name="current_035_path",
            label="Current 0.35 path",
            rows=current_path_rows,
            base_quality_rate=base_quality_rate,
            contract=contract,
            path_parameters={
                "pipeline_mode": "raw_threshold",
                "raw_threshold": current_threshold,
                "stage1_selection_mode": getattr(self.config, "stage1_selection_mode", None),
                "live_selection_mode": getattr(self.config, "live_selection_mode", None),
                "selection_semantics": "exact current raw-threshold + legacy visible shortlist behavior",
            },
        )
        contract_summary = self._build_path_summary(
            path_name="recalibrated_contract_path",
            label="Recalibrated contract-aligned path",
            rows=contract_path_rows,
            base_quality_rate=base_quality_rate,
            contract=contract,
            path_parameters={
                "selection_semantics": "strong-edge-or-better objective bands, ranking preserved, top-5 cap, one near-strong fallback when no strong-edge row exists",
                "source_contract": "raw_score_baseline objective semantics",
                "ranking_preserved": True,
                "top_cap": min(5, int(getattr(self.config, "stage2_decision_focus_top_n", 5) or 5), int(getattr(self.config, "utility_shortlist_target_max_names", 8) or 8)),
                "strong_edge_floor": _f(contract.get("strong_edge_floor")),
                "confirmed_shortlist_floor": _f(contract.get("confirmed_shortlist_floor"), current_threshold),
            },
        )
        widening_summary = self._build_path_summary(
            path_name="widening_028_reference_path",
            label="0.28 widening reference path",
            rows=widening_path_rows,
            base_quality_rate=base_quality_rate,
            contract=contract,
            path_parameters={
                "pipeline_mode": "raw_threshold",
                "raw_threshold": widening_threshold,
                "stage1_selection_mode": getattr(self.config, "stage1_selection_mode", None),
                "live_selection_mode": getattr(self.config, "live_selection_mode", None),
                "selection_semantics": "raw-threshold widening reference only; not a live recommendation in this tranche",
            },
        )

        comparison_rows = [current_summary, contract_summary, widening_summary]
        best_path = self._choose_best_path(comparison_rows)
        headline, summary_text, obvious_effects = self._headline_and_effects(
            current_summary=current_summary,
            contract_summary=contract_summary,
            widening_summary=widening_summary,
            best_path=best_path,
        )
        code_truth_note = self._code_truth_note(current_threshold=current_threshold)
        scope_note = self._scope_note()

        summary = {
            "available": True,
            "app_version": APP_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "headline": headline,
            "summary": summary_text,
            "tranche_title": "Offline/replay score-semantics recalibration around the existing ranking",
            "objective": "Surface a small, trustworthy visible shortlist that beats the hidden remainder for a quality +2.0% move within 240 minutes.",
            "replay_frame": {
                "hours": int(hours),
                "step_minutes": int(step_minutes),
                "max_scans": int(max_scans),
                "max_symbols": int(max_symbols),
                "scan_count": len(shared["timestamps"]),
                "symbol_count": len(shared["selected_symbols"]),
                "start_utc": shared["timestamps"][0].isoformat() if shared["timestamps"] else None,
                "end_utc": shared["timestamps"][-1].isoformat() if shared["timestamps"] else None,
                "model_bundle_path": shared["bundle_path"],
                "model_bundle_label": shared["bundle_label"],
                "shared_frame_guarantee": "All three paths use the same replay timestamps, symbol cohort, histories, model bundle, Stage 1 mode, and raw ranking source before their final visible-selection semantics diverge.",
            },
            "objective_semantics_contract": contract,
            "paths": {
                "current_035_path": current_summary,
                "recalibrated_contract_path": contract_summary,
                "widening_028_reference_path": widening_summary,
            },
            "comparison_table": [self._compact_comparison_row(row) for row in comparison_rows],
            "best_path_now": best_path,
            "obvious_effects": obvious_effects,
            "code_truth_note": code_truth_note,
            "scope_note": scope_note,
            "download_paths": {
                "summary_json": "/api/semantics-comparison/latest-summary",
                "summary_txt": "/api/semantics-comparison/latest-summary.txt",
                "pack_zip": "/api/semantics-comparison/latest-pack.zip",
            },
            "next_evidence_to_return": [
                "the semantics comparison pack zip",
                "the semantics comparison summary",
                "ledger_input_pack.zip",
                "the next post-maturity current-version bundle once this deployed version has resolved rows",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        self._build_pack(
            summary=summary,
            current_rows=current_path_rows,
            contract_rows=contract_path_rows,
            widening_rows=widening_path_rows,
        )
        return {
            "ok": True,
            "summary": summary,
            "download_path": "/api/semantics-comparison/latest-pack.zip",
            "summary_path": "/api/semantics-comparison/latest-summary",
        }

    def _build_shared_frame(self, *, hours: int, step_minutes: int, max_scans: int, max_symbols: int) -> dict:
        end_dt = self.replay._align_5m(self.replay._parse_utc(None) or datetime.now(timezone.utc))  # noqa: SLF001
        start_dt = self.replay._align_5m(end_dt - timedelta(hours=max(1, int(hours or 168))))  # noqa: SLF001
        timestamps = self.replay._build_timestamps(start_dt, end_dt, step_minutes=max(5, int(step_minutes or 120)), max_scans=max(1, int(max_scans or 84)))  # noqa: SLF001
        bundle_path = str(getattr(self.config, "model_path_pt2", "") or "").strip()
        bundle = ModelBundle.load(bundle_path)
        if bundle is None:
            raise FileNotFoundError(f"trained model bundle not found: {bundle_path or getattr(self.config, 'model_path_pt2', '')}")
        bundle_label = str(Path(bundle_path).name or "pt2")

        products = self.replay.client.list_products()
        currencies = self.replay.client.list_currencies()
        volume_map = self.replay.client.get_volume_summary()
        locked_symbols = self.replay.scanner._locked_live_cohort()  # noqa: SLF001
        universe = UniverseBuilder(self.config).build(
            products,
            currencies,
            volume_map,
            locked_symbols=locked_symbols,
            selection_label=self.replay.scanner._selection_label(locked_symbols),  # noqa: SLF001
        )
        selected_for_fetch = list(universe.selected_for_fetch)
        if max_symbols > 0:
            selected_for_fetch = selected_for_fetch[: max(1, int(max_symbols))]
        selected_symbols = [str(p.get("id") or "") for p in selected_for_fetch if str(p.get("id") or "")]
        if not selected_symbols:
            raise ValueError("no selected symbols available for semantics comparison")

        warmup_bars = max(int(getattr(self.config, "stage1_light_calendar_5m_bars", 864) or 864), int(getattr(self.config, "stage2_lookback_5m_bars", 2400) or 2400))
        horizon_bars = max(1, int(getattr(self.config, "candles_per_horizon", 48) or 48))
        history_start = timestamps[0] - timedelta(minutes=5 * max(1, warmup_bars - 1))
        history_end = timestamps[-1] + timedelta(minutes=5 * horizon_bars)
        prefetch_symbols = sorted(set(selected_symbols + ["BTC-USD", "ETH-USD"]))
        histories = self.replay._prefetch_histories(prefetch_symbols, history_start, history_end)  # noqa: SLF001
        return {
            "timestamps": timestamps,
            "bundle": bundle,
            "bundle_path": bundle_path,
            "bundle_label": bundle_label,
            "selected_for_fetch": selected_for_fetch,
            "selected_symbols": selected_symbols,
            "universe": universe,
            "histories": histories,
        }

    def _normalize_replay_rows(self, rows: list[dict], *, path_name: str) -> list[dict]:
        grouped = self._group_by_scan(rows)
        out: list[dict] = []
        for scan_rows in grouped.values():
            ordered = self._ordered_rows(scan_rows)
            visible_rank = 0
            for row in ordered:
                item = dict(row)
                is_visible = str(item.get("row_type") or "") == "visible"
                item["comparison_path_name"] = path_name
                item["comparison_path_label"] = path_name
                item["comparison_row_type"] = "visible" if is_visible else "hidden"
                if is_visible:
                    visible_rank += 1
                    item["comparison_visible_rank"] = visible_rank
                    item["comparison_selection_reason"] = "current_path_visible" if path_name == "current_035_path" else "widening_reference_visible"
                else:
                    item["comparison_visible_rank"] = None
                    item["comparison_selection_reason"] = str(item.get("suppression_reason") or item.get("row_type") or "hidden")
                out.append(item)
        return out

    def _apply_contract_path(self, *, widening_rows: list[dict], path_name: str, contract: dict) -> list[dict]:
        grouped = self._group_by_scan(widening_rows)
        out: list[dict] = []
        for scan_rows in grouped.values():
            ordered = self._ordered_rows(scan_rows)
            selected = self._select_contract_rows(ordered, contract=contract)
            selected_ids = {id(row): reason for row, reason in selected}
            visible_rank = 0
            for row in ordered:
                item = dict(row)
                reason = selected_ids.get(id(row))
                is_visible = reason is not None
                item["comparison_path_name"] = path_name
                item["comparison_path_label"] = path_name
                item["comparison_row_type"] = "visible" if is_visible else "hidden"
                if is_visible:
                    visible_rank += 1
                    item["comparison_visible_rank"] = visible_rank
                    item["comparison_selection_reason"] = reason
                else:
                    band = str(item.get("objective_score_band") or "")
                    if band in STRONGER_OBJECTIVE_BANDS:
                        item["comparison_selection_reason"] = "beyond_contract_top_cap"
                    elif band in VISIBLE_OBJECTIVE_BANDS:
                        item["comparison_selection_reason"] = "confirmed_but_not_contract_selected"
                    else:
                        item["comparison_selection_reason"] = "below_contract_semantics"
                out.append(item)
        return out

    def _select_contract_rows(self, ordered: list[dict], *, contract: dict) -> list[tuple[dict, str]]:
        strong_floor = _f(contract.get("strong_edge_floor"))
        confirmed_floor = _f(contract.get("confirmed_shortlist_floor"), _f(getattr(self.config, "live_raw_threshold", 0.35), 0.35)) or 0.35
        top_cap = min(
            5,
            max(1, int(getattr(self.config, "stage2_decision_focus_top_n", 5) or 5)),
            max(1, int(getattr(self.config, "utility_shortlist_target_max_names", 8) or 8)),
        )
        strong_rows: list[tuple[dict, str]] = []
        for row in ordered:
            band = str(row.get("objective_score_band") or "")
            live_score = _f(row.get("live_score"), 0.0) or 0.0
            if band in STRONGER_OBJECTIVE_BANDS:
                strong_rows.append((row, f"contract_{band}"))
                continue
            if strong_floor is not None and live_score >= strong_floor:
                strong_rows.append((row, "contract_strong_floor"))
        if strong_rows:
            return strong_rows[:top_cap]

        fallback_gap = 0.015
        fallback: list[tuple[dict, str]] = []
        for row in ordered:
            band = str(row.get("objective_score_band") or "")
            live_score = _f(row.get("live_score"), 0.0) or 0.0
            if band not in VISIBLE_OBJECTIVE_BANDS:
                continue
            if live_score < confirmed_floor:
                continue
            if strong_floor is not None and live_score < strong_floor - fallback_gap:
                continue
            fallback.append((row, "contract_near_strong_fallback"))
            break
        return fallback

    def _group_by_scan(self, rows: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in list(rows or []):
            grouped[str(row.get("as_of_utc") or "unknown")].append(row)
        return dict(sorted(grouped.items(), key=lambda item: item[0]))

    def _ordered_rows(self, rows: list[dict]) -> list[dict]:
        return sorted(
            list(rows or []),
            key=lambda row: (
                _i(row.get("candidate_rank_all"), 10**9),
                _i(row.get("pre_policy_rank"), 10**9),
                -(_f(row.get("live_score"), 0.0) or 0.0),
                str(row.get("symbol") or ""),
            ),
        )

    def _build_path_summary(self, *, path_name: str, label: str, rows: list[dict], base_quality_rate: float | None, contract: dict, path_parameters: dict) -> dict:
        grouped = self._group_by_scan(rows)
        visible = [r for r in rows if str(r.get("comparison_row_type") or "") == "visible"]
        hidden = [r for r in rows if str(r.get("comparison_row_type") or "") != "visible"]
        visible_bucket = _bucket_summary(visible)
        hidden_bucket = _bucket_summary(hidden)
        visible_quality = _f(visible_bucket.get("quality_hit_rate"), 0.0) or 0.0
        hidden_quality = _f(hidden_bucket.get("quality_hit_rate"), 0.0) or 0.0
        scan_topk = self._scan_topk_summary(grouped=grouped, base_quality_rate=base_quality_rate)
        shortlist_distribution = self._shortlist_size_distribution(grouped)
        score_band_distribution = self._score_band_distribution(rows)
        symbol_concentration = self._symbol_concentration(visible)
        visible_quality_gap = round(visible_quality - hidden_quality, 6) if visible or hidden else None
        return {
            "path_name": path_name,
            "path_label": label,
            "path_parameters": path_parameters,
            "scan_count": len(grouped),
            "row_count": len(rows),
            "visible": visible_bucket,
            "hidden": hidden_bucket,
            "visible_vs_hidden_quality_gap": visible_quality_gap,
            "visible_vs_hidden_raw_gap": round((_f(visible_bucket.get("raw_hit_rate"), 0.0) or 0.0) - (_f(hidden_bucket.get("raw_hit_rate"), 0.0) or 0.0), 6) if visible or hidden else None,
            "topk_quality": scan_topk,
            "shortlist_size_distribution": shortlist_distribution,
            "score_band_distribution": score_band_distribution,
            "symbol_concentration": symbol_concentration,
            "contract_context": {
                "confirmed_shortlist_floor": _f(contract.get("confirmed_shortlist_floor")),
                "strong_edge_floor": _f(contract.get("strong_edge_floor")),
                "priority_edge_floor": _f(contract.get("priority_edge_floor")),
                "elite_edge_floor": _f(contract.get("elite_edge_floor")),
            },
        }

    def _scan_topk_summary(self, *, grouped: dict[str, list[dict]], base_quality_rate: float | None) -> dict:
        out: dict[str, dict] = {}
        scan_count = max(1, len(grouped))
        for k in (1, 3, 5):
            per_scan_rates: list[float] = []
            hit_scans = 0
            for rows in grouped.values():
                visible = [r for r in self._ordered_rows(rows) if str(r.get("comparison_row_type") or "") == "visible"]
                subset = visible[:k]
                if not subset:
                    per_scan_rates.append(0.0)
                    continue
                rate = sum(int(r.get("quality_touched") or 0) for r in subset) / len(subset)
                per_scan_rates.append(rate)
                if any(int(r.get("quality_touched") or 0) == 1 for r in subset):
                    hit_scans += 1
            mean_rate = sum(per_scan_rates) / scan_count
            out[f"top_{k}"] = {
                "k": k,
                "scan_count": scan_count,
                "mean_quality_rate": round(mean_rate, 6),
                "lift_vs_base": round(mean_rate / base_quality_rate, 6) if base_quality_rate and base_quality_rate > 0 else None,
                "share_of_scans_with_hit": round(hit_scans / scan_count, 6),
            }
        return out

    def _shortlist_size_distribution(self, grouped: dict[str, list[dict]]) -> dict:
        counts = sorted(sum(1 for r in rows if str(r.get("comparison_row_type") or "") == "visible") for rows in grouped.values())
        if not counts:
            return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "max": None, "zero_visible_scan_fraction": None}
        return {
            "count": len(counts),
            "mean": round(sum(counts) / len(counts), 4),
            "min": counts[0],
            "p50": self._percentile(counts, 0.50),
            "p90": self._percentile(counts, 0.90),
            "max": counts[-1],
            "zero_visible_scan_fraction": round(sum(1 for c in counts if c == 0) / len(counts), 6),
        }

    def _score_band_distribution(self, rows: list[dict]) -> dict:
        visible_counter: Counter[str] = Counter()
        hidden_counter: Counter[str] = Counter()
        for row in rows:
            band = str(row.get("objective_score_band") or row.get("score_band") or "unknown")
            if str(row.get("comparison_row_type") or "") == "visible":
                visible_counter[band] += 1
            else:
                hidden_counter[band] += 1
        return {
            "visible": dict(visible_counter),
            "hidden": dict(hidden_counter),
        }

    def _symbol_concentration(self, visible_rows: list[dict]) -> dict:
        counts = Counter(str(r.get("symbol") or "") for r in visible_rows if str(r.get("symbol") or ""))
        total = sum(counts.values())
        if total == 0:
            return {
                "visible_symbol_count": 0,
                "top_symbols": [],
                "top_symbol_share": None,
                "symbol_hhi": None,
            }
        top_symbols = [
            {"symbol": symbol, "visible_count": int(count), "share": round(count / total, 6)}
            for symbol, count in counts.most_common(10)
        ]
        hhi = sum((count / total) ** 2 for count in counts.values())
        return {
            "visible_symbol_count": len(counts),
            "top_symbols": top_symbols,
            "top_symbol_share": round(top_symbols[0]["share"], 6) if top_symbols else None,
            "symbol_hhi": round(hhi, 6),
        }

    def _choose_best_path(self, summaries: list[dict]) -> dict:
        def _score(item: dict) -> tuple:
            visible_quality = _f(((item.get("visible") or {}).get("quality_hit_rate")), 0.0) or 0.0
            quality_gap = _f(item.get("visible_vs_hidden_quality_gap"), -999.0) or -999.0
            top3 = _f((((item.get("topk_quality") or {}).get("top_3") or {}).get("mean_quality_rate")), 0.0) or 0.0
            visible_count = _i(((item.get("visible") or {}).get("count")), 0) or 0
            return (quality_gap, visible_quality, top3, -visible_count)
        winner = max(summaries, key=_score)
        return {
            "path_name": winner.get("path_name"),
            "path_label": winner.get("path_label"),
            "reason": "Highest visible-vs-hidden quality gap, then highest visible quality rate, then strongest top-3 quality.",
        }

    def _headline_and_effects(self, *, current_summary: dict, contract_summary: dict, widening_summary: dict, best_path: dict) -> tuple[str, str, list[str]]:
        effects: list[str] = []
        current_visible_count = _i((current_summary.get("visible") or {}).get("count"), 0) or 0
        widening_visible_count = _i((widening_summary.get("visible") or {}).get("count"), 0) or 0
        current_visible_quality = _f((current_summary.get("visible") or {}).get("quality_hit_rate"), 0.0) or 0.0
        widening_visible_quality = _f((widening_summary.get("visible") or {}).get("quality_hit_rate"), 0.0) or 0.0
        contract_visible_quality = _f((contract_summary.get("visible") or {}).get("quality_hit_rate"), 0.0) or 0.0
        contract_visible_count = _i((contract_summary.get("visible") or {}).get("count"), 0) or 0
        if widening_visible_count > current_visible_count and widening_visible_quality < current_visible_quality:
            effects.append(
                f"The 0.28 widening reference is wider but weaker: visible rows rise from {current_visible_count} to {widening_visible_count} while visible quality hit rate drops from {current_visible_quality:.3f} to {widening_visible_quality:.3f}."
            )
        if contract_visible_quality > current_visible_quality and contract_visible_count <= current_visible_count:
            effects.append(
                f"The contract-aligned path is tighter but stronger than the current 0.35 path: visible rows move from {current_visible_count} to {contract_visible_count} while visible quality hit rate improves from {current_visible_quality:.3f} to {contract_visible_quality:.3f}."
            )
        if not effects:
            effects.append("No path dominates on every metric yet; use the pack to judge quality edge, shortlist width, and top-k utility together.")

        best_name = str(best_path.get("path_name") or "")
        if best_name == "recalibrated_contract_path":
            headline = "Contract-aligned semantics currently look stronger than the exact 0.35 path without relying on 0.28 widening"
        elif best_name == "current_035_path":
            headline = "The exact current 0.35 path still leads on this replay frame; widening and semantics tightening did not beat it cleanly"
        else:
            headline = "The 0.28 widening reference leads on this replay frame, but check shortlist-noise trade-offs before any live change"
        summary_text = (
            f"Compared on one shared replay frame: current 0.35 path visible quality={current_visible_quality:.3f}, "
            f"contract-aligned visible quality={contract_visible_quality:.3f}, "
            f"0.28 widening visible quality={widening_visible_quality:.3f}. Best path by quality edge right now: {best_path.get('path_label') or best_name}."
        )
        return headline, summary_text, effects

    def _code_truth_note(self, *, current_threshold: float) -> dict:
        return {
            "headline": "Stage 1 mode and live selection mode are different layers in the live path",
            "summary": (
                "STAGE1_SELECTION_MODE controls which names pass Stage 1 into Stage 2 scoring. "
                "In the current raw-threshold pipeline, LIVE_RAW_THRESHOLD gates the scored rows first, then LIVE_SELECTION_MODE controls only the final visible-shortlist trimming. "
                "With LIVE_SELECTION_MODE=legacy, the app uses the legacy actionability/watchlist + concentration-cap shortlist logic instead of the utility_constrained shortlist engine."
            ),
            "current_stage1_selection_mode": str(getattr(self.config, "stage1_selection_mode", "") or ""),
            "current_live_selection_mode": str(getattr(self.config, "live_selection_mode", "") or ""),
            "current_live_pipeline_mode": str(getattr(self.config, "live_pipeline_mode", "") or ""),
            "current_live_raw_threshold": round(float(current_threshold), 4),
        }

    def _scope_note(self) -> dict:
        current_version_summary: dict
        try:
            current_version_summary = self.review_packs.get_current_version_summary() or {}
        except Exception:
            current_version_summary = {}
        evidence = dict(current_version_summary.get("evidence") or {})
        visible_rows = _i(evidence.get("visible_rows"), 0) or 0
        resolved_rows = _i(evidence.get("resolved_rows"), 0) or 0
        latest_evaluated_pack = Path(self.config.model_dir) / "review_packs" / "latest_evaluated_pack.zip"
        mismatch = latest_evaluated_pack.exists() and resolved_rows == 0
        if mismatch:
            summary = (
                "A latest evaluated pack can exist while current-version resolved rows stay at zero because the latest evaluated artifact is a global latest file, "
                "while current-version evidence is filtered to the current deployed version/deployment scope. Treat this as a scope filter note unless the current-version bundle later disagrees with its own scoped rows."
            )
        else:
            summary = "No current evaluated-scope mismatch is visible right now."
        return {
            "mismatch_detected": mismatch,
            "current_version_visible_rows": visible_rows,
            "current_version_resolved_rows": resolved_rows,
            "latest_evaluated_pack_exists": latest_evaluated_pack.exists(),
            "summary": summary,
        }

    def _compact_comparison_row(self, row: dict) -> dict:
        return {
            "path_name": row.get("path_name"),
            "path_label": row.get("path_label"),
            "visible_count": (row.get("visible") or {}).get("count"),
            "hidden_count": (row.get("hidden") or {}).get("count"),
            "visible_quality_hit_rate": (row.get("visible") or {}).get("quality_hit_rate"),
            "hidden_quality_hit_rate": (row.get("hidden") or {}).get("quality_hit_rate"),
            "visible_raw_hit_rate": (row.get("visible") or {}).get("raw_hit_rate"),
            "hidden_raw_hit_rate": (row.get("hidden") or {}).get("raw_hit_rate"),
            "visible_vs_hidden_quality_gap": row.get("visible_vs_hidden_quality_gap"),
            "top_1_quality_rate": (((row.get("topk_quality") or {}).get("top_1") or {}).get("mean_quality_rate")),
            "top_3_quality_rate": (((row.get("topk_quality") or {}).get("top_3") or {}).get("mean_quality_rate")),
            "top_5_quality_rate": (((row.get("topk_quality") or {}).get("top_5") or {}).get("mean_quality_rate")),
            "mean_shortlist_size": ((row.get("shortlist_size_distribution") or {}).get("mean")),
            "zero_visible_scan_fraction": ((row.get("shortlist_size_distribution") or {}).get("zero_visible_scan_fraction")),
        }

    def _summary_text(self, summary: dict) -> str:
        lines = [
            f"Coinbase Crypto Prob Scanner v{APP_VERSION}",
            "",
            str(summary.get("headline") or ""),
            str(summary.get("summary") or ""),
            "",
            "Best path now:",
            f"- {((summary.get('best_path_now') or {}).get('path_label')) or '-'}",
            f"- Reason: {((summary.get('best_path_now') or {}).get('reason')) or '-'}",
            "",
            "Comparison table:",
        ]
        for row in list(summary.get("comparison_table") or []):
            lines.append(
                f"- {row.get('path_label')}: visible_q={row.get('visible_quality_hit_rate')} hidden_q={row.get('hidden_quality_hit_rate')} visible_rows={row.get('visible_count')} mean_shortlist={row.get('mean_shortlist_size')}"
            )
        lines.extend([
            "",
            "Obvious effects:",
        ])
        for item in list(summary.get("obvious_effects") or []):
            lines.append(f"- {item}")
        lines.extend([
            "",
            "Code-truth note:",
            f"- {((summary.get('code_truth_note') or {}).get('summary')) or '-'}",
            "",
            "Scope note:",
            f"- {((summary.get('scope_note') or {}).get('summary')) or '-'}",
        ])
        return "\n".join(lines).strip() + "\n"

    def _build_pack(self, *, summary: dict, current_rows: list[dict], contract_rows: list[dict], widening_rows: list[dict]) -> None:
        summary_txt = self._summary_text(summary)
        current_summary = dict((summary.get("paths") or {}).get("current_035_path") or {})
        contract_summary = dict((summary.get("paths") or {}).get("recalibrated_contract_path") or {})
        widening_summary = dict((summary.get("paths") or {}).get("widening_028_reference_path") or {})
        all_rows = current_rows + contract_rows + widening_rows
        with zipfile.ZipFile(self.pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("semantics_comparison_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr("semantics_comparison_summary.txt", summary_txt)
            zf.writestr("comparison_table.json", json.dumps(summary.get("comparison_table") or [], indent=2, sort_keys=True))
            zf.writestr("current_035_path_summary.json", json.dumps(current_summary, indent=2, sort_keys=True))
            zf.writestr("recalibrated_contract_path_summary.json", json.dumps(contract_summary, indent=2, sort_keys=True))
            zf.writestr("widening_028_reference_path_summary.json", json.dumps(widening_summary, indent=2, sort_keys=True))
            zf.writestr("code_truth_note.json", json.dumps(summary.get("code_truth_note") or {}, indent=2, sort_keys=True))
            zf.writestr("scope_note.json", json.dumps(summary.get("scope_note") or {}, indent=2, sort_keys=True))
            zf.writestr("comparison_rows.csv", self._csv_bytes(all_rows))
            zf.writestr("current_035_visible_rows.csv", self._csv_bytes([r for r in current_rows if str(r.get("comparison_row_type") or "") == "visible"]))
            zf.writestr("recalibrated_contract_visible_rows.csv", self._csv_bytes([r for r in contract_rows if str(r.get("comparison_row_type") or "") == "visible"]))
            zf.writestr("widening_028_visible_rows.csv", self._csv_bytes([r for r in widening_rows if str(r.get("comparison_row_type") or "") == "visible"]))

    def _csv_bytes(self, rows: list[dict]) -> bytes:
        rows = list(rows or [])
        buffer = io.StringIO()
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
        return buffer.getvalue().encode("utf-8")

    def _percentile(self, values: list[int], pct: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        rank = max(0.0, min(1.0, float(pct))) * (len(values) - 1)
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            return float(values[low])
        frac = rank - low
        return round(float(values[low] + (values[high] - values[low]) * frac), 4)
