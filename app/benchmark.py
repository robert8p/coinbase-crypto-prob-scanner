from __future__ import annotations

import csv
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List
from datetime import datetime, timezone

from .config import AppConfig
from .model_audit import ModelAuditService
from .persist import atomic_write_json, read_json
from .replay import HistoricalReplayService
from .review_runs import ReviewPackService


def _rate(row: dict, key: str):
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_replay_evidence(summary: dict) -> dict:
    summary = dict(summary or {})
    surfaced = dict(summary.get("surfaced_evidence") or {})
    visible_bucket = dict(summary.get("visible_bucket") or {})
    non_visible_bucket = dict(summary.get("non_visible_bucket") or {})
    evidence = {
        "visible_quality_hit_rate": surfaced.get("visible_quality_hit_rate"),
        "non_visible_quality_hit_rate": surfaced.get("non_visible_quality_hit_rate"),
        "visible_raw_hit_rate": surfaced.get("visible_raw_hit_rate"),
        "non_visible_raw_hit_rate": surfaced.get("non_visible_raw_hit_rate"),
        "visible_avg_end_ret": surfaced.get("visible_avg_end_ret"),
        "non_visible_avg_end_ret": surfaced.get("non_visible_avg_end_ret"),
        "visible_rows": surfaced.get("visible_rows"),
        "non_visible_rows": surfaced.get("non_visible_rows"),
        "resolved_rows": surfaced.get("resolved_rows"),
    }
    if evidence["visible_quality_hit_rate"] is None:
        evidence["visible_quality_hit_rate"] = visible_bucket.get("quality_hit_rate")
    if evidence["non_visible_quality_hit_rate"] is None:
        evidence["non_visible_quality_hit_rate"] = non_visible_bucket.get("quality_hit_rate")
    if evidence["visible_raw_hit_rate"] is None:
        evidence["visible_raw_hit_rate"] = visible_bucket.get("raw_hit_rate")
    if evidence["non_visible_raw_hit_rate"] is None:
        evidence["non_visible_raw_hit_rate"] = non_visible_bucket.get("raw_hit_rate")
    if evidence["visible_avg_end_ret"] is None:
        evidence["visible_avg_end_ret"] = visible_bucket.get("avg_end_ret")
    if evidence["non_visible_avg_end_ret"] is None:
        evidence["non_visible_avg_end_ret"] = non_visible_bucket.get("avg_end_ret")
    if evidence["visible_rows"] is None:
        evidence["visible_rows"] = visible_bucket.get("total")
    if evidence["non_visible_rows"] is None:
        evidence["non_visible_rows"] = non_visible_bucket.get("total")
    if evidence["resolved_rows"] is None:
        evidence["resolved_rows"] = surfaced.get("resolved_rows")
    return evidence


def _classify_symbol_repeatability_rows(rows: List[dict], *, source: str = "current_version") -> dict:
    rows = list(rows or [])
    winners = []
    disappointments = []
    hidden_outperformers = []
    visible_underperformers = []
    for row in rows:
        visible_rows = int(row.get("visible_rows") or 0)
        hidden_rows = int(row.get("non_visible_rows") or 0)
        q_hit = _rate(row, "quality_hit_rate") or 0.0
        vis_q_hit = _rate(row, "visible_quality_hit_rate")
        hid_q_hit = _rate(row, "non_visible_quality_hit_rate")
        avg_ret = _rate(row, "avg_end_ret") or 0.0
        item = {
            "symbol": row.get("symbol"),
            "resolved_rows": int(row.get("resolved_rows") or 0),
            "visible_rows": visible_rows,
            "non_visible_rows": hidden_rows,
            "quality_hit_rate": q_hit,
            "visible_quality_hit_rate": vis_q_hit,
            "non_visible_quality_hit_rate": hid_q_hit,
            "avg_end_ret": avg_ret,
            "max_live_score": row.get("max_live_score"),
        }
        if visible_rows >= 2 and q_hit >= 0.40 and avg_ret >= 0.0:
            winners.append(item)
        if visible_rows >= 2 and q_hit <= 0.10 and avg_ret <= 0.0:
            disappointments.append(item)
        if hidden_rows >= 2 and (hid_q_hit or 0.0) >= 0.40:
            hidden_outperformers.append(item)
        if visible_rows >= 2 and vis_q_hit is not None and hid_q_hit is not None and vis_q_hit < hid_q_hit:
            visible_underperformers.append(item)
    sort_key = lambda item: ((item.get("quality_hit_rate") or 0.0), (item.get("avg_end_ret") or -999.0), int(item.get("resolved_rows") or 0))
    winners = sorted(winners, key=sort_key, reverse=True)[:12]
    hidden_outperformers = sorted(hidden_outperformers, key=lambda item: ((item.get("non_visible_quality_hit_rate") or 0.0), (item.get("avg_end_ret") or -999.0), int(item.get("non_visible_rows") or 0)), reverse=True)[:12]
    disappointments = sorted(disappointments, key=lambda item: (-(item.get("avg_end_ret") or 999.0), -(item.get("quality_hit_rate") or 999.0), -int(item.get("visible_rows") or 0)))[:12]
    visible_underperformers = sorted(visible_underperformers, key=lambda item: (((item.get("non_visible_quality_hit_rate") or 0.0) - (item.get("visible_quality_hit_rate") or 0.0)), int(item.get("resolved_rows") or 0)), reverse=True)[:12]
    return {
        "available": bool(rows),
        "source": source,
        "headline": "Symbol classifications from evaluated evidence" if rows else "No evaluated symbol evidence yet",
        "summary": "Repeat winners, disappointments, and hidden outperformers are derived from evaluated symbol repeatability so you can spot where the scanner is helping or failing." if rows else "Wait for evaluated packs before classifying symbols.",
        "repeat_winners": winners,
        "repeat_disappointments": disappointments,
        "hidden_outperformers": hidden_outperformers,
        "visible_underperformers": visible_underperformers,
    }

def _csv_text(rows: List[dict], fieldnames: List[str]) -> str:
    buffer = []
    import io
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    return sio.getvalue()


def _atomic_zip_write(path: Path, write_fn) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + '_', suffix='.tmp', dir=str(path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            write_fn(zf)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return path

def _parse_thresholds(value: str | Iterable[float] | None) -> List[float]:
    if value is None:
        return [0.25, 0.30, 0.35, 0.40]
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    else:
        items = list(value)
    out: List[float] = []
    for item in items:
        try:
            threshold = max(0.0, min(1.0, float(item)))
        except Exception:
            continue
        if threshold not in out:
            out.append(threshold)
    return out or [0.25, 0.30, 0.35, 0.40]


class BenchmarkLabService:
    def __init__(self, config: AppConfig, replay: HistoricalReplayService, review_packs: ReviewPackService, model_audit: ModelAuditService):
        self.config = config
        self.replay = replay
        self.review_packs = review_packs
        self.model_audit = model_audit
        self.summary_path = Path(config.model_dir) / "benchmark_lab_summary.json"

    def latest_summary(self) -> dict:
        summary = read_json(self.summary_path, {})
        if not summary:
            return {}
        return self._attach_live_context(summary)

    def build_benchmark_pack(self) -> Path:
        summary = self.latest_summary()
        if not summary or not summary.get("rows"):
            raise FileNotFoundError("no benchmark summary available")
        pack_path = Path(self.config.model_dir) / "benchmark_lab_pack.zip"

        def write_zip(zf):
            rows = list(summary.get("rows") or [])
            recommendation = summary.get("recommendation") or {}
            zf.writestr("benchmark_summary.json", json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr(
                "benchmark_threshold_rows.csv",
                _csv_text(rows, [
                    "threshold", "headline", "visible_quality_hit_rate", "non_visible_quality_hit_rate",
                    "visible_raw_hit_rate", "non_visible_raw_hit_rate", "visible_avg_end_ret",
                    "non_visible_avg_end_ret", "visible_rows", "non_visible_rows", "resolved_rows",
                    "stage1_quality_recall", "missed_quality_opportunities", "top_symbol_at_0_45",
                    "top_symbol_share_at_0_45", "download_path", "summary_path",
                ]),
            )
            zf.writestr("benchmark_recommendation.json", json.dumps(recommendation, indent=2, sort_keys=True))
            zf.writestr("current_version_summary.json", json.dumps(summary.get("live_current_version") or {}, indent=2, sort_keys=True))
            zf.writestr("model_audit_summary.json", json.dumps(summary.get("model_audit") or {}, indent=2, sort_keys=True))
            zf.writestr("symbol_classification.json", json.dumps(summary.get("symbol_classification") or {}, indent=2, sort_keys=True))

        return _atomic_zip_write(pack_path, write_zip)

    def build_symbol_classification_pack(self) -> Path:
        summary = self.latest_summary()
        classification = (summary.get("symbol_classification") if summary else None) or self._build_symbol_classification()
        if not classification:
            classification = {
                "available": False,
                "headline": "No symbol classification available yet",
                "summary": "Run the benchmark lab or wait for evaluated packs to generate symbol classifications.",
                "repeat_winners": [],
                "repeat_disappointments": [],
                "hidden_outperformers": [],
                "visible_underperformers": [],
            }
        pack_path = Path(self.config.model_dir) / "symbol_classification_pack.zip"

        def write_zip(zf):
            zf.writestr("symbol_classification.json", json.dumps(classification, indent=2, sort_keys=True))
            sections = {
                "repeat_winners.csv": classification.get("repeat_winners") or [],
                "repeat_disappointments.csv": classification.get("repeat_disappointments") or [],
                "hidden_outperformers.csv": classification.get("hidden_outperformers") or [],
                "visible_underperformers.csv": classification.get("visible_underperformers") or [],
            }
            fieldnames = [
                "symbol", "resolved_rows", "visible_rows", "non_visible_rows", "quality_hit_rate",
                "visible_quality_hit_rate", "non_visible_quality_hit_rate", "avg_end_ret", "max_live_score",
            ]
            for name, rows in sections.items():
                zf.writestr(name, _csv_text(list(rows), fieldnames))
            zf.writestr("current_version_summary.json", json.dumps((summary or {}).get("live_current_version") or self.review_packs.get_current_version_summary(), indent=2, sort_keys=True))

        return _atomic_zip_write(pack_path, write_zip)

    def run_threshold_sweep(
        self,
        *,
        hours: int,
        step_minutes: int,
        max_scans: int,
        max_symbols: int,
        thresholds: str | Iterable[float] | None = None,
    ) -> dict:
        parsed_thresholds = _parse_thresholds(thresholds)
        rows = []
        replay_classifications = []
        for threshold in parsed_thresholds:
            result = self.replay.run(
                hours=hours,
                step_minutes=step_minutes,
                max_scans=max_scans,
                max_symbols=max_symbols,
                pipeline_mode="raw_threshold",
                raw_threshold=threshold,
            )
            summary = result.get("summary") or {}
            evidence = _extract_replay_evidence(summary)
            counter = summary.get("counterfactual") or summary.get("counterfactual_summary") or {}
            concentration = (summary.get("outlier_concentration") or {}).get("thresholds") or {}
            upper_band = concentration.get("0.45") or concentration.get(0.45) or {}
            replay_symbol_rows = list(((summary.get("symbol_repeatability") or {}).get("rows") or []))
            replay_classifications.append({
                "threshold": round(float(threshold), 2),
                "classification": _classify_symbol_repeatability_rows(replay_symbol_rows, source=f"benchmark_replay_{round(float(threshold), 2):.2f}"),
            })
            row = {
                "threshold": round(float(threshold), 2),
                "headline": summary.get("headline"),
                "visible_quality_hit_rate": evidence.get("visible_quality_hit_rate"),
                "non_visible_quality_hit_rate": evidence.get("non_visible_quality_hit_rate"),
                "visible_raw_hit_rate": evidence.get("visible_raw_hit_rate"),
                "non_visible_raw_hit_rate": evidence.get("non_visible_raw_hit_rate"),
                "visible_avg_end_ret": evidence.get("visible_avg_end_ret"),
                "non_visible_avg_end_ret": evidence.get("non_visible_avg_end_ret"),
                "visible_rows": evidence.get("visible_rows"),
                "non_visible_rows": evidence.get("non_visible_rows"),
                "resolved_rows": evidence.get("resolved_rows"),
                "stage1_quality_recall": counter.get("stage1_quality_recall"),
                "missed_quality_opportunities": counter.get("missed_quality_opportunities"),
                "top_symbol_at_0_45": upper_band.get("top_symbol"),
                "top_symbol_share_at_0_45": upper_band.get("top_symbol_share"),
                "download_path": result.get("download_path"),
                "summary_path": result.get("summary_path"),
            }
            rows.append(row)
        recommendation = self._recommend_threshold(rows, max_scans=max_scans)
        recommended_threshold = recommendation.get("recommended_threshold")
        chosen = next((entry for entry in replay_classifications if entry.get("threshold") == recommended_threshold), None)
        if chosen and (chosen.get("classification") or {}).get("available"):
            symbol_classification = dict(chosen.get("classification") or {})
            symbol_classification["summary"] = (
                f"Symbol classifications are derived from the benchmark replay at threshold {recommended_threshold:.2f}. "
                + str(symbol_classification.get("summary") or "")
            ).strip()
        else:
            symbol_classification = self._build_symbol_classification()
        benchmark = {
            "generated_at_utc": _utc_now_iso(),
            "hours": int(hours),
            "step_minutes": int(step_minutes),
            "max_scans": int(max_scans),
            "max_symbols": int(max_symbols),
            "pipeline_mode": "raw_threshold",
            "thresholds": parsed_thresholds,
            "rows": rows,
            "recommendation": recommendation,
            "notes": [
                "Benchmark sweep replays the same window at multiple raw thresholds to compare shortlist quality without disturbing the live scanner.",
                "The sweep uses historical replay and should be treated as evidence for operating-point selection, not proof of live performance.",
            ],
        }
        benchmark["symbol_classification"] = symbol_classification
        benchmark = self._attach_live_context(benchmark)
        atomic_write_json(self.summary_path, benchmark)
        return benchmark

    def _attach_live_context(self, summary: dict) -> dict:
        enriched = dict(summary or {})
        try:
            enriched.setdefault("live_current_version", self.review_packs.get_current_version_summary())
        except Exception:
            enriched.setdefault("live_current_version", {})
        try:
            enriched.setdefault("model_audit", self.model_audit.latest_summary())
        except Exception:
            enriched.setdefault("model_audit", {})
        enriched.setdefault("symbol_classification", self._build_symbol_classification(current_version_summary=enriched.get("live_current_version") or {}))
        return enriched

    def _recommend_threshold(self, rows: List[dict], *, max_scans: int) -> dict:
        if not rows:
            return {
                "recommended_threshold": None,
                "reason": "No benchmark rows available yet.",
            }
        min_visible_rows = max(12, int(max_scans))
        eligible = [
            row for row in rows
            if (_rate(row, "visible_avg_end_ret") or -1.0) > 0.0
            and int(row.get("visible_rows") or 0) >= min_visible_rows
            and (_rate(row, "visible_quality_hit_rate") or -1.0) > (_rate(row, "non_visible_quality_hit_rate") or -1.0)
        ]
        pool = eligible or list(rows)
        best = max(
            pool,
            key=lambda row: (
                round((_rate(row, "visible_avg_end_ret") or -1.0), 6),
                round((_rate(row, "visible_quality_hit_rate") or -1.0), 6),
                int(row.get("visible_rows") or 0),
            ),
        )
        threshold = best.get("threshold")
        if eligible:
            reason = (
                f"Threshold {threshold:.2f} is the best balance point so far because it keeps visible average end return positive, "
                f"maintains a usable visible row count ({int(best.get('visible_rows') or 0)}), and still beats the hidden remainder on quality hit rate."
            )
        else:
            reason = (
                f"Threshold {threshold:.2f} was selected as the least-bad benchmark row because no threshold met the full positive-return and usable-row criteria."
            )
        return {
            "recommended_threshold": threshold,
            "reason": reason,
            "min_visible_rows_target": min_visible_rows,
        }

    def _build_symbol_classification(self, current_version_summary: dict | None = None) -> dict:
        try:
            summary = current_version_summary or self.review_packs.get_current_version_summary()
        except Exception:
            summary = current_version_summary or {}
        rows = list(((summary.get("symbol_repeatability") or {}).get("rows") or []))
        return _classify_symbol_repeatability_rows(rows, source="current_version")
