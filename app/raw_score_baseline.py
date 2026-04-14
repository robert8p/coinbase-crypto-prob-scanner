from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .replay import HistoricalReplayService
from .version import APP_VERSION


class RawScoreBaselineService:
    """Replay-backed raw-score baseline for current deployed model semantics.

    This is deliberately offline-only. It runs a full historical live-emulation
    replay, captures every stage-2 rankable row before display trimming, and
    summarizes whether raw model ranking quality exists independently of the
    current shortlist policy.
    """

    def __init__(self, config: AppConfig, replay: HistoricalReplayService):
        self.config = config
        self.replay = replay
        self.root_dir = ensure_dir(Path(config.model_dir) / "raw_score_baseline")
        self.pack_dir = ensure_dir(self.root_dir / "packs")
        self.summary_path = self.root_dir / "latest_raw_score_baseline_summary.json"
        self.latest_pack_link = self.pack_dir / "latest_raw_score_baseline_pack.zip"

    def latest_summary(self) -> dict:
        data = read_json(self.summary_path, {})
        return data if isinstance(data, dict) else {}

    def latest_pack(self) -> Path | None:
        if self.latest_pack_link.exists():
            return self.latest_pack_link
        packs = sorted(self.pack_dir.glob("raw_score_baseline_pack_*.zip"))
        return packs[-1] if packs else None

    def run(
        self,
        *,
        hours: int = 168,
        step_minutes: int = 120,
        max_scans: int = 84,
        max_symbols: int = 100,
    ) -> dict:
        replay_result = self.replay.run(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            pipeline_mode="full",
            raw_threshold=float(self.config.live_raw_threshold),
            capture_full_rankable_rows=True,
        )
        replay_summary = dict(replay_result.get("summary") or {})
        captured_rows = list(replay_result.get("captured_rankable_rows") or [])
        if not captured_rows:
            raise ValueError("replay completed but no captured rankable rows were returned")

        df = pd.DataFrame(captured_rows)
        if df.empty:
            raise ValueError("captured rankable rows are empty")
        for col, default in {
            "resolved": 0,
            "quality_touched": 0,
            "raw_touched": 0,
            "prob_2_model": 0.0,
            "pre_policy_score": 0.0,
            "live_score": 0.0,
            "post_model_total_penalty": 0.0,
            "was_capped": 0,
            "as_of_utc": None,
            "symbol": None,
        }.items():
            if col not in df.columns:
                df[col] = default
        df["resolved"] = pd.to_numeric(df["resolved"], errors="coerce").fillna(0).astype(int)
        df = df[df["resolved"] == 1].copy()
        if df.empty:
            raise ValueError("captured rankable rows contain no resolved rows")

        df["quality_touched"] = pd.to_numeric(df["quality_touched"], errors="coerce").fillna(0).astype(int)
        df["raw_touched"] = pd.to_numeric(df["raw_touched"], errors="coerce").fillna(0).astype(int)
        for col in ["prob_2_model", "pre_policy_score", "live_score", "post_model_total_penalty", "end_ret", "mae", "mfe"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df["was_capped"] = pd.to_numeric(df["was_capped"], errors="coerce").fillna(0).astype(int)
        df["as_of_utc"] = pd.to_datetime(df["as_of_utc"], utc=True, errors="coerce")
        df["symbol"] = df["symbol"].astype(str)

        base_quality_rate = float(df["quality_touched"].mean()) if len(df) else 0.0
        raw_distribution = self._score_distribution(df, score_col="prob_2_model", outcome_col="quality_touched")
        pre_policy_distribution = self._quantile_distribution(df["pre_policy_score"])
        live_distribution = self._quantile_distribution(df["live_score"])
        scan_topk = self._scan_topk_summary(df, score_col="prob_2_model", outcome_col="quality_touched")
        compression = self._compression_summary(
            df,
            raw_quantiles=raw_distribution.get("score_quantiles") or {},
            pre_quantiles=pre_policy_distribution,
            live_quantiles=live_distribution,
        )
        diagnosis = self._diagnosis(
            base_quality_rate=base_quality_rate,
            raw_distribution=raw_distribution,
            scan_topk=scan_topk,
            compression=compression,
        )

        top_percentile_rows = self._top_percentile_rows(raw_distribution)
        scan_topk_rows = self._scan_topk_rows(scan_topk)
        quantile_rows = self._quantile_rows(raw_distribution, pre_policy_distribution, live_distribution)
        summary = {
            "available": True,
            "app_version": APP_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "headline": diagnosis.get("headline"),
            "source_replay": {
                "headline": replay_summary.get("headline"),
                "window": replay_summary.get("window") or {},
                "pipeline_mode": replay_summary.get("pipeline_mode"),
                "raw_threshold": replay_summary.get("raw_threshold"),
            },
            "inputs": {
                "hours": int(hours),
                "step_minutes": int(step_minutes),
                "max_scans": int(max_scans),
                "max_symbols": int(max_symbols),
            },
            "resolved_row_count": int(len(df)),
            "scan_count": int(df["as_of_utc"].nunique()),
            "symbol_count": int(df["symbol"].nunique()),
            "base_quality_rate": round(base_quality_rate, 6),
            "raw_model_score_distribution": raw_distribution,
            "pre_policy_score_distribution": {
                "score_quantiles": pre_policy_distribution,
            },
            "live_score_distribution": {
                "score_quantiles": live_distribution,
            },
            "scan_topk_quality": scan_topk,
            "compression_summary": compression,
            "diagnosis": diagnosis,
            "notes": [
                "This baseline uses prob_2_model ordering across every resolved stage-2 rankable replay row before display trimming.",
                "It is intended to answer whether the raw model ranking has useful signal before more shortlist-policy iteration.",
                "The replay still inherits historical-emulation limitations already disclosed by the replay service.",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        pack_path = self._build_pack(
            summary=summary,
            quantile_rows=quantile_rows,
            top_percentile_rows=top_percentile_rows,
            scan_topk_rows=scan_topk_rows,
            rows_df=df,
            replay_summary=replay_summary,
        )
        try:
            self.latest_pack_link.unlink(missing_ok=True)
            self.latest_pack_link.write_bytes(pack_path.read_bytes())
        except Exception:
            pass
        return {
            "ok": True,
            "summary": summary,
            "pack_path": str(pack_path),
            "download_path": "/api/reviews/raw-score-baseline/latest-pack.zip",
            "summary_path": "/api/reviews/raw-score-baseline/summary",
        }

    def _quantile_distribution(self, values: Iterable[Any]) -> Dict[str, float | None]:
        series = pd.to_numeric(pd.Series(list(values), dtype="float64"), errors="coerce").dropna()
        if series.empty:
            return {"q50": None, "q75": None, "q90": None, "q95": None, "q99": None, "max": None}
        return {
            "q50": round(float(series.quantile(0.50)), 6),
            "q75": round(float(series.quantile(0.75)), 6),
            "q90": round(float(series.quantile(0.90)), 6),
            "q95": round(float(series.quantile(0.95)), 6),
            "q99": round(float(series.quantile(0.99)), 6),
            "max": round(float(series.max()), 6),
        }

    def _score_distribution(self, df: pd.DataFrame, *, score_col: str, outcome_col: str) -> dict:
        work = df[[score_col, outcome_col]].copy()
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
        work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce").fillna(0).astype(int)
        work = work.dropna(subset=[score_col]).sort_values(score_col, ascending=False).reset_index(drop=True)
        quantiles = self._quantile_distribution(work[score_col])
        base_rate = float(work[outcome_col].mean()) if len(work) else 0.0
        top_rates: Dict[str, dict] = {}
        top_lifts: Dict[str, float | None] = {}
        for frac, label in ((0.01, "top_1pct"), (0.05, "top_5pct"), (0.10, "top_10pct")):
            count = max(1, int(math.ceil(len(work) * frac))) if len(work) else 0
            subset = work.head(count) if count else work.head(0)
            rate = float(subset[outcome_col].mean()) if len(subset) else 0.0
            top_rates[label] = {
                "count": int(len(subset)),
                "quality_rate": round(rate, 6),
                "score_min": round(float(subset[score_col].min()), 6) if len(subset) else None,
                "score_max": round(float(subset[score_col].max()), 6) if len(subset) else None,
            }
            top_lifts[label] = round(rate / base_rate, 6) if base_rate > 0 else None
        q99 = float(quantiles.get("q99") or 0.0)
        vmax = float(quantiles.get("max") or 0.0)
        dead_upper_tail = bool(q99 < 0.60 and vmax < 0.70)
        return {
            "score_quantiles": quantiles,
            "base_quality_rate": round(base_rate, 6),
            "top_bucket_quality_rate": top_rates,
            "top_bucket_lift": top_lifts,
            "dead_upper_tail": dead_upper_tail,
        }

    def _scan_topk_summary(self, df: pd.DataFrame, *, score_col: str, outcome_col: str) -> dict:
        work = df[["as_of_utc", score_col, outcome_col, "symbol"]].copy()
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
        work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce").fillna(0).astype(int)
        work = work.dropna(subset=["as_of_utc", score_col]).sort_values(["as_of_utc", score_col], ascending=[True, False]).reset_index(drop=True)
        base_rate = float(work[outcome_col].mean()) if len(work) else 0.0
        rows: Dict[str, dict] = {}
        for k in (1, 3, 5):
            per_scan: List[float] = []
            hit_scans = 0
            scan_count = 0
            for _, scan in work.groupby("as_of_utc", sort=False):
                ordered = scan.head(min(k, len(scan)))
                if ordered.empty:
                    continue
                scan_count += 1
                rate = float(ordered[outcome_col].mean())
                per_scan.append(rate)
                if int(ordered[outcome_col].max()) == 1:
                    hit_scans += 1
            mean_rate = float(sum(per_scan) / len(per_scan)) if per_scan else None
            rows[f"top_{k}"] = {
                "k": int(k),
                "scan_count": int(scan_count),
                "mean_quality_rate": round(mean_rate, 6) if mean_rate is not None else None,
                "lift_vs_base": round(mean_rate / base_rate, 6) if mean_rate is not None and base_rate > 0 else None,
                "share_of_scans_with_hit": round(float(hit_scans) / float(scan_count), 6) if scan_count else None,
            }
        return rows

    def _compression_summary(self, df: pd.DataFrame, *, raw_quantiles: dict, pre_quantiles: dict, live_quantiles: dict) -> dict:
        def _gap(left: float | None, right: float | None) -> float | None:
            if left is None or right is None:
                return None
            return round(float(left) - float(right), 6)

        penalty_mean = pd.to_numeric(df["post_model_total_penalty"], errors="coerce").dropna()
        capped_fraction = float((df["was_capped"] == 1).mean()) if len(df) else 0.0
        penalized_fraction = float((pd.to_numeric(df["post_model_total_penalty"], errors="coerce").fillna(0.0) > 1e-9).mean()) if len(df) else 0.0
        return {
            "raw_minus_pre_policy_q99": _gap(raw_quantiles.get("q99"), pre_quantiles.get("q99")),
            "raw_minus_live_q99": _gap(raw_quantiles.get("q99"), live_quantiles.get("q99")),
            "raw_minus_pre_policy_max": _gap(raw_quantiles.get("max"), pre_quantiles.get("max")),
            "raw_minus_live_max": _gap(raw_quantiles.get("max"), live_quantiles.get("max")),
            "average_post_model_total_penalty": round(float(penalty_mean.mean()), 6) if not penalty_mean.empty else 0.0,
            "capped_row_fraction": round(capped_fraction, 6),
            "penalized_row_fraction": round(penalized_fraction, 6),
        }

    def _diagnosis(self, *, base_quality_rate: float, raw_distribution: dict, scan_topk: dict, compression: dict) -> dict:
        lifts = raw_distribution.get("top_bucket_lift") or {}
        top10_lift = float(lifts.get("top_10pct") or 0.0)
        top5_lift = float(lifts.get("top_5pct") or 0.0)
        scan_top1_lift = float(((scan_topk.get("top_1") or {}).get("lift_vs_base") or 0.0))
        scan_top3_lift = float(((scan_topk.get("top_3") or {}).get("lift_vs_base") or 0.0))
        q99_gap = float(compression.get("raw_minus_live_q99") or 0.0)
        max_gap = float(compression.get("raw_minus_live_max") or 0.0)
        penalized_fraction = float(compression.get("penalized_row_fraction") or 0.0)
        dead_upper_tail = bool(raw_distribution.get("dead_upper_tail"))

        if top10_lift >= 1.35 and scan_top3_lift >= 1.25:
            ranking_strength = "strong"
        elif top10_lift >= 1.15 or scan_top3_lift >= 1.10 or scan_top1_lift >= 1.20:
            ranking_strength = "moderate"
        else:
            ranking_strength = "weak"

        compression_significant = bool(q99_gap >= 0.08 or max_gap >= 0.10 or penalized_fraction >= 0.08)
        if ranking_strength == "strong" and compression_significant:
            blocker = "post_model_compression"
            recommended = "Reform post-model score shaping and caps before another shortlist-policy experiment."
            headline = "Raw ranking looks useful, but the deployed pipeline appears to compress the tail."
            rationale = "Top-of-score quality is materially better than base, while live/pre-policy scores sit meaningfully below raw-model tail levels."
        elif ranking_strength in {"strong", "moderate"} and not dead_upper_tail and not compression_significant:
            blocker = "calibration_semantics"
            recommended = "Recalibrate score semantics and thresholding around the existing raw ranking rather than adding new governance."
            headline = "Raw ranking is usable and the tail exists; semantics look likelier than policy to be the remaining blocker."
            rationale = "Top-of-score buckets beat base and the raw upper tail is present without large downstream compression."
        elif ranking_strength == "weak" and dead_upper_tail:
            blocker = "model_path"
            recommended = "Move to model/feature/target reform. Another shortlist-policy loop is unlikely to help."
            headline = "The current model path looks like the primary blocker."
            rationale = "Raw top buckets do not separate enough from base and the upper tail appears dead."
        else:
            blocker = "model_path"
            recommended = "Prioritize model-selection and feature/target work over more shortlist-policy iteration."
            headline = "Raw ranking improvement looks more important than policy refinement."
            rationale = "The baseline does not show enough raw-score separation to justify more policy search as the main path."

        return {
            "base_quality_rate": round(base_quality_rate, 6),
            "ranking_strength": ranking_strength,
            "tail_state": "dead_upper_tail" if dead_upper_tail else "tail_present",
            "compression_significant": compression_significant,
            "primary_blocker": blocker,
            "recommended_next_tranche": recommended,
            "headline": headline,
            "rationale": rationale,
        }

    def _top_percentile_rows(self, raw_distribution: dict) -> List[dict]:
        rows: List[dict] = []
        bucket_rates = raw_distribution.get("top_bucket_quality_rate") or {}
        lifts = raw_distribution.get("top_bucket_lift") or {}
        for key in ("top_1pct", "top_5pct", "top_10pct"):
            bucket = bucket_rates.get(key) or {}
            rows.append({
                "bucket": key,
                "count": int(bucket.get("count") or 0),
                "quality_rate": bucket.get("quality_rate"),
                "lift_vs_base": lifts.get(key),
                "score_min": bucket.get("score_min"),
                "score_max": bucket.get("score_max"),
            })
        return rows

    def _scan_topk_rows(self, scan_topk: dict) -> List[dict]:
        rows: List[dict] = []
        for key in ("top_1", "top_3", "top_5"):
            bucket = dict(scan_topk.get(key) or {})
            bucket["bucket"] = key
            rows.append(bucket)
        return rows

    def _quantile_rows(self, raw_distribution: dict, pre_policy_distribution: dict, live_distribution: dict) -> List[dict]:
        rows: List[dict] = []
        raw_quantiles = raw_distribution.get("score_quantiles") or {}
        for q in ("q50", "q75", "q90", "q95", "q99", "max"):
            rows.append({
                "quantile": q,
                "raw_model": raw_quantiles.get(q),
                "pre_policy": pre_policy_distribution.get(q),
                "live": live_distribution.get(q),
            })
        return rows

    def _summary_text(self, summary: dict) -> str:
        def _render(value: Any, indent: int = 0) -> str:
            prefix = " " * indent
            if isinstance(value, dict):
                lines: List[str] = []
                for key, val in value.items():
                    if isinstance(val, (dict, list)):
                        lines.append(f"{prefix}{key}:")
                        lines.append(_render(val, indent + 2))
                    else:
                        lines.append(f"{prefix}{key}: {'-' if val is None else val}")
                return "\n".join(lines)
            if isinstance(value, list):
                lines = []
                for item in value:
                    if isinstance(item, (dict, list)):
                        lines.append(f"{prefix}-")
                        lines.append(_render(item, indent + 2))
                    else:
                        lines.append(f"{prefix}- {'-' if item is None else item}")
                return "\n".join(lines)
            return f"{prefix}{'-' if value is None else value}"

        return _render(summary)

    def _build_pack(
        self,
        *,
        summary: dict,
        quantile_rows: List[dict],
        top_percentile_rows: List[dict],
        scan_topk_rows: List[dict],
        rows_df: pd.DataFrame,
        replay_summary: dict,
    ) -> Path:
        generated = str(summary.get("generated_at_utc") or datetime.now(timezone.utc).isoformat()).replace(":", "").replace("-", "")[:15]
        pack_path = self.pack_dir / f"raw_score_baseline_pack_{APP_VERSION.replace('.', '_')}_{generated}.zip"
        with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("raw_score_baseline_summary.json", json.dumps(summary, indent=2, default=str))
            zf.writestr("raw_score_baseline_summary.txt", self._summary_text(summary))
            zf.writestr("raw_score_baseline_quantiles.csv", self._csv_bytes(quantile_rows))
            zf.writestr("raw_score_baseline_top_percentiles.csv", self._csv_bytes(top_percentile_rows))
            zf.writestr("raw_score_baseline_scan_topk.csv", self._csv_bytes(scan_topk_rows))
            zf.writestr(
                "raw_score_baseline_resolved_rows.csv",
                rows_df.sort_values(["as_of_utc", "prob_2_model"], ascending=[True, False]).to_csv(index=False),
            )
            zf.writestr("raw_score_baseline_source_replay_summary.json", json.dumps(replay_summary, indent=2, default=str))
            zf.writestr(
                "raw_score_baseline_manifest.json",
                json.dumps(
                    {
                        "app_version": APP_VERSION,
                        "generated_at_utc": summary.get("generated_at_utc"),
                        "headline": summary.get("headline"),
                        "resolved_row_count": summary.get("resolved_row_count"),
                        "scan_count": summary.get("scan_count"),
                        "source_replay_window": ((summary.get("source_replay") or {}).get("window") or {}),
                    },
                    indent=2,
                    default=str,
                ),
            )
        return pack_path

    def _csv_bytes(self, rows: List[dict]) -> bytes:
        rows = list(rows or [])
        if not rows:
            return b""
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
        return buf.getvalue().encode("utf-8")
