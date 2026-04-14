from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import AppConfig
from .persist import atomic_write_json


class ModelAuditService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.replay_latest_pack_path = Path(config.model_dir) / "replay_packs" / "latest_replay_pack.zip"
        self.summary_path = Path(config.model_dir) / "latest_model_audit_summary.json"

    def latest_summary(self) -> dict:
        if self.summary_path.exists():
            try:
                return json.loads(self.summary_path.read_text())
            except Exception:
                return {}
        return {}

    def build_from_latest_replay_pack(self) -> dict:
        if not self.replay_latest_pack_path.exists():
            raise FileNotFoundError("latest replay pack not found")

        with zipfile.ZipFile(self.replay_latest_pack_path, "r") as zf:
            names = set(zf.namelist())
            required = {"replay_visible_rows.csv", "replay_non_visible_rows.csv", "replay_manifest.json", "replay_summary.json"}
            missing = required - names
            if missing:
                raise FileNotFoundError(f"latest replay pack missing required files: {sorted(missing)}")
            visible_df = pd.read_csv(io.BytesIO(zf.read("replay_visible_rows.csv")))
            hidden_df = pd.read_csv(io.BytesIO(zf.read("replay_non_visible_rows.csv")))
            manifest = json.loads(zf.read("replay_manifest.json"))
            replay_summary = json.loads(zf.read("replay_summary.json"))

        df = pd.concat([visible_df, hidden_df], ignore_index=True)
        if df.empty:
            raise ValueError("latest replay pack has no replay rows")
        for col in ["resolved", "quality_touched", "raw_touched", "prob_2_model", "live_score", "pre_policy_score", "end_ret", "mae"]:
            if col not in df.columns:
                df[col] = None
        df = df[pd.to_numeric(df["resolved"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if df.empty:
            raise ValueError("latest replay pack has no resolved replay rows")
        df["quality_touched"] = pd.to_numeric(df["quality_touched"], errors="coerce").fillna(0).astype(int)
        df["raw_touched"] = pd.to_numeric(df["raw_touched"], errors="coerce").fillna(0).astype(int)
        df["prob_2_model"] = pd.to_numeric(df["prob_2_model"], errors="coerce").fillna(0.0)
        df["end_ret"] = pd.to_numeric(df["end_ret"], errors="coerce")
        df["mae"] = pd.to_numeric(df["mae"], errors="coerce")

        scores = df["prob_2_model"].astype(float)
        y = df["quality_touched"].astype(int)
        auc = None
        if y.nunique() >= 2:
            try:
                auc = float(roc_auc_score(y, scores))
            except Exception:
                auc = None
        try:
            brier = float(brier_score_loss(y, scores.clip(0.0, 1.0)))
        except Exception:
            brier = None

        calibration = self._calibration_table(df, score_col="prob_2_model", outcome_col="quality_touched")
        tail_precision = self._tail_precision(df, score_col="prob_2_model", outcome_col="quality_touched")
        per_symbol = self._per_symbol_summary(df)

        summary = {
            "available": True,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_replay_pack": self.replay_latest_pack_path.name,
            "source_replay_generated_at_utc": manifest.get("generated_at_utc"),
            "source_replay_app_version": manifest.get("app_version") or replay_summary.get("app_version"),
            "source_replay_headline": manifest.get("headline") or replay_summary.get("headline"),
            "resolved_row_count": int(len(df)),
            "positive_rate": round(float(y.mean()), 4),
            "auc": round(float(auc), 4) if auc is not None else None,
            "brier": round(float(brier), 4) if brier is not None else None,
            "calibration_deciles": calibration,
            "tail_precision": tail_precision,
            "per_symbol_rows": per_symbol,
            "notes": [
                "This is a diagnostic of replay-resolved model ordering using prob_2_model from the latest replay pack.",
                "Treat this as a model-quality checkpoint, not as live trading validation.",
            ],
        }
        atomic_write_json(self.summary_path, summary)
        return summary

    def _calibration_table(self, df: pd.DataFrame, *, score_col: str, outcome_col: str) -> List[dict]:
        work = df[[score_col, outcome_col]].copy()
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
        work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce").fillna(0).astype(int)
        try:
            work["decile"] = pd.qcut(work[score_col], 10, duplicates="drop")
        except Exception:
            work["decile"] = pd.cut(work[score_col], bins=10, include_lowest=True)
        rows: List[dict] = []
        for idx, (_, group) in enumerate(work.groupby("decile", observed=False), start=1):
            rows.append({
                "bucket": int(idx),
                "count": int(len(group)),
                "predicted_mean": round(float(group[score_col].mean()), 4),
                "actual_rate": round(float(group[outcome_col].mean()), 4),
                "min_score": round(float(group[score_col].min()), 4),
                "max_score": round(float(group[score_col].max()), 4),
            })
        return rows

    def _tail_precision(self, df: pd.DataFrame, *, score_col: str, outcome_col: str) -> Dict[str, dict]:
        work = df[[score_col, outcome_col]].copy()
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
        work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce").fillna(0).astype(int)
        work = work.sort_values(score_col, ascending=False).reset_index(drop=True)
        out: Dict[str, dict] = {}
        for label, frac in (("top_10pct", 0.10), ("top_5pct", 0.05)):
            n = max(1, int(round(len(work) * frac)))
            subset = work.head(n)
            out[label] = {
                "count": int(len(subset)),
                "precision": round(float(subset[outcome_col].mean()), 4),
                "score_min": round(float(subset[score_col].min()), 4),
                "score_max": round(float(subset[score_col].max()), 4),
            }
        return out

    def _per_symbol_summary(self, df: pd.DataFrame) -> List[dict]:
        work = df.copy()
        work["symbol"] = work["symbol"].astype(str)
        rows: List[dict] = []
        for symbol, group in work.groupby("symbol"):
            if len(group) < 10:
                continue
            rows.append({
                "symbol": symbol,
                "resolved_rows": int(len(group)),
                "quality_hit_rate": round(float(group["quality_touched"].mean()), 4),
                "raw_hit_rate": round(float(group["raw_touched"].mean()), 4),
                "avg_end_ret": round(float(group["end_ret"].mean()), 6) if group["end_ret"].notna().any() else None,
                "avg_mae": round(float(group["mae"].mean()), 6) if group["mae"].notna().any() else None,
                "mean_prob_2_model": round(float(group["prob_2_model"].mean()), 4),
                "max_prob_2_model": round(float(group["prob_2_model"].max()), 4),
            })
        rows.sort(key=lambda r: (r["quality_hit_rate"], r["resolved_rows"]), reverse=True)
        return rows
