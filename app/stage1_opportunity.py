from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import AppConfig
from .persist import atomic_write_json

STAGE1_OPPORTUNITY_FEATURES = [
    "ret_15m", "ret_60m", "ret_6h", "ret_24h", "asset_vs_btc_1h", "rvol_1h",
    "path_smoothness", "candle_efficiency", "wickiness", "downside_impulse",
    "momentum_persistence_1h", "move_vs_atr_ratio", "volume_acceleration", "uncertainty",
]


class Stage1OpportunityService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.replay_latest_pack_path = Path(config.model_dir) / "replay_packs" / "latest_replay_pack.zip"
        self.model_path = Path(config.model_dir) / "stage1_opportunity_scorer.joblib"
        self.summary_path = Path(config.model_dir) / "stage1_opportunity_scorer_summary.json"
        self._artifact = None

    def latest_summary(self) -> dict:
        if self.summary_path.exists():
            try:
                return json.loads(self.summary_path.read_text())
            except Exception:
                return {}
        return {}

    def available(self) -> bool:
        return self.model_path.exists()

    def _load_artifact(self):
        if self._artifact is None and self.model_path.exists():
            self._artifact = joblib.load(self.model_path)
        return self._artifact

    def build_from_latest_replay_pack(self) -> dict:
        if not self.replay_latest_pack_path.exists():
            raise FileNotFoundError("latest replay pack not found")
        with zipfile.ZipFile(self.replay_latest_pack_path, "r") as zf:
            if "replay_counterfactual_rows.csv" not in zf.namelist():
                raise FileNotFoundError("replay_counterfactual_rows.csv not found in latest replay pack")
            with zf.open("replay_counterfactual_rows.csv") as fh:
                df = pd.read_csv(io.BytesIO(fh.read()))
            manifest = {}
            if "replay_manifest.json" in zf.namelist():
                with zf.open("replay_manifest.json") as fh:
                    manifest = json.load(fh)

        if df.empty:
            raise ValueError("latest replay pack has no counterfactual rows")

        df = df.copy()
        df = df[df.get("stage1_blocked", False) != True]
        df = df[df.get("resolved", 0) == 1]
        if df.empty:
            raise ValueError("latest replay pack has no resolved, non-blocked stage1 rows")

        for col in STAGE1_OPPORTUNITY_FEATURES + ["quality_touched"]:
            if col not in df.columns:
                df[col] = 0.0
        df[STAGE1_OPPORTUNITY_FEATURES] = df[STAGE1_OPPORTUNITY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df["quality_touched"] = pd.to_numeric(df["quality_touched"], errors="coerce").fillna(0).astype(int)
        if df["quality_touched"].nunique() < 2:
            raise ValueError("latest replay pack does not have both positive and negative quality rows")

        if "as_of_utc" in df.columns:
            df["as_of_utc"] = pd.to_datetime(df["as_of_utc"], utc=True, errors="coerce")
            df = df.sort_values("as_of_utc").reset_index(drop=True)

        split_idx = max(1, int(len(df) * 0.8))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy() if split_idx < len(df) else df.iloc[-max(1, len(df)//5):].copy()

        X_train = train_df[STAGE1_OPPORTUNITY_FEATURES]
        y_train = train_df["quality_touched"]
        X_val = val_df[STAGE1_OPPORTUNITY_FEATURES]
        y_val = val_df["quality_touched"]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict_proba(X_val)[:, 1]

        auc_val = None
        try:
            if y_val.nunique() >= 2:
                auc_val = float(roc_auc_score(y_val, val_pred))
        except Exception:
            auc_val = None
        brier_val = float(brier_score_loss(y_val, val_pred))

        lr = pipeline.named_steps["model"]
        coefs = list(zip(STAGE1_OPPORTUNITY_FEATURES, [float(x) for x in lr.coef_[0]]))
        coefs_sorted = sorted(coefs, key=lambda x: x[1], reverse=True)
        summary = {
            "available": True,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_replay_pack": str(self.replay_latest_pack_path.name),
            "source_replay_generated_at_utc": manifest.get("generated_at_utc"),
            "source_replay_scan_count": manifest.get("scan_count"),
            "row_count_all": int(len(df)),
            "row_count_train": int(len(train_df)),
            "row_count_validation": int(len(val_df)),
            "positive_rate_all": round(float(df["quality_touched"].mean()), 4),
            "positive_rate_validation": round(float(y_val.mean()), 4),
            "auc_validation": round(float(auc_val), 4) if auc_val is not None else None,
            "brier_validation": round(float(brier_val), 4),
            "features": list(STAGE1_OPPORTUNITY_FEATURES),
            "top_positive_weights": [{"feature": k, "weight": round(v, 6)} for k, v in coefs_sorted[:8]],
            "top_negative_weights": [{"feature": k, "weight": round(v, 6)} for k, v in sorted(coefs, key=lambda x: x[1])[:8]],
            "selection_modes_supported": ["stage1_opportunity_model", "primary_plus_opportunity_reserve"],
            "notes": [
                "This scorer is trained from replay-labeled stage1 rows, not from live trade outcomes.",
                "Treat it as a stage1 opportunity-ranking aid, not a calibrated probability model.",
            ],
        }
        artifact = {
            "pipeline": pipeline,
            "features": list(STAGE1_OPPORTUNITY_FEATURES),
            "summary": summary,
        }
        joblib.dump(artifact, self.model_path)
        atomic_write_json(self.summary_path, summary)
        self._artifact = artifact
        return summary

    def score_feature_rows(self, feature_rows: Dict[str, dict], guardrails: Dict[str, dict] | None = None) -> Dict[str, float]:
        artifact = self._load_artifact()
        if artifact is None:
            return {}
        guardrails = guardrails or {}
        features = list(artifact.get("features") or STAGE1_OPPORTUNITY_FEATURES)
        rows = []
        symbols = []
        for symbol, row in (feature_rows or {}).items():
            guard = guardrails.get(symbol) or {}
            payload = dict(row or {})
            payload.setdefault("uncertainty", float(guard.get("uncertainty", 0.0) or 0.0))
            rows.append({k: float(payload.get(k, 0.0) or 0.0) for k in features})
            symbols.append(str(symbol))
        if not rows:
            return {}
        frame = pd.DataFrame(rows, columns=features).fillna(0.0)
        preds = artifact["pipeline"].predict_proba(frame)[:, 1]
        return {symbol: round(float(score), 6) for symbol, score in zip(symbols, preds)}
