"""
Paper trade forward validation service.

Lifecycle:
1. After each scan, `log_predictions()` writes every scored asset to SQLite
   with the scan timestamp and entry price.
2. At the start of each scan, `resolve_pending()` finds predictions whose
   prediction horizon has elapsed, fetches the actual candle data, and records
   what really happened: did the quality touch occur? what was the actual
   high / low / close at the horizon?
3. `/api/paper-trade/summary` computes rolling forward accuracy from resolved
   predictions.
4. `/api/reliability-lab` exposes a richer live reliability view for the
   currently active target definition and model fingerprint only.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

from .coinbase_client import CoinbaseClient
from .config import AppConfig

logger = logging.getLogger(__name__)

from .version import APP_VERSION

_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_utc                 TEXT    NOT NULL,
    symbol                   TEXT    NOT NULL,
    entry_price              REAL    NOT NULL,
    prob_2_model             REAL,
    prob_2                   REAL    NOT NULL,
    risk                     REAL    NOT NULL,
    block_code               TEXT    NOT NULL DEFAULT 'OK',
    btc_regime               TEXT    NOT NULL DEFAULT 'unknown',
    pt2                      TEXT    NOT NULL DEFAULT 'heuristic',
    target_hash              TEXT,
    model_hash               TEXT,
    app_version              TEXT,
    target_move_pct          REAL,
    target_horizon_minutes   INTEGER,
    quality_max_mae          REAL,
    quality_min_end_ret      REAL,
    scan_interval_minutes    INTEGER,
    was_capped               INTEGER,
    activity_bucket          TEXT,
    liquidity_bucket         TEXT,
    score_rank               INTEGER,
    resolved                 INTEGER NOT NULL DEFAULT 0,
    resolve_utc              TEXT,
    actual_high              REAL,
    actual_low               REAL,
    actual_close             REAL,
    raw_touched              INTEGER,
    mae                      REAL,
    end_ret                  REAL,
    quality_touched          INTEGER
);
"""

_REQUIRED_COLUMNS = {
    "prob_2_model": "REAL",
    "target_hash": "TEXT",
    "model_hash": "TEXT",
    "app_version": "TEXT",
    "target_move_pct": "REAL",
    "target_horizon_minutes": "INTEGER",
    "quality_max_mae": "REAL",
    "quality_min_end_ret": "REAL",
    "scan_interval_minutes": "INTEGER",
    "was_capped": "INTEGER",
    "activity_bucket": "TEXT",
    "liquidity_bucket": "TEXT",
    "score_rank": "INTEGER",
}

_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_pred_pending ON predictions (resolved, scan_utc)",
    "CREATE INDEX IF NOT EXISTS idx_pred_resolved ON predictions (resolved, quality_touched)",
    "CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions (symbol, scan_utc)",
    "CREATE INDEX IF NOT EXISTS idx_pred_target_model ON predictions (target_hash, model_hash, resolved, scan_utc)",
]


def _existing_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(r[1]) for r in rows}


def _ensure_prediction_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_TABLE_SCHEMA)
    existing = _existing_columns(conn, "predictions")
    for col_name, col_type in _REQUIRED_COLUMNS.items():
        if col_name not in existing:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
    for stmt in _INDEX_STATEMENTS:
        conn.execute(stmt)


def _compute_target_hash(config: "AppConfig") -> str:
    return f"{config.target_move_pct}|{config.target_horizon_minutes}|{config.quality_max_mae}|{config.quality_min_end_ret}"


def _load_current_model_hash(config: "AppConfig") -> str:
    try:
        from .modeling import ModelBundle
        bundle = ModelBundle.load(config.model_path_pt2)
        if bundle is not None:
            return str(bundle.metadata.get("model_fingerprint") or "unversioned")
    except Exception as exc:  # pragma: no cover
        logger.warning("paper_trade_model_hash_load_failed error=%s", exc)
    return "untrained"


def _wilson_lower_bound(hits: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = hits / total
    denom = 1 + (z * z) / total
    centre = phat + (z * z) / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def _quantile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    q = min(1.0, max(0.0, q))
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] + (vals[hi] - vals[lo]) * frac


def _avg(values: List[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _evidence_label(n: int, strong_cut: int = 100, moderate_cut: int = 30) -> str:
    if n >= strong_cut:
        return "strong"
    if n >= moderate_cut:
        return "moderate"
    if n > 0:
        return "low"
    return "none"


class PaperTradeService:
    def __init__(self, config: AppConfig, client: CoinbaseClient):
        self.config = config
        self.client = client
        self._lock = threading.Lock()
        self._db_path = Path(config.model_dir) / "paper_trade.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._target_hash = _compute_target_hash(config)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        try:
            conn = self._connect()
            _ensure_prediction_schema(conn)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("paper_trade_db_init_failed error=%s", exc)

    def log_predictions(self, scores: List[dict]) -> int:
        if not scores:
            return 0
        cooldown_minutes = self.config.target_horizon_minutes
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        cutoff = (now - timedelta(minutes=cooldown_minutes)).isoformat()
        current_model_hash = _load_current_model_hash(self.config)
        try:
            with self._lock:
                conn = self._connect()
                recent = conn.execute(
                    """SELECT DISTINCT symbol FROM predictions
                       WHERE scan_utc > ? AND target_hash = ? AND model_hash = ?""",
                    (cutoff, self._target_hash, current_model_hash),
                ).fetchall()
                recent_symbols = {r["symbol"] for r in recent}
                rows = []
                for s in scores:
                    sym = s["symbol"]
                    if sym in recent_symbols:
                        continue
                    model_hash = str(s.get("model_hash") or current_model_hash)
                    rows.append((
                        now_iso,
                        sym,
                        float(s["price"]),
                        float(s.get("prob_2_model", s["prob_2"])),
                        float(s["prob_2"]),
                        float(s.get("risk", 0.0)),
                        s.get("block_code", "OK"),
                        s.get("btc_regime_context", "unknown"),
                        s.get("pt2", "heuristic"),
                        self._target_hash,
                        model_hash,
                        str(s.get("app_version") or APP_VERSION),
                        float(self.config.target_move_pct),
                        int(self.config.target_horizon_minutes),
                        float(self.config.quality_max_mae),
                        float(self.config.quality_min_end_ret),
                        int(self.config.scan_interval_minutes),
                        int(bool(s.get("was_capped", False))),
                        str(s.get("activity_bucket") or "unknown"),
                        str(s.get("liquidity_bucket") or "unknown"),
                        int(s.get("score_rank") or 0),
                    ))
                if rows:
                    conn.executemany(
                        """INSERT INTO predictions
                           (scan_utc, symbol, entry_price, prob_2_model, prob_2, risk,
                            block_code, btc_regime, pt2, target_hash, model_hash, app_version,
                            target_move_pct, target_horizon_minutes, quality_max_mae,
                            quality_min_end_ret, scan_interval_minutes, was_capped,
                            activity_bucket, liquidity_bucket, score_rank)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        rows,
                    )
                    conn.commit()
                conn.close()
            logged = len(rows)
            skipped = len(scores) - logged
            if logged or skipped:
                logger.info("paper_trade_logged new=%d skipped_cooldown=%d", logged, skipped)
            return logged
        except Exception as exc:
            logger.warning("paper_trade_log_failed error=%s", exc)
            return 0

    def resolve_pending(self, max_resolve: int = 200) -> int:
        now_utc = datetime.now(timezone.utc)
        try:
            with self._lock:
                conn = self._connect()
                raw_pending = conn.execute(
                    """SELECT id, symbol, entry_price, scan_utc, target_move_pct,
                              target_horizon_minutes, quality_max_mae, quality_min_end_ret,
                              target_hash, model_hash
                       FROM predictions
                       WHERE resolved = 0
                       ORDER BY scan_utc ASC
                       LIMIT ?""",
                    (max_resolve * 4,),
                ).fetchall()
                conn.close()
        except Exception as exc:
            logger.warning("paper_trade_resolve_query_failed error=%s", exc)
            return 0
        pending = []
        for row in raw_pending:
            row_d = dict(row)
            horizon_minutes = int(row_d.get("target_horizon_minutes") or self.config.target_horizon_minutes)
            scan_time = datetime.fromisoformat(row_d["scan_utc"])
            if scan_time <= now_utc - timedelta(minutes=horizon_minutes + 5):
                pending.append(row_d)
            if len(pending) >= max_resolve:
                break
        if not pending:
            return 0
        by_symbol: Dict[str, List[dict]] = {}
        for row in pending:
            by_symbol.setdefault(row["symbol"], []).append(row)
        resolved_count = 0
        for symbol, preds in by_symbol.items():
            try:
                import pandas as pd
                oldest_scan = min(pd.to_datetime(p["scan_utc"], utc=True) for p in preds)
                now = pd.Timestamp.now(tz="UTC")
                max_horizon_bars = max(1, max(int((p.get("target_horizon_minutes") or self.config.target_horizon_minutes) / 5) for p in preds))
                minutes_since_oldest = max(1, (now - oldest_scan).total_seconds() / 60)
                bars_needed = int(minutes_since_oldest / 5) + max_horizon_bars + 12
                bars_needed = min(bars_needed, 2000)
                df = self.client.get_candles(symbol, bars_needed)
                if df.empty:
                    continue
                df = df.sort_values("ts").reset_index(drop=True)
                df["ts_dt"] = pd.to_datetime(df["ts"], utc=True)
                for pred in preds:
                    outcome = self._resolve_one(pred, df)
                    if outcome is not None:
                        self._write_outcome(pred["id"], outcome)
                        resolved_count += 1
                time.sleep(self.config.request_pause_seconds)
            except Exception as exc:
                logger.warning("paper_trade_resolve_symbol_failed symbol=%s error=%s", symbol, exc)
        logger.info("paper_trade_resolved count=%d of %d pending", resolved_count, len(pending))
        return resolved_count

    def _resolve_one(self, pred: dict, df) -> dict | None:
        import pandas as pd
        scan_time = pd.to_datetime(pred["scan_utc"], utc=True)
        entry_price = float(pred["entry_price"])
        if entry_price <= 0:
            return None
        target_move_pct = float(pred.get("target_move_pct") or self.config.target_move_pct)
        target_horizon_minutes = int(pred.get("target_horizon_minutes") or self.config.target_horizon_minutes)
        quality_max_mae = float(pred.get("quality_max_mae") or self.config.quality_max_mae)
        quality_min_end_ret = float(pred.get("quality_min_end_ret") or self.config.quality_min_end_ret)
        horizon_bars = max(1, target_horizon_minutes // 5)
        future = df.loc[df["ts_dt"] > scan_time].head(horizon_bars)
        if len(future) < max(1, horizon_bars // 2):
            return None
        actual_high = float(future["high"].max())
        actual_low = float(future["low"].min())
        actual_close = float(future["close"].iloc[-1])
        raw_touched = int(actual_high >= entry_price * (1.0 + target_move_pct))
        mae = (actual_low / entry_price) - 1.0
        end_ret = (actual_close / entry_price) - 1.0
        quality_touched = int(raw_touched and mae > quality_max_mae and end_ret > quality_min_end_ret)
        return {
            "actual_high": round(actual_high, 8),
            "actual_low": round(actual_low, 8),
            "actual_close": round(actual_close, 8),
            "raw_touched": raw_touched,
            "mae": round(mae, 6),
            "end_ret": round(end_ret, 6),
            "quality_touched": quality_touched,
        }

    def _write_outcome(self, pred_id: int, outcome: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                conn = self._connect()
                conn.execute(
                    """UPDATE predictions SET
                         resolved = 1,
                         resolve_utc = ?,
                         actual_high = ?,
                         actual_low = ?,
                         actual_close = ?,
                         raw_touched = ?,
                         mae = ?,
                         end_ret = ?,
                         quality_touched = ?
                       WHERE id = ?""",
                    (
                        now,
                        outcome["actual_high"],
                        outcome["actual_low"],
                        outcome["actual_close"],
                        outcome["raw_touched"],
                        outcome["mae"],
                        outcome["end_ret"],
                        outcome["quality_touched"],
                        pred_id,
                    ),
                )
                conn.commit()
                conn.close()
        except Exception as exc:
            logger.warning("paper_trade_write_outcome_failed id=%s error=%s", pred_id, exc)

    def _load_resolved_current(self) -> tuple[list[dict], dict]:
        current_hash = self._target_hash
        current_model_hash = _load_current_model_hash(self.config)
        with self._lock:
            conn = self._connect()
            total_preds = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE target_hash = ? AND model_hash = ?",
                (current_hash, current_model_hash),
            ).fetchone()[0]
            total_resolved = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE resolved = 1 AND target_hash = ? AND model_hash = ?",
                (current_hash, current_model_hash),
            ).fetchone()[0]
            total_pending = total_preds - total_resolved
            other_target_count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE target_hash IS NOT NULL AND target_hash != ?",
                (current_hash,),
            ).fetchone()[0]
            other_model_count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE target_hash = ? AND model_hash IS NOT NULL AND model_hash != ?",
                (current_hash, current_model_hash),
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT scan_utc, symbol, entry_price, prob_2_model, prob_2, risk,
                          btc_regime, pt2, resolve_utc, actual_high, actual_low, actual_close,
                          raw_touched, mae, end_ret, quality_touched, target_horizon_minutes,
                          was_capped, activity_bucket, liquidity_bucket, score_rank
                   FROM predictions
                   WHERE resolved = 1 AND target_hash = ? AND model_hash = ?
                   ORDER BY scan_utc""",
                (current_hash, current_model_hash),
            ).fetchall()
            conn.close()
        meta = {
            "target_hash": current_hash,
            "model_hash": current_model_hash,
            "total_predictions": int(total_preds),
            "total_resolved": int(total_resolved),
            "total_pending": int(total_pending),
            "excluded_other_target": int(other_target_count),
            "excluded_other_model": int(other_model_count),
        }
        return [dict(r) for r in rows], meta

    def _build_base_metrics(self, resolved: list[dict], meta: dict) -> dict:
        if not resolved:
            return {
                "ok": True,
                **meta,
                "message": "no resolved predictions yet for the active target + model fingerprint",
            }
        probs = [float(r["prob_2"]) for r in resolved]
        probs_model = [float(r.get("prob_2_model") or r["prob_2"]) for r in resolved]
        raw = [int(r.get("raw_touched") or 0) for r in resolved]
        quality = [int(r.get("quality_touched") or 0) for r in resolved]
        maes = [float(r.get("mae") or 0.0) for r in resolved]
        end_rets = [float(r.get("end_ret") or 0.0) for r in resolved]
        regimes = [str(r.get("btc_regime") or "unknown") for r in resolved]
        episode_preds = []
        seen = {}
        for r in resolved:
            sym = r["symbol"]
            scan_t = datetime.fromisoformat(r["scan_utc"])
            last_t = seen.get(sym)
            cooldown = timedelta(minutes=int(r.get("target_horizon_minutes") or self.config.target_horizon_minutes))
            if last_t is None or (scan_t - last_t) >= cooldown:
                episode_preds.append(r)
                seen[sym] = scan_t
        ep_probs = [float(r["prob_2"]) for r in episode_preds]
        ep_quality = [int(r.get("quality_touched") or 0) for r in episode_preds]
        thresholds = {}
        for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            adj_idx = [i for i, p in enumerate(probs) if p >= th]
            mdl_idx = [i for i, p in enumerate(probs_model) if p >= th]
            epi_idx = [i for i, p in enumerate(ep_probs) if p >= th]
            hits = sum(quality[i] for i in adj_idx)
            m_hits = sum(quality[i] for i in mdl_idx)
            e_hits = sum(ep_quality[i] for i in epi_idx)
            count = len(adj_idx)
            m_count = len(mdl_idx)
            e_count = len(epi_idx)
            precision = hits / count if count else None
            m_precision = m_hits / m_count if m_count else None
            e_precision = e_hits / e_count if e_count else None
            thresholds[f"precision_at_{th:.2f}"] = {
                "precision": round(precision, 4) if precision is not None else None,
                "count": count,
                "hits": hits,
                "model_precision": round(m_precision, 4) if m_precision is not None else None,
                "model_count": m_count,
                "episode_precision": round(e_precision, 4) if e_precision is not None else None,
                "episode_count": e_count,
                "episode_hits": e_hits,
                "wilson_lower": round(_wilson_lower_bound(hits, count), 4) if count else None,
                "episode_wilson_lower": round(_wilson_lower_bound(e_hits, e_count), 4) if e_count else None,
            }
        calibration = []
        for lo, hi in [(0.0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.0)]:
            idx = [i for i, p in enumerate(probs) if lo <= p < hi]
            if not idx:
                continue
            actual = sum(quality[i] for i in idx) / len(idx)
            pred = sum(probs[i] for i in idx) / len(idx)
            calibration.append({
                "band": f"{lo:.1f}-{hi:.1f}",
                "count": len(idx),
                "avg_predicted": round(pred, 4),
                "actual_quality_rate": round(actual, 4),
                "gap": round(pred - actual, 4),
            })
        regime_stats = {}
        for regime in sorted(set(regimes)):
            idx = [i for i, r in enumerate(regimes) if r == regime]
            regime_stats[regime] = {
                "count": len(idx),
                "quality_rate": round(sum(quality[i] for i in idx) / len(idx), 4),
                "raw_touch_rate": round(sum(raw[i] for i in idx) / len(idx), 4),
                "avg_predicted_prob": round(sum(probs[i] for i in idx) / len(idx), 4),
                "avg_mae": round(sum(maes[i] for i in idx) / len(idx), 4),
            }
        model_stats = {}
        pt2_list = [str(r.get("pt2") or "unknown") for r in resolved]
        for pt2 in sorted(set(pt2_list)):
            idx = [i for i, v in enumerate(pt2_list) if v == pt2]
            model_stats[pt2] = {
                "count": len(idx),
                "quality_rate": round(sum(quality[i] for i in idx) / len(idx), 4),
                "avg_predicted_prob": round(sum(probs[i] for i in idx) / len(idx), 4),
            }
        return {
            "ok": True,
            **meta,
            "total_episodes": len(episode_preds),
            "overall": {
                "raw_touch_rate": round(sum(raw) / len(raw), 4),
                "quality_touch_rate": round(sum(quality) / len(quality), 4),
                "avg_mae": round(sum(maes) / len(maes), 4),
                "avg_end_ret": round(sum(end_rets) / len(end_rets), 4),
            },
            "precision_at_threshold": thresholds,
            "calibration": calibration,
            "by_regime": regime_stats,
            "by_model": model_stats,
        }

    def get_summary(self) -> dict:
        try:
            resolved, meta = self._load_resolved_current()
            return self._build_base_metrics(resolved, meta)
        except Exception as exc:
            logger.warning("paper_trade_summary_failed error=%s", exc)
            return {"ok": False, "message": str(exc)}

    def get_reliability_lab(self) -> dict:
        try:
            resolved, meta = self._load_resolved_current()
            if not resolved:
                return {
                    "ok": True,
                    **meta,
                    "lab_ready": False,
                    "message": "no resolved predictions yet for the active target + model fingerprint",
                    "evidence": {
                        "overall": {"level": "none", "resolved_predictions": 0},
                        "high_confidence": {"level": "none", "resolved_predictions": 0},
                    },
                }
            probs = [float(r["prob_2"]) for r in resolved]
            probs_model = [float(r.get("prob_2_model") or r["prob_2"]) for r in resolved]
            quality = [int(r.get("quality_touched") or 0) for r in resolved]
            raw = [int(r.get("raw_touched") or 0) for r in resolved]
            maes = [float(r.get("mae") or 0.0) for r in resolved]
            end_rets = [float(r.get("end_ret") or 0.0) for r in resolved]
            capped_flags = [int(r.get("was_capped") or 0) for r in resolved]
            activity_buckets = [str(r.get("activity_bucket") or "unknown") for r in resolved]
            liquidity_buckets = [str(r.get("liquidity_bucket") or "unknown") for r in resolved]
            regimes = [str(r.get("btc_regime") or "unknown") for r in resolved]
            score_ranks = [int(r.get("score_rank") or 0) for r in resolved]
            # Calibration / honesty buckets
            bucket_defs = [
                (0.00, 0.40), (0.40, 0.50), (0.50, 0.55), (0.55, 0.60),
                (0.60, 0.65), (0.65, 0.70), (0.70, 1.01),
            ]
            score_buckets = []
            weighted_gap_num = 0.0
            weighted_gap_den = 0
            for lo, hi in bucket_defs:
                idx = [i for i, p in enumerate(probs) if lo <= p < hi]
                if not idx:
                    continue
                avg_pred = sum(probs[i] for i in idx) / len(idx)
                actual = sum(quality[i] for i in idx) / len(idx)
                avg_mae = sum(maes[i] for i in idx) / len(idx)
                avg_end_ret = sum(end_rets[i] for i in idx) / len(idx)
                gap = avg_pred - actual
                weighted_gap_num += abs(gap) * len(idx)
                weighted_gap_den += len(idx)
                score_buckets.append({
                    "band": f"{lo:.2f}-{min(hi,1.0):.2f}",
                    "count": len(idx),
                    "avg_predicted": round(avg_pred, 4),
                    "actual_quality_rate": round(actual, 4),
                    "gap": round(gap, 4),
                    "avg_mae": round(avg_mae, 4),
                    "avg_end_ret": round(avg_end_ret, 4),
                })
            avg_abs_gap = (weighted_gap_num / weighted_gap_den) if weighted_gap_den else None
            honesty_score = None if avg_abs_gap is None else max(0.0, 1.0 - min(1.0, avg_abs_gap / 0.25))

            top_tail = []
            for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                idx = [i for i, p in enumerate(probs) if p >= th]
                m_idx = [i for i, p in enumerate(probs_model) if p >= th]
                hits = sum(quality[i] for i in idx)
                count = len(idx)
                model_hits = sum(quality[i] for i in m_idx)
                model_count = len(m_idx)
                avg_mae = _avg([maes[i] for i in idx])
                avg_end = _avg([end_rets[i] for i in idx])
                top_tail.append({
                    "threshold": round(th, 2),
                    "count": count,
                    "hits": hits,
                    "precision": round(hits / count, 4) if count else None,
                    "wilson_lower": round(_wilson_lower_bound(hits, count), 4) if count else None,
                    "avg_mae": round(avg_mae, 4) if avg_mae is not None else None,
                    "avg_end_ret": round(avg_end, 4) if avg_end is not None else None,
                    "model_count": model_count,
                    "model_precision": round(model_hits / model_count, 4) if model_count else None,
                })

            def segment_by(values: list[str | int], key_name: str) -> dict:
                out = {}
                for key in sorted(set(values), key=lambda x: str(x)):
                    idx = [i for i, v in enumerate(values) if v == key]
                    count = len(idx)
                    hits = sum(quality[i] for i in idx)
                    out[str(key)] = {
                        "count": count,
                        "quality_rate": round(hits / count, 4) if count else None,
                        "avg_predicted": round(sum(probs[i] for i in idx) / count, 4) if count else None,
                        "avg_mae": round(sum(maes[i] for i in idx) / count, 4) if count else None,
                        "avg_end_ret": round(sum(end_rets[i] for i in idx) / count, 4) if count else None,
                    }
                return out

            rolling = []
            for size in [20, 50, 100]:
                rows = resolved[-size:]
                if not rows:
                    continue
                r_probs = [float(r["prob_2"]) for r in rows]
                r_quality = [int(r.get("quality_touched") or 0) for r in rows]
                avg_pred = sum(r_probs) / len(r_probs)
                actual = sum(r_quality) / len(r_quality)
                top_idx = [i for i, p in enumerate(r_probs) if p >= 0.60]
                top_hits = sum(r_quality[i] for i in top_idx)
                rolling.append({
                    "window": size,
                    "count": len(rows),
                    "avg_predicted": round(avg_pred, 4),
                    "actual_quality_rate": round(actual, 4),
                    "gap": round(avg_pred - actual, 4),
                    "top_tail_count": len(top_idx),
                    "top_tail_precision": round(top_hits / len(top_idx), 4) if top_idx else None,
                    "top_tail_wilson_lower": round(_wilson_lower_bound(top_hits, len(top_idx)), 4) if top_idx else None,
                })

            recent = []
            for row in resolved[-20:][::-1]:
                recent.append({
                    "scan_utc": row["scan_utc"],
                    "symbol": row["symbol"],
                    "adjusted_score": round(float(row["prob_2"]), 4),
                    "model_score": round(float(row.get("prob_2_model") or row["prob_2"]), 4),
                    "quality_touched": int(row.get("quality_touched") or 0),
                    "mae": round(float(row.get("mae") or 0.0), 4),
                    "end_ret": round(float(row.get("end_ret") or 0.0), 4),
                    "btc_regime": row.get("btc_regime") or "unknown",
                    "was_capped": bool(row.get("was_capped") or 0),
                    "activity_bucket": row.get("activity_bucket") or "unknown",
                    "liquidity_bucket": row.get("liquidity_bucket") or "unknown",
                    "score_rank": int(row.get("score_rank") or 0),
                })

            top60_count = sum(1 for p in probs if p >= 0.60)
            evidence = {
                "overall": {
                    "level": _evidence_label(len(resolved), strong_cut=150, moderate_cut=50),
                    "resolved_predictions": len(resolved),
                },
                "high_confidence": {
                    "level": _evidence_label(top60_count, strong_cut=75, moderate_cut=25),
                    "resolved_predictions": top60_count,
                },
            }
            reliability_gate = {
                "status": "strong" if evidence["overall"]["level"] == "strong" and evidence["high_confidence"]["level"] in ("moderate", "strong") else ("moderate" if evidence["overall"]["level"] in ("moderate", "strong") else "low"),
                "message": (
                    "High-confidence evidence is still thin." if top60_count < 25 else
                    "Model has a usable amount of live evidence, but keep watching recent drift." if top60_count < 75 else
                    "Model has accumulated meaningful live evidence for the current target and model version."
                ),
            }
            return {
                "ok": True,
                **meta,
                "lab_ready": True,
                "evidence": evidence,
                "reliability_gate": reliability_gate,
                "headline": {
                    "live_confidence_honesty": round(honesty_score, 4) if honesty_score is not None else None,
                    "avg_abs_calibration_gap": round(avg_abs_gap, 4) if avg_abs_gap is not None else None,
                    "resolved_predictions": len(resolved),
                    "quality_touch_rate": round(sum(quality) / len(quality), 4),
                    "raw_touch_rate": round(sum(raw) / len(raw), 4),
                    "dead_upper_tail_live": top60_count == 0,
                },
                "score_distribution": {
                    "min": round(min(probs), 4),
                    "p50": round(_quantile(probs, 0.50) or 0.0, 4),
                    "p90": round(_quantile(probs, 0.90) or 0.0, 4),
                    "p95": round(_quantile(probs, 0.95) or 0.0, 4),
                    "p99": round(_quantile(probs, 0.99) or 0.0, 4),
                    "max": round(max(probs), 4),
                },
                "score_buckets": score_buckets,
                "top_tail": top_tail,
                "rolling_windows": rolling,
                "by_regime": segment_by(regimes, "btc_regime"),
                "by_capped": segment_by(capped_flags, "was_capped"),
                "by_activity_bucket": segment_by(activity_buckets, "activity_bucket"),
                "by_liquidity_bucket": segment_by(liquidity_buckets, "liquidity_bucket"),
                "by_rank_bucket": segment_by([
                    "1-5" if 1 <= r <= 5 else "6-15" if 6 <= r <= 15 else "16+" if r >= 16 else "unknown"
                    for r in score_ranks
                ], "rank_bucket"),
                "recent_resolved": recent,
            }
        except Exception as exc:
            logger.warning("reliability_lab_failed error=%s", exc)
            return {"ok": False, "message": str(exc)}

    def get_recent(self, limit: int = 50) -> List[dict]:
        try:
            with self._lock:
                conn = self._connect()
                rows = conn.execute(
                    """SELECT scan_utc, symbol, entry_price, prob_2_model, prob_2, risk,
                              btc_regime, pt2, resolve_utc, was_capped,
                              activity_bucket, liquidity_bucket, score_rank,
                              actual_high, actual_low, actual_close,
                              raw_touched, mae, end_ret, quality_touched
                       FROM predictions
                       WHERE resolved = 1
                       ORDER BY scan_utc DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
                conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("paper_trade_recent_failed error=%s", exc)
            return []

    def get_counts(self) -> dict:
        try:
            with self._lock:
                conn = self._connect()
                total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                resolved = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved = 1").fetchone()[0]
                conn.close()
            return {"total": total, "resolved": resolved, "pending": total - resolved}
        except Exception:
            return {"total": 0, "resolved": 0, "pending": 0}
