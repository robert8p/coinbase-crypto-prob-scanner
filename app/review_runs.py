from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from .coinbase_client import CoinbaseClient
from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json

logger = logging.getLogger(__name__)

from .version import APP_VERSION

RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_runs (
    run_id TEXT PRIMARY KEY,
    scan_started_utc TEXT,
    scan_finished_utc TEXT,
    trigger_source TEXT,
    app_version TEXT,
    model_fingerprint TEXT,
    target_move_pct REAL,
    target_horizon_minutes INTEGER,
    quality_max_mae REAL,
    quality_min_end_ret REAL,
    live_universe_mode TEXT,
    market_regime_state TEXT,
    market_regime_actionability TEXT,
    cooldown_active INTEGER,
    visible_rows_count INTEGER NOT NULL DEFAULT 0,
    suppressed_rows_count INTEGER NOT NULL DEFAULT 0,
    evaluation_due_utc TEXT,
    evaluation_complete INTEGER NOT NULL DEFAULT 0,
    evaluation_started_utc TEXT,
    evaluation_finished_utc TEXT,
    latest_scan_pack_path TEXT,
    latest_scan_pack_generated_utc TEXT,
    latest_evaluated_pack_path TEXT,
    latest_evaluated_pack_generated_utc TEXT,
    review_status_path TEXT,
    review_visible_rows_path TEXT,
    review_suppressed_rows_path TEXT,
    review_summary_path TEXT,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_run_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    row_type TEXT NOT NULL,
    suppression_reason TEXT,
    symbol TEXT NOT NULL,
    entry_utc TEXT NOT NULL,
    entry_price REAL NOT NULL,
    target_move_pct REAL NOT NULL,
    target_horizon_minutes INTEGER NOT NULL,
    quality_max_mae REAL NOT NULL,
    quality_min_end_ret REAL NOT NULL,
    candidate_stage TEXT,
    prob_2_model REAL,
    prob_2_pre_regime REAL,
    prob_2_rank REAL,
    prob_2 REAL,
    live_score REAL,
    opportunity_score REAL,
    risk REAL,
    liquidity_tier TEXT,
    actionability_tier TEXT,
    actionability_type TEXT,
    actionability_reason TEXT,
    policy_constraint_reason TEXT,
    suppression_reason_detail TEXT,
    pre_policy_score REAL,
    pre_policy_rank INTEGER,
    candidate_rank_all INTEGER,
    informational_rank INTEGER,
    live_threshold REAL,
    validated_floor REAL,
    distance_to_validated REAL,
    distance_to_live_threshold REAL,
    score_band TEXT,
    visibility_band TEXT,
    visibility_band_label TEXT,
    score_band_label TEXT,
    monitor_priority TEXT,
    contract_truth_state TEXT,
    contract_truth_semantics TEXT,
    temporal_tail_state TEXT,
    temporal_tail_semantics TEXT,
    market_regime_state TEXT,
    cooldown_active INTEGER NOT NULL DEFAULT 0,
    reasons_json TEXT,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolve_utc TEXT,
    actual_high REAL,
    actual_low REAL,
    actual_close REAL,
    raw_touched INTEGER,
    quality_touched INTEGER,
    mae REAL,
    end_ret REAL,
    mfe REAL,
    time_to_touch_minutes INTEGER,
    was_actionable INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES review_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_review_runs_due ON review_runs (evaluation_complete, evaluation_due_utc);
CREATE INDEX IF NOT EXISTS idx_review_run_rows_pending ON review_run_rows (resolved, entry_utc);
CREATE INDEX IF NOT EXISTS idx_review_run_rows_run ON review_run_rows (run_id, row_type, symbol);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _compact_status_for_pack(status: dict) -> dict:
    status = dict(status or {})
    # keep the compact status as-is; it is already slimmed in v4.1.7.
    return status


def _non_visible_row_types() -> tuple[str, ...]:
    return ("suppressed", "informational", "overflow")


ROW_CSV_FIELDS = [
    "id", "run_id", "row_type", "suppression_reason", "suppression_reason_detail", "symbol",
    "entry_utc", "entry_price", "target_move_pct", "target_horizon_minutes", "quality_max_mae",
    "quality_min_end_ret", "candidate_stage", "prob_2_model", "prob_2_pre_regime", "pre_policy_score",
    "prob_2_rank", "prob_2", "live_score", "live_threshold", "validated_floor", "distance_to_validated", "distance_to_live_threshold", "score_band",
    "score_band_label", "visibility_band", "visibility_band_label", "monitor_priority", "opportunity_score", "risk", "liquidity_tier",
    "actionability_tier", "actionability_type", "actionability_reason", "policy_constraint_reason",
    "pre_policy_rank", "candidate_rank_all", "informational_rank", "contract_truth_state",
    "contract_truth_semantics", "temporal_tail_state", "temporal_tail_semantics", "market_regime_state",
    "cooldown_active", "reasons_json", "resolved", "resolve_utc", "actual_high", "actual_low",
    "actual_close", "raw_touched", "quality_touched", "mae", "end_ret", "mfe",
    "time_to_touch_minutes", "was_actionable", "created_at_utc", "updated_at_utc",
]

BLOCKED_FOCUS_FIELDS = [
    "symbol", "pre_policy_rank", "liquidity_tier", "pre_policy_score", "live_score", "live_threshold", "pre_policy_distance_to_validated", "distance_to_validated",
    "distance_to_live_threshold", "distance_to_live_threshold_pct_points", "pre_policy_score_band", "pre_policy_score_band_label", "score_band", "score_band_label",
    "visibility_band", "visibility_band_label", "suppression_reason", "suppression_reason_detail",
]

FOLLOWUP_CHANGE_FIELDS = [
    "symbol", "prior_pre_policy_score", "prior_live_score", "prior_live_threshold", "prior_visibility_band", "prior_score_band",
    "current_row_type", "current_actionability_tier", "current_pre_policy_score", "current_live_score", "current_live_threshold",
    "current_visibility_band", "current_score_band", "current_distance_to_live_threshold", "delta_pre_policy_score", "delta_live_score",
    "became_visible", "still_blocked", "missing_current",
]

TRACKED_VISIBLE_FIELDS = [
    "symbol", "tracked_rank", "row_type", "actionability_tier", "pre_policy_rank", "candidate_rank_all",
    "pre_policy_score", "live_score", "live_threshold", "distance_to_validated", "distance_to_live_threshold",
    "score_band", "score_band_label", "visibility_band", "visibility_band_label", "delta_live_score", "delta_pre_policy_score",
]

TOP_PRETRIM_FIELDS = [
    "symbol", "liquidity_tier", "candidate_stage", "row_type", "display_bucket",
    "actionability_tier", "suppression_reason", "prob_2_model", "pre_policy_score",
    "live_score", "distance_to_validated", "distance_to_live_threshold",
]

STAGE1_TRACE_FIELDS = [
    "symbol", "liquidity_tier", "stage1_selected", "stage1_rank", "stage1_selection_source", "stage1_blocked", "stage1_block_code",
    "stage2_fetched", "stage2_scored", "final_row_type", "final_display_bucket", "final_suppression_reason",
    "final_actionability_tier", "candidate_rank_all", "pre_policy_rank", "prob_2_model", "pre_policy_score", "live_score",
]

CANDIDATE_QUALITY_TIER_FIELDS = [
    "liquidity_tier", "stage1_feature_ready", "stage1_blocked", "stage1_selected", "stage1_selected_share",
    "stage2_scored", "stage2_visible", "stage2_hidden", "stage2_live_max", "stage2_live_median",
    "stage2_count_ge_0_30", "stage2_count_ge_0_35", "stage2_count_ge_0_45",
]

RECENT_RESOLVED_FIELDS = ROW_CSV_FIELDS + [
    "source_run_finished_utc", "source_model_fingerprint", "source_trigger_source", "source_run_market_regime_state",
]

POLICY_AUDIT_REASON_FIELDS = [
    "suppression_reason", "count", "quality_hit_rate", "raw_hit_rate", "avg_end_ret", "avg_mae", "avg_mfe", "avg_time_to_touch_minutes",
]

POLICY_AUDIT_REGIME_FIELDS = [
    "market_regime_state", "total", "visible_count", "visible_quality_hit_rate", "visible_avg_end_ret",
    "suppressed_count", "suppressed_quality_hit_rate", "suppressed_avg_end_ret",
]
CURRENT_VERSION_THRESHOLD_FIELDS = [
    "threshold", "count", "visible_count", "non_visible_count", "quality_hit_rate", "raw_hit_rate", "avg_end_ret", "avg_mae",
]
CURRENT_VERSION_REGIME_FIELDS = [
    "market_regime_state", "market_regime_actionability", "run_count", "evaluated_run_count", "visible_rows", "suppressed_rows",
]
CURRENT_VERSION_REPEATABILITY_FIELDS = [
    "symbol", "resolved_rows", "visible_rows", "non_visible_rows", "quality_hits", "raw_hits", "quality_hit_rate",
    "raw_hit_rate", "visible_quality_hit_rate", "non_visible_quality_hit_rate", "avg_end_ret", "avg_mae", "max_live_score",
    "count_ge_0_35", "count_ge_0_45", "count_ge_0_60",
]
CURRENT_VERSION_REGIME_EVIDENCE_FIELDS = [
    "market_regime_state", "market_regime_actionability", "resolved_rows", "visible_rows", "non_visible_rows",
    "visible_quality_hit_rate", "non_visible_quality_hit_rate", "visible_raw_hit_rate", "non_visible_raw_hit_rate",
    "visible_avg_end_ret", "non_visible_avg_end_ret", "visible_avg_mae", "non_visible_avg_mae",
]
CURRENT_VERSION_REGIME_THRESHOLD_FIELDS = [
    "market_regime_state", "market_regime_actionability", "threshold", "count", "visible_count", "non_visible_count",
    "quality_hit_rate", "raw_hit_rate", "avg_end_ret", "avg_mae",
]

def _is_non_visible_row(row: dict) -> bool:
    return str(row.get("row_type") or "") in _non_visible_row_types()


class ReviewPackService:
    def __init__(self, config: AppConfig, client: CoinbaseClient):
        self.config = config
        self.client = client
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.root_dir = ensure_dir(Path(config.model_dir) / "review_runs")
        self.pack_dir = ensure_dir(Path(config.model_dir) / "review_packs")
        self.db_path = Path(config.model_dir) / "review_runs.db"
        self.latest_scan_link = self.pack_dir / "latest_scan_pack.zip"
        self.latest_eval_link = self.pack_dir / "latest_evaluated_pack.zip"
        self.post_evaluation_callback = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(RUNS_SCHEMA)
            existing = {str(r[1]) for r in conn.execute("PRAGMA table_info(review_run_rows)").fetchall()}
            migrations = {
                "suppression_reason_detail": "TEXT",
                "pre_policy_score": "REAL",
                "pre_policy_rank": "INTEGER",
                "candidate_rank_all": "INTEGER",
                "informational_rank": "INTEGER",
                "validated_floor": "REAL",
                "distance_to_validated": "REAL",
                "score_band": "TEXT",
                "score_band_label": "TEXT",
                "monitor_priority": "TEXT",
                "live_threshold": "REAL",
                "distance_to_live_threshold": "REAL",
                "visibility_band": "TEXT",
                "visibility_band_label": "TEXT",
            }
            for col, typ in migrations.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE review_run_rows ADD COLUMN {col} {typ}")
            conn.commit()

    def start_background_threads(self) -> None:
        if not self.config.review_packs_enabled or not self.config.review_evaluator_enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="review-pack-evaluator")
        self._thread.start()

    def stop_background_threads(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        interval = max(60, int(self.config.review_evaluate_interval_minutes * 60))
        while not self._stop.wait(interval):
            try:
                resolved = self.resolve_due_runs(max_runs=20)
                if resolved:
                    logger.info("review_pack_resolved rows=%d", resolved)
                self.prune_old_runs()
            except Exception as exc:  # pragma: no cover
                logger.warning("review_pack_loop_failed error=%s", exc)

    def record_scan(self, *, status: dict, visible_rows: List[dict], suppressed_rows: List[dict], informational_rows: List[dict], overflow_rows: List[dict], trigger_source: str) -> str | None:
        if not self.config.review_packs_enabled:
            return None
        run_id = f"run_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        run_dir = ensure_dir(self.root_dir / run_id)
        now_iso = _utc_now_iso()
        scan = status.get("scan") or {}
        target = status.get("target") or {}
        model_pt2 = ((status.get("model") or {}).get("pt2") or {})
        market_regime = status.get("market_regime") or {}
        compact_status = _compact_status_for_pack(status)
        visible_rows = list(visible_rows or [])
        suppressed_rows = list(suppressed_rows or [])
        informational_rows = list(informational_rows or [])
        overflow_rows = list(overflow_rows or [])
        all_rows = visible_rows + suppressed_rows + informational_rows + overflow_rows
        finished_at = scan.get("finished_at_utc") or scan.get("heartbeat_utc") or now_iso
        entry_utc = finished_at
        evaluation_due = (datetime.fromisoformat(finished_at) + timedelta(minutes=int(target.get("horizon_minutes") or self.config.target_horizon_minutes) + self.config.review_outcome_buffer_minutes)).isoformat()

        status_path = run_dir / "run_status.json"
        visible_path = run_dir / "visible_rows.json"
        suppressed_path = run_dir / "suppressed_rows.json"
        summary_path = run_dir / "summary.txt"
        atomic_write_json(status_path, compact_status)
        atomic_write_json(visible_path, visible_rows)
        atomic_write_json(suppressed_path, suppressed_rows)
        recent_resolved = self._load_recent_resolved_rows(model_fingerprint=str(model_pt2.get("model_fingerprint") or model_pt2.get("trained_at_utc") or "unknown"))
        recent_evidence_summary = self._build_recent_evidence_summary(recent_resolved, model_fingerprint=str(model_pt2.get("model_fingerprint") or model_pt2.get("trained_at_utc") or "unknown"))
        summary_txt = self._build_summary_text(run_id=run_id, app_version=str(model_pt2.get("app_version") or APP_VERSION), status=compact_status, visible_rows=visible_rows, suppressed_rows=suppressed_rows + informational_rows + overflow_rows, outcomes=None, recent_evidence_summary=recent_evidence_summary)
        summary_path.write_text(summary_txt, encoding="utf-8")

        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO review_runs (
                    run_id, scan_started_utc, scan_finished_utc, trigger_source, app_version, model_fingerprint,
                    target_move_pct, target_horizon_minutes, quality_max_mae, quality_min_end_ret,
                    live_universe_mode, market_regime_state, market_regime_actionability, cooldown_active,
                    visible_rows_count, suppressed_rows_count, evaluation_due_utc, review_status_path,
                    review_visible_rows_path, review_suppressed_rows_path, review_summary_path,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    scan.get("started_at_utc"),
                    finished_at,
                    trigger_source,
                    str(model_pt2.get("app_version") or APP_VERSION),
                    str(model_pt2.get("model_fingerprint") or model_pt2.get("trained_at_utc") or "unknown"),
                    float(target.get("move_pct") or self.config.target_move_pct),
                    int(target.get("horizon_minutes") or self.config.target_horizon_minutes),
                    float(target.get("quality_max_mae") or self.config.quality_max_mae),
                    float(target.get("quality_min_end_ret") or self.config.quality_min_end_ret),
                    str(status.get("live_universe_mode_effective") or status.get("live_universe_mode") or "unknown"),
                    str(market_regime.get("state") or "unknown"),
                    str(market_regime.get("effective_actionability_state") or market_regime.get("actionability_state") or "unknown"),
                    int(bool(market_regime.get("cooldown_active"))),
                    len(visible_rows),
                    len(suppressed_rows) + len(informational_rows) + len(overflow_rows),
                    evaluation_due,
                    str(status_path),
                    str(visible_path),
                    str(suppressed_path),
                    str(summary_path),
                    now_iso,
                    now_iso,
                ),
            )
            def _rank_int(value):
                try:
                    if value is None or value == "":
                        return None
                    value = int(value)
                    return value if value > 0 else None
                except Exception:
                    return None

            row_records = []
            for row_type, rows in (("visible", visible_rows), ("suppressed", suppressed_rows), ("informational", informational_rows), ("overflow", overflow_rows)):
                for row in rows:
                    row_records.append(
                        (
                            run_id,
                            row_type,
                            str(row.get("suppression_reason") or "") if row_type in _non_visible_row_types() else None,
                            str(row.get("symbol") or ""),
                            entry_utc,
                            float(row.get("price") or 0.0),
                            float(target.get("move_pct") or self.config.target_move_pct),
                            int(target.get("horizon_minutes") or self.config.target_horizon_minutes),
                            float(target.get("quality_max_mae") or self.config.quality_max_mae),
                            float(target.get("quality_min_end_ret") or self.config.quality_min_end_ret),
                            str(row.get("candidate_stage") or ""),
                            _f(row.get("prob_2_model")),
                            _f(row.get("prob_2_pre_regime")),
                            _f(row.get("prob_2_rank")),
                            _f(row.get("prob_2")),
                            _f(row.get("live_score")),
                            _f(row.get("opportunity_score")),
                            _f(row.get("risk")),
                            str(row.get("liquidity_tier") or ""),
                            str(row.get("actionability_tier") or ""),
                            str(row.get("actionability_type") or ""),
                            str(row.get("actionability_reason") or ""),
                            str(row.get("policy_constraint_reason") or ""),
                            str(row.get("suppression_reason_detail") or ""),
                            _f(row.get("pre_policy_score") if row.get("pre_policy_score") is not None else row.get("prob_2_pre_regime")),
                            _rank_int(row.get("pre_policy_rank") or row.get("candidate_rank_all")),
                            _rank_int(row.get("candidate_rank_all")),
                            _rank_int(row.get("informational_rank")),
                            _f(row.get("live_threshold")),
                            _f(row.get("validated_floor")),
                            _f(row.get("distance_to_validated")),
                            _f(row.get("distance_to_live_threshold")),
                            str(row.get("score_band") or ""),
                            str(row.get("visibility_band") or ""),
                            str(row.get("visibility_band_label") or ""),
                            str(row.get("score_band_label") or ""),
                            str(row.get("monitor_priority") or ""),
                            str(row.get("contract_truth_state") or row.get("tail_trust_state") or ""),
                            str(row.get("contract_truth_semantics") or row.get("probability_semantics") or ""),
                            str(row.get("temporal_tail_state") or ""),
                            str(row.get("temporal_tail_semantics") or ""),
                            str(row.get("market_regime_state") or ""),
                            int(bool(row.get("cooldown_active"))),
                            json.dumps(row.get("reasons") or [], ensure_ascii=False),
                            0,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            int(row_type == "visible"),
                            now_iso,
                            now_iso,
                        )
                    )
            conn.executemany(
                """INSERT INTO review_run_rows (
                    run_id, row_type, suppression_reason, symbol, entry_utc, entry_price,
                    target_move_pct, target_horizon_minutes, quality_max_mae, quality_min_end_ret,
                    candidate_stage, prob_2_model, prob_2_pre_regime, prob_2_rank, prob_2,
                    live_score, opportunity_score, risk, liquidity_tier, actionability_tier,
                    actionability_type, actionability_reason, policy_constraint_reason,
                    suppression_reason_detail, pre_policy_score, pre_policy_rank, candidate_rank_all, informational_rank,
                    live_threshold, validated_floor, distance_to_validated, distance_to_live_threshold, score_band, visibility_band, visibility_band_label, score_band_label, monitor_priority,
                    contract_truth_state, contract_truth_semantics, temporal_tail_state,
                    temporal_tail_semantics, market_regime_state, cooldown_active, reasons_json,
                    resolved, resolve_utc, actual_high, actual_low, actual_close, raw_touched,
                    quality_touched, mae, end_ret, mfe, time_to_touch_minutes, was_actionable,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row_records,
            )
            conn.commit()
        scan_pack = self._build_run_pack(run_id, include_outcomes=False)
        self._update_run_pack(run_id, scan_pack, evaluated=False)
        return run_id

    def resolve_due_runs(self, max_runs: int = 20) -> int:
        if not self.config.review_packs_enabled:
            return 0
        now = _utc_now()
        with self._lock, self._connect() as conn:
            due_runs = conn.execute(
                "SELECT run_id FROM review_runs WHERE evaluation_complete = 0 AND evaluation_due_utc <= ? ORDER BY evaluation_due_utc ASC LIMIT ?",
                (now.isoformat(), max_runs),
            ).fetchall()
        total_resolved = 0
        for run in due_runs:
            total_resolved += self._resolve_run(str(run["run_id"]))
        return total_resolved

    def _resolve_run(self, run_id: str) -> int:
        with self._lock, self._connect() as conn:
            run = conn.execute("SELECT * FROM review_runs WHERE run_id = ?", (run_id,)).fetchone()
            if not run:
                return 0
            pending = conn.execute(
                "SELECT * FROM review_run_rows WHERE run_id = ? AND resolved = 0 ORDER BY symbol ASC",
                (run_id,),
            ).fetchall()
            if not pending:
                conn.execute(
                    "UPDATE review_runs SET evaluation_complete = 1, evaluation_finished_utc = ?, updated_at_utc = ? WHERE run_id = ?",
                    (_utc_now_iso(), _utc_now_iso(), run_id),
                )
                conn.commit()
                self._build_and_store_evaluated_pack(run_id)
                return 0
            conn.execute(
                "UPDATE review_runs SET evaluation_started_utc = COALESCE(evaluation_started_utc, ?), updated_at_utc = ? WHERE run_id = ?",
                (_utc_now_iso(), _utc_now_iso(), run_id),
            )
            conn.commit()
        by_symbol: Dict[str, List[sqlite3.Row]] = {}
        for row in pending:
            by_symbol.setdefault(str(row["symbol"]), []).append(row)
        resolved = 0
        for symbol, rows in by_symbol.items():
            try:
                bars_needed = self._bars_needed_for_rows(rows)
                df = self.client.get_candles(symbol, bars_needed)
                if df.empty:
                    continue
                df = df.sort_values("ts").reset_index(drop=True)
                df["ts_dt"] = pd_to_datetime(df["ts"])
                for row in rows:
                    outcome = self._resolve_one_row(dict(row), df)
                    if outcome is None:
                        continue
                    self._write_row_outcome(int(row["id"]), outcome)
                    resolved += 1
                time.sleep(self.config.request_pause_seconds)
            except Exception as exc:  # pragma: no cover
                logger.warning("review_run_resolve_symbol_failed run_id=%s symbol=%s error=%s", run_id, symbol, exc)
        with self._lock, self._connect() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM review_run_rows WHERE run_id = ? AND resolved = 0",
                (run_id,),
            ).fetchone()[0]
            if int(remaining) == 0:
                now_iso = _utc_now_iso()
                conn.execute(
                    "UPDATE review_runs SET evaluation_complete = 1, evaluation_finished_utc = ?, updated_at_utc = ? WHERE run_id = ?",
                    (now_iso, now_iso, run_id),
                )
                conn.commit()
                self._build_and_store_evaluated_pack(run_id)
            else:
                conn.execute("UPDATE review_runs SET updated_at_utc = ? WHERE run_id = ?", (_utc_now_iso(), run_id))
                conn.commit()
        return resolved

    def _bars_needed_for_rows(self, rows: List[sqlite3.Row]) -> int:
        earliest = min(datetime.fromisoformat(str(r["entry_utc"])) for r in rows)
        max_horizon = max(int(r["target_horizon_minutes"] or self.config.target_horizon_minutes) for r in rows)
        minutes_since = max(1, (_utc_now() - earliest).total_seconds() / 60)
        bars = int(minutes_since / 5) + max(1, max_horizon // 5) + 24
        return min(max(bars, 100), 2400)

    def _resolve_one_row(self, row: dict, df) -> dict | None:
        scan_time = datetime.fromisoformat(str(row["entry_utc"]))
        entry_price = float(row["entry_price"] or 0.0)
        if entry_price <= 0:
            return None
        target_move_pct = float(row.get("target_move_pct") or self.config.target_move_pct)
        target_horizon_minutes = int(row.get("target_horizon_minutes") or self.config.target_horizon_minutes)
        quality_max_mae = float(row.get("quality_max_mae") or self.config.quality_max_mae)
        quality_min_end_ret = float(row.get("quality_min_end_ret") or self.config.quality_min_end_ret)
        horizon_bars = max(1, target_horizon_minutes // 5)
        future = df.loc[df["ts_dt"] > scan_time].head(horizon_bars)
        if len(future) < max(1, horizon_bars // 2):
            return None
        actual_high = float(future["high"].max())
        actual_low = float(future["low"].min())
        actual_close = float(future["close"].iloc[-1])
        target_px = entry_price * (1.0 + target_move_pct)
        touch_rows = future.loc[future["high"] >= target_px]
        raw_touched = int(not touch_rows.empty)
        first_touch_minutes = None
        if raw_touched:
            first_touch = touch_rows.iloc[0]["ts_dt"]
            first_touch_minutes = int(max(0, round((first_touch - scan_time).total_seconds() / 60.0)))
        mae = (actual_low / entry_price) - 1.0
        mfe = (actual_high / entry_price) - 1.0
        end_ret = (actual_close / entry_price) - 1.0
        quality_touched = int(raw_touched and mae > quality_max_mae and end_ret > quality_min_end_ret)
        return {
            "resolve_utc": _utc_now_iso(),
            "actual_high": round(actual_high, 8),
            "actual_low": round(actual_low, 8),
            "actual_close": round(actual_close, 8),
            "raw_touched": raw_touched,
            "quality_touched": quality_touched,
            "mae": round(mae, 6),
            "mfe": round(mfe, 6),
            "end_ret": round(end_ret, 6),
            "time_to_touch_minutes": first_touch_minutes,
        }

    def _write_row_outcome(self, row_id: int, outcome: dict) -> None:
        now_iso = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE review_run_rows SET
                    resolved = 1,
                    resolve_utc = ?,
                    actual_high = ?,
                    actual_low = ?,
                    actual_close = ?,
                    raw_touched = ?,
                    quality_touched = ?,
                    mae = ?,
                    end_ret = ?,
                    mfe = ?,
                    time_to_touch_minutes = ?,
                    updated_at_utc = ?
                   WHERE id = ?""",
                (
                    outcome["resolve_utc"],
                    outcome["actual_high"],
                    outcome["actual_low"],
                    outcome["actual_close"],
                    outcome["raw_touched"],
                    outcome["quality_touched"],
                    outcome["mae"],
                    outcome["end_ret"],
                    outcome["mfe"],
                    outcome["time_to_touch_minutes"],
                    now_iso,
                    row_id,
                ),
            )
            conn.commit()

    def _load_run_rows(self, run_id: str) -> List[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM review_run_rows WHERE run_id = ? ORDER BY CASE row_type WHEN 'visible' THEN 0 WHEN 'suppressed' THEN 1 WHEN 'informational' THEN 2 WHEN 'overflow' THEN 3 ELSE 9 END ASC, COALESCE(NULLIF(informational_rank, 0), NULLIF(pre_policy_rank, 0), NULLIF(candidate_rank_all, 0), 999999) ASC, symbol ASC", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def _load_run(self, run_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM review_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_runs(self, limit: int = 20) -> List[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, scan_finished_utc, trigger_source, app_version, market_regime_state, market_regime_actionability, visible_rows_count, suppressed_rows_count, evaluation_complete, latest_scan_pack_path, latest_evaluated_pack_path, review_status_path FROM review_runs ORDER BY scan_finished_utc DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_runs_for_app_version(self, app_version: str, limit: int | None = None) -> List[dict]:
        query = (
            "SELECT run_id, scan_finished_utc, trigger_source, app_version, market_regime_state, market_regime_actionability, "
            "visible_rows_count, suppressed_rows_count, evaluation_complete, latest_scan_pack_path, latest_evaluated_pack_path, review_status_path "
            "FROM review_runs WHERE app_version = ? ORDER BY scan_finished_utc DESC"
        )
        params: list[Any] = [str(app_version or APP_VERSION)]
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def _load_rows_for_run_ids(self, run_ids: List[str], *, resolved_only: bool = False) -> List[dict]:
        ids = [str(x) for x in (run_ids or []) if x]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        where = f"run_id IN ({placeholders})"
        if resolved_only:
            where += " AND resolved = 1"
        query = (
            f"SELECT * FROM review_run_rows WHERE {where} ORDER BY entry_utc DESC, "
            "COALESCE(NULLIF(informational_rank, 0), NULLIF(pre_policy_rank, 0), NULLIF(candidate_rank_all, 0), 999999) ASC, symbol ASC"
        )
        with self._lock, self._connect() as conn:
            return [dict(r) for r in conn.execute(query, ids).fetchall()]

    def _current_version_regime_rows(self, runs: List[dict]) -> List[dict]:
        buckets: Dict[tuple[str, str], dict] = {}
        for run in list(runs or []):
            key = (str(run.get("market_regime_state") or "unknown"), str(run.get("market_regime_actionability") or "unknown"))
            bucket = buckets.setdefault(key, {
                "market_regime_state": key[0],
                "market_regime_actionability": key[1],
                "run_count": 0,
                "evaluated_run_count": 0,
                "visible_rows": 0,
                "suppressed_rows": 0,
            })
            bucket["run_count"] += 1
            bucket["evaluated_run_count"] += 1 if bool(run.get("evaluation_complete")) else 0
            bucket["visible_rows"] += int(run.get("visible_rows_count") or 0)
            bucket["suppressed_rows"] += int(run.get("suppressed_rows_count") or 0)
        return sorted(buckets.values(), key=lambda r: (r["market_regime_state"], r["market_regime_actionability"]))

    def _build_scan_score_diagnostics_summary(self, runs: List[dict]) -> dict:
        diagnostics = []
        for run in runs:
            status_path = run.get("review_status_path")
            if not status_path:
                continue
            status = read_json(status_path, {})
            diag = status.get("score_diagnostics") or {}
            if diag:
                diagnostics.append(diag)
        if not diagnostics:
            return {"available": False, "scan_count": 0, "counts_above_thresholds": []}
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        summary_rows = []
        for threshold in thresholds:
            live_count = pre_count = model_count = 0
            for diag in diagnostics:
                for row in (diag.get("counts_above_thresholds") or []):
                    try:
                        if float(row.get("threshold")) == float(threshold):
                            live_count += int(row.get("live_count") or 0)
                            pre_count += int(row.get("pre_policy_count") or 0)
                            model_count += int(row.get("model_count") or 0)
                    except Exception:
                        continue
            summary_rows.append({"threshold": round(float(threshold), 2), "live_count": live_count, "pre_policy_count": pre_count, "model_count": model_count})
        return {
            "available": True,
            "scan_count": len(diagnostics),
            "guardrail_cap_values": sorted({round(float(d.get("guardrail_cap") or 0.0), 4) for d in diagnostics if d.get("guardrail_cap") is not None}),
            "headline": diagnostics[-1].get("headline"),
            "counts_above_thresholds": summary_rows,
        }

    def _candidate_quality_rows_from_status(self, status: dict) -> List[dict]:
        quality = status.get("candidate_quality") or {}
        stage1_by_tier = quality.get("stage1_by_tier") or {}
        stage2_by_tier = quality.get("stage2_by_tier") or {}
        rows = []
        for tier in ("tier1", "tier2", "tier3"):
            s1 = stage1_by_tier.get(tier) or {}
            s2 = stage2_by_tier.get(tier) or {}
            rows.append({
                "liquidity_tier": tier,
                "stage1_feature_ready": int(s1.get("feature_ready") or 0),
                "stage1_blocked": int(s1.get("blocked") or 0),
                "stage1_selected": int(s1.get("selected") or 0),
                "stage1_selected_share": s1.get("selected_share"),
                "stage2_scored": int(s2.get("scored") or 0),
                "stage2_visible": int(s2.get("visible") or 0),
                "stage2_hidden": int(s2.get("hidden") or 0),
                "stage2_live_max": ((s2.get("live_score") or {}).get("max")),
                "stage2_live_median": ((s2.get("live_score") or {}).get("median")),
                "stage2_count_ge_0_30": int(s2.get("count_ge_0_30") or 0),
                "stage2_count_ge_0_35": int(s2.get("count_ge_0_35") or 0),
                "stage2_count_ge_0_45": int(s2.get("count_ge_0_45") or 0),
            })
        return rows

    def _build_latest_stage1_omission_audit(self, runs: List[dict]) -> dict:
        for run in runs:
            status = read_json(run.get("review_status_path"), {}) if run.get("review_status_path") else {}
            audit = status.get("stage1_omission_audit") or {}
            if audit:
                return audit
        return {"available": False}

    def _build_latest_stage1_selection_repair_review(self, runs: List[dict]) -> dict:
        for run in runs:
            status = read_json(run.get("review_status_path"), {}) if run.get("review_status_path") else {}
            review = status.get("stage1_selection_repair_review") or {}
            if review:
                return review
        return {"available": False}

    def _build_latest_threshold_experiment_review(self, runs: List[dict]) -> dict:
        for run in runs:
            status = read_json(run.get("review_status_path"), {}) if run.get("review_status_path") else {}
            review = status.get("threshold_experiment_review") or {}
            if review:
                return review
        return {"available": False}

    def _build_candidate_quality_summary(self, runs: List[dict]) -> dict:
        status_rows = []
        for run in runs:
            status_path = run.get("review_status_path")
            if not status_path:
                continue
            status = read_json(status_path, {})
            if not (status.get("candidate_quality") or {}):
                continue
            status_rows.append(self._candidate_quality_rows_from_status(status))
        if not status_rows:
            return {"available": False, "tiers": []}
        agg = {tier: {"liquidity_tier": tier, "scans": 0, "stage1_feature_ready": 0, "stage1_blocked": 0, "stage1_selected": 0, "stage2_scored": 0, "stage2_visible": 0, "stage2_hidden": 0, "stage2_count_ge_0_30": 0, "stage2_count_ge_0_35": 0, "stage2_count_ge_0_45": 0, "max_live_score": None} for tier in ("tier1", "tier2", "tier3")}
        for rows in status_rows:
            for row in rows:
                tier = str(row.get("liquidity_tier") or "tier3")
                bucket = agg[tier]
                bucket["scans"] += 1
                for key in ("stage1_feature_ready", "stage1_blocked", "stage1_selected", "stage2_scored", "stage2_visible", "stage2_hidden", "stage2_count_ge_0_30", "stage2_count_ge_0_35", "stage2_count_ge_0_45"):
                    bucket[key] += int(row.get(key) or 0)
                max_live = row.get("stage2_live_max")
                if max_live is not None:
                    bucket["max_live_score"] = max(float(max_live), float(bucket["max_live_score"] or 0.0))
        tier_rows = []
        for tier in ("tier1", "tier2", "tier3"):
            bucket = agg[tier]
            denom = max(1, int(bucket["stage1_feature_ready"]) - int(bucket["stage1_blocked"]))
            bucket["avg_stage1_selected_share"] = round(float(bucket["stage1_selected"]) / float(denom), 4)
            if bucket["max_live_score"] is not None:
                bucket["max_live_score"] = round(float(bucket["max_live_score"]), 4)
            tier_rows.append(bucket)
        return {"available": True, "tiers": tier_rows}

    def _build_cohort_symbol_summary(self, runs: List[dict]) -> dict:
        buckets: Dict[str, dict] = {}
        for run in runs:
            status_path = run.get("review_status_path")
            if not status_path:
                continue
            status = read_json(status_path, {})
            trace = ((status.get("candidate_quality") or {}).get("stage1_to_stage2_trace") or [])
            for row in trace:
                symbol = str(row.get("symbol") or "")
                if not symbol:
                    continue
                bucket = buckets.setdefault(symbol, {
                    "symbol": symbol,
                    "liquidity_tier": row.get("liquidity_tier"),
                    "selected_scans": 0,
                    "visible_scans": 0,
                    "hidden_scans": 0,
                    "max_live_score": None,
                    "count_ge_0_30": 0,
                    "count_ge_0_35": 0,
                })
                if bool(row.get("stage1_selected")):
                    bucket["selected_scans"] += 1
                final_row_type = str(row.get("final_row_type") or "")
                if final_row_type == "visible":
                    bucket["visible_scans"] += 1
                elif final_row_type:
                    bucket["hidden_scans"] += 1
                score = row.get("live_score")
                if score not in (None, ""):
                    score_f = float(score)
                    bucket["max_live_score"] = max(float(bucket["max_live_score"] or 0.0), score_f)
                    if score_f >= 0.30:
                        bucket["count_ge_0_30"] += 1
                    if score_f >= 0.35:
                        bucket["count_ge_0_35"] += 1
        rows = sorted(buckets.values(), key=lambda r: (-(r.get("selected_scans") or 0), -(r.get("max_live_score") or 0.0), r.get("symbol") or ""))
        for row in rows:
            if row.get("max_live_score") is not None:
                row["max_live_score"] = round(float(row["max_live_score"]), 4)
        return {"available": bool(rows), "rows": rows}

    def _build_symbol_repeatability_summary(self, rows: List[dict]) -> dict:
        buckets: Dict[str, dict] = {}
        for row in list(rows or []):
            symbol = str(row.get("symbol") or "")
            if not symbol:
                continue
            bucket = buckets.setdefault(symbol, {
                "symbol": symbol,
                "resolved_rows": 0,
                "visible_rows": 0,
                "non_visible_rows": 0,
                "quality_hits": 0,
                "raw_hits": 0,
                "visible_quality_hits": 0,
                "non_visible_quality_hits": 0,
                "end_ret_sum": 0.0,
                "end_ret_count": 0,
                "mae_sum": 0.0,
                "mae_count": 0,
                "max_live_score": None,
                "count_ge_0_35": 0,
                "count_ge_0_45": 0,
                "count_ge_0_60": 0,
            })
            bucket["resolved_rows"] += 1
            is_visible = str(row.get("row_type") or "") == "visible"
            if is_visible:
                bucket["visible_rows"] += 1
            elif _is_non_visible_row(row):
                bucket["non_visible_rows"] += 1
            quality = int(row.get("quality_touched") or 0)
            raw = int(row.get("raw_touched") or 0)
            bucket["quality_hits"] += quality
            bucket["raw_hits"] += raw
            if is_visible:
                bucket["visible_quality_hits"] += quality
            elif _is_non_visible_row(row):
                bucket["non_visible_quality_hits"] += quality
            end_ret = _f(row.get("end_ret"))
            if end_ret is not None:
                bucket["end_ret_sum"] += end_ret
                bucket["end_ret_count"] += 1
            mae = _f(row.get("mae"))
            if mae is not None:
                bucket["mae_sum"] += mae
                bucket["mae_count"] += 1
            score = _f(row.get("live_score"))
            if score is not None:
                bucket["max_live_score"] = max(float(bucket["max_live_score"] or 0.0), score)
                if score >= 0.35:
                    bucket["count_ge_0_35"] += 1
                if score >= 0.45:
                    bucket["count_ge_0_45"] += 1
                if score >= 0.60:
                    bucket["count_ge_0_60"] += 1
        result_rows = []
        for bucket in buckets.values():
            resolved = max(1, int(bucket["resolved_rows"]))
            visible = int(bucket["visible_rows"])
            non_visible = int(bucket["non_visible_rows"])
            result_rows.append({
                "symbol": bucket["symbol"],
                "resolved_rows": resolved,
                "visible_rows": visible,
                "non_visible_rows": non_visible,
                "quality_hits": int(bucket["quality_hits"]),
                "raw_hits": int(bucket["raw_hits"]),
                "quality_hit_rate": round(float(bucket["quality_hits"]) / resolved, 4),
                "raw_hit_rate": round(float(bucket["raw_hits"]) / resolved, 4),
                "visible_quality_hit_rate": round(float(bucket["visible_quality_hits"]) / max(1, visible), 4) if visible else None,
                "non_visible_quality_hit_rate": round(float(bucket["non_visible_quality_hits"]) / max(1, non_visible), 4) if non_visible else None,
                "avg_end_ret": round(float(bucket["end_ret_sum"]) / max(1, int(bucket["end_ret_count"])) , 6) if bucket["end_ret_count"] else None,
                "avg_mae": round(float(bucket["mae_sum"]) / max(1, int(bucket["mae_count"])) , 6) if bucket["mae_count"] else None,
                "max_live_score": round(float(bucket["max_live_score"]), 4) if bucket["max_live_score"] is not None else None,
                "count_ge_0_35": int(bucket["count_ge_0_35"]),
                "count_ge_0_45": int(bucket["count_ge_0_45"]),
                "count_ge_0_60": int(bucket["count_ge_0_60"]),
            })
        result_rows.sort(key=lambda r: (-(r.get("count_ge_0_60") or 0), -(r.get("count_ge_0_45") or 0), -(r.get("quality_hits") or 0), -(r.get("max_live_score") or 0.0), r.get("symbol") or ""))
        return {
            "available": bool(result_rows),
            "rows": result_rows,
            "top_symbols": result_rows[:10],
        }

    def _build_outlier_concentration_summary(self, rows: List[dict]) -> dict:
        rows = list(rows or [])
        thresholds = [0.35, 0.45, 0.60]
        threshold_buckets: Dict[str, dict] = {}
        for threshold in thresholds:
            band_rows = [r for r in rows if _f(r.get("live_score")) is not None and float(r.get("live_score") or 0.0) >= threshold]
            counts: Dict[str, int] = {}
            for row in band_rows:
                symbol = str(row.get("symbol") or "")
                if not symbol:
                    continue
                counts[symbol] = counts.get(symbol, 0) + 1
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            total = len(band_rows)
            top1_count = ranked[0][1] if ranked else 0
            top3_count = sum(c for _, c in ranked[:3]) if ranked else 0
            threshold_buckets[f"{threshold:.2f}"] = {
                "threshold": round(float(threshold), 2),
                "row_count": total,
                "unique_symbols": len(ranked),
                "top_symbol": ranked[0][0] if ranked else None,
                "top_symbol_count": top1_count,
                "top_symbol_share": round(float(top1_count) / max(1, total), 4) if total else None,
                "top_3_share": round(float(top3_count) / max(1, total), 4) if total else None,
                "top_symbols": [{"symbol": sym, "count": count} for sym, count in ranked[:10]],
            }
        quality_rows = [r for r in rows if int(r.get("quality_touched") or 0) == 1]
        quality_counts: Dict[str, int] = {}
        for row in quality_rows:
            symbol = str(row.get("symbol") or "")
            if not symbol:
                continue
            quality_counts[symbol] = quality_counts.get(symbol, 0) + 1
        ranked_quality = sorted(quality_counts.items(), key=lambda item: (-item[1], item[0]))
        quality_total = len(quality_rows)
        top_quality_count = ranked_quality[0][1] if ranked_quality else 0
        top3_quality = sum(c for _, c in ranked_quality[:3]) if ranked_quality else 0
        focus = threshold_buckets.get("0.45") or threshold_buckets.get("0.35") or {}
        focus_share = focus.get("top_symbol_share")
        if not rows:
            headline = "No resolved evidence rows yet"
            summary = "Wait for evaluated packs before judging concentration risk."
        elif (focus.get("row_count") or 0) == 0:
            headline = "No higher-band resolved rows yet"
            summary = "There are not yet any resolved rows in the stronger research bands, so concentration risk cannot be judged there."
        elif focus_share is not None and focus_share >= 0.5:
            headline = "Stronger rows are concentrated in a small number of symbols"
            summary = "Treat standout names carefully until you know whether the strongest rows are repeatable across multiple symbols rather than driven by one or two recurring outliers."
        else:
            headline = "Stronger rows are distributed across multiple symbols"
            summary = "Higher-band resolved rows are not dominated by a single symbol, which is a healthier sign for repeatability."
        return {
            "available": bool(rows),
            "headline": headline,
            "summary": summary,
            "thresholds": threshold_buckets,
            "quality_hits": {
                "row_count": quality_total,
                "unique_symbols": len(ranked_quality),
                "top_symbol": ranked_quality[0][0] if ranked_quality else None,
                "top_symbol_count": top_quality_count,
                "top_symbol_share": round(float(top_quality_count) / max(1, quality_total), 4) if quality_total else None,
                "top_3_share": round(float(top3_quality) / max(1, quality_total), 4) if quality_total else None,
                "top_symbols": [{"symbol": sym, "count": count} for sym, count in ranked_quality[:10]],
            },
        }

    def _build_regime_evidence_summary(self, runs: List[dict], resolved_rows: List[dict]) -> dict:
        run_lookup = {str(r.get("run_id")): dict(r) for r in list(runs or []) if r.get("run_id")}
        buckets: Dict[tuple[str, str], List[dict]] = {}
        for row in list(resolved_rows or []):
            run = run_lookup.get(str(row.get("run_id") or ""), {})
            state = str(run.get("market_regime_state") or row.get("market_regime_state") or "unknown")
            actionability = str(run.get("market_regime_actionability") or "unknown")
            buckets.setdefault((state, actionability), []).append(dict(row))
        threshold_levels = [0.30, 0.35, 0.40, 0.45]
        rows = []
        flattened_threshold_rows = []
        for (state, actionability), bucket_rows in sorted(buckets.items(), key=lambda item: item[0]):
            visible_rows = [r for r in bucket_rows if str(r.get("row_type") or "") == "visible"]
            non_visible_rows = [r for r in bucket_rows if _is_non_visible_row(r)]
            threshold_bands = []
            for threshold in threshold_levels:
                band_rows = [r for r in bucket_rows if _f(r.get("live_score")) is not None and float(r.get("live_score") or 0.0) >= threshold]
                band = {
                    "threshold": round(float(threshold), 2),
                    "count": len(band_rows),
                    "visible_count": sum(1 for r in band_rows if str(r.get("row_type") or "") == "visible"),
                    "non_visible_count": sum(1 for r in band_rows if _is_non_visible_row(r)),
                    "quality_hit_rate": _rate(band_rows, "quality_touched"),
                    "raw_hit_rate": _rate(band_rows, "raw_touched"),
                    "avg_end_ret": _avg_metric(band_rows, "end_ret"),
                    "avg_mae": _avg_metric(band_rows, "mae"),
                }
                threshold_bands.append(band)
                flattened_threshold_rows.append({
                    "market_regime_state": state,
                    "market_regime_actionability": actionability,
                    **band,
                })
            rows.append({
                "market_regime_state": state,
                "market_regime_actionability": actionability,
                "resolved_rows": len(bucket_rows),
                "visible_rows": len(visible_rows),
                "non_visible_rows": len(non_visible_rows),
                "visible_quality_hit_rate": _rate(visible_rows, "quality_touched"),
                "non_visible_quality_hit_rate": _rate(non_visible_rows, "quality_touched"),
                "visible_raw_hit_rate": _rate(visible_rows, "raw_touched"),
                "non_visible_raw_hit_rate": _rate(non_visible_rows, "raw_touched"),
                "visible_avg_end_ret": _avg_metric(visible_rows, "end_ret"),
                "non_visible_avg_end_ret": _avg_metric(non_visible_rows, "end_ret"),
                "visible_avg_mae": _avg_metric(visible_rows, "mae"),
                "non_visible_avg_mae": _avg_metric(non_visible_rows, "mae"),
                "threshold_bands": threshold_bands,
            })
        headline = None
        summary = None
        if not rows:
            headline = "No regime-sliced evaluated evidence yet"
            summary = "Wait for more evaluated packs before judging whether the simplified path behaves differently across green, amber, and red regimes."
        else:
            best = max(rows, key=lambda r: float(r.get("visible_quality_hit_rate") or -1.0))
            worst = min(rows, key=lambda r: float(r.get("visible_quality_hit_rate") or 2.0))
            headline = "Regime-sliced evaluated evidence is available"
            best_rate = best.get('visible_quality_hit_rate')
            worst_rate = worst.get('visible_quality_hit_rate')
            best_pct = f"{float(best_rate) * 100.0:.2f}%" if best_rate is not None else "-"
            worst_pct = f"{float(worst_rate) * 100.0:.2f}%" if worst_rate is not None else "-"
            summary = (
                f"Best visible quality-hit regime bucket so far: {best.get('market_regime_state')} / {best.get('market_regime_actionability')} "
                f"at {best_pct}. Weakest visible quality-hit bucket: {worst.get('market_regime_state')} / {worst.get('market_regime_actionability')} "
                f"at {worst_pct}."
            )
        return {"available": bool(rows), "headline": headline, "summary": summary, "rows": rows, "threshold_rows": flattened_threshold_rows}

    def _build_current_version_summary(self, *, version: str, runs: List[dict]) -> dict:
        runs = list(runs or [])
        deployed_since_utc = min((r.get("scan_finished_utc") or "") for r in runs if r.get("scan_finished_utc")) or None
        evaluated_runs = [r for r in runs if bool(r.get("evaluation_complete"))]
        resolved_rows = self._load_rows_for_run_ids([str(r.get("run_id")) for r in evaluated_runs], resolved_only=True)
        live_scores = [float(r.get("live_score")) for r in resolved_rows if r.get("live_score") not in (None, "")]
        pre_scores = [float(r.get("pre_policy_score")) for r in resolved_rows if r.get("pre_policy_score") not in (None, "")]
        visible_rows = [r for r in resolved_rows if str(r.get("row_type") or "") == "visible"]
        non_visible_rows = [r for r in resolved_rows if _is_non_visible_row(r)]
        symbol_repeatability = self._build_symbol_repeatability_summary(resolved_rows)
        outlier_concentration = self._build_outlier_concentration_summary(resolved_rows)
        regime_evidence = self._build_regime_evidence_summary(runs, resolved_rows)
        regime_semantics_note = None
        for run in runs:
            status = read_json(run.get("review_status_path"), {}) if run.get("review_status_path") else {}
            market_regime = status.get("market_regime") or {}
            note = market_regime.get("effective_actionability_note")
            if note:
                regime_semantics_note = str(note)
                break
        thresholds = [0.30, 0.35, 0.40, 0.45]
        threshold_rows = []
        for threshold in thresholds:
            band_rows = [r for r in resolved_rows if _f(r.get("live_score")) is not None and float(r.get("live_score") or 0.0) >= threshold]
            threshold_rows.append({
                "threshold": round(float(threshold), 2),
                "count": len(band_rows),
                "visible_count": sum(1 for r in band_rows if str(r.get("row_type") or "") == "visible"),
                "non_visible_count": sum(1 for r in band_rows if _is_non_visible_row(r)),
                "quality_hit_rate": _rate(band_rows, "quality_touched"),
                "raw_hit_rate": _rate(band_rows, "raw_touched"),
                "avg_end_ret": _avg_metric(band_rows, "end_ret"),
                "avg_mae": _avg_metric(band_rows, "mae"),
            })
        suppression_quality_hits = {
            reason: sum(1 for r in non_visible_rows if str(r.get("suppression_reason") or "other") == reason and int(r.get("quality_touched") or 0) == 1)
            for reason in ("display_trim", "threshold", "regime", "cooldown", "other")
        }
        summary = {
            "app_version": version,
            "generated_at_utc": _utc_now_iso(),
            "deployed_since_utc": deployed_since_utc,
            "scan_pack_count": len(runs),
            "evaluated_pack_count": len(evaluated_runs),
            "total_visible_rows": sum(int(r.get("visible_rows_count") or 0) for r in runs),
            "total_suppressed_rows": sum(int(r.get("suppressed_rows_count") or 0) for r in runs),
            "regime_breakdown": self._current_version_regime_rows(runs),
            "regime_semantics_note": regime_semantics_note,
            "regime_evidence": regime_evidence,
            "scan_score_diagnostics": self._build_scan_score_diagnostics_summary(runs),
            "candidate_quality": self._build_candidate_quality_summary(runs),
            "stage1_omission_audit_latest": self._build_latest_stage1_omission_audit(runs),
            "stage1_selection_repair_review_latest": self._build_latest_stage1_selection_repair_review(runs),
            "threshold_experiment_review_latest": self._build_latest_threshold_experiment_review(runs),
            "cohort_symbols": self._build_cohort_symbol_summary(runs),
            "symbol_repeatability": symbol_repeatability,
            "outlier_concentration": outlier_concentration,
            "evidence": {
                "available": bool(resolved_rows),
                "resolved_rows": len(resolved_rows),
                "visible_rows": len(visible_rows),
                "non_visible_rows": len(non_visible_rows),
                "visible_quality_hit_rate": _rate(visible_rows, "quality_touched"),
                "non_visible_quality_hit_rate": _rate(non_visible_rows, "quality_touched"),
                "visible_raw_hit_rate": _rate(visible_rows, "raw_touched"),
                "non_visible_raw_hit_rate": _rate(non_visible_rows, "raw_touched"),
                "visible_avg_end_ret": _avg_metric(visible_rows, "end_ret"),
                "non_visible_avg_end_ret": _avg_metric(non_visible_rows, "end_ret"),
                "visible_avg_mae": _avg_metric(visible_rows, "mae"),
                "non_visible_avg_mae": _avg_metric(non_visible_rows, "mae"),
                "display_trim_quality_hits": suppression_quality_hits.get("display_trim", 0),
                "threshold_quality_hits": suppression_quality_hits.get("threshold", 0),
                "regime_quality_hits": suppression_quality_hits.get("regime", 0),
                "cooldown_quality_hits": suppression_quality_hits.get("cooldown", 0),
                "score_range": {
                    "max_live_score": _quantile(live_scores, 1.0),
                    "p95_live_score": _quantile(live_scores, 0.95),
                    "median_live_score": _quantile(live_scores, 0.50),
                    "max_pre_policy_score": _quantile(pre_scores, 1.0),
                    "p95_pre_policy_score": _quantile(pre_scores, 0.95),
                    "median_pre_policy_score": _quantile(pre_scores, 0.50),
                },
                "threshold_bands": threshold_rows,
                "validated_bands_dormant": all(int(row.get("count") or 0) == 0 for row in threshold_rows if float(row.get("threshold") or 0.0) >= 0.45),
                "headline": None,
                "summary": None,
            },
        }
        if not resolved_rows:
            summary["evidence"]["headline"] = "No evaluated rows yet for this deployed version"
            summary["evidence"]["summary"] = "Wait for evaluated packs before judging whether validated score bands are reachable in live practice."
            return summary
        max_live = summary["evidence"]["score_range"]["max_live_score"]
        if max_live is not None and max_live < 0.45:
            summary["evidence"]["headline"] = "Validated research bands have not been reached in this deployment window"
            summary["evidence"]["summary"] = (
                f"Max live score since deployment is {max_live:.4f}; no resolved rows have reached 0.45+, so lowering the live floor alone would not be evidence-based."
            )
        else:
            summary["evidence"]["headline"] = "Some resolved rows have reached the lower research bands"
            summary["evidence"]["summary"] = "Use the threshold-band counts to judge whether live score ranges are approaching validated territory."
        concentration_headline = (outlier_concentration.get("headline") or "") if isinstance(outlier_concentration, dict) else ""
        if concentration_headline:
            summary["evidence"]["summary"] += f" {concentration_headline}."
        return summary

    def get_current_version_summary(self, app_version: str | None = None) -> dict:
        version = str(app_version or APP_VERSION)
        runs = self.get_runs_for_app_version(version)
        if not runs:
            raise FileNotFoundError(f"no review packs for app version {version}")
        return self._build_current_version_summary(version=version, runs=runs)

    def get_run(self, run_id: str) -> dict | None:
        run = self._load_run(run_id)
        if not run:
            return None
        status = read_json(run.get("review_status_path"), {})
        rows = self._load_run_rows(run_id)
        visible = [r for r in rows if r.get("row_type") == "visible"]
        suppressed = [r for r in rows if r.get("row_type") == "suppressed"]
        informational = [r for r in rows if r.get("row_type") == "informational"]
        overflow = [r for r in rows if r.get("row_type") == "overflow"]
        return {
            "run": run,
            "status": status,
            "visible_rows": visible,
            "suppressed_rows": suppressed,
            "informational_rows": informational,
            "overflow_rows": overflow,
            "outcome_summary": self._summarize_outcomes(rows),
            "policy_audit": self._policy_audit(rows, runs=[run]),
        }

    def get_policy_audit(self, *, hours: int = 24) -> dict:
        cutoff = (_utc_now() - timedelta(hours=hours)).isoformat()
        with self._lock, self._connect() as conn:
            runs = [dict(r) for r in conn.execute(
                "SELECT * FROM review_runs WHERE scan_finished_utc >= ? ORDER BY scan_finished_utc DESC LIMIT ?",
                (cutoff, self.config.review_max_runs_in_aggregate),
            ).fetchall()]
        run_ids = [str(r.get("run_id")) for r in runs if r.get("run_id")]
        rows: List[dict] = []
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            with self._lock, self._connect() as conn:
                rows = [dict(r) for r in conn.execute(
                    f"SELECT * FROM review_run_rows WHERE run_id IN ({placeholders}) ORDER BY entry_utc DESC, COALESCE(NULLIF(informational_rank, 0), NULLIF(pre_policy_rank, 0), NULLIF(candidate_rank_all, 0), 999999) ASC, symbol ASC",
                    run_ids,
                ).fetchall()]
        audit = self._policy_audit(rows, runs=runs)
        audit["hours"] = hours
        return audit

    def _load_recent_resolved_rows(self, *, model_fingerprint: str | None = None, limit: int | None = None, lookback_days: int | None = None) -> List[dict]:
        limit = max(1, int(limit or getattr(self.config, "review_pack_recent_resolved_limit", 250) or 250))
        lookback_days = max(1, int(lookback_days or getattr(self.config, "review_pack_recent_resolved_lookback_days", 30) or 30))
        params = []
        where = ["rr.resolved = 1"]
        cutoff = (_utc_now() - timedelta(days=lookback_days)).isoformat()
        where.append("COALESCE(rr.resolve_utc, r.scan_finished_utc) >= ?")
        params.append(cutoff)
        if model_fingerprint:
            where.append("r.model_fingerprint = ?")
            params.append(str(model_fingerprint))
        query = f"""
            SELECT rr.*,
                   r.scan_finished_utc AS source_run_finished_utc,
                   r.model_fingerprint AS source_model_fingerprint,
                   r.trigger_source AS source_trigger_source,
                   r.market_regime_state AS source_run_market_regime_state
            FROM review_run_rows rr
            JOIN review_runs r ON r.run_id = rr.run_id
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(rr.resolve_utc, r.scan_finished_utc) DESC, rr.symbol ASC
            LIMIT ?
        """
        params.append(limit)
        with self._lock, self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _build_recent_evidence_summary(self, rows: List[dict], *, model_fingerprint: str | None = None) -> dict:
        thresholds = [0.45, 0.50, 0.55, 0.60]
        rows = list(rows or [])
        summary = {
            "available": bool(rows),
            "model_fingerprint": model_fingerprint,
            "resolved_rows": len(rows),
            "quality_hit_rate": _rate(rows, "quality_touched"),
            "raw_hit_rate": _rate(rows, "raw_touched"),
            "avg_end_ret": _avg_metric(rows, "end_ret"),
            "avg_mae": _avg_metric(rows, "mae"),
            "threshold_bands": {},
        }
        for threshold in thresholds:
            band_rows = [r for r in rows if _f(r.get("live_score")) is not None and float(r.get("live_score") or 0.0) >= threshold]
            summary["threshold_bands"][f"{threshold:.2f}"] = {
                "count": len(band_rows),
                "quality_hit_rate": _rate(band_rows, "quality_touched"),
                "raw_hit_rate": _rate(band_rows, "raw_touched"),
                "avg_end_ret": _avg_metric(band_rows, "end_ret"),
                "avg_mae": _avg_metric(band_rows, "mae"),
                "visible_count": sum(1 for r in band_rows if str(r.get("row_type") or "") == "visible"),
                "non_visible_count": sum(1 for r in band_rows if _is_non_visible_row(r)),
            }
        return summary

    def _build_summary_text(self, *, run_id: str, app_version: str, status: dict, visible_rows: List[dict], suppressed_rows: List[dict], outcomes: List[dict] | None, recent_evidence_summary: dict | None = None) -> str:
        scan = status.get("scan") or {}
        regime = status.get("market_regime") or {}
        suppression = status.get("suppression_summary") or {}
        target = status.get("target") or {}
        decision = status.get("decision_summary") or {}
        follow_up_scan = status.get("follow_up_scan") or {}
        blocked_context = status.get("blocked_monitoring_context") or {}
        cooldown_campaign = status.get("cooldown_campaign") or {}
        followup_comparison = status.get("followup_comparison") or {}
        lines = [
            f"Coinbase Crypto Prob Scanner Review Pack {app_version}",
            f"Run ID: {run_id}",
            f"Scan finished UTC: {scan.get('finished_at_utc') or '-'}",
            f"Regime: {regime.get('state') or '-'} / actionability={regime.get('actionability_state') or '-'} / cooldown={bool(regime.get('cooldown_active'))}",
            f"Visible rows: {len(visible_rows)}",
            f"Non-visible evidence rows: {len(suppressed_rows)}",
            f"Suppressed by regime/cooldown/threshold/display: {suppression.get('regime_suppressed_rows', 0)}/{suppression.get('cooldown_suppressed_rows', 0)}/{suppression.get('threshold_suppressed_rows', 0)}/{suppression.get('display_trimmed_rows', 0)}",
            f"Target: +{float(target.get('move_pct') or self.config.target_move_pct) * 100:.1f}% within {int(target.get('horizon_minutes') or self.config.target_horizon_minutes)} minutes",
            f"Decision headline: {decision.get('headline') or '-'}",
            f"Decision summary: {decision.get('summary') or '-'}",
            f"Validated floor / near floor: {decision.get('validated_floor') or '-'} / {decision.get('near_validated_floor') or '-'}",
            f"Blocked near-band rows: {decision.get('blocked_near_validated_rows') or 0}",
            f"Blocked near-threshold rows: {decision.get('blocked_near_threshold_rows') or 0}",
            f"Best blocked threshold gap: {decision.get('best_blocked_threshold_gap') or '-'}",
            f"Follow-up scan: {follow_up_scan.get('run_after_utc') or '-'}",
            f"Follow-up reason / sequence: {(follow_up_scan.get('reason') or '-')} / {(follow_up_scan.get('sequence') or '-')}",
            f"Blocked tracking count: {blocked_context.get('tracked_count') or 0}",
            f"Cooldown campaign runs / unique symbols: {cooldown_campaign.get('merged_from_runs') or 0} / {cooldown_campaign.get('merged_unique_symbols') or 0}",
            f"Follow-up comparison available: {bool(followup_comparison.get('available'))}",
            "",
            "Top visible rows:",
        ]
        for row in visible_rows[:10]:
            lines.append(
                f"- {row.get('symbol')}: live={row.get('live_score')} actionability={row.get('actionability_tier')} regime={row.get('market_regime_state')} reason={row.get('actionability_reason')}"
            )
        blocked_focus = list(decision.get("blocked_focus_symbols") or [])
        if blocked_focus:
            lines.extend(["", "Top blocked monitoring rows:"])
            for row in blocked_focus[:5]:
                lines.append(
                    f"- {row.get('symbol')}: pre={row.get('pre_policy_score')} live={row.get('live_score')} live_threshold={row.get('live_threshold')} threshold_gap={row.get('distance_to_live_threshold')} pre_gap={row.get('pre_policy_distance_to_validated')} suppression={row.get('suppression_reason')} detail={row.get('suppression_reason_detail') or '-'}"
                )
        if followup_comparison.get("available"):
            lines.extend(["", "Cooldown follow-up comparison:"])
            lines.append(
                f"- tracked={followup_comparison.get('tracked_count') or 0} visible_now={followup_comparison.get('visible_now_count') or 0} still_blocked={followup_comparison.get('still_blocked_count') or 0} near_visibility_now={followup_comparison.get('near_visibility_now_count') or 0} improved_live={followup_comparison.get('improved_live_count') or 0}"
            )
            tracked_visible_rows = list(followup_comparison.get("tracked_visible_rows") or [])
            if tracked_visible_rows:
                lines.append("- tracked now visible: " + ", ".join(str(r.get('symbol')) for r in tracked_visible_rows[:5] if r.get('symbol')))
            for row in list(followup_comparison.get("top_changes") or [])[:5]:
                lines.append(
                    f"- {row.get('symbol')}: current_row_type={row.get('current_row_type') or '-'} current_live={row.get('current_live_score') or '-'} current_threshold={row.get('current_live_threshold') or '-'} delta_live={row.get('delta_live_score') or '-'} became_visible={bool(row.get('became_visible'))}"
                )
        if suppressed_rows:
            lines.extend(["", "Top non-visible evidence rows:"])
            for row in suppressed_rows[:10]:
                lines.append(
                    f"- {row.get('symbol')}: pre={row.get('pre_policy_score') or row.get('prob_2_pre_regime')} live={row.get('live_score')} threshold_gap={row.get('distance_to_live_threshold') or '-'} suppression={row.get('suppression_reason')} detail={row.get('suppression_reason_detail') or row.get('policy_constraint_reason') or '-'} rank={row.get('informational_rank') or row.get('pre_policy_rank') or row.get('candidate_rank_all') or '-'}"
                )
        if recent_evidence_summary is not None:
            lines.extend([
                "",
                "Rolling resolved evidence (same model fingerprint):",
                f"- Available: {bool(recent_evidence_summary.get('available'))}",
                f"- Resolved rows: {recent_evidence_summary.get('resolved_rows') or 0}",
                f"- Quality/raw hit rate: {recent_evidence_summary.get('quality_hit_rate')} / {recent_evidence_summary.get('raw_hit_rate')}",
                f"- Avg end ret / MAE: {recent_evidence_summary.get('avg_end_ret')} / {recent_evidence_summary.get('avg_mae')}",
            ])
            for threshold, bucket in (recent_evidence_summary.get('threshold_bands') or {}).items():
                lines.append(
                    f"- >= {threshold}: count={bucket.get('count') or 0} quality_hit_rate={bucket.get('quality_hit_rate')} avg_end_ret={bucket.get('avg_end_ret')} visible/non_visible={bucket.get('visible_count') or 0}/{bucket.get('non_visible_count') or 0}"
                )

        if outcomes is not None:
            summary = self._summarize_outcomes(outcomes)
            audit = self._policy_audit(outcomes)
            lines.extend(
                [
                    "",
                    "Outcome summary:",
                    f"- Evaluated rows: {summary['evaluated_rows']}",
                    f"- Visible quality hit rate: {summary['visible_quality_hit_rate']}",
                    f"- Suppressed quality hit rate: {summary['suppressed_quality_hit_rate']}",
                    f"- Visible avg end ret: {summary['visible_avg_end_ret']}",
                    f"- Suppressed avg end ret: {summary['suppressed_avg_end_ret']}",
                    f"- Visible avg MAE: {summary['visible_avg_mae']}",
                    f"- Suppressed avg MAE: {summary['suppressed_avg_mae']}",
                    "",
                    "Policy audit:",
                    f"- False suppressions (quality/raw): {audit['false_suppressions_quality_count']}/{audit['false_suppressions_raw_count']}",
                    f"- Bad visible rows: {audit['bad_visible_rows_count']}",
                    f"- Policy overblock gap (quality hit rate): {audit['policy_overblock_gap_quality']}",
                    f"- Policy protection gap (avg end ret): {audit['policy_protection_gap_end_ret']}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _summarize_outcomes(self, rows: List[dict]) -> dict:
        evaluated = [r for r in rows if int(r.get("resolved") or 0) == 1]
        visible = [r for r in evaluated if r.get("row_type") == "visible"]
        suppressed = [r for r in evaluated if _is_non_visible_row(r)]
        return {
            "evaluated_rows": len(evaluated),
            "visible_rows": len(visible),
            "suppressed_rows": len(suppressed),
            "visible_quality_hit_rate": _rate(visible, "quality_touched"),
            "suppressed_quality_hit_rate": _rate(suppressed, "quality_touched"),
            "visible_avg_end_ret": _avg_metric(visible, "end_ret"),
            "suppressed_avg_end_ret": _avg_metric(suppressed, "end_ret"),
            "visible_avg_mae": _avg_metric(visible, "mae"),
            "suppressed_avg_mae": _avg_metric(suppressed, "mae"),
        }

    def _bucket_summary(self, rows: List[dict]) -> dict:
        return {
            "count": len(rows),
            "quality_hit_rate": _rate(rows, "quality_touched"),
            "raw_hit_rate": _rate(rows, "raw_touched"),
            "avg_end_ret": _avg_metric(rows, "end_ret"),
            "avg_mae": _avg_metric(rows, "mae"),
            "avg_mfe": _avg_metric(rows, "mfe"),
            "avg_time_to_touch_minutes": _avg_metric(rows, "time_to_touch_minutes"),
        }

    def _policy_audit(self, rows: List[dict], runs: List[dict] | None = None) -> dict:
        evaluated = [r for r in rows if int(r.get("resolved") or 0) == 1]
        visible = [r for r in evaluated if r.get("row_type") == "visible"]
        suppressed = [r for r in evaluated if _is_non_visible_row(r)]
        by_reason: Dict[str, List[dict]] = {k: [] for k in ("regime", "cooldown", "threshold", "display_trim", "other")}
        for row in suppressed:
            reason = str(row.get("suppression_reason") or "other")
            by_reason[reason if reason in by_reason else "other"].append(row)
        regime_breakdown: Dict[str, dict] = {}
        for state in sorted({str(r.get("market_regime_state") or "unknown") for r in evaluated}):
            state_rows = [r for r in evaluated if str(r.get("market_regime_state") or "unknown") == state]
            regime_breakdown[state] = {
                "visible": self._bucket_summary([r for r in state_rows if r.get("row_type") == "visible"]),
                "suppressed": self._bucket_summary([r for r in state_rows if _is_non_visible_row(r)]),
                "total": len(state_rows),
            }
        false_suppressions_quality = [r for r in suppressed if int(r.get("quality_touched") or 0) == 1]
        false_suppressions_raw = [r for r in suppressed if int(r.get("raw_touched") or 0) == 1]
        def _is_bad_visible(r: dict) -> bool:
            if int(r.get("quality_touched") or 0) == 1:
                return False
            mae = _f(r.get("mae"))
            end_ret = _f(r.get("end_ret"))
            quality_max_mae = _f(r.get("quality_max_mae"))
            quality_min_end_ret = _f(r.get("quality_min_end_ret"))
            return ((mae is not None and quality_max_mae is not None and mae <= quality_max_mae) or
                    (end_ret is not None and quality_min_end_ret is not None and end_ret <= quality_min_end_ret))
        bad_visible = [r for r in visible if _is_bad_visible(r)]
        visible_quality = _rate(visible, "quality_touched")
        suppressed_quality = _rate(suppressed, "quality_touched")
        visible_end = _avg_metric(visible, "end_ret")
        suppressed_end = _avg_metric(suppressed, "end_ret")
        audit = {
            "run_count": len(runs or []),
            "completed_run_count": sum(1 for r in (runs or []) if bool(r.get("evaluation_complete"))) if runs is not None else None,
            "evaluated_rows": len(evaluated),
            "visible": self._bucket_summary(visible),
            "suppressed": self._bucket_summary(suppressed),
            "suppressed_by_reason": {k: self._bucket_summary(v) for k, v in by_reason.items()},
            "regime_breakdown": regime_breakdown,
            "false_suppressions_quality_count": len(false_suppressions_quality),
            "false_suppressions_raw_count": len(false_suppressions_raw),
            "bad_visible_rows_count": len(bad_visible),
            "policy_overblock_gap_quality": round((suppressed_quality or 0) - (visible_quality or 0), 4) if visible_quality is not None and suppressed_quality is not None else None,
            "policy_protection_gap_end_ret": round((visible_end or 0) - (suppressed_end or 0), 6) if visible_end is not None and suppressed_end is not None else None,
        }
        return audit

    def _policy_audit_reason_rows(self, audit: dict) -> List[dict]:
        rows = []
        for reason, bucket in (audit.get("suppressed_by_reason") or {}).items():
            rows.append({"suppression_reason": reason, **bucket})
        return rows

    def _policy_audit_regime_rows(self, audit: dict) -> List[dict]:
        rows = []
        for regime_state, info in (audit.get("regime_breakdown") or {}).items():
            rows.append({
                "market_regime_state": regime_state,
                "total": info.get("total"),
                "visible_count": ((info.get("visible") or {}).get("count")),
                "visible_quality_hit_rate": ((info.get("visible") or {}).get("quality_hit_rate")),
                "visible_avg_end_ret": ((info.get("visible") or {}).get("avg_end_ret")),
                "suppressed_count": ((info.get("suppressed") or {}).get("count")),
                "suppressed_quality_hit_rate": ((info.get("suppressed") or {}).get("quality_hit_rate")),
                "suppressed_avg_end_ret": ((info.get("suppressed") or {}).get("avg_end_ret")),
            })
        return rows

    def _build_run_pack(self, run_id: str, *, include_outcomes: bool) -> Path:
        run = self._load_run(run_id)
        if not run:
            raise FileNotFoundError(run_id)
        rows = self._load_run_rows(run_id)
        status = read_json(run.get("review_status_path"), {})
        visible = [r for r in rows if r.get("row_type") == "visible"]
        suppressed = [r for r in rows if r.get("row_type") == "suppressed"]
        informational = [r for r in rows if r.get("row_type") == "informational"]
        overflow = [r for r in rows if r.get("row_type") == "overflow"]
        outcomes = [r for r in rows if int(r.get("resolved") or 0) == 1]
        pending_rows = [r for r in rows if int(r.get("resolved") or 0) == 0]
        audit = self._policy_audit(outcomes if include_outcomes else rows, runs=[run])
        pack_name = f"{run_id}_{'evaluated' if include_outcomes else 'scan'}.zip"
        pack_path = self.pack_dir / pack_name
        recent_resolved = self._load_recent_resolved_rows(model_fingerprint=str(run.get("model_fingerprint") or "unknown"))
        recent_evidence_summary = self._build_recent_evidence_summary(recent_resolved, model_fingerprint=str(run.get("model_fingerprint") or "unknown"))
        summary_txt = self._build_summary_text(run_id=run_id, app_version=str(run.get("app_version") or APP_VERSION), status=status, visible_rows=visible, suppressed_rows=suppressed + informational + overflow, outcomes=outcomes if include_outcomes else None, recent_evidence_summary=recent_evidence_summary)
        blocked_focus = list((status.get("decision_summary") or {}).get("blocked_focus_symbols") or [])
        followup_comparison = status.get("followup_comparison") or {}
        blocked_context = status.get("blocked_monitoring_context") or {}
        cooldown_campaign = status.get("cooldown_campaign") or {}
        follow_up_scan = status.get("follow_up_scan") or {}
        tracked_visible_rows = list(followup_comparison.get("tracked_visible_rows") or [])
        with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("summary.txt", summary_txt)
            zf.writestr("run_status.json", json.dumps(status, indent=2, default=str))
            zf.writestr("score_contract.json", json.dumps(status.get("score_contract") or {}, indent=2, default=str))
            zf.writestr("score_diagnostics.json", json.dumps(status.get("score_diagnostics") or {}, indent=2, default=str))
            zf.writestr("top_pretrim_candidates.csv", _csv_bytes(list((status.get("score_diagnostics") or {}).get("top_pretrim_candidates") or []), fieldnames=TOP_PRETRIM_FIELDS))
            zf.writestr("candidate_quality.json", json.dumps(status.get("candidate_quality") or {}, indent=2, default=str))
            zf.writestr("stage1_omission_audit.json", json.dumps(status.get("stage1_omission_audit") or {}, indent=2, default=str))
            zf.writestr("stage1_selection_repair_review.json", json.dumps(status.get("stage1_selection_repair_review") or {}, indent=2, default=str))
            zf.writestr("threshold_experiment_review.json", json.dumps(status.get("threshold_experiment_review") or {}, indent=2, default=str))
            zf.writestr("stage1_to_stage2_trace.csv", _csv_bytes(list((status.get("candidate_quality") or {}).get("stage1_to_stage2_trace") or []), fieldnames=STAGE1_TRACE_FIELDS))
            zf.writestr("candidate_quality_by_tier.csv", _csv_bytes(self._candidate_quality_rows_from_status(status), fieldnames=CANDIDATE_QUALITY_TIER_FIELDS))
            zf.writestr("regime_snapshot.json", json.dumps(status.get("market_regime") or {}, indent=2, default=str))
            zf.writestr("model_summary.json", json.dumps(((status.get("model") or {}).get("pt2") or {}), indent=2, default=str))
            zf.writestr("visible_rows.csv", _csv_bytes(visible, fieldnames=ROW_CSV_FIELDS))
            zf.writestr("suppressed_rows.csv", _csv_bytes(suppressed, fieldnames=ROW_CSV_FIELDS))
            zf.writestr("informational_rows.csv", _csv_bytes(informational, fieldnames=ROW_CSV_FIELDS))
            zf.writestr("informational_overflow_rows.csv", _csv_bytes(overflow, fieldnames=ROW_CSV_FIELDS))
            zf.writestr("blocked_focus_rows.csv", _csv_bytes(blocked_focus, fieldnames=BLOCKED_FOCUS_FIELDS))
            zf.writestr("follow_up_scan.json", json.dumps(follow_up_scan, indent=2, default=str))
            zf.writestr("blocked_monitoring_context.json", json.dumps(blocked_context, indent=2, default=str))
            zf.writestr("cooldown_campaign.json", json.dumps(cooldown_campaign, indent=2, default=str))
            zf.writestr("followup_comparison.json", json.dumps(followup_comparison, indent=2, default=str))
            zf.writestr("followup_changes.csv", _csv_bytes(list(followup_comparison.get("top_changes") or []), fieldnames=FOLLOWUP_CHANGE_FIELDS))
            zf.writestr("tracked_visible_rows.csv", _csv_bytes(tracked_visible_rows, fieldnames=TRACKED_VISIBLE_FIELDS))
            zf.writestr("recent_resolved_outcomes.csv", _csv_bytes(recent_resolved, fieldnames=RECENT_RESOLVED_FIELDS))
            zf.writestr("recent_evidence_summary.json", json.dumps(recent_evidence_summary, indent=2, default=str))
            zf.writestr("pending_outcomes.csv", _csv_bytes(pending_rows, fieldnames=ROW_CSV_FIELDS))
            zf.writestr("outcomes.csv", _csv_bytes(outcomes, fieldnames=ROW_CSV_FIELDS))
            zf.writestr("policy_audit.json", json.dumps(audit, indent=2, default=str))
            zf.writestr("policy_audit_by_reason.csv", _csv_bytes(self._policy_audit_reason_rows(audit), fieldnames=POLICY_AUDIT_REASON_FIELDS))
            zf.writestr("policy_audit_by_regime.csv", _csv_bytes(self._policy_audit_regime_rows(audit), fieldnames=POLICY_AUDIT_REGIME_FIELDS))
            zf.writestr("run_manifest.json", json.dumps({
                "run_id": run_id,
                "evaluation_complete": bool(run.get("evaluation_complete")),
                "visible_rows": len(visible),
                "suppressed_rows": len(suppressed),
                "informational_rows": len(informational),
                "informational_overflow_rows": len(overflow),
                "blocked_focus_rows": len(blocked_focus),
                "blocked_tracking_count": int(blocked_context.get("tracked_count") or 0),
                "cooldown_campaign_runs": int(cooldown_campaign.get("merged_from_runs") or 0),
                "cooldown_campaign_unique_symbols": int(cooldown_campaign.get("merged_unique_symbols") or 0),
                "follow_up_scheduled": bool(follow_up_scan.get("scheduled")),
                "followup_comparison_available": bool(followup_comparison.get("available")),
                "followup_visible_now_count": int(followup_comparison.get("visible_now_count") or 0),
                "tracked_visible_rows": len(tracked_visible_rows),
                "non_visible_rows": len(suppressed) + len(informational) + len(overflow),
                "pending_rows": len(pending_rows),
                "evaluated_rows": len(outcomes),
                "recent_resolved_rows": len(recent_resolved),
                "recent_evidence_available": bool(recent_evidence_summary.get("available")),
                "generated_at_utc": _utc_now_iso(),
                "app_version": str(run.get("app_version") or APP_VERSION),
            }, indent=2))
        return pack_path

    def _update_run_pack(self, run_id: str, pack_path: Path, *, evaluated: bool) -> None:
        now_iso = _utc_now_iso()
        with self._lock, self._connect() as conn:
            if evaluated:
                conn.execute(
                    "UPDATE review_runs SET latest_evaluated_pack_path = ?, latest_evaluated_pack_generated_utc = ?, updated_at_utc = ? WHERE run_id = ?",
                    (str(pack_path), now_iso, now_iso, run_id),
                )
            else:
                conn.execute(
                    "UPDATE review_runs SET latest_scan_pack_path = ?, latest_scan_pack_generated_utc = ?, updated_at_utc = ? WHERE run_id = ?",
                    (str(pack_path), now_iso, now_iso, run_id),
                )
            conn.commit()
        if evaluated:
            try:
                self.latest_eval_link.unlink(missing_ok=True)
                self.latest_eval_link.write_bytes(pack_path.read_bytes())
            except Exception:
                pass
        else:
            try:
                self.latest_scan_link.unlink(missing_ok=True)
                self.latest_scan_link.write_bytes(pack_path.read_bytes())
            except Exception:
                pass

    def _build_and_store_evaluated_pack(self, run_id: str) -> None:
        pack = self._build_run_pack(run_id, include_outcomes=True)
        self._update_run_pack(run_id, pack, evaluated=True)
        callback = getattr(self, "post_evaluation_callback", None)
        if callable(callback):
            try:
                callback(run_id, pack)
            except Exception as exc:  # pragma: no cover
                logger.warning("post_evaluation_callback_failed run_id=%s error=%s", run_id, exc)

    def get_pack_for_run(self, run_id: str, evaluated: bool = True) -> Path | None:
        run = self._load_run(run_id)
        if not run:
            return None
        key = "latest_evaluated_pack_path" if evaluated else "latest_scan_pack_path"
        path = run.get(key)
        if path and Path(path).exists():
            return Path(path)
        pack = self._build_run_pack(run_id, include_outcomes=evaluated)
        self._update_run_pack(run_id, pack, evaluated=evaluated)
        return pack

    def build_aggregate_pack(self, *, hours: int) -> Path:
        cutoff = (_utc_now() - timedelta(hours=hours)).isoformat()
        with self._lock, self._connect() as conn:
            runs = conn.execute(
                "SELECT run_id, scan_finished_utc, market_regime_state, market_regime_actionability, visible_rows_count, suppressed_rows_count, evaluation_complete FROM review_runs WHERE scan_finished_utc >= ? ORDER BY scan_finished_utc DESC LIMIT ?",
                (cutoff, self.config.review_max_runs_in_aggregate),
            ).fetchall()
        pack_path = self.pack_dir / f"review_pack_last_{hours}h.zip"
        summary_rows = [dict(r) for r in runs]
        audit = self.get_policy_audit(hours=hours)
        with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("runs_summary.csv", _csv_bytes(summary_rows))
            zf.writestr("runs_summary.json", json.dumps(summary_rows, indent=2, default=str))
            zf.writestr("policy_audit.json", json.dumps(audit, indent=2, default=str))
            zf.writestr("policy_audit_by_reason.csv", _csv_bytes(self._policy_audit_reason_rows(audit)))
            zf.writestr("policy_audit_by_regime.csv", _csv_bytes(self._policy_audit_regime_rows(audit)))
            for row in summary_rows:
                run_id = row["run_id"]
                run_pack = self.get_pack_for_run(run_id, evaluated=bool(row.get("evaluation_complete")))
                if run_pack and run_pack.exists():
                    zf.writestr(f"runs/{run_id}/{run_pack.name}", run_pack.read_bytes())
        return pack_path

    def build_current_version_pack(self, *, app_version: str | None = None, include_evaluated: bool = True) -> Path:
        version = str(app_version or APP_VERSION)
        runs = self.get_runs_for_app_version(version)
        if not runs:
            raise FileNotFoundError(f"no review packs for app version {version}")
        deployed_since_utc = min((r.get("scan_finished_utc") or "") for r in runs if r.get("scan_finished_utc")) or None
        generated_at_utc = _utc_now_iso()
        pack_path = self.pack_dir / f"review_pack_current_version_{version.replace('.', '_')}.zip"
        summary_rows = list(runs)
        total_visible = sum(int(r.get("visible_rows_count") or 0) for r in runs)
        total_suppressed = sum(int(r.get("suppressed_rows_count") or 0) for r in runs)
        evaluated_count = sum(1 for r in runs if r.get("evaluation_complete"))
        current_summary = self._build_current_version_summary(version=version, runs=runs)
        with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("runs_summary.csv", _csv_bytes(summary_rows))
            zf.writestr("runs_summary.json", json.dumps(summary_rows, indent=2, default=str))
            zf.writestr("current_version_manifest.json", json.dumps({
                "app_version": version,
                "deployed_since_utc": deployed_since_utc,
                "generated_at_utc": generated_at_utc,
                "scan_pack_count": len(runs),
                "evaluated_pack_count": evaluated_count,
                "total_visible_rows": total_visible,
                "total_suppressed_rows": total_suppressed,
                "includes_evaluated_packs": bool(include_evaluated),
                "note": "This bundle includes all scan packs recorded under the currently deployed app version.",
            }, indent=2))
            zf.writestr("current_version_evidence_summary.json", json.dumps(current_summary, indent=2, default=str))
            zf.writestr("current_version_threshold_bands.csv", _csv_bytes(list((current_summary.get("evidence") or {}).get("threshold_bands") or []), fieldnames=CURRENT_VERSION_THRESHOLD_FIELDS))
            zf.writestr("current_version_regime_breakdown.csv", _csv_bytes(list(current_summary.get("regime_breakdown") or []), fieldnames=CURRENT_VERSION_REGIME_FIELDS))
            zf.writestr("current_version_regime_evidence.csv", _csv_bytes(list((current_summary.get("regime_evidence") or {}).get("rows") or []), fieldnames=CURRENT_VERSION_REGIME_EVIDENCE_FIELDS))
            zf.writestr("current_version_regime_threshold_bands.csv", _csv_bytes(list((current_summary.get("regime_evidence") or {}).get("threshold_rows") or []), fieldnames=CURRENT_VERSION_REGIME_THRESHOLD_FIELDS))
            zf.writestr("current_version_scan_score_diagnostics.csv", _csv_bytes(list((current_summary.get("scan_score_diagnostics") or {}).get("counts_above_thresholds") or []), fieldnames=["threshold", "live_count", "pre_policy_count", "model_count"]))
            zf.writestr("current_version_candidate_quality_by_tier.csv", _csv_bytes(list((current_summary.get("candidate_quality") or {}).get("tiers") or []), fieldnames=["liquidity_tier", "scans", "stage1_feature_ready", "stage1_blocked", "stage1_selected", "avg_stage1_selected_share", "stage2_scored", "stage2_visible", "stage2_hidden", "stage2_count_ge_0_30", "stage2_count_ge_0_35", "stage2_count_ge_0_45", "max_live_score"]))
            zf.writestr("current_version_cohort_symbol_summary.csv", _csv_bytes(list((current_summary.get("cohort_symbols") or {}).get("rows") or []), fieldnames=["symbol", "liquidity_tier", "selected_scans", "visible_scans", "hidden_scans", "max_live_score", "count_ge_0_30", "count_ge_0_35"]))
            zf.writestr("current_version_symbol_repeatability.csv", _csv_bytes(list((current_summary.get("symbol_repeatability") or {}).get("rows") or []), fieldnames=CURRENT_VERSION_REPEATABILITY_FIELDS))
            zf.writestr("current_version_outlier_concentration.json", json.dumps(current_summary.get("outlier_concentration") or {}, indent=2, default=str))
            for row in runs:
                run_id = str(row["run_id"])
                scan_pack = self.get_pack_for_run(run_id, evaluated=False)
                if scan_pack and scan_pack.exists():
                    zf.writestr(f"scan_packs/{run_id}/{scan_pack.name}", scan_pack.read_bytes())
                if include_evaluated and bool(row.get("evaluation_complete")):
                    eval_pack = self.get_pack_for_run(run_id, evaluated=True)
                    if eval_pack and eval_pack.exists():
                        zf.writestr(f"evaluated_packs/{run_id}/{eval_pack.name}", eval_pack.read_bytes())
        return pack_path

    def prune_old_runs(self) -> int:
        cutoff = (_utc_now() - timedelta(days=self.config.review_retention_days)).isoformat()
        removed = 0
        with self._lock, self._connect() as conn:
            old_runs = conn.execute("SELECT run_id, review_status_path, review_visible_rows_path, review_suppressed_rows_path, review_summary_path, latest_scan_pack_path, latest_evaluated_pack_path FROM review_runs WHERE scan_finished_utc < ?", (cutoff,)).fetchall()
            for row in old_runs:
                for key in ("review_status_path", "review_visible_rows_path", "review_suppressed_rows_path", "review_summary_path", "latest_scan_pack_path", "latest_evaluated_pack_path"):
                    path = row[key]
                    if path:
                        try:
                            Path(path).unlink(missing_ok=True)
                        except Exception:
                            pass
                run_dir = self.root_dir / str(row["run_id"])
                if run_dir.exists():
                    for child in run_dir.iterdir():
                        child.unlink(missing_ok=True)
                    run_dir.rmdir()
                conn.execute("DELETE FROM review_run_rows WHERE run_id = ?", (row["run_id"],))
                conn.execute("DELETE FROM review_runs WHERE run_id = ?", (row["run_id"],))
                removed += 1
            conn.commit()
        return removed


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _rate(rows: List[dict], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(int(r.get(key) or 0) for r in rows) / len(rows), 4)


def _avg_metric(rows: List[dict], key: str) -> float | None:
    vals = [float(r.get(key)) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _quantile(values: List[float], q: float) -> float | None:
    vals = sorted(float(v) for v in (values or []) if v is not None)
    if not vals:
        return None
    if q <= 0:
        return round(vals[0], 4)
    if q >= 1:
        return round(vals[-1], 4)
    idx = int(round((len(vals) - 1) * q))
    idx = max(0, min(idx, len(vals) - 1))
    return round(vals[idx], 4)


def _csv_bytes(rows: List[dict], fieldnames: List[str] | None = None) -> bytes:
    rows = list(rows or [])
    inferred = sorted({k for row in rows for k in row.keys()})
    if fieldnames is None:
        fieldnames = inferred
    else:
        fieldnames = list(dict.fromkeys(list(fieldnames) + inferred))
    if not fieldnames:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})
    return output.getvalue().encode("utf-8")


def _csv_value(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


def pd_to_datetime(series):
    import pandas as pd
    return pd.to_datetime(series, utc=True)
