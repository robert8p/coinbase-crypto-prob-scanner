from __future__ import annotations

import csv
import io
import json
import logging
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .coinbase_client import CoinbaseClient
from .config import AppConfig
from .features import FEATURE_COLUMNS, compute_guardrails, compute_live_features, heuristic_probability, stage1_rank, stage1_select
from .live_scoring import apply_live_post_model_adjustments
from .modeling import ModelBundle
from .persist import atomic_write_json, ensure_dir, read_json
from .regime import assess_market_regime_readiness, build_market_regime, classify_liquidity_tier, live_policy_for
from .review_runs import ReviewPackService
from .scanner import ScannerService
from .universe import UniverseBuilder
from .version import APP_VERSION

logger = logging.getLogger(__name__)


class HistoricalReplayService:
    """Historical live-emulation replay for the currently deployed scanner logic.

    This is intentionally not a naive backtest. It reuses the current live cohort,
    model, stage1/stage2 logic, market-regime engine, visibility rules, and
    outcome definitions. It still has limitations (for example no historical
    Binance cross-exchange replay), and those are surfaced in the output pack.
    """

    def __init__(self, config: AppConfig, client: CoinbaseClient, scanner: ScannerService, review_packs: ReviewPackService):
        self.config = config
        self.client = client
        self.scanner = scanner
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / "replay_runs")
        self.pack_dir = ensure_dir(Path(config.model_dir) / "replay_packs")
        self.latest_summary_path = Path(config.model_dir) / "latest_replay_summary.json"
        self.latest_pack_link = self.pack_dir / "latest_replay_pack.zip"
        self._lock = threading.RLock()

    def latest_summary(self) -> dict:
        return read_json(self.latest_summary_path, {})

    def latest_pack(self) -> Path | None:
        if self.latest_pack_link.exists():
            return self.latest_pack_link
        return None

    def run(
        self,
        *,
        start_utc: str | None = None,
        end_utc: str | None = None,
        hours: int | None = None,
        step_minutes: int | None = None,
        max_scans: int | None = None,
        max_symbols: int | None = None,
        pipeline_mode: str = "full",
        raw_threshold: float = 0.30,
        stage1_selection_mode_override: str | None = None,
        stage1_max_candidates_override: int | None = None,
        model_bundle_path_override: str | None = None,
        model_bundle_label_override: str | None = None,
        capture_full_rankable_rows: bool = False,
    ) -> dict:
        hours = max(1, int(hours or self.config.replay_default_hours))
        step_minutes = max(5, int(step_minutes or self.config.replay_default_step_minutes))
        max_scans = max(1, int(max_scans or self.config.replay_max_scans))
        max_symbols = max(0, int(max_symbols or self.config.replay_max_symbols))
        pipeline_mode = str(pipeline_mode or "full").strip().lower()
        if pipeline_mode not in {"full", "raw_threshold"}:
            raise ValueError("pipeline_mode must be 'full' or 'raw_threshold'")
        raw_threshold = max(0.0, min(1.0, float(raw_threshold or 0.30)))

        end_dt = self._align_5m(self._parse_utc(end_utc) or datetime.now(timezone.utc))
        start_dt = self._align_5m(self._parse_utc(start_utc) or (end_dt - timedelta(hours=hours)))
        if start_dt >= end_dt:
            raise ValueError("start_utc must be earlier than end_utc")
        timestamps = self._build_timestamps(start_dt, end_dt, step_minutes=step_minutes, max_scans=max_scans)
        if not timestamps:
            raise ValueError("no replay timestamps generated")

        bundle_path = str(model_bundle_path_override or self.config.model_path_pt2 or '').strip()
        bundle = ModelBundle.load(bundle_path)
        if bundle is None:
            raise FileNotFoundError(f"trained model bundle not found: {bundle_path or self.config.model_path_pt2}")
        bundle_label = str(model_bundle_label_override or Path(bundle_path).name or 'pt2')

        products = self.client.list_products()
        currencies = self.client.list_currencies()
        volume_map = self.client.get_volume_summary()
        locked_symbols = self.scanner._locked_live_cohort()
        universe = UniverseBuilder(self.config).build(
            products,
            currencies,
            volume_map,
            locked_symbols=locked_symbols,
            selection_label=self.scanner._selection_label(locked_symbols),
        )
        selected_for_fetch = list(universe.selected_for_fetch)
        if max_symbols > 0:
            selected_for_fetch = selected_for_fetch[:max_symbols]
        selected_symbols = [str(p.get("id") or "") for p in selected_for_fetch if str(p.get("id") or "")]
        if not selected_symbols:
            raise ValueError("no selected symbols available for replay")

        warmup_bars = max(self.config.stage1_light_calendar_5m_bars, self.config.stage2_lookback_5m_bars)
        horizon_bars = max(1, self.config.candles_per_horizon)
        history_start = timestamps[0] - timedelta(minutes=5 * max(1, warmup_bars - 1))
        history_end = timestamps[-1] + timedelta(minutes=5 * horizon_bars)
        prefetch_symbols = sorted(set(selected_symbols + ["BTC-USD", "ETH-USD"]))
        histories = self._prefetch_histories(prefetch_symbols, history_start, history_end)

        limitations = [
            "historical replay reuses the current locked live cohort and current product metadata rather than a point-in-time historical universe",
            "historical replay does not currently reproduce historical Binance cross-exchange penalties; those paths are replayed with Coinbase-only inputs",
        ]

        primary = self._execute_replay_window(
            timestamps=timestamps,
            selected_for_fetch=selected_for_fetch,
            universe=universe,
            bundle=bundle,
            histories=histories,
            pipeline_mode=pipeline_mode,
            raw_threshold=raw_threshold,
            stage1_selection_mode_override=stage1_selection_mode_override,
            stage1_max_candidates_override=stage1_max_candidates_override,
            capture_full_rankable_rows=capture_full_rankable_rows,
        )
        summary = self._build_replay_summary(
            timestamps=timestamps,
            scan_summaries=primary["scan_summaries"],
            replay_rows=primary["replay_rows"],
            counterfactual_rows=primary["counterfactual_rows"],
            universe=universe,
            limitations=limitations,
            pipeline_mode=pipeline_mode,
            raw_threshold=raw_threshold,
        )
        summary['replay_model_context'] = {
            'bundle_path': bundle_path,
            'bundle_label': bundle_label,
            'using_override': bool(model_bundle_path_override),
        }
        if pipeline_mode == "full":
            alt = self._execute_replay_window(
                timestamps=timestamps,
                selected_for_fetch=selected_for_fetch,
                universe=universe,
                bundle=bundle,
                histories=histories,
                pipeline_mode="raw_threshold",
                raw_threshold=raw_threshold,
                stage1_selection_mode_override=stage1_selection_mode_override,
                stage1_max_candidates_override=stage1_max_candidates_override,
                capture_full_rankable_rows=False,
            )
            alt_summary = self._build_replay_summary(
                timestamps=timestamps,
                scan_summaries=alt["scan_summaries"],
                replay_rows=alt["replay_rows"],
                counterfactual_rows=alt["counterfactual_rows"],
                universe=universe,
                limitations=limitations,
                pipeline_mode="raw_threshold",
                raw_threshold=raw_threshold,
            )
            alt_summary['replay_model_context'] = {
                'bundle_path': bundle_path,
                'bundle_label': bundle_label,
                'using_override': bool(model_bundle_path_override),
            }
            summary["pipeline_ablation"] = self._pipeline_ablation_summary(summary, alt_summary)
        pack_path = self._build_replay_pack(summary)
        atomic_write_json(self.latest_summary_path, summary)
        try:
            self.latest_pack_link.unlink(missing_ok=True)
            self.latest_pack_link.write_bytes(pack_path.read_bytes())
        except Exception:
            pass
        result = {
            "ok": True,
            "summary": summary,
            "pack_path": str(pack_path),
            "download_path": "/api/replay/latest.zip",
            "summary_path": "/api/replay/latest-summary",
        }
        if capture_full_rankable_rows:
            result["captured_rankable_rows"] = primary.get("captured_rankable_rows") or []
        return result

    def _parse_utc(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def _align_5m(self, value: datetime) -> datetime:
        value = value.astimezone(timezone.utc)
        floored_minute = value.minute - (value.minute % 5)
        return value.replace(minute=floored_minute, second=0, microsecond=0)

    def _build_timestamps(self, start_dt: datetime, end_dt: datetime, *, step_minutes: int, max_scans: int) -> List[datetime]:
        step = timedelta(minutes=max(5, step_minutes))
        cur = start_dt
        out: List[datetime] = []
        while cur <= end_dt:
            out.append(cur)
            cur += step
        if len(out) > max_scans:
            out = out[-max_scans:]
        return out

    def _prefetch_histories(self, symbols: Iterable[str], start_dt: datetime, end_dt: datetime) -> Dict[str, pd.DataFrame]:
        histories: Dict[str, pd.DataFrame] = {}
        workers = max(1, min(int(self.config.replay_prefetch_max_workers), self.config.max_workers, 6))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.client.get_candles_range, symbol, start_dt, end_dt): symbol for symbol in symbols}
            for fut in as_completed(futures):
                symbol = futures[fut]
                try:
                    histories[symbol] = fut.result(timeout=max(30.0, self.config.http_timeout_seconds + 30.0))
                except Exception as exc:
                    logger.warning("replay_prefetch_failed symbol=%s error=%s", symbol, exc)
                    histories[symbol] = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        return histories

    def _execute_replay_window(self, *, timestamps: List[datetime], selected_for_fetch: List[dict], universe, bundle: ModelBundle, histories: Dict[str, pd.DataFrame], pipeline_mode: str, raw_threshold: float, stage1_selection_mode_override: str | None = None, stage1_max_candidates_override: int | None = None, capture_full_rankable_rows: bool = False) -> dict:
        scan_summaries: List[dict] = []
        replay_rows: List[dict] = []
        counterfactual_rows: List[dict] = []
        captured_rankable_rows: List[dict] = []
        for as_of in timestamps:
            result = self._run_replay_scan(
                as_of=as_of,
                selected_products=selected_for_fetch,
                universe=universe,
                bundle=bundle,
                histories=histories,
                pipeline_mode=pipeline_mode,
                raw_threshold=raw_threshold,
                stage1_selection_mode_override=stage1_selection_mode_override,
                stage1_max_candidates_override=stage1_max_candidates_override,
                capture_full_rankable_rows=capture_full_rankable_rows,
            )
            scan_summaries.append(result["scan_summary"])
            replay_rows.extend(result["rows"])
            counterfactual_rows.extend(result["counterfactual_rows"])
            if capture_full_rankable_rows:
                captured_rankable_rows.extend(result.get("captured_rankable_rows") or [])
        return {
            "scan_summaries": scan_summaries,
            "replay_rows": replay_rows,
            "counterfactual_rows": counterfactual_rows,
            "captured_rankable_rows": captured_rankable_rows,
        }

    def _window_to_end(self, frame: pd.DataFrame | None, *, end_ts: datetime, bars: int) -> pd.DataFrame:
        if frame is None or frame.empty:
            return self.client._regularize_candles(pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]), lookback_bars=bars, end_ts=end_ts)
        start_ts = end_ts - timedelta(minutes=5 * max(1, bars - 1))
        window = frame[(pd.to_datetime(frame["ts"], utc=True) >= start_ts) & (pd.to_datetime(frame["ts"], utc=True) <= end_ts)].copy()
        return self.client._regularize_candles(window, lookback_bars=bars, end_ts=end_ts)

    def _future_window(self, frame: pd.DataFrame | None, *, as_of: datetime) -> pd.DataFrame:
        horizon_bars = max(1, self.config.candles_per_horizon)
        end_ts = as_of + timedelta(minutes=5 * horizon_bars)
        window = self._window_to_end(frame, end_ts=end_ts, bars=horizon_bars + 1)
        if window.empty:
            return window
        return window[pd.to_datetime(window["ts"], utc=True) > as_of].copy().reset_index(drop=True)

    def _resolve_outcome(self, *, symbol: str, entry_utc: datetime, entry_price: float, frame: pd.DataFrame | None) -> dict:
        future = self._future_window(frame, as_of=entry_utc)
        if future.empty or entry_price <= 0:
            return {
                "symbol": symbol,
                "entry_utc": entry_utc.isoformat(),
                "resolved": 0,
                "resolve_utc": None,
                "actual_high": None,
                "actual_low": None,
                "actual_close": None,
                "raw_touched": None,
                "quality_touched": None,
                "mae": None,
                "mfe": None,
                "end_ret": None,
                "time_to_touch_minutes": None,
            }
        target_px = entry_price * (1.0 + float(self.config.target_move_pct))
        actual_high = float(future["high"].max())
        actual_low = float(future["low"].min())
        actual_close = float(future["close"].iloc[-1])
        touch_rows = future[future["high"] >= target_px]
        raw_touched = int(not touch_rows.empty)
        time_to_touch_minutes = None
        if raw_touched:
            first_touch_ts = pd.to_datetime(touch_rows["ts"].iloc[0], utc=True)
            time_to_touch_minutes = int(max(0.0, (first_touch_ts - entry_utc).total_seconds()) // 60)
        mae = (actual_low / entry_price) - 1.0
        mfe = (actual_high / entry_price) - 1.0
        end_ret = (actual_close / entry_price) - 1.0
        quality_touched = int(raw_touched and mae > float(self.config.quality_max_mae) and end_ret > float(self.config.quality_min_end_ret))
        return {
            "symbol": symbol,
            "entry_utc": entry_utc.isoformat(),
            "resolved": 1,
            "resolve_utc": pd.to_datetime(future["ts"].iloc[-1], utc=True).isoformat(),
            "actual_high": round(actual_high, 8),
            "actual_low": round(actual_low, 8),
            "actual_close": round(actual_close, 8),
            "raw_touched": raw_touched,
            "quality_touched": quality_touched,
            "mae": round(float(mae), 6),
            "mfe": round(float(mfe), 6),
            "end_ret": round(float(end_ret), 6),
            "time_to_touch_minutes": time_to_touch_minutes,
        }

    def _stage1_rank_details(self, feature_rows: Dict[str, dict], guardrails: Dict[str, dict], *, btc_regime: str) -> List[dict]:
        is_panic = btc_regime == "BTC panic"
        is_weak = btc_regime == "BTC weak"
        ranked = []
        for symbol, row in feature_rows.items():
            guard = dict(guardrails.get(symbol) or {})
            if str(guard.get("block_code") or "") == "BLOCKED":
                continue
            liquidity_penalty = float(guard.get("liquidity_penalty", 0.0) or 0.0)
            history_bonus = 0.08 * float(row.get("history_bars_ratio_24h", 0.0) or 0.0) + 0.05 * float(row.get("history_bars_ratio_7d", 0.0) or 0.0)
            observed_bonus = 0.08 * float(row.get("observed_bar_density_24h", 0.0) or 0.0) + 0.03 * float(row.get("nonzero_volume_rate_24h", 0.0) or 0.0)
            score = (
                0.28 * float(row.get("ret_60m", 0.0) or 0.0)
                + 0.18 * float(row.get("ret_24h", 0.0) or 0.0)
                + 0.08 * float(row.get("ret_6h", 0.0) or 0.0)
                + 0.10 * float(row.get("asset_vs_btc_1h", 0.0) or 0.0)
                + 0.06 * float(row.get("adx_proxy", 0.0) or 0.0) / 100.0
                + 0.08 * float(row.get("path_smoothness", 0.0) or 0.0)
                + 0.06 * float(row.get("candle_efficiency", 0.0) or 0.0)
                + 0.08 * min(float(row.get("rvol_1h", 0.0) or 0.0), 4.0) / 4.0
                + history_bonus
                + observed_bonus
                + 0.04 * min(float(row.get("dollar_vol_24h_log", 0.0) or 0.0), 18.0) / 18.0
                - 0.10 * float(row.get("wickiness", 0.0) or 0.0)
                - 0.09 * max(0.0, -float(row.get("downside_impulse", 0.0) or 0.0))
                - 0.10 * float(guard.get("uncertainty", 0.0) or 0.0)
                - 0.12 * liquidity_penalty
            )
            if (is_panic or is_weak) and float(row.get("asset_vs_btc_1h", 0.0) or 0.0) < 0:
                score -= 0.06
            if is_panic:
                score -= 0.03
            if float(row.get("dollar_vol_24h_log", 0.0) or 0.0) < 12.42922:  # log1p(250_000)
                score -= 0.05
            ranked.append({"symbol": symbol, "stage1_score": round(float(score), 6)})
        ranked.sort(key=lambda r: (float(r.get("stage1_score") or 0.0), str(r.get("symbol") or "")), reverse=True)
        for idx, row in enumerate(ranked, start=1):
            row["stage1_rank_all"] = idx
        return ranked

    def _run_replay_scan(self, *, as_of: datetime, selected_products: List[dict], universe, bundle: ModelBundle, histories: Dict[str, pd.DataFrame], pipeline_mode: str = "full", raw_threshold: float = 0.30, stage1_selection_mode_override: str | None = None, stage1_max_candidates_override: int | None = None, capture_full_rankable_rows: bool = False) -> dict:
        requested = len(selected_products)
        score_contract = self.scanner._score_contract()
        stage1_input_rows: Dict[str, dict] = {}
        stage1_guardrails: Dict[str, dict] = {}
        stage1_diags: Dict[str, dict] = {}
        stage2_seed_products: Dict[str, dict] = {str(p.get("id") or ""): p for p in selected_products}
        skip_reasons: Dict[str, int] = {}
        returned_light = 0
        stage1_feature_ready = 0

        btc_light_df = self._window_to_end(histories.get("BTC-USD"), end_ts=as_of, bars=self.config.stage1_light_calendar_5m_bars)
        eth_light_df = self._window_to_end(histories.get("ETH-USD"), end_ts=as_of, bars=self.config.stage1_light_calendar_5m_bars)
        btc_light_ctx = self.scanner._make_ctx(btc_light_df)
        eth_light_ctx = self.scanner._make_ctx(eth_light_df)
        btc_regime = self.scanner._btc_regime_label(btc_light_ctx)
        btc_deep_df = self._window_to_end(histories.get("BTC-USD"), end_ts=as_of, bars=self.config.stage2_lookback_5m_bars)

        for product in selected_products:
            symbol = str(product.get("id") or "")
            if not symbol:
                continue
            light_df = self._window_to_end(histories.get(symbol), end_ts=as_of, bars=self.config.stage1_light_calendar_5m_bars)
            if not light_df.empty:
                returned_light += 1
            feature_df = self.scanner._prepare_feature_frame(light_df, self.config.stage1_light_feature_5m_bars)
            history_bars = len(feature_df)
            observed_bars = int(feature_df.attrs.get("observed_bars", int((feature_df["volume"] > 0).sum()) if not feature_df.empty else 0))
            if history_bars < self.config.stage1_min_history_5m_bars:
                skip_reasons["stage1_insufficient_history"] = skip_reasons.get("stage1_insufficient_history", 0) + 1
                continue
            if observed_bars < self.config.stage1_min_observed_5m_bars:
                skip_reasons["stage1_insufficient_observed"] = skip_reasons.get("stage1_insufficient_observed", 0) + 1
                continue
            feat = compute_live_features(symbol, feature_df, btc_ctx=btc_light_ctx, eth_ctx=eth_light_ctx, btc_df=btc_deep_df)
            diag = {**feat.diagnostics, "rolling_dollar_volume": float(product.get("rolling_dollar_volume", 0.0) or 0.0)}
            guard = compute_guardrails(symbol, feat.feature_row, diag, feat.block_reason, self.scanner.state.model_metadata.get("pt2"), self.config)
            stage1_input_rows[symbol] = feat.feature_row
            stage1_diags[symbol] = diag
            stage1_guardrails[symbol] = guard
            stage1_feature_ready += 1

        readiness = assess_market_regime_readiness(self.config, btc_light_ctx, eth_light_ctx, stage1_input_rows)
        market_regime = build_market_regime(
            self.config,
            btc_light_ctx,
            eth_light_ctx,
            stage1_input_rows,
            previous={},
            readiness=readiness,
            publish_meta={
                "partial_publish_attempts": 0,
                "partial_publish_successes": 0,
                "partial_publish_failures": 0,
                "last_partial_publish_attempt_utc": None,
                "last_partial_publish_error": None,
            },
        )
        regime_candidate_cap = int(stage1_max_candidates_override or self.config.stage1_max_candidates)
        if market_regime.state == "amber":
            regime_candidate_cap = min(regime_candidate_cap, max(12, int(self.config.stage1_max_candidates * 0.75)))
        elif market_regime.state == "red":
            regime_candidate_cap = min(regime_candidate_cap, max(8, int(self.config.stage1_max_candidates * 0.45)))

        blocked_stage1 = sum(1 for g in stage1_guardrails.values() if str(g.get("block_code") or "") == "BLOCKED")
        opportunity_scores = {}
        selection_mode = str(stage1_selection_mode_override or getattr(self.config, "stage1_selection_mode", "primary_only") or "primary_only")
        stage1_opportunity = getattr(self.scanner, "stage1_opportunity", None)
        if stage1_opportunity is not None:
            try:
                opportunity_scores = stage1_opportunity.score_feature_rows(stage1_input_rows, stage1_guardrails)
            except Exception:
                opportunity_scores = {}
        stage1_candidates, stage1_selection_meta = stage1_select(
            stage1_input_rows,
            stage1_guardrails,
            regime_candidate_cap,
            btc_regime=btc_regime,
            selection_mode=selection_mode,
            recall_reserve_frac=float(getattr(self.config, "stage1_recall_reserve_frac", 0.25) or 0.25),
            recall_reserve_min=int(getattr(self.config, "stage1_recall_reserve_min", 6) or 6),
            recall_reserve_max=int(getattr(self.config, "stage1_recall_reserve_max", 12) or 12),
            promotion_overflow_window=int(getattr(self.config, "stage1_promotion_overflow_window", 20) or 20),
            opportunity_model_scores=opportunity_scores,
        )
        stage1_dropped_by_rank = max(0, len(stage1_input_rows) - blocked_stage1 - len(stage1_candidates))
        stage1_rank_details = self._stage1_rank_details(stage1_input_rows, stage1_guardrails, btc_regime=btc_regime)
        rank_map = {str(r["symbol"]): r for r in stage1_rank_details}

        btc_deep_ctx = self.scanner._make_ctx(btc_deep_df)
        eth_deep_df = self._window_to_end(histories.get("ETH-USD"), end_ts=as_of, bars=self.config.stage2_lookback_5m_bars)
        eth_deep_ctx = self.scanner._make_ctx(eth_deep_df)

        stage2_rows: Dict[str, dict] = {}
        stage2_guardrails: Dict[str, dict] = {}
        stage2_diags: Dict[str, dict] = {}
        stage2_feature_ready = 0
        stage2_returned = 0
        for symbol in stage1_candidates:
            deep_df = self._window_to_end(histories.get(symbol), end_ts=as_of, bars=self.config.stage2_lookback_5m_bars)
            if not deep_df.empty:
                stage2_returned += 1
            history_bars = len(deep_df)
            observed_bars = int(deep_df.attrs.get("observed_bars", int((deep_df["volume"] > 0).sum()) if not deep_df.empty else 0))
            if history_bars < self.config.stage2_min_history_5m_bars:
                skip_reasons["stage2_insufficient_history"] = skip_reasons.get("stage2_insufficient_history", 0) + 1
                continue
            if observed_bars < self.config.stage2_min_observed_5m_bars:
                skip_reasons["stage2_insufficient_observed"] = skip_reasons.get("stage2_insufficient_observed", 0) + 1
                continue
            feat = compute_live_features(symbol, deep_df, btc_ctx=btc_deep_ctx, eth_ctx=eth_deep_ctx, btc_df=btc_deep_df, cross_exchange=None)
            product = stage2_seed_products.get(symbol, {})
            diag = {**feat.diagnostics, "rolling_dollar_volume": float(product.get("rolling_dollar_volume", 0.0) or 0.0)}
            guard = compute_guardrails(symbol, feat.feature_row, diag, feat.block_reason, self.scanner.state.model_metadata.get("pt2"), self.config)
            stage2_rows[symbol] = feat.feature_row
            stage2_diags[symbol] = diag
            stage2_guardrails[symbol] = guard
            stage2_feature_ready += 1

        pipeline_mode = str(pipeline_mode or "full")
        raw_threshold = max(0.0, min(1.0, float(raw_threshold or 0.30)))
        is_panic = btc_regime == "BTC panic"
        threshold_boost = self.config.panic_threshold_boost if is_panic else 0.0
        sector_leader_rets = self.scanner._compute_sector_leader_rets(stage2_rows)
        raw_scores: List[dict] = []
        suppressed_rows: List[dict] = []
        threshold_candidates: List[dict] = []
        dropped_stage2_blocked = 0
        capped = 0
        event_risk = 0
        suppressed_regime = 0
        suppressed_threshold = 0
        suppressed_cooldown = 0
        model_meta = self.scanner.state.model_metadata.get("pt2") or {}
        active_model_hash = str(model_meta.get("model_fingerprint") or "untrained")

        for symbol, row in stage2_rows.items():
            guard = stage2_guardrails[symbol]
            if str(guard.get("block_code") or "") == "BLOCKED":
                dropped_stage2_blocked += 1
                skip_reasons["stage2_blocked"] = skip_reasons.get("stage2_blocked", 0) + 1
                continue
            if str(guard.get("block_code") or "") == "EVENT_RISK":
                event_risk += 1
            if bool(guard.get("capped")):
                capped += 1

            if bundle is not None:
                prob_model = float(bundle.predict_proba(pd.DataFrame([{k: row.get(k) for k in FEATURE_COLUMNS}]))[0])
                pt2_label = "trained"
            else:
                prob_model = heuristic_probability(row, guard, guardrail_cap=self.config.tail_unvalidated_cap)
                pt2_label = "heuristic"

            sector_penalty = self.scanner._get_sector_penalty(symbol, sector_leader_rets)
            liquidity_bucket = self.scanner._liquidity_bucket(stage2_diags[symbol])
            liquidity_tier = classify_liquidity_tier(symbol, stage2_diags[symbol], self.config)
            if pipeline_mode == "raw_threshold":
                adjustment_detail = {
                    "guardrail_capped": False,
                    "panic_penalty": 0.0,
                    "sector_penalty": 0.0,
                    "binance_gap_penalty": 0.0,
                    "binance_lead_penalty": 0.0,
                    "total_penalty": 0.0,
                }
                prob_pre_regime = max(0.0, min(1.0, float(prob_model)))
                prob_adjusted = prob_pre_regime
                live_policy = {"threshold": raw_threshold, "factor": 1.0, "cap": 1.0, "suppress": False}
                suppress_reason = None
                cooldown_blocked = False
            else:
                prob_adjusted, adjustment_detail = apply_live_post_model_adjustments(
                    prob_model,
                    row,
                    guard,
                    is_panic=is_panic,
                    threshold_boost=threshold_boost,
                    sector_penalty=sector_penalty,
                    guardrail_cap=self.config.tail_unvalidated_cap,
                )
                prob_pre_regime = prob_adjusted
                live_policy = live_policy_for(market_regime.state, liquidity_tier, self.config)
                suppress_reason = "regime" if bool(live_policy.get("suppress")) else None
                cooldown_blocked = False
                if bool(market_regime.suppress_new_entries):
                    if liquidity_tier == "tier3":
                        cooldown_blocked = True
                    elif liquidity_tier == "tier2" and liquidity_bucket != "high":
                        cooldown_blocked = True
                prob_adjusted = max(0.0, prob_adjusted * float(live_policy.get("factor", 1.0) or 1.0))
                prob_adjusted = min(prob_adjusted, float(live_policy.get("cap", 0.95) or 0.95))
            trust = self.scanner._apply_tail_trust(prob_adjusted, score_contract)
            actionability = self.scanner._assess_actionability(
                adjusted_score=prob_adjusted,
                trust=trust,
                score_contract=score_contract,
                market_regime=market_regime,
                liquidity_tier=liquidity_tier,
                guard=guard,
                objective_band=self.scanner._score_band(live_score=trust["display_score"], score_contract=score_contract),
            )
            score_band = self.scanner._score_band(live_score=trust["display_score"], score_contract=score_contract)
            pre_policy_band = self.scanner._score_band(live_score=prob_pre_regime, score_contract=score_contract)
            reasons = self.scanner._build_reasons(row, guard)
            reasons.append(f"market regime: {market_regime.state}")
            if float(live_policy.get("factor", 1.0) or 1.0) < 0.999:
                reasons.append(f"event-risk haircut x{float(live_policy['factor']):.2f}")
            if float(live_policy.get("cap", 1.0) or 1.0) < 0.99:
                reasons.append(f"live cap {float(live_policy['cap']):.2f}")
            if bool(market_regime.cooldown_active):
                reasons.append("cooldown active")
            if market_regime.override_state:
                reasons.append("operator override")
            reasons.append(actionability["actionability_reason"])
            if trust.get("tail_trust_note"):
                reasons.append(str(trust["tail_trust_note"]))
            row_payload = {
                "symbol": symbol,
                "price": round(float(stage2_diags[symbol].get("latest_price", 0.0) or 0.0), 8),
                "pt2": pt2_label,
                "prob_2_model": round(float(prob_model), 4),
                "prob_2_pre_regime": round(float(prob_pre_regime), 4),
                "pre_policy_score": round(float(prob_pre_regime), 4),
                "prob_2_rank": round(float(prob_adjusted), 4),
                "prob_2": trust["display_score"],
                "live_score": trust["display_score"],
                "validated_floor": score_band["validated_floor"],
                "near_validated_floor": score_band["near_validated_floor"],
                "pre_policy_validated_floor": pre_policy_band["validated_floor"],
                "pre_policy_near_validated_floor": pre_policy_band["near_validated_floor"],
                "pre_policy_distance_to_validated": pre_policy_band["distance_to_validated"],
                "pre_policy_distance_to_validated_pct_points": pre_policy_band["distance_to_validated_pct_points"],
                "pre_policy_score_band": pre_policy_band["score_band"],
                "pre_policy_score_band_label": pre_policy_band["score_band_label"],
                "distance_to_validated": score_band["distance_to_validated"],
                "distance_to_validated_pct_points": score_band["distance_to_validated_pct_points"],
                "score_band": score_band["score_band"],
                "score_band_label": score_band["score_band_label"],
                "monitor_priority": score_band["monitor_priority"],
                "objective_score_band": score_band.get("objective_score_band"),
                "objective_score_band_label": score_band.get("objective_score_band_label"),
                "objective_monitor_priority": score_band.get("objective_monitor_priority"),
                "objective_quality_reference_rate": score_band.get("objective_quality_reference_rate"),
                "objective_quality_reference_source": score_band.get("objective_quality_reference_source"),
                "objective_distance_to_confirmed_shortlist": score_band.get("objective_distance_to_confirmed_shortlist"),
                "objective_distance_to_confirmed_shortlist_pct_points": score_band.get("objective_distance_to_confirmed_shortlist_pct_points"),
                "objective_confirmed_shortlist_floor": score_band.get("objective_confirmed_shortlist_floor"),
                "objective_strong_edge_floor": score_band.get("objective_strong_edge_floor"),
                "objective_priority_edge_floor": score_band.get("objective_priority_edge_floor"),
                "objective_elite_edge_floor": score_band.get("objective_elite_edge_floor"),
                "opportunity_score": trust["opportunity_score"],
                "probability_semantics": trust["probability_semantics"],
                "tail_trust_state": trust["tail_trust_state"],
                "tail_validated_threshold": trust["tail_validated_threshold"],
                "tail_trust_note": trust["tail_trust_note"],
                "risk": guard.get("risk"),
                "risk_reasons": guard.get("risk_reasons"),
                "downside_risk": guard.get("downside_risk"),
                "uncertainty": guard.get("uncertainty"),
                "uncertainty_reasons": guard.get("uncertainty_reasons"),
                "btc_regime_context": btc_regime,
                "market_regime_state": market_regime.state,
                "headline_risk": market_regime.headline_risk,
                "market_regime_score": market_regime.score,
                "market_regime_reasons": list(market_regime.reasons),
                "market_regime_actionability": market_regime.actionability_state,
                "cooldown_active": bool(market_regime.cooldown_active),
                "cooldown_until_utc": market_regime.cooldown_until_utc,
                "liquidity_tier": liquidity_tier,
                "actionability_tier": actionability["actionability_tier"],
                "actionability_rank": actionability["actionability_rank"],
                "actionability_type": actionability["actionability_type"],
                "actionability_evidence": actionability["actionability_evidence"],
                "actionability_reason": actionability["actionability_reason"],
                "policy_constraint_reason": actionability["policy_constraint_reason"],
                "contract_truth_state": actionability["contract_truth_state"],
                "contract_truth_semantics": actionability["contract_truth_semantics"],
                "temporal_tail_state": actionability["temporal_tail_state"],
                "temporal_tail_semantics": actionability["temporal_tail_semantics"],
                "live_threshold": round(float(live_policy.get("threshold", 0.0) or 0.0), 4),
                "base_live_threshold": round(float(live_policy.get("threshold", 0.0) or 0.0), 4),
                "threshold_policy_mode": "raw_threshold" if pipeline_mode == "raw_threshold" else "absolute",
                "threshold_math": ({"mode": "raw_threshold", "raw_threshold": round(float(raw_threshold), 4)} if pipeline_mode == "raw_threshold" else self.scanner._policy_math(factor=float(live_policy.get("factor", 1.0) or 1.0), cap=float(live_policy.get("cap", 0.95) or 0.95), threshold=float(live_policy.get("threshold", 0.0) or 0.0))),
                "regime_haircut_factor": round(float(live_policy.get("factor", 1.0) or 1.0), 4),
                "regime_cap": round(float(live_policy.get("cap", 0.95) or 0.95), 4),
                **self.scanner._visibility_band(live_score=trust["display_score"], live_threshold=float(live_policy.get("threshold", 0.0) or 0.0)),
                "operator_override_active": bool(market_regime.override_state),
                "reasons": reasons,
                "block_code": guard.get("block_code"),
                "model_hash": active_model_hash,
                "app_version": APP_VERSION,
                "was_capped": bool(adjustment_detail.get("guardrail_capped", False)),
                "panic_penalty": round(float(adjustment_detail.get("panic_penalty", 0.0) or 0.0), 4),
                "sector_penalty": round(float(adjustment_detail.get("sector_penalty", 0.0) or 0.0), 4),
                "binance_gap_penalty": round(float(adjustment_detail.get("binance_gap_penalty", 0.0) or 0.0), 4),
                "binance_lead_penalty": round(float(adjustment_detail.get("binance_lead_penalty", 0.0) or 0.0), 4),
                "post_model_total_penalty": round(float(adjustment_detail.get("total_penalty", 0.0) or 0.0), 4),
                "activity_bucket": self.scanner._activity_bucket(row),
                "liquidity_bucket": liquidity_bucket,
                "cohort_member": True,
                "cohort_mode": universe.diagnostics.get("selection_mode", "dynamic"),
                "candidate_stage": "stage2_final",
                "provisional": False,
                "deep_confirmed": True,
                "row_type": "candidate",
                "tracked_followup_symbol": False,
                "suppression_reason": None,
                "suppression_reason_detail": None,
                "display_bucket": "candidate",
                "informational_only": False,
                "is_actionable_now": True,
            }
            if pipeline_mode != "raw_threshold" and suppress_reason == "regime":
                suppressed_regime += 1
                row_payload["suppression_reason"] = "regime"
                row_payload["suppression_reason_detail"] = row_payload.get("policy_constraint_reason") or "blocked by live market regime policy"
                row_payload["display_bucket"] = "informational_suppressed"
                row_payload["informational_only"] = True
                row_payload["is_actionable_now"] = False
                row_payload["row_type"] = "suppressed"
                suppressed_rows.append(row_payload)
                continue
            if pipeline_mode != "raw_threshold" and cooldown_blocked:
                suppressed_cooldown += 1
                row_payload["suppression_reason"] = "cooldown"
                row_payload["suppression_reason_detail"] = row_payload.get("policy_constraint_reason") or "blocked by active cooldown"
                row_payload["display_bucket"] = "informational_suppressed"
                row_payload["informational_only"] = True
                row_payload["is_actionable_now"] = False
                row_payload["row_type"] = "suppressed"
                suppressed_rows.append(row_payload)
                continue
            threshold_candidates.append(row_payload)

        threshold_plan = self.scanner._build_threshold_plan(regime_state=market_regime.state, threshold_candidates=threshold_candidates) if pipeline_mode != "raw_threshold" else {}
        for row_payload in threshold_candidates:
            if pipeline_mode == "raw_threshold":
                effective_threshold = raw_threshold
                tier_plan = {"mode": "raw_threshold", "effective_math": {"raw_threshold": round(float(raw_threshold), 4)}}
            else:
                effective_threshold, tier_plan = self.scanner._effective_threshold_for_row(row_payload, threshold_plan)
            row_payload["base_live_threshold"] = round(float(row_payload.get("base_live_threshold", row_payload.get("live_threshold") or 0.0) or 0.0), 4)
            row_payload["live_threshold"] = round(float(effective_threshold), 4)
            row_payload["threshold_policy_mode"] = str(tier_plan.get("mode") or "absolute")
            row_payload["threshold_math"] = dict(tier_plan.get("effective_math") or row_payload.get("threshold_math") or {})
            row_payload.update(self.scanner._visibility_band(live_score=float(row_payload.get("live_score", row_payload.get("prob_2_rank") or 0.0) or 0.0), live_threshold=float(effective_threshold)))
            threshold_blocked = float(row_payload.get("prob_2_rank", 0.0) or 0.0) < float(effective_threshold)
            if threshold_blocked:
                suppressed_threshold += 1
                row_payload["suppression_reason"] = "threshold"
                row_payload["suppression_reason_detail"] = row_payload.get("policy_constraint_reason") or "below current live threshold"
                row_payload["display_bucket"] = "informational_suppressed"
                row_payload["informational_only"] = True
                row_payload["is_actionable_now"] = False
                row_payload["row_type"] = "suppressed"
                suppressed_rows.append(row_payload)
                continue
            row_payload["row_type"] = "visible"
            raw_scores.append(row_payload)

        rankable_rows = list(raw_scores) + list(suppressed_rows)
        rankable_rows.sort(key=self.scanner._informational_sort_key, reverse=True)
        for idx, score in enumerate(rankable_rows, start=1):
            score["candidate_rank_all"] = idx
            score["would_be_rank"] = idx
            score["pre_policy_rank"] = idx

        raw_scores = [row for row in rankable_rows if not bool(row.get("informational_only"))]
        raw_scores.sort(key=self.scanner._row_sort_key, reverse=True)
        for idx, score in enumerate(raw_scores, start=1):
            score["score_rank"] = idx
            score["display_bucket"] = "actionable"
            score["informational_only"] = False
            score["is_actionable_now"] = True
            score["row_type"] = "visible"
            score["tracked_followup_visible"] = False

        effective_max = self.config.stage2_panic_max_names if is_panic else self.config.stage2_max_names
        if market_regime.state == "amber":
            effective_max = min(effective_max, max(6, int(self.config.stage2_max_names * 0.65)))
        elif market_regime.state == "red":
            effective_max = min(effective_max, max(2, int(self.config.stage2_max_names * 0.20)))
        scores, trimmed_visible_rows, shortlist_meta = self.scanner._limit_visible_shortlist(raw_scores, effective_max=effective_max, tracked_priority_symbols=[])
        for idx, score in enumerate(scores, start=1):
            score["score_rank"] = idx
            score["tracked_followup_visible"] = False
        informational_rows: List[dict] = list(suppressed_rows)
        if self.config.informational_rankings_enabled and self.config.informational_include_display_trimmed and trimmed_visible_rows:
            for score in trimmed_visible_rows:
                trimmed = dict(score)
                trimmed["suppression_reason"] = "display_trim"
                trimmed["suppression_reason_detail"] = (
                    "watchlist candidate trimmed to keep the visible shortlist focused" if str(trimmed.get("actionability_tier") or "") == "watchlist" else "ranked candidate trimmed by output cap"
                )
                trimmed["display_bucket"] = "informational_suppressed"
                trimmed["informational_only"] = True
                trimmed["is_actionable_now"] = False
                informational_rows.append(trimmed)
        informational_overflow_rows: List[dict] = []
        if self.config.informational_rankings_enabled:
            informational_rows.sort(key=self.scanner._informational_sort_key, reverse=True)
            informational_cap = max(1, int(self.config.informational_rankings_max_names))
            informational_overflow_rows = informational_rows[informational_cap:]
            informational_rows = informational_rows[:informational_cap]
            for idx, row in enumerate(informational_rows, start=1):
                row["informational_rank"] = idx
                row["pre_policy_rank"] = row.get("pre_policy_rank") or row.get("candidate_rank_all") or idx
                row["display_bucket"] = "informational_suppressed"
                row["informational_only"] = True
                row["is_actionable_now"] = False
                row["row_type"] = "informational"
            base_rank = len(informational_rows)
            for idx, row in enumerate(informational_overflow_rows, start=1):
                row["informational_rank"] = base_rank + idx
                row["pre_policy_rank"] = row.get("pre_policy_rank") or row.get("candidate_rank_all") or (base_rank + idx)
                row["display_bucket"] = "informational_overflow"
                row["informational_only"] = True
                row["is_actionable_now"] = False
                row["row_type"] = "overflow"
                row.setdefault("suppression_reason", "display_trim")
                row.setdefault("suppression_reason_detail", "trimmed row preserved only in the replay pack because the informational cap was reached")
        else:
            informational_rows = []
            informational_overflow_rows = []

        all_ranked_rows = self.scanner._unique_rows_by_symbol(list(scores) + list(suppressed_rows) + list(trimmed_visible_rows))
        stage_summary = self.scanner._score_stage_summary(scores)
        decision_summary = self.scanner._build_decision_summary(
            visible_rows=scores,
            score_contract=score_contract,
            market_regime=market_regime,
            hidden_watchlist_rows=len(trimmed_visible_rows),
            blocked_rows=suppressed_rows,
        )
        score_diagnostics = self.scanner._score_diagnostics(
            visible_rows=scores,
            suppressed_rows=suppressed_rows,
            informational_rows=informational_rows,
            informational_overflow_rows=informational_overflow_rows,
            score_contract=score_contract,
        )
        candidate_quality = self.scanner._candidate_quality_diagnostics(
            stage1_input_rows=stage1_input_rows,
            stage1_guardrails=stage1_guardrails,
            stage1_diags=stage1_diags,
            stage1_candidates=stage1_candidates,
            stage1_selection_meta=stage1_selection_meta,
            stage2_diags=stage2_diags,
            final_rows=all_ranked_rows,
        )

        evaluated_rows: List[dict] = []
        for row in all_ranked_rows:
            outcome = self._resolve_outcome(
                symbol=str(row.get("symbol") or ""),
                entry_utc=as_of,
                entry_price=float(row.get("price") or 0.0),
                frame=histories.get(str(row.get("symbol") or "")),
            )
            evaluated_rows.append({
                **row,
                **outcome,
                "as_of_utc": as_of.isoformat(),
                "scan_scope": "replay",
            })

        selected_set = set(stage1_candidates)
        blocked_set = {symbol for symbol, guard in stage1_guardrails.items() if str(guard.get("block_code") or "") == "BLOCKED"}
        counterfactual_rows: List[dict] = []
        for symbol in stage1_input_rows.keys():
            price = float((stage1_diags.get(symbol) or {}).get("latest_price") or 0.0)
            outcome = self._resolve_outcome(symbol=symbol, entry_utc=as_of, entry_price=price, frame=histories.get(symbol))
            if symbol in selected_set:
                disposition = "selected"
            elif symbol in blocked_set:
                disposition = "blocked"
            else:
                disposition = "stage1_not_selected"
            stage1_row = dict(stage1_input_rows.get(symbol) or {})
            stage1_guard = dict(stage1_guardrails.get(symbol) or {})
            counterfactual_rows.append({
                "symbol": symbol,
                "as_of_utc": as_of.isoformat(),
                "stage1_disposition": disposition,
                "stage1_rank_all": (rank_map.get(symbol) or {}).get("stage1_rank_all"),
                "stage1_score": (rank_map.get(symbol) or {}).get("stage1_score"),
                "stage1_primary_rank": (stage1_selection_meta or {}).get("primary_ranks", {}).get(symbol),
                "stage1_recall_rank": (stage1_selection_meta or {}).get("recall_ranks", {}).get(symbol),
                "stage1_opportunity_rank": (stage1_selection_meta or {}).get("opportunity_ranks", {}).get(symbol),
                "stage1_opportunity_score": (stage1_selection_meta or {}).get("opportunity_scores", {}).get(symbol),
                "stage1_selection_source": (stage1_selection_meta or {}).get("selected_sources", {}).get(symbol),
                "stage1_selected": symbol in selected_set,
                "stage1_blocked": symbol in blocked_set,
                "liquidity_tier": self.scanner._liquidity_bucket(stage1_diags.get(symbol) or {}),
                "entry_price": round(price, 8) if price else None,
                "ret_15m": round(float(stage1_row.get("ret_15m", 0.0) or 0.0), 6),
                "ret_60m": round(float(stage1_row.get("ret_60m", 0.0) or 0.0), 6),
                "ret_6h": round(float(stage1_row.get("ret_6h", 0.0) or 0.0), 6),
                "ret_24h": round(float(stage1_row.get("ret_24h", 0.0) or 0.0), 6),
                "asset_vs_btc_1h": round(float(stage1_row.get("asset_vs_btc_1h", 0.0) or 0.0), 6),
                "rvol_1h": round(float(stage1_row.get("rvol_1h", 0.0) or 0.0), 6),
                "path_smoothness": round(float(stage1_row.get("path_smoothness", 0.0) or 0.0), 6),
                "candle_efficiency": round(float(stage1_row.get("candle_efficiency", 0.0) or 0.0), 6),
                "wickiness": round(float(stage1_row.get("wickiness", 0.0) or 0.0), 6),
                "downside_impulse": round(float(stage1_row.get("downside_impulse", 0.0) or 0.0), 6),
                "momentum_persistence_1h": round(float(stage1_row.get("momentum_persistence_1h", 0.0) or 0.0), 6),
                "move_vs_atr_ratio": round(float(stage1_row.get("move_vs_atr_ratio", 0.0) or 0.0), 6),
                "volume_acceleration": round(float(stage1_row.get("volume_acceleration", 0.0) or 0.0), 6),
                "uncertainty": round(float(stage1_guard.get("uncertainty", 0.0) or 0.0), 6),
                **outcome,
            })

        captured_rankable_rows: List[dict] = []
        if capture_full_rankable_rows:
            full_rankable_rows = self.scanner._unique_rows_by_symbol(list(rankable_rows))
            for row in full_rankable_rows:
                outcome = self._resolve_outcome(
                    symbol=str(row.get("symbol") or ""),
                    entry_utc=as_of,
                    entry_price=float(row.get("price") or 0.0),
                    frame=histories.get(str(row.get("symbol") or "")),
                )
                captured_rankable_rows.append({
                    **row,
                    **outcome,
                    "as_of_utc": as_of.isoformat(),
                    "scan_scope": "replay",
                })

        scan_summary = {
            "as_of_utc": as_of.isoformat(),
            "requested_symbols": requested,
            "symbols_returned_with_bars_count": returned_light,
            "stage1_feature_ready": stage1_feature_ready,
            "stage1_candidates": len(stage1_candidates),
            "stage1_dropped_by_rank": stage1_dropped_by_rank,
            "stage1_selection_mode": (stage1_selection_meta or {}).get("selection_mode") or "primary_only",
            "stage1_primary_slots": int((stage1_selection_meta or {}).get("primary_slots") or 0),
            "stage1_max_candidates": int(regime_candidate_cap),
            "stage1_selection_mode_override": stage1_selection_mode_override,
            "stage1_recall_reserve_slots": int((stage1_selection_meta or {}).get("recall_reserve_slots") or 0),
            "stage2_feature_ready": stage2_feature_ready,
            "stage2_scored": len(all_ranked_rows),
            "visible_rows": stage_summary.get("visible_rows", 0),
            "hidden_rows": max(0, len(all_ranked_rows) - int(stage_summary.get("visible_rows", 0) or 0)),
            "market_regime_state": market_regime.state,
            "market_regime_actionability": market_regime.actionability_state,
            "score_headline": score_diagnostics.get("headline"),
            "decision_headline": decision_summary.get("headline"),
            "pre_policy_max": ((score_diagnostics.get("pre_policy_score") or {}).get("max")),
            "live_max": ((score_diagnostics.get("live_score") or {}).get("max")),
            "validated_rows": int(decision_summary.get("validated_rows") or 0),
            "quality_hit_visible_count": sum(1 for r in evaluated_rows if str(r.get("row_type") or "") == "visible" and int(r.get("quality_touched") or 0) == 1),
            "quality_hit_non_visible_count": sum(1 for r in evaluated_rows if str(r.get("row_type") or "") != "visible" and int(r.get("quality_touched") or 0) == 1),
            "pipeline_mode": pipeline_mode,
        }
        return {
            "scan_summary": scan_summary,
            "rows": evaluated_rows,
            "counterfactual_rows": counterfactual_rows,
            "captured_rankable_rows": captured_rankable_rows,
        }

    def _build_counterfactual_summary(self, rows: List[dict]) -> dict:
        rows = list(rows or [])
        if not rows:
            return {
                "available": False,
                "headline": "No counterfactual rows yet",
                "summary": "Run a replay before judging missed opportunity rates.",
                "rows": [],
                "top_missed_quality_rows": [],
            }
        selected = [r for r in rows if bool(r.get("stage1_selected"))]
        not_selected = [r for r in rows if str(r.get("stage1_disposition") or "") == "stage1_not_selected"]
        blocked = [r for r in rows if bool(r.get("stage1_blocked"))]
        selectable = selected + not_selected
        selectable_quality = [r for r in selectable if int(r.get("quality_touched") or 0) == 1]
        selected_quality = [r for r in selected if int(r.get("quality_touched") or 0) == 1]
        missed_quality = [r for r in not_selected if int(r.get("quality_touched") or 0) == 1]
        selected_raw = [r for r in selected if int(r.get("raw_touched") or 0) == 1]
        missed_raw = [r for r in not_selected if int(r.get("raw_touched") or 0) == 1]
        recall = round(float(len(selected_quality)) / max(1, len(selectable_quality)), 4) if selectable_quality else None
        top_missed = sorted(missed_quality, key=lambda r: (float(r.get("mfe") or -999.0), float(r.get("end_ret") or -999.0), str(r.get("symbol") or "")), reverse=True)[:25]
        rows_out = []
        for label, bucket in (("selected", selected), ("stage1_not_selected", not_selected), ("blocked", blocked)):
            rows_out.append({
                "bucket": label,
                "count": len(bucket),
                "quality_hit_rate": self.review_packs._bucket_summary(bucket).get("quality_hit_rate"),
                "raw_hit_rate": self.review_packs._bucket_summary(bucket).get("raw_hit_rate"),
                "avg_end_ret": self.review_packs._bucket_summary(bucket).get("avg_end_ret"),
                "avg_mae": self.review_packs._bucket_summary(bucket).get("avg_mae"),
                "quality_hit_count": sum(1 for r in bucket if int(r.get("quality_touched") or 0) == 1),
                "raw_hit_count": sum(1 for r in bucket if int(r.get("raw_touched") or 0) == 1),
            })
        selection_source_rows = []
        for label in ("primary", "recall_reserve", "primary_backfill", "followup_reserve"):
            bucket = [r for r in selected if str(r.get("stage1_selection_source") or "") == label]
            if not bucket:
                continue
            source_summary = self.review_packs._bucket_summary(bucket)
            selection_source_rows.append({
                "selection_source": label,
                "count": len(bucket),
                "quality_hit_rate": source_summary.get("quality_hit_rate"),
                "raw_hit_rate": source_summary.get("raw_hit_rate"),
                "avg_end_ret": source_summary.get("avg_end_ret"),
                "avg_mae": source_summary.get("avg_mae"),
                "quality_hit_count": sum(1 for r in bucket if int(r.get("quality_touched") or 0) == 1),
                "raw_hit_count": sum(1 for r in bucket if int(r.get("raw_touched") or 0) == 1),
            })
        if selectable_quality and recall is not None and recall < 0.75:
            headline = "Stage1 is missing a meaningful share of later quality opportunities"
            summary = "The replay shows that a notable share of later quality touches were in stage1-eligible names that never advanced into stage2. That is a recall problem, not just a scoring problem."
        elif selectable_quality:
            headline = "Stage1 is retaining most later quality opportunities"
            summary = "Most later quality touches were already inside the stage1-selected set, which is a healthier sign for recall."
        else:
            headline = "No later quality opportunities were observed in the replay window"
            summary = "This window did not produce any stage1-selectable quality opportunities, so recall cannot be judged from it."
        return {
            "available": True,
            "headline": headline,
            "summary": summary,
            "selectable_rows": len(selectable),
            "selectable_quality_opportunities": len(selectable_quality),
            "selected_quality_opportunities": len(selected_quality),
            "missed_quality_opportunities": len(missed_quality),
            "selected_raw_opportunities": len(selected_raw),
            "missed_raw_opportunities": len(missed_raw),
            "stage1_quality_recall": recall,
            "rows": rows_out,
            "selection_source_rows": selection_source_rows,
            "top_missed_quality_rows": top_missed,
        }

    def _rank_bucket_label(self, rank: Any) -> str:
        try:
            rank_i = int(rank)
        except Exception:
            return "unranked"
        if rank_i <= 0:
            return "unranked"
        if rank_i <= 10:
            return "1-10"
        if rank_i <= 20:
            return "11-20"
        if rank_i <= 30:
            return "21-30"
        if rank_i <= 40:
            return "31-40"
        if rank_i <= 60:
            return "41-60"
        return "61+"

    def _build_stage1_rank_bucket_summary(self, rows: List[dict], *, rank_field: str) -> List[dict]:
        buckets: Dict[str, List[dict]] = {}
        for row in list(rows or []):
            label = self._rank_bucket_label(row.get(rank_field))
            buckets.setdefault(label, []).append(row)
        order = ["1-10", "11-20", "21-30", "31-40", "41-60", "61+", "unranked"]
        out = []
        for label in order:
            bucket = buckets.get(label) or []
            if not bucket:
                continue
            selected = [r for r in bucket if bool(r.get("stage1_selected"))]
            missed_quality = [r for r in bucket if str(r.get("stage1_disposition") or "") == "stage1_not_selected" and int(r.get("quality_touched") or 0) == 1]
            out.append({
                "rank_bucket": label,
                "count": len(bucket),
                "selected_count": len(selected),
                "selected_share": round(float(len(selected)) / float(max(1, len(bucket))), 4),
                "quality_hit_rate": self.review_packs._bucket_summary(bucket).get("quality_hit_rate"),
                "raw_hit_rate": self.review_packs._bucket_summary(bucket).get("raw_hit_rate"),
                "avg_end_ret": self.review_packs._bucket_summary(bucket).get("avg_end_ret"),
                "avg_mae": self.review_packs._bucket_summary(bucket).get("avg_mae"),
                "missed_quality_count": len(missed_quality),
                "missed_quality_share": round(float(len(missed_quality)) / float(max(1, sum(1 for r in bucket if int(r.get("quality_touched") or 0) == 1))), 4),
            })
        return out

    def _build_stage1_feature_delta_summary(self, rows: List[dict]) -> dict:
        features = [
            "ret_15m", "ret_60m", "ret_6h", "ret_24h", "asset_vs_btc_1h", "rvol_1h",
            "path_smoothness", "candle_efficiency", "wickiness", "downside_impulse",
            "momentum_persistence_1h", "move_vs_atr_ratio", "volume_acceleration", "uncertainty",
        ]
        groups = {
            "selected_quality_hits": [r for r in rows if bool(r.get("stage1_selected")) and int(r.get("quality_touched") or 0) == 1],
            "missed_quality_hits": [r for r in rows if str(r.get("stage1_disposition") or "") == "stage1_not_selected" and int(r.get("quality_touched") or 0) == 1],
            "selected_non_hits": [r for r in rows if bool(r.get("stage1_selected")) and int(r.get("quality_touched") or 0) == 0],
        }
        def _mean(bucket: List[dict], key: str):
            vals = []
            for row in bucket:
                try:
                    vals.append(float(row.get(key)))
                except Exception:
                    continue
            if not vals:
                return None
            return round(sum(vals) / float(len(vals)), 6)
        rows_out = []
        for feat in features:
            selected_quality = _mean(groups["selected_quality_hits"], feat)
            missed_quality = _mean(groups["missed_quality_hits"], feat)
            selected_non = _mean(groups["selected_non_hits"], feat)
            rows_out.append({
                "feature": feat,
                "selected_quality_hits_mean": selected_quality,
                "missed_quality_hits_mean": missed_quality,
                "selected_non_hits_mean": selected_non,
                "missed_minus_selected_quality": round((missed_quality - selected_quality), 6) if selected_quality is not None and missed_quality is not None else None,
                "missed_minus_selected_non_hits": round((missed_quality - selected_non), 6) if selected_non is not None and missed_quality is not None else None,
            })
        return {
            "available": True,
            "group_sizes": {name: len(bucket) for name, bucket in groups.items()},
            "rows": rows_out,
        }

    def _simulate_stage1_mode(self, rows: List[dict], *, mode: str, max_candidates: int, recall_reserve_frac: float, recall_reserve_min: int, recall_reserve_max: int, promotion_overflow_window: int) -> List[dict]:
        rows = [r for r in rows if not bool(r.get("stage1_blocked"))]
        primary_sorted = sorted([r for r in rows if r.get("stage1_primary_rank") is not None], key=lambda r: (int(r.get("stage1_primary_rank") or 999999), str(r.get("symbol") or "")))
        recall_sorted = sorted([r for r in rows if r.get("stage1_recall_rank") is not None], key=lambda r: (int(r.get("stage1_recall_rank") or 999999), str(r.get("symbol") or "")))
        opportunity_sorted = sorted([r for r in rows if r.get("stage1_opportunity_rank") is not None], key=lambda r: (int(r.get("stage1_opportunity_rank") or 999999), str(r.get("symbol") or "")))
        effective_max = max(1, int(max_candidates))
        reserve_n = int(round(effective_max * max(0.0, float(recall_reserve_frac))))
        reserve_n = max(0, reserve_n)
        reserve_n = max(reserve_n, max(0, int(recall_reserve_min))) if effective_max > 8 else min(reserve_n, effective_max)
        reserve_n = min(reserve_n, max(0, int(recall_reserve_max)))
        reserve_n = min(reserve_n, max(0, effective_max // 2))
        primary_slots = max(1, effective_max - reserve_n)
        selected: List[dict] = []
        selected_symbols = set()
        if mode == "primary_only":
            for row in primary_sorted[:effective_max]:
                selected.append(row)
                selected_symbols.add(str(row.get("symbol") or ""))
            return selected
        if mode == "stage1_opportunity_model":
            ranked = opportunity_sorted if opportunity_sorted else primary_sorted
            for row in ranked[:effective_max]:
                selected.append(row)
                selected_symbols.add(str(row.get("symbol") or ""))
            return selected
        if mode == "hybrid_primary_plus_recall_reserve":
            for row in primary_sorted[:primary_slots]:
                selected.append(row)
                selected_symbols.add(str(row.get("symbol") or ""))
            ranked = recall_sorted
        elif mode == "primary_plus_near_miss_recall_promotion":
            for row in primary_sorted[:primary_slots]:
                selected.append(row)
                selected_symbols.add(str(row.get("symbol") or ""))
            ranked = [r for r in recall_sorted if int(r.get("stage1_primary_rank") or 999999) <= (effective_max + max(1, int(promotion_overflow_window)))]
        elif mode == "primary_plus_opportunity_reserve":
            for row in primary_sorted[:primary_slots]:
                selected.append(row)
                selected_symbols.add(str(row.get("symbol") or ""))
            ranked = opportunity_sorted if opportunity_sorted else recall_sorted
        else:
            ranked = primary_sorted
        added = 0
        for row in ranked:
            symbol = str(row.get("symbol") or "")
            if symbol in selected_symbols:
                continue
            if added >= reserve_n or len(selected) >= effective_max:
                break
            selected.append(row)
            selected_symbols.add(symbol)
            added += 1
        for row in primary_sorted:
            symbol = str(row.get("symbol") or "")
            if len(selected) >= effective_max:
                break
            if symbol in selected_symbols:
                continue
            selected.append(row)
            selected_symbols.add(symbol)
        return selected

    def _build_stage1_promotion_audit(self, rows: List[dict]) -> dict:
        rows = list(rows or [])
        if not rows:
            return {"available": False, "headline": "No stage1 promotion audit available yet", "rows": []}
        grouped: Dict[str, List[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("as_of_utc") or "unknown"), []).append(row)
        selectable_quality_total = sum(1 for row in rows if not bool(row.get("stage1_blocked")) and int(row.get("quality_touched") or 0) == 1)
        mode_names = ["primary_only", "hybrid_primary_plus_recall_reserve", "primary_plus_near_miss_recall_promotion", "stage1_opportunity_model", "primary_plus_opportunity_reserve"]
        out = []
        for mode in mode_names:
            selected_rows: List[dict] = []
            for scan_rows in grouped.values():
                selected_rows.extend(self._simulate_stage1_mode(
                    scan_rows,
                    mode=mode,
                    max_candidates=int(getattr(self.config, "stage1_max_candidates", 40) or 40),
                    recall_reserve_frac=float(getattr(self.config, "stage1_recall_reserve_frac", 0.25) or 0.25),
                    recall_reserve_min=int(getattr(self.config, "stage1_recall_reserve_min", 6) or 6),
                    recall_reserve_max=int(getattr(self.config, "stage1_recall_reserve_max", 12) or 12),
                    promotion_overflow_window=int(getattr(self.config, "stage1_promotion_overflow_window", 20) or 20),
                ))
            selected_quality = sum(1 for row in selected_rows if int(row.get("quality_touched") or 0) == 1)
            selected_raw = sum(1 for row in selected_rows if int(row.get("raw_touched") or 0) == 1)
            summary = self.review_packs._bucket_summary(selected_rows)
            out.append({
                "mode": mode,
                "selected_count": len(selected_rows),
                "quality_hit_count": selected_quality,
                "raw_hit_count": selected_raw,
                "quality_hit_rate": summary.get("quality_hit_rate"),
                "raw_hit_rate": summary.get("raw_hit_rate"),
                "avg_end_ret": summary.get("avg_end_ret"),
                "avg_mae": summary.get("avg_mae"),
                "stage1_quality_recall": round(float(selected_quality) / float(max(1, selectable_quality_total)), 4) if selectable_quality_total else None,
            })
        baseline = next((r for r in out if r.get("mode") == "primary_only"), None)
        best = max(out, key=lambda r: ((float(r.get("stage1_quality_recall") or 0.0) - 0.5 * abs(float(r.get("avg_end_ret") or 0.0))), float(r.get("quality_hit_rate") or 0.0))) if out else None
        if best and baseline and best.get("mode") != baseline.get("mode") and (float(best.get("stage1_quality_recall") or 0.0) > float(baseline.get("stage1_quality_recall") or 0.0) + 0.03):
            headline = f"Alternative stage1 mode may improve recall relative to primary_only: {best.get('mode')}"
        else:
            headline = "No audited stage1 promotion rule clearly beats primary_only yet"
        return {
            "available": True,
            "headline": headline,
            "baseline_mode": "primary_only",
            "rows": out,
        }

    def _build_replay_summary(self, *, timestamps: List[datetime], scan_summaries: List[dict], replay_rows: List[dict], counterfactual_rows: List[dict], universe, limitations: List[str], pipeline_mode: str = "full", raw_threshold: float = 0.30) -> dict:
        surfaced_summary = self.review_packs._build_recent_evidence_summary(replay_rows, model_fingerprint=str((self.scanner.state.model_metadata.get("pt2") or {}).get("model_fingerprint") or "unknown"))
        policy_audit = self.review_packs._policy_audit(replay_rows, runs=[{"evaluation_complete": True} for _ in scan_summaries])
        symbol_repeatability = self.review_packs._build_symbol_repeatability_summary(replay_rows)
        outlier_concentration = self.review_packs._build_outlier_concentration_summary(replay_rows)
        counterfactual_summary = self._build_counterfactual_summary(counterfactual_rows)
        stage1_primary_rank_buckets = self._build_stage1_rank_bucket_summary(counterfactual_rows, rank_field="stage1_primary_rank")
        stage1_recall_rank_buckets = self._build_stage1_rank_bucket_summary(counterfactual_rows, rank_field="stage1_recall_rank")
        stage1_feature_deltas = self._build_stage1_feature_delta_summary(counterfactual_rows)
        stage1_promotion_audit = self._build_stage1_promotion_audit(counterfactual_rows)
        visible_rows = [r for r in replay_rows if str(r.get("row_type") or "") == "visible"]
        non_visible_rows = [r for r in replay_rows if str(r.get("row_type") or "") != "visible"]
        visible_bucket = self.review_packs._bucket_summary(visible_rows)
        non_visible_bucket = self.review_packs._bucket_summary(non_visible_rows)
        top_scans = sorted(scan_summaries, key=lambda r: (int(r.get("validated_rows") or 0), float(r.get("live_max") or 0.0), float(r.get("pre_policy_max") or 0.0)), reverse=True)[:10]
        headline = "Replay evidence is too thin to judge ranking quality"
        if replay_rows:
            if (visible_bucket.get("quality_hit_rate") or 0.0) > ((non_visible_bucket.get("quality_hit_rate") or 0.0) + 0.05):
                headline = "Visible replay rows beat the hidden remainder in this historical window"
            else:
                headline = "Replay ranking edge is weak or unclear in this historical window"
        return {
            "available": True,
            "app_version": APP_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "window": {
                "start_utc": timestamps[0].isoformat(),
                "end_utc": timestamps[-1].isoformat(),
                "scan_count": len(timestamps),
                "scan_step_minutes": int(round((timestamps[1] - timestamps[0]).total_seconds() / 60.0)) if len(timestamps) > 1 else None,
            },
            "headline": headline,
            "pipeline_mode": pipeline_mode,
            "raw_threshold": round(float(raw_threshold), 4),
            "limitations": limitations,
            "universe": {
                "selection_mode": universe.diagnostics.get("selection_mode"),
                "selected_for_fetch_count": len(universe.selected_for_fetch),
                "eligible_count": len(universe.eligible),
            },
            "surfaced_evidence": surfaced_summary,
            "policy_audit": policy_audit,
            "symbol_repeatability": symbol_repeatability,
            "outlier_concentration": outlier_concentration,
            "counterfactual": counterfactual_summary,
            "stage1_primary_rank_buckets": stage1_primary_rank_buckets,
            "stage1_recall_rank_buckets": stage1_recall_rank_buckets,
            "stage1_feature_deltas": stage1_feature_deltas,
            "stage1_promotion_audit": stage1_promotion_audit,
            "visible_bucket": visible_bucket,
            "non_visible_bucket": non_visible_bucket,
            "scan_summaries": scan_summaries,
            "top_scans": top_scans,
            "replay_rows": replay_rows,
            "counterfactual_rows": counterfactual_rows,
        }

    def _build_replay_pack(self, summary: dict) -> Path:
        window = dict(summary.get("window") or {})
        safe_start = str(window.get("start_utc") or "unknown").replace(":", "").replace("-", "")[:13]
        safe_end = str(window.get("end_utc") or "unknown").replace(":", "").replace("-", "")[:13]
        pack_path = self.pack_dir / f"replay_pack_{APP_VERSION.replace('.', '_')}_{safe_start}_{safe_end}.zip"
        replay_rows = list(summary.get("replay_rows") or [])
        counter_rows = list(summary.get("counterfactual_rows") or [])
        scan_summaries = list(summary.get("scan_summaries") or [])
        counter_summary = dict(summary.get("counterfactual") or {})
        top_missed = list(counter_summary.get("top_missed_quality_rows") or [])
        with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("replay_summary.json", json.dumps({k: v for k, v in summary.items() if k not in {"replay_rows", "counterfactual_rows", "scan_summaries"}}, indent=2, default=str))
            zf.writestr("replay_manifest.json", json.dumps({
                "app_version": APP_VERSION,
                "generated_at_utc": summary.get("generated_at_utc"),
                "scan_count": len(scan_summaries),
                "replay_row_count": len(replay_rows),
                "counterfactual_row_count": len(counter_rows),
                "headline": summary.get("headline"),
            }, indent=2))
            zf.writestr("replay_scan_summaries.csv", _csv_bytes(scan_summaries))
            zf.writestr("replay_visible_rows.csv", _csv_bytes([r for r in replay_rows if str(r.get("row_type") or "") == "visible"]))
            zf.writestr("replay_non_visible_rows.csv", _csv_bytes([r for r in replay_rows if str(r.get("row_type") or "") != "visible"]))
            zf.writestr("replay_counterfactual_rows.csv", _csv_bytes(counter_rows))
            zf.writestr("replay_top_missed_quality_rows.csv", _csv_bytes(top_missed))
            zf.writestr("replay_threshold_bands.csv", _csv_bytes(self._threshold_band_rows(summary.get("surfaced_evidence") or {}), fieldnames=["threshold", "count", "visible_count", "non_visible_count", "quality_hit_rate", "raw_hit_rate", "avg_end_ret", "avg_mae"]))
            zf.writestr("replay_symbol_repeatability.csv", _csv_bytes(list((summary.get("symbol_repeatability") or {}).get("rows") or [])))
            zf.writestr("replay_outlier_concentration.json", json.dumps(summary.get("outlier_concentration") or {}, indent=2, default=str))
            zf.writestr("replay_counterfactual_summary.json", json.dumps(counter_summary, indent=2, default=str))
            zf.writestr("replay_stage1_primary_rank_buckets.csv", _csv_bytes(list(summary.get("stage1_primary_rank_buckets") or [])))
            zf.writestr("replay_stage1_recall_rank_buckets.csv", _csv_bytes(list(summary.get("stage1_recall_rank_buckets") or [])))
            zf.writestr("replay_stage1_feature_deltas.json", json.dumps(summary.get("stage1_feature_deltas") or {}, indent=2, default=str))
            zf.writestr("replay_stage1_promotion_audit.json", json.dumps(summary.get("stage1_promotion_audit") or {}, indent=2, default=str))
            zf.writestr("replay_policy_audit.json", json.dumps(summary.get("policy_audit") or {}, indent=2, default=str))
            zf.writestr("replay_pipeline_ablation.json", json.dumps(summary.get("pipeline_ablation") or {}, indent=2, default=str))
        return pack_path

    def _pipeline_ablation_summary(self, full_summary: dict, raw_summary: dict) -> dict:
        full_visible = dict(full_summary.get("visible_bucket") or {})
        raw_visible = dict(raw_summary.get("visible_bucket") or {})
        full_counter = dict(full_summary.get("counterfactual") or {})
        raw_counter = dict(raw_summary.get("counterfactual") or {})
        full_q = float(full_visible.get("quality_hit_rate") or 0.0)
        raw_q = float(raw_visible.get("quality_hit_rate") or 0.0)
        headline = "Full pipeline still looks stronger than raw-threshold ablation"
        if raw_q > (full_q + 0.01):
            headline = "Raw-threshold ablation outperformed the full pipeline in this replay window"
        elif abs(raw_q - full_q) <= 0.01:
            headline = "Raw-threshold ablation and full pipeline were broadly similar in this replay window"
        return {
            "available": True,
            "headline": headline,
            "baseline_mode": full_summary.get("pipeline_mode") or "full",
            "comparison_mode": raw_summary.get("pipeline_mode") or "raw_threshold",
            "rows": [
                {
                    "mode": full_summary.get("pipeline_mode") or "full",
                    "visible_quality_hit_rate": round(full_q, 4),
                    "visible_raw_hit_rate": round(float(full_visible.get("raw_hit_rate") or 0.0), 4),
                    "visible_avg_end_ret": round(float(full_visible.get("avg_end_ret") or 0.0), 6),
                    "stage1_quality_recall": round(float(full_counter.get("stage1_quality_recall") or 0.0), 4),
                    "resolved_rows": int((full_summary.get("surfaced_evidence") or {}).get("resolved_rows") or 0),
                },
                {
                    "mode": raw_summary.get("pipeline_mode") or "raw_threshold",
                    "visible_quality_hit_rate": round(raw_q, 4),
                    "visible_raw_hit_rate": round(float(raw_visible.get("raw_hit_rate") or 0.0), 4),
                    "visible_avg_end_ret": round(float(raw_visible.get("avg_end_ret") or 0.0), 6),
                    "stage1_quality_recall": round(float(raw_counter.get("stage1_quality_recall") or 0.0), 4),
                    "resolved_rows": int((raw_summary.get("surfaced_evidence") or {}).get("resolved_rows") or 0),
                },
            ],
        }

    def _threshold_band_rows(self, surfaced_evidence: dict) -> List[dict]:
        rows = []
        for threshold, bucket in (surfaced_evidence.get("threshold_bands") or {}).items():
            rows.append({"threshold": threshold, **dict(bucket or {})})
        return rows


def _csv_bytes(rows: List[dict], fieldnames: List[str] | None = None) -> bytes:
    rows = list(rows or [])
    if not rows and not fieldnames:
        return b""
    buf = io.StringIO()
    names = list(fieldnames or [])
    if not names:
        union = set()
        for row in rows:
            union.update(dict(row).keys())
        names = sorted(union)
    writer = csv.DictWriter(buf, fieldnames=names, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _normalize_csv_value(v) for k, v in dict(row).items()})
    return buf.getvalue().encode("utf-8")


def _normalize_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value
