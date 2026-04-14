from __future__ import annotations

import gc
import hashlib
import logging
import threading
from typing import List

import numpy as np
import pandas as pd

from .coinbase_client import CoinbaseClient
from .config import AppConfig
from .demo_data import STABLES
from .features import build_training_frame
from .modeling import reconcile_runtime_metadata, train_pt2
from .state import AppState
from .universe import UniverseBuilder

logger = logging.getLogger(__name__)

from .version import APP_VERSION



def _is_stablecoin_pair(symbol: str) -> bool:
    base = str(symbol).split("-", 1)[0].upper()
    return base in STABLES

def _downcast_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce training memory footprint without changing the feature set."""
    if frame.empty:
        return frame
    frame = frame.copy()
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_float_dtype(series):
            frame[col] = pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_integer_dtype(series):
            frame[col] = pd.to_numeric(series, downcast="integer")
    return frame



class TrainingService:
    def __init__(self, config: AppConfig, state: AppState, client: CoinbaseClient):
        self.config = config
        self.state = state
        self.client = client
        self._lock = threading.Lock()

    def start_training(self) -> bool:
        if self._lock.locked():
            return False
        t = threading.Thread(target=self._train, daemon=True, name="trainer")
        t.start()
        return True

    def _train(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        self.state.training_started("discovering training universe")
        try:
            result = self._build_and_train()
            self.state.training_finished(result=result, error=None)
        except Exception as exc:
            logger.exception("training_failed error=%s", exc)
            self.state.training_finished(result=None, error=f"{type(exc).__name__}: {exc}")
        finally:
            self._lock.release()

    def _build_and_train(self) -> dict:
        self.state.training_progress("discover_universe", "loading Coinbase products for training")
        products = self.client.list_products()
        currencies = self.client.list_currencies()
        volume_map = self.client.get_volume_summary()
        universe = UniverseBuilder(self.config).build(products, currencies, volume_map)
        ordered_symbols = [p["id"] for p in universe.eligible]
        symbols = self._select_training_symbols(ordered_symbols)
        symbols = self._ensure_context_symbols(symbols, ordered_symbols)
        self.state.training_progress(
            "discover_universe",
            f"selected {len(symbols)} training symbols from {len(ordered_symbols)} eligible",
            symbols_total=len(symbols),
        )

        lookback_bars = max(self.config.stage2_lookback_5m_bars, int((self.config.train_lookback_days * 24 * 60) / 5))
        self.state.training_progress("context_fetch", f"fetching BTC/ETH context with {lookback_bars} bars", symbols_total=len(symbols))
        btc_df = self.client.get_candles("BTC-USD", lookback_bars) if "BTC-USD" in symbols else None
        eth_df = self.client.get_candles("ETH-USD", lookback_bars) if "ETH-USD" in symbols else None

        frames: List[pd.DataFrame] = []
        skipped: List[dict] = []
        rows_accumulated = 0
        for idx, symbol in enumerate(symbols, start=1):
            try:
                self.state.training_progress("build_frames", f"fetching training history for {symbol} ({idx}/{len(symbols)})", symbols_total=len(symbols))
                logger.info("training_symbol_start %s/%s symbol=%s", idx, len(symbols), symbol)
                df = self.client.get_candles(symbol, lookback_bars)
                observed_bars = int(df.attrs.get("observed_bars", int((df["volume"] > 0).sum()) if not df.empty else 0))
                if len(df) < max(self.config.train_feature_warmup_5m_bars, self.config.candles_per_horizon + 48):
                    skipped.append({"symbol": symbol, "reason": f"insufficient_history bars={len(df)}"})
                    self.state.training_progress("build_frames", f"skipped {symbol}: insufficient history", inc_done=True, inc_skipped=True, rows_accumulated=rows_accumulated)
                    continue
                if observed_bars < max(self.config.stage2_min_observed_5m_bars, 48):
                    skipped.append({"symbol": symbol, "reason": f"insufficient_observed_bars observed_bars={observed_bars}"})
                    self.state.training_progress("build_frames", f"skipped {symbol}: insufficient observed bars", inc_done=True, inc_skipped=True, rows_accumulated=rows_accumulated)
                    continue
                # v2.6.0: pass quality label thresholds to training frame builder
                frame = build_training_frame(
                    symbol=symbol,
                    df=df,
                    btc_df=btc_df,
                    eth_df=eth_df,
                    sample_every=self.config.train_sample_every_n_bars,
                    horizon_bars=self.config.candles_per_horizon,
                    target_move_pct=self.config.target_move_pct,
                    warmup_bars=self.config.train_feature_warmup_5m_bars,
                    quality_max_mae=self.config.quality_max_mae,
                    quality_min_end_ret=self.config.quality_min_end_ret,
                )
                if frame.empty:
                    skipped.append({"symbol": symbol, "reason": "empty_training_frame"})
                    self.state.training_progress("build_frames", f"skipped {symbol}: empty frame", inc_done=True, inc_skipped=True, rows_accumulated=rows_accumulated)
                    continue
                frame = _downcast_training_frame(frame)
                frames.append(frame)
                rows_accumulated += len(frame)
                self.state.training_progress(
                    "build_frames",
                    f"prepared {symbol} rows={len(frame)}",
                    inc_done=True,
                    rows_accumulated=rows_accumulated,
                    symbols_total=len(symbols),
                )
                del df, frame
                gc.collect()
            except Exception as exc:
                skipped.append({"symbol": symbol, "reason": str(exc)})
                self.state.training_progress("build_frames", f"failed {symbol}: {exc}", inc_done=True, inc_skipped=True, rows_accumulated=rows_accumulated)
                logger.warning("training_symbol_skipped symbol=%s error=%s", symbol, exc)

        training_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        del frames
        gc.collect()
        if training_df.empty:
            raise RuntimeError("training produced no rows")

        # v2.6.0: log quality label statistics
        quality_event_rate = float(training_df["y"].mean())
        raw_touch_rate = float(training_df["y_raw_touch"].mean()) if "y_raw_touch" in training_df else quality_event_rate
        logger.info(
            "training_data_summary rows=%s quality_event_rate=%.3f raw_touch_rate=%.3f",
            len(training_df), quality_event_rate, raw_touch_rate,
        )

        self.state.training_progress("fit_model", f"fitting pt2 on {len(training_df)} rows (quality_rate={quality_event_rate:.2%})", rows_accumulated=len(training_df), symbols_total=len(symbols))
        # v2.7.0: pass config for adjusted-score simulation during training
        cfg_dict = {
            "app_version": APP_VERSION,
            "btc_panic_threshold": self.config.btc_panic_threshold,
            "panic_threshold_boost": self.config.panic_threshold_boost,
            "downside_cap": self.config.downside_cap,
            "uncertainty_cap": self.config.uncertainty_cap,
            "target_move_pct": self.config.target_move_pct,
            "target_horizon_minutes": self.config.target_horizon_minutes,
            "quality_max_mae": self.config.quality_max_mae,
            "quality_min_end_ret": self.config.quality_min_end_ret,
            "app_version": APP_VERSION,
        }
        bundle = train_pt2(training_df, cfg_dict=cfg_dict)
        used_symbol_set = set(training_df["symbol"].unique().tolist())
        requested_symbols = [symbol for symbol in symbols if not _is_stablecoin_pair(symbol)]
        trained_cohort_symbols = [symbol for symbol in requested_symbols if symbol in used_symbol_set]
        cohort_hash = hashlib.sha256("|".join(trained_cohort_symbols).encode("utf-8")).hexdigest()[:16] if trained_cohort_symbols else "none"
        bundle.metadata.update({
            "trained_cohort_symbols": trained_cohort_symbols,
            "trained_cohort_size": len(trained_cohort_symbols),
            "trained_cohort_hash": cohort_hash,
            "training_symbol_selection_method": "top_liquidity_locked",
            "live_universe_mode": self.config.live_universe_mode,
            "training_candidate_pool_size": len(ordered_symbols),
            "training_symbols_requested": requested_symbols,
        })
        meta_seed = {"trained": True, "path": self.config.model_path_pt2, **bundle.metadata}
        meta, bundle_contracts = reconcile_runtime_metadata(
            meta_seed,
            existing_status=self.state.get_status(),
            min_count=self.config.tail_validation_min_count,
            min_wilson_lift=self.config.tail_validation_min_wilson_lift,
            min_precision_floor=self.config.tail_validation_min_precision_floor,
            unvalidated_tail_cap=self.config.tail_unvalidated_cap,
            scanner_contract_source="recomputed_runtime_adjusted",
            threshold_suppression_contract_source="recomputed_runtime_adjusted",
        )
        bundle.metadata.update({k: v for k, v in meta.items() if k not in {"trained", "path"}})
        bundle.save(self.config.model_path_pt2)
        self.state.set_model_metadata(meta)
        self.state.update_status(
            score_contract=bundle_contracts["score_contract"],
            score_contract_live=bundle_contracts["score_contract_live"],
            score_contract_raw=bundle_contracts["score_contract_raw"],
            score_reconciliation=bundle_contracts["score_reconciliation"],
        )

        # v2.6.0: improved threshold recommendation with bootstrap
        recommended_threshold, high_confidence_ready, readiness_details = self._recommend_live_threshold(bundle.metadata)

        training_result = {
            **meta,
            "training_universe_count": len(ordered_symbols),
            "training_symbols_requested": list(meta.get("training_symbols_requested") or requested_symbols),
            "training_symbols_used": list(meta.get("training_symbols_used") or trained_cohort_symbols),
            "trained_cohort_symbols": list(meta.get("trained_cohort_symbols") or trained_cohort_symbols),
            "trained_cohort_size": int(meta.get("trained_cohort_size", len(trained_cohort_symbols)) or len(trained_cohort_symbols)),
            "trained_cohort_hash": meta.get("trained_cohort_hash", cohort_hash),
            "training_symbol_selection_method": "top_liquidity_locked",
            "live_universe_mode": self.config.live_universe_mode,
            "training_rows": int(len(training_df)),
            "quality_event_rate": quality_event_rate,
            "raw_touch_rate": raw_touch_rate,
            "training_skipped": skipped,
            "feature_group_counts": {
                "momentum": len([c for c in bundle.metadata["feature_cols"] if c.startswith("ret_") or c in {"impulse_60m", "accel_30_60"}]),
                "trend_vol": len([c for c in bundle.metadata["feature_cols"] if c in {"ema_fast_gap", "ema_slow_gap", "adx_proxy", "atr_pct", "rv_1h", "rv_6h", "rv_24h", "bb_width"}]),
                "structure": len([c for c in bundle.metadata["feature_cols"] if c in {"wickiness", "jumpiness", "path_smoothness", "reversal_rate", "failed_breakout"}]),
                "path_quality": len([c for c in bundle.metadata["feature_cols"] if c in {"momentum_persistence_1h", "up_volume_ratio_1h", "time_since_impulse", "rv_ratio_1h_24h"}]),
                "history_liquidity": len([c for c in bundle.metadata["feature_cols"] if c in {"history_bars_ratio_24h", "history_bars_ratio_7d", "observed_bar_density_24h", "observed_bar_density_7d", "nonzero_volume_rate_24h", "dollar_vol_24h_log"}]),
                "context": len([c for c in bundle.metadata["feature_cols"] if "btc" in c or "eth" in c or c in {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}]),
            },
            "recommended_live_threshold": recommended_threshold,
            "high_confidence_ready": high_confidence_ready,
            "readiness_details": readiness_details,
            "label_definition": f"quality-conditioned: touched +{self.config.target_move_pct:.1%} within {self.config.target_horizon_minutes}m AND mae > {self.config.quality_max_mae} AND end_ret > {self.config.quality_min_end_ret}",
            "model_type": bundle.model_type,
        }
        self.state.training_progress(
            "persist_model",
            f"training complete rows={len(training_df)} used={len(training_result['training_symbols_used'])} model={bundle.model_type}",
            rows_accumulated=len(training_df),
            symbols_total=len(symbols),
        )
        return training_result

    def _select_training_symbols(self, ordered_symbols: List[str]) -> List[str]:
        max_symbols = max(1, self.config.train_max_symbols)
        selected = list(ordered_symbols[:max_symbols])
        deduped: List[str] = []
        seen = set()
        for symbol in selected:
            if symbol not in seen:
                deduped.append(symbol)
                seen.add(symbol)
        return deduped[:max_symbols]

    def _ensure_context_symbols(self, selected: List[str], ordered_symbols: List[str]) -> List[str]:
        out = list(selected)
        for ctx in ["BTC-USD", "ETH-USD"]:
            if ctx in ordered_symbols and ctx not in out:
                if len(out) < self.config.train_max_symbols:
                    out.append(ctx)
                else:
                    out[-1] = ctx
        deduped = []
        seen = set()
        for symbol in out:
            if symbol not in seen:
                deduped.append(symbol)
                seen.add(symbol)
        return deduped

    def _recommend_live_threshold(self, metrics: dict) -> tuple[float, bool, dict]:
        """Use adjusted-score metrics for readiness, with upper-tail diagnostics."""
        auc = float(metrics.get("auc_holdout", 0.0))
        adjusted_auc = float(metrics.get("adjusted_auc_holdout", auc))
        details: dict = {
            "auc": auc,
            "adjusted_auc": adjusted_auc,
            "auc_gate_passed": min(auc, adjusted_auc) >= 0.58,
        }

        if min(auc, adjusted_auc) < 0.58:
            details["reason"] = f"AUC gate failed raw={auc:.3f} adjusted={adjusted_auc:.3f}"
            return 0.60, False, details

        stability = metrics.get("temporal_stability_adjusted") or metrics.get("temporal_stability_model") or {}
        worst_auc = stability.get("worst_auc") if isinstance(stability, dict) else None

        for threshold in (0.80, 0.75, 0.70, 0.60):
            key = f"{threshold:.2f}".replace(".", "_")
            count = int(metrics.get(f"adjusted_count_at_{key}", metrics.get(f"count_at_{key}", 0)))
            precision = float(metrics.get(f"adjusted_precision_at_{key}", metrics.get(f"precision_at_{key}", 0.0)))
            wilson = float(metrics.get(f"adjusted_wilson_lower_{key}", metrics.get(f"wilson_lower_{key}", 0.0)))

            if count >= 20 and wilson >= 0.50:
                challenge_prec = float(metrics.get("adjusted_challenge_set_precision_at_0_60", metrics.get("challenge_set_precision_at_0_60", 0.0)))
                btc_panic_prec = float(metrics.get("adjusted_btc_panic_challenge_precision", metrics.get("btc_panic_challenge_precision", 0.0)))
                btc_panic_count = int(metrics.get("adjusted_btc_panic_challenge_count", metrics.get("btc_panic_challenge_count", 0)))

                details["recommended_threshold"] = threshold
                details["precision"] = precision
                details["wilson_lower"] = wilson
                details["count"] = count
                details["challenge_precision"] = challenge_prec
                details["btc_panic_precision"] = btc_panic_prec
                details["worst_window_auc"] = worst_auc
                details["score_quantiles"] = metrics.get("adjusted_score_quantiles", metrics.get("score_quantiles", {}))
                details["top_bucket_lift"] = metrics.get("adjusted_top_bucket_lift", metrics.get("top_bucket_lift", {}))

                if challenge_prec < 0.40 and count > 10:
                    details["reason"] = f"challenge set precision {challenge_prec:.3f} too low"
                    return threshold, False, details

                if btc_panic_count >= 5 and btc_panic_prec < 0.30:
                    details["reason"] = f"BTC panic precision {btc_panic_prec:.3f} too low (n={btc_panic_count})"
                    return threshold, False, details

                if worst_auc is not None and worst_auc < 0.52:
                    details["reason"] = f"worst adjusted temporal window AUC {worst_auc:.3f} too low"
                    return threshold, False, details

                return threshold, True, details

        details["reason"] = "no threshold met count >= 20 AND Wilson lower >= 0.50 on adjusted score"
        details["score_quantiles"] = metrics.get("adjusted_score_quantiles", metrics.get("score_quantiles", {}))
        details["top_bucket_lift"] = metrics.get("adjusted_top_bucket_lift", metrics.get("top_bucket_lift", {}))
        details["dead_upper_tail"] = bool(metrics.get("adjusted_dead_upper_tail", metrics.get("dead_upper_tail", False)))
        return 0.60, False, details
