
from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .coinbase_client import CoinbaseClient
from .config import AppConfig
from .demo_data import STABLES
from .features import build_training_frame
from .modeling import (
    ModelBundle,
    _purged_time_split,
    _simulate_adjusted_scores,
    evaluate_predictions,
)
from .persist import atomic_write_json, ensure_dir, read_json
from .review_runs import ReviewPackService
from .runtime_scope import current_runtime_scope
from .version import APP_VERSION

logger = logging.getLogger(__name__)


class ComparisonCancelledError(RuntimeError):
    """Raised when a background comparison run is cancelled by operator request."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ''):
            return default
        return float(value)
    except Exception:
        return default


def _is_stablecoin_pair(symbol: str) -> bool:
    base = str(symbol).split('-', 1)[0].upper()
    return base in STABLES


def _downcast_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_float_dtype(series):
            frame[col] = pd.to_numeric(series, downcast='float')
        elif pd.api.types.is_integer_dtype(series):
            frame[col] = pd.to_numeric(series, downcast='integer')
    return frame


class OfflineChallengerComparisonService:
    def __init__(self, config: AppConfig, client: CoinbaseClient, review_packs: ReviewPackService):
        self.config = config
        self.client = client
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / 'challenger_comparison')
        self.summary_path = self.root_dir / 'latest_challenger_comparison_summary.json'
        self.pack_path = self.root_dir / 'latest_challenger_comparison_pack.zip'
        self.source_audit_summary_path = Path(config.model_dir) / 'fresh_retrain_audit' / 'latest_fresh_retrain_audit_summary.json'
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def latest_summary(self) -> dict:
        summary = read_json(self.summary_path, {
            'available': False,
            'running': False,
            'status': 'idle',
            'headline': 'No offline challenger comparison has run yet.',
            'summary': 'Run the comparison after the fresh retrain audit builds a non-promoting shadow candidate.',
            'app_version': APP_VERSION,
        })
        return self._repair_stale_summary(summary)

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def start_run(self) -> dict:
        if self._lock.locked():
            current = self.latest_summary()
            current['already_running'] = True
            return current
        self._stop_event.clear()
        thread = threading.Thread(target=self._run, daemon=True, name='challenger_comparison')
        self._thread = thread
        thread.start()
        return self.latest_summary()

    def stop_run(self) -> dict:
        summary = self.latest_summary()
        thread_alive = (self._thread is not None) and self._thread.is_alive()
        if not (self._lock.locked() or thread_alive or bool(summary.get('running'))):
            summary['already_stopped'] = True
            summary['stop_requested'] = False
            return summary
        self._stop_event.set()
        updated = dict(summary)
        updated.update({
            'available': True,
            'running': True,
            'status': 'stopping',
            'generated_at_utc': _utc_now_iso(),
            'last_heartbeat_at_utc': _utc_now_iso(),
            'stop_requested': True,
            'headline': 'Stopping offline challenger comparison…',
            'summary': 'Cancellation requested. Waiting for the current stage to exit safely.',
        })
        atomic_write_json(self.summary_path, updated)
        return updated

    def _repair_stale_summary(self, summary: dict) -> dict:
        if not isinstance(summary, dict):
            return summary
        running_like = bool(summary.get('running')) or str(summary.get('status') or '').lower() in {'running', 'stopping'}
        thread_alive = (self._thread is not None) and self._thread.is_alive()
        if running_like and (not self._lock.locked()) and (not thread_alive):
            repaired = dict(summary)
            repaired.update({
                'available': True,
                'running': False,
                'status': 'interrupted',
                'generated_at_utc': _utc_now_iso(),
                'finished_at_utc': repaired.get('finished_at_utc') or _utc_now_iso(),
                'headline': 'Offline challenger comparison was interrupted',
                'summary': 'A previous challenger comparison no longer has a live worker thread. Treat this as interrupted/stale state, not an active run.',
                'stale_run_detected': True,
                'stop_requested': False,
            })
            atomic_write_json(self.summary_path, repaired)
            return repaired
        return summary

    def _raise_if_cancel_requested(self) -> None:
        if self._stop_event.is_set():
            raise ComparisonCancelledError('Offline challenger comparison cancelled by operator request.')

    def _source_audit_summary(self) -> dict:
        data = read_json(self.source_audit_summary_path, {})
        return data if isinstance(data, dict) else {}

    def _running_summary_base(self, current_version: dict, audit_summary: dict) -> dict:
        scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        started = _utc_now_iso()
        source_context = dict((audit_summary or {}).get('source_context') or {})
        current_live = source_context.get('current_live_path') or audit_summary.get('current_live_path') or {}
        return {
            'available': True,
            'running': True,
            'status': 'running',
            'app_version': APP_VERSION,
            'generated_at_utc': started,
            'started_at_utc': started,
            'last_heartbeat_at_utc': started,
            'state_scope_key': scope.get('state_scope_key'),
            'headline': 'Offline challenger comparison is running',
            'summary': 'Scoring incumbent and shadow candidate on a shared offline evaluation frame. Live scoring remains unchanged.',
            'current_live_path': {
                'decision_checkpoint_outcome': current_live.get('decision_checkpoint_outcome'),
                'resolved_visible_rows': int(current_live.get('resolved_visible_rows') or 0),
                'visible_quality_hit_rate': current_live.get('visible_quality_hit_rate'),
                'non_visible_quality_hit_rate': current_live.get('non_visible_quality_hit_rate'),
                'stage1_selection_mode': current_live.get('stage1_selection_mode') or self.config.stage1_selection_mode,
                'live_raw_threshold': current_live.get('live_raw_threshold') or self.config.live_raw_threshold,
            },
            'source_context': source_context,
            'source_audit_headline': audit_summary.get('headline'),
            'source_audit_verdict': audit_summary.get('verdict'),
            'stop_requested': False,
            'progress': {
                'stage': 'queued',
                'detail': 'Queued',
                'fraction': 0.0,
                'completed_symbols': 0,
                'total_symbols': 0,
                'current_symbol': None,
            },
        }

    def _write_running_summary(self, current_version: dict, audit_summary: dict) -> None:
        atomic_write_json(self.summary_path, self._running_summary_base(current_version, audit_summary))

    def _update_running_progress(
        self,
        *,
        stage: str,
        detail: str,
        fraction: float | None = None,
        completed_symbols: int | None = None,
        total_symbols: int | None = None,
        current_symbol: str | None = None,
        extra: dict | None = None,
    ) -> None:
        summary = self.latest_summary()
        if not summary:
            summary = {
                'available': True,
                'running': True,
                'status': 'running',
                'app_version': APP_VERSION,
                'started_at_utc': _utc_now_iso(),
            }
        progress = dict(summary.get('progress') or {})
        progress.update({
            'stage': stage,
            'detail': detail,
            'fraction': round(max(0.0, min(1.0, float(fraction if fraction is not None else progress.get('fraction') or 0.0))), 4),
            'completed_symbols': int(completed_symbols if completed_symbols is not None else progress.get('completed_symbols') or 0),
            'total_symbols': int(total_symbols if total_symbols is not None else progress.get('total_symbols') or 0),
            'current_symbol': current_symbol,
        })
        summary.update({
            'available': True,
            'running': True,
            'status': 'running',
            'generated_at_utc': _utc_now_iso(),
            'last_heartbeat_at_utc': _utc_now_iso(),
            'headline': 'Offline challenger comparison is running',
            'summary': detail,
            'progress': progress,
            'stop_requested': bool(self._stop_event.is_set()),
        })
        if extra:
            summary.update(extra)
        atomic_write_json(self.summary_path, summary)

    def _run(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            audit_summary = self._source_audit_summary()
            try:
                current_version = self.review_packs.get_current_version_summary() or {}
            except FileNotFoundError:
                current_version = {}
            self._raise_if_cancel_requested()
            self._write_running_summary(current_version, audit_summary)
            summary = self._build_summary(current_version=current_version, audit_summary=audit_summary)
            atomic_write_json(self.summary_path, summary)
        except ComparisonCancelledError as exc:
            cancelled = self.latest_summary()
            cancelled.update({
                'available': True,
                'running': False,
                'status': 'cancelled',
                'generated_at_utc': _utc_now_iso(),
                'finished_at_utc': _utc_now_iso(),
                'headline': 'Offline challenger comparison cancelled',
                'summary': str(exc),
                'error': None,
                'stop_requested': True,
                'progress': {
                    'stage': 'cancelled',
                    'detail': str(exc),
                    'fraction': 1.0,
                    'completed_symbols': int(((cancelled.get('progress') or {}).get('completed_symbols')) or 0),
                    'total_symbols': int(((cancelled.get('progress') or {}).get('total_symbols')) or 0),
                    'current_symbol': None,
                },
            })
            atomic_write_json(self.summary_path, cancelled)
        except Exception as exc:
            logger.exception('challenger_comparison_failed error=%s', exc)
            failed = self.latest_summary()
            failed.update({
                'available': True,
                'running': False,
                'status': 'failed',
                'generated_at_utc': _utc_now_iso(),
                'finished_at_utc': _utc_now_iso(),
                'headline': 'Offline challenger comparison failed',
                'summary': f'{type(exc).__name__}: {exc}',
                'error': f'{type(exc).__name__}: {exc}',
                'stop_requested': bool(self._stop_event.is_set()),
                'progress': {
                    'stage': 'failed',
                    'detail': f'{type(exc).__name__}: {exc}',
                    'fraction': 1.0,
                    'completed_symbols': int(((failed.get('progress') or {}).get('completed_symbols')) or 0),
                    'total_symbols': int(((failed.get('progress') or {}).get('total_symbols')) or 0),
                    'current_symbol': None,
                },
            })
            atomic_write_json(self.summary_path, failed)
        finally:
            self._stop_event.clear()
            self._thread = None
            self._lock.release()

    def _build_summary(self, *, current_version: dict, audit_summary: dict) -> dict:
        self._raise_if_cancel_requested()
        if str((audit_summary or {}).get('status')) != 'completed':
            raise FileNotFoundError('Run the fresh retrain audit first so a completed shadow candidate exists.')
        if not bool((audit_summary or {}).get('non_promoting', False)):
            raise RuntimeError('Source audit is not marked non-promoting; refusing to compare for live action.')

        artifacts = dict(audit_summary.get('artifact_paths') or {})
        shadow_model_path = str((artifacts.get('shadow_model_path')) or '')
        if not shadow_model_path:
            raise FileNotFoundError('Shadow model path missing from fresh retrain audit summary.')
        incumbent_model_path = str((artifacts.get('incumbent_model_path')) or self.config.model_path_pt2 or '')
        if not incumbent_model_path:
            raise FileNotFoundError('Incumbent model path missing for challenger comparison.')

        incumbent = ModelBundle.load(incumbent_model_path)
        if incumbent is None:
            raise FileNotFoundError('Incumbent pt2 bundle not found for challenger comparison.')
        challenger = ModelBundle.load(shadow_model_path)
        if challenger is None:
            raise FileNotFoundError('Shadow challenger bundle not found.')

        spec = audit_summary.get('shadow_training_spec') or {}
        symbols = list(((audit_summary.get('shadow_model_result') or {}).get('training_symbols_used') or []))
        if not symbols:
            raise FileNotFoundError('No training symbols found in fresh retrain audit summary.')

        eval_frame, eval_meta = self._build_shared_eval_frame(symbols=symbols, spec=spec)

        source_context = dict((audit_summary or {}).get('source_context') or {})
        source_live = source_context.get('current_live_path') or audit_summary.get('current_live_path') or {}
        source_threshold = _f(source_live.get('live_raw_threshold'), self.config.live_raw_threshold) or self.config.live_raw_threshold
        cfg_dict = {
            'target_move_pct': self.config.target_move_pct,
            'target_horizon_minutes': self.config.target_horizon_minutes,
            'quality_max_mae': self.config.quality_max_mae,
            'quality_min_end_ret': self.config.quality_min_end_ret,
            'live_raw_threshold': source_threshold,
            'tail_unvalidated_cap': self.config.tail_unvalidated_cap,
            'downside_cap': self.config.downside_cap,
            'uncertainty_cap': self.config.uncertainty_cap,
            'btc_panic_threshold': self.config.btc_panic_threshold,
            'panic_threshold_boost': self.config.panic_threshold_boost,
            'app_version': APP_VERSION,
        }
        incumbent_metrics = self._evaluate_bundle(label='incumbent_live_pt2', bundle=incumbent, eval_frame=eval_frame, cfg_dict=cfg_dict)
        self._raise_if_cancel_requested()
        self._update_running_progress(stage='score_shadow', detail='Scoring shadow challenger on shared evaluation frame…', fraction=0.90, completed_symbols=eval_meta.get('symbols_used_count'), total_symbols=eval_meta.get('symbols_used_count'))
        challenger_metrics = self._evaluate_bundle(label='shadow_retrain_candidate', bundle=challenger, eval_frame=eval_frame, cfg_dict=cfg_dict)
        comparison = self._compare_models(incumbent=incumbent_metrics, challenger=challenger_metrics)

        source_context = dict((audit_summary or {}).get('source_context') or {})
        current_live = source_context.get('current_live_path') or audit_summary.get('current_live_path') or {}
        headline = 'Shadow challenger comparison is ready for offline review'
        summary = comparison.get('summary') or 'Offline challenger comparison finished.'
        finished_at = _utc_now_iso()
        eval_symbols = int((eval_meta or {}).get('symbols_used_count') or 0)
        result = {
            'available': True,
            'running': False,
            'status': 'completed',
            'app_version': APP_VERSION,
            'generated_at_utc': finished_at,
            'finished_at_utc': finished_at,
            'state_scope_key': current_runtime_scope(self.config.model_dir, app_version=APP_VERSION).get('state_scope_key'),
            'source_context': source_context,
            'headline': headline,
            'summary': summary,
            'verdict': comparison.get('verdict'),
            'recommended_action': comparison.get('recommended_action'),
            'recommended_action_reason': comparison.get('recommended_action_reason'),
            'non_promoting': True,
            'live_promotion_blocked': True,
            'current_live_path': {
                'decision_checkpoint_outcome': current_live.get('decision_checkpoint_outcome'),
                'resolved_visible_rows': int(current_live.get('resolved_visible_rows') or 0),
                'visible_quality_hit_rate': current_live.get('visible_quality_hit_rate'),
                'non_visible_quality_hit_rate': current_live.get('non_visible_quality_hit_rate'),
                'stage1_selection_mode': current_live.get('stage1_selection_mode') or self.config.stage1_selection_mode,
                'live_raw_threshold': current_live.get('live_raw_threshold') or self.config.live_raw_threshold,
            },
            'source_fresh_retrain_audit': {
                'generated_at_utc': audit_summary.get('generated_at_utc'),
                'headline': audit_summary.get('headline'),
                'verdict': audit_summary.get('verdict'),
                'shadow_model_path': shadow_model_path,
                'incumbent_model_path': incumbent_model_path,
            },
            'shared_evaluation_frame': eval_meta,
            'incumbent_model': incumbent_metrics,
            'challenger_model': challenger_metrics,
            'comparison': comparison,
            'progress': {
                'stage': 'completed',
                'detail': 'Run finished. Use the summary and pack links above.',
                'fraction': 1.0,
                'completed_symbols': eval_symbols,
                'total_symbols': eval_symbols,
                'current_symbol': None,
            },
            'notes': [
                'This is an offline-only incumbent-vs-challenger comparison. It must not promote a live model automatically.',
                'Treat this as a decision-grade filter for the next causal move, not as proof of live success on its own.',
            ],
        }
        self._build_pack(result=result, current_version=current_version, audit_summary=audit_summary)
        return result

    def _build_shared_eval_frame(self, *, symbols: List[str], spec: dict) -> tuple[pd.DataFrame, dict]:
        train_lookback_days = int(spec.get('train_lookback_days') or 90)
        sample_every_n_bars = int(spec.get('train_sample_every_n_bars') or 2)
        lookback_bars = max(self.config.stage2_lookback_5m_bars, int((train_lookback_days * 24 * 60) / 5))
        unique_symbols = []
        seen = set()
        for symbol in symbols:
            symbol = str(symbol)
            if symbol and symbol not in seen:
                unique_symbols.append(symbol)
                seen.add(symbol)
        total_symbols = max(1, len(unique_symbols))
        self._raise_if_cancel_requested()
        self._update_running_progress(stage='context', detail='Loading shared BTC/ETH context candles…', fraction=0.06, total_symbols=total_symbols)
        btc_df = self.client.get_candles('BTC-USD', lookback_bars) if 'BTC-USD' in seen else None
        eth_df = self.client.get_candles('ETH-USD', lookback_bars) if 'ETH-USD' in seen else None

        frames: List[pd.DataFrame] = []
        skipped: List[dict] = []
        rows_accumulated = 0
        for idx, symbol in enumerate(unique_symbols, start=1):
            self._raise_if_cancel_requested()
            frac = 0.08 + (0.72 * ((idx - 1) / total_symbols))
            self._update_running_progress(
                stage='shared_eval_frame_build',
                detail=f'Building shared evaluation rows for {symbol} ({idx}/{total_symbols})…',
                fraction=frac,
                completed_symbols=idx - 1,
                total_symbols=total_symbols,
                current_symbol=symbol,
            )
            try:
                if _is_stablecoin_pair(symbol):
                    skipped.append({'symbol': symbol, 'reason': 'stablecoin_pair'})
                    continue
                df = self.client.get_candles(symbol, lookback_bars)
                observed_bars = int(df.attrs.get('observed_bars', int((df['volume'] > 0).sum()) if not df.empty else 0))
                if len(df) < max(self.config.train_feature_warmup_5m_bars, self.config.candles_per_horizon + 48):
                    skipped.append({'symbol': symbol, 'reason': f'insufficient_history bars={len(df)}'})
                    continue
                if observed_bars < max(self.config.stage2_min_observed_5m_bars, 48):
                    skipped.append({'symbol': symbol, 'reason': f'insufficient_observed_bars observed_bars={observed_bars}'})
                    continue
                frame = build_training_frame(
                    symbol=symbol,
                    df=df,
                    btc_df=btc_df,
                    eth_df=eth_df,
                    sample_every=sample_every_n_bars,
                    horizon_bars=self.config.candles_per_horizon,
                    target_move_pct=self.config.target_move_pct,
                    warmup_bars=self.config.train_feature_warmup_5m_bars,
                    quality_max_mae=self.config.quality_max_mae,
                    quality_min_end_ret=self.config.quality_min_end_ret,
                )
                if frame is None or frame.empty:
                    skipped.append({'symbol': symbol, 'reason': 'empty_training_frame'})
                    continue
                frame['symbol'] = symbol
                frames.append(frame)
                rows_accumulated += int(len(frame))
            except Exception as exc:
                skipped.append({'symbol': symbol, 'reason': f'{type(exc).__name__}: {exc}'})
            if idx % 10 == 0:
                gc.collect()
        self._raise_if_cancel_requested()
        if not frames:
            raise RuntimeError('No shared evaluation frame rows were built for the challenger comparison.')
        training_df = pd.concat(frames, ignore_index=True)
        training_df = _downcast_training_frame(training_df)
        training_df = training_df.sort_values('ts').reset_index(drop=True)
        if len(training_df) < 300:
            raise RuntimeError(f'Not enough shared evaluation rows for challenger comparison: {len(training_df)}')
        df_train, df_val, df_test, embargo_dropped = _purged_time_split(training_df)
        self._raise_if_cancel_requested()
        self._update_running_progress(stage='shared_eval_frame_ready', detail='Shared evaluation frame built. Scoring incumbent…', fraction=0.84, completed_symbols=total_symbols, total_symbols=total_symbols)
        return df_test.reset_index(drop=True), {
            'train_lookback_days': train_lookback_days,
            'sample_every_n_bars': sample_every_n_bars,
            'symbols_requested_count': len(symbols),
            'symbols_used_count': len(unique_symbols),
            'rows_all': int(len(training_df)),
            'rows_test': int(len(df_test)),
            'embargo_dropped': int(embargo_dropped),
            'skipped_symbols': skipped,
        }

    def _threshold_concentration(self, symbols: pd.Series, scores: np.ndarray, threshold: float) -> dict:
        mask = scores >= threshold
        count = int(mask.sum())
        if count == 0:
            return {
                'threshold': threshold,
                'row_count': 0,
                'top_symbol': None,
                'top_symbol_count': 0,
                'top_symbol_share': None,
                'top_3_share': None,
                'top_symbols': [],
                'unique_symbols': 0,
            }
        subset = symbols.loc[mask].astype(str)
        counts = subset.value_counts()
        top_items = [
            {'symbol': str(sym), 'count': int(cnt)}
            for sym, cnt in counts.head(10).items()
        ]
        top_symbol, top_count = next(iter(counts.items()))
        top_3_share = float(counts.head(3).sum()) / float(count)
        return {
            'threshold': threshold,
            'row_count': count,
            'top_symbol': str(top_symbol),
            'top_symbol_count': int(top_count),
            'top_symbol_share': float(top_count) / float(count),
            'top_3_share': top_3_share,
            'top_symbols': top_items,
            'unique_symbols': int(counts.shape[0]),
        }

    def _top_bucket(self, y: np.ndarray, scores: np.ndarray, frac: float) -> dict:
        if len(scores) == 0:
            return {'count': 0, 'quality_rate': None}
        count = max(1, int(round(len(scores) * frac)))
        order = np.argsort(scores)[::-1][:count]
        return {
            'count': int(count),
            'quality_rate': float(np.mean(y[order])) if count > 0 else None,
        }

    def _shortlist_proxy(self, *, eval_frame: pd.DataFrame, adjusted_scores: np.ndarray, threshold: float) -> dict:
        df = pd.DataFrame({
            'ts': pd.to_datetime(eval_frame['ts'], utc=True),
            'symbol': eval_frame['symbol'].astype(str).values,
            'score': np.asarray(adjusted_scores, dtype=float),
            'y': eval_frame['y'].astype(int).values,
        })
        if df.empty:
            return {'threshold': threshold, 'visible_rows': 0, 'hidden_rows': 0, 'visible_quality_rate': None, 'hidden_quality_rate': None, 'quality_gap': None, 'scans': 0, 'scans_with_visible': 0, 'avg_visible_rows_per_scan': None}
        df['visible'] = df['score'] >= float(threshold)
        visible = df[df['visible']]
        hidden = df[~df['visible']]
        scans = int(df['ts'].nunique())
        visible_counts = df.groupby('ts', sort=False)['visible'].sum()
        visible_rate = float(visible['y'].mean()) if not visible.empty else None
        hidden_rate = float(hidden['y'].mean()) if not hidden.empty else None
        gap = None if visible_rate is None or hidden_rate is None else float(visible_rate - hidden_rate)
        return {
            'threshold': float(threshold),
            'visible_rows': int(len(visible)),
            'hidden_rows': int(len(hidden)),
            'visible_quality_rate': visible_rate,
            'hidden_quality_rate': hidden_rate,
            'quality_gap': gap,
            'scans': scans,
            'scans_with_visible': int((visible_counts > 0).sum()) if scans else 0,
            'avg_visible_rows_per_scan': float(visible_counts.mean()) if scans else None,
        }

    def _evaluate_bundle(self, *, label: str, bundle: ModelBundle, eval_frame: pd.DataFrame, cfg_dict: dict) -> dict:
        pred_model = bundle.predict_proba(eval_frame)
        adjusted_scores, parity = _simulate_adjusted_scores(pred_model, eval_frame, cfg_dict or {})
        shortlist_threshold = float(cfg_dict.get('live_raw_threshold', 0.35) or 0.35)
        raw_metrics = evaluate_predictions(eval_frame, pred_model, shortlist_threshold=shortlist_threshold)
        adjusted_metrics = evaluate_predictions(eval_frame, adjusted_scores, shortlist_threshold=shortlist_threshold)
        y = eval_frame['y'].astype(int).values
        threshold_stats = {}
        for th in (0.35, 0.40, 0.45, 0.50):
            suffix = {0.35: '0_35', 0.40: '0_40', 0.45: '0_45', 0.50: '0_50'}[th]
            threshold_stats[str(th)] = {
                'threshold': th,
                'count': int(adjusted_metrics.get(f'count_at_{suffix}', 0) or 0),
                'precision': adjusted_metrics.get(f'precision_at_{suffix}'),
                'wilson_lower': adjusted_metrics.get(f'wilson_lower_{suffix}'),
            }
        live_threshold = float(cfg_dict.get('live_raw_threshold', 0.35) or 0.35)
        result = {
            'label': label,
            'model_type': bundle.model_type,
            'adjusted_auc_holdout': adjusted_metrics.get('auc_holdout'),
            'adjusted_brier_holdout': adjusted_metrics.get('brier_holdout'),
            'raw_auc_holdout': raw_metrics.get('auc_holdout'),
            'raw_brier_holdout': raw_metrics.get('brier_holdout'),
            'quality_event_rate_holdout': adjusted_metrics.get('quality_event_rate_holdout'),
            'raw_touch_rate_holdout': adjusted_metrics.get('raw_touch_rate_holdout'),
            'score_quantiles_adjusted': {
                'max': adjusted_metrics.get('max'),
                'q50': adjusted_metrics.get('q50'),
                'q75': adjusted_metrics.get('q75'),
                'q90': adjusted_metrics.get('q90'),
                'q95': adjusted_metrics.get('q95'),
                'q99': adjusted_metrics.get('q99'),
            },
            'dead_upper_tail': bool((adjusted_metrics.get('max') or 0) < 0.60),
            'threshold_stats_adjusted': threshold_stats,
            'top_bucket_quality_rate': {
                'top_1pct': self._top_bucket(y, adjusted_scores, 0.01),
                'top_5pct': self._top_bucket(y, adjusted_scores, 0.05),
                'top_10pct': self._top_bucket(y, adjusted_scores, 0.10),
            },
            'concentration': {
                '0.35': self._threshold_concentration(eval_frame['symbol'], adjusted_scores, 0.35),
                '0.45': self._threshold_concentration(eval_frame['symbol'], adjusted_scores, 0.45),
                '0.50': self._threshold_concentration(eval_frame['symbol'], adjusted_scores, 0.50),
            },
            'shortlist_proxy': {
                'live_threshold': self._shortlist_proxy(eval_frame=eval_frame, adjusted_scores=adjusted_scores, threshold=live_threshold),
                '0.45': self._shortlist_proxy(eval_frame=eval_frame, adjusted_scores=adjusted_scores, threshold=0.45),
            },
            'scan_shortlist_utility': {
                'threshold': adjusted_metrics.get('scan_shortlist_threshold'),
                'utility_score': adjusted_metrics.get('scan_shortlist_utility_score'),
                'mean_gap': adjusted_metrics.get('scan_shortlist_mean_gap'),
                'pairwise_win_rate': adjusted_metrics.get('scan_shortlist_pairwise_win_rate'),
                'top1_mean_quality': adjusted_metrics.get('scan_shortlist_top1_mean_quality'),
                'top3_mean_quality': adjusted_metrics.get('scan_shortlist_top3_mean_quality'),
                'top5_mean_quality': adjusted_metrics.get('scan_shortlist_top5_mean_quality'),
                'avg_visible_rows_per_scan': adjusted_metrics.get('scan_shortlist_avg_visible_rows_per_scan'),
                'scan_capture_rate': adjusted_metrics.get('scan_shortlist_scan_capture_rate'),
                'visible_quality_rate_mean': adjusted_metrics.get('scan_shortlist_visible_quality_rate_mean'),
                'hidden_quality_rate_mean': adjusted_metrics.get('scan_shortlist_hidden_quality_rate_mean'),
                'pairwise_comparable_scans': adjusted_metrics.get('scan_shortlist_pairwise_comparable_scans'),
            },
            'score_reconciliation': parity,
        }
        return result

    def _compare_models(self, *, incumbent: dict, challenger: dict) -> dict:
        inc_auc = _f(incumbent.get('adjusted_auc_holdout'), 0.0) or 0.0
        chal_auc = _f(challenger.get('adjusted_auc_holdout'), 0.0) or 0.0
        inc_brier = _f(incumbent.get('adjusted_brier_holdout'), 1.0) or 1.0
        chal_brier = _f(challenger.get('adjusted_brier_holdout'), 1.0) or 1.0
        inc_t45 = ((incumbent.get('threshold_stats_adjusted') or {}).get('0.45') or {})
        chal_t45 = ((challenger.get('threshold_stats_adjusted') or {}).get('0.45') or {})
        inc_t50 = ((incumbent.get('threshold_stats_adjusted') or {}).get('0.50') or {})
        chal_t50 = ((challenger.get('threshold_stats_adjusted') or {}).get('0.50') or {})
        inc_top1 = _f((((incumbent.get('top_bucket_quality_rate') or {}).get('top_1pct') or {}).get('quality_rate')), 0.0) or 0.0
        chal_top1 = _f((((challenger.get('top_bucket_quality_rate') or {}).get('top_1pct') or {}).get('quality_rate')), 0.0) or 0.0
        inc_conc45 = (((incumbent.get('concentration') or {}).get('0.45')) or {})
        chal_conc45 = (((challenger.get('concentration') or {}).get('0.45')) or {})
        inc_share45 = _f(inc_conc45.get('top_symbol_share'))
        chal_share45 = _f(chal_conc45.get('top_symbol_share'))

        inc_short = (((incumbent.get('shortlist_proxy') or {}).get('live_threshold')) or {})
        chal_short = (((challenger.get('shortlist_proxy') or {}).get('live_threshold')) or {})
        inc_scan = dict(incumbent.get('scan_shortlist_utility') or {})
        chal_scan = dict(challenger.get('scan_shortlist_utility') or {})
        deltas = {
            'adjusted_auc_delta': round(chal_auc - inc_auc, 6),
            'adjusted_brier_delta': round(chal_brier - inc_brier, 6),
            'count_ge_0_45_delta': int((chal_t45.get('count') or 0) - (inc_t45.get('count') or 0)),
            'precision_ge_0_45_delta': round((_f(chal_t45.get('precision'), 0.0) or 0.0) - (_f(inc_t45.get('precision'), 0.0) or 0.0), 6),
            'count_ge_0_50_delta': int((chal_t50.get('count') or 0) - (inc_t50.get('count') or 0)),
            'precision_ge_0_50_delta': round((_f(chal_t50.get('precision'), 0.0) or 0.0) - (_f(inc_t50.get('precision'), 0.0) or 0.0), 6),
            'top_1pct_quality_rate_delta': round(chal_top1 - inc_top1, 6),
            'top_symbol_share_ge_0_45_delta': None if chal_share45 is None or inc_share45 is None else round(chal_share45 - inc_share45, 6),
            'shortlist_quality_gap_delta': None if _f(chal_short.get('quality_gap')) is None or _f(inc_short.get('quality_gap')) is None else round((_f(chal_short.get('quality_gap'),0.0) or 0.0) - (_f(inc_short.get('quality_gap'),0.0) or 0.0), 6),
            'shortlist_visible_quality_rate_delta': None if _f(chal_short.get('visible_quality_rate')) is None or _f(inc_short.get('visible_quality_rate')) is None else round((_f(chal_short.get('visible_quality_rate'),0.0) or 0.0) - (_f(inc_short.get('visible_quality_rate'),0.0) or 0.0), 6),
            'shortlist_avg_visible_rows_per_scan_delta': None if _f(chal_short.get('avg_visible_rows_per_scan')) is None or _f(inc_short.get('avg_visible_rows_per_scan')) is None else round((_f(chal_short.get('avg_visible_rows_per_scan'),0.0) or 0.0) - (_f(inc_short.get('avg_visible_rows_per_scan'),0.0) or 0.0), 6),
            'scan_shortlist_utility_score_delta': None if _f(chal_scan.get('utility_score')) is None or _f(inc_scan.get('utility_score')) is None else round((_f(chal_scan.get('utility_score'), 0.0) or 0.0) - (_f(inc_scan.get('utility_score'), 0.0) or 0.0), 6),
            'scan_shortlist_mean_gap_delta': None if _f(chal_scan.get('mean_gap')) is None or _f(inc_scan.get('mean_gap')) is None else round((_f(chal_scan.get('mean_gap'), 0.0) or 0.0) - (_f(inc_scan.get('mean_gap'), 0.0) or 0.0), 6),
            'scan_shortlist_pairwise_win_rate_delta': None if _f(chal_scan.get('pairwise_win_rate')) is None or _f(inc_scan.get('pairwise_win_rate')) is None else round((_f(chal_scan.get('pairwise_win_rate'), 0.0) or 0.0) - (_f(inc_scan.get('pairwise_win_rate'), 0.0) or 0.0), 6),
            'scan_shortlist_top1_mean_quality_delta': None if _f(chal_scan.get('top1_mean_quality')) is None or _f(inc_scan.get('top1_mean_quality')) is None else round((_f(chal_scan.get('top1_mean_quality'), 0.0) or 0.0) - (_f(inc_scan.get('top1_mean_quality'), 0.0) or 0.0), 6),
            'scan_shortlist_top3_mean_quality_delta': None if _f(chal_scan.get('top3_mean_quality')) is None or _f(inc_scan.get('top3_mean_quality')) is None else round((_f(chal_scan.get('top3_mean_quality'), 0.0) or 0.0) - (_f(inc_scan.get('top3_mean_quality'), 0.0) or 0.0), 6),
            'scan_shortlist_avg_visible_rows_per_scan_delta': None if _f(chal_scan.get('avg_visible_rows_per_scan')) is None or _f(inc_scan.get('avg_visible_rows_per_scan')) is None else round((_f(chal_scan.get('avg_visible_rows_per_scan'), 0.0) or 0.0) - (_f(inc_scan.get('avg_visible_rows_per_scan'), 0.0) or 0.0), 6),
            'scan_shortlist_scan_capture_rate_delta': None if _f(chal_scan.get('scan_capture_rate')) is None or _f(inc_scan.get('scan_capture_rate')) is None else round((_f(chal_scan.get('scan_capture_rate'), 0.0) or 0.0) - (_f(inc_scan.get('scan_capture_rate'), 0.0) or 0.0), 6),
        }

        positive = 0
        negative = 0
        utility_positive = 0
        utility_negative = 0
        rationale = []
        if deltas.get('scan_shortlist_utility_score_delta') is not None and deltas['scan_shortlist_utility_score_delta'] > 0.015:
            positive += 1
            utility_positive += 1
            rationale.append(f"Shadow scan-level shortlist utility score improved by {deltas['scan_shortlist_utility_score_delta']:+.4f}.")
        elif deltas.get('scan_shortlist_utility_score_delta') is not None and deltas['scan_shortlist_utility_score_delta'] < -0.015:
            negative += 1
            utility_negative += 1
            rationale.append(f"Shadow scan-level shortlist utility score worsened by {abs(deltas['scan_shortlist_utility_score_delta']):.4f}.")
        if deltas.get('scan_shortlist_mean_gap_delta') is not None and deltas['scan_shortlist_mean_gap_delta'] > 0.015:
            positive += 1
            utility_positive += 1
            rationale.append(f"Shadow per-scan visible-vs-hidden quality gap improved by {deltas['scan_shortlist_mean_gap_delta']:.2%}.")
        elif deltas.get('scan_shortlist_mean_gap_delta') is not None and deltas['scan_shortlist_mean_gap_delta'] < -0.015:
            negative += 1
            utility_negative += 1
            rationale.append(f"Shadow per-scan visible-vs-hidden quality gap worsened by {abs(deltas['scan_shortlist_mean_gap_delta']):.2%}.")
        if deltas.get('scan_shortlist_pairwise_win_rate_delta') is not None and deltas['scan_shortlist_pairwise_win_rate_delta'] > 0.04:
            positive += 1
            utility_positive += 1
            rationale.append(f"Shadow per-scan win rate improved by {deltas['scan_shortlist_pairwise_win_rate_delta']:.2%}.")
        elif deltas.get('scan_shortlist_pairwise_win_rate_delta') is not None and deltas['scan_shortlist_pairwise_win_rate_delta'] < -0.04:
            negative += 1
            utility_negative += 1
            rationale.append(f"Shadow per-scan win rate worsened by {abs(deltas['scan_shortlist_pairwise_win_rate_delta']):.2%}.")
        if deltas.get('scan_shortlist_top1_mean_quality_delta') is not None and deltas['scan_shortlist_top1_mean_quality_delta'] > 0.03:
            positive += 1
            utility_positive += 1
            rationale.append(f"Shadow top-of-scan quality improved by {deltas['scan_shortlist_top1_mean_quality_delta']:.2%}.")
        elif deltas.get('scan_shortlist_top1_mean_quality_delta') is not None and deltas['scan_shortlist_top1_mean_quality_delta'] < -0.03:
            negative += 1
            utility_negative += 1
            rationale.append(f"Shadow top-of-scan quality worsened by {abs(deltas['scan_shortlist_top1_mean_quality_delta']):.2%}.")
        if deltas.get('scan_shortlist_top3_mean_quality_delta') is not None and deltas['scan_shortlist_top3_mean_quality_delta'] > 0.02:
            positive += 1
            utility_positive += 1
            rationale.append(f"Shadow top-3-per-scan quality improved by {deltas['scan_shortlist_top3_mean_quality_delta']:.2%}.")
        elif deltas.get('scan_shortlist_top3_mean_quality_delta') is not None and deltas['scan_shortlist_top3_mean_quality_delta'] < -0.02:
            negative += 1
            utility_negative += 1
            rationale.append(f"Shadow top-3-per-scan quality worsened by {abs(deltas['scan_shortlist_top3_mean_quality_delta']):.2%}.")
        if deltas.get('scan_shortlist_avg_visible_rows_per_scan_delta') is not None and deltas['scan_shortlist_avg_visible_rows_per_scan_delta'] > 1.0:
            negative += 1
            rationale.append(f"Shadow average visible rows per scan widened by {deltas['scan_shortlist_avg_visible_rows_per_scan_delta']:+.2f}, which weakens shortlist discipline.")
        elif deltas.get('scan_shortlist_avg_visible_rows_per_scan_delta') is not None and deltas['scan_shortlist_avg_visible_rows_per_scan_delta'] < -0.5:
            positive += 1
            rationale.append(f"Shadow reduced average visible rows per scan by {abs(deltas['scan_shortlist_avg_visible_rows_per_scan_delta']):.2f} while staying on the shared evaluation frame.")
        if deltas['precision_ge_0_45_delta'] > 0.03:
            positive += 1
            rationale.append(f"Shadow >=0.45 precision improved by {deltas['precision_ge_0_45_delta']:.2%}.")
        elif deltas['precision_ge_0_45_delta'] < -0.03:
            negative += 1
            rationale.append(f"Shadow >=0.45 precision worsened by {abs(deltas['precision_ge_0_45_delta']):.2%}.")
        if deltas['adjusted_auc_delta'] > 0.005:
            rationale.append(f"Shadow adjusted AUC improved by {deltas['adjusted_auc_delta']:.4f}.")
        elif deltas['adjusted_auc_delta'] < -0.005:
            rationale.append(f"Shadow adjusted AUC worsened by {abs(deltas['adjusted_auc_delta']):.4f}.")
        if deltas['adjusted_brier_delta'] < -0.002:
            rationale.append(f"Shadow adjusted Brier improved by {abs(deltas['adjusted_brier_delta']):.4f} lower-is-better.")
        elif deltas['adjusted_brier_delta'] > 0.002:
            rationale.append(f"Shadow adjusted Brier worsened by {deltas['adjusted_brier_delta']:.4f}.")

        concentration_ok = chal_share45 is None or inc_share45 is None or chal_share45 <= max(0.60, (inc_share45 or 0.0) + 0.10)
        if not concentration_ok:
            negative += 1
            rationale.append('Shadow >=0.45 concentration became materially worse than the incumbent.')

        if utility_positive >= 3 and utility_negative == 0 and concentration_ok and negative <= 1:
            verdict = 'shadow_candidate_preferred_offline'
            recommended_action = 'prepare_fresh_live_proof_window_for_shadow_candidate'
            recommended_action_reason = 'The shadow challenger improves scan-level shortlist usefulness on the shared offline frame and is ready for a later fresh live proof window, while remaining non-promoting now.'
        elif utility_negative >= 3 and utility_positive == 0:
            verdict = 'incumbent_still_preferred_offline'
            recommended_action = 'reject_current_shadow_candidate_and_retrain_again'
            recommended_action_reason = 'The incumbent still produces the stronger scan-level shortlist on the shared offline evaluation frame.'
        else:
            verdict = 'mixed_offline_result_manual_review_required'
            recommended_action = 'manual_review_before_any_live_candidate_selection'
            recommended_action_reason = 'The offline comparison remains mixed on shortlist utility, so it should not trigger a live candidate choice automatically.'

        utility_text = 'n/a' if deltas.get('scan_shortlist_utility_score_delta') is None else f"{deltas['scan_shortlist_utility_score_delta']:+.4f}"
        gap_text = 'n/a' if deltas.get('scan_shortlist_mean_gap_delta') is None else f"{deltas['scan_shortlist_mean_gap_delta']:+.2%}"
        top1_text = 'n/a' if deltas.get('scan_shortlist_top1_mean_quality_delta') is None else f"{deltas['scan_shortlist_top1_mean_quality_delta']:+.2%}"
        summary = (
            f"Offline challenger comparison verdict: {verdict}. "
            f"Scan-level utility delta {utility_text}, per-scan visible-vs-hidden gap delta {gap_text}, "
            f"top-of-scan quality delta {top1_text}, >=0.45 precision delta {deltas['precision_ge_0_45_delta']:+.2%}."
        )
        return {
            'verdict': verdict,
            'recommended_action': recommended_action,
            'recommended_action_reason': recommended_action_reason,
            'summary': summary,
            'deltas': deltas,
            'positive_signals': positive,
            'negative_signals': negative,
            'concentration_ok': concentration_ok,
            'rationale': rationale,
        }

    def _build_pack(self, *, result: dict, current_version: dict, audit_summary: dict) -> None:
        ensure_dir(self.pack_path.parent)
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('challenger_comparison_summary.json', json.dumps(result, indent=2, sort_keys=True))
            zf.writestr('incumbent_model_metrics.json', json.dumps(result.get('incumbent_model') or {}, indent=2, sort_keys=True))
            zf.writestr('challenger_model_metrics.json', json.dumps(result.get('challenger_model') or {}, indent=2, sort_keys=True))
            zf.writestr('comparison_deltas.json', json.dumps(result.get('comparison') or {}, indent=2, sort_keys=True))
            zf.writestr('fresh_retrain_audit_summary_excerpt.json', json.dumps({
                'generated_at_utc': audit_summary.get('generated_at_utc'),
                'headline': audit_summary.get('headline'),
                'verdict': audit_summary.get('verdict'),
                'summary': audit_summary.get('summary'),
                'shadow_training_spec': audit_summary.get('shadow_training_spec') or {},
                'source_context': audit_summary.get('source_context') or {},
                'artifact_paths': audit_summary.get('artifact_paths') or {},
            }, indent=2, sort_keys=True))
            zf.writestr('current_version_summary_excerpt.json', json.dumps({
                'app_version': current_version.get('app_version'),
                'generated_at_utc': current_version.get('generated_at_utc'),
                'decision_checkpoint': current_version.get('decision_checkpoint') or current_version.get('decision_rule_checkpoint') or {},
                'evidence': current_version.get('evidence') or {},
                'model_output_distribution': current_version.get('model_output_distribution') or {},
            }, indent=2, sort_keys=True))
            zf.writestr('decision_memo.md', self._decision_memo_markdown(result))

    def _decision_memo_markdown(self, result: dict) -> str:
        comp = result.get('comparison') or {}
        deltas = comp.get('deltas') or {}
        live = result.get('current_live_path') or {}
        eval_meta = result.get('shared_evaluation_frame') or {}
        return (
            "# Offline challenger comparison\n\n"
            f"- **Headline:** {result.get('headline')}\n"
            f"- **Verdict:** {result.get('verdict')}\n"
            f"- **Recommended action:** {result.get('recommended_action')}\n"
            f"- **Why:** {result.get('recommended_action_reason')}\n\n"
            "## Current live branch context\n"
            f"- **Checkpoint outcome:** {live.get('decision_checkpoint_outcome')}\n"
            f"- **Visible q-hit:** {live.get('visible_quality_hit_rate')}\n"
            f"- **Non-visible q-hit:** {live.get('non_visible_quality_hit_rate')}\n"
            f"- **Stage 1 mode:** {live.get('stage1_selection_mode')}\n"
            f"- **Threshold:** {live.get('live_raw_threshold')}\n\n"
            "## Shared evaluation frame\n"
            f"- **Rows test:** {eval_meta.get('rows_test')}\n"
            f"- **Symbols used:** {eval_meta.get('symbols_used_count')}\n"
            f"- **Lookback days:** {eval_meta.get('train_lookback_days')}\n"
            f"- **Sample every n bars:** {eval_meta.get('sample_every_n_bars')}\n\n"
            "## Key deltas\n"
            f"- **Scan-level shortlist utility delta:** {deltas.get('scan_shortlist_utility_score_delta')}\n"
            f"- **Per-scan visible-vs-hidden gap delta:** {deltas.get('scan_shortlist_mean_gap_delta')}\n"
            f"- **Per-scan win-rate delta:** {deltas.get('scan_shortlist_pairwise_win_rate_delta')}\n"
            f"- **Top-of-scan quality delta:** {deltas.get('scan_shortlist_top1_mean_quality_delta')}\n"
            f"- **Top-3-per-scan quality delta:** {deltas.get('scan_shortlist_top3_mean_quality_delta')}\n"
            f"- **Avg visible rows per scan delta:** {deltas.get('scan_shortlist_avg_visible_rows_per_scan_delta')}\n"
            f"- **Adjusted AUC delta:** {deltas.get('adjusted_auc_delta')}\n"
            f"- **Adjusted Brier delta:** {deltas.get('adjusted_brier_delta')}\n"
            f"- **>=0.45 precision delta:** {deltas.get('precision_ge_0_45_delta')}\n"
        )
