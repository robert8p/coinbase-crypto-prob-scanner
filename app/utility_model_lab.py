from __future__ import annotations

import gc
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import pandas as pd

from .coinbase_client import CoinbaseClient
from .config import AppConfig
from .demo_data import STABLES
from .features import FEATURE_COLUMNS, build_training_frame
from .modeling import ModelBundle, _purged_time_split, sanitize_feature_frame, train_pt2
from .persist import atomic_write_json, ensure_dir, read_json
from .universe import UniverseBuilder
from .version import APP_VERSION

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _select_training_symbols(config: AppConfig, ordered_symbols: List[str]) -> List[str]:
    max_symbols = max(1, int(config.train_max_symbols or 1))
    selected = list(ordered_symbols[:max_symbols])
    deduped: List[str] = []
    seen = set()
    for symbol in selected:
        if symbol not in seen:
            deduped.append(symbol)
            seen.add(symbol)
    for ctx in ['BTC-USD', 'ETH-USD']:
        if ctx in ordered_symbols and ctx not in deduped:
            if len(deduped) < max_symbols:
                deduped.append(ctx)
            else:
                deduped[-1] = ctx
    out: List[str] = []
    seen.clear()
    for symbol in deduped:
        if symbol not in seen:
            out.append(symbol)
            seen.add(symbol)
    return out


def _utility_target(df: pd.DataFrame, config: AppConfig) -> pd.Series:
    target_move_pct = float(config.target_move_pct or 0.02)
    mae_scale = max(0.01, abs(float(config.quality_max_mae or -0.02)))
    end_ret = df.get('end_ret', pd.Series(0.0, index=df.index)).astype(float).clip(-0.10, 0.20)
    mae = df.get('mae', pd.Series(0.0, index=df.index)).astype(float)
    y = df.get('y', pd.Series(0, index=df.index)).astype(float)
    raw_touch = df.get('y_raw_touch', pd.Series(0, index=df.index)).astype(float)
    clean_touch = df.get('touched_before_major_adverse', pd.Series(0, index=df.index)).astype(float)
    path_ugliness = df.get('path_ugliness', pd.Series(0.0, index=df.index)).astype(float).clip(0.0, 0.20)
    upside_term = (end_ret / max(target_move_pct, 1e-6)).clip(-1.5, 2.0)
    drawdown_penalty = (np.maximum(0.0, -mae) / mae_scale).clip(0.0, 4.0)
    utility = (
        1.00 * y
        + 0.15 * raw_touch
        + 0.10 * clean_touch
        + 0.35 * upside_term
        - 0.30 * drawdown_penalty
        - 0.20 * path_ugliness
    )
    return utility.astype(float)


@dataclass(slots=True)
class UtilityModelCandidate:
    pipeline: object
    metadata: dict

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = sanitize_feature_frame(df, FEATURE_COLUMNS)
        preds = self.pipeline.predict(X)
        return np.asarray(preds, dtype=float)

    def save(self, path: str) -> None:
        ensure_dir(Path(path).parent)
        joblib.dump({'pipeline': self.pipeline, 'metadata': self.metadata}, path)


def _scan_rank_metrics(df: pd.DataFrame, score_col: str, utility_col: str, *, visible_cap: int) -> dict:
    rows = []
    if df.empty:
        return {
            'scan_count': 0,
            'visible_rows': 0,
            'hidden_rows': 0,
            'visible_quality_hit_rate': None,
            'hidden_quality_hit_rate': None,
            'visible_hidden_quality_gap': None,
            'visible_true_utility_mean': None,
            'hidden_true_utility_mean': None,
            'visible_hidden_utility_gap': None,
            'scan_pairwise_win_rate': None,
            'scan_pairwise_comparable_scans': 0,
            'scan_top1_quality': None,
            'scan_top3_quality': None,
            'scan_shortlist_utility_score': None,
            'avg_visible_rows_per_scan': None,
        }
    overall_quality = float(df['y'].astype(float).mean()) if len(df) else 0.0
    for ts, grp in df.groupby('ts', sort=False):
        ranked = grp.sort_values(score_col, ascending=False).copy()
        visible = ranked.iloc[: min(visible_cap, len(ranked))].copy()
        hidden = ranked.iloc[min(visible_cap, len(ranked)) :].copy()
        rows.append({
            'scan_ts': ts,
            'visible_n': int(len(visible)),
            'hidden_n': int(len(hidden)),
            'visible_quality_rate': float(visible['y'].mean()) if len(visible) else None,
            'hidden_quality_rate': float(hidden['y'].mean()) if len(hidden) else None,
            'visible_utility_mean': float(visible[utility_col].mean()) if len(visible) else None,
            'hidden_utility_mean': float(hidden[utility_col].mean()) if len(hidden) else None,
            'top1_quality': float(visible.iloc[:1]['y'].mean()) if len(visible) else None,
            'top3_quality': float(visible.iloc[: min(3, len(visible))]['y'].mean()) if len(visible) else None,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            'scan_count': 0,
            'visible_rows': 0,
            'hidden_rows': 0,
            'visible_quality_hit_rate': None,
            'hidden_quality_hit_rate': None,
            'visible_hidden_quality_gap': None,
            'visible_true_utility_mean': None,
            'hidden_true_utility_mean': None,
            'visible_hidden_utility_gap': None,
            'scan_pairwise_win_rate': None,
            'scan_pairwise_comparable_scans': 0,
            'scan_top1_quality': None,
            'scan_top3_quality': None,
            'scan_shortlist_utility_score': None,
            'avg_visible_rows_per_scan': None,
        }
    frame['utility_gap'] = frame['visible_utility_mean'] - frame['hidden_utility_mean']
    frame['quality_gap'] = frame['visible_quality_rate'] - frame['hidden_quality_rate']
    comparable = frame[frame['hidden_n'] > 0].copy()
    pairwise_wins = 0.0
    for _, row in comparable.iterrows():
        gap = float(row['utility_gap']) if pd.notna(row['utility_gap']) else 0.0
        if gap > 0:
            pairwise_wins += 1.0
        elif gap == 0:
            pairwise_wins += 0.5
    pairwise_rate = (pairwise_wins / len(comparable)) if len(comparable) else None
    mean_utility_gap = float(frame['utility_gap'].mean()) if frame['utility_gap'].notna().any() else None
    mean_quality_gap = float(frame['quality_gap'].mean()) if frame['quality_gap'].notna().any() else None
    top1 = float(frame['top1_quality'].mean()) if frame['top1_quality'].notna().any() else None
    top3 = float(frame['top3_quality'].mean()) if frame['top3_quality'].notna().any() else None
    utility_score = None
    if mean_utility_gap is not None:
        utility_score = mean_utility_gap
        if pairwise_rate is not None:
            utility_score += 0.25 * (pairwise_rate - 0.5)
        if top1 is not None:
            utility_score += 0.10 * (top1 - overall_quality)
        if top3 is not None:
            utility_score += 0.05 * (top3 - overall_quality)
    return {
        'scan_count': int(frame['scan_ts'].nunique()),
        'visible_rows': int(frame['visible_n'].sum()),
        'hidden_rows': int(frame['hidden_n'].sum()),
        'visible_quality_hit_rate': float(df.sort_values([score_col], ascending=False).iloc[:0]['y'].mean()) if False else (float(frame['visible_quality_rate'].mean()) if frame['visible_quality_rate'].notna().any() else None),
        'hidden_quality_hit_rate': float(frame['hidden_quality_rate'].mean()) if frame['hidden_quality_rate'].notna().any() else None,
        'visible_hidden_quality_gap': round(mean_quality_gap, 6) if mean_quality_gap is not None else None,
        'visible_true_utility_mean': float(frame['visible_utility_mean'].mean()) if frame['visible_utility_mean'].notna().any() else None,
        'hidden_true_utility_mean': float(frame['hidden_utility_mean'].mean()) if frame['hidden_utility_mean'].notna().any() else None,
        'visible_hidden_utility_gap': round(mean_utility_gap, 6) if mean_utility_gap is not None else None,
        'scan_pairwise_win_rate': round(pairwise_rate, 6) if pairwise_rate is not None else None,
        'scan_pairwise_comparable_scans': int(len(comparable)),
        'scan_top1_quality': round(top1, 6) if top1 is not None else None,
        'scan_top3_quality': round(top3, 6) if top3 is not None else None,
        'scan_shortlist_utility_score': round(utility_score, 6) if utility_score is not None else None,
        'avg_visible_rows_per_scan': round(float(frame['visible_n'].mean()), 6) if len(frame) else None,
    }


class UtilityModelLabService:
    def __init__(self, config: AppConfig, client: CoinbaseClient):
        self.config = config
        self.client = client
        self.root_dir = ensure_dir(Path(config.model_dir) / 'utility_model_lab')
        self.summary_path = self.root_dir / 'latest_utility_model_lab_summary.json'
        self.pack_path = self.root_dir / 'latest_utility_model_lab_pack.zip'
        self.model_path = self.root_dir / 'latest_utility_model_candidate.joblib'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def run(self, *, max_symbols: int | None = None, visible_cap: int | None = None) -> dict:
        products = self.client.list_products()
        currencies = self.client.list_currencies()
        volume_map = self.client.get_volume_summary()
        universe = UniverseBuilder(self.config).build(products, currencies, volume_map)
        ordered_symbols = [p['id'] for p in universe.eligible]
        orig_max = self.config.train_max_symbols
        try:
            if max_symbols is not None:
                self.config.train_max_symbols = max(1, int(max_symbols))
            symbols = _select_training_symbols(self.config, ordered_symbols)
        finally:
            self.config.train_max_symbols = orig_max
        lookback_bars = max(self.config.stage2_lookback_5m_bars, int((self.config.train_lookback_days * 24 * 60) / 5))
        btc_df = self.client.get_candles('BTC-USD', lookback_bars) if 'BTC-USD' in symbols else None
        eth_df = self.client.get_candles('ETH-USD', lookback_bars) if 'ETH-USD' in symbols else None
        frames: list[pd.DataFrame] = []
        skipped: list[dict] = []
        for symbol in symbols:
            try:
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
                    sample_every=self.config.train_sample_every_n_bars,
                    horizon_bars=self.config.candles_per_horizon,
                    target_move_pct=self.config.target_move_pct,
                    warmup_bars=self.config.train_feature_warmup_5m_bars,
                    quality_max_mae=self.config.quality_max_mae,
                    quality_min_end_ret=self.config.quality_min_end_ret,
                )
                if frame.empty:
                    skipped.append({'symbol': symbol, 'reason': 'empty_training_frame'})
                    continue
                frames.append(_downcast_training_frame(frame))
                del df, frame
                gc.collect()
            except Exception as exc:
                skipped.append({'symbol': symbol, 'reason': str(exc)})
        training_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        del frames
        gc.collect()
        if training_df.empty:
            raise RuntimeError('utility model lab produced no rows')
        training_df['utility_target'] = _utility_target(training_df, self.config)
        df_train, df_val, df_test, embargo_dropped = _purged_time_split(training_df)
        if min(len(df_train), len(df_val), len(df_test)) < 80:
            raise RuntimeError('utility model lab split too small')

        cfg_dict = {
            'app_version': APP_VERSION,
            'target_move_pct': self.config.target_move_pct,
            'target_horizon_minutes': self.config.target_horizon_minutes,
            'quality_max_mae': self.config.quality_max_mae,
            'quality_min_end_ret': self.config.quality_min_end_ret,
            'live_raw_threshold': self.config.live_raw_threshold,
            'utility_selection_engine_label': 'utility_model_challenger_lab',
        }
        incumbent_bundle = train_pt2(training_df, cfg_dict=cfg_dict)
        incumbent_test_pred = incumbent_bundle.predict_proba(df_test)

        if lgb is None:
            raise RuntimeError('lightgbm not available for utility model challenger')
        reg = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=0.20,
            random_state=42,
            n_jobs=-1,
        )
        X_train = sanitize_feature_frame(df_train, FEATURE_COLUMNS)
        X_val = sanitize_feature_frame(df_val, FEATURE_COLUMNS)
        X_test = sanitize_feature_frame(df_test, FEATURE_COLUMNS)
        reg.fit(
            X_train,
            df_train['utility_target'].astype(float),
            eval_set=[(X_val, df_val['utility_target'].astype(float))],
            eval_metric='l2',
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        utility_pred = np.asarray(reg.predict(X_test), dtype=float)
        candidate = UtilityModelCandidate(
            pipeline=reg,
            metadata={
                'feature_cols': FEATURE_COLUMNS,
                'trained_at_utc': _utc_now_iso(),
                'rows_train': int(len(df_train)),
                'rows_validation': int(len(df_val)),
                'rows_test': int(len(df_test)),
                'utility_target': 'quality + end_ret reward - mae penalty - path ugliness penalty',
                'selection_method': 'topk_by_model_score_same_scan',
                'app_version': APP_VERSION,
            },
        )
        candidate.save(str(self.model_path))

        visible_cap = max(1, int(visible_cap or self.config.utility_shortlist_target_max_names or 8))
        eval_df = df_test[['ts', 'symbol', 'y', 'utility_target']].copy()
        eval_df['incumbent_score'] = incumbent_test_pred
        eval_df['utility_challenger_score'] = utility_pred

        incumbent_metrics = _scan_rank_metrics(eval_df, 'incumbent_score', 'utility_target', visible_cap=visible_cap)
        challenger_metrics = _scan_rank_metrics(eval_df, 'utility_challenger_score', 'utility_target', visible_cap=visible_cap)
        deltas = {
            'scan_shortlist_utility_score_delta': round((challenger_metrics.get('scan_shortlist_utility_score') or 0.0) - (incumbent_metrics.get('scan_shortlist_utility_score') or 0.0), 6),
            'visible_hidden_utility_gap_delta': round((challenger_metrics.get('visible_hidden_utility_gap') or 0.0) - (incumbent_metrics.get('visible_hidden_utility_gap') or 0.0), 6),
            'visible_hidden_quality_gap_delta': round((challenger_metrics.get('visible_hidden_quality_gap') or 0.0) - (incumbent_metrics.get('visible_hidden_quality_gap') or 0.0), 6),
            'scan_pairwise_win_rate_delta': round((challenger_metrics.get('scan_pairwise_win_rate') or 0.0) - (incumbent_metrics.get('scan_pairwise_win_rate') or 0.0), 6),
            'scan_top1_quality_delta': round((challenger_metrics.get('scan_top1_quality') or 0.0) - (incumbent_metrics.get('scan_top1_quality') or 0.0), 6),
        }
        verdict = 'utility_model_candidate_not_supported_offline'
        headline = 'Utility-trained challenger does not yet clearly beat the incumbent shortlist model offline'
        recommended_action = 'keep_current_live_scoring_but_review_candidate_outputs'
        summary = 'The first utility-aligned challenger model did not yet clearly beat the incumbent event model on scan-level shortlist utility.'
        if (
            deltas['scan_shortlist_utility_score_delta'] >= 0.03
            and deltas['visible_hidden_utility_gap_delta'] >= 0.03
            and deltas['visible_hidden_quality_gap_delta'] >= 0.01
            and deltas['scan_pairwise_win_rate_delta'] >= 0.05
        ):
            verdict = 'utility_model_candidate_supported_offline'
            headline = 'Utility-trained challenger beats the incumbent shortlist model offline'
            recommended_action = 'prepare_isolated_live_proof_for_the_utility_model_candidate'
            summary = 'The utility-aligned challenger model improved shortlist utility, visible-vs-hidden separation, and scan win rate strongly enough to justify the next offline-to-live proof step.'
        payload = {
            'generated_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'headline': headline,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'summary': summary,
            'lab_inputs': {
                'max_symbols': int(max_symbols or self.config.train_max_symbols),
                'visible_cap': visible_cap,
                'lookback_bars': lookback_bars,
            },
            'training_frame': {
                'rows_all': int(len(training_df)),
                'rows_train': int(len(df_train)),
                'rows_validation': int(len(df_val)),
                'rows_test': int(len(df_test)),
                'embargo_dropped': int(embargo_dropped),
                'symbols_used': sorted(training_df['symbol'].astype(str).unique().tolist()),
                'skipped_symbols': skipped,
            },
            'incumbent_shortlist_metrics': incumbent_metrics,
            'utility_model_shortlist_metrics': challenger_metrics,
            'best_vs_incumbent_deltas': deltas,
            'candidate_model_path': str(self.model_path),
            'candidate_model_metadata': candidate.metadata,
            'decision_memo_markdown': (
                '# Utility model challenger lab\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n'
                f'- **Summary:** {summary}\n\n'
                '## Why this tranche exists\n'
                '- The app now applies utility-aware shortlist logic, but its model is still trained as an event classifier.\n'
                '- This lab tests the first real modeling step of the redesign: train directly toward utility and compare on scan-level shortlist outcomes.\n'
                '- The challenger is judged on the same replay-style grouped test frame as the incumbent.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, eval_df, training_df)
        return payload

    def _build_pack(self, summary: dict, eval_df: pd.DataFrame, training_df: pd.DataFrame) -> None:
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('utility_model_lab_summary.json', json.dumps(summary, indent=2, default=str))
            zf.writestr('utility_model_lab_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('utility_model_candidate_metadata.json', json.dumps(summary.get('candidate_model_metadata') or {}, indent=2, default=str))
            zf.writestr('utility_model_lab_eval_rows.csv', eval_df.to_csv(index=False).encode('utf-8'))
            preview_cols = [c for c in ['ts', 'symbol', 'y', 'y_raw_touch', 'mae', 'end_ret', 'utility_target'] if c in training_df.columns]
            zf.writestr('utility_model_training_frame_preview.csv', training_df[preview_cols].head(5000).to_csv(index=False).encode('utf-8'))
