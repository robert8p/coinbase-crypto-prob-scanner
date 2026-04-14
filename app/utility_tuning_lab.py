from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any

import pandas as pd

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .replay import HistoricalReplayService
from .review_runs import ReviewPackService
from .utility_shortlist import optimize_visible_shortlist
from .version import APP_VERSION


def _config_to_dict(config: object) -> dict:
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    data = getattr(config, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    slots = getattr(type(config), "__slots__", None) or getattr(config, "__slots__", None) or []
    if isinstance(slots, str):
        slots = [slots]
    slot_payload = {name: getattr(config, name) for name in slots if isinstance(name, str) and hasattr(config, name)}
    if slot_payload:
        return slot_payload
    payload = {}
    for name in dir(config):
        if name.startswith('_'):
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        payload[name] = value
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ''):
            return default
        return float(value)
    except Exception:
        return default


class UtilityTuningLabService:
    def __init__(self, config: AppConfig, replay: HistoricalReplayService, review_packs: ReviewPackService):
        self.config = config
        self.replay = replay
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / 'utility_tuning_lab')
        self.summary_path = self.root_dir / 'latest_utility_tuning_lab_summary.json'
        self.pack_path = self.root_dir / 'latest_utility_tuning_lab_pack.zip'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def run(self, *, hours: int = 168, step_minutes: int = 120, max_scans: int = 84, max_symbols: int = 100) -> dict:
        try:
            current_version = self.review_packs.get_current_version_summary() or {}
        except FileNotFoundError:
            current_version = {}
        live_threshold = self._current_live_threshold(current_version)
        replay_result = self.replay.run(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            pipeline_mode='raw_threshold',
            raw_threshold=live_threshold,
        )
        replay_summary = dict(replay_result.get('summary') or {})
        replay_rows = list(replay_summary.get('replay_rows') or [])
        scans = self._build_scan_groups(replay_rows)

        candidate_defs = self._candidate_definitions()
        all_rows: dict[str, list[dict]] = {c['label']: [] for c in candidate_defs}
        scan_rows: list[dict] = []

        for as_of, scan_rows_all in scans.items():
            candidate_pool = self._candidate_pool(scan_rows_all)
            regime_state = str((scan_rows_all[0] if scan_rows_all else {}).get('market_regime_state') or 'green').lower()
            effective_max = self._effective_max_for_regime(regime_state)
            scan_record = {
                'as_of_utc': as_of,
                'market_regime_state': regime_state,
                'candidate_pool_count': len(candidate_pool),
                'effective_max': effective_max,
            }
            for cand in candidate_defs:
                cfg = self._proxy_config(cand['params'])
                result = optimize_visible_shortlist(candidate_pool, effective_max=effective_max, config=cfg, tracked_priority_symbols=[])
                eval_out = self._evaluate_candidate(as_of=as_of, visible_rows=result.visible_rows, pool_rows=candidate_pool)
                tagged_rows = [dict(r, tuning_candidate_label=cand['label']) for r in (eval_out['visible_rows'] + eval_out['hidden_rows'])]
                all_rows[cand['label']].extend(tagged_rows)
                scan_record[f"{cand['label']}_visible_count"] = len(result.visible_rows)
                scan_record[f"{cand['label']}_gap"] = eval_out['visible_hidden_gap']
                scan_record[f"{cand['label']}_top1"] = result.visible_rows[0].get('symbol') if result.visible_rows else None
            scan_rows.append(scan_record)

        candidate_summaries = []
        for cand in candidate_defs:
            summary = self._engine_summary(cand['label'], all_rows[cand['label']])
            summary['params'] = dict(cand['params'])
            candidate_summaries.append(summary)
        candidate_summaries.sort(
            key=lambda item: (
                float(item.get('scan_shortlist_utility_score') or -9.0),
                float(item.get('scan_shortlist_mean_gap') or -9.0),
                float(item.get('scan_shortlist_pairwise_win_rate') or 0.0),
                float(item.get('scan_shortlist_top1_visible_quality') or 0.0),
                -float(item.get('scan_shortlist_avg_visible_rows_per_scan') or 999.0),
            ),
            reverse=True,
        )
        baseline = next((c for c in candidate_summaries if c.get('engine_label') == 'baseline_current_utility_v1'), candidate_summaries[0] if candidate_summaries else None)
        best = candidate_summaries[0] if candidate_summaries else None
        summary = self._build_summary(
            current_version=current_version,
            replay_summary=replay_summary,
            candidate_summaries=candidate_summaries,
            baseline=baseline,
            best=best,
            scan_rows=scan_rows,
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
        )
        atomic_write_json(self.summary_path, summary)
        self._build_pack(summary, replay_summary, candidate_summaries, all_rows, scan_rows)
        return summary

    def _current_live_threshold(self, current_version: dict) -> float:
        checkpoint = current_version.get('decision_checkpoint') or current_version.get('decision_rule_checkpoint') or {}
        return _f(
            checkpoint.get('live_raw_threshold')
            or checkpoint.get('effective_live_raw_threshold')
            or (current_version.get('decision_branch_automation') or {}).get('effective_live_raw_threshold')
            or self.config.live_raw_threshold,
            0.35,
        ) or 0.35

    def _candidate_definitions(self) -> list[dict]:
        b = {
            'utility_expected_edge_weight': float(self.config.utility_expected_edge_weight),
            'utility_confidence_weight': float(self.config.utility_confidence_weight),
            'utility_probability_weight': float(self.config.utility_probability_weight),
            'utility_shortlist_target_max_names': int(self.config.utility_shortlist_target_max_names),
            'utility_shortlist_score_floor': float(self.config.utility_shortlist_score_floor),
            'utility_shortlist_score_dropoff': float(self.config.utility_shortlist_score_dropoff),
            'utility_confidence_floor': float(self.config.utility_confidence_floor),
            'utility_tier3_max_frac': float(self.config.utility_tier3_max_frac),
            'utility_pinned_visible_cap': int(self.config.utility_pinned_visible_cap),
            'utility_tracked_symbol_floor_relaxation': float(self.config.utility_tracked_symbol_floor_relaxation),
            'utility_tracked_symbol_confidence_relaxation': float(self.config.utility_tracked_symbol_confidence_relaxation),
        }
        return [
            {'label': 'baseline_current_utility_v1', 'params': b},
            {'label': 'edge_heavy_compact_v1', 'params': {**b, 'utility_expected_edge_weight': 0.55, 'utility_confidence_weight': 0.30, 'utility_probability_weight': 0.15, 'utility_shortlist_target_max_names': max(4, b['utility_shortlist_target_max_names'] - 2), 'utility_shortlist_score_floor': min(0.95, b['utility_shortlist_score_floor'] + 0.02), 'utility_confidence_floor': min(0.95, b['utility_confidence_floor'] + 0.02), 'utility_shortlist_score_dropoff': max(0.08, b['utility_shortlist_score_dropoff'] - 0.02), 'utility_tier3_max_frac': min(b['utility_tier3_max_frac'], 0.20)}},
            {'label': 'confidence_heavy_compact_v1', 'params': {**b, 'utility_expected_edge_weight': 0.35, 'utility_confidence_weight': 0.45, 'utility_probability_weight': 0.20, 'utility_shortlist_target_max_names': max(4, b['utility_shortlist_target_max_names'] - 2), 'utility_shortlist_score_floor': min(0.95, b['utility_shortlist_score_floor'] + 0.02), 'utility_confidence_floor': min(0.95, b['utility_confidence_floor'] + 0.04), 'utility_shortlist_score_dropoff': max(0.08, b['utility_shortlist_score_dropoff'] - 0.03), 'utility_tier3_max_frac': min(b['utility_tier3_max_frac'], 0.15)}},
            {'label': 'balanced_compact_v1', 'params': {**b, 'utility_expected_edge_weight': 0.45, 'utility_confidence_weight': 0.30, 'utility_probability_weight': 0.25, 'utility_shortlist_target_max_names': max(5, b['utility_shortlist_target_max_names'] - 1), 'utility_shortlist_score_floor': min(0.95, b['utility_shortlist_score_floor'] + 0.01), 'utility_confidence_floor': min(0.95, b['utility_confidence_floor'] + 0.01), 'utility_shortlist_score_dropoff': max(0.08, b['utility_shortlist_score_dropoff'] - 0.02)}},
            {'label': 'balanced_wider_v1', 'params': {**b, 'utility_expected_edge_weight': 0.45, 'utility_confidence_weight': 0.25, 'utility_probability_weight': 0.30, 'utility_shortlist_target_max_names': min(12, b['utility_shortlist_target_max_names'] + 2), 'utility_shortlist_score_floor': max(0.30, b['utility_shortlist_score_floor'] - 0.02), 'utility_confidence_floor': max(0.20, b['utility_confidence_floor'] - 0.02), 'utility_shortlist_score_dropoff': min(0.30, b['utility_shortlist_score_dropoff'] + 0.03), 'utility_tier3_max_frac': max(b['utility_tier3_max_frac'], 0.25)}},
            {'label': 'probability_heavy_v1', 'params': {**b, 'utility_expected_edge_weight': 0.30, 'utility_confidence_weight': 0.20, 'utility_probability_weight': 0.50, 'utility_shortlist_target_max_names': b['utility_shortlist_target_max_names'], 'utility_shortlist_score_floor': b['utility_shortlist_score_floor'], 'utility_confidence_floor': b['utility_confidence_floor'], 'utility_shortlist_score_dropoff': b['utility_shortlist_score_dropoff']}},
            {'label': 'ultra_tight_quality_v1', 'params': {**b, 'utility_expected_edge_weight': 0.50, 'utility_confidence_weight': 0.35, 'utility_probability_weight': 0.15, 'utility_shortlist_target_max_names': max(4, b['utility_shortlist_target_max_names'] - 3), 'utility_shortlist_score_floor': min(0.97, b['utility_shortlist_score_floor'] + 0.04), 'utility_confidence_floor': min(0.97, b['utility_confidence_floor'] + 0.05), 'utility_shortlist_score_dropoff': max(0.08, b['utility_shortlist_score_dropoff'] - 0.05), 'utility_tier3_max_frac': min(b['utility_tier3_max_frac'], 0.10)}},
        ]

    def _proxy_config(self, params: dict) -> SimpleNamespace:
        payload = _config_to_dict(self.config)
        payload.update(params)
        return SimpleNamespace(**payload)

    def _build_scan_groups(self, replay_rows: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for row in replay_rows:
            as_of = str(row.get('as_of_utc') or '')
            if not as_of:
                continue
            out.setdefault(as_of, []).append(dict(row))
        return out

    def _candidate_pool(self, rows: list[dict]) -> list[dict]:
        blocked_reasons = {'threshold', 'regime', 'cooldown'}
        pool = []
        for row in rows:
            reason = str(row.get('suppression_reason') or '')
            if reason in blocked_reasons:
                continue
            candidate = dict(row)
            candidate['row_type'] = 'candidate_pool'
            pool.append(candidate)
        pool.sort(key=lambda r: (
            {'action_ready': 3, 'selective': 2, 'watchlist': 1}.get(str(r.get('actionability_tier') or 'watchlist'), 1),
            float(r.get('prob_2_rank', r.get('prob_2') or 0.0) or 0.0),
            float(r.get('opportunity_score', 0.0) or 0.0),
            float(r.get('prob_2', 0.0) or 0.0),
            -float(r.get('risk', 0.0) or 0.0),
            str(r.get('symbol') or ''),
        ), reverse=True)
        return pool

    def _effective_max_for_regime(self, regime_state: str) -> int:
        effective_max = int(self.config.stage2_max_names)
        if regime_state == 'amber':
            effective_max = min(effective_max, max(6, int(self.config.stage2_max_names * 0.65)))
        elif regime_state == 'red':
            effective_max = min(effective_max, max(2, int(self.config.stage2_max_names * 0.20)))
        return max(0, effective_max)

    def _evaluate_candidate(self, *, as_of: str, visible_rows: list[dict], pool_rows: list[dict]) -> dict:
        visible_symbols = {str(r.get('symbol') or '') for r in visible_rows}
        visible = []
        hidden = []
        for row in pool_rows:
            item = dict(row)
            item['as_of_utc'] = as_of
            if str(item.get('symbol') or '') in visible_symbols:
                item['row_type'] = 'visible'
                visible.append(item)
            else:
                item['row_type'] = 'hidden'
                hidden.append(item)
        visible_summary = self.review_packs._bucket_summary(visible) if visible else {}
        hidden_summary = self.review_packs._bucket_summary(hidden) if hidden else {}
        return {
            'visible_rows': visible,
            'hidden_rows': hidden,
            'visible_quality_hit_rate': visible_summary.get('quality_hit_rate'),
            'hidden_quality_hit_rate': hidden_summary.get('quality_hit_rate'),
            'visible_hidden_gap': None if _f(visible_summary.get('quality_hit_rate')) is None or _f(hidden_summary.get('quality_hit_rate')) is None else round((_f(visible_summary.get('quality_hit_rate'), 0.0) or 0.0) - (_f(hidden_summary.get('quality_hit_rate'), 0.0) or 0.0), 6),
        }

    def _scan_shortlist_utility(self, rows: list[dict]) -> dict:
        if not rows:
            return {
                'scan_shortlist_scans': 0,
                'scan_shortlist_scans_with_visible': 0,
                'scan_shortlist_avg_visible_rows_per_scan': None,
                'scan_shortlist_visible_quality_rate_mean': None,
                'scan_shortlist_hidden_quality_rate_mean': None,
                'scan_shortlist_mean_gap': None,
                'scan_shortlist_pairwise_win_rate': None,
                'scan_shortlist_pairwise_comparable_scans': 0,
                'scan_shortlist_top1_visible_quality': None,
                'scan_shortlist_top3_visible_quality': None,
                'scan_shortlist_overwide_penalty': None,
                'scan_shortlist_utility_score': None,
            }
        frame = pd.DataFrame([
            {
                'scan_id': str(r.get('as_of_utc') or ''),
                'row_type': str(r.get('row_type') or ''),
                'y': int(r.get('quality_touched') or 0),
            }
            for r in rows
        ])
        if frame.empty:
            return {
                'scan_shortlist_scans': 0,
                'scan_shortlist_scans_with_visible': 0,
                'scan_shortlist_avg_visible_rows_per_scan': None,
                'scan_shortlist_visible_quality_rate_mean': None,
                'scan_shortlist_hidden_quality_rate_mean': None,
                'scan_shortlist_mean_gap': None,
                'scan_shortlist_pairwise_win_rate': None,
                'scan_shortlist_pairwise_comparable_scans': 0,
                'scan_shortlist_top1_visible_quality': None,
                'scan_shortlist_top3_visible_quality': None,
                'scan_shortlist_overwide_penalty': None,
                'scan_shortlist_utility_score': None,
            }
        base_event_rate = float(frame['y'].mean()) if len(frame) else 0.0
        scan_count = int(frame['scan_id'].nunique())
        scans_with_visible = 0
        pairwise_wins = 0.0
        pairwise_comparable = 0
        visible_counts = []
        visible_rates = []
        hidden_rates = []
        gaps = []
        top1_visible = []
        top3_visible = []
        for _, scan in frame.groupby('scan_id', sort=False):
            visible = scan[scan['row_type'] == 'visible']
            hidden = scan[scan['row_type'] != 'visible']
            visible_counts.append(int(len(visible)))
            if not visible.empty:
                scans_with_visible += 1
                visible_rate = float(visible['y'].mean())
                visible_rates.append(visible_rate)
                top1_visible.append(float(visible.iloc[:1]['y'].mean()))
                top3_visible.append(float(visible.iloc[: min(3, len(visible))]['y'].mean()))
                if not hidden.empty:
                    hidden_rate = float(hidden['y'].mean())
                    hidden_rates.append(hidden_rate)
                    gap = visible_rate - hidden_rate
                    gaps.append(gap)
                    pairwise_comparable += 1
                    if gap > 0:
                        pairwise_wins += 1.0
                    elif gap == 0:
                        pairwise_wins += 0.5
        avg_visible_rows = float(pd.Series(visible_counts).mean()) if visible_counts else None
        visible_quality_mean = float(pd.Series(visible_rates).mean()) if visible_rates else None
        hidden_quality_mean = float(pd.Series(hidden_rates).mean()) if hidden_rates else None
        mean_gap = float(pd.Series(gaps).mean()) if gaps else None
        pairwise_win_rate = float(pairwise_wins) / float(pairwise_comparable) if pairwise_comparable else None
        top1_mean = float(pd.Series(top1_visible).mean()) if top1_visible else None
        top3_mean = float(pd.Series(top3_visible).mean()) if top3_visible else None
        overwide_penalty = max(0.0, (avg_visible_rows or 0.0) - 5.0) / 5.0 if avg_visible_rows is not None else None
        utility_score = None
        if mean_gap is not None:
            utility_score = (
                float(mean_gap)
                + 0.25 * (((pairwise_win_rate if pairwise_win_rate is not None else 0.5) - 0.5))
                + 0.10 * (((top1_mean if top1_mean is not None else base_event_rate) - base_event_rate))
                + 0.05 * (((top3_mean if top3_mean is not None else base_event_rate) - base_event_rate))
                - 0.02 * (overwide_penalty or 0.0)
            )
        return {
            'scan_shortlist_scans': scan_count,
            'scan_shortlist_scans_with_visible': scans_with_visible,
            'scan_shortlist_avg_visible_rows_per_scan': round(avg_visible_rows, 6) if avg_visible_rows is not None else None,
            'scan_shortlist_visible_quality_rate_mean': round(visible_quality_mean, 6) if visible_quality_mean is not None else None,
            'scan_shortlist_hidden_quality_rate_mean': round(hidden_quality_mean, 6) if hidden_quality_mean is not None else None,
            'scan_shortlist_mean_gap': round(mean_gap, 6) if mean_gap is not None else None,
            'scan_shortlist_pairwise_win_rate': round(pairwise_win_rate, 6) if pairwise_win_rate is not None else None,
            'scan_shortlist_pairwise_comparable_scans': int(pairwise_comparable),
            'scan_shortlist_top1_visible_quality': round(top1_mean, 6) if top1_mean is not None else None,
            'scan_shortlist_top3_visible_quality': round(top3_mean, 6) if top3_mean is not None else None,
            'scan_shortlist_overwide_penalty': round(overwide_penalty, 6) if overwide_penalty is not None else None,
            'scan_shortlist_utility_score': round(utility_score, 6) if utility_score is not None else None,
        }

    def _engine_summary(self, label: str, rows: list[dict]) -> dict:
        utility = self._scan_shortlist_utility(rows)
        visible = [r for r in rows if str(r.get('row_type') or '') == 'visible']
        hidden = [r for r in rows if str(r.get('row_type') or '') != 'visible']
        visible_summary = self.review_packs._bucket_summary(visible) if visible else {}
        hidden_summary = self.review_packs._bucket_summary(hidden) if hidden else {}
        return {
            'engine_label': label,
            'visible_row_count': len(visible),
            'hidden_row_count': len(hidden),
            'visible_quality_hit_rate': visible_summary.get('quality_hit_rate'),
            'hidden_quality_hit_rate': hidden_summary.get('quality_hit_rate'),
            'visible_avg_end_ret': visible_summary.get('avg_end_ret'),
            'hidden_avg_end_ret': hidden_summary.get('avg_end_ret'),
            'visible_avg_mae': visible_summary.get('avg_mae'),
            'hidden_avg_mae': hidden_summary.get('avg_mae'),
            **utility,
        }

    def _build_summary(self, *, current_version: dict, replay_summary: dict, candidate_summaries: list[dict], baseline: dict | None, best: dict | None, scan_rows: list[dict], hours: int, step_minutes: int, max_scans: int, max_symbols: int) -> dict:
        deltas = {}
        headline = 'Utility tuning lab produced a mixed result'
        verdict = 'mixed_offline_result'
        recommended_action = 'review_candidate_grid_before_changing_live_utility_settings'
        recommended_params = None
        recommended_env_overrides = None
        if baseline and best:
            metric_names = [
                'scan_shortlist_utility_score', 'scan_shortlist_mean_gap', 'scan_shortlist_pairwise_win_rate',
                'scan_shortlist_top1_visible_quality', 'scan_shortlist_avg_visible_rows_per_scan', 'visible_quality_hit_rate'
            ]
            for name in metric_names:
                best_v = _f(best.get(name))
                base_v = _f(baseline.get(name))
                deltas[f'{name}_delta'] = None if best_v is None or base_v is None else round(best_v - base_v, 6)
            if best.get('engine_label') == baseline.get('engine_label'):
                headline = 'Current utility settings remain the best offline candidate in this tuning grid'
                verdict = 'current_utility_settings_hold_offline'
                recommended_action = 'keep_current_live_utility_settings_and_expand_the_grid_if_needed'
            else:
                clear_support = (
                    (deltas.get('scan_shortlist_utility_score_delta') or 0.0) >= 0.015 and
                    (deltas.get('scan_shortlist_mean_gap_delta') or 0.0) >= 0.010 and
                    (deltas.get('scan_shortlist_pairwise_win_rate_delta') or 0.0) >= 0.03 and
                    ((deltas.get('scan_shortlist_avg_visible_rows_per_scan_delta') or 0.0) <= 1.25)
                )
                if clear_support:
                    headline = f"{best.get('engine_label')} beats the current utility settings offline"
                    verdict = 'utility_tuning_candidate_supported_offline'
                    recommended_action = 'prepare_a_controlled_live_update_of_the_utility_settings'
                    recommended_params = dict(best.get('params') or {})
                    recommended_env_overrides = {
                        'UTILITY_EXPECTED_EDGE_WEIGHT': recommended_params.get('utility_expected_edge_weight'),
                        'UTILITY_CONFIDENCE_WEIGHT': recommended_params.get('utility_confidence_weight'),
                        'UTILITY_PROBABILITY_WEIGHT': recommended_params.get('utility_probability_weight'),
                        'UTILITY_SHORTLIST_TARGET_MAX_NAMES': recommended_params.get('utility_shortlist_target_max_names'),
                        'UTILITY_SHORTLIST_SCORE_FLOOR': recommended_params.get('utility_shortlist_score_floor'),
                        'UTILITY_SHORTLIST_SCORE_DROPOFF': recommended_params.get('utility_shortlist_score_dropoff'),
                        'UTILITY_CONFIDENCE_FLOOR': recommended_params.get('utility_confidence_floor'),
                        'UTILITY_TIER3_MAX_FRAC': recommended_params.get('utility_tier3_max_frac'),
                    }
                else:
                    headline = f"{best.get('engine_label')} is interesting, but not decisively better offline"
                    verdict = 'utility_tuning_candidate_mixed_offline'
                    recommended_action = 'do_not_change_live_utility_settings_yet'
        decision_memo_markdown = (
            '# Utility tuning lab\n\n'
            f'- **Headline:** {headline}\n'
            f'- **Verdict:** {verdict}\n'
            f'- **Recommended action:** {recommended_action}\n\n'
            '## Why this exists\n'
            '- The utility-constrained selection engine is now the live semantics candidate.\n'
            '- The next missing step is to tune its score weights and shortlist gates against the real shortlist objective, not by feel.\n'
            '- This lab searches a focused candidate grid on the same replay frame.\n'
        )
        return {
            'available': True,
            'generated_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'headline': headline,
            'summary': 'The offline lab searches a focused grid of utility-score weights and shortlist gates on the same replay frame and compares them to the current live utility settings.',
            'verdict': verdict,
            'recommended_action': recommended_action,
            'lab_inputs': {
                'hours': int(hours),
                'step_minutes': int(step_minutes),
                'max_scans': int(max_scans),
                'max_symbols': int(max_symbols),
                'scan_count': int((replay_summary.get('window') or {}).get('scan_count') or 0),
            },
            'baseline_candidate': baseline,
            'best_candidate': best,
            'best_vs_baseline_deltas': deltas,
            'candidate_summaries': candidate_summaries,
            'recommended_params': recommended_params,
            'recommended_env_overrides': recommended_env_overrides,
            'scan_rows': scan_rows,
            'decision_memo_markdown': decision_memo_markdown,
        }

    def _build_pack(self, summary: dict, replay_summary: dict, candidate_summaries: list[dict], all_rows: dict[str, list[dict]], scan_rows: list[dict]):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('utility_tuning_lab_summary.json', json.dumps(summary, indent=2, default=str))
            zf.writestr('utility_tuning_lab_decision_memo.md', str(summary.get('decision_memo_markdown') or ''))
            zf.writestr('utility_tuning_lab_scan_rows.json', json.dumps(scan_rows, indent=2, default=str))
            zf.writestr('replay_summary_snapshot.json', json.dumps({k: v for k, v in replay_summary.items() if k not in {'replay_rows', 'counterfactual_rows', 'scan_summaries'}}, indent=2, default=str))
            zf.writestr('candidate_summaries.json', json.dumps(candidate_summaries, indent=2, default=str))
            for label, rows in all_rows.items():
                safe = label.replace('/', '_').replace(' ', '_')
                zf.writestr(f'{safe}_rows.json', json.dumps(rows, indent=2, default=str))
