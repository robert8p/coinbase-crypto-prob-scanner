from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .replay import HistoricalReplayService
from .review_runs import ReviewPackService
from .utility_shortlist import optimize_visible_shortlist, legacy_visible_shortlist
from .version import APP_VERSION


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ''):
            return default
        return float(value)
    except Exception:
        return default


class UtilitySelectionLabService:
    def __init__(self, config: AppConfig, replay: HistoricalReplayService, review_packs: ReviewPackService):
        self.config = config
        self.replay = replay
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / 'utility_selection_lab')
        self.summary_path = self.root_dir / 'latest_utility_selection_lab_summary.json'
        self.pack_path = self.root_dir / 'latest_utility_selection_lab_pack.zip'

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
        current_engine = str(getattr(self.config, 'live_selection_mode', 'utility_constrained') or 'utility_constrained').lower()

        utility_rows: list[dict] = []
        legacy_rows: list[dict] = []
        scan_rows: list[dict] = []
        for as_of, scan_rows_all in scans.items():
            candidate_pool = self._candidate_pool(scan_rows_all)
            regime_state = str((scan_rows_all[0] if scan_rows_all else {}).get('market_regime_state') or 'green').lower()
            effective_max = self._effective_max_for_regime(regime_state)
            utility_result = optimize_visible_shortlist(candidate_pool, effective_max=effective_max, config=self.config, tracked_priority_symbols=[])
            legacy_result = legacy_visible_shortlist(candidate_pool, effective_max=effective_max, config=self.config, tracked_priority_symbols=[])
            utility_eval = self._evaluate_engine(as_of=as_of, visible_rows=utility_result.visible_rows, pool_rows=candidate_pool)
            legacy_eval = self._evaluate_engine(as_of=as_of, visible_rows=legacy_result.visible_rows, pool_rows=candidate_pool)
            utility_rows.extend(utility_eval['visible_rows'] + utility_eval['hidden_rows'])
            legacy_rows.extend(legacy_eval['visible_rows'] + legacy_eval['hidden_rows'])
            scan_rows.append({
                'as_of_utc': as_of,
                'market_regime_state': regime_state,
                'candidate_pool_count': len(candidate_pool),
                'effective_max': effective_max,
                'utility_visible_count': len(utility_result.visible_rows),
                'legacy_visible_count': len(legacy_result.visible_rows),
                'utility_visible_quality_hit_rate': utility_eval['visible_quality_hit_rate'],
                'legacy_visible_quality_hit_rate': legacy_eval['visible_quality_hit_rate'],
                'utility_hidden_quality_hit_rate': utility_eval['hidden_quality_hit_rate'],
                'legacy_hidden_quality_hit_rate': legacy_eval['hidden_quality_hit_rate'],
                'utility_gap': utility_eval['visible_hidden_gap'],
                'legacy_gap': legacy_eval['visible_hidden_gap'],
                'utility_top1_symbol': (utility_result.visible_rows[0].get('symbol') if utility_result.visible_rows else None),
                'legacy_top1_symbol': (legacy_result.visible_rows[0].get('symbol') if legacy_result.visible_rows else None),
            })

        utility_summary = self._engine_summary(str(getattr(self.config, 'utility_selection_engine_label', 'utility_constrained_v7') or 'utility_constrained_v7'), utility_rows)
        legacy_summary = self._engine_summary('legacy_ranked_cap_v1', legacy_rows)
        summary = self._build_summary(
            current_version=current_version,
            replay_summary=replay_summary,
            utility_summary=utility_summary,
            legacy_summary=legacy_summary,
            scan_rows=scan_rows,
            current_engine=current_engine,
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
        )
        atomic_write_json(self.summary_path, summary)
        self._build_pack(summary, replay_summary, utility_rows, legacy_rows, scan_rows)
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

    def _evaluate_engine(self, *, as_of: str, visible_rows: list[dict], pool_rows: list[dict]) -> dict:
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
        empty = {
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
        if not rows:
            return empty
        frame = pd.DataFrame([{
            'scan_id': r.get('as_of_utc'),
            'row_type': str(r.get('row_type') or ''),
            'score': _f(r.get('utility_decision_score', r.get('prob_2_rank') or r.get('prob_2') or 0.0), 0.0) or 0.0,
            'y': int(r.get('quality_touched') or 0),
        } for r in rows]).dropna(subset=['scan_id'])
        if frame.empty:
            return empty
        frame = frame.sort_values(['scan_id', 'score'], ascending=[True, False]).reset_index(drop=True)
        base_event_rate = float(frame['y'].mean()) if len(frame) else 0.0
        scan_count = int(frame['scan_id'].nunique())
        scans_with_visible = 0
        pairwise_wins = 0.0
        pairwise_comparable = 0
        visible_counts, visible_rates, hidden_rates, gaps, top1_visible, top3_visible = [], [], [], [], [], []
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

    def _build_summary(self, *, current_version: dict, replay_summary: dict, utility_summary: dict, legacy_summary: dict, scan_rows: list[dict], current_engine: str, hours: int, step_minutes: int, max_scans: int, max_symbols: int) -> dict:
        utility_score = _f(utility_summary.get('scan_shortlist_utility_score'))
        legacy_score = _f(legacy_summary.get('scan_shortlist_utility_score'))
        utility_gap = _f(utility_summary.get('scan_shortlist_mean_gap'))
        legacy_gap = _f(legacy_summary.get('scan_shortlist_mean_gap'))
        utility_pairwise = _f(utility_summary.get('scan_shortlist_pairwise_win_rate'))
        legacy_pairwise = _f(legacy_summary.get('scan_shortlist_pairwise_win_rate'))
        utility_top1 = _f(utility_summary.get('scan_shortlist_top1_visible_quality'))
        legacy_top1 = _f(legacy_summary.get('scan_shortlist_top1_visible_quality'))
        summary = 'The offline lab compares the new utility-constrained shortlist engine against the legacy ranked-cap shortlist on the same replay frame.'
        headline = 'Utility selection lab produced a mixed result'
        verdict = 'mixed_offline_result'
        recommended_action = 'keep_current_engine_and_review_summary'
        if utility_score is not None and legacy_score is not None:
            clearly_better = (utility_score - legacy_score) >= 0.02 and ((utility_gap or -9) - (legacy_gap or -9)) >= 0.01 and ((utility_pairwise or 0.0) - (legacy_pairwise or 0.0)) >= 0.03
            clearly_worse = (legacy_score - utility_score) >= 0.02 and ((legacy_gap or -9) - (utility_gap or -9)) >= 0.01 and ((legacy_pairwise or 0.0) - (utility_pairwise or 0.0)) >= 0.03
            if clearly_better:
                headline = 'Utility-constrained shortlist beats the legacy shortlist offline'
                verdict = 'utility_engine_supported_offline'
                recommended_action = 'keep_utility_constrained_engine_and_tune_it_further'
            elif clearly_worse:
                headline = 'Legacy shortlist beats the utility-constrained shortlist offline'
                verdict = 'legacy_engine_preferred_offline'
                recommended_action = 'rework_utility_selection_before_treating_it_as_the_new_default'
        decision_memo_markdown = (
            '# Utility selection lab\n\n'
            f'- **Headline:** {headline}\n'
            f'- **Verdict:** {verdict}\n'
            f'- **Recommended action:** {recommended_action}\n\n'
            '## Why this exists\n'
            '- v4.12.0 changed live shortlist selection semantics.\n'
            '- The next missing step is an objective-aligned offline validator for the new selection engine.\n'
            '- This lab compares the utility-constrained shortlist against the legacy ranked-cap shortlist on the same replay frame.\n'
        )
        return {
            'available': True,
            'generated_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'headline': headline,
            'summary': summary,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'current_live_selection_mode': current_engine,
            'lab_inputs': {
                'hours': int(hours),
                'step_minutes': int(step_minutes),
                'max_scans': int(max_scans),
                'max_symbols': int(max_symbols),
                'scan_count': int((replay_summary.get('window') or {}).get('scan_count') or 0),
            },
            'utility_engine': utility_summary,
            'legacy_engine': legacy_summary,
            'scan_rows': scan_rows,
            'decision_memo_markdown': decision_memo_markdown,
        }

    def _build_pack(self, summary: dict, replay_summary: dict, utility_rows: list[dict], legacy_rows: list[dict], scan_rows: list[dict]):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('utility_selection_lab_summary.json', json.dumps(summary, indent=2, default=str))
            zf.writestr('utility_selection_lab_decision_memo.md', str(summary.get('decision_memo_markdown') or ''))
            zf.writestr('utility_selection_lab_scan_rows.json', json.dumps(scan_rows, indent=2, default=str))
            zf.writestr('replay_summary_snapshot.json', json.dumps({k: v for k, v in replay_summary.items() if k not in {'replay_rows', 'counterfactual_rows', 'scan_summaries'}}, indent=2, default=str))
            zf.writestr('utility_engine_rows.json', json.dumps(utility_rows, indent=2, default=str))
            zf.writestr('legacy_engine_rows.json', json.dumps(legacy_rows, indent=2, default=str))
