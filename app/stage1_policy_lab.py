from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .replay import HistoricalReplayService
from .review_runs import ReviewPackService
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


def _csv_bytes(rows: list[dict], fieldnames: Iterable[str] | None = None) -> bytes:
    rows = list(rows or [])
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update((row or {}).keys())
        fieldnames = sorted(keys)
    fieldnames = list(fieldnames or [])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (row or {}).get(k) for k in fieldnames})
    return buffer.getvalue().encode('utf-8')


class Stage1PolicyLabService:
    def __init__(self, config: AppConfig, replay: HistoricalReplayService, review_packs: ReviewPackService):
        self.config = config
        self.replay = replay
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / 'stage1_policy_lab')
        self.summary_path = self.root_dir / 'latest_stage1_policy_lab_summary.json'
        self.pack_path = self.root_dir / 'latest_stage1_policy_lab_pack.zip'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def run(self, *, hours: int = 168, step_minutes: int = 120, max_scans: int = 84, max_symbols: int = 100) -> dict:
        try:
            current_version = self.review_packs.get_current_version_summary() or {}
        except FileNotFoundError:
            current_version = {}
        checkpoint = dict(current_version.get('decision_checkpoint') or current_version.get('decision_rule_checkpoint') or {})
        live_mode = str(checkpoint.get('stage1_selection_mode') or self.config.stage1_selection_mode or 'stage1_opportunity_model')
        live_cap = int(getattr(self.config, 'stage1_max_candidates', 40) or 40)
        raw_threshold = _f(
            checkpoint.get('live_raw_threshold')
            or checkpoint.get('effective_live_raw_threshold')
            or (current_version.get('decision_branch_automation') or {}).get('effective_live_raw_threshold')
            or self.config.live_raw_threshold,
            0.35,
        ) or 0.35

        policies: list[dict] = []
        def add(mode: str, cap: int, kind: str):
            mode = str(mode or '').strip() or live_mode
            cap = max(1, int(cap))
            if any(p['mode'] == mode and int(p['max_candidates']) == cap for p in policies):
                return
            policies.append({
                'label': f'{mode}@{cap}',
                'mode': mode,
                'max_candidates': cap,
                'policy_kind': kind,
            })

        add(live_mode, live_cap, 'baseline')
        add(live_mode, min(80, live_cap + 10), 'looser_cap')
        add(live_mode, min(80, live_cap + 20), 'looser_cap')
        add('primary_plus_opportunity_reserve', live_cap, 'reserve_blend')
        add('primary_plus_opportunity_reserve', min(80, live_cap + 10), 'reserve_blend')

        rows: list[dict] = []
        policy_summaries: dict[str, dict] = {}
        for policy in policies:
            result = self.replay.run(
                hours=hours,
                step_minutes=step_minutes,
                max_scans=max_scans,
                max_symbols=max_symbols,
                pipeline_mode='full',
                raw_threshold=raw_threshold,
                stage1_selection_mode_override=policy['mode'],
                stage1_max_candidates_override=policy['max_candidates'],
            )
            summary = dict(result.get('summary') or {})
            policy_summaries[policy['label']] = summary
            utility = self._display_shortlist_utility(summary)
            visible_bucket = dict(summary.get('visible_bucket') or {})
            hidden_bucket = dict(summary.get('non_visible_bucket') or {})
            counter = dict(summary.get('counterfactual') or {})
            replay_rows = list(summary.get('replay_rows') or [])
            rows.append({
                **policy,
                'window_start_utc': ((summary.get('window') or {}).get('start_utc')),
                'window_end_utc': ((summary.get('window') or {}).get('end_utc')),
                'scan_count': int(((summary.get('window') or {}).get('scan_count')) or 0),
                'resolved_rows': int((summary.get('surfaced_evidence') or {}).get('resolved_rows') or 0),
                'visible_quality_hit_rate': visible_bucket.get('quality_hit_rate'),
                'hidden_quality_hit_rate': hidden_bucket.get('quality_hit_rate'),
                'visible_hidden_gap': None if _f(visible_bucket.get('quality_hit_rate')) is None or _f(hidden_bucket.get('quality_hit_rate')) is None else round((_f(visible_bucket.get('quality_hit_rate'), 0.0) or 0.0) - (_f(hidden_bucket.get('quality_hit_rate'), 0.0) or 0.0), 6),
                'stage1_quality_recall': counter.get('stage1_quality_recall'),
                'visible_count': sum(1 for r in replay_rows if str(r.get('row_type') or '') == 'visible'),
                **utility,
            })

        rows.sort(key=self._policy_sort_key, reverse=True)
        baseline = next((r for r in rows if r.get('policy_kind') == 'baseline'), rows[0] if rows else None)
        best = rows[0] if rows else None
        summary = self._build_summary(
            current_version=current_version,
            checkpoint=checkpoint,
            baseline=baseline,
            best=best,
            rows=rows,
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            raw_threshold=raw_threshold,
        )
        atomic_write_json(self.summary_path, summary)
        self._build_pack(summary=summary, rows=rows, policy_summaries=policy_summaries)
        return summary

    def _policy_sort_key(self, row: dict) -> tuple:
        return (
            float(row.get('scan_shortlist_utility_score') or -9.0),
            float(row.get('scan_shortlist_mean_gap') or -9.0),
            float(row.get('scan_shortlist_pairwise_win_rate') or 0.0),
            float(row.get('scan_shortlist_top1_visible_quality') or 0.0),
            float(row.get('scan_shortlist_top3_visible_quality') or 0.0),
            -float(row.get('scan_shortlist_avg_visible_rows_per_scan') or 999.0),
            float(row.get('visible_hidden_gap') or -9.0),
            float(row.get('visible_quality_hit_rate') or 0.0),
            float(row.get('stage1_quality_recall') or 0.0),
        )

    def _display_shortlist_utility(self, summary: dict) -> dict:
        replay_rows = list(summary.get('replay_rows') or [])
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
        if not replay_rows:
            return empty
        frame = pd.DataFrame([{ 
            'ts': r.get('as_of_utc') or r.get('entry_utc'),
            'row_type': str(r.get('row_type') or ''),
            'score': _f(r.get('live_score'), 0.0) or 0.0,
            'y': int(r.get('quality_touched') or 0),
        } for r in replay_rows]).dropna(subset=['ts'])
        if frame.empty:
            return empty
        frame['ts'] = pd.to_datetime(frame['ts'], utc=True, errors='coerce')
        frame = frame.dropna(subset=['ts']).sort_values(['ts', 'score'], ascending=[True, False]).reset_index(drop=True)
        if frame.empty:
            return empty
        base_event_rate = float(frame['y'].mean()) if len(frame) else 0.0
        scan_count = int(frame['ts'].nunique())
        scans_with_visible = 0
        pairwise_wins = 0.0
        pairwise_comparable = 0
        visible_counts: list[int] = []
        visible_rates: list[float] = []
        hidden_rates: list[float] = []
        gaps: list[float] = []
        top1_visible: list[float] = []
        top3_visible: list[float] = []
        for _, scan in frame.groupby('ts', sort=False):
            scan = scan.sort_values('score', ascending=False)
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
                    elif abs(gap) <= 1e-12:
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

    def _build_summary(self, *, current_version: dict, checkpoint: dict, baseline: dict | None, best: dict | None, rows: list[dict], hours: int, step_minutes: int, max_scans: int, max_symbols: int, raw_threshold: float) -> dict:
        live_mode = str(checkpoint.get('stage1_selection_mode') or self.config.stage1_selection_mode or 'stage1_opportunity_model')
        live_cap = int(getattr(self.config, 'stage1_max_candidates', 40) or 40)
        visible_q = _f((current_version.get('evidence') or {}).get('visible_quality_hit_rate'))
        hidden_q = _f((current_version.get('evidence') or {}).get('non_visible_quality_hit_rate'))
        verdict = 'no_stage1_change_supported'
        headline = 'No looser Stage 1 policy clearly beats the current policy offline'
        recommended_action = 'keep_live_stage1_unchanged'
        recommended_policy = None
        recommended_action_reason = 'The offline Stage 1 policy lab did not find a materially better visible-shortlist policy than the current Stage 1 policy.'
        deltas: dict[str, float | None] = {}
        rationale: list[str] = []
        if baseline and best:
            for name in [
                'scan_shortlist_utility_score',
                'scan_shortlist_mean_gap',
                'scan_shortlist_pairwise_win_rate',
                'scan_shortlist_top1_visible_quality',
                'scan_shortlist_top3_visible_quality',
                'scan_shortlist_avg_visible_rows_per_scan',
                'visible_hidden_gap',
                'visible_quality_hit_rate',
                'stage1_quality_recall',
            ]:
                best_v = _f(best.get(name))
                base_v = _f(baseline.get(name))
                deltas[f'{name}_delta'] = None if best_v is None or base_v is None else round(best_v - base_v, 6)
            if deltas.get('scan_shortlist_utility_score_delta') is not None:
                rationale.append(f"Best policy utility delta vs live baseline: {deltas['scan_shortlist_utility_score_delta']:+.4f}.")
            if deltas.get('scan_shortlist_mean_gap_delta') is not None:
                rationale.append(f"Visible-vs-hidden mean gap delta: {deltas['scan_shortlist_mean_gap_delta']:+.4f}.")
            if deltas.get('scan_shortlist_pairwise_win_rate_delta') is not None:
                rationale.append(f"Per-scan win-rate delta: {deltas['scan_shortlist_pairwise_win_rate_delta']:+.2%}.")
            if deltas.get('scan_shortlist_avg_visible_rows_per_scan_delta') is not None:
                rationale.append(f"Average visible rows/scan delta: {deltas['scan_shortlist_avg_visible_rows_per_scan_delta']:+.2f}.")
            support = (
                best.get('label') != baseline.get('label')
                and (deltas.get('scan_shortlist_utility_score_delta') or 0.0) >= 0.015
                and (deltas.get('scan_shortlist_mean_gap_delta') or 0.0) >= 0.015
                and (deltas.get('scan_shortlist_pairwise_win_rate_delta') or 0.0) >= 0.04
                and ((deltas.get('scan_shortlist_avg_visible_rows_per_scan_delta') or 0.0) <= 1.50)
            )
            if support:
                verdict = 'stage1_live_candidate_supported_offline'
                headline = f"{best.get('label')} is the best offline Stage 1 relaxation candidate"
                recommended_action = 'prepare_single_live_stage1_candidate_only_after_challenger_branch_is_clear'
                recommended_policy = {
                    'stage1_selection_mode': best.get('mode'),
                    'stage1_max_candidates': int(best.get('max_candidates') or live_cap),
                }
                recommended_action_reason = (
                    f"Relative to the current live-style baseline {baseline.get('label')}, {best.get('label')} improved scan-level shortlist utility, visible-vs-hidden separation, and per-scan win rate without obviously blowing out shortlist width."
                )
            elif best.get('label') != baseline.get('label'):
                verdict = 'mixed_stage1_result_keep_live_unchanged'
                headline = f"{best.get('label')} looks interesting offline, but not decisively enough"
                recommended_action_reason = 'The best offline Stage 1 variant improved at least one utility dimension, but not strongly enough to justify a live Stage 1 change yet.'
        decision_memo_markdown = (
            "# Stage 1 policy lab\n\n"
            f"- **Headline:** {headline}\n"
            f"- **Verdict:** {verdict}\n"
            f"- **Recommended action:** {recommended_action}\n"
            f"- **Why:** {recommended_action_reason}\n\n"
            "## Live baseline\n"
            f"- Stage 1 mode: {live_mode}\n"
            f"- Stage 1 max candidates: {live_cap}\n"
            f"- Raw threshold: {raw_threshold:.2f}\n"
            f"- Current visible quality-hit rate: {visible_q}\n"
            f"- Current hidden quality-hit rate: {hidden_q}\n\n"
            "## Offline comparison rules\n"
            "- Same replay window for every candidate.\n"
            "- Same Stage 2 and live threshold semantics.\n"
            "- Rank candidates by visible-shortlist utility before generic counts.\n"
            "- Penalize overly wide visible shortlists.\n\n"
            "## Best-vs-baseline deltas\n"
            + ''.join(f"- {line}\n" for line in rationale)
        )
        return {
            'available': True,
            'generated_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'headline': headline,
            'summary': recommended_action_reason,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'recommended_action_reason': recommended_action_reason,
            'recommended_policy': recommended_policy,
            'live_baseline': {
                'stage1_selection_mode': live_mode,
                'stage1_max_candidates': live_cap,
                'raw_threshold': round(float(raw_threshold), 4),
            },
            'current_live_evidence': {
                'visible_quality_hit_rate': visible_q,
                'non_visible_quality_hit_rate': hidden_q,
            },
            'lab_inputs': {
                'hours': int(hours),
                'step_minutes': int(step_minutes),
                'max_scans': int(max_scans),
                'max_symbols': int(max_symbols),
                'candidate_policy_count': len(rows),
            },
            'best_policy': best,
            'baseline_policy': baseline,
            'best_vs_baseline_deltas': deltas,
            'policy_rows': rows,
            'decision_memo_markdown': decision_memo_markdown,
        }

    def _build_pack(self, *, summary: dict, rows: list[dict], policy_summaries: dict[str, dict]) -> Path:
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('stage1_policy_lab_summary.json', json.dumps(summary, indent=2, default=str))
            zf.writestr('stage1_policy_lab_rows.csv', _csv_bytes(rows))
            zf.writestr('stage1_policy_lab_decision_memo.md', str(summary.get('decision_memo_markdown') or ''))
            for label, payload in sorted(policy_summaries.items()):
                safe = label.replace('@', '_cap_').replace('/', '_').replace('.', '_')
                zf.writestr(f'policy_{safe}_replay_summary.json', json.dumps(payload, indent=2, default=str))
        return self.pack_path
