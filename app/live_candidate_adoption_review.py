from __future__ import annotations

from datetime import datetime, timezone
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .review_runs import ReviewPackService
from .version import APP_VERSION


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso(value: Any) -> datetime | None:
    try:
        if value in (None, ''):
            return None
        text = str(value).replace('Z', '+00:00')
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode('utf-8')


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ''):
            return default
        return float(value)
    except Exception:
        return default


def _fmt_pct(value: Any) -> str:
    value = _f(value)
    return '-' if value is None else f"{value * 100.0:.2f}%"


def _summary_txt(payload: dict) -> str:
    lines = [
        f"Headline: {payload.get('headline') or '-'}",
        f"Verdict: {payload.get('verdict') or '-'}",
        f"Recommended action: {payload.get('recommended_action') or '-'}",
        f"Summary: {payload.get('summary') or '-'}",
        '',
    ]
    sess = dict(payload.get('adoption_session') or {})
    if sess:
        lines.extend([
            'Adoption session',
            f"- Adoption session id: {sess.get('adoption_session_id') or '-'}",
            f"- Adopted at UTC: {sess.get('adopted_at_utc') or '-'}",
            f"- Candidate label: {payload.get('candidate_label') or '-'}",
            '',
        ])
    runs = dict(payload.get('adoption_runs') or {})
    if runs:
        lines.extend([
            'Matching adoption-window runs',
            f"- Matching runs: {runs.get('matching_runs') or 0}",
            f"- Evaluated runs: {runs.get('evaluated_runs') or 0}",
            '',
        ])
    evidence = dict(payload.get('adoption_evidence') or {})
    if evidence:
        lines.extend([
            'Adoption-window evidence',
            f"- Resolved rows: {evidence.get('resolved_rows') or 0}",
            f"- Visible rows: {evidence.get('visible_rows') or 0}",
            f"- Hidden rows: {evidence.get('hidden_rows') or 0}",
            f"- Visible quality hit rate: {_fmt_pct(evidence.get('visible_quality_hit_rate'))}",
            f"- Hidden quality hit rate: {_fmt_pct(evidence.get('hidden_quality_hit_rate'))}",
            f"- Visible-hidden gap: {_fmt_pct(evidence.get('visible_hidden_gap'))}",
            '',
        ])
    utility = dict(payload.get('scan_shortlist_utility') or {})
    if utility:
        lines.extend([
            'Scan-level shortlist utility',
            f"- Utility score: {utility.get('scan_shortlist_utility_score') if utility.get('scan_shortlist_utility_score') is not None else '-'}",
            f"- Mean gap: {_fmt_pct(utility.get('scan_shortlist_mean_gap'))}",
            f"- Pairwise win rate: {_fmt_pct(utility.get('scan_shortlist_pairwise_win_rate'))}",
            f"- Top-1 visible quality: {_fmt_pct(utility.get('scan_shortlist_top1_visible_quality'))}",
            f"- Top-3 visible quality: {_fmt_pct(utility.get('scan_shortlist_top3_visible_quality'))}",
            '',
        ])
    deltas = dict(payload.get('deltas_vs_activation_baseline') or {})
    if deltas:
        lines.extend([
            'Deltas vs activation baseline',
            f"- Visible quality delta: {_fmt_pct(deltas.get('visible_quality_hit_rate_delta_vs_activation'))}",
            f"- Visible-hidden gap delta: {_fmt_pct(deltas.get('visible_hidden_gap_delta_vs_activation'))}",
            '',
        ])
    return "\n".join(lines).strip() + "\n"


class LiveCandidateAdoptionReviewService:
    def __init__(self, config: AppConfig, review_packs: ReviewPackService):
        self.config = config
        self.review_packs = review_packs
        self.adoption_root = ensure_dir(Path(config.model_dir) / 'live_candidate_adoption')
        self.root_dir = ensure_dir(Path(config.model_dir) / 'live_candidate_adoption_review')
        self.summary_path = self.root_dir / 'latest_live_candidate_adoption_review_summary.json'
        self.pack_path = self.root_dir / 'latest_live_candidate_adoption_review_pack.zip'
        self.state_path = self.adoption_root / 'latest_live_candidate_adoption_state.json'
        self.summary_source_path = self.adoption_root / 'latest_live_candidate_adoption_summary.json'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _load_session(self) -> dict:
        session = read_json(self.state_path, {})
        if not isinstance(session, dict) or not session:
            return {}
        return session

    def _matching_runs(self, session: dict) -> list[dict]:
        adopted_at = _parse_iso(session.get('adopted_at_utc'))
        if adopted_at is None:
            return []
        app_version = str(session.get('app_version') or APP_VERSION)
        with self.review_packs._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_runs WHERE app_version = ? AND scan_finished_utc >= ? ORDER BY scan_finished_utc ASC",
                (app_version, adopted_at.isoformat()),
            ).fetchall()
        target_session_id = str(session.get('adoption_session_id') or '').strip()
        target_scope = str(session.get('state_scope_key') or '').strip()
        target_model_path = str(session.get('model_bundle_path_override') or '').strip()
        target_model_label = str(session.get('model_bundle_label_override') or '').strip()
        target_stage1_mode = str(session.get('stage1_selection_mode_override') or '').strip()
        target_stage1_cap = int(session.get('stage1_max_candidates_override') or 0)
        target_threshold = _f(session.get('live_raw_threshold_override'))
        run_rows: list[dict] = []
        for row in rows:
            item = dict(row)
            status = read_json(item.get('review_status_path'), {})
            adoption = dict(status.get('live_candidate_adoption') or {})
            if not adoption:
                continue
            if target_session_id and str(adoption.get('adoption_session_id') or '').strip() != target_session_id:
                continue
            if target_scope and str(adoption.get('state_scope_key') or '').strip() != target_scope:
                continue
            if target_model_path and str(adoption.get('model_bundle_path_override') or '').strip() != target_model_path:
                continue
            if target_model_label and str(adoption.get('model_bundle_label_override') or '').strip() != target_model_label:
                continue
            if target_stage1_mode and str(adoption.get('stage1_selection_mode_override') or '').strip() != target_stage1_mode:
                continue
            if target_stage1_cap and int(adoption.get('stage1_max_candidates_override') or 0) != target_stage1_cap:
                continue
            adoption_threshold = _f(adoption.get('live_raw_threshold_override'))
            if target_threshold is not None and adoption_threshold is not None and abs(adoption_threshold - target_threshold) > 1e-9:
                continue
            item['live_candidate_adoption'] = adoption
            run_rows.append(item)
        return run_rows

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
        frame = pd.DataFrame([
            {
                'scan_id': str(r.get('run_id') or ''),
                'row_type': str(r.get('row_type') or ''),
                'score': _f(r.get('live_score'), 0.0) or 0.0,
                'y': int(r.get('quality_touched') or 0),
            }
            for r in rows
        ])
        if frame.empty:
            return empty
        frame = frame.sort_values(['scan_id', 'score'], ascending=[True, False]).reset_index(drop=True)
        scan_count = int(frame['scan_id'].nunique())
        scans_with_visible = 0
        pairwise_wins = 0.0
        pairwise_comparable = 0
        visible_counts: list[int] = []
        visible_rates: list[float] = []
        hidden_rates: list[float] = []
        gaps: list[float] = []
        top1_visible: list[float] = []
        top3_visible: list[float] = []
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
            utility_score = mean_gap
            if pairwise_win_rate is not None:
                utility_score += 0.25 * (pairwise_win_rate - 0.5)
            if top1_mean is not None and hidden_quality_mean is not None:
                utility_score += 0.15 * (top1_mean - hidden_quality_mean)
            if overwide_penalty is not None:
                utility_score -= 0.10 * overwide_penalty
        return {
            'scan_shortlist_scans': scan_count,
            'scan_shortlist_scans_with_visible': scans_with_visible,
            'scan_shortlist_avg_visible_rows_per_scan': round(avg_visible_rows, 6) if avg_visible_rows is not None else None,
            'scan_shortlist_visible_quality_rate_mean': round(visible_quality_mean, 6) if visible_quality_mean is not None else None,
            'scan_shortlist_hidden_quality_rate_mean': round(hidden_quality_mean, 6) if hidden_quality_mean is not None else None,
            'scan_shortlist_mean_gap': round(mean_gap, 6) if mean_gap is not None else None,
            'scan_shortlist_pairwise_win_rate': round(pairwise_win_rate, 6) if pairwise_win_rate is not None else None,
            'scan_shortlist_pairwise_comparable_scans': pairwise_comparable,
            'scan_shortlist_top1_visible_quality': round(top1_mean, 6) if top1_mean is not None else None,
            'scan_shortlist_top3_visible_quality': round(top3_mean, 6) if top3_mean is not None else None,
            'scan_shortlist_overwide_penalty': round(overwide_penalty, 6) if overwide_penalty is not None else None,
            'scan_shortlist_utility_score': round(utility_score, 6) if utility_score is not None else None,
        }

    def build_summary(self, *, reason: str = 'refresh') -> dict:
        session = self._load_session()
        adoption_summary = read_json(self.summary_source_path, {})
        baseline = dict(adoption_summary.get('baseline_current_version_evidence_at_activation') or {})
        candidate = dict(adoption_summary.get('candidate') or {})
        active_adoption = dict(adoption_summary.get('active_adoption') or {})
        if not session:
            payload = {
                'generated_at_utc': _utc_now_iso(),
                'reason': reason,
                'headline': 'No controlled live candidate adoption session is available',
                'verdict': 'no_live_candidate_adoption_session',
                'recommended_action': 'activate_or_load_a_controlled_live_candidate_adoption_first',
                'summary': 'The app does not yet have an adoption session to review.',
            }
            atomic_write_json(self.summary_path, payload)
            self._build_pack(payload, {}, [], [], {})
            return payload

        runs = self._matching_runs(session)
        evaluated_runs = [r for r in runs if bool(r.get('evaluation_complete'))]
        resolved_rows = self.review_packs._load_rows_for_run_ids([str(r.get('run_id')) for r in evaluated_runs], resolved_only=True) if evaluated_runs else []
        visible_rows = [r for r in resolved_rows if str(r.get('row_type') or '') == 'visible']
        hidden_rows = [r for r in resolved_rows if str(r.get('row_type') or '') != 'visible']
        visible_q = _f(sum(int(r.get('quality_touched') or 0) for r in visible_rows) / len(visible_rows) if visible_rows else None)
        hidden_q = _f(sum(int(r.get('quality_touched') or 0) for r in hidden_rows) / len(hidden_rows) if hidden_rows else None)
        gap = round(visible_q - hidden_q, 6) if visible_q is not None and hidden_q is not None else None
        utility = self._scan_shortlist_utility(resolved_rows)
        base_visible_q = _f(baseline.get('visible_quality_hit_rate'))
        base_hidden_q = _f(baseline.get('non_visible_quality_hit_rate'))
        base_gap = round(base_visible_q - base_hidden_q, 6) if base_visible_q is not None and base_hidden_q is not None else None
        deltas = {
            'visible_quality_hit_rate_delta_vs_activation': round(visible_q - base_visible_q, 6) if visible_q is not None and base_visible_q is not None else None,
            'visible_hidden_gap_delta_vs_activation': round(gap - base_gap, 6) if gap is not None and base_gap is not None else None,
        }
        evidence = {
            'resolved_rows': len(resolved_rows),
            'visible_rows': len(visible_rows),
            'hidden_rows': len(hidden_rows),
            'visible_quality_hit_rate': round(visible_q, 6) if visible_q is not None else None,
            'hidden_quality_hit_rate': round(hidden_q, 6) if hidden_q is not None else None,
            'visible_hidden_gap': gap,
        }
        verdict = 'waiting_for_adoption_window_evidence'
        headline = 'Controlled live candidate adoption review is waiting for evidence'
        recommended_action = 'keep_collecting_adoption_scope_evidence'
        summary = 'The adopted path has not yet produced enough isolated matured evidence for a keep-versus-rollback decision.'
        visible_n = len(visible_rows)
        utility_score = _f(utility.get('scan_shortlist_utility_score'))
        gap_delta = _f(deltas.get('visible_hidden_gap_delta_vs_activation'))
        if not runs:
            verdict = 'no_matching_adoption_window_runs'
            headline = 'No matching adoption-window runs are available yet'
            recommended_action = 'wait_for_scans_under_the_adopted_candidate'
            summary = 'The controlled adoption exists, but no matching review-pack runs have been resolved for it yet.'
        elif visible_n < 40:
            verdict = 'waiting_for_more_resolved_visible_rows'
            headline = 'Adoption review needs more resolved visible rows'
            recommended_action = 'keep_collecting_isolated_adoption_scope_evidence'
            summary = 'The adopted path has some evidence, but not enough resolved visible rows for a decision-grade keep-versus-rollback verdict.'
        elif gap is not None and gap < 0 and (gap_delta is None or gap_delta <= 0):
            verdict = 'adoption_review_recommends_rollback'
            headline = 'Adoption review recommends rollback'
            recommended_action = 'clear_the_controlled_live_candidate_adoption'
            summary = 'The isolated adopted-path evidence is underperforming the hidden remainder and does not beat the activation baseline.'
        elif utility_score is not None and utility_score > 0 and gap is not None and gap > 0 and (gap_delta is None or gap_delta >= 0):
            verdict = 'adoption_review_supports_keeping_candidate'
            headline = 'Adoption review supports keeping the adopted candidate'
            recommended_action = 'keep_the_adopted_candidate_and_continue_monitoring'
            summary = 'The isolated adopted-path evidence continues to beat the hidden remainder and is at least as strong as the activation baseline.'
        else:
            verdict = 'adoption_review_inconclusive'
            headline = 'Adoption review is mixed'
            recommended_action = 'continue_monitoring_or_rollback_if_operator_risk_tolerance_is_low'
            summary = 'The adopted path is producing evidence, but it is not yet decisive enough to lock in a keep-versus-rollback conclusion.'

        payload = {
            'generated_at_utc': _utc_now_iso(),
            'reason': reason,
            'headline': headline,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'summary': summary,
            'candidate_label': session.get('utility_selection_engine_label') or session.get('adoption_session_id') or '-',
            'adoption_session': session,
            'adoption_runs': {
                'matching_runs': len(runs),
                'evaluated_runs': len(evaluated_runs),
            },
            'adoption_evidence': evidence,
            'scan_shortlist_utility': utility,
            'baseline_current_version_evidence_at_activation': baseline or None,
            'deltas_vs_activation_baseline': deltas,
            'candidate': candidate or None,
            'active_adoption': active_adoption or None,
            'decision_memo_markdown': (
                '# Controlled live candidate adoption review\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n\n'
                '## Why this tranche exists\n'
                '- The adoption gate can activate a rollback-aware adopted live path.\n'
                '- The remaining missing step is a clean keep-versus-rollback verdict for the adopted path itself.\n'
                '- This service isolates adoption-window evidence instead of relying on contaminated current-version aggregates.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, session, runs, evaluated_runs, adoption_summary)
        return payload

    def _build_pack(self, summary: dict, session: dict, runs: list[dict], evaluated_runs: list[dict], adoption_summary: dict):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('live_candidate_adoption_review_summary.json', _json_bytes(summary))
            zf.writestr('live_candidate_adoption_review_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('live_candidate_adoption_session.json', _json_bytes(session or {}))
            zf.writestr('live_candidate_adoption_summary.json', _json_bytes(adoption_summary or {}))
            zf.writestr('matching_adoption_window_runs.json', _json_bytes(runs or []))
            zf.writestr('evaluated_adoption_window_runs.json', _json_bytes(evaluated_runs or []))
