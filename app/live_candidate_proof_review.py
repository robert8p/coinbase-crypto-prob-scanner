from __future__ import annotations

from datetime import datetime, timezone
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .review_runs import ReviewPackService, ROW_CSV_FIELDS, _csv_bytes, _f, _is_non_visible_row
from .runtime_scope import current_runtime_scope
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


def _fmt_pct(value: Any) -> str:
    try:
        if value in (None, ''):
            return '-'
        return f"{float(value):.2%}"
    except Exception:
        return str(value)


def _summary_txt(payload: dict) -> str:
    lines = [
        f"Headline: {payload.get('headline') or '-'}",
        f"Verdict: {payload.get('verdict') or '-'}",
        f"Recommended action: {payload.get('recommended_action') or '-'}",
        f"Summary: {payload.get('summary') or '-'}",
        '',
    ]
    session = dict(payload.get('proof_session') or {})
    if session:
        lines.extend([
            'Proof session',
            f"- Session id: {session.get('proof_session_id') or '-'}",
            f"- Candidate label: {session.get('candidate_label') or '-'}",
            f"- Activated at UTC: {session.get('activated_at_utc') or '-'}",
            f"- Expires at UTC: {session.get('expires_at_utc') or '-'}",
            f"- Cleared at UTC: {session.get('cleared_at_utc') or '-'}",
            '',
        ])
    stats = dict(payload.get('proof_runs') or {})
    if stats:
        lines.extend([
            'Proof-window run coverage',
            f"- Matching runs: {stats.get('matching_runs') or 0}",
            f"- Evaluated runs: {stats.get('evaluated_runs') or 0}",
            f"- Matching visible rows: {stats.get('matching_visible_rows') or 0}",
            f"- Matching non-visible rows: {stats.get('matching_non_visible_rows') or 0}",
            '',
        ])
    evidence = dict(payload.get('proof_evidence') or {})
    if evidence:
        lines.extend([
            'Proof evidence',
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
    return "\n".join(lines).strip() + "\n"


class LiveCandidateProofReviewService:
    def __init__(self, config: AppConfig, review_packs: ReviewPackService):
        self.config = config
        self.review_packs = review_packs
        self.proof_root = ensure_dir(Path(config.model_dir) / 'live_candidate_proof')
        self.root_dir = ensure_dir(Path(config.model_dir) / 'live_candidate_proof_review')
        self.summary_path = self.root_dir / 'latest_live_candidate_proof_review_summary.json'
        self.pack_path = self.root_dir / 'latest_live_candidate_proof_review_pack.zip'
        self.session_path = self.proof_root / 'latest_live_candidate_proof_session.json'
        self.live_proof_summary_path = self.proof_root / 'latest_live_candidate_proof_summary.json'
        self.next_candidate_summary_path = Path(config.model_dir) / 'next_live_candidate_lab' / 'latest_next_live_candidate_lab_summary.json'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _load_session(self) -> dict:
        session = read_json(self.session_path, {})
        if isinstance(session, dict) and session:
            return session
        summary = read_json(self.live_proof_summary_path, {})
        active = dict(summary.get('active_override') or {})
        candidate = dict(summary.get('recommended_candidate') or {})
        if active:
            return {
                'proof_session_id': active.get('proof_session_id'),
                'activated_at_utc': active.get('activated_at_utc'),
                'expires_at_utc': active.get('expires_at_utc'),
                'state_scope_key': active.get('state_scope_key'),
                'app_version': active.get('app_version') or APP_VERSION,
                'recommended_candidate': candidate,
                'candidate_label': candidate.get('model_source') or candidate.get('model_kind'),
                'model_bundle_path_override': active.get('model_bundle_path_override'),
                'model_bundle_label_override': active.get('model_bundle_label_override'),
                'stage1_selection_mode_override': active.get('stage1_selection_mode_override'),
                'stage1_max_candidates_override': active.get('stage1_max_candidates_override'),
            }
        return {}

    def _matching_runs(self, session: dict) -> list[dict]:
        activated_at = _parse_iso(session.get('activated_at_utc'))
        if activated_at is None:
            return []
        app_version = str(session.get('app_version') or APP_VERSION)
        with self.review_packs._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_runs WHERE app_version = ? AND scan_finished_utc >= ? ORDER BY scan_finished_utc ASC",
                (app_version, activated_at.isoformat()),
            ).fetchall()
        target_session_id = str(session.get('proof_session_id') or '').strip()
        target_scope = str(session.get('state_scope_key') or '').strip()
        target_model_path = str(session.get('model_bundle_path_override') or '').strip()
        target_model_label = str(session.get('model_bundle_label_override') or '').strip()
        target_stage1_mode = str(session.get('stage1_selection_mode_override') or '').strip()
        target_stage1_cap = int(session.get('stage1_max_candidates_override') or 0)
        run_rows = []
        for row in rows:
            item = dict(row)
            status = read_json(item.get('review_status_path'), {})
            proof = dict(status.get('live_candidate_proof') or {})
            if not proof:
                continue
            if target_session_id:
                if str(proof.get('proof_session_id') or '').strip() != target_session_id:
                    continue
            else:
                if not bool(proof.get('active')):
                    continue
                if target_scope and str(proof.get('state_scope_key') or '').strip() != target_scope:
                    continue
                if target_model_path and str(proof.get('model_bundle_path_override') or '').strip() != target_model_path:
                    continue
                if target_model_label and str(proof.get('model_bundle_label_override') or '').strip() != target_model_label:
                    continue
                if target_stage1_mode and str(proof.get('stage1_selection_mode_override') or '').strip() != target_stage1_mode:
                    continue
                if target_stage1_cap and int(proof.get('stage1_max_candidates_override') or 0) != target_stage1_cap:
                    continue
            item['live_candidate_proof'] = proof
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
        base_event_rate = float(frame['y'].mean()) if len(frame) else 0.0
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

    def build_summary(self, *, reason: str = 'refresh') -> dict:
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        session = self._load_session()
        if not session:
            payload = {
                'available': False,
                'generated_at_utc': _utc_now_iso(),
                'app_version': APP_VERSION,
                'headline': 'No retained live proof session is available',
                'verdict': 'no_live_candidate_proof_session',
                'recommended_action': 'activate_a_controlled_live_candidate_proof_window_first',
                'summary': 'There is no retained proof session on disk, so no isolated proof-window evidence can be reviewed yet.',
                'reason': reason,
                'current_scope': current_scope,
            }
            atomic_write_json(self.summary_path, payload)
            self._build_pack(payload, [], [], {}, {}, {})
            return payload

        runs = self._matching_runs(session)
        run_ids = [str(r.get('run_id')) for r in runs if r.get('run_id')]
        resolved_rows = self.review_packs._load_rows_for_run_ids(run_ids, resolved_only=True) if run_ids else []
        visible_rows = [r for r in resolved_rows if str(r.get('row_type') or '') == 'visible']
        hidden_rows = [r for r in resolved_rows if _is_non_visible_row(r)]
        visible_bucket = self.review_packs._bucket_summary(visible_rows) if visible_rows else {}
        hidden_bucket = self.review_packs._bucket_summary(hidden_rows) if hidden_rows else {}
        utility = self._scan_shortlist_utility(resolved_rows)
        evidence = {
            'resolved_rows': len(resolved_rows),
            'visible_rows': len(visible_rows),
            'hidden_rows': len(hidden_rows),
            'visible_quality_hit_rate': visible_bucket.get('quality_hit_rate'),
            'hidden_quality_hit_rate': hidden_bucket.get('quality_hit_rate'),
            'visible_hidden_gap': None,
            'visible_avg_end_ret': visible_bucket.get('avg_end_ret'),
            'hidden_avg_end_ret': hidden_bucket.get('avg_end_ret'),
            'visible_avg_mae': visible_bucket.get('avg_mae'),
            'hidden_avg_mae': hidden_bucket.get('avg_mae'),
        }
        if _f(evidence['visible_quality_hit_rate']) is not None and _f(evidence['hidden_quality_hit_rate']) is not None:
            evidence['visible_hidden_gap'] = round((_f(evidence['visible_quality_hit_rate'], 0.0) or 0.0) - (_f(evidence['hidden_quality_hit_rate'], 0.0) or 0.0), 6)

        evaluated_runs = [r for r in runs if bool(r.get('evaluation_complete'))]
        proof_runs = {
            'matching_runs': len(runs),
            'evaluated_runs': len(evaluated_runs),
            'matching_visible_rows': sum(int(r.get('visible_rows_count') or 0) for r in runs),
            'matching_non_visible_rows': sum(int(r.get('suppressed_rows_count') or 0) for r in runs),
        }
        next_summary = read_json(self.next_candidate_summary_path, {})
        candidate = dict(session.get('recommended_candidate') or {})
        candidate_label = str(session.get('candidate_label') or candidate.get('model_source') or candidate.get('model_kind') or 'candidate')
        baseline_snapshot = dict(session.get('baseline_current_version_summary') or {})
        baseline_evidence = dict(baseline_snapshot.get('evidence') or {})

        headline = 'Controlled live proof has not yet produced reviewable evidence'
        verdict = 'waiting_for_proof_window_evidence'
        recommended_action = 'keep_candidate_scoped_and_wait_for_matured_outcomes'
        summary = 'The proof window exists, but there is not yet enough isolated matured evidence to judge the candidate.'
        visible_rows_count = int(evidence.get('visible_rows') or 0)
        pairwise = _f(utility.get('scan_shortlist_pairwise_win_rate'))
        mean_gap = _f(utility.get('scan_shortlist_mean_gap'))
        utility_score = _f(utility.get('scan_shortlist_utility_score'))
        if proof_runs['matching_runs'] <= 0:
            headline = 'No matching proof-window runs have been captured yet'
            verdict = 'no_matching_proof_window_runs'
            recommended_action = 'keep_proof_window_active_until_scans_are_captured'
            summary = 'The candidate proof session exists, but no scan runs attributable to that exact proof window have been recorded yet.'
        elif len(evaluated_runs) <= 0 or visible_rows_count < 30:
            headline = 'Controlled live proof is collecting evidence'
            verdict = 'waiting_for_more_resolved_visible_rows'
            recommended_action = 'allow_more_proof_window_runs_to_mature'
            summary = 'The proof window has matching runs, but decision-grade visible evidence has not matured yet.'
        elif mean_gap is not None and mean_gap > 0 and (pairwise is None or pairwise >= 0.55) and (utility_score is None or utility_score > 0):
            headline = 'Controlled live proof currently supports the candidate'
            verdict = 'live_proof_supports_candidate'
            recommended_action = 'prepare_a_formal_decision_checkpoint_before_any_promotion'
            summary = 'Isolated proof-window evidence shows the visible shortlist beating the hidden remainder on the candidate runs, so the candidate has earned a formal next decision checkpoint.'
        elif mean_gap is not None and mean_gap < 0 and (pairwise is None or pairwise <= 0.45) and (utility_score is None or utility_score < 0):
            headline = 'Controlled live proof currently rejects the candidate'
            verdict = 'live_proof_rejects_candidate'
            recommended_action = 'clear_the_proof_window_and_do_not_promote_this_candidate'
            summary = 'Isolated proof-window evidence shows the candidate failing the shortlist-vs-hidden test in live conditions, so it should not be promoted.'
        else:
            headline = 'Controlled live proof remains mixed'
            verdict = 'live_proof_inconclusive'
            recommended_action = 'keep_candidate_scoped_or_end_the_window_without_promotion'
            summary = 'The candidate proof window produced some matured evidence, but the live shortlist-vs-hidden result is still mixed.'

        payload = {
            'available': True,
            'generated_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'headline': headline,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'summary': summary,
            'reason': reason,
            'current_scope': current_scope,
            'proof_session': session,
            'proof_runs': proof_runs,
            'proof_evidence': evidence,
            'scan_shortlist_utility': utility,
            'candidate_label': candidate_label,
            'recommended_candidate': candidate,
            'next_live_candidate_summary_verdict': next_summary.get('verdict'),
            'baseline_current_version_evidence_at_activation': {
                'resolved_rows': int(baseline_evidence.get('resolved_rows') or 0),
                'visible_rows': int(baseline_evidence.get('visible_rows') or 0),
                'visible_quality_hit_rate': baseline_evidence.get('visible_quality_hit_rate'),
                'non_visible_quality_hit_rate': baseline_evidence.get('non_visible_quality_hit_rate'),
            } if baseline_snapshot else None,
            'decision_memo_markdown': (
                '# Controlled live proof review\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n'
                f'- **Candidate:** {candidate_label}\n\n'
                '## Why this tranche exists\n'
                '- A proof window is only useful if it creates isolated evidence for that exact candidate.\n'
                '- This review filters proof-window-attributable runs rather than relying on contaminated current-version aggregates.\n'
                '- The result is meant to decide whether the candidate earned a formal next decision checkpoint, not to auto-promote anything.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, runs, resolved_rows, session, next_summary, baseline_snapshot)
        return payload

    def _build_pack(self, summary: dict, runs: list[dict], resolved_rows: list[dict], session: dict, next_summary: dict, baseline_snapshot: dict):
        proof_run_rows = []
        for run in runs:
            proof = dict(run.get('live_candidate_proof') or {})
            proof_run_rows.append({
                'run_id': run.get('run_id'),
                'scan_finished_utc': run.get('scan_finished_utc'),
                'evaluation_complete': run.get('evaluation_complete'),
                'visible_rows_count': run.get('visible_rows_count'),
                'suppressed_rows_count': run.get('suppressed_rows_count'),
                'market_regime_state': run.get('market_regime_state'),
                'proof_session_id': proof.get('proof_session_id'),
                'state_scope_key': proof.get('state_scope_key'),
                'model_bundle_label_override': proof.get('model_bundle_label_override'),
                'stage1_selection_mode_override': proof.get('stage1_selection_mode_override'),
                'stage1_max_candidates_override': proof.get('stage1_max_candidates_override'),
            })
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('live_candidate_proof_review_summary.json', _json_bytes(summary))
            zf.writestr('live_candidate_proof_review_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('proof_session.json', _json_bytes(session))
            zf.writestr('next_live_candidate_summary.json', _json_bytes(next_summary))
            zf.writestr('activation_baseline_current_version_summary.json', _json_bytes(baseline_snapshot))
            zf.writestr('proof_runs.csv', _csv_bytes(proof_run_rows, fieldnames=list(proof_run_rows[0].keys()) if proof_run_rows else ['run_id']))
            zf.writestr('proof_resolved_rows.csv', _csv_bytes(resolved_rows, fieldnames=ROW_CSV_FIELDS))
