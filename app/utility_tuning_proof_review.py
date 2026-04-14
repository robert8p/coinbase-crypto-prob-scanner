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


def _summary_txt(payload: dict) -> str:
    lines = [
        f"Headline: {payload.get('headline') or '-'}",
        f"Verdict: {payload.get('verdict') or '-'}",
        f"Recommended action: {payload.get('recommended_action') or '-'}",
        f"Summary: {payload.get('summary') or '-'}",
        '',
    ]
    proof = dict(payload.get('proof_evidence') or {})
    if proof:
        lines.extend([
            'Proof-window evidence',
            f"- Resolved rows: {proof.get('resolved_rows') or 0}",
            f"- Visible rows: {proof.get('visible_rows') or 0}",
            f"- Hidden rows: {proof.get('hidden_rows') or 0}",
            f"- Visible quality hit rate: {proof.get('visible_quality_hit_rate')}",
            f"- Hidden quality hit rate: {proof.get('hidden_quality_hit_rate')}",
            f"- Visible-hidden gap: {proof.get('visible_hidden_gap')}",
            '',
        ])
    return "\n".join(lines).strip() + "\n"


class UtilityTuningProofReviewService:
    def __init__(self, config: AppConfig, review_packs: ReviewPackService):
        self.config = config
        self.review_packs = review_packs
        self.proof_root = ensure_dir(Path(config.model_dir) / 'utility_tuning_proof')
        self.root_dir = ensure_dir(Path(config.model_dir) / 'utility_tuning_proof_review')
        self.summary_path = self.root_dir / 'latest_utility_tuning_proof_review_summary.json'
        self.pack_path = self.root_dir / 'latest_utility_tuning_proof_review_pack.zip'
        self.session_path = self.proof_root / 'latest_utility_tuning_proof_session.json'
        self.proof_summary_path = self.proof_root / 'latest_utility_tuning_proof_summary.json'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _load_session(self) -> dict:
        return read_json(self.session_path, {})

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
        target_session = str(session.get('proof_session_id') or '').strip()
        out = []
        for row in rows:
            item = dict(row)
            status = read_json(item.get('review_status_path'), {})
            proof = dict(status.get('live_utility_tuning_proof') or {})
            if not proof:
                continue
            if target_session and str(proof.get('proof_session_id') or '').strip() != target_session:
                continue
            item['live_utility_tuning_proof'] = proof
            out.append(item)
        return out

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
            'scan_id': str(r.get('run_id') or ''),
            'row_type': str(r.get('row_type') or ''),
            'score': _f(r.get('utility_decision_score'), _f(r.get('live_score'), 0.0)) or 0.0,
            'y': int(r.get('quality_touched') or 0),
        } for r in rows])
        if frame.empty:
            return empty
        frame = frame.sort_values(['scan_id', 'score'], ascending=[True, False]).reset_index(drop=True)
        base_event_rate = float(frame['y'].mean()) if len(frame) else 0.0
        scan_count = int(frame['scan_id'].nunique())
        scans_with_visible = 0
        pairwise_wins = 0.0
        pairwise_comparable = 0
        visible_counts=[]; visible_rates=[]; hidden_rates=[]; gaps=[]; top1=[]; top3=[]
        for _, scan in frame.groupby('scan_id', sort=False):
            visible = scan[scan['row_type']=='visible']
            hidden = scan[scan['row_type']!='visible']
            visible_counts.append(int(len(visible)))
            if not visible.empty:
                scans_with_visible += 1
                vr = float(visible['y'].mean())
                visible_rates.append(vr)
                top1.append(float(visible.iloc[:1]['y'].mean()))
                top3.append(float(visible.iloc[: min(3, len(visible))]['y'].mean()))
                if not hidden.empty:
                    hr = float(hidden['y'].mean())
                    hidden_rates.append(hr)
                    gap = vr - hr
                    gaps.append(gap)
                    pairwise_comparable += 1
                    if gap > 0:
                        pairwise_wins += 1.0
                    elif gap == 0:
                        pairwise_wins += 0.5
        avg_visible = float(pd.Series(visible_counts).mean()) if visible_counts else None
        visible_mean = float(pd.Series(visible_rates).mean()) if visible_rates else None
        hidden_mean = float(pd.Series(hidden_rates).mean()) if hidden_rates else None
        mean_gap = float(pd.Series(gaps).mean()) if gaps else None
        pairwise = float(pairwise_wins)/float(pairwise_comparable) if pairwise_comparable else None
        top1_mean = float(pd.Series(top1).mean()) if top1 else None
        top3_mean = float(pd.Series(top3).mean()) if top3 else None
        overwide = max(0.0, (avg_visible or 0.0) - 5.0) / 5.0 if avg_visible is not None else None
        utility_score = None
        if mean_gap is not None:
            utility_score = mean_gap + 0.25 * (((pairwise if pairwise is not None else 0.5) - 0.5)) + 0.10 * (((top1_mean if top1_mean is not None else base_event_rate) - base_event_rate)) + 0.05 * (((top3_mean if top3_mean is not None else base_event_rate) - base_event_rate)) - 0.02 * (overwide or 0.0)
        return {
            'scan_shortlist_scans': scan_count,
            'scan_shortlist_scans_with_visible': scans_with_visible,
            'scan_shortlist_avg_visible_rows_per_scan': round(avg_visible, 6) if avg_visible is not None else None,
            'scan_shortlist_visible_quality_rate_mean': round(visible_mean, 6) if visible_mean is not None else None,
            'scan_shortlist_hidden_quality_rate_mean': round(hidden_mean, 6) if hidden_mean is not None else None,
            'scan_shortlist_mean_gap': round(mean_gap, 6) if mean_gap is not None else None,
            'scan_shortlist_pairwise_win_rate': round(pairwise, 6) if pairwise is not None else None,
            'scan_shortlist_pairwise_comparable_scans': int(pairwise_comparable),
            'scan_shortlist_top1_visible_quality': round(top1_mean, 6) if top1_mean is not None else None,
            'scan_shortlist_top3_visible_quality': round(top3_mean, 6) if top3_mean is not None else None,
            'scan_shortlist_overwide_penalty': round(overwide, 6) if overwide is not None else None,
            'scan_shortlist_utility_score': round(utility_score, 6) if utility_score is not None else None,
        }

    def build_summary(self, *, reason: str = 'refresh') -> dict:
        session = self._load_session()
        proof_summary = read_json(self.proof_summary_path, {})
        if not session:
            payload = {
                'available': False,
                'generated_at_utc': _utc_now_iso(),
                'app_version': APP_VERSION,
                'headline': 'No retained utility tuning proof session is available',
                'verdict': 'no_utility_tuning_proof_session',
                'recommended_action': 'activate_a_controlled_utility_tuning_proof_first',
                'summary': 'There is no retained utility tuning proof session on disk, so no isolated proof-window evidence can be reviewed yet.',
                'reason': reason,
            }
            atomic_write_json(self.summary_path, payload)
            self._build_pack(payload, {}, [], [])
            return payload
        runs = self._matching_runs(session)
        evaluated_runs = [r for r in runs if bool(r.get('evaluation_complete'))]
        resolved_rows = self.review_packs._load_rows_for_run_ids([str(r.get('run_id')) for r in evaluated_runs], resolved_only=True) if evaluated_runs else []
        visible_rows = [r for r in resolved_rows if str(r.get('row_type') or '') == 'visible']
        hidden_rows = [r for r in resolved_rows if str(r.get('row_type') or '') != 'visible']
        visible_summary = self.review_packs._bucket_summary(visible_rows) if visible_rows else {}
        hidden_summary = self.review_packs._bucket_summary(hidden_rows) if hidden_rows else {}
        gap = None if _f(visible_summary.get('quality_hit_rate')) is None or _f(hidden_summary.get('quality_hit_rate')) is None else round((_f(visible_summary.get('quality_hit_rate'),0.0) or 0.0) - (_f(hidden_summary.get('quality_hit_rate'),0.0) or 0.0), 6)
        utility = self._scan_shortlist_utility(resolved_rows)
        baseline_snapshot = dict(session.get('baseline_current_version_summary') or {})
        base_evidence = dict(baseline_snapshot.get('evidence') or {})
        base_gap = None if _f(base_evidence.get('visible_quality_hit_rate')) is None or _f(base_evidence.get('non_visible_quality_hit_rate')) is None else round((_f(base_evidence.get('visible_quality_hit_rate'),0.0) or 0.0) - (_f(base_evidence.get('non_visible_quality_hit_rate'),0.0) or 0.0), 6)
        deltas = {
            'visible_quality_hit_rate_delta_vs_activation': None if _f(visible_summary.get('quality_hit_rate')) is None or _f(base_evidence.get('visible_quality_hit_rate')) is None else round((_f(visible_summary.get('quality_hit_rate'),0.0) or 0.0) - (_f(base_evidence.get('visible_quality_hit_rate'),0.0) or 0.0), 6),
            'visible_hidden_gap_delta_vs_activation': None if gap is None or base_gap is None else round(gap - base_gap, 6),
        }
        proof_evidence = {
            'resolved_rows': len(resolved_rows),
            'visible_rows': len(visible_rows),
            'hidden_rows': len(hidden_rows),
            'visible_quality_hit_rate': visible_summary.get('quality_hit_rate'),
            'hidden_quality_hit_rate': hidden_summary.get('quality_hit_rate'),
            'visible_hidden_gap': gap,
        }
        headline = 'Controlled utility tuning proof is waiting for evidence'
        verdict = 'waiting_for_utility_tuning_proof_evidence'
        recommended_action = 'keep_collecting_isolated_evidence_for_the_tuned_bundle'
        summary = 'The proof window for the tuned utility parameters has not yet matured enough for a decision-grade verdict.'
        visible_n = len(visible_rows)
        utility_score = _f(utility.get('scan_shortlist_utility_score'))
        if not runs:
            headline = 'No matching utility proof-window runs have been captured yet'
            verdict = 'no_matching_utility_proof_runs'
            recommended_action = 'wait_for_scans_under_the_tuned_utility_bundle'
            summary = 'The tuned utility proof session exists, but no matching attributed runs have been recorded yet.'
        elif visible_n < 30:
            headline = 'Controlled utility tuning proof is collecting evidence'
            verdict = 'waiting_for_more_resolved_visible_rows'
            recommended_action = 'allow_more_tuned_utility_runs_to_mature'
            summary = 'The proof window has matching runs, but not enough resolved visible rows yet.'
        elif gap is not None and gap > 0 and (utility_score is None or utility_score > 0):
            headline = 'Controlled utility tuning proof currently supports the tuned bundle'
            verdict = 'utility_tuning_proof_supports_candidate'
            recommended_action = 'prepare_a_controlled_live_update_of_the_utility_settings'
            summary = 'The tuned utility parameter bundle is beating the hidden remainder in isolated live evidence.'
        elif gap is not None and gap < 0 and (utility_score is None or utility_score < 0):
            headline = 'Controlled utility tuning proof currently rejects the tuned bundle'
            verdict = 'utility_tuning_proof_rejects_candidate'
            recommended_action = 'clear_the_utility_tuning_proof_and_keep_current_live_settings'
            summary = 'The tuned utility parameter bundle is not beating the hidden remainder in isolated live evidence.'
        else:
            headline = 'Controlled utility tuning proof remains mixed'
            verdict = 'utility_tuning_proof_inconclusive'
            recommended_action = 'continue_the_proof_or_keep_current_live_settings'
            summary = 'The tuned utility parameter bundle has some isolated live evidence, but it is not decisive yet.'
        payload = {
            'available': True,
            'generated_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'reason': reason,
            'headline': headline,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'summary': summary,
            'proof_session': session,
            'proof_runs': {'matching_runs': len(runs), 'evaluated_runs': len(evaluated_runs)},
            'proof_evidence': proof_evidence,
            'scan_shortlist_utility': utility,
            'baseline_current_version_evidence_at_activation': {
                'resolved_rows': int(base_evidence.get('resolved_rows') or 0),
                'visible_rows': int(base_evidence.get('visible_rows') or 0),
                'visible_quality_hit_rate': base_evidence.get('visible_quality_hit_rate'),
                'non_visible_quality_hit_rate': base_evidence.get('non_visible_quality_hit_rate'),
            } if baseline_snapshot else None,
            'deltas_vs_activation_baseline': deltas,
            'proof_summary': proof_summary,
            'decision_memo_markdown': (
                '# Controlled utility tuning proof review\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n\n'
                '## Why this tranche exists\n'
                '- The tuning lab can find a better offline utility bundle.\n'
                '- The next missing step is isolated live proof for that exact parameter bundle.\n'
                '- This review decides whether the tuned bundle deserves a real live update.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, session, runs, evaluated_runs)
        return payload

    def _build_pack(self, summary: dict, session: dict, runs: list[dict], evaluated_runs: list[dict]):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('utility_tuning_proof_review_summary.json', _json_bytes(summary))
            zf.writestr('utility_tuning_proof_review_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('utility_tuning_proof_session.json', _json_bytes(session or {}))
            zf.writestr('matching_utility_tuning_proof_runs.json', _json_bytes(runs or []))
            zf.writestr('evaluated_utility_tuning_proof_runs.json', _json_bytes(evaluated_runs or []))
