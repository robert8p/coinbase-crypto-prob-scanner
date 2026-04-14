from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .review_runs import ReviewPackService
from .runtime_scope import current_runtime_scope, scope_key
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


def _summary_txt(payload: dict) -> str:
    lines = [
        f"Headline: {payload.get('headline') or '-'}",
        f"Verdict: {payload.get('verdict') or '-'}",
        f"Recommended action: {payload.get('recommended_action') or '-'}",
        f"Summary: {payload.get('summary') or '-'}",
        '',
    ]
    proof = dict(payload.get('proof_window') or {})
    if proof:
        lines.extend([
            'Proof window',
            f"- Active: {proof.get('active')}",
            f"- Activated at UTC: {proof.get('activated_at_utc') or '-'}",
            f"- Expires at UTC: {proof.get('expires_at_utc') or '-'}",
            f"- Remaining minutes: {proof.get('remaining_minutes') if proof.get('remaining_minutes') is not None else '-'}",
            '',
        ])
    candidate = dict(payload.get('recommended_candidate') or {})
    if candidate:
        lines.extend([
            'Recommended candidate',
            f"- Model kind: {candidate.get('model_kind') or '-'}",
            f"- Model source: {candidate.get('model_source') or '-'}",
            f"- Model bundle path: {candidate.get('model_bundle_path') or '-'}",
            f"- Stage 1 mode: {candidate.get('stage1_selection_mode') or '-'}",
            f"- Stage 1 max candidates: {candidate.get('stage1_max_candidates') or '-'}",
            f"- Raw threshold: {candidate.get('raw_threshold') or '-'}",
            '',
        ])
    evidence = dict(payload.get('current_scope_evidence') or {})
    if evidence:
        lines.extend([
            'Current-scope evidence snapshot',
            f"- Resolved rows: {evidence.get('resolved_rows') or '-'}",
            f"- Visible rows: {evidence.get('visible_rows') or '-'}",
            f"- Visible quality hit rate: {evidence.get('visible_quality_hit_rate') or '-'}",
            f"- Hidden quality hit rate: {evidence.get('non_visible_quality_hit_rate') or '-'}",
            '',
        ])
    return "\n".join(lines).strip() + "\n"


def _override_matches_current_scope(model_dir: str | Path, raw: dict, *, app_version: str = APP_VERSION) -> bool:
    current_scope = current_runtime_scope(model_dir, app_version=app_version)
    current_scope_key = current_scope.get('state_scope_key')
    override_scope_key = raw.get('state_scope_key') or scope_key(raw.get('app_version'), raw.get('deployed_since_utc'))
    if current_scope_key and override_scope_key and current_scope_key != override_scope_key:
        return False
    return True


def load_active_live_candidate_proof_override(model_dir: str | Path, *, app_version: str = APP_VERSION) -> dict:
    path = Path(model_dir) / 'runtime_live_overrides.json'
    raw = read_json(path, {})
    if not isinstance(raw, dict) or raw.get('source') not in {'live_candidate_proof', 'utility_model_proof'}:
        return {}
    if not bool(raw.get('proof_window_active')):
        return {}
    if not _override_matches_current_scope(model_dir, raw, app_version=app_version):
        return {}
    expires_at = _parse_iso(raw.get('expires_at_utc'))
    if expires_at is not None and expires_at <= _utc_now():
        return {}
    return raw


def load_active_live_candidate_override(model_dir: str | Path, *, app_version: str = APP_VERSION) -> dict:
    path = Path(model_dir) / 'runtime_live_overrides.json'
    raw = read_json(path, {})
    if not isinstance(raw, dict):
        return {}
    source = str(raw.get('source') or '')
    if source in {'live_candidate_proof', 'utility_model_proof'}:
        return load_active_live_candidate_proof_override(model_dir, app_version=app_version)
    if source != 'live_candidate_adoption':
        return {}
    if not bool(raw.get('adopted_live_candidate_active')):
        return {}
    if not _override_matches_current_scope(model_dir, raw, app_version=app_version):
        return {}
    return raw


class LiveCandidateProofService:
    def __init__(self, config: AppConfig, review_packs: ReviewPackService):
        self.config = config
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / 'live_candidate_proof')
        self.summary_path = self.root_dir / 'latest_live_candidate_proof_summary.json'
        self.pack_path = self.root_dir / 'latest_live_candidate_proof_pack.zip'
        self.next_candidate_summary_path = Path(config.model_dir) / 'next_live_candidate_lab' / 'latest_next_live_candidate_lab_summary.json'
        self.overrides_path = Path(config.model_dir) / 'runtime_live_overrides.json'
        self.session_path = self.root_dir / 'latest_live_candidate_proof_session.json'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def latest_session(self) -> dict:
        return read_json(self.session_path, {})

    def activate(self, *, proof_hours: int = 24) -> dict:
        next_summary = read_json(self.next_candidate_summary_path, {})
        verdict = str(next_summary.get('verdict') or '')
        candidate = dict(next_summary.get('recommended_candidate') or {})
        if verdict != 'single_live_candidate_supported_offline' or not candidate:
            raise RuntimeError('No exact next live candidate is currently supported. Run the Next Live Candidate Lab first and require a single supported candidate.')
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        try:
            baseline_current_version_summary = self.review_packs.get_current_version_summary() or {}
        except FileNotFoundError:
            baseline_current_version_summary = {}
        proof_hours = max(1, int(proof_hours or 24))
        activated_at = _utc_now()
        expires_at = activated_at + timedelta(hours=proof_hours)
        proof_session_id = f"proof_{activated_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        payload = {
            'source': 'live_candidate_proof',
            'proof_window_active': True,
            'proof_session_id': proof_session_id,
            'activated_at_utc': activated_at.isoformat(),
            'expires_at_utc': expires_at.isoformat(),
            'proof_hours': proof_hours,
            'note': 'Scoped live proof window activated from next-live-candidate lab recommendation.',
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
            'recommended_candidate': candidate,
            'baseline_candidate': dict(next_summary.get('live_baseline') or {}),
            'baseline_current_version_summary': baseline_current_version_summary,
            'next_live_candidate_generated_at_utc': next_summary.get('generated_at_utc'),
            'next_live_candidate_verdict': verdict,
            'model_bundle_path_override': str(candidate.get('model_bundle_path') or self.config.model_path_pt2),
            'model_bundle_label_override': str(candidate.get('model_source') or candidate.get('model_kind') or 'candidate'),
            'stage1_selection_mode_override': str(candidate.get('stage1_selection_mode') or self.config.stage1_selection_mode),
            'stage1_max_candidates_override': int(candidate.get('stage1_max_candidates') or self.config.stage1_max_candidates),
        }
        session = {
            **payload,
            'candidate_label': str(candidate.get('model_source') or candidate.get('model_kind') or 'candidate'),
            'session_status': 'active',
        }
        atomic_write_json(self.overrides_path, payload)
        atomic_write_json(self.session_path, session)
        return self.build_summary(reason='activated')

    def clear(self) -> dict:
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        existing_session = self.latest_session()
        cleared_at = _utc_now_iso()
        payload = {
            'source': 'live_candidate_proof',
            'proof_window_active': False,
            'proof_session_id': existing_session.get('proof_session_id'),
            'cleared_at_utc': cleared_at,
            'note': 'Live candidate proof window cleared by operator action.',
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
            'model_bundle_path_override': None,
            'model_bundle_label_override': None,
            'stage1_selection_mode_override': None,
            'stage1_max_candidates_override': None,
        }
        if existing_session:
            existing_session.update({
                'proof_window_active': False,
                'cleared_at_utc': cleared_at,
                'session_status': 'cleared',
            })
            atomic_write_json(self.session_path, existing_session)
        atomic_write_json(self.overrides_path, payload)
        return self.build_summary(reason='cleared')

    def build_summary(self, *, reason: str = 'refresh') -> dict:
        next_summary = read_json(self.next_candidate_summary_path, {})
        recommended_candidate = dict(next_summary.get('recommended_candidate') or {})
        next_verdict = str(next_summary.get('verdict') or '')
        raw_overrides = read_json(self.overrides_path, {})
        active_override = load_active_live_candidate_proof_override(self.config.model_dir, app_version=APP_VERSION)
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        now = _utc_now()
        try:
            current_version = self.review_packs.get_current_version_summary() or {}
        except FileNotFoundError:
            current_version = {}
        evidence = dict(current_version.get('evidence') or {})
        proof_window = {
            'active': bool(active_override),
            'activated_at_utc': active_override.get('activated_at_utc') if active_override else None,
            'expires_at_utc': active_override.get('expires_at_utc') if active_override else None,
            'remaining_minutes': None,
            'state_scope_key': active_override.get('state_scope_key') if active_override else current_scope.get('state_scope_key'),
            'reason': reason,
        }
        if active_override:
            expires_at = _parse_iso(active_override.get('expires_at_utc'))
            if expires_at is not None:
                proof_window['remaining_minutes'] = max(0, int((expires_at - now).total_seconds() // 60))
        expired_override = False
        if isinstance(raw_overrides, dict) and raw_overrides.get('source') == 'live_candidate_proof' and bool(raw_overrides.get('proof_window_active')) and not active_override:
            expired_override = True

        headline = 'No exact live candidate is ready for a proof window'
        verdict = 'no_supported_live_candidate'
        recommended_action = 'rerun_next_live_candidate_lab'
        summary = 'The next-live-candidate lab has not yet produced a single exact candidate that deserves a controlled live proof window.'
        if active_override:
            headline = 'Controlled live candidate proof window is active'
            verdict = 'live_candidate_proof_window_active'
            recommended_action = 'monitor_current_scope_and_collect_matured_evidence'
            summary = 'A single exact candidate is active in the live scanner for a scoped proof window. Keep all other live semantics fixed and let current-scope evidence mature.'
        elif expired_override:
            headline = 'Controlled live candidate proof window has expired'
            verdict = 'live_candidate_proof_window_expired'
            recommended_action = 'review_current_scope_evidence_then_clear_or_reactivate'
            summary = 'A prior proof-window override exists on disk but is no longer active. Review current-scope evidence before reactivating anything.'
        elif next_verdict == 'single_live_candidate_supported_offline' and recommended_candidate:
            headline = 'A single exact live candidate is ready for controlled proof'
            verdict = 'live_candidate_ready_for_activation'
            recommended_action = 'activate_controlled_live_candidate_proof_window'
            summary = 'Offline evidence supports one exact live candidate. Activate a scoped proof window rather than making an open-ended live change.'

        current_scope_evidence = {
            'resolved_rows': int(evidence.get('resolved_rows') or current_version.get('resolved_visible_rows') or 0),
            'visible_rows': int(evidence.get('visible_rows') or 0),
            'visible_quality_hit_rate': evidence.get('visible_quality_hit_rate'),
            'non_visible_quality_hit_rate': evidence.get('non_visible_quality_hit_rate'),
        }
        payload = {
            'available': True,
            'generated_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'headline': headline,
            'summary': summary,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'reason': reason,
            'proof_window': proof_window,
            'current_scope': current_scope,
            'recommended_candidate': recommended_candidate or None,
            'live_baseline': dict(next_summary.get('live_baseline') or {}),
            'next_live_candidate_verdict': next_verdict or None,
            'next_live_candidate_generated_at_utc': next_summary.get('generated_at_utc'),
            'activation_supported': bool(next_verdict == 'single_live_candidate_supported_offline' and recommended_candidate),
            'active_override': active_override or None,
            'current_scope_evidence': current_scope_evidence,
            'decision_memo_markdown': (
                '# Controlled live candidate proof window\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n'
                f'- **Summary:** {summary}\n\n'
                '## Why this tranche exists\n'
                '- The previous offline tranche can identify at most one exact next live candidate.\n'
                '- The next step should be a scoped proof window, not an open-ended live change.\n'
                '- This harness applies only the exact candidate recommendation and keeps the rest of live semantics fixed.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, next_summary, raw_overrides)
        return payload

    def _build_pack(self, summary: dict, next_summary: dict, raw_overrides: dict):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('live_candidate_proof_summary.json', _json_bytes(summary))
            zf.writestr('live_candidate_proof_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('next_live_candidate_lab_summary_snapshot.json', _json_bytes(next_summary))
            zf.writestr('runtime_live_overrides_snapshot.json', _json_bytes(raw_overrides if isinstance(raw_overrides, dict) else {}))
