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
from .runtime_scope import current_runtime_scope
from .version import APP_VERSION


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


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
    params = dict(payload.get('recommended_params') or {})
    if params:
        lines.append('Recommended params')
        for k, v in sorted(params.items()):
            lines.append(f"- {k}: {v}")
        lines.append('')
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
    return "\n".join(lines).strip() + "\n"


class UtilityTuningProofService:
    def __init__(self, config: AppConfig, review_packs: ReviewPackService):
        self.config = config
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / 'utility_tuning_proof')
        self.summary_path = self.root_dir / 'latest_utility_tuning_proof_summary.json'
        self.pack_path = self.root_dir / 'latest_utility_tuning_proof_pack.zip'
        self.session_path = self.root_dir / 'latest_utility_tuning_proof_session.json'
        self.tuning_summary_path = Path(config.model_dir) / 'utility_tuning_lab' / 'latest_utility_tuning_lab_summary.json'
        self.overrides_path = Path(config.model_dir) / 'runtime_live_overrides.json'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def latest_session(self) -> dict:
        return read_json(self.session_path, {})

    def _active_override(self) -> dict:
        raw = read_json(self.overrides_path, {})
        if not isinstance(raw, dict) or raw.get('source') != 'utility_tuning_proof':
            return {}
        if not bool(raw.get('proof_window_active')):
            return {}
        return raw

    def activate(self, *, proof_hours: int = 24) -> dict:
        tuning = read_json(self.tuning_summary_path, {})
        if str(tuning.get('verdict') or '') != 'utility_tuning_candidate_supported_offline':
            raise RuntimeError('Utility tuning proof cannot be activated because the tuning lab does not currently support an offline candidate.')
        params = dict(tuning.get('recommended_params') or {})
        if not params:
            raise RuntimeError('No recommended params are available from the utility tuning lab.')
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        try:
            baseline_current_version_summary = self.review_packs.get_current_version_summary() or {}
        except FileNotFoundError:
            baseline_current_version_summary = {}
        activated_at = _utc_now()
        expires_at = activated_at + timedelta(hours=max(1, int(proof_hours or 24)))
        proof_session_id = f"utility_proof_{activated_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        payload = {
            'source': 'utility_tuning_proof',
            'proof_window_active': True,
            'proof_session_id': proof_session_id,
            'activated_at_utc': activated_at.isoformat(),
            'expires_at_utc': expires_at.isoformat(),
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
            'utility_selection_engine_label': 'utility_constrained_proof_v1',
            **params,
            'baseline_current_version_summary': baseline_current_version_summary,
            'tuning_lab_generated_at_utc': tuning.get('generated_at_utc'),
            'tuning_lab_verdict': tuning.get('verdict'),
        }
        atomic_write_json(self.overrides_path, payload)
        atomic_write_json(self.session_path, payload)
        return self.build_summary(reason='activated')

    def clear(self) -> dict:
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        existing = self.latest_session() or {}
        payload = {
            'source': 'utility_tuning_proof',
            'proof_window_active': False,
            'proof_session_id': existing.get('proof_session_id'),
            'activated_at_utc': existing.get('activated_at_utc'),
            'expires_at_utc': existing.get('expires_at_utc'),
            'cleared_at_utc': _utc_now_iso(),
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
        }
        atomic_write_json(self.overrides_path, payload)
        if existing:
            existing.update({'proof_window_active': False, 'cleared_at_utc': payload['cleared_at_utc']})
            atomic_write_json(self.session_path, existing)
        return self.build_summary(reason='cleared')

    def build_summary(self, *, reason: str = 'refresh') -> dict:
        tuning = read_json(self.tuning_summary_path, {})
        active = self._active_override()
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        headline = 'No utility tuning candidate is ready for a controlled live proof'
        verdict = 'no_supported_utility_tuning_candidate'
        recommended_action = 'rerun_utility_tuning_lab_or_keep_current_live_settings'
        summary = 'The utility tuning lab has not yet produced a parameter bundle that clearly deserves a live proof window.'
        proof_window = {'active': bool(active), 'activated_at_utc': None, 'expires_at_utc': None, 'remaining_minutes': None}
        recommended_params = dict(tuning.get('recommended_params') or {})
        if active:
            expires_at = datetime.fromisoformat(str(active.get('expires_at_utc')).replace('Z', '+00:00')) if active.get('expires_at_utc') else None
            remaining = None if expires_at is None else max(0, int((expires_at - _utc_now()).total_seconds() // 60))
            proof_window.update({'activated_at_utc': active.get('activated_at_utc'), 'expires_at_utc': active.get('expires_at_utc'), 'remaining_minutes': remaining})
            headline = 'Controlled utility tuning proof window is active'
            verdict = 'utility_tuning_proof_window_active'
            recommended_action = 'collect_matured_evidence_for_the_exact_tuned_parameter_bundle'
            summary = 'The tuned utility parameter bundle is active for the current deployment scope in a bounded proof window.'
        elif str(tuning.get('verdict') or '') == 'utility_tuning_candidate_supported_offline' and recommended_params:
            headline = 'A tuned utility parameter bundle is ready for controlled live proof'
            verdict = 'utility_tuning_candidate_ready_for_live_proof'
            recommended_action = 'activate_a_controlled_live_proof_for_the_tuned_utility_settings'
            summary = 'The tuning lab produced a clearly better offline parameter bundle. The next step is a bounded live proof, not a casual live switch.'
        payload = {
            'available': True,
            'generated_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'reason': reason,
            'headline': headline,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'summary': summary,
            'current_scope': current_scope,
            'recommended_params': recommended_params or None,
            'recommended_env_overrides': dict(tuning.get('recommended_env_overrides') or {}),
            'tuning_summary': tuning,
            'proof_window': proof_window,
            'active_override': active or None,
            'decision_memo_markdown': (
                '# Controlled utility tuning proof\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n\n'
                '## Why this tranche exists\n'
                '- The utility selection semantics and tuning candidate are now offline concepts.\n'
                '- The next missing step is a bounded live proof for the exact tuned parameter bundle.\n'
                '- This tranche avoids manual env editing and preserves deployment-scope attribution.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, tuning)
        return payload

    def _build_pack(self, summary: dict, tuning: dict):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('utility_tuning_proof_summary.json', _json_bytes(summary))
            zf.writestr('utility_tuning_proof_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('utility_tuning_lab_summary.json', _json_bytes(tuning or {}))
            zf.writestr('utility_tuning_proof_session.json', _json_bytes(self.latest_session() or {}))
