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
from .utility_shortlist import load_active_utility_tuning_override
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
    cand = dict(payload.get('candidate_model') or {})
    if cand:
        lines.extend([
            'Candidate model',
            f"- Path: {cand.get('path') or '-'}",
            f"- Label: {cand.get('label') or '-'}",
            '',
        ])
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


class UtilityModelProofService:
    def __init__(self, config: AppConfig, review_packs: ReviewPackService):
        self.config = config
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / 'utility_model_proof')
        self.summary_path = self.root_dir / 'latest_utility_model_proof_summary.json'
        self.pack_path = self.root_dir / 'latest_utility_model_proof_pack.zip'
        self.session_path = self.root_dir / 'latest_utility_model_proof_session.json'
        self.lab_summary_path = Path(config.model_dir) / 'utility_model_lab' / 'latest_utility_model_lab_summary.json'
        self.overrides_path = Path(config.model_dir) / 'runtime_live_overrides.json'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def latest_session(self) -> dict:
        return read_json(self.session_path, {})

    def _active_override(self) -> dict:
        raw = read_json(self.overrides_path, {})
        if not isinstance(raw, dict) or raw.get('source') != 'utility_model_proof':
            return {}
        if not bool(raw.get('proof_window_active')):
            return {}
        return raw

    def activate(self, *, proof_hours: int = 24) -> dict:
        lab = read_json(self.lab_summary_path, {})
        if str(lab.get('verdict') or '') != 'utility_model_candidate_supported_offline':
            raise RuntimeError('Utility model proof cannot be activated because the utility model lab does not currently support an offline candidate.')
        candidate_path = str(lab.get('candidate_model_path') or '').strip()
        if not candidate_path:
            raise RuntimeError('No candidate model path is available from the utility model lab.')
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        try:
            baseline_current_version_summary = self.review_packs.get_current_version_summary() or {}
        except FileNotFoundError:
            baseline_current_version_summary = {}
        utility_override = load_active_utility_tuning_override(self.config.model_dir)
        activated_at = _utc_now()
        expires_at = activated_at + timedelta(hours=max(1, int(proof_hours or 24)))
        proof_session_id = f"utility_model_proof_{activated_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        payload = {
            'source': 'utility_model_proof',
            'proof_window_active': True,
            'proof_session_id': proof_session_id,
            'activated_at_utc': activated_at.isoformat(),
            'expires_at_utc': expires_at.isoformat(),
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
            'model_bundle_path_override': candidate_path,
            'model_bundle_label_override': 'utility_model_candidate',
            'baseline_current_version_summary': baseline_current_version_summary,
            'utility_model_lab_generated_at_utc': lab.get('generated_at_utc'),
            'utility_model_lab_verdict': lab.get('verdict'),
            'utility_model_candidate_metadata': dict(lab.get('candidate_model_metadata') or {}),
            # carry current utility semantics so only the model changes during proof
            'utility_selection_engine_label': str((utility_override or {}).get('utility_selection_engine_label') or getattr(self.config, 'utility_selection_engine_label', 'utility_constrained_v1')),
            'utility_expected_edge_weight': float((utility_override or {}).get('utility_expected_edge_weight') or getattr(self.config, 'utility_expected_edge_weight', 0.45)),
            'utility_confidence_weight': float((utility_override or {}).get('utility_confidence_weight') or getattr(self.config, 'utility_confidence_weight', 0.30)),
            'utility_probability_weight': float((utility_override or {}).get('utility_probability_weight') or getattr(self.config, 'utility_probability_weight', 0.25)),
            'utility_shortlist_target_max_names': int((utility_override or {}).get('utility_shortlist_target_max_names') or getattr(self.config, 'utility_shortlist_target_max_names', 8)),
            'utility_shortlist_score_floor': float((utility_override or {}).get('utility_shortlist_score_floor') or getattr(self.config, 'utility_shortlist_score_floor', 0.52)),
            'utility_shortlist_score_dropoff': float((utility_override or {}).get('utility_shortlist_score_dropoff') or getattr(self.config, 'utility_shortlist_score_dropoff', 0.16)),
            'utility_confidence_floor': float((utility_override or {}).get('utility_confidence_floor') or getattr(self.config, 'utility_confidence_floor', 0.35)),
            'utility_tier3_max_frac': float((utility_override or {}).get('utility_tier3_max_frac') or getattr(self.config, 'utility_tier3_max_frac', 0.25)),
        }
        atomic_write_json(self.overrides_path, payload)
        atomic_write_json(self.session_path, payload)
        return self.build_summary(reason='activated')

    def clear(self) -> dict:
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        existing = self.latest_session() or {}
        payload = {
            'source': 'utility_model_proof',
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
        lab = read_json(self.lab_summary_path, {})
        active = self._active_override()
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        headline = 'No utility-model challenger is ready for a controlled live proof'
        verdict = 'no_supported_utility_model_candidate'
        recommended_action = 'rerun_utility_model_lab_or_keep_current_live_model'
        summary = 'The utility model lab has not yet produced a model candidate that clearly deserves a live proof window.'
        proof_window = {'active': bool(active), 'activated_at_utc': None, 'expires_at_utc': None, 'remaining_minutes': None}
        candidate_model = {
            'path': lab.get('candidate_model_path'),
            'label': 'utility_model_candidate',
        } if lab.get('candidate_model_path') else None
        if active:
            expires_at = datetime.fromisoformat(str(active.get('expires_at_utc')).replace('Z', '+00:00')) if active.get('expires_at_utc') else None
            remaining = None if expires_at is None else max(0, int((expires_at - _utc_now()).total_seconds() // 60))
            proof_window.update({'activated_at_utc': active.get('activated_at_utc'), 'expires_at_utc': active.get('expires_at_utc'), 'remaining_minutes': remaining})
            headline = 'Controlled utility-model proof window is active'
            verdict = 'utility_model_proof_window_active'
            recommended_action = 'collect_matured_evidence_for_the_exact_utility_model_candidate'
            summary = 'The utility-model challenger is active for the current deployment scope in a bounded proof window, while current utility semantics are held constant.'
        elif str(lab.get('verdict') or '') == 'utility_model_candidate_supported_offline' and candidate_model:
            headline = 'A utility-model challenger is ready for controlled live proof'
            verdict = 'utility_model_candidate_ready_for_live_proof'
            recommended_action = 'activate_a_controlled_live_proof_for_the_utility_model_candidate'
            summary = 'The utility-model challenger improved shortlist utility strongly enough offline to justify a bounded live proof.'
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
            'candidate_model': candidate_model,
            'utility_model_lab_summary': lab,
            'proof_window': proof_window,
            'active_override': active or None,
            'decision_memo_markdown': (
                '# Controlled utility-model proof\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n\n'
                '## Why this tranche exists\n'
                '- The app can now train a utility-aligned challenger offline.\n'
                '- The next missing step is isolated live proof for the exact utility-model candidate.\n'
                '- This tranche changes the model while preserving current utility selection semantics.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, lab)
        return payload

    def _build_pack(self, summary: dict, lab: dict):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('utility_model_proof_summary.json', _json_bytes(summary))
            zf.writestr('utility_model_proof_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('utility_model_lab_summary.json', _json_bytes(lab or {}))
            zf.writestr('utility_model_proof_session.json', _json_bytes(self.latest_session() or {}))
