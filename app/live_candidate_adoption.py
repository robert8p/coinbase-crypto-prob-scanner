from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
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
    cand = dict(payload.get('candidate') or {})
    if cand:
        lines.extend([
            'Candidate',
            f"- Model source: {cand.get('model_source') or '-'}",
            f"- Model kind: {cand.get('model_kind') or '-'}",
            f"- Model bundle path: {cand.get('model_bundle_path') or '-'}",
            f"- Stage 1 mode: {cand.get('stage1_selection_mode') or '-'}",
            f"- Stage 1 max candidates: {cand.get('stage1_max_candidates') or '-'}",
            f"- Raw threshold: {cand.get('raw_threshold') or '-'}",
            '',
        ])
    proof = dict(payload.get('proof_review') or {})
    if proof:
        evidence = dict(proof.get('proof_evidence') or {})
        utility = dict(proof.get('scan_shortlist_utility') or {})
        lines.extend([
            'Isolated proof review',
            f"- Verdict: {proof.get('verdict') or '-'}",
            f"- Visible rows: {evidence.get('visible_rows') or 0}",
            f"- Visible-hidden gap: {evidence.get('visible_hidden_gap') or '-'}",
            f"- Utility score: {utility.get('scan_shortlist_utility_score') if utility.get('scan_shortlist_utility_score') is not None else '-'}",
            '',
        ])
    adoption = dict(payload.get('active_adoption') or {})
    if adoption:
        lines.extend([
            'Active adoption override',
            f"- Active: {adoption.get('active')}",
            f"- Adoption session id: {adoption.get('adoption_session_id') or '-'}",
            f"- Adopted at UTC: {adoption.get('adopted_at_utc') or '-'}",
            '',
        ])
    return "\n".join(lines).strip() + "\n"


class LiveCandidateAdoptionService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.root_dir = ensure_dir(Path(config.model_dir) / 'live_candidate_adoption')
        self.summary_path = self.root_dir / 'latest_live_candidate_adoption_summary.json'
        self.pack_path = self.root_dir / 'latest_live_candidate_adoption_pack.zip'
        self.state_path = self.root_dir / 'latest_live_candidate_adoption_state.json'
        self.overrides_path = Path(config.model_dir) / 'runtime_live_overrides.json'
        self.next_summary_path = Path(config.model_dir) / 'next_live_candidate_lab' / 'latest_next_live_candidate_lab_summary.json'
        self.proof_summary_path = Path(config.model_dir) / 'live_candidate_proof' / 'latest_live_candidate_proof_summary.json'
        self.proof_review_path = Path(config.model_dir) / 'live_candidate_proof_review' / 'latest_live_candidate_proof_review_summary.json'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _load_override(self) -> dict:
        raw = read_json(self.overrides_path, {})
        if not isinstance(raw, dict) or raw.get('source') != 'live_candidate_adoption':
            return {}
        if not bool(raw.get('adopted_live_candidate_active')):
            return {}
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        current_scope_key = current_scope.get('state_scope_key')
        override_scope_key = raw.get('state_scope_key') or scope_key(raw.get('app_version'), raw.get('deployed_since_utc'))
        if current_scope_key and override_scope_key and current_scope_key != override_scope_key:
            return {}
        out = dict(raw)
        out['active'] = True
        return out

    def _candidate(self, next_summary: dict, proof_summary: dict, proof_review: dict) -> dict:
        cand = dict(next_summary.get('recommended_candidate') or {})
        if cand:
            return cand
        cand = dict(proof_summary.get('recommended_candidate') or {})
        if cand:
            return cand
        return dict(proof_review.get('recommended_candidate') or {})

    def build_summary(self, *, reason: str = 'refresh') -> dict:
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        next_summary = read_json(self.next_summary_path, {})
        proof_summary = read_json(self.proof_summary_path, {})
        proof_review = read_json(self.proof_review_path, {})
        active_adoption = self._load_override()
        candidate = self._candidate(next_summary, proof_summary, proof_review)
        proof_evidence = dict(proof_review.get('proof_evidence') or {})
        proof_utility = dict(proof_review.get('scan_shortlist_utility') or {})
        baseline_at_activation = dict(proof_review.get('baseline_current_version_evidence_at_activation') or {})
        proof_visible_q = _f(proof_evidence.get('visible_quality_hit_rate'))
        proof_hidden_q = _f(proof_evidence.get('hidden_quality_hit_rate'))
        proof_gap = _f(proof_evidence.get('visible_hidden_gap'))
        proof_visible_rows = int(proof_evidence.get('visible_rows') or 0)
        proof_utility_score = _f(proof_utility.get('scan_shortlist_utility_score'))
        base_visible_q = _f(baseline_at_activation.get('visible_quality_hit_rate'))
        base_hidden_q = _f(baseline_at_activation.get('non_visible_quality_hit_rate'))
        base_gap = None
        if base_visible_q is not None and base_hidden_q is not None:
            base_gap = round(base_visible_q - base_hidden_q, 6)
        deltas = {
            'visible_quality_hit_rate_delta_vs_activation': round(proof_visible_q - base_visible_q, 6) if proof_visible_q is not None and base_visible_q is not None else None,
            'visible_hidden_gap_delta_vs_activation': round(proof_gap - base_gap, 6) if proof_gap is not None and base_gap is not None else None,
        }

        headline = 'No candidate is ready for controlled adoption'
        verdict = 'no_candidate_ready_for_adoption'
        recommended_action = 'keep_current_live_path_unchanged'
        summary = 'The app does not yet have the combined offline + isolated live proof evidence needed for an adoption decision.'

        next_verdict = str(next_summary.get('verdict') or '')
        proof_verdict = str(proof_review.get('verdict') or '')
        if active_adoption:
            headline = 'Controlled live candidate adoption is active'
            verdict = 'live_candidate_adoption_active'
            recommended_action = 'monitor_current_scope_and_preserve_rollback_path'
            summary = 'An evidence-gated adopted candidate is active for the current deployment scope. Preserve the rollback path and continue collecting current-scope evidence.'
        elif next_verdict != 'single_live_candidate_supported_offline' or not candidate:
            headline = 'No exact candidate has cleared the offline adoption gate'
            verdict = 'offline_candidate_not_ready_for_adoption'
            recommended_action = 'rerun_next_live_candidate_lab_or_keep_live_unchanged'
            summary = 'The app does not currently have one exact offline candidate that deserves adoption consideration.'
        elif proof_verdict in {'no_live_candidate_proof_session', 'no_matching_proof_window_runs', 'waiting_for_more_resolved_visible_rows', 'waiting_for_proof_window_evidence'}:
            headline = 'Adoption gate is waiting for more isolated live proof evidence'
            verdict = 'waiting_for_more_live_proof_evidence'
            recommended_action = 'keep_or_complete_the_proof_window_before_any_adoption'
            summary = 'The candidate may be promising offline, but the isolated live proof window has not matured enough to justify adoption.'
        elif proof_verdict == 'live_proof_rejects_candidate':
            headline = 'Adoption gate rejects the candidate'
            verdict = 'reject_candidate_for_live_adoption'
            recommended_action = 'clear_the_proof_window_and_keep_the_live_baseline'
            summary = 'The isolated live proof window rejected the candidate, so it should not replace the baseline live path.'
        elif proof_verdict == 'live_proof_inconclusive':
            headline = 'Adoption gate remains mixed'
            verdict = 'adoption_requires_manual_review'
            recommended_action = 'extend_the_proof_or_keep_the_live_baseline'
            summary = 'The proof window is directionally informative but not decisive enough for a safe adoption decision.'
        elif proof_verdict == 'live_proof_supports_candidate':
            stronger_support = proof_visible_rows >= 50 and (proof_gap is None or proof_gap > 0) and (proof_utility_score is None or proof_utility_score > 0)
            if stronger_support:
                headline = 'Candidate is ready for controlled adoption'
                verdict = 'ready_for_controlled_live_candidate_adoption'
                recommended_action = 'activate_controlled_candidate_adoption'
                summary = 'The candidate cleared the offline single-candidate gate and the isolated live proof gate, so it is ready for a controlled adoption with rollback preserved.'
            else:
                headline = 'Candidate is supported, but the adoption gate wants a bit more proof'
                verdict = 'candidate_supported_but_more_live_proof_preferred'
                recommended_action = 'consider_extending_the_proof_before_adoption'
                summary = 'The isolated live proof supports the candidate, but the evidence margin is still thin for a clean adoption decision.'

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
            'candidate': candidate or None,
            'next_live_candidate_summary': next_summary,
            'proof_summary': proof_summary,
            'proof_review': proof_review,
            'active_adoption': active_adoption or None,
            'baseline_current_version_evidence_at_activation': baseline_at_activation or None,
            'adoption_deltas_vs_activation_baseline': deltas,
            'decision_memo_markdown': (
                '# Controlled live candidate adoption gate\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n\n'
                '## Why this tranche exists\n'
                '- The isolated proof review answers whether the exact candidate works in live conditions.\n'
                '- The remaining missing step is a gated adoption decision with rollback preserved.\n'
                '- This service refuses to make adoption look casual: it requires the offline winner and the isolated live proof winner to line up.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, next_summary, proof_summary, proof_review, active_adoption)
        return payload

    def activate(self) -> dict:
        summary = self.build_summary(reason='activate')
        if str(summary.get('verdict') or '') != 'ready_for_controlled_live_candidate_adoption':
            raise RuntimeError('Controlled live candidate adoption is not currently justified by the evidence gate.')
        candidate = dict(summary.get('candidate') or {})
        if not candidate:
            raise RuntimeError('No candidate payload is available for controlled live adoption.')
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        proof_summary = dict(summary.get('proof_summary') or {})
        proof_session = dict((proof_summary.get('active_override') or {}))
        payload = {
            'source': 'live_candidate_adoption',
            'adopted_live_candidate_active': True,
            'adoption_session_id': f"adopt-{uuid.uuid4().hex[:12]}",
            'adopted_at_utc': _utc_now_iso(),
            'note': 'Evidence-gated live candidate adoption activated from the controlled proof review gate.',
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
            'model_bundle_path_override': candidate.get('model_bundle_path'),
            'model_bundle_label_override': candidate.get('model_source') or candidate.get('model_kind'),
            'stage1_selection_mode_override': candidate.get('stage1_selection_mode'),
            'stage1_max_candidates_override': int(candidate.get('stage1_max_candidates') or getattr(self.config, 'stage1_max_candidates', 40) or 40),
            'live_raw_threshold_override': _f(candidate.get('raw_threshold')),
            'evidence_gate_verdict': summary.get('verdict'),
            'offline_next_live_candidate_verdict': dict(summary.get('next_live_candidate_summary') or {}).get('verdict'),
            'isolated_live_proof_verdict': dict(summary.get('proof_review') or {}).get('verdict'),
            'proof_session_id': proof_session.get('proof_session_id'),
            'rollback_note': 'Clear this adoption override to restore configured live semantics for the current deployment scope.',
            'rollback_live_baseline': dict(dict(summary.get('next_live_candidate_summary') or {}).get('live_baseline') or {}),
        }
        atomic_write_json(self.overrides_path, payload)
        atomic_write_json(self.state_path, payload)
        return self.build_summary(reason='post_activate')

    def clear(self) -> dict:
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        existing = self._load_override() or read_json(self.state_path, {}) or {}
        payload = {
            'source': 'live_candidate_adoption',
            'adopted_live_candidate_active': False,
            'adoption_session_id': existing.get('adoption_session_id'),
            'adopted_at_utc': existing.get('adopted_at_utc'),
            'model_bundle_path_override': existing.get('model_bundle_path_override'),
            'model_bundle_label_override': existing.get('model_bundle_label_override'),
            'stage1_selection_mode_override': existing.get('stage1_selection_mode_override'),
            'stage1_max_candidates_override': existing.get('stage1_max_candidates_override'),
            'cleared_at_utc': _utc_now_iso(),
            'reason': 'controlled_live_candidate_adoption_cleared',
            'note': 'Controlled live candidate adoption override cleared by operator action.',
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
            'live_raw_threshold_override': None,
        }
        atomic_write_json(self.overrides_path, payload)
        atomic_write_json(self.state_path, payload)
        return self.build_summary(reason='post_clear')

    def _build_pack(self, summary: dict, next_summary: dict, proof_summary: dict, proof_review: dict, active_adoption: dict):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('live_candidate_adoption_summary.json', _json_bytes(summary))
            zf.writestr('live_candidate_adoption_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('next_live_candidate_summary.json', _json_bytes(next_summary))
            zf.writestr('live_candidate_proof_summary.json', _json_bytes(proof_summary))
            zf.writestr('live_candidate_proof_review_summary.json', _json_bytes(proof_review))
            zf.writestr('active_live_candidate_adoption_override.json', _json_bytes(active_adoption or {}))
