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
from .utility_shortlist import load_active_utility_tuning_override
from .version import APP_VERSION


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


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
    cand = dict(payload.get('candidate_model') or {})
    if cand:
        lines.extend([
            'Candidate model',
            f"- Path: {cand.get('path') or '-'}",
            f"- Label: {cand.get('label') or '-'}",
            '',
        ])
    params = dict(payload.get('utility_params') or {})
    if params:
        lines.extend([
            'Utility shortlist semantics',
            f"- Engine: {params.get('utility_selection_engine_label') or '-'}",
            f"- Target max names: {params.get('utility_shortlist_target_max_names') or '-'}",
            f"- Score floor / dropoff: {params.get('utility_shortlist_score_floor') or '-'} / {params.get('utility_shortlist_score_dropoff') or '-'}",
            f"- Confidence floor: {params.get('utility_confidence_floor') or '-'}",
            f"- Edge/confidence/probability weights: {params.get('utility_expected_edge_weight') or '-'} / {params.get('utility_confidence_weight') or '-'} / {params.get('utility_probability_weight') or '-'}",
            '',
        ])
    proof = dict(payload.get('proof_review') or {})
    if proof:
        evidence = dict(proof.get('proof_evidence') or {})
        utility = dict(proof.get('scan_shortlist_utility') or {})
        lines.extend([
            'Isolated utility-model proof review',
            f"- Verdict: {proof.get('verdict') or '-'}",
            f"- Visible rows: {evidence.get('visible_rows') or 0}",
            f"- Visible-hidden gap: {evidence.get('visible_hidden_gap') or '-'}",
            f"- Utility score: {utility.get('scan_shortlist_utility_score') if utility.get('scan_shortlist_utility_score') is not None else '-'}",
            '',
        ])
    active = dict(payload.get('active_adoption') or {})
    if active:
        lines.extend([
            'Active adoption override',
            f"- Active: {active.get('active')}",
            f"- Adoption session id: {active.get('adoption_session_id') or '-'}",
            f"- Adopted at UTC: {active.get('adopted_at_utc') or '-'}",
            '',
        ])
    return "\n".join(lines).strip() + "\n"


class UtilityModelAdoptionService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.root_dir = ensure_dir(Path(config.model_dir) / 'utility_model_adoption')
        self.summary_path = self.root_dir / 'latest_utility_model_adoption_summary.json'
        self.pack_path = self.root_dir / 'latest_utility_model_adoption_pack.zip'
        self.state_path = self.root_dir / 'latest_utility_model_adoption_state.json'
        self.overrides_path = Path(config.model_dir) / 'runtime_live_overrides.json'
        self.lab_summary_path = Path(config.model_dir) / 'utility_model_lab' / 'latest_utility_model_lab_summary.json'
        self.proof_summary_path = Path(config.model_dir) / 'utility_model_proof' / 'latest_utility_model_proof_summary.json'
        self.proof_review_path = Path(config.model_dir) / 'utility_model_proof_review' / 'latest_utility_model_proof_review_summary.json'

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _load_override(self) -> dict:
        raw = read_json(self.overrides_path, {})
        if not isinstance(raw, dict) or raw.get('source') != 'utility_model_adoption':
            return {}
        if not bool(raw.get('adopted_utility_model_active')):
            return {}
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        current_scope_key = current_scope.get('state_scope_key')
        override_scope_key = raw.get('state_scope_key') or scope_key(raw.get('app_version'), raw.get('deployed_since_utc'))
        if current_scope_key and override_scope_key and current_scope_key != override_scope_key:
            return {}
        out = dict(raw)
        out['active'] = True
        return out

    def _candidate_model(self, lab_summary: dict, proof_summary: dict) -> dict:
        candidate_path = str(lab_summary.get('candidate_model_path') or '').strip()
        if not candidate_path:
            candidate_path = str((proof_summary.get('candidate_model') or {}).get('path') or '').strip()
        if not candidate_path:
            return {}
        return {'path': candidate_path, 'label': 'utility_model_candidate'}

    def _utility_params(self, proof_summary: dict) -> dict:
        active = dict(proof_summary.get('active_override') or {})
        if active:
            return {
                'utility_selection_engine_label': active.get('utility_selection_engine_label'),
                'utility_expected_edge_weight': active.get('utility_expected_edge_weight'),
                'utility_confidence_weight': active.get('utility_confidence_weight'),
                'utility_probability_weight': active.get('utility_probability_weight'),
                'utility_shortlist_target_max_names': active.get('utility_shortlist_target_max_names'),
                'utility_shortlist_score_floor': active.get('utility_shortlist_score_floor'),
                'utility_shortlist_score_dropoff': active.get('utility_shortlist_score_dropoff'),
                'utility_confidence_floor': active.get('utility_confidence_floor'),
                'utility_tier3_max_frac': active.get('utility_tier3_max_frac'),
            }
        utility_override = load_active_utility_tuning_override(self.config.model_dir)
        if utility_override:
            return {
                'utility_selection_engine_label': utility_override.get('utility_selection_engine_label'),
                'utility_expected_edge_weight': utility_override.get('utility_expected_edge_weight'),
                'utility_confidence_weight': utility_override.get('utility_confidence_weight'),
                'utility_probability_weight': utility_override.get('utility_probability_weight'),
                'utility_shortlist_target_max_names': utility_override.get('utility_shortlist_target_max_names'),
                'utility_shortlist_score_floor': utility_override.get('utility_shortlist_score_floor'),
                'utility_shortlist_score_dropoff': utility_override.get('utility_shortlist_score_dropoff'),
                'utility_confidence_floor': utility_override.get('utility_confidence_floor'),
                'utility_tier3_max_frac': utility_override.get('utility_tier3_max_frac'),
            }
        return {
            'utility_selection_engine_label': getattr(self.config, 'utility_selection_engine_label', 'utility_constrained_v1'),
            'utility_expected_edge_weight': getattr(self.config, 'utility_expected_edge_weight', 0.45),
            'utility_confidence_weight': getattr(self.config, 'utility_confidence_weight', 0.30),
            'utility_probability_weight': getattr(self.config, 'utility_probability_weight', 0.25),
            'utility_shortlist_target_max_names': getattr(self.config, 'utility_shortlist_target_max_names', 8),
            'utility_shortlist_score_floor': getattr(self.config, 'utility_shortlist_score_floor', 0.52),
            'utility_shortlist_score_dropoff': getattr(self.config, 'utility_shortlist_score_dropoff', 0.16),
            'utility_confidence_floor': getattr(self.config, 'utility_confidence_floor', 0.35),
            'utility_tier3_max_frac': getattr(self.config, 'utility_tier3_max_frac', 0.25),
        }

    def build_summary(self, *, reason: str = 'refresh') -> dict:
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        lab_summary = read_json(self.lab_summary_path, {})
        proof_summary = read_json(self.proof_summary_path, {})
        proof_review = read_json(self.proof_review_path, {})
        active_adoption = self._load_override()
        candidate_model = self._candidate_model(lab_summary, proof_summary)
        utility_params = self._utility_params(proof_summary)
        proof_evidence = dict(proof_review.get('proof_evidence') or {})
        proof_utility = dict(proof_review.get('scan_shortlist_utility') or {})
        baseline_at_activation = dict(proof_review.get('baseline_current_version_evidence_at_activation') or {})
        proof_gap = _f(proof_evidence.get('visible_hidden_gap'))
        proof_visible_rows = int(proof_evidence.get('visible_rows') or 0)
        proof_utility_score = _f(proof_utility.get('scan_shortlist_utility_score'))
        base_visible_q = _f(baseline_at_activation.get('visible_quality_hit_rate'))
        base_hidden_q = _f(baseline_at_activation.get('non_visible_quality_hit_rate'))
        base_gap = None
        if base_visible_q is not None and base_hidden_q is not None:
            base_gap = round(base_visible_q - base_hidden_q, 6)
        deltas = {
            'visible_hidden_gap_delta_vs_activation': round(proof_gap - base_gap, 6) if proof_gap is not None and base_gap is not None else None,
        }

        headline = 'No utility-model challenger is ready for controlled adoption'
        verdict = 'no_utility_model_ready_for_adoption'
        recommended_action = 'keep_current_live_model'
        summary = 'The app does not yet have the combined offline and isolated live proof evidence needed for a controlled adoption of the utility-model challenger.'

        lab_verdict = str(lab_summary.get('verdict') or '')
        proof_verdict = str(proof_review.get('verdict') or '')
        if active_adoption:
            headline = 'Controlled utility-model adoption is active'
            verdict = 'utility_model_adoption_active'
            recommended_action = 'monitor_current_scope_and_preserve_rollback_path'
            summary = 'An evidence-gated utility-model challenger is active for the current deployment scope. Preserve the rollback path and continue collecting current-scope evidence.'
        elif lab_verdict != 'utility_model_candidate_supported_offline' or not candidate_model:
            headline = 'No exact utility-model challenger has cleared the offline gate'
            verdict = 'offline_utility_model_not_ready_for_adoption'
            recommended_action = 'rerun_the_utility_model_lab_or_keep_current_live_model'
            summary = 'The app does not currently have one exact utility-model challenger that deserves adoption consideration.'
        elif proof_verdict in {'no_utility_model_proof_session', 'no_matching_utility_model_proof_runs', 'waiting_for_more_resolved_visible_rows', 'waiting_for_utility_model_proof_evidence'}:
            headline = 'Adoption gate is waiting for more isolated utility-model proof evidence'
            verdict = 'waiting_for_more_utility_model_proof_evidence'
            recommended_action = 'keep_or_complete_the_utility_model_proof_before_any_adoption'
            summary = 'The utility-model challenger may be promising offline, but the isolated live proof has not matured enough to justify adoption.'
        elif proof_verdict == 'utility_model_proof_rejects_candidate':
            headline = 'Adoption gate rejects the utility-model challenger'
            verdict = 'reject_utility_model_for_adoption'
            recommended_action = 'clear_the_utility_model_proof_and_keep_current_live_model'
            summary = 'The isolated live proof rejected the utility-model challenger, so it should not replace the current live model.'
        elif proof_verdict == 'utility_model_proof_inconclusive':
            headline = 'Adoption gate remains mixed'
            verdict = 'utility_model_adoption_requires_manual_review'
            recommended_action = 'extend_the_utility_model_proof_or_keep_current_live_model'
            summary = 'The utility-model proof is informative but not decisive enough for a safe adoption decision.'
        elif proof_verdict == 'utility_model_proof_supports_candidate':
            stronger_support = proof_visible_rows >= 50 and (proof_gap is None or proof_gap > 0) and (proof_utility_score is None or proof_utility_score > 0)
            if stronger_support:
                headline = 'Utility-model challenger is ready for controlled adoption'
                verdict = 'ready_for_controlled_utility_model_adoption'
                recommended_action = 'activate_controlled_utility_model_adoption'
                summary = 'The utility-model challenger cleared the offline gate and the isolated live proof gate, so it is ready for a controlled adoption with rollback preserved.'
            else:
                headline = 'Utility-model challenger is supported, but the adoption gate wants a bit more proof'
                verdict = 'utility_model_supported_but_more_live_proof_preferred'
                recommended_action = 'consider_extending_the_utility_model_proof_before_adoption'
                summary = 'The isolated utility-model proof supports the challenger, but the evidence margin is still thin for a clean adoption decision.'

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
            'candidate_model': candidate_model or None,
            'utility_params': utility_params,
            'utility_model_lab_summary': lab_summary,
            'proof_summary': proof_summary,
            'proof_review': proof_review,
            'active_adoption': active_adoption or None,
            'baseline_current_version_evidence_at_activation': baseline_at_activation or None,
            'adoption_deltas_vs_activation_baseline': deltas,
            'decision_memo_markdown': (
                '# Controlled utility-model adoption gate\n\n'
                f'- **Headline:** {headline}\n'
                f'- **Verdict:** {verdict}\n'
                f'- **Recommended action:** {recommended_action}\n\n'
                '## Why this tranche exists\n'
                '- The isolated utility-model proof answers whether the exact model challenger works in live conditions.\n'
                '- The remaining missing step is a gated adoption decision with rollback preserved.\n'
                '- This service refuses to make utility-model adoption look casual: it requires the offline winner and the isolated live proof winner to line up.\n'
            ),
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, lab_summary, proof_summary, proof_review, active_adoption)
        return payload

    def activate(self) -> dict:
        summary = self.build_summary(reason='activate')
        if str(summary.get('verdict') or '') != 'ready_for_controlled_utility_model_adoption':
            raise RuntimeError('Controlled utility-model adoption is not currently justified by the evidence gate.')
        candidate_model = dict(summary.get('candidate_model') or {})
        if not candidate_model:
            raise RuntimeError('No candidate model payload is available for controlled utility-model adoption.')
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        utility_params = dict(summary.get('utility_params') or {})
        proof_summary = dict(summary.get('proof_summary') or {})
        proof_session = dict((proof_summary.get('active_override') or {}))
        payload = {
            'source': 'utility_model_adoption',
            'adopted_utility_model_active': True,
            'adoption_session_id': f"utility_model_adopt-{uuid.uuid4().hex[:12]}",
            'adopted_at_utc': _utc_now_iso(),
            'note': 'Evidence-gated utility-model adoption activated from the controlled utility-model proof review gate.',
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
            'model_bundle_path_override': candidate_model.get('path'),
            'model_bundle_label_override': candidate_model.get('label'),
            'utility_selection_engine_label': utility_params.get('utility_selection_engine_label'),
            'utility_expected_edge_weight': utility_params.get('utility_expected_edge_weight'),
            'utility_confidence_weight': utility_params.get('utility_confidence_weight'),
            'utility_probability_weight': utility_params.get('utility_probability_weight'),
            'utility_shortlist_target_max_names': int(utility_params.get('utility_shortlist_target_max_names') or getattr(self.config, 'utility_shortlist_target_max_names', 8)),
            'utility_shortlist_score_floor': utility_params.get('utility_shortlist_score_floor'),
            'utility_shortlist_score_dropoff': utility_params.get('utility_shortlist_score_dropoff'),
            'utility_confidence_floor': utility_params.get('utility_confidence_floor'),
            'utility_tier3_max_frac': utility_params.get('utility_tier3_max_frac'),
            'evidence_gate_verdict': summary.get('verdict'),
            'offline_utility_model_verdict': dict(summary.get('utility_model_lab_summary') or {}).get('verdict'),
            'isolated_live_proof_verdict': dict(summary.get('proof_review') or {}).get('verdict'),
            'proof_session_id': proof_session.get('proof_session_id'),
            'rollback_note': 'Clear this adoption override to restore the configured live model and utility semantics for the current deployment scope.',
        }
        atomic_write_json(self.overrides_path, payload)
        atomic_write_json(self.state_path, payload)
        return self.build_summary(reason='post_activate')

    def clear(self) -> dict:
        current_scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        existing = self._load_override() or read_json(self.state_path, {}) or {}
        payload = {
            'source': 'utility_model_adoption',
            'adopted_utility_model_active': False,
            'adoption_session_id': existing.get('adoption_session_id'),
            'adopted_at_utc': existing.get('adopted_at_utc'),
            'model_bundle_path_override': existing.get('model_bundle_path_override'),
            'model_bundle_label_override': existing.get('model_bundle_label_override'),
            'utility_selection_engine_label': existing.get('utility_selection_engine_label'),
            'utility_expected_edge_weight': existing.get('utility_expected_edge_weight'),
            'utility_confidence_weight': existing.get('utility_confidence_weight'),
            'utility_probability_weight': existing.get('utility_probability_weight'),
            'utility_shortlist_target_max_names': existing.get('utility_shortlist_target_max_names'),
            'utility_shortlist_score_floor': existing.get('utility_shortlist_score_floor'),
            'utility_shortlist_score_dropoff': existing.get('utility_shortlist_score_dropoff'),
            'utility_confidence_floor': existing.get('utility_confidence_floor'),
            'utility_tier3_max_frac': existing.get('utility_tier3_max_frac'),
            'cleared_at_utc': _utc_now_iso(),
            'reason': 'controlled_utility_model_adoption_cleared',
            'note': 'Controlled utility-model adoption override cleared by operator action.',
            'app_version': current_scope.get('app_version') or APP_VERSION,
            'deployed_since_utc': current_scope.get('deployed_since_utc'),
            'state_scope_key': current_scope.get('state_scope_key'),
        }
        atomic_write_json(self.overrides_path, payload)
        atomic_write_json(self.state_path, payload)
        return self.build_summary(reason='post_clear')

    def _build_pack(self, summary: dict, lab_summary: dict, proof_summary: dict, proof_review: dict, active_adoption: dict):
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('utility_model_adoption_summary.json', _json_bytes(summary))
            zf.writestr('utility_model_adoption_summary.txt', _summary_txt(summary).encode('utf-8'))
            zf.writestr('utility_model_lab_summary.json', _json_bytes(lab_summary or {}))
            zf.writestr('utility_model_proof_summary.json', _json_bytes(proof_summary or {}))
            zf.writestr('utility_model_proof_review_summary.json', _json_bytes(proof_review or {}))
            zf.writestr('active_utility_model_adoption_override.json', _json_bytes(active_adoption or {}))
