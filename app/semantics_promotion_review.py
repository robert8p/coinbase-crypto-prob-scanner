from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .persist import atomic_write_json, ensure_dir, read_json
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


def _summary_txt(payload: dict) -> str:
    lines = [
        f"Headline: {payload.get('headline') or '-'}",
        f"Verdict: {payload.get('verdict') or '-'}",
        f"Recommended action: {payload.get('recommended_action') or '-'}",
        f"Summary: {payload.get('summary') or '-'}",
        '',
    ]
    baseline = dict(payload.get('baseline_legacy_reference') or {})
    current = dict(payload.get('current_promoted_evidence') or {})
    deltas = dict(payload.get('deltas_vs_legacy_baseline') or {})
    lines.extend([
        'Legacy baseline reference',
        f"- Version: {baseline.get('baseline_version') or '-'}",
        f"- Visible quality hit rate: {baseline.get('visible_quality_hit_rate')}",
        f"- Visible-hidden quality gap: {baseline.get('visible_hidden_quality_gap')}",
        f"- Visible avg end ret: {baseline.get('visible_avg_end_ret')}",
        '',
        'Current promoted evidence',
        f"- Visible rows: {current.get('visible_rows')}",
        f"- Visible quality hit rate: {current.get('visible_quality_hit_rate')}",
        f"- Visible-hidden quality gap: {current.get('visible_hidden_quality_gap')}",
        f"- Visible avg end ret: {current.get('visible_avg_end_ret')}",
        '',
        'Deltas vs baseline',
        f"- Visible quality hit rate delta: {deltas.get('visible_quality_hit_rate_delta')}",
        f"- Visible-hidden quality gap delta: {deltas.get('visible_hidden_quality_gap_delta')}",
        f"- Visible avg end ret delta: {deltas.get('visible_avg_end_ret_delta')}",
    ])
    under = payload.get('underperforming_dimensions') or []
    if under:
        lines.extend(['', 'Underperforming dimensions'] + [f"- {item}" for item in under])
    return "\n".join(lines).strip() + "\n"


class SemanticsPromotionReviewService:
    def __init__(self, config, review_packs: ReviewPackService):
        self.config = config
        self.review_packs = review_packs
        self.repo_root = Path(__file__).resolve().parent.parent
        self.root_dir = ensure_dir(Path(config.model_dir) / 'semantics_promotion_review')
        self.summary_path = self.root_dir / 'latest_semantics_promotion_review_summary.json'
        self.pack_path = self.root_dir / 'latest_semantics_promotion_review_pack.zip'

    def latest_summary(self) -> dict:
        payload = read_json(self.summary_path, {})
        if isinstance(payload, dict) and payload:
            return payload
        return {
            'available': False,
            'app_version': APP_VERSION,
            'headline': 'No semantics promotion review has been generated yet',
            'summary': 'Load the semantics promotion review after the promoted version accumulates resolved rows.',
        }

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _load_release_manifest(self) -> dict:
        path = self.repo_root / 'release_manifest.json'
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def build_summary(self, *, reason: str = 'refresh') -> dict:
        manifest = self._load_release_manifest()
        baseline = dict(manifest.get('promotion_baseline_reference') or {})
        current = self.review_packs.get_current_version_summary(APP_VERSION) or {}
        evidence = dict(current.get('evidence') or {})
        visible_q = _f(evidence.get('visible_quality_hit_rate'))
        hidden_q = _f(evidence.get('non_visible_quality_hit_rate'))
        visible_end_ret = _f(evidence.get('visible_avg_end_ret'))
        current_gap = None if visible_q is None or hidden_q is None else round((visible_q or 0.0) - (hidden_q or 0.0), 6)
        baseline_q = _f(baseline.get('visible_quality_hit_rate'))
        baseline_gap = _f(baseline.get('visible_hidden_quality_gap'))
        baseline_end_ret = _f(baseline.get('visible_avg_end_ret'))
        min_visible_rows = int(baseline.get('min_visible_rows_for_guardrail') or 30)
        visible_rows = int(evidence.get('visible_rows') or 0)
        deltas = {
            'visible_quality_hit_rate_delta': None if visible_q is None or baseline_q is None else round((visible_q or 0.0) - (baseline_q or 0.0), 6),
            'visible_hidden_quality_gap_delta': None if current_gap is None or baseline_gap is None else round((current_gap or 0.0) - (baseline_gap or 0.0), 6),
            'visible_avg_end_ret_delta': None if visible_end_ret is None or baseline_end_ret is None else round((visible_end_ret or 0.0) - (baseline_end_ret or 0.0), 6),
        }
        under = []
        if deltas['visible_quality_hit_rate_delta'] is not None and deltas['visible_quality_hit_rate_delta'] < 0:
            under.append('visible_quality_hit_rate')
        if deltas['visible_hidden_quality_gap_delta'] is not None and deltas['visible_hidden_quality_gap_delta'] < 0:
            under.append('visible_vs_hidden_quality_gap')
        if deltas['visible_avg_end_ret_delta'] is not None and deltas['visible_avg_end_ret_delta'] < 0:
            under.append('visible_avg_end_ret')

        if not baseline:
            headline = 'Semantics promotion review is missing the retained 4.20.9 baseline reference'
            verdict = 'missing_baseline_reference'
            recommended_action = 'inspect_release_manifest_and_restore_the_baseline_reference'
            summary = 'The promoted path cannot be judged cleanly because the retained legacy baseline reference is missing from release_manifest.json.'
        elif visible_rows < min_visible_rows:
            headline = 'Semantics promotion review is waiting for a meaningful resolved batch'
            verdict = 'waiting_for_meaningful_resolved_batch'
            recommended_action = 'keep_collecting_promoted_version_evidence'
            summary = f'The promoted path has {visible_rows} resolved visible rows so far. Wait until at least {min_visible_rows} are resolved before making a keep-versus-rollback recommendation.'
        elif under:
            headline = 'Semantics promotion review recommends rollback'
            verdict = 'semantics_promotion_review_recommends_rollback'
            recommended_action = 'rollback_to_legacy_visible_shortlist_semantics'
            summary = 'The promoted contract-aligned live shortlist is underperforming the confirmed 4.20.9 legacy baseline on at least one required guardrail metric.'
        else:
            headline = 'Semantics promotion review supports keeping the promoted live shortlist'
            verdict = 'semantics_promotion_review_supports_keep'
            recommended_action = 'keep_the_promoted_contract_aligned_live_shortlist'
            summary = 'The promoted contract-aligned live shortlist is at least matching the confirmed 4.20.9 legacy baseline on the required guardrail metrics.'

        payload = {
            'available': True,
            'app_version': APP_VERSION,
            'generated_at_utc': _utc_now_iso(),
            'reason': reason,
            'headline': headline,
            'verdict': verdict,
            'recommended_action': recommended_action,
            'summary': summary,
            'baseline_legacy_reference': baseline or None,
            'current_promoted_evidence': {
                'visible_rows': visible_rows,
                'resolved_rows': int(evidence.get('resolved_rows') or 0),
                'visible_quality_hit_rate': visible_q,
                'non_visible_quality_hit_rate': hidden_q,
                'visible_hidden_quality_gap': current_gap,
                'visible_avg_end_ret': visible_end_ret,
                'visible_avg_mae': _f(evidence.get('visible_avg_mae')),
            },
            'deltas_vs_legacy_baseline': deltas,
            'underperforming_dimensions': under,
            'minimum_visible_rows_for_guardrail': min_visible_rows,
            'current_version_summary_excerpt': {
                'generated_at_utc': current.get('generated_at_utc'),
                'deployed_since_utc': current.get('deployed_since_utc'),
                'scan_pack_count': current.get('scan_pack_count'),
                'evaluated_pack_count': current.get('evaluated_pack_count'),
            },
        }
        atomic_write_json(self.summary_path, payload)
        self._build_pack(payload, manifest, current)
        return payload

    def _build_pack(self, summary: dict, manifest: dict, current_version_summary: dict) -> None:
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('semantics_promotion_review_summary.json', json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr('semantics_promotion_review_summary.txt', _summary_txt(summary))
            zf.writestr('release_manifest.json', json.dumps(manifest or {}, indent=2, sort_keys=True))
            zf.writestr('current_version_summary.json', json.dumps(current_version_summary or {}, indent=2, sort_keys=True, default=str))
