from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .objective_semantics import load_objective_semantics_contract
from .persist import atomic_write_json, ensure_dir, read_json
from .review_runs import ReviewPackService
from .version import APP_VERSION

VISIBLE_OBJECTIVE_BANDS = {"confirmed_shortlist", "strong_edge", "priority_edge", "elite_edge"}
STRONGER_OBJECTIVE_BANDS = {"strong_edge", "priority_edge", "elite_edge"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except Exception:
        return default


class SemanticsShadowComparisonService:
    """Evidence-only shadow comparison for the contract-aligned semantics challenger.

    Live runtime remains unchanged. This service records how the current live legacy
    shortlist compares against the contract-aligned challenger on the same completed scan.
    """

    def __init__(self, config: AppConfig, review_packs: ReviewPackService, semantics_comparison_service: Any):
        self.config = config
        self.review_packs = review_packs
        self.semantics_comparison_service = semantics_comparison_service
        self.root_dir = ensure_dir(Path(config.model_dir) / 'semantics_shadow_comparison')
        self.summary_path = self.root_dir / 'latest_semantics_shadow_comparison_summary.json'
        self.pack_path = self.root_dir / 'latest_semantics_shadow_comparison_pack.zip'
        self.history_path = self.root_dir / 'comparison_history.jsonl'

    def latest_summary(self) -> dict:
        summary = read_json(self.summary_path, {})
        if summary:
            summary.setdefault('available', True)
            summary.setdefault('app_version', APP_VERSION)
            summary['pack_available'] = self.pack_path.exists() and str(summary.get('status') or '') == 'recorded'
            return summary
        return {
            'available': False,
            'app_version': APP_VERSION,
            'headline': 'No semantics shadow comparison summary available yet',
            'summary': 'The first shadow comparison will appear after a completed scan once an objective-aligned semantics challenger is supported offline.',
            'status': 'waiting',
            'pack_available': False,
        }

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _read_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        rows: list[dict] = []
        for line in self.history_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _append_history(self, payload: dict) -> None:
        line = json.dumps(payload, sort_keys=True)
        with self.history_path.open('a', encoding='utf-8') as fh:
            fh.write(line + "\n")

    def _resolve_live_selection_state(self, status: dict[str, Any] | None) -> dict[str, str]:
        payload = dict(status or {})
        configured = str(
            payload.get('configured_live_selection_mode')
            or getattr(self.config, 'live_selection_mode', 'legacy')
            or 'legacy'
        ).strip().lower() or 'legacy'
        effective_mode = str(
            payload.get('effective_live_selection_mode')
            or payload.get('effective_live_selection_engine')
            or payload.get('selection_engine')
            or configured
        ).strip().lower() or configured
        effective_engine = str(
            payload.get('effective_live_selection_engine')
            or payload.get('selection_engine')
            or effective_mode
        ).strip().lower() or effective_mode
        return {
            'configured_live_selection_mode': configured,
            'effective_live_selection_mode': effective_mode,
            'effective_live_selection_engine': effective_engine,
        }

    def _match_run(self, *, app_version: str, generated_at_utc: str) -> dict | None:
        generated = _parse_utc(generated_at_utc)
        if generated is None:
            return None
        try:
            runs = self.review_packs.get_runs_for_app_version(str(app_version or APP_VERSION), limit=30)
        except Exception:
            return None
        best = None
        best_delta = None
        for run in runs:
            finished = _parse_utc(run.get('scan_finished_utc'))
            if finished is None:
                continue
            delta = abs((generated - finished).total_seconds())
            if delta > 900:
                continue
            if best is None or delta < (best_delta or 10**18):
                best = run
                best_delta = delta
        return best

    def _candidate_pool(self, live_rows: list[dict], trimmed_rows: list[dict], suppressed_rows: list[dict]) -> list[dict]:
        dedup: dict[str, dict] = {}
        for row in list(live_rows or []) + list(trimmed_rows or []) + list(suppressed_rows or []):
            symbol = str(row.get('symbol') or '')
            if not symbol:
                continue
            existing = dedup.get(symbol)
            current_rank = int(row.get('candidate_rank_all') or row.get('pre_policy_rank') or row.get('score_rank') or 999999)
            existing_rank = int((existing or {}).get('candidate_rank_all') or (existing or {}).get('pre_policy_rank') or (existing or {}).get('score_rank') or 999999)
            if existing is None or current_rank < existing_rank:
                dedup[symbol] = dict(row)
        rows = list(dedup.values())
        rows.sort(key=lambda r: (
            int(r.get('candidate_rank_all') or r.get('pre_policy_rank') or r.get('score_rank') or 999999),
            -(float(r.get('live_score', r.get('prob_2') or 0.0) or 0.0)),
            str(r.get('symbol') or ''),
        ))
        return rows

    def _mean_score(self, rows: list[dict]) -> float | None:
        vals = [float(r.get('live_score', r.get('prob_2') or 0.0) or 0.0) for r in rows if r.get('live_score', r.get('prob_2')) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 6)

    def _top_band_counts(self, rows: list[dict]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for row in rows:
            counter[str(row.get('objective_score_band') or row.get('score_band') or 'unknown')] += 1
        return dict(counter)

    def _objective_recommendation(self, summary: dict) -> dict:
        current = dict((summary.get('paths') or {}).get('current_035_path') or {})
        contract = dict((summary.get('paths') or {}).get('recalibrated_contract_path') or {})
        widening = dict((summary.get('paths') or {}).get('widening_028_reference_path') or {})
        if not current or not contract:
            return {
                'status': 'missing_comparison',
                'recommended_path_name': None,
                'recommended_path_label': None,
                'reason': 'No completed semantics comparison is available yet.',
                'shadow_ready': False,
            }
        current_visible_q = _safe_float(((current.get('visible') or {}).get('quality_hit_rate')), 0.0) or 0.0
        current_top3 = _safe_float((((current.get('topk_quality') or {}).get('top_3') or {}).get('mean_quality_rate')), 0.0) or 0.0
        current_shortlist = _safe_float(((current.get('shortlist_size_distribution') or {}).get('mean')), 999.0) or 999.0

        contract_visible_q = _safe_float(((contract.get('visible') or {}).get('quality_hit_rate')), 0.0) or 0.0
        contract_top3 = _safe_float((((contract.get('topk_quality') or {}).get('top_3') or {}).get('mean_quality_rate')), 0.0) or 0.0
        contract_shortlist = _safe_float(((contract.get('shortlist_size_distribution') or {}).get('mean')), 999.0) or 999.0

        widening_visible_q = _safe_float(((widening.get('visible') or {}).get('quality_hit_rate')), 0.0) or 0.0
        widening_shortlist = _safe_float(((widening.get('shortlist_size_distribution') or {}).get('mean')), 999.0) or 999.0

        contract_beats_current = (
            contract_visible_q >= current_visible_q
            and contract_top3 >= (current_top3 - 0.01)
            and contract_shortlist <= current_shortlist
        )
        widening_objective_blocked = widening_visible_q < current_visible_q or widening_shortlist > (current_shortlist * 1.5)

        if contract_beats_current:
            return {
                'status': 'supported_offline',
                'recommended_path_name': 'recalibrated_contract_path',
                'recommended_path_label': 'Recalibrated contract-aligned path',
                'reason': 'The contract-aligned path improves visible quality without widening the shortlist, so it is the objective-aligned challenger for a live shadow proof window.',
                'shadow_ready': True,
                'widening_objective_blocked': widening_objective_blocked,
            }
        return {
            'status': 'not_supported_offline',
            'recommended_path_name': None,
            'recommended_path_label': None,
            'reason': "No challenger improved the visible shortlist on the app's true objective without adding shortlist noise.",
            'shadow_ready': False,
            'widening_objective_blocked': widening_objective_blocked,
        }

    def _ordered_rows(self, rows: list[dict]) -> list[dict]:
        return sorted(
            list(rows or []),
            key=lambda row: (
                _safe_int(row.get('candidate_rank_all'), 10**9),
                _safe_int(row.get('pre_policy_rank'), 10**9),
                -(_safe_float(row.get('live_score'), 0.0) or 0.0),
                str(row.get('symbol') or ''),
            ),
        )

    def _select_contract_rows(self, ordered: list[dict], *, contract: dict) -> list[tuple[dict, str]]:
        strong_floor = _safe_float(contract.get('strong_edge_floor'))
        confirmed_floor = _safe_float(contract.get('confirmed_shortlist_floor'), _safe_float(getattr(self.config, 'live_raw_threshold', 0.35), 0.35)) or 0.35
        top_cap = min(
            5,
            max(1, int(getattr(self.config, 'stage2_decision_focus_top_n', 5) or 5)),
            max(1, int(getattr(self.config, 'utility_shortlist_target_max_names', 8) or 8)),
        )
        strong_rows: list[tuple[dict, str]] = []
        for row in ordered:
            band = str(row.get('objective_score_band') or '')
            live_score = _safe_float(row.get('live_score'), 0.0) or 0.0
            if band in STRONGER_OBJECTIVE_BANDS:
                strong_rows.append((row, f'contract_{band}'))
                continue
            if strong_floor is not None and live_score >= strong_floor:
                strong_rows.append((row, 'contract_strong_floor'))
        if strong_rows:
            return strong_rows[:top_cap]
        fallback_gap = 0.015
        for row in ordered:
            band = str(row.get('objective_score_band') or '')
            live_score = _safe_float(row.get('live_score'), 0.0) or 0.0
            if band not in VISIBLE_OBJECTIVE_BANDS:
                continue
            if live_score < confirmed_floor:
                continue
            if strong_floor is not None and live_score < strong_floor - fallback_gap:
                continue
            return [(row, 'contract_near_strong_fallback')]
        return []

    def _build_pack(self, summary: dict, candidate_pool: list[dict], live_rows: list[dict], challenger_rows: list[dict]) -> None:
        recent_history = self._read_history()[-100:]
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('latest_semantics_shadow_comparison_summary.json', json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr('comparison_history_recent.json', json.dumps(recent_history, indent=2, sort_keys=True))
            zf.writestr('incumbent_visible_rows.json', json.dumps(live_rows, indent=2, sort_keys=True))
            zf.writestr('challenger_visible_rows.json', json.dumps(challenger_rows, indent=2, sort_keys=True))
            zf.writestr('candidate_pool_snapshot.json', json.dumps(candidate_pool[:200], indent=2, sort_keys=True))
            zf.writestr('README.txt', (
                'Semantics Shadow Comparison Pack\n\n'
                'This pack records the unchanged live legacy shortlist against the contract-aligned semantics challenger in controlled shadow on the same completed scan.\n\n'
                f"Generated: {summary.get('generated_at_utc')}\n"
                f"Live engine: {summary.get('effective_live_selection_engine')}\n"
                f"Recommendation: {((summary.get('objective_aligned_recommendation') or {}).get('recommended_path_label')) or '-'}\n"
            ))

    def record_scan(
        self,
        *,
        status: dict,
        live_rows: list[dict],
        trimmed_visible_rows: list[dict],
        suppressed_rows: list[dict],
        trigger_source: str = 'manual',
    ) -> dict:
        generated_at = _utc_now_iso()
        live_state = self._resolve_live_selection_state(status)
        if str(live_state.get('effective_live_selection_mode') or 'legacy').lower() != 'legacy':
            summary = {
                'available': True,
                'generated_at_utc': generated_at,
                'app_version': APP_VERSION,
                'headline': 'Semantics shadow comparison skipped because legacy is not the live incumbent',
                'summary': 'This evidence-only shadow comparison only runs while legacy remains the effective live shortlist engine.',
                'status': 'skipped',
                'skip_reason': 'live_engine_not_legacy',
                **live_state,
                'live_path_unchanged': False,
                'pack_available': False,
            }
            atomic_write_json(self.summary_path, summary)
            return summary

        semantics_summary = self.semantics_comparison_service.latest_summary() or {}
        recommendation = self._objective_recommendation(semantics_summary)
        if not recommendation.get('shadow_ready'):
            summary = {
                'available': True,
                'generated_at_utc': generated_at,
                'app_version': APP_VERSION,
                'headline': 'Semantics shadow comparison blocked by the offline objective gate',
                'summary': recommendation.get('reason') or 'No objective-aligned semantics challenger is currently supported offline.',
                'status': 'blocked_offline_gate',
                'skip_reason': 'objective_gate_not_met',
                **live_state,
                'objective_aligned_recommendation': recommendation,
                'live_path_unchanged': True,
                'pack_available': False,
            }
            if self.pack_path.exists():
                try:
                    self.pack_path.unlink()
                except Exception:
                    pass
            atomic_write_json(self.summary_path, summary)
            return summary

        matched_run = self._match_run(app_version=str((status or {}).get('version') or APP_VERSION), generated_at_utc=generated_at)
        source_run_id = str((matched_run or {}).get('run_id') or '') or None
        source_scan_finished_utc = str((matched_run or {}).get('scan_finished_utc') or generated_at)
        contract = dict((semantics_summary.get('objective_semantics_contract') or {}))
        if not contract:
            contract = load_objective_semantics_contract(
                self.config.model_dir,
                live_threshold=float(getattr(self.config, 'live_raw_threshold', 0.35) or 0.35),
                stage1_selection_mode=getattr(self.config, 'stage1_selection_mode', None),
            ) or {}

        candidate_pool = self._candidate_pool(live_rows, trimmed_visible_rows, suppressed_rows)
        ordered = self._ordered_rows(candidate_pool)
        selected = self._select_contract_rows(ordered, contract=contract)
        challenger_rows = [dict(row) for row, _reason in selected]
        incumbent_rows = [dict(r) for r in (live_rows or [])]
        incumbent_symbols = [str(r.get('symbol') or '') for r in incumbent_rows if str(r.get('symbol') or '')]
        challenger_symbols = [str(r.get('symbol') or '') for r in challenger_rows if str(r.get('symbol') or '')]
        overlap = sorted(set(incumbent_symbols) & set(challenger_symbols))
        incumbent_only = sorted(set(incumbent_symbols) - set(challenger_symbols))
        challenger_only = sorted(set(challenger_symbols) - set(incumbent_symbols))

        headline = 'Recorded contract-aligned semantics challenger against the live legacy shortlist'
        detail = 'The live legacy path stayed unchanged while the contract-aligned semantics challenger was recorded in shadow on the same scan.'
        summary = {
            'available': True,
            'generated_at_utc': generated_at,
            'app_version': APP_VERSION,
            'headline': headline,
            'summary': detail,
            'status': 'recorded',
            **live_state,
            'trigger_source': trigger_source,
            'source_run_id': source_run_id,
            'source_scan_finished_utc': source_scan_finished_utc,
            'current_live_pipeline_mode': str((status or {}).get('live_pipeline_mode') or 'raw_threshold'),
            'current_live_raw_threshold': _safe_float((status or {}).get('live_raw_threshold'), _safe_float(getattr(self.config, 'live_raw_threshold', 0.35), 0.35)),
            'current_stage1_selection_mode': str((status or {}).get('stage1_selection_mode') or getattr(self.config, 'stage1_selection_mode', '') or ''),
            'live_path_unchanged': True,
            'live_path_statement': 'Legacy remains the effective live selection engine; the contract-aligned semantics challenger is evidence-only in shadow.',
            'objective_aligned_recommendation': recommendation,
            'offline_reference': {
                'generated_at_utc': semantics_summary.get('generated_at_utc'),
                'headline': semantics_summary.get('headline'),
                'summary': semantics_summary.get('summary'),
            },
            'incumbent': {
                'engine': 'current_035_legacy',
                'visible_count': len(incumbent_rows),
                'symbols': incumbent_symbols,
                'mean_live_score': self._mean_score(incumbent_rows),
                'score_band_distribution': self._top_band_counts(incumbent_rows),
            },
            'challenger_policy': {
                'policy_name': 'recalibrated_contract_path',
                'policy_label': 'Recalibrated contract-aligned path',
                'selection_semantics': 'strong-edge-or-better objective bands, ranking preserved, top-5 cap, one near-strong fallback when no strong-edge row exists',
            },
            'challenger': {
                'engine': 'recalibrated_contract_path',
                'visible_count': len(challenger_rows),
                'symbols': challenger_symbols,
                'mean_live_score': self._mean_score(challenger_rows),
                'score_band_distribution': self._top_band_counts(challenger_rows),
            },
            'comparison': {
                'overlap_count': len(overlap),
                'incumbent_only_count': len(incumbent_only),
                'challenger_only_count': len(challenger_only),
                'overlap_symbols': overlap[:25],
                'incumbent_only_symbols': incumbent_only[:25],
                'challenger_only_symbols': challenger_only[:25],
                'candidate_pool_count': len(candidate_pool),
            },
            'pack_available': False,
        }
        self._append_history(summary)
        atomic_write_json(self.summary_path, summary)
        self._build_pack(summary, candidate_pool, incumbent_rows, challenger_rows)
        summary['pack_available'] = True
        atomic_write_json(self.summary_path, summary)
        return summary
