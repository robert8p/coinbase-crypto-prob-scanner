
from __future__ import annotations

import gc
import hashlib
import json
import logging
import shutil
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .coinbase_client import CoinbaseClient
from .config import AppConfig
from .demo_data import STABLES
from .features import build_training_frame
from .modeling import reconcile_runtime_metadata, train_pt2
from .persist import atomic_write_json, ensure_dir, read_json
from .review_runs import ReviewPackService
from .runtime_scope import current_runtime_scope
from .state import AppState
from .universe import UniverseBuilder
from .version import APP_VERSION

logger = logging.getLogger(__name__)


class AuditCancelledError(RuntimeError):
    """Raised when a background run is cancelled by operator request."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ''):
            return default
        return float(value)
    except Exception:
        return default


def _is_stablecoin_pair(symbol: str) -> bool:
    base = str(symbol).split('-', 1)[0].upper()
    return base in STABLES


def _downcast_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_float_dtype(series):
            frame[col] = pd.to_numeric(series, downcast='float')
        elif pd.api.types.is_integer_dtype(series):
            frame[col] = pd.to_numeric(series, downcast='integer')
    return frame


class FreshRetrainAuditService:
    def __init__(self, config: AppConfig, state: AppState, client: CoinbaseClient, review_packs: ReviewPackService):
        self.config = config
        self.state = state
        self.client = client
        self.review_packs = review_packs
        self.root_dir = ensure_dir(Path(config.model_dir) / 'fresh_retrain_audit')
        self.summary_path = self.root_dir / 'latest_fresh_retrain_audit_summary.json'
        self.pack_path = self.root_dir / 'latest_fresh_retrain_audit_pack.zip'
        self.shadow_model_path = self.root_dir / 'latest_shadow_pt2.joblib'
        self.incumbent_model_copy_path = self.root_dir / 'source_incumbent_pt2.joblib'
        self.source_context_path = self.root_dir / 'latest_source_context.json'
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def latest_summary(self) -> dict:
        summary = read_json(self.summary_path, {
            'available': False,
            'running': False,
            'status': 'idle',
            'headline': 'No fresh retrain audit has run yet.',
            'summary': 'Run the non-promoting fresh retrain + symbol concentration audit after falsification.',
            'app_version': APP_VERSION,
        })
        return self._repair_stale_summary(summary)

    def latest_pack(self) -> Path | None:
        summary = self.latest_summary()
        if not self.pack_path.exists():
            return None
        if str(summary.get('status') or '').lower() != 'completed':
            return None
        if str(summary.get('app_version') or APP_VERSION) != APP_VERSION:
            return None
        return self.pack_path

    def start_run(self) -> dict:
        if self._lock.locked():
            current = self.latest_summary()
            current['already_running'] = True
            return current
        self._stop_event.clear()
        thread = threading.Thread(target=self._run, daemon=True, name='fresh_retrain_audit')
        self._thread = thread
        thread.start()
        return self.latest_summary()

    def stop_run(self) -> dict:
        summary = self.latest_summary()
        thread_alive = (self._thread is not None) and self._thread.is_alive()
        if not (self._lock.locked() or thread_alive or bool(summary.get('running'))):
            summary['already_stopped'] = True
            summary['stop_requested'] = False
            return summary
        self._stop_event.set()
        updated = dict(summary)
        updated.update({
            'available': True,
            'running': True,
            'status': 'stopping',
            'generated_at_utc': _utc_now_iso(),
            'last_heartbeat_at_utc': _utc_now_iso(),
            'stop_requested': True,
            'headline': 'Stopping fresh retrain audit…',
            'summary': 'Cancellation requested. Waiting for the current stage to exit safely.',
        })
        atomic_write_json(self.summary_path, updated)
        return updated

    def _repair_stale_summary(self, summary: dict) -> dict:
        if not isinstance(summary, dict):
            return summary
        running_like = bool(summary.get('running')) or str(summary.get('status') or '').lower() in {'running', 'stopping'}
        thread_alive = (self._thread is not None) and self._thread.is_alive()
        if running_like and (not self._lock.locked()) and (not thread_alive):
            repaired = dict(summary)
            repaired.update({
                'available': True,
                'running': False,
                'status': 'interrupted',
                'generated_at_utc': _utc_now_iso(),
                'finished_at_utc': repaired.get('finished_at_utc') or _utc_now_iso(),
                'headline': 'Fresh retrain audit was interrupted',
                'summary': 'A previous fresh retrain audit no longer has a live worker thread. Treat this as interrupted/stale state, not an active run.',
                'stale_run_detected': True,
                'stop_requested': False,
            })
            atomic_write_json(self.summary_path, repaired)
            return repaired
        return summary

    def _raise_if_cancel_requested(self) -> None:
        if self._stop_event.is_set():
            raise AuditCancelledError('Fresh retrain audit cancelled by operator request.')

    def _select_training_symbols(self, ordered_symbols: List[str], max_symbols: int) -> List[str]:
        selected = list(ordered_symbols[: max(1, max_symbols)])
        deduped: List[str] = []
        seen = set()
        for symbol in selected:
            if symbol not in seen:
                deduped.append(symbol)
                seen.add(symbol)
        return deduped[: max(1, max_symbols)]

    def _ensure_context_symbols(self, selected: List[str], ordered_symbols: List[str], max_symbols: int) -> List[str]:
        out = list(selected)
        for ctx in ['BTC-USD', 'ETH-USD']:
            if ctx in ordered_symbols and ctx not in out and len(out) < max_symbols:
                out.append(ctx)
        return out[:max_symbols]

    def _live_path_snapshot(self, *, checkpoint: dict, evidence: dict) -> dict:
        return {
            'decision_checkpoint_outcome': checkpoint.get('current_outcome') or checkpoint.get('decision_checkpoint_outcome'),
            'resolved_visible_rows': int(checkpoint.get('resolved_visible_rows') or 0),
            'visible_quality_hit_rate': evidence.get('visible_quality_hit_rate'),
            'non_visible_quality_hit_rate': evidence.get('non_visible_quality_hit_rate'),
            'stage1_selection_mode': checkpoint.get('stage1_selection_mode') or self.config.stage1_selection_mode,
            'live_raw_threshold': checkpoint.get('live_raw_threshold') or checkpoint.get('effective_live_raw_threshold') or self.config.live_raw_threshold,
        }

    def _summary_checkpoint(self, summary: dict | None) -> dict:
        return dict((summary or {}).get('decision_checkpoint') or (summary or {}).get('decision_rule_checkpoint') or {})

    def _summary_evidence(self, summary: dict | None) -> dict:
        return dict((summary or {}).get('evidence') or {})

    def _summary_outcome(self, summary: dict | None) -> str | None:
        checkpoint = self._summary_checkpoint(summary)
        outcome = checkpoint.get('current_outcome') or checkpoint.get('decision_checkpoint_outcome')
        return str(outcome) if outcome not in (None, '') else None

    def _summary_is_falsified(self, summary: dict | None) -> bool:
        return self._summary_outcome(summary) == 'falsified'

    def _summary_branch_outcome(self, summary: dict | None) -> str | None:
        summary = dict(summary or {})
        branch = dict(summary.get('decision_branch_automation') or {})
        outcome = branch.get('checkpoint_outcome')
        if outcome in (None, ''):
            checkpoint = self._summary_checkpoint(summary)
            outcome = checkpoint.get('current_outcome') or checkpoint.get('decision_checkpoint_outcome')
        return str(outcome) if outcome not in (None, '') else None

    def _current_scope_key(self) -> str | None:
        scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        value = scope.get('state_scope_key') if isinstance(scope, dict) else None
        return str(value) if value not in (None, '') else None

    def _current_deployed_since(self) -> str | None:
        scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        value = scope.get('deployed_since_utc') if isinstance(scope, dict) else None
        return str(value) if value not in (None, '') else None

    def _parse_utc(self, value: Any) -> datetime | None:
        raw = str(value or '').strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace('Z', '+00:00'))
        except Exception:
            return None

    def _lineage_prefix(self, version: str | None) -> str:
        raw = str(version or '').strip()
        if not raw:
            return ''
        parts = raw.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[:2])
        return raw

    def _summary_matches_current_scope(self, summary: dict | None) -> bool:
        checkpoint = self._summary_checkpoint(summary)
        state_scope_key = checkpoint.get('state_scope_key')
        current_scope_key = self._current_scope_key()
        return bool(current_scope_key and state_scope_key and str(state_scope_key) == str(current_scope_key))

    def _clear_latest_artifacts(self) -> None:
        for path in [
            self.summary_path,
            self.pack_path,
            self.shadow_model_path,
            self.incumbent_model_copy_path,
        ]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                logger.warning('fresh_retrain_audit_cleanup_failed path=%s', path)

    def _discover_latest_falsified_summary(self) -> tuple[dict | None, dict]:
        rows = []
        try:
            with self.review_packs._connect() as conn:  # internal app service use
                rows = conn.execute(
                    "SELECT app_version, MAX(scan_finished_utc) AS latest_finished_utc FROM review_runs GROUP BY app_version ORDER BY latest_finished_utc DESC LIMIT 25"
                ).fetchall()
        except Exception as exc:
            logger.warning('fresh_retrain_audit_discovery_failed error=%s', exc)
            return None, {'origin': 'review_runs_discovery_failed', 'reason': str(exc)}

        current_lineage = self._lineage_prefix(APP_VERSION)
        current_deployed_at = self._parse_utc(self._current_deployed_since())
        for row in rows:
            version = str(row['app_version'] or '').strip()
            if not version or version == APP_VERSION:
                continue
            if self._lineage_prefix(version) != current_lineage:
                continue
            latest_finished = self._parse_utc(row['latest_finished_utc'])
            if current_deployed_at is not None and latest_finished is not None and latest_finished >= current_deployed_at:
                continue
            try:
                candidate = self.review_packs.get_current_version_summary(app_version=version) or {}
            except Exception:
                continue
            if not (self._summary_is_falsified(candidate) or (self._summary_branch_outcome(candidate) == 'falsified')):
                continue
            checkpoint = self._summary_checkpoint(candidate)
            return candidate, {
                'origin': 'review_runs_latest_prior_lineage_falsified',
                'source_app_version': version,
                'source_generated_at_utc': candidate.get('generated_at_utc'),
                'source_state_scope_key': checkpoint.get('state_scope_key'),
                'source_checkpoint_resolution': 'latest_prior_lineage_summary',
            }
        return None, {
            'origin': 'no_prior_lineage_falsified_summary_found',
            'source_app_version': APP_VERSION,
            'lineage_prefix': current_lineage,
        }

    def _resolve_source_summary(self, current_version: dict) -> tuple[dict, dict]:
        if self._summary_is_falsified(current_version) or (self._summary_branch_outcome(current_version) == 'falsified'):
            checkpoint = self._summary_checkpoint(current_version)
            return current_version, {
                'origin': 'current_version_scope',
                'source_app_version': current_version.get('app_version') or APP_VERSION,
                'source_generated_at_utc': current_version.get('generated_at_utc'),
                'source_state_scope_key': checkpoint.get('state_scope_key'),
                'source_checkpoint_resolution': 'current_version_summary',
            }

        cached = read_json(self.source_context_path, {})
        cached_summary = dict(cached.get('source_current_version_summary') or {}) if isinstance(cached, dict) else {}
        cached_meta = dict(cached.get('source_context') or {}) if isinstance(cached, dict) else {}
        if (
            cached_meta.get('source_app_version') == APP_VERSION
            and self._summary_matches_current_scope(cached_summary)
            and (self._summary_is_falsified(cached_summary) or (self._summary_branch_outcome(cached_summary) == 'falsified'))
        ):
            cached_meta.setdefault('origin', 'cached_current_scope_source_context')
            cached_meta.setdefault('source_checkpoint_resolution', 'cached_current_scope_summary')
            return cached_summary, cached_meta

        discovered_summary, meta = self._discover_latest_falsified_summary()
        if discovered_summary:
            meta.setdefault('source_checkpoint_resolution', 'discovered_current_app_summary')
            return discovered_summary, meta

        raise RuntimeError('No eligible falsified source checkpoint could be resolved for the fresh retrain audit branch.')

    def _persist_source_context(self, *, source_summary: dict, source_context: dict) -> None:
        payload = {
            'stored_at_utc': _utc_now_iso(),
            'app_version': APP_VERSION,
            'source_context': dict(source_context or {}),
            'source_current_version_summary': dict(source_summary or {}),
        }
        atomic_write_json(self.source_context_path, payload)

    def _copy_incumbent_artifact(self) -> str | None:
        src = Path(self.config.model_path_pt2)
        if not src.exists():
            return None
        ensure_dir(self.incumbent_model_copy_path.parent)
        shutil.copy2(src, self.incumbent_model_copy_path)
        return str(self.incumbent_model_copy_path)

    def _running_summary_base(self, source_summary: dict, source_context: dict) -> dict:
        scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)
        checkpoint = self._summary_checkpoint(source_summary)
        evidence = self._summary_evidence(source_summary)
        started = _utc_now_iso()
        return {
            'available': True,
            'running': True,
            'status': 'running',
            'app_version': APP_VERSION,
            'generated_at_utc': started,
            'started_at_utc': started,
            'last_heartbeat_at_utc': started,
            'state_scope_key': scope.get('state_scope_key') or checkpoint.get('state_scope_key'),
            'source_context': dict(source_context or {}),
            'headline': 'Fresh retrain + symbol concentration audit is running',
            'summary': 'Building a non-promoting shadow retrain candidate and audit pack. Live scoring remains unchanged.',
            'current_live_threshold': checkpoint.get('live_raw_threshold') or checkpoint.get('effective_live_raw_threshold') or self.config.live_raw_threshold,
            'current_stage1_selection_mode': checkpoint.get('stage1_selection_mode') or self.config.stage1_selection_mode,
            'current_live_path': self._live_path_snapshot(checkpoint=checkpoint, evidence=evidence),
            'stop_requested': False,
            'progress': {
                'stage': 'queued',
                'detail': 'Queued',
                'fraction': 0.0,
                'completed_symbols': 0,
                'total_symbols': 0,
                'current_symbol': None,
            },
        }

    def _write_running_summary(self, source_summary: dict, source_context: dict) -> None:
        atomic_write_json(self.summary_path, self._running_summary_base(source_summary, source_context))

    def _update_running_progress(
        self,
        *,
        stage: str,
        detail: str,
        fraction: float | None = None,
        completed_symbols: int | None = None,
        total_symbols: int | None = None,
        current_symbol: str | None = None,
        extra: dict | None = None,
    ) -> None:
        summary = self.latest_summary()
        if not summary:
            summary = {
                'available': True,
                'running': True,
                'status': 'running',
                'app_version': APP_VERSION,
                'started_at_utc': _utc_now_iso(),
            }
        progress = dict(summary.get('progress') or {})
        progress.update({
            'stage': stage,
            'detail': detail,
            'fraction': round(max(0.0, min(1.0, float(fraction if fraction is not None else progress.get('fraction') or 0.0))), 4),
            'completed_symbols': int(completed_symbols if completed_symbols is not None else progress.get('completed_symbols') or 0),
            'total_symbols': int(total_symbols if total_symbols is not None else progress.get('total_symbols') or 0),
            'current_symbol': current_symbol,
        })
        summary.update({
            'available': True,
            'running': True,
            'status': 'running',
            'generated_at_utc': _utc_now_iso(),
            'last_heartbeat_at_utc': _utc_now_iso(),
            'headline': 'Fresh retrain + symbol concentration audit is running',
            'summary': detail,
            'progress': progress,
            'stop_requested': bool(self._stop_event.is_set()),
        })
        if extra:
            summary.update(extra)
        atomic_write_json(self.summary_path, summary)

    def _run(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        current_version: dict = {}
        source_context: dict = {}
        try:
            self._clear_latest_artifacts()
            current_version = self.review_packs.get_current_version_summary() or {}
            source_summary, source_context = self._resolve_source_summary(current_version)
            self._raise_if_cancel_requested()
            self._persist_source_context(source_summary=source_summary, source_context=source_context)
            self._write_running_summary(source_summary, source_context)
            summary = self._build_summary(current_version=current_version, source_summary=source_summary, source_context=source_context)
            atomic_write_json(self.summary_path, summary)
        except AuditCancelledError as exc:
            cancelled = {
                'available': True,
                'running': False,
                'status': 'cancelled',
                'generated_at_utc': _utc_now_iso(),
                'finished_at_utc': _utc_now_iso(),
                'app_version': APP_VERSION,
                'headline': 'Fresh retrain audit cancelled',
                'summary': str(exc),
                'error': None,
                'stop_requested': True,
                'source_context': dict(source_context or {}),
                'progress': {
                    'stage': 'cancelled',
                    'detail': str(exc),
                    'fraction': 1.0,
                    'completed_symbols': 0,
                    'total_symbols': 0,
                    'current_symbol': None,
                },
            }
            atomic_write_json(self.summary_path, cancelled)
        except Exception as exc:
            logger.exception('fresh_retrain_audit_failed error=%s', exc)
            checkpoint = self._summary_checkpoint(current_version)
            evidence = self._summary_evidence(current_version)
            failed = {
                'available': True,
                'running': False,
                'status': 'failed',
                'generated_at_utc': _utc_now_iso(),
                'finished_at_utc': _utc_now_iso(),
                'app_version': APP_VERSION,
                'state_scope_key': self._current_scope_key() or checkpoint.get('state_scope_key'),
                'headline': 'Fresh retrain audit failed',
                'summary': f'{type(exc).__name__}: {exc}',
                'error': f'{type(exc).__name__}: {exc}',
                'stop_requested': bool(self._stop_event.is_set()),
                'source_context': dict(source_context or {}),
                'current_live_path': self._live_path_snapshot(checkpoint=checkpoint, evidence=evidence) if current_version else {},
                'progress': {
                    'stage': 'failed',
                    'detail': f'{type(exc).__name__}: {exc}',
                    'fraction': 1.0,
                    'completed_symbols': 0,
                    'total_symbols': 0,
                    'current_symbol': None,
                },
                'artifact_paths': {
                    'shadow_model_path': str(self.shadow_model_path),
                    'incumbent_model_path': str(self.incumbent_model_copy_path),
                    'pack_path': str(self.pack_path),
                },
                'notes': [
                    'This run did not produce a valid new shadow candidate summary.',
                    'Stale latest artifacts are cleared at run start so a failed run cannot masquerade as a current result.',
                ],
            }
            atomic_write_json(self.summary_path, failed)
        finally:
            self._stop_event.clear()
            self._thread = None
            self._lock.release()

    def _build_summary(self, *, current_version: dict, source_summary: dict, source_context: dict) -> dict:
        status = self.state.get_status() or {}
        checkpoint = self._summary_checkpoint(source_summary)
        stage1_omission = (source_summary.get('stage1_omission_audit_latest') or {})
        stage1_repair = (source_summary.get('stage1_selection_repair_review_latest') or {})
        evidence = self._summary_evidence(source_summary)
        model_output = (source_summary.get('model_output_distribution') or {})
        outlier = (source_summary.get('outlier_concentration') or {})
        scope = current_runtime_scope(self.config.model_dir, app_version=APP_VERSION)

        incumbent_model_copy_path = self._copy_incumbent_artifact()
        candidate = self._build_shadow_candidate()
        live_035 = (outlier.get('thresholds') or {}).get('0.35') or {}
        live_045 = (outlier.get('thresholds') or {}).get('0.45') or {}
        live_060 = (outlier.get('thresholds') or {}).get('0.60') or {}
        train_conc = candidate.get('training_symbol_concentration') or {}
        quality_conc = candidate.get('quality_symbol_concentration') or {}

        top_045_share = _f(live_045.get('top_symbol_share'), 0.0) or 0.0
        top_060_share = _f(live_060.get('top_symbol_share'), 0.0) or 0.0
        concentration_controls = bool((live_045.get('row_count') or 0) >= 5 and top_045_share >= 0.40)

        source_app_version = str((source_context or {}).get('source_app_version') or source_summary.get('app_version') or 'unknown')
        headline = 'Shadow retrain candidate built; live promotion remains blocked'
        summary = (
            f'The source checkpoint from {source_app_version} is falsified, Stage 1 alternatives still do not clearly beat the current mode, ' 
            'and the model-output window remains upper-tail starved. This run built a non-promoting fresh retrain candidate and a symbol concentration audit pack for offline review.'
        )
        if stage1_omission.get('verdict') != 'stage2_score_compression_likely':
            headline = 'Shadow retrain candidate built, but Stage 2 isolation remains imperfect'

        finished_at = _utc_now_iso()
        trained_symbols_used = list((candidate.get('shadow_model_result') or {}).get('training_symbols_used') or [])
        total_symbols = int(((candidate.get('training_spec') or {}).get('train_max_symbols')) or len(trained_symbols_used) or 0)
        completed_symbols = int(len(trained_symbols_used) or total_symbols or 0)
        result = {
            'available': True,
            'running': False,
            'status': 'completed',
            'generated_at_utc': finished_at,
            'finished_at_utc': finished_at,
            'app_version': APP_VERSION,
            'state_scope_key': scope.get('state_scope_key') or checkpoint.get('state_scope_key'),
            'source_context': dict(source_context or {}),
            'headline': headline,
            'summary': summary,
            'verdict': 'shadow_retrain_candidate_ready_for_offline_review',
            'non_promoting': True,
            'live_promotion_blocked': True,
            'current_live_path': {
                'decision_checkpoint_outcome': checkpoint.get('current_outcome') or checkpoint.get('decision_checkpoint_outcome'),
                'resolved_visible_rows': int(checkpoint.get('resolved_visible_rows') or 0),
                'visible_quality_hit_rate': evidence.get('visible_quality_hit_rate'),
                'non_visible_quality_hit_rate': evidence.get('non_visible_quality_hit_rate'),
                'stage1_selection_mode': checkpoint.get('stage1_selection_mode') or self.config.stage1_selection_mode,
                'live_raw_threshold': checkpoint.get('live_raw_threshold') or self.config.live_raw_threshold,
            },
            'justification': {
                'checkpoint_falsified': str(checkpoint.get('current_outcome') or checkpoint.get('decision_checkpoint_outcome')) == 'falsified',
                'stage1_omission_verdict': stage1_omission.get('verdict'),
                'stage1_repair_headline': stage1_repair.get('headline'),
                'model_output_headline': model_output.get('headline'),
                'average_ge_0_45_per_scan': (model_output.get('average_upper_tail_counts_per_scan') or {}).get('ge_0.45'),
                'fraction_zero_ge_0_45_scans': model_output.get('fraction_of_scans_with_zero_ge_0.45_rows'),
            },
            'shadow_training_spec': candidate.get('training_spec'),
            'shadow_model_result': candidate.get('shadow_model_result'),
            'training_symbol_concentration': train_conc,
            'quality_symbol_concentration': quality_conc,
            'live_outlier_concentration': {
                'threshold_0_35': live_035,
                'threshold_0_45': live_045,
                'threshold_0_60': live_060,
            },
            'future_retrain_spec_note': {
                'include_symbol_concentration_controls': concentration_controls,
                'reason': (
                    'The current >=0.45 live rows are sparse and concentrated enough that future retrain specs should include an explicit concentration-control check.'
                    if concentration_controls
                    else 'Current >=0.45 live rows are sparse but not dominated by a single symbol strongly enough to force a hard cap yet; keep concentration auditing explicit.'
                ),
            },
            'progress': {
                'stage': 'completed',
                'detail': 'Run finished. Use the summary and pack links below.',
                'fraction': 1.0,
                'completed_symbols': completed_symbols,
                'total_symbols': total_symbols,
                'current_symbol': None,
            },
            'artifact_paths': {
                'shadow_model_path': str(self.shadow_model_path),
                'incumbent_model_path': incumbent_model_copy_path,
                'pack_path': str(self.pack_path),
            },
            'notes': [
                'This is a non-promoting shadow branch. It must not overwrite the live pt2 bundle.',
                'Use the pack to compare whether the next causal move should be a Stage 2 tranche, a retrain candidate, or a revert/reassess decision.',
                'Threshold lowering is still not justified by this branch on its own.',
            ],
        }
        self._build_pack(result=result, current_version=current_version, source_summary=source_summary, status=status, candidate=candidate)
        return result

    def _build_pack(self, *, result: dict, current_version: dict, source_summary: dict, status: dict, candidate: dict) -> None:
        ensure_dir(self.pack_path.parent)
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('fresh_retrain_audit_summary.json', json.dumps(result, indent=2, sort_keys=True))
            zf.writestr('source_current_version_summary_excerpt.json', json.dumps({
                'app_version': source_summary.get('app_version'),
                'generated_at_utc': source_summary.get('generated_at_utc'),
                'decision_checkpoint': source_summary.get('decision_checkpoint') or source_summary.get('decision_rule_checkpoint') or {},
                'evidence': source_summary.get('evidence') or {},
                'outlier_concentration': source_summary.get('outlier_concentration') or {},
                'model_output_distribution': source_summary.get('model_output_distribution') or {},
                'stage1_omission_audit_latest': source_summary.get('stage1_omission_audit_latest') or {},
                'stage1_selection_repair_review_latest': source_summary.get('stage1_selection_repair_review_latest') or {},
            }, indent=2, sort_keys=True))
            zf.writestr('current_runtime_version_summary_excerpt.json', json.dumps({
                'app_version': current_version.get('app_version'),
                'generated_at_utc': current_version.get('generated_at_utc'),
                'decision_checkpoint': current_version.get('decision_checkpoint') or current_version.get('decision_rule_checkpoint') or {},
                'evidence': current_version.get('evidence') or {},
                'outlier_concentration': current_version.get('outlier_concentration') or {},
                'model_output_distribution': current_version.get('model_output_distribution') or {},
                'stage1_omission_audit_latest': current_version.get('stage1_omission_audit_latest') or {},
                'stage1_selection_repair_review_latest': current_version.get('stage1_selection_repair_review_latest') or {},
            }, indent=2, sort_keys=True))
            zf.writestr('status_excerpt.json', json.dumps({
                'app_version': status.get('app_version'),
                'updated_at_utc': status.get('updated_at_utc'),
                'stage1_selection_mode': status.get('stage1_selection_mode'),
                'effective_live_raw_threshold': status.get('effective_live_raw_threshold'),
                'score_diagnostics': status.get('score_diagnostics') or {},
                'candidate_quality': status.get('candidate_quality') or {},
            }, indent=2, sort_keys=True))
            zf.writestr('shadow_model_metadata.json', json.dumps(candidate.get('shadow_model_result') or {}, indent=2, sort_keys=True))
            zf.writestr('training_symbol_concentration.json', json.dumps(candidate.get('training_symbol_concentration') or {}, indent=2, sort_keys=True))
            zf.writestr('quality_symbol_concentration.json', json.dumps(candidate.get('quality_symbol_concentration') or {}, indent=2, sort_keys=True))
            zf.writestr('source_context.json', json.dumps(result.get('source_context') or {}, indent=2, sort_keys=True))
            zf.writestr('decision_memo.md', self._decision_memo_markdown(result, candidate))

    def _decision_memo_markdown(self, result: dict, candidate: dict) -> str:
        live = result.get('current_live_path') or {}
        spec_lines = "\n".join([f"- **{k}**: {v}" for k, v in (result.get('shadow_training_spec') or {}).items()])
        return (
            "# Fresh retrain + symbol concentration audit\n\n"
            f"- **Headline:** {result.get('headline')}\n"
            f"- **Verdict:** {result.get('verdict')}\n"
            f"- **Checkpoint outcome:** {live.get('decision_checkpoint_outcome')}\n"
            f"- **Visible quality-hit rate:** {live.get('visible_quality_hit_rate')}\n"
            f"- **Non-visible quality-hit rate:** {live.get('non_visible_quality_hit_rate')}\n"
            f"- **Live threshold:** {live.get('live_raw_threshold')}\n"
            f"- **Stage 1 mode:** {live.get('stage1_selection_mode')}\n\n"
            "## Why this branch is now justified\n"
            "- The current deployment window falsified at the visible-vs-hidden checkpoint.\n"
            "- Stage 1 omission still points to Stage 2 compression, not a clearly superior Stage 1 alternative.\n"
            "- This branch is non-promoting and does not alter live scoring.\n\n"
            "## Shadow training spec\n"
            + spec_lines + "\n\n"
            + "## Training concentration\n"
            + f"- Top all-row symbol share: {(candidate.get('training_symbol_concentration') or {}).get('top_symbol_share')}\n"
            + f"- Top quality-row symbol share: {(candidate.get('quality_symbol_concentration') or {}).get('top_symbol_share')}\n"
            + f"- Include concentration controls in future retrain spec: {((result.get('future_retrain_spec_note') or {}).get('include_symbol_concentration_controls'))}\n"
        )

    def _build_shadow_candidate(self) -> dict:
        train_lookback_days = 90
        train_max_symbols = 120
        sample_every_n_bars = 2
        self._raise_if_cancel_requested()
        self._update_running_progress(stage='universe', detail='Loading eligible training universe…', fraction=0.02)
        products = self.client.list_products()
        currencies = self.client.list_currencies()
        volume_map = self.client.get_volume_summary()
        universe = UniverseBuilder(self.config).build(products, currencies, volume_map)
        ordered_symbols = [p['id'] for p in universe.eligible]
        symbols = self._select_training_symbols(ordered_symbols, train_max_symbols)
        symbols = self._ensure_context_symbols(symbols, ordered_symbols, train_max_symbols)

        lookback_bars = max(self.config.stage2_lookback_5m_bars, int((train_lookback_days * 24 * 60) / 5))
        self._raise_if_cancel_requested()
        self._update_running_progress(stage='context', detail='Loading BTC/ETH context candles…', fraction=0.06, total_symbols=len(symbols))
        btc_df = self.client.get_candles('BTC-USD', lookback_bars) if 'BTC-USD' in symbols else None
        eth_df = self.client.get_candles('ETH-USD', lookback_bars) if 'ETH-USD' in symbols else None

        frames: List[pd.DataFrame] = []
        skipped: List[dict] = []
        rows_accumulated = 0
        total_symbols = max(1, len(symbols))
        for idx, symbol in enumerate(symbols, start=1):
            try:
                self._raise_if_cancel_requested()
                frac = 0.08 + (0.62 * ((idx - 1) / total_symbols))
                self._update_running_progress(stage='training_frame_build', detail=f'Building training rows for {symbol} ({idx}/{total_symbols})…', fraction=frac, completed_symbols=idx-1, total_symbols=total_symbols, current_symbol=symbol)
                df = self.client.get_candles(symbol, lookback_bars)
                observed_bars = int(df.attrs.get('observed_bars', int((df['volume'] > 0).sum()) if not df.empty else 0))
                if len(df) < max(self.config.train_feature_warmup_5m_bars, self.config.candles_per_horizon + 48):
                    skipped.append({'symbol': symbol, 'reason': f'insufficient_history bars={len(df)}'})
                    continue
                if observed_bars < max(self.config.stage2_min_observed_5m_bars, 48):
                    skipped.append({'symbol': symbol, 'reason': f'insufficient_observed_bars observed_bars={observed_bars}'})
                    continue
                frame = build_training_frame(
                    symbol=symbol,
                    df=df,
                    btc_df=btc_df,
                    eth_df=eth_df,
                    sample_every=sample_every_n_bars,
                    horizon_bars=self.config.candles_per_horizon,
                    target_move_pct=self.config.target_move_pct,
                    warmup_bars=self.config.train_feature_warmup_5m_bars,
                    quality_max_mae=self.config.quality_max_mae,
                    quality_min_end_ret=self.config.quality_min_end_ret,
                )
                if frame.empty:
                    skipped.append({'symbol': symbol, 'reason': 'empty_training_frame'})
                    continue
                frame = _downcast_training_frame(frame)
                frames.append(frame)
                rows_accumulated += len(frame)
                del df, frame
                gc.collect()
                frac = 0.08 + (0.62 * (idx / total_symbols))
                self._update_running_progress(stage='training_frame_build', detail=f'Built training rows for {symbol} ({idx}/{total_symbols}) — rows accumulated: {rows_accumulated}', fraction=frac, completed_symbols=idx, total_symbols=total_symbols, current_symbol=symbol)
            except Exception as exc:
                skipped.append({'symbol': symbol, 'reason': str(exc)})
                logger.warning('fresh_retrain_symbol_skipped symbol=%s error=%s', symbol, exc)
                self._update_running_progress(stage='training_frame_build', detail=f'Skipped {symbol}: {exc}', fraction=0.08 + (0.62 * (idx / total_symbols)), completed_symbols=idx, total_symbols=total_symbols, current_symbol=symbol)

        self._raise_if_cancel_requested()
        self._update_running_progress(stage='training_dataframe', detail='Combining training rows…', fraction=0.72, completed_symbols=total_symbols, total_symbols=total_symbols)
        training_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        del frames
        gc.collect()
        if training_df.empty:
            raise RuntimeError('fresh retrain audit produced no training rows')

        quality_event_rate = float(training_df['y'].mean())
        raw_touch_rate = float(training_df['y_raw_touch'].mean()) if 'y_raw_touch' in training_df else quality_event_rate
        cfg_dict = {
            'app_version': APP_VERSION,
            'btc_panic_threshold': self.config.btc_panic_threshold,
            'panic_threshold_boost': self.config.panic_threshold_boost,
            'downside_cap': self.config.downside_cap,
            'uncertainty_cap': self.config.uncertainty_cap,
            'target_move_pct': self.config.target_move_pct,
            'target_horizon_minutes': self.config.target_horizon_minutes,
            'quality_max_mae': self.config.quality_max_mae,
            'quality_min_end_ret': self.config.quality_min_end_ret,
        }
        self._raise_if_cancel_requested()
        self._update_running_progress(stage='shadow_training', detail=f'Training shadow Stage 2 model on {len(training_df)} rows…', fraction=0.8, completed_symbols=total_symbols, total_symbols=total_symbols)
        bundle = train_pt2(training_df, cfg_dict=cfg_dict)
        used_symbol_set = set(training_df['symbol'].unique().tolist())
        requested_symbols = [symbol for symbol in symbols if not _is_stablecoin_pair(symbol)]
        trained_cohort_symbols = [symbol for symbol in requested_symbols if symbol in used_symbol_set]
        cohort_hash = hashlib.sha256('|'.join(trained_cohort_symbols).encode('utf-8')).hexdigest()[:16] if trained_cohort_symbols else 'none'
        bundle.metadata.update({
            'trained_cohort_symbols': trained_cohort_symbols,
            'trained_cohort_size': len(trained_cohort_symbols),
            'trained_cohort_hash': cohort_hash,
            'training_symbol_selection_method': 'top_liquidity_locked_shadow',
            'live_universe_mode': self.config.live_universe_mode,
            'training_candidate_pool_size': len(ordered_symbols),
            'training_symbols_requested': requested_symbols,
            'shadow_candidate': True,
            'shadow_candidate_label': 'fresh_retrain_and_symbol_concentration_audit',
            'train_lookback_days': train_lookback_days,
            'train_max_symbols': train_max_symbols,
            'train_sample_every_n_bars': sample_every_n_bars,
        })
        meta_seed = {'trained': True, 'path': str(self.shadow_model_path), **bundle.metadata}
        meta, _ = reconcile_runtime_metadata(
            meta_seed,
            existing_status=self.state.get_status(),
            min_count=self.config.tail_validation_min_count,
            min_wilson_lift=self.config.tail_validation_min_wilson_lift,
            min_precision_floor=self.config.tail_validation_min_precision_floor,
            unvalidated_tail_cap=self.config.tail_unvalidated_cap,
            scanner_contract_source='shadow_retrain_candidate',
            threshold_suppression_contract_source='shadow_retrain_candidate',
        )
        bundle.metadata.update({k: v for k, v in meta.items() if k not in {'trained', 'path'}})
        bundle.save(str(self.shadow_model_path))
        self._raise_if_cancel_requested()
        self._update_running_progress(stage='concentration_audit', detail='Computing symbol concentration audit…', fraction=0.92, completed_symbols=total_symbols, total_symbols=total_symbols)

        train_concentration = self._symbol_concentration(training_df)
        quality_concentration = self._symbol_concentration(training_df[training_df['y'] == 1].copy())
        self._raise_if_cancel_requested()
        self._update_running_progress(stage='packaging', detail='Packaging fresh retrain audit artifacts…', fraction=0.96, completed_symbols=total_symbols, total_symbols=total_symbols)

        return {
            'training_spec': {
                'label': 'fresh_retrain_and_symbol_concentration_audit',
                'train_lookback_days': train_lookback_days,
                'train_max_symbols': train_max_symbols,
                'train_sample_every_n_bars': sample_every_n_bars,
                'save_path': str(self.shadow_model_path),
                'switch_live_model_automatically': False,
                'purpose': 'Build a non-promoting shadow Stage 2 retrain candidate after checkpoint falsification.',
            },
            'shadow_model_result': {
                'trained_at_utc': bundle.metadata.get('trained_at_utc'),
                'model_type': bundle.model_type,
                'auc_holdout': bundle.metadata.get('auc_holdout'),
                'adjusted_auc_holdout': bundle.metadata.get('adjusted_auc_holdout'),
                'brier_holdout': bundle.metadata.get('brier_holdout'),
                'adjusted_brier_holdout': bundle.metadata.get('adjusted_brier_holdout'),
                'trained_cohort_size': bundle.metadata.get('trained_cohort_size'),
                'trained_cohort_hash': bundle.metadata.get('trained_cohort_hash'),
                'training_rows': int(len(training_df)),
                'quality_event_rate': quality_event_rate,
                'raw_touch_rate': raw_touch_rate,
                'training_symbols_requested': requested_symbols,
                'training_symbols_used': trained_cohort_symbols,
                'training_skipped': skipped,
                'path': str(self.shadow_model_path),
                'score_distribution_adjusted': bundle.metadata.get('score_distribution_adjusted') or {},
                'score_contract_live': bundle.metadata.get('score_contract_live') or {},
                'tail_validation_state': bundle.metadata.get('tail_validation_state'),
                'highest_validated_threshold': bundle.metadata.get('highest_validated_threshold'),
            },
            'training_symbol_concentration': train_concentration,
            'quality_symbol_concentration': quality_concentration,
        }

    def _symbol_concentration(self, frame: pd.DataFrame) -> dict:
        if frame is None or frame.empty or 'symbol' not in frame.columns:
            return {'row_count': 0, 'unique_symbols': 0, 'top_symbol': None, 'top_symbol_count': 0, 'top_symbol_share': None, 'top_3_share': None, 'top_symbols': []}
        counts = frame['symbol'].value_counts()
        total = int(counts.sum())
        top = counts.head(10)
        top_symbol = str(top.index[0]) if not top.empty else None
        top_symbol_count = int(top.iloc[0]) if not top.empty else 0
        top_symbol_share = round(top_symbol_count / total, 4) if total else None
        top3_share = round(top.head(3).sum() / total, 4) if total else None
        return {
            'row_count': total,
            'unique_symbols': int(counts.shape[0]),
            'top_symbol': top_symbol,
            'top_symbol_count': top_symbol_count,
            'top_symbol_share': top_symbol_share,
            'top_3_share': top3_share,
            'top_symbols': [{'symbol': str(sym), 'count': int(cnt)} for sym, cnt in top.items()],
        }
