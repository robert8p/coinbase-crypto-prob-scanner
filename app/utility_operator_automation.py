from __future__ import annotations

from datetime import datetime, timezone
import json
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .version import APP_VERSION


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True, default=str).encode('utf-8')


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


def _summary_matches_session(summary: dict, state: dict) -> bool:
    if not isinstance(summary, dict) or not summary:
        return False
    session_dt = _parse_iso(state.get('started_at_utc'))
    summary_dt = _parse_iso(summary.get('generated_at_utc'))
    if session_dt is None or summary_dt is None:
        return False
    if summary_dt < session_dt:
        return False
    expected_version = str(state.get('app_version') or APP_VERSION)
    summary_version = str(summary.get('app_version') or expected_version)
    return summary_version == expected_version


class UtilityOperatorAutomationService:
    def __init__(
        self,
        config: AppConfig,
        utility_selection_lab,
        utility_tuning_lab,
        utility_model_lab,
        utility_model_proof,
        utility_model_proof_review,
        utility_model_adoption,
        utility_model_adoption_review,
        utility_tuning_proof,
        utility_tuning_proof_review,
        utility_tuning_adoption,
        utility_tuning_adoption_review,
    ):
        self.config = config
        self.utility_selection_lab = utility_selection_lab
        self.utility_tuning_lab = utility_tuning_lab
        self.utility_model_lab = utility_model_lab
        self.utility_model_proof = utility_model_proof
        self.utility_model_proof_review = utility_model_proof_review
        self.utility_model_adoption = utility_model_adoption
        self.utility_model_adoption_review = utility_model_adoption_review
        self.utility_tuning_proof = utility_tuning_proof
        self.utility_tuning_proof_review = utility_tuning_proof_review
        self.utility_tuning_adoption = utility_tuning_adoption
        self.utility_tuning_adoption_review = utility_tuning_adoption_review
        self.root_dir = ensure_dir(Path(config.model_dir) / 'utility_operator_automation')
        self.sessions_dir = ensure_dir(self.root_dir / 'sessions')
        self.state_path = self.root_dir / 'state.json'
        self.summary_path = self.root_dir / 'latest_summary.json'
        self.pack_path = self.root_dir / 'latest_pack.zip'
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start_background_threads(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='utility-operator-automation')
        self._thread.start()

    def stop_background_threads(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def latest_summary(self) -> dict:
        return read_json(self.summary_path, {})

    def latest_state(self) -> dict:
        return read_json(self.state_path, {})

    def latest_pack(self) -> Path | None:
        return self.pack_path if self.pack_path.exists() else None

    def _session_dir(self, state: dict) -> Path:
        return ensure_dir(self.sessions_dir / str(state.get('session_id') or 'unknown-session'))

    def _session_snapshot_path(self, state: dict, name: str) -> Path:
        return self._session_dir(state) / f"{name}.json"

    def _write_session_snapshot(self, state: dict, name: str, summary: dict) -> None:
        path = self._session_snapshot_path(state, name)
        atomic_write_json(path, summary)
        snapshots = dict(state.get('offline_lab_snapshot_paths') or {})
        snapshots[name] = str(path)
        state['offline_lab_snapshot_paths'] = snapshots

    def _load_session_snapshot(self, state: dict, name: str) -> dict:
        snapshots = dict(state.get('offline_lab_snapshot_paths') or {})
        path = snapshots.get(name)
        if not path:
            candidate = self._session_snapshot_path(state, name)
            if candidate.exists():
                path = str(candidate)
        if not path:
            return {}
        return read_json(path, {})

    def _status_lab_payload(self, state: dict, name: str, latest_summary: dict) -> dict:
        session_summary = self._load_session_snapshot(state, name)
        if session_summary:
            payload = dict(session_summary)
            payload['session_fresh'] = True
            payload['status_source'] = 'session_snapshot'
            return payload
        latest = dict(latest_summary or {})
        if not latest:
            return {}
        return {
            'available': False,
            'headline': 'Lab output hidden until current-session freshness is proven',
            'summary': 'A latest lab summary exists on disk, but it is not being surfaced into automation status because it is not pinned to the current automation session.',
            'generated_at_utc': latest.get('generated_at_utc'),
            'app_version': latest.get('app_version'),
            'session_fresh': False,
            'status_source': 'redacted_stale_latest_summary',
            'stale_for_current_session': True,
        }

    def start(self) -> dict:
        state = self.latest_state()
        session_id = f"utility-auto-{uuid.uuid4().hex[:12]}"
        payload = {
            'active': True,
            'session_id': session_id,
            'app_version': APP_VERSION,
            'started_at_utc': _utc_now_iso(),
            'phase': 'running_utility_selection_lab_quick',
            'selected_branch': None,
            'attempted_branches': [],
            'offline_labs_completed': False,
            'last_action': 'start',
            'last_error': None,
            'current_operation': 'utility_selection_lab_quick',
            'current_operation_started_at_utc': _utc_now_iso(),
            'offline_lab_freshness': {
                'utility_selection_lab': False,
                'utility_tuning_lab': False,
                'utility_model_lab': False,
            },
            'selection_gate_verdict': None,
            'selection_quick_gate_verdict': None,
            'offline_lab_snapshot_paths': {},
        }
        if state.get('active'):
            payload['restarted_from_session_id'] = state.get('session_id')
        atomic_write_json(self.state_path, payload)
        self._wake.set()
        return self.refresh(reason='start')

    def stop(self) -> dict:
        state = self.latest_state()
        state.update({
            'active': False,
            'stopped_at_utc': _utc_now_iso(),
            'last_action': 'stop',
            'current_operation': None,
        })
        atomic_write_json(self.state_path, state)
        return self.refresh(reason='stop')

    def refresh(self, *, reason: str = 'manual_refresh') -> dict:
        state = self.latest_state()
        summary = self._build_summary(state=state, reason=reason)
        atomic_write_json(self.summary_path, summary)
        self._build_pack(summary)
        return summary

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=30)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self._tick()
            except Exception as exc:
                state = self.latest_state()
                if state:
                    state['last_error'] = str(exc)
                    state['last_action'] = 'tick_error'
                    state['phase'] = 'automation_error'
                    state['active'] = False
                    atomic_write_json(self.state_path, state)
                    self.refresh(reason='tick_error')

    def _run_selection_lab(self, state: dict) -> dict:
        quick_inputs = {'hours': 48, 'step_minutes': 120, 'max_scans': 24, 'max_symbols': 60}
        state['phase'] = 'running_utility_selection_lab_quick'
        state['current_operation'] = 'utility_selection_lab_quick'
        state['current_operation_inputs'] = quick_inputs
        state['current_operation_started_at_utc'] = _utc_now_iso()
        state['last_action'] = 'running_utility_selection_lab_quick'
        atomic_write_json(self.state_path, state)
        self.refresh(reason='running_utility_selection_lab_quick')
        quick_summary = self.utility_selection_lab.run(**quick_inputs)
        if not self._pin_selection_summary_or_block(state, quick_summary, snapshot_name='utility_selection_lab_quick'):
            return {}
        quick_verdict = str(quick_summary.get('verdict') or '')
        state['selection_quick_gate_verdict'] = quick_verdict
        if quick_verdict == 'legacy_engine_preferred_offline':
            state['selection_gate_verdict'] = quick_verdict
            state['phase'] = f"selection_gate_blocked_{quick_verdict}"
            state['last_action'] = 'selection_gate_blocked_quick'
            state['last_error'] = None
            state['active'] = False
            state['current_operation'] = None
            state['current_operation_inputs'] = None
            atomic_write_json(self.state_path, state)
            self.refresh(reason='selection_gate_blocked_quick')
            return quick_summary

        full_inputs = {'hours': 168, 'step_minutes': 120, 'max_scans': 84, 'max_symbols': 100}
        state['phase'] = 'running_utility_selection_lab_full'
        state['current_operation'] = 'utility_selection_lab_full'
        state['current_operation_inputs'] = full_inputs
        state['current_operation_started_at_utc'] = _utc_now_iso()
        state['last_action'] = 'running_utility_selection_lab_full'
        atomic_write_json(self.state_path, state)
        self.refresh(reason='running_utility_selection_lab_full')
        summary = self.utility_selection_lab.run(**full_inputs)
        if not self._pin_selection_summary_or_block(state, summary, snapshot_name='utility_selection_lab'):
            return {}
        state['selection_gate_verdict'] = str(summary.get('verdict') or '')
        if state['selection_gate_verdict'] != 'utility_engine_supported_offline':
            state['phase'] = f"selection_gate_blocked_{state['selection_gate_verdict'] or 'unknown'}"
            state['last_action'] = 'selection_gate_blocked'
            state['last_error'] = None
            state['active'] = False
            state['current_operation'] = None
            state['current_operation_inputs'] = None
            atomic_write_json(self.state_path, state)
            self.refresh(reason='selection_gate_blocked')
            return summary
        state['phase'] = 'running_utility_tuning_lab'
        state['current_operation'] = None
        state['current_operation_inputs'] = None
        state['last_action'] = 'selection_gate_passed'
        atomic_write_json(self.state_path, state)
        self.refresh(reason='selection_gate_passed')
        return summary

    def _pin_selection_summary_or_block(self, state: dict, summary: dict, *, snapshot_name: str) -> bool:
        freshness = dict(state.get('offline_lab_freshness') or {})
        fresh = _summary_matches_session(summary, state)
        freshness['utility_selection_lab'] = fresh
        state['offline_lab_freshness'] = freshness
        if not fresh:
            state['phase'] = 'selection_lab_not_session_fresh'
            state['last_action'] = 'selection_lab_not_session_fresh'
            state['last_error'] = 'automation_refused_non_session_fresh_selection_lab'
            state['active'] = False
            state['current_operation'] = None
            state['current_operation_inputs'] = None
            atomic_write_json(self.state_path, state)
            self.refresh(reason='selection_lab_not_session_fresh')
            return False
        self._write_session_snapshot(state, snapshot_name, summary)
        atomic_write_json(self.state_path, state)
        self.refresh(reason=f'{snapshot_name}_completed')
        return True

    def _run_tuning_lab(self, state: dict) -> dict:
        state['phase'] = 'running_utility_tuning_lab'
        state['current_operation'] = 'utility_tuning_lab'
        state['current_operation_started_at_utc'] = _utc_now_iso()
        state['last_action'] = 'running_utility_tuning_lab'
        atomic_write_json(self.state_path, state)
        self.refresh(reason='running_utility_tuning_lab')
        summary = self.utility_tuning_lab.run(hours=168, step_minutes=120, max_scans=84, max_symbols=100)
        freshness = dict(state.get('offline_lab_freshness') or {})
        fresh = _summary_matches_session(summary, state)
        freshness['utility_tuning_lab'] = fresh
        state['offline_lab_freshness'] = freshness
        if fresh:
            self._write_session_snapshot(state, 'utility_tuning_lab', summary)
            state['phase'] = 'running_utility_model_lab'
            state['last_action'] = 'utility_tuning_lab_completed'
            state['last_error'] = None
        else:
            state['phase'] = 'utility_tuning_lab_not_session_fresh'
            state['last_action'] = 'utility_tuning_lab_not_session_fresh'
            state['last_error'] = 'automation_refused_non_session_fresh_tuning_lab'
            state['active'] = False
            state['current_operation'] = None
        atomic_write_json(self.state_path, state)
        self.refresh(reason=state['last_action'])
        return summary

    def _run_model_lab(self, state: dict) -> dict:
        state['phase'] = 'running_utility_model_lab'
        state['current_operation'] = 'utility_model_lab'
        state['current_operation_started_at_utc'] = _utc_now_iso()
        state['last_action'] = 'running_utility_model_lab'
        atomic_write_json(self.state_path, state)
        self.refresh(reason='running_utility_model_lab')
        summary = self.utility_model_lab.run(max_symbols=100)
        freshness = dict(state.get('offline_lab_freshness') or {})
        fresh = _summary_matches_session(summary, state)
        freshness['utility_model_lab'] = fresh
        state['offline_lab_freshness'] = freshness
        if fresh:
            self._write_session_snapshot(state, 'utility_model_lab', summary)
            state['offline_labs_completed'] = True
            state['phase'] = 'offline_labs_completed'
            state['last_action'] = 'utility_model_lab_completed'
            state['last_error'] = None
        else:
            state['phase'] = 'utility_model_lab_not_session_fresh'
            state['last_action'] = 'utility_model_lab_not_session_fresh'
            state['last_error'] = 'automation_refused_non_session_fresh_model_lab'
            state['active'] = False
            state['current_operation'] = None
        atomic_write_json(self.state_path, state)
        self.refresh(reason=state['last_action'])
        return summary

    def _tick(self) -> None:
        state = self.latest_state()
        if not state.get('active'):
            return

        selection_summary = self._load_session_snapshot(state, 'utility_selection_lab')
        if not selection_summary:
            self._run_selection_lab(state)
            state = self.latest_state()
            if not state.get('active'):
                return

        tuning_summary = self._load_session_snapshot(state, 'utility_tuning_lab')
        if not tuning_summary:
            self._run_tuning_lab(state)
            state = self.latest_state()
            if not state.get('active'):
                return

        model_summary = self._load_session_snapshot(state, 'utility_model_lab')
        if not model_summary:
            self._run_model_lab(state)
            state = self.latest_state()
            if not state.get('active'):
                return

        state['offline_labs_completed'] = True
        state['current_operation'] = None
        if not state.get('selected_branch'):
            model_summary = self._load_session_snapshot(state, 'utility_model_lab')
            tuning_summary = self._load_session_snapshot(state, 'utility_tuning_lab')
            model_supported = str(model_summary.get('verdict') or '') == 'utility_model_candidate_supported_offline'
            tuning_supported = str(tuning_summary.get('verdict') or '') == 'utility_tuning_candidate_supported_offline'
            if model_supported:
                state['selected_branch'] = 'utility_model'
                state['phase'] = 'model_branch_selected'
                state['last_action'] = 'selected_utility_model_branch'
            elif tuning_supported:
                state['selected_branch'] = 'utility_tuning'
                state['phase'] = 'tuning_branch_selected'
                state['last_action'] = 'selected_utility_tuning_branch'
            else:
                state['phase'] = 'waiting_for_supported_offline_candidate'
                state['last_action'] = 'waiting_for_supported_offline_candidate'
                state['active'] = False
            state['current_operation'] = None
            atomic_write_json(self.state_path, state)
            self.refresh(reason=state['last_action'])
            if not state.get('active'):
                return

        changed = False
        if state.get('selected_branch') == 'utility_model':
            changed = self._advance_model_branch(state)
        elif state.get('selected_branch') == 'utility_tuning':
            changed = self._advance_tuning_branch(state)
        if changed:
            atomic_write_json(self.state_path, state)
            self.refresh(reason='tick_progress')
        if state.get('active'):
            self._wake.set()

    def _fallback_to_tuning(self, state: dict, *, reason: str) -> bool:
        attempts = list(state.get('attempted_branches') or [])
        if 'utility_model' not in attempts:
            attempts.append('utility_model')
            state['attempted_branches'] = attempts
        if 'utility_tuning' not in attempts:
            state['selected_branch'] = 'utility_tuning'
            state['phase'] = 'tuning_branch_selected'
            state['last_action'] = reason
            return True
        state['phase'] = 'completed_without_supported_path'
        state['last_action'] = reason
        state['active'] = False
        return True

    def _advance_model_branch(self, state: dict) -> bool:
        changed = False
        adoption_summary = self.utility_model_adoption.build_summary(reason='automation_tick')
        if str(adoption_summary.get('verdict') or '') == 'utility_model_adoption_active':
            review = self.utility_model_adoption_review.build_summary(reason='automation_tick')
            verdict = str(review.get('verdict') or '')
            if verdict == 'utility_model_adoption_review_supports_keeping_candidate':
                state['phase'] = 'completed_keep_model_adoption'
                state['last_action'] = 'model_adoption_supported_keep'
                state['active'] = False
                return True
            if verdict == 'utility_model_adoption_review_recommends_rollback':
                self.utility_model_adoption.clear()
                return self._fallback_to_tuning(state, reason='model_adoption_recommended_rollback')
            state['phase'] = 'waiting_for_model_adoption_review'
            state['last_action'] = 'waiting_for_model_adoption_review'
            return True

        proof_summary = self.utility_model_proof.build_summary(reason='automation_tick')
        proof_verdict = str(proof_summary.get('verdict') or '')
        if proof_verdict == 'utility_model_candidate_ready_for_live_proof':
            self.utility_model_proof.activate(proof_hours=24)
            state['phase'] = 'model_proof_active'
            state['last_action'] = 'activated_model_proof'
            return True
        if proof_verdict == 'utility_model_proof_window_active':
            review = self.utility_model_proof_review.build_summary(reason='automation_tick')
            verdict = str(review.get('verdict') or '')
            if verdict == 'utility_model_proof_supports_candidate':
                self.utility_model_adoption.activate()
                state['phase'] = 'model_adoption_active'
                state['last_action'] = 'activated_model_adoption'
                return True
            if verdict == 'utility_model_proof_rejects_candidate':
                self.utility_model_proof.clear()
                return self._fallback_to_tuning(state, reason='model_proof_rejected_candidate')
            state['phase'] = 'waiting_for_model_proof_review'
            state['last_action'] = 'waiting_for_model_proof_review'
            return True

        state['phase'] = 'waiting_for_model_branch_gate'
        state['last_action'] = 'waiting_for_model_branch_gate'
        return changed

    def _advance_tuning_branch(self, state: dict) -> bool:
        adoption_summary = self.utility_tuning_adoption.build_summary(reason='automation_tick')
        if str(adoption_summary.get('verdict') or '') == 'utility_tuning_adoption_active':
            review = self.utility_tuning_adoption_review.build_summary(reason='automation_tick')
            verdict = str(review.get('verdict') or '')
            if verdict == 'utility_tuning_adoption_review_supports_keeping_candidate':
                state['phase'] = 'completed_keep_tuning_adoption'
                state['last_action'] = 'tuning_adoption_supported_keep'
                state['active'] = False
                return True
            if verdict == 'utility_tuning_adoption_review_recommends_rollback':
                self.utility_tuning_adoption.clear()
                state['phase'] = 'completed_tuning_adoption_rollback'
                state['last_action'] = 'tuning_adoption_recommended_rollback'
                state['active'] = False
                return True
            state['phase'] = 'waiting_for_tuning_adoption_review'
            state['last_action'] = 'waiting_for_tuning_adoption_review'
            return True

        proof_summary = self.utility_tuning_proof.build_summary(reason='automation_tick')
        proof_verdict = str(proof_summary.get('verdict') or '')
        if proof_verdict == 'utility_tuning_candidate_ready_for_live_proof':
            self.utility_tuning_proof.activate(proof_hours=24)
            state['phase'] = 'tuning_proof_active'
            state['last_action'] = 'activated_tuning_proof'
            return True
        if proof_verdict == 'utility_tuning_proof_window_active':
            review = self.utility_tuning_proof_review.build_summary(reason='automation_tick')
            verdict = str(review.get('verdict') or '')
            if verdict == 'utility_tuning_proof_supports_candidate':
                self.utility_tuning_adoption.activate()
                state['phase'] = 'tuning_adoption_active'
                state['last_action'] = 'activated_tuning_adoption'
                return True
            if verdict == 'utility_tuning_proof_rejects_candidate':
                self.utility_tuning_proof.clear()
                state['phase'] = 'completed_tuning_proof_rejected'
                state['last_action'] = 'tuning_proof_rejected_candidate'
                state['active'] = False
                return True
            state['phase'] = 'waiting_for_tuning_proof_review'
            state['last_action'] = 'waiting_for_tuning_proof_review'
            return True

        state['phase'] = 'waiting_for_tuning_branch_gate'
        state['last_action'] = 'waiting_for_tuning_branch_gate'
        return False

    def _build_summary(self, *, state: dict, reason: str) -> dict:
        selection = self._status_lab_payload(state, 'utility_selection_lab', self.utility_selection_lab.latest_summary())
        tuning = self._status_lab_payload(state, 'utility_tuning_lab', self.utility_tuning_lab.latest_summary())
        model = self._status_lab_payload(state, 'utility_model_lab', self.utility_model_lab.latest_summary())
        freshness = dict(state.get('offline_lab_freshness') or {})
        model_proof = self.utility_model_proof.latest_summary()
        model_proof_review = self.utility_model_proof_review.latest_summary()
        model_adoption = self.utility_model_adoption.latest_summary()
        model_adoption_review = self.utility_model_adoption_review.latest_summary()
        tuning_proof = self.utility_tuning_proof.latest_summary()
        tuning_proof_review = self.utility_tuning_proof_review.latest_summary()
        tuning_adoption = self.utility_tuning_adoption.latest_summary()
        tuning_adoption_review = self.utility_tuning_adoption_review.latest_summary()
        phase = str(state.get('phase') or 'idle')
        current_op_started = _parse_iso(state.get('current_operation_started_at_utc'))
        elapsed_seconds = None
        if state.get('active') and current_op_started is not None:
            elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - current_op_started).total_seconds())
        headline = 'Utility operator automation idle'
        summary = 'Automation is not active.'
        recommended_action = 'start_utility_operator_automation'
        if state.get('active'):
            headline = 'Utility operator automation is running'
            summary = f"Current phase: {phase.replace('_', ' ')}."
            recommended_action = 'leave_it_running'
        elif phase.startswith('selection_gate_blocked_'):
            headline = 'Utility operator automation stopped at the first evidence gate'
            verdict = str(state.get('selection_gate_verdict') or 'unknown')
            summary = f'The current utility-selection redesign did not produce an explicit offline win for the utility engine (selection verdict: {verdict}), so automation refused to proceed into proof or adoption.'
            recommended_action = 'review_utility_selection_summary_before_restart'
        elif phase == 'automation_error':
            headline = 'Utility operator automation hit a runtime error'
            summary = f"Automation failed during {state.get('current_operation') or 'unknown operation'}: {state.get('last_error') or 'unknown error'}."
            recommended_action = 'deploy_runtime_hotfix_before_restart'
        elif phase.endswith('_not_session_fresh'):
            headline = 'Utility operator automation rejected non-session-fresh lab output'
            summary = 'A lab ran, but the resulting summary could not be safely attributed to the active session, so automation halted.'
            recommended_action = 'review_lab_pinning_hotfix'
        elif phase.startswith('completed_'):
            headline = 'Utility operator automation reached a terminal state'
            summary = f"Completed phase: {phase.replace('_', ' ')}."
            recommended_action = 'review_summary_and_packs'
        payload = {
            'available': True,
            'app_version': APP_VERSION,
            'generated_at_utc': _utc_now_iso(),
            'reason': reason,
            'headline': headline,
            'summary': summary,
            'recommended_action': recommended_action,
            'state': state,
            'offline_lab_freshness': freshness,
            'current_operation_elapsed_seconds': elapsed_seconds,
            'utility_selection_lab': selection,
            'utility_tuning_lab': tuning,
            'utility_model_lab': model,
            'utility_model_proof': model_proof,
            'utility_model_proof_review': model_proof_review,
            'utility_model_adoption': model_adoption,
            'utility_model_adoption_review': model_adoption_review,
            'utility_tuning_proof': tuning_proof,
            'utility_tuning_proof_review': tuning_proof_review,
            'utility_tuning_adoption': tuning_adoption,
            'utility_tuning_adoption_review': tuning_adoption_review,
        }
        return payload

    def _build_pack(self, summary: dict) -> None:
        state = self.latest_state()
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('utility_operator_automation_summary.json', _safe_json(summary))
            zf.writestr('utility_operator_automation_summary.txt', _safe_json(summary))
            zf.writestr('utility_operator_automation_state.json', _safe_json(state))
            for name in ('utility_selection_lab_quick', 'utility_selection_lab', 'utility_tuning_lab', 'utility_model_lab'):
                snap = self._load_session_snapshot(state, name)
                if snap:
                    zf.writestr(f'session_snapshots/{name}.json', _safe_json(snap))
