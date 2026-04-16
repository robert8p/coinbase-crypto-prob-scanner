from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .persist import atomic_write_json, ensure_dir, read_json
from .version import APP_VERSION


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ''):
            return None
        return float(value)
    except Exception:
        return None


def _rate(rows: list[dict], field: str) -> float | None:
    if not rows:
        return None
    vals = [int(r.get(field) or 0) for r in rows]
    return round(sum(vals) / len(vals), 6)


def _avg_metric(rows: list[dict], field: str) -> float | None:
    vals = [float(r.get(field)) for r in rows if _safe_float(r.get(field)) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b''
    buf = io.StringIO()
    fieldnames = sorted({k for row in rows for k in row.keys()})
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode('utf-8')


class SemanticsShadowOutcomeReviewService:
    def __init__(self, config: AppConfig, review_packs: Any, semantics_shadow_comparison_service: Any):
        self.config = config
        self.review_packs = review_packs
        self.semantics_shadow_comparison_service = semantics_shadow_comparison_service
        self.root_dir = ensure_dir(Path(config.model_dir) / 'semantics_shadow_outcome_review')
        self.summary_path = self.root_dir / 'latest_semantics_shadow_outcome_review_summary.json'
        self.pack_path = self.root_dir / 'latest_semantics_shadow_outcome_review_pack.zip'
        self.history_path = Path(config.model_dir) / 'semantics_shadow_comparison' / 'comparison_history.jsonl'

    def latest_summary(self) -> dict:
        summary = self._build_summary()
        atomic_write_json(self.summary_path, summary)
        if summary.get('pack_available'):
            self._build_pack(summary)
        return summary

    def latest_pack(self) -> Path | None:
        summary = self.latest_summary()
        return self.pack_path if summary.get('pack_available') and self.pack_path.exists() else None

    def _history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        rows = []
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

    def _match_run(self, item: dict[str, Any]) -> dict | None:
        existing = str(item.get('source_run_id') or '')
        if existing:
            try:
                run = self.review_packs.get_run(existing)
            except Exception:
                run = None
            if run:
                return run
        app_version = str(item.get('app_version') or APP_VERSION)
        generated_at = _parse_utc(item.get('generated_at_utc'))
        source_finished = _parse_utc(item.get('source_scan_finished_utc'))
        anchor = source_finished or generated_at
        try:
            runs = self.review_packs.get_runs_for_app_version(app_version, limit=500)
        except Exception:
            return None
        best = None
        best_delta = None
        for run in runs:
            finished = _parse_utc(run.get('scan_finished_utc'))
            if finished is None or anchor is None:
                continue
            delta = abs((anchor - finished).total_seconds())
            if delta > 1800:
                continue
            if best is None or delta < (best_delta or 10**18):
                best = run
                best_delta = delta
        return best

    def _resolved_rows_from_evaluated_pack(self, path: str) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        try:
            with zipfile.ZipFile(p) as zf:
                if 'outcomes.csv' not in zf.namelist():
                    return []
                raw = zf.read('outcomes.csv').decode('utf-8', errors='replace')
        except Exception:
            return []
        try:
            rows = list(csv.DictReader(io.StringIO(raw)))
        except Exception:
            return []
        return [dict(r) for r in rows if isinstance(r, dict)]

    def _load_resolved_rows_for_run(self, run_id: str, run_detail: dict[str, Any] | None) -> tuple[list[dict[str, Any]], str | None]:
        rows = self.review_packs._load_rows_for_run_ids([run_id], resolved_only=True)
        if rows:
            return rows, 'review_db'
        pack_path = str((run_detail or {}).get('latest_evaluated_pack_path') or '').strip()
        if pack_path:
            pack_rows = self._resolved_rows_from_evaluated_pack(pack_path)
            if pack_rows:
                return pack_rows, 'evaluated_pack'
        return [], None

    def _engine_rows(self, rows: list[dict], symbols: set[str], scan_id: str) -> list[dict]:
        out = []
        for row in rows:
            symbol = str(row.get('symbol') or '')
            out.append({
                'as_of_utc': scan_id,
                'row_type': 'visible' if symbol in symbols else 'hidden',
                'quality_touched': int(bool(row.get('quality_touched'))),
                'raw_touched': int(bool(row.get('raw_touched'))),
                'end_ret': _safe_float(row.get('end_ret')),
                'mae': _safe_float(row.get('mae')),
                'symbol': symbol,
            })
        return out

    def _matured_records(self) -> tuple[list[dict], int, int, dict[str, Any]]:
        horizon = int(getattr(self.config, 'target_horizon_minutes', 240) or 240)
        threshold = _utc_now() - timedelta(minutes=horizon)
        now = _utc_now()
        matured: list[dict] = []
        waiting = 0
        pending = 0
        pending_reason_counts: Counter[str] = Counter()
        resolved_source_counts: Counter[str] = Counter()
        for item in self._history():
            ts = _parse_utc(item.get('generated_at_utc'))
            if ts is None:
                continue
            if str(item.get('status') or '') != 'recorded':
                continue
            run = self._match_run(item)
            run_id = str((run or {}).get('run_id') or item.get('source_run_id') or '')
            run_detail = None
            if run_id:
                try:
                    run_detail = self.review_packs._load_run(run_id)
                except Exception:
                    run_detail = None
                if run_detail is None and run is not None:
                    run_detail = dict(run)
            due = _parse_utc((run_detail or {}).get('evaluation_due_utc'))
            if run_detail is not None and due is not None and now < due:
                waiting += 1
                continue
            if run_detail is None and ts > threshold:
                waiting += 1
                continue
            if not run_id:
                pending += 1
                pending_reason_counts['missing_run'] += 1
                continue
            rows, resolved_source = self._load_resolved_rows_for_run(run_id, run_detail)
            if not rows:
                if due is not None and now < due:
                    waiting += 1
                    continue
                pending += 1
                pending_reason_counts['resolved_rows_missing'] += 1
                continue
            resolved_source_counts[resolved_source or 'unknown'] += 1
            incumbent_symbols = {str(x) for x in ((item.get('incumbent') or {}).get('symbols') or []) if x}
            challenger_symbols = {str(x) for x in ((item.get('challenger') or {}).get('symbols') or []) if x}
            scan_id = str(item.get('generated_at_utc') or (run_detail or {}).get('scan_finished_utc') or run_id)
            matured.append({
                'summary': item,
                'run': run_detail or run or {},
                'run_id': run_id,
                'scan_id': scan_id,
                'resolved_source': resolved_source or 'unknown',
                'incumbent_rows': self._engine_rows(rows, incumbent_symbols, scan_id),
                'challenger_rows': self._engine_rows(rows, challenger_symbols, scan_id),
            })
        diagnostics = {
            'pending_reason_counts': dict(sorted(pending_reason_counts.items())),
            'resolved_source_counts': dict(sorted(resolved_source_counts.items())),
        }
        return matured, waiting, pending, diagnostics

    def _bucket(self, rows: list[dict]) -> dict[str, Any]:
        visible = [r for r in rows if str(r.get('row_type') or '') == 'visible']
        hidden = [r for r in rows if str(r.get('row_type') or '') != 'visible']
        return {
            'visible_count': len(visible),
            'hidden_count': len(hidden),
            'visible_quality_hit_rate': _rate(visible, 'quality_touched'),
            'hidden_quality_hit_rate': _rate(hidden, 'quality_touched'),
            'visible_raw_hit_rate': _rate(visible, 'raw_touched'),
            'hidden_raw_hit_rate': _rate(hidden, 'raw_touched'),
            'visible_avg_end_ret': _avg_metric(visible, 'end_ret'),
            'hidden_avg_end_ret': _avg_metric(hidden, 'end_ret'),
            'visible_avg_mae': _avg_metric(visible, 'mae'),
            'hidden_avg_mae': _avg_metric(hidden, 'mae'),
            'visible_vs_hidden_quality_gap': round(((_rate(visible, 'quality_touched') or 0.0) - (_rate(hidden, 'quality_touched') or 0.0)), 6) if (visible or hidden) else None,
        }

    def _pairwise(self, matured: list[dict]) -> dict[str, Any]:
        if not matured:
            return {'scan_count': 0, 'challenger_quality_win_rate': None, 'challenger_end_ret_win_rate': None}
        quality_wins = 0
        end_ret_wins = 0
        ties = 0
        for item in matured:
            inc = self._bucket(item['incumbent_rows'])
            ch = self._bucket(item['challenger_rows'])
            inc_q = _safe_float(inc.get('visible_quality_hit_rate'))
            ch_q = _safe_float(ch.get('visible_quality_hit_rate'))
            inc_e = _safe_float(inc.get('visible_avg_end_ret'))
            ch_e = _safe_float(ch.get('visible_avg_end_ret'))
            if ch_q is not None and inc_q is not None:
                if ch_q > inc_q:
                    quality_wins += 1
                elif ch_q == inc_q:
                    ties += 1
            if ch_e is not None and inc_e is not None and ch_e > inc_e:
                end_ret_wins += 1
        n = len(matured)
        return {
            'scan_count': n,
            'challenger_quality_win_rate': round(quality_wins / n, 6),
            'challenger_end_ret_win_rate': round(end_ret_wins / n, 6),
            'quality_tie_count': ties,
        }

    def _build_summary(self) -> dict:
        matured, waiting, pending, diagnostics = self._matured_records()
        generated_at = _utc_now_iso()
        shadow_latest = self.semantics_shadow_comparison_service.latest_summary() or {}
        if not matured:
            summary = {
                'available': True,
                'generated_at_utc': generated_at,
                'app_version': APP_VERSION,
                'headline': 'Semantics shadow outcome review waiting for resolved rows',
                'summary': 'The contract-aligned semantics challenger is being recorded in shadow, but there are not yet any matured comparisons with resolved rows to review.',
                'status': 'waiting',
                'matured_comparisons': 0,
                'waiting_for_maturity': waiting,
                'pending_resolution': pending,
                'diagnostics': diagnostics,
                'shadow_latest_summary': {
                    'headline': shadow_latest.get('headline'),
                    'status': shadow_latest.get('status'),
                },
                'pack_available': False,
            }
            return summary

        incumbent_rows = [row for item in matured for row in item['incumbent_rows']]
        challenger_rows = [row for item in matured for row in item['challenger_rows']]
        incumbent = self._bucket(incumbent_rows)
        challenger = self._bucket(challenger_rows)
        pairwise = self._pairwise(matured)
        challenger_leads = (
            (_safe_float(challenger.get('visible_quality_hit_rate')) or 0.0) > (_safe_float(incumbent.get('visible_quality_hit_rate')) or 0.0)
            and (_safe_float(challenger.get('visible_vs_hidden_quality_gap')) or 0.0) >= (_safe_float(incumbent.get('visible_vs_hidden_quality_gap')) or 0.0)
        )
        headline = 'Contract-aligned semantics challenger leads legacy in the matured shadow window' if challenger_leads else 'Legacy still leads or the shadow result is inconclusive so far'
        summary = {
            'available': True,
            'generated_at_utc': generated_at,
            'app_version': APP_VERSION,
            'headline': headline,
            'summary': 'Resolved rows from the same completed scans were compared across the unchanged live legacy shortlist and the contract-aligned semantics challenger recorded in shadow.',
            'status': 'completed',
            'matured_comparisons': len(matured),
            'waiting_for_maturity': waiting,
            'pending_resolution': pending,
            'incumbent': incumbent,
            'challenger': challenger,
            'pairwise': pairwise,
            'challenger_leads_now': challenger_leads,
            'live_path_unchanged': True,
            'shadow_latest_summary': {
                'headline': shadow_latest.get('headline'),
                'status': shadow_latest.get('status'),
            },
            'diagnostics': diagnostics,
            'pack_available': True,
        }
        return summary

    def _build_pack(self, summary: dict) -> None:
        matured, _waiting, _pending, _diagnostics = self._matured_records()
        incumbent_rows = [row for item in matured for row in item['incumbent_rows']]
        challenger_rows = [row for item in matured for row in item['challenger_rows']]
        with zipfile.ZipFile(self.pack_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('latest_semantics_shadow_outcome_review_summary.json', json.dumps(summary, indent=2, sort_keys=True))
            zf.writestr('matured_incumbent_rows.csv', _csv_bytes(incumbent_rows))
            zf.writestr('matured_challenger_rows.csv', _csv_bytes(challenger_rows))
            zf.writestr('matured_history.json', json.dumps(matured, indent=2, sort_keys=True))
            zf.writestr('README.txt', (
                'Semantics Shadow Outcome Review Pack\n\n'
                'This pack compares resolved outcomes for the unchanged live legacy shortlist and the contract-aligned semantics challenger recorded in shadow on the same completed scans.\n\n'
                f"Generated: {summary.get('generated_at_utc')}\n"
                f"Matured comparisons: {summary.get('matured_comparisons')}\n"
            ))
