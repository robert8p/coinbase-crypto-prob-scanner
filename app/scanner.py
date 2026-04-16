from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from .binance_client import BinanceClient
from .coinbase_client import CoinbaseClient
from .config import AppConfig
from .decision_branch_automation import effective_live_raw_threshold
from .live_candidate_proof import load_active_live_candidate_override
from .features import FEATURE_COLUMNS, compute_guardrails, compute_live_features, heuristic_probability, stage1_rank, stage1_select
from .live_scoring import apply_live_post_model_adjustments
from .modeling import ModelBundle, build_model_status_summary, reconcile_runtime_metadata
from .paper_trade import PaperTradeService
from .review_runs import ReviewPackService
from .regime import assess_market_regime_readiness, build_market_regime, classify_liquidity_tier, live_policy_for, mark_market_regime_applied, pending_market_regime
from .state import AppState
from .utility_shortlist import annotate_rows_for_utility, optimize_visible_shortlist, load_active_utility_tuning_override, utility_config_with_runtime_override
from .universe import UniverseBuilder

logger = logging.getLogger(__name__)

from .version import APP_VERSION
from .objective_semantics import load_objective_semantics_contract, score_objective_band


@dataclass(slots=True)
class ScanArtifacts:
    scores: List[dict]
    informational_rows: List[dict]
    informational_overflow_rows: List[dict]
    coverage: dict
    status_updates: dict
    suppressed_rows: List[dict]
    trimmed_visible_rows: List[dict]


class ScannerService:
    def __init__(self, config: AppConfig, state: AppState, client: CoinbaseClient, paper_trade: PaperTradeService | None = None, review_packs: ReviewPackService | None = None, shadow_selection_comparison_service: Any | None = None, semantics_shadow_comparison_service: Any | None = None):
        self.config = config
        self.state = state
        self.client = client
        self.paper_trade = paper_trade
        self.review_packs = review_packs
        self.shadow_selection_comparison_service = shadow_selection_comparison_service
        self.semantics_shadow_comparison_service = semantics_shadow_comparison_service
        self.model_output_distribution_service = None
        self.binance = BinanceClient(timeout=config.http_timeout_seconds, pause=config.request_pause_seconds)
        self._scheduler_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._current_thread: threading.Thread | None = None
        self._followup_lock = threading.RLock()
        self._followup_token = 0
        self.stage1_opportunity = None
        self._objective_semantics_cache: dict | None = None
        self._objective_semantics_cache_ts: float = 0.0

    def _locked_live_cohort(self) -> List[str] | None:
        if str(self.config.live_universe_mode).lower() != "trained_cohort":
            return None
        model_meta = self.state.model_metadata.get("pt2") or {}
        if not model_meta.get("trained"):
            return None
        raw = model_meta.get("trained_cohort_symbols") or model_meta.get("training_symbols_used") or []
        stable_bases = {"USDT", "USDC", "DAI", "PYUSD", "USDP", "EURC", "USDS", "GUSD", "FDUSD", "TUSD", "RLUSD", "USD1"}
        cohort = []
        for symbol in raw:
            s = str(symbol)
            if not s:
                continue
            base = s.split("-", 1)[0].upper() if "-" in s else s.upper()
            if self.config.stablecoin_exclusion_enabled and base in stable_bases:
                continue
            cohort.append(s)
        return cohort or None

    def _selection_label(self, locked_symbols: List[str] | None) -> str:
        if locked_symbols:
            return "trained_cohort_locked"
        mode = str(self.config.live_universe_mode).lower()
        return "dynamic_fallback" if mode == "trained_cohort" else "dynamic"

    def _parse_utc(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def _set_followup_status(self, payload: dict | None = None) -> None:
        payload = dict(payload or {})
        payload.setdefault("scheduled", False)
        payload.setdefault("reason", None)
        payload.setdefault("trigger", None)
        payload.setdefault("run_after_utc", None)
        payload.setdefault("tracked_symbols", [])
        payload.setdefault("tracked_count", len(payload.get("tracked_symbols") or []))
        payload.setdefault("source_scan_finished_utc", None)
        payload.setdefault("campaign_run_count", 0)
        payload.setdefault("campaign_unique_symbols", 0)
        payload.setdefault("max_wait_minutes", int(getattr(self.config, "cooldown_followup_scan_max_wait_minutes", 120) or 120))
        self.state.update_status(follow_up_scan=payload)

    def _cancel_followup_scan(self, *, reason: str = "cleared") -> None:
        with self._followup_lock:
            self._followup_token += 1
        self._set_followup_status({"scheduled": False, "reason": reason})

    def _schedule_followup_scan(self, *, run_after: datetime, reason: str, trigger: str = "cooldown_followup", tracked_symbols: List[str] | None = None, source_scan_finished_utc: str | None = None, sequence: str = "primary", campaign_run_count: int = 0, campaign_unique_symbols: int = 0) -> None:
        run_after = run_after.astimezone(timezone.utc)
        tracked_symbols = list(tracked_symbols or [])
        max_wait_minutes = int(getattr(self.config, "cooldown_followup_scan_max_wait_minutes", 120) or 120)
        with self._followup_lock:
            self._followup_token += 1
            token = self._followup_token
        self._set_followup_status({
            "scheduled": True,
            "reason": reason,
            "trigger": trigger,
            "sequence": sequence,
            "run_after_utc": run_after.isoformat(),
            "scheduled_at_utc": datetime.now(timezone.utc).isoformat(),
            "tracked_symbols": tracked_symbols,
            "tracked_count": len(tracked_symbols),
            "source_scan_finished_utc": source_scan_finished_utc,
            "campaign_run_count": int(campaign_run_count or 0),
            "campaign_unique_symbols": int(campaign_unique_symbols or 0),
            "max_wait_minutes": max_wait_minutes,
        })

        def _runner(local_token: int, when: datetime) -> None:
            delay = max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            if self._stop.wait(delay):
                return
            with self._followup_lock:
                if local_token != self._followup_token:
                    return
            self._set_followup_status({
                "scheduled": False,
                "reason": reason,
                "trigger": trigger,
                "sequence": sequence,
                "run_after_utc": when.isoformat(),
                "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
                "tracked_symbols": tracked_symbols,
                "tracked_count": len(tracked_symbols),
                "source_scan_finished_utc": source_scan_finished_utc,
                "campaign_run_count": int(campaign_run_count or 0),
                "campaign_unique_symbols": int(campaign_unique_symbols or 0),
                "max_wait_minutes": max_wait_minutes,
            })
            self.trigger_scan(trigger)

        threading.Thread(target=_runner, args=(token, run_after), daemon=True, name=f"{trigger}-scheduler").start()

    def _recover_persisted_followup(self, *, startup: bool = False) -> bool:
        status = self.state.get_status()
        plan = dict(status.get("follow_up_scan") or {})
        if not bool(plan.get("scheduled")):
            return False
        run_after = self._parse_utc(plan.get("run_after_utc"))
        if run_after is None:
            return False
        trigger = str(plan.get("trigger") or "cooldown_followup")
        reason = str(plan.get("reason") or ("cooldown_followup_resume" if startup else "cooldown_followup_catchup"))
        sequence = str(plan.get("sequence") or "primary")
        tracked_symbols = list(plan.get("tracked_symbols") or [])
        source_scan_finished_utc = plan.get("source_scan_finished_utc")
        campaign_run_count = int(plan.get("campaign_run_count") or 0)
        campaign_unique_symbols = int(plan.get("campaign_unique_symbols") or len(tracked_symbols))
        now = datetime.now(timezone.utc)
        if run_after <= now:
            self._set_followup_status({
                "scheduled": False,
                "reason": reason,
                "trigger": trigger,
                "sequence": sequence,
                "run_after_utc": run_after.isoformat(),
                "triggered_at_utc": now.isoformat(),
                "tracked_symbols": tracked_symbols,
                "tracked_count": len(tracked_symbols),
                "source_scan_finished_utc": source_scan_finished_utc,
                "campaign_run_count": campaign_run_count,
                "campaign_unique_symbols": campaign_unique_symbols,
                "recovered": True,
            })
            return self.trigger_scan(trigger)
        if startup:
            self._schedule_followup_scan(
                run_after=run_after,
                reason=reason,
                trigger=trigger,
                tracked_symbols=tracked_symbols,
                source_scan_finished_utc=source_scan_finished_utc,
                sequence=sequence,
                campaign_run_count=campaign_run_count,
                campaign_unique_symbols=campaign_unique_symbols,
            )
            return True
        return False

    def _maybe_schedule_followup_scan(self, *, trigger: str = "scheduler") -> None:
        if not bool(getattr(self.config, "cooldown_followup_scan_enabled", True)):
            self._cancel_followup_scan(reason="disabled")
            return
        status = self.state.get_status()
        regime = status.get("market_regime") or {}
        decision = status.get("decision_summary") or {}
        blocked_ctx = status.get("blocked_monitoring_context") or {}
        followup_comparison = status.get("followup_comparison") or {}
        cooldown_until = self._parse_utc(regime.get("cooldown_until_utc"))
        now = datetime.now(timezone.utc)

        # Step 1: schedule the primary cooldown-expiry follow-up even when the wait is long,
        # so blocked runs do not become dead ends during extended cooldowns.
        if bool(regime.get("cooldown_active")) and cooldown_until is not None:
            delay_seconds = (cooldown_until - now).total_seconds() + float(getattr(self.config, "cooldown_followup_scan_delay_seconds", 10) or 10)
            max_wait_seconds = max(0.0, float(getattr(self.config, "cooldown_followup_scan_max_wait_minutes", 120) or 120) * 60.0)
            min_blocked = max(1, int(getattr(self.config, "cooldown_followup_scan_min_blocked_rows", 3) or 3))
            blocked_rows = int(decision.get("blocked_rows") or 0)
            tracked_symbols = list(blocked_ctx.get("tracked_symbols") or [])
            if delay_seconds <= 0:
                self.state.update_status(cooldown_campaign=self._empty_cooldown_campaign())
                self._cancel_followup_scan(reason="cooldown_elapsed")
                return
            if blocked_rows < min_blocked:
                self.state.update_status(cooldown_campaign=self._empty_cooldown_campaign())
                self._cancel_followup_scan(reason="insufficient_blocked_rows")
                return
            if delay_seconds > max_wait_seconds and max_wait_seconds > 0:
                self.state.update_status(cooldown_campaign=self._empty_cooldown_campaign())
                self._cancel_followup_scan(reason="cooldown_wait_exceeds_cap")
                return
            existing_campaign = status.get("cooldown_campaign") or {}
            run_after = now + timedelta(seconds=delay_seconds)
            merged_campaign = self._merge_cooldown_campaign(existing_campaign, blocked_ctx, run_after_utc=run_after.isoformat())
            self.state.update_status(cooldown_campaign=merged_campaign)
            tracked_symbols = list(merged_campaign.get("tracked_symbols") or tracked_symbols)
            schedule_reason = "cooldown_expiry_followup" if trigger not in {"cooldown_followup", "cooldown_followup_confirmation"} else "cooldown_extended_followup"
            sequence = "primary" if trigger not in {"cooldown_followup", "cooldown_followup_confirmation"} else "extended"
            self._schedule_followup_scan(
                run_after=run_after,
                reason=schedule_reason,
                tracked_symbols=tracked_symbols,
                source_scan_finished_utc=merged_campaign.get("latest_source_run_finished_utc") or blocked_ctx.get("source_run_finished_utc"),
                sequence=sequence,
                campaign_run_count=int(merged_campaign.get("merged_from_runs") or 0),
                campaign_unique_symbols=int(merged_campaign.get("merged_unique_symbols") or len(tracked_symbols)),
            )
            return

        # Step 2: after the primary follow-up runs and cooldown is gone, schedule one short confirmation
        # recheck if tracked names improved but have not yet become visible.
        confirmation_enabled = bool(getattr(self.config, "cooldown_followup_confirmation_enabled", True))
        if trigger == "cooldown_followup" and confirmation_enabled and bool(followup_comparison.get("available")):
            visible_now = int(followup_comparison.get("visible_now_count") or 0)
            still_blocked = int(followup_comparison.get("still_blocked_count") or 0)
            near_visibility_now = int(followup_comparison.get("near_visibility_now_count") or 0)
            improved_live_count = int(followup_comparison.get("improved_live_count") or 0)
            min_improved_live = max(1, int(getattr(self.config, "cooldown_followup_confirmation_min_improved_live", 1) or 1))
            tracked_symbols = list((self._active_followup_context() or {}).get("tracked_symbols") or list((status.get("follow_up_scan") or {}).get("tracked_symbols") or []))
            if visible_now == 0 and still_blocked > 0 and (near_visibility_now > 0 or improved_live_count >= min_improved_live) and tracked_symbols:
                delay_minutes = max(1, int(getattr(self.config, "cooldown_followup_confirmation_delay_minutes", 10) or 10))
                self._schedule_followup_scan(
                    run_after=now + timedelta(minutes=delay_minutes),
                    reason="post_cooldown_confirmation",
                    trigger="cooldown_followup_confirmation",
                    tracked_symbols=tracked_symbols,
                    source_scan_finished_utc=(blocked_ctx or {}).get("source_run_finished_utc"),
                    sequence="confirmation",
                )
                return

        self.state.update_status(cooldown_campaign=self._empty_cooldown_campaign())
        self._cancel_followup_scan(reason="cooldown_inactive")

    def _active_blocked_monitoring_context(self) -> dict:
        ctx = dict((self.state.get_status().get("blocked_monitoring_context") or {}))
        tracked = list(ctx.get("tracked_symbols") or [])
        if not bool(ctx.get("context_active")) or not tracked:
            return {}
        max_age_seconds = max(0, int(getattr(self.config, "cooldown_followup_context_max_age_minutes", 180) or 180) * 60)
        source_finished = self._parse_utc(ctx.get("source_run_finished_utc"))
        if source_finished is not None and max_age_seconds > 0:
            age_seconds = (datetime.now(timezone.utc) - source_finished).total_seconds()
            if age_seconds > max_age_seconds:
                return {}
        return ctx

    def _resolve_review_runs_catchup(self, *, phase: str) -> int:
        if self.review_packs is None:
            return 0
        batch_runs = max(1, int(getattr(self.config, "review_resolve_batch_runs", 50) or 50))
        max_loops = max(1, int(getattr(self.config, "review_resolve_max_loops", 4) or 4))
        total_resolved = 0
        for loop_idx in range(max_loops):
            resolved_rows = int(self.review_packs.resolve_due_runs(max_runs=batch_runs) or 0)
            if resolved_rows <= 0:
                break
            total_resolved += resolved_rows
            logger.info("review_runs_resolved phase=%s loop=%d rows=%d", phase, loop_idx + 1, resolved_rows)
        return total_resolved

    def _empty_cooldown_campaign(self) -> dict:
        return {
            "active": False,
            "cooldown_until_utc": None,
            "run_after_utc": None,
            "tracked_symbols": [],
            "tracked_rows": [],
            "tracked_count": 0,
            "merged_from_runs": 0,
            "source_runs": [],
            "latest_source_run_finished_utc": None,
            "reason": "cleared",
        }

    def _active_cooldown_campaign(self) -> dict:
        campaign = dict((self.state.get_status().get("cooldown_campaign") or {}))
        tracked = list(campaign.get("tracked_symbols") or [])
        if not bool(campaign.get("active")) or not tracked:
            return {}
        max_age_seconds = max(0, int(getattr(self.config, "cooldown_followup_context_max_age_minutes", 180) or 180) * 60)
        latest_source = self._parse_utc(campaign.get("latest_source_run_finished_utc") or campaign.get("source_run_finished_utc"))
        if latest_source is not None and max_age_seconds > 0:
            age_seconds = (datetime.now(timezone.utc) - latest_source).total_seconds()
            if age_seconds > max_age_seconds:
                return {}
        return campaign

    def _active_followup_context(self) -> dict:
        campaign = self._active_cooldown_campaign()
        if campaign:
            return campaign
        return self._active_blocked_monitoring_context()

    def _merge_cooldown_campaign(self, existing: dict | None, new_context: dict | None, *, run_after_utc: str | None = None) -> dict:
        existing = dict(existing or {})
        new_context = dict(new_context or {})
        if not bool(new_context.get("context_active")):
            return self._empty_cooldown_campaign()
        cooldown_until_utc = new_context.get("cooldown_until_utc") or existing.get("cooldown_until_utc")
        if not cooldown_until_utc:
            return self._empty_cooldown_campaign()
        same_window = bool(existing.get("active")) and str(existing.get("cooldown_until_utc") or "") == str(cooldown_until_utc)
        candidate_rows = []
        if same_window:
            candidate_rows.extend(list(existing.get("tracked_rows") or []))
        candidate_rows.extend(list(new_context.get("tracked_rows") or []))
        by_symbol: Dict[str, dict] = {}
        for row in candidate_rows:
            symbol = str((row or {}).get("symbol") or "")
            if not symbol:
                continue
            normalized = dict(row)
            current = by_symbol.get(symbol)
            if current is None or self._blocked_focus_sort_key(normalized) > self._blocked_focus_sort_key(current):
                by_symbol[symbol] = normalized
        tracked_limit = max(
            max(1, int(getattr(self.config, "cooldown_followup_track_top_n", 5) or 5)),
            max(1, int(getattr(self.config, "cooldown_campaign_max_tracked_symbols", 12) or 12)),
        )
        tracked_rows_all = sorted(by_symbol.values(), key=self._blocked_focus_sort_key, reverse=True)
        tracked_rows = tracked_rows_all[:tracked_limit]
        max_runs = max(1, int(getattr(self.config, "cooldown_campaign_max_source_runs", 8) or 8))
        source_runs = []
        seen_run_keys = set()
        if same_window:
            for item in list(existing.get("source_runs") or []):
                finished = str((item or {}).get("source_run_finished_utc") or "")
                if finished and finished not in seen_run_keys:
                    source_runs.append(dict(item))
                    seen_run_keys.add(finished)
        new_run = {
            "source_run_finished_utc": new_context.get("source_run_finished_utc"),
            "market_regime_state": new_context.get("market_regime_state"),
            "market_regime_actionability": new_context.get("market_regime_actionability"),
            "tracked_count": int(new_context.get("tracked_count") or len(list(new_context.get("tracked_rows") or []))),
            "tracked_symbols": list(new_context.get("tracked_symbols") or []),
        }
        finished = str(new_run.get("source_run_finished_utc") or "")
        if finished and finished not in seen_run_keys:
            source_runs.append(new_run)
        source_runs = source_runs[-max_runs:]
        latest_source_run_finished_utc = source_runs[-1].get("source_run_finished_utc") if source_runs else new_context.get("source_run_finished_utc")
        return {
            "active": bool(tracked_rows),
            "cooldown_until_utc": cooldown_until_utc,
            "run_after_utc": run_after_utc or existing.get("run_after_utc"),
            "tracked_symbols": [r.get("symbol") for r in tracked_rows if r.get("symbol")],
            "tracked_rows": tracked_rows,
            "tracked_count": len(tracked_rows),
            "merged_unique_symbols": len(tracked_rows_all),
            "merged_from_runs": len(source_runs),
            "source_runs": source_runs,
            "latest_source_run_finished_utc": latest_source_run_finished_utc,
            "reason": "cooldown_campaign",
            "market_regime_state": new_context.get("market_regime_state"),
            "market_regime_actionability": new_context.get("market_regime_actionability"),
        }

    def _apply_followup_candidate_reserve(self, stage1_candidates: List[str], stage1_input_rows: Dict[str, dict], stage1_guardrails: Dict[str, dict], tracked_symbols: List[str] | None) -> tuple[List[str], dict]:
        reserve_n = max(0, int(getattr(self.config, "cooldown_followup_stage1_reserve_count", 5) or 5))
        requested = [str(s) for s in (tracked_symbols or []) if s]
        existing = set(stage1_candidates)
        eligible = []
        missing = []
        blocked = []
        injected = []
        already_present = []
        for symbol in requested:
            if symbol not in stage1_input_rows:
                missing.append(symbol)
                continue
            guard = stage1_guardrails.get(symbol) or {}
            if str(guard.get("block_code") or "") == "BLOCKED":
                blocked.append(symbol)
                continue
            eligible.append(symbol)
            if symbol in existing:
                already_present.append(symbol)
                continue
            if len(injected) < reserve_n:
                stage1_candidates.append(symbol)
                existing.add(symbol)
                injected.append(symbol)
        meta = {
            "triggered": bool(requested and reserve_n > 0),
            "requested_symbols": requested,
            "eligible_symbols": eligible,
            "missing_symbols": missing,
            "blocked_symbols": blocked,
            "already_present_symbols": already_present,
            "injected_symbols": injected,
            "reserve_count": reserve_n,
        }
        return stage1_candidates, meta

    def _build_blocked_monitoring_context(self, *, trigger: str, market_regime, blocked_rows: List[dict], decision_summary: dict, effective_market_regime_actionability: str | None = None) -> dict:
        tracked_n = max(1, int(getattr(self.config, "cooldown_followup_track_top_n", 5) or 5))
        blocked_sorted = [dict(r) for r in (blocked_rows or []) if r.get("symbol")]
        blocked_sorted.sort(key=self._blocked_focus_sort_key, reverse=True)
        tracked_rows = []
        for row in blocked_sorted[:tracked_n]:
            tracked_rows.append({
                "symbol": row.get("symbol"),
                "pre_policy_score": row.get("pre_policy_score"),
                "live_score": row.get("live_score"),
                "live_threshold": row.get("live_threshold"),
                "pre_policy_distance_to_validated": row.get("pre_policy_distance_to_validated"),
                "distance_to_live_threshold": row.get("distance_to_live_threshold"),
                "pre_policy_score_band": row.get("pre_policy_score_band") or row.get("score_band"),
                "visibility_band": row.get("visibility_band"),
                "liquidity_tier": row.get("liquidity_tier"),
                "pre_policy_rank": row.get("pre_policy_rank") or row.get("candidate_rank_all") or row.get("informational_rank"),
                "suppression_reason": row.get("suppression_reason"),
                "suppression_reason_detail": row.get("suppression_reason_detail") or row.get("policy_constraint_reason"),
            })
        active = bool(tracked_rows) and int(decision_summary.get("blocked_rows") or 0) > 0
        return {
            "context_active": active,
            "source_trigger": trigger,
            "source_run_finished_utc": datetime.now(timezone.utc).isoformat(),
            "market_regime_state": getattr(market_regime, "state", None),
            "market_regime_actionability": effective_market_regime_actionability if effective_market_regime_actionability is not None else getattr(market_regime, "actionability_state", None),
            "cooldown_active": bool(getattr(market_regime, "cooldown_active", False)),
            "cooldown_until_utc": getattr(market_regime, "cooldown_until_utc", None),
            "tracked_symbols": [r.get("symbol") for r in tracked_rows if r.get("symbol")],
            "tracked_rows": tracked_rows,
            "tracked_count": len(tracked_rows),
            "reason": "policy_blocked_monitoring" if active else "cleared",
        }

    def _derive_row_type(self, row: dict | None) -> str:
        row = dict(row or {})
        explicit = str(row.get("row_type") or "").strip().lower()
        if explicit:
            return explicit
        bucket = str(row.get("review_bucket") or row.get("display_bucket") or "").lower()
        suppression_reason = str(row.get("suppression_reason") or "").strip().lower()
        informational_only = bool(row.get("informational_only"))
        if bucket == "informational_overflow" or "overflow" in bucket:
            return "overflow"
        if bucket in {"informational_retained", "display_trim"}:
            return "informational"
        if bucket == "informational_suppressed":
            return "suppressed" if suppression_reason in {"threshold", "regime", "cooldown"} else "informational"
        if suppression_reason in {"threshold", "regime", "cooldown"}:
            return "suppressed"
        if informational_only:
            return "informational"
        if str(row.get("visibility_band") or "") == "cleared_visible_threshold":
            return "visible"
        return "visible"

    def _is_visible_now(self, row: dict | None) -> bool:
        row = dict(row or {})
        row_type = self._derive_row_type(row)
        if row_type == "visible":
            return True
        if row_type in {"suppressed", "informational", "overflow"}:
            return False
        if bool(row.get("informational_only")):
            return False
        if str(row.get("suppression_reason") or ""):
            return False
        return str(row.get("visibility_band") or "") == "cleared_visible_threshold"

    def _build_followup_comparison(self, *, trigger: str, prior_context: dict, current_rows: List[dict]) -> dict:
        empty = {
            "available": False,
            "tracked_count": 0,
            "visible_now_count": 0,
            "still_blocked_count": 0,
            "missing_count": 0,
            "near_visibility_now_count": 0,
            "near_validated_now_count": 0,
            "improved_live_count": 0,
            "improved_pre_policy_count": 0,
            "comparison_mode": None,
            "comparison_reason": None,
            "tracked_visible_rows": [],
            "tracked_visible_symbols": [],
            "top_changes": [],
        }
        if not prior_context or not bool(prior_context.get("context_active")):
            return empty
        now = datetime.now(timezone.utc)
        cooldown_until = self._parse_utc(prior_context.get("cooldown_until_utc"))
        explicit_followup = trigger in {"cooldown_followup", "cooldown_followup_confirmation"}
        overdue_catchup = (cooldown_until is not None and now >= cooldown_until and trigger not in {"cooldown_followup", "cooldown_followup_confirmation"})
        if not explicit_followup and not overdue_catchup:
            return empty
        priority = {"visible": 4, "suppressed": 3, "informational": 2, "overflow": 1}
        by_symbol: Dict[str, dict] = {}
        for row in current_rows:
            symbol = str(row.get("symbol") or "")
            if not symbol:
                continue
            normalized = dict(row)
            normalized["row_type"] = self._derive_row_type(normalized)
            current = by_symbol.get(symbol)
            if current is None or priority.get(str(normalized.get("row_type") or ""), 0) > priority.get(str(current.get("row_type") or ""), 0):
                by_symbol[symbol] = normalized
        changes = []
        tracked_visible_rows = []
        visible_now_count = 0
        still_blocked_count = 0
        missing_count = 0
        near_visibility_now_count = 0
        near_validated_now_count = 0
        improved_live_count = 0
        improved_pre_policy_count = 0
        for prev in list(prior_context.get("tracked_rows") or []):
            symbol = str(prev.get("symbol") or "")
            if not symbol:
                continue
            cur = by_symbol.get(symbol)
            missing_current = cur is None
            current_row_type = self._derive_row_type(cur)
            became_visible = bool(cur and self._is_visible_now(cur))
            still_blocked = bool(cur and not became_visible)
            if became_visible:
                visible_now_count += 1
            elif still_blocked:
                still_blocked_count += 1
            else:
                missing_count += 1
            current_visibility_band = (cur or {}).get("visibility_band")
            current_score_band = (cur or {}).get("score_band")
            if current_visibility_band == "near_visibility":
                near_visibility_now_count += 1
            if current_score_band == "near_validated":
                near_validated_now_count += 1
            prev_live = float(prev.get("live_score") or 0.0)
            prev_pre = float(prev.get("pre_policy_score") or 0.0)
            cur_live = float((cur or {}).get("live_score") or 0.0)
            cur_pre = float((cur or {}).get("pre_policy_score") or (cur or {}).get("prob_2_pre_regime") or 0.0)
            delta_live = round(cur_live - prev_live, 4) if cur else None
            delta_pre = round(cur_pre - prev_pre, 4) if cur else None
            if delta_live is not None and delta_live > 0.01:
                improved_live_count += 1
            if delta_pre is not None and delta_pre > 0.01:
                improved_pre_policy_count += 1
            change = {
                "symbol": symbol,
                "prior_pre_policy_score": prev.get("pre_policy_score"),
                "prior_live_score": prev.get("live_score"),
                "prior_live_threshold": prev.get("live_threshold"),
                "prior_visibility_band": prev.get("visibility_band"),
                "prior_score_band": prev.get("pre_policy_score_band"),
                "current_row_type": None if missing_current else current_row_type,
                "current_actionability_tier": (cur or {}).get("actionability_tier"),
                "current_pre_policy_score": (cur or {}).get("pre_policy_score"),
                "current_live_score": (cur or {}).get("live_score"),
                "current_live_threshold": (cur or {}).get("live_threshold"),
                "current_visibility_band": current_visibility_band,
                "current_score_band": current_score_band,
                "current_distance_to_live_threshold": (cur or {}).get("distance_to_live_threshold"),
                "delta_pre_policy_score": delta_pre,
                "delta_live_score": delta_live,
                "became_visible": became_visible,
                "still_blocked": still_blocked,
                "missing_current": missing_current,
            }
            changes.append(change)
            if became_visible and cur:
                tracked_visible_rows.append({
                    "symbol": symbol,
                    "tracked_rank": len(tracked_visible_rows) + 1,
                    "row_type": current_row_type,
                    "actionability_tier": (cur or {}).get("actionability_tier"),
                    "pre_policy_rank": (cur or {}).get("pre_policy_rank"),
                    "candidate_rank_all": (cur or {}).get("candidate_rank_all"),
                    "pre_policy_score": (cur or {}).get("pre_policy_score"),
                    "live_score": (cur or {}).get("live_score"),
                    "live_threshold": (cur or {}).get("live_threshold"),
                    "distance_to_validated": (cur or {}).get("distance_to_validated"),
                    "distance_to_live_threshold": (cur or {}).get("distance_to_live_threshold"),
                    "score_band": current_score_band,
                    "score_band_label": (cur or {}).get("score_band_label"),
                    "visibility_band": current_visibility_band,
                    "visibility_band_label": (cur or {}).get("visibility_band_label"),
                    "delta_live_score": delta_live,
                    "delta_pre_policy_score": delta_pre,
                })
        changes.sort(key=lambda row: (
            1 if row.get("became_visible") else 0,
            1 if row.get("current_score_band") == "near_validated" else 0,
            1 if row.get("current_visibility_band") == "near_visibility" else 0,
            float(row.get("current_live_score") or 0.0),
            float(row.get("delta_live_score") or -999.0),
        ), reverse=True)
        tracked_visible_rows.sort(key=lambda row: (
            1 if row.get("score_band") == "near_validated" else 0,
            float(row.get("live_score") or 0.0),
            float(row.get("delta_live_score") or -999.0),
        ), reverse=True)
        top_n = max(1, int(getattr(self.config, "cooldown_followup_comparison_top_n", 5) or 5))
        return {
            "available": True,
            "comparison_mode": "scheduled_followup" if explicit_followup else "post_cooldown_catchup",
            "comparison_reason": "triggered_followup_scan" if explicit_followup else "regular_scan_after_missed_or_elapsed_followup",
            "source_run_finished_utc": prior_context.get("source_run_finished_utc"),
            "source_market_regime_state": prior_context.get("market_regime_state"),
            "source_market_regime_actionability": prior_context.get("market_regime_actionability"),
            "tracked_count": len(list(prior_context.get("tracked_rows") or [])),
            "visible_now_count": visible_now_count,
            "still_blocked_count": still_blocked_count,
            "missing_count": missing_count,
            "near_visibility_now_count": near_visibility_now_count,
            "near_validated_now_count": near_validated_now_count,
            "improved_live_count": improved_live_count,
            "improved_pre_policy_count": improved_pre_policy_count,
            "tracked_visible_rows": tracked_visible_rows[:top_n],
            "tracked_visible_symbols": [r.get("symbol") for r in tracked_visible_rows[:top_n] if r.get("symbol")],
            "top_changes": changes[:top_n],
        }

    def _apply_followup_comparison_to_decision_summary(self, decision_summary: dict, followup_comparison: dict) -> dict:
        decision = dict(decision_summary or {})
        if not bool((followup_comparison or {}).get("available")):
            decision["followup_headline"] = None
            decision["followup_summary"] = None
            decision["followup_comparison"] = followup_comparison or {"available": False}
            return decision
        tracked_count = int(followup_comparison.get("tracked_count") or 0)
        visible_now = int(followup_comparison.get("visible_now_count") or 0)
        near_visibility_now = int(followup_comparison.get("near_visibility_now_count") or 0)
        top_changes = list(followup_comparison.get("top_changes") or [])
        tracked_visible_rows = list(followup_comparison.get("tracked_visible_rows") or [])
        top_visible_symbols = ", ".join(str(r.get("symbol")) for r in tracked_visible_rows[:3] if r.get("symbol"))
        top_symbols = ", ".join(str(r.get("symbol")) for r in top_changes[:3] if r.get("symbol"))
        if visible_now > 0:
            followup_headline = f"Cooldown follow-up: {visible_now}/{tracked_count} tracked name{'s' if visible_now != 1 else ''} now visible"
            followup_summary = f"The scheduled follow-up rechecked {tracked_count} previously blocked monitoring names. Visible now: {top_visible_symbols or top_symbols or 'see shortlist'}; {int(followup_comparison.get('still_blocked_count') or 0)} remain blocked."
        elif near_visibility_now > 0:
            followup_headline = f"Cooldown follow-up: {near_visibility_now} tracked name{'s' if near_visibility_now != 1 else ''} close to visibility"
            followup_summary = f"The scheduled follow-up rechecked {tracked_count} previously blocked monitoring names. None are visible yet, but {near_visibility_now} now sit near the live threshold ({top_symbols or 'see blocked monitoring rows'})."
        else:
            followup_headline = "Cooldown follow-up: tracked names remain blocked"
            followup_summary = f"The scheduled follow-up rechecked {tracked_count} previously blocked monitoring names. None became visible; strongest tracked symbols remain {top_symbols or 'in blocked monitoring'} while policy pressure persists."
        existing_summary = str(decision.get("summary") or "").strip()
        decision["summary"] = (followup_summary + (" " + existing_summary if existing_summary else "")).strip()
        decision["followup_headline"] = followup_headline
        decision["followup_summary"] = followup_summary
        decision["tracked_visible_count"] = int(followup_comparison.get("visible_now_count") or 0)
        decision["tracked_visible_symbols"] = list(followup_comparison.get("tracked_visible_symbols") or [])
        decision["followup_comparison"] = followup_comparison
        return decision


    def _coverage_snapshot(
        self,
        universe,
        *,
        requested: int,
        returned_light: int = 0,
        stage1_feature_ready: int = 0,
        stage2_requested: int = 0,
        stage2_returned: int = 0,
        stage2_feature_ready: int = 0,
        symbols_scored: int = 0,
        skip_reasons: Counter | None = None,
        blocked_stage1: int = 0,
        dropped_stage1_by_rank: int = 0,
        dropped_stage2_blocked: int = 0,
        suppressed_regime: int = 0,
        suppressed_threshold: int = 0,
        suppressed_cooldown: int = 0,
        dropped_by_output_cap: int = 0,
    ) -> dict:
        skip_reasons = skip_reasons or Counter()
        return {
            "universe_count": len(universe.eligible),
            "cohort_mode": universe.diagnostics.get("selection_mode", "dynamic"),
            "trained_cohort_requested_count": int(universe.diagnostics.get("trained_cohort_requested_count", 0) or 0),
            "trained_cohort_available_count": int(universe.diagnostics.get("trained_cohort_available_count", 0) or 0),
            "trained_cohort_missing_count": int(universe.diagnostics.get("trained_cohort_missing_count", 0) or 0),
            "symbols_requested_count": requested,
            "symbols_returned_with_bars_count": returned_light,
            "symbols_with_sufficient_bars_count": stage1_feature_ready,
            "symbols_scored_count": symbols_scored,
            "stage1_feature_ready_count": stage1_feature_ready,
            "stage2_fetch_requested_count": stage2_requested,
            "stage2_fetch_returned_count": stage2_returned,
            "stage2_feature_ready_count": stage2_feature_ready,
            "dropped_stage1_insufficient_history": int(skip_reasons.get("stage1_insufficient_history", 0)),
            "dropped_stage1_fetch_failed": int(skip_reasons.get("stage1_fetch_failed", 0)),
            "dropped_stage1_insufficient_observed": int(skip_reasons.get("stage1_insufficient_observed", 0)),
            "dropped_stage1_blocked": blocked_stage1,
            "dropped_stage1_by_rank": dropped_stage1_by_rank,
            "dropped_stage2_insufficient_history": int(skip_reasons.get("stage2_insufficient_history", 0)),
            "dropped_stage2_fetch_failed": int(skip_reasons.get("stage2_fetch_failed", 0)),
            "dropped_stage2_insufficient_observed": int(skip_reasons.get("stage2_insufficient_observed", 0)),
            "dropped_stage2_blocked": dropped_stage2_blocked,
            "dropped_stage2_regime_suppressed": suppressed_regime,
            "dropped_stage2_threshold_suppressed": suppressed_threshold,
            "dropped_stage2_cooldown_suppressed": suppressed_cooldown,
            "dropped_stage2_display_trimmed": dropped_by_output_cap,
            "dropped_stage2_output_cap": dropped_by_output_cap,
            "top_skip_reasons": [{"reason": k, "count": int(v)} for k, v in skip_reasons.most_common(10)],
            "followup_reserved_symbols": 0,
            "followup_reserved_existing_symbols": 0,
        }

    def _guardrail_snapshot(
        self,
        *,
        blocked_stage1: int = 0,
        blocked_stage2: int = 0,
        event_risk: int = 0,
        probability_capped: int = 0,
        suppressed_regime: int = 0,
        suppressed_threshold: int = 0,
        suppressed_cooldown: int = 0,
    ) -> dict:
        total_blocked = blocked_stage1 + blocked_stage2
        return {
            "blocked": total_blocked,
            "blocked_stage1": blocked_stage1,
            "blocked_stage2": blocked_stage2,
            "event_risk": event_risk,
            "probability_capped": probability_capped,
            "capped": probability_capped,
            "suppressed_regime": suppressed_regime,
            "suppressed_threshold": suppressed_threshold,
            "suppressed_cooldown": suppressed_cooldown,
        }

    def _score_stage_summary(self, rows: List[dict] | None) -> dict:
        rows = rows or []
        stage1_preview_rows = sum(1 for r in rows if str(r.get("candidate_stage") or "") == "stage1_preview")
        stage2_partial_rows = sum(1 for r in rows if str(r.get("candidate_stage") or "") == "stage2_partial")
        stage2_final_rows = sum(1 for r in rows if str(r.get("candidate_stage") or "") == "stage2_final")
        preview_rows = sum(1 for r in rows if bool(r.get("provisional")))
        deep_confirmed_rows = sum(1 for r in rows if bool(r.get("deep_confirmed")))
        action_ready_rows = sum(1 for r in rows if str(r.get("actionability_tier") or "") == "action_ready")
        selective_rows = sum(1 for r in rows if str(r.get("actionability_tier") or "") == "selective")
        watchlist_rows = sum(1 for r in rows if str(r.get("actionability_tier") or "") == "watchlist")
        informational_rows = sum(1 for r in rows if bool(r.get("informational_only")))
        informational_regime_rows = sum(1 for r in rows if str(r.get("suppression_reason") or "") == "regime")
        informational_cooldown_rows = sum(1 for r in rows if str(r.get("suppression_reason") or "") == "cooldown")
        informational_threshold_rows = sum(1 for r in rows if str(r.get("suppression_reason") or "") == "threshold")
        informational_display_trim_rows = sum(1 for r in rows if str(r.get("suppression_reason") or "") == "display_trim")
        return {
            "visible_rows": len(rows),
            "informational_rows": informational_rows,
            "informational_regime_rows": informational_regime_rows,
            "informational_cooldown_rows": informational_cooldown_rows,
            "informational_threshold_rows": informational_threshold_rows,
            "informational_display_trim_rows": informational_display_trim_rows,
            "preview_rows": preview_rows,
            "deep_confirmed_rows": deep_confirmed_rows,
            "stage1_preview_rows": stage1_preview_rows,
            "stage2_partial_rows": stage2_partial_rows,
            "stage2_final_rows": stage2_final_rows,
            "stage2_scored": stage2_partial_rows + stage2_final_rows,
            "action_ready_rows": action_ready_rows,
            "selective_rows": selective_rows,
            "watchlist_rows": watchlist_rows,
        }

    @staticmethod
    def _row_sort_key(row: dict) -> tuple:
        action_order = {"action_ready": 3, "selective": 2, "watchlist": 1}
        return (
            action_order.get(str(row.get("actionability_tier") or "watchlist"), 1),
            float(row.get("utility_decision_score", row.get("prob_2_rank", row.get("prob_2") or 0.0)) or 0.0),
            float(row.get("utility_confidence", row.get("opportunity_score", 0.0)) or 0.0),
            float(row.get("prob_2_rank", row.get("prob_2") or 0.0) or 0.0),
            -float(row.get("risk", 0.0) or 0.0),
        )

    @staticmethod
    def _unique_rows_by_symbol(rows: List[dict] | None) -> List[dict]:
        unique: List[dict] = []
        seen: set[str] = set()
        for row in list(rows or []):
            symbol = str(row.get("symbol") or "").strip()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            unique.append(row)
        return unique

    @staticmethod
    def _informational_sort_key(row: dict) -> tuple:
        decision = float(row.get("utility_decision_score", row.get("pre_policy_score", row.get("prob_2_pre_regime") or row.get("prob_2_model") or 0.0)) or 0.0)
        confidence = float(row.get("utility_confidence", 0.0) or 0.0)
        live_score = float(row.get("live_score", row.get("prob_2") or 0.0) or 0.0)
        risk = float(row.get("risk", 0.0) or 0.0)
        return (
            decision,
            confidence,
            live_score,
            -risk,
        )

    @staticmethod
    def _blocked_focus_sort_key(row: dict) -> tuple:
        band_order = {"validated": 3, "near_validated": 2, "exploratory": 1}
        visibility_order = {"cleared_visible_threshold": 3, "near_visibility": 2, "below_visibility": 1}
        liquidity_order = {"tier1": 3, "tier2": 2, "tier3": 1}
        pre_policy = float(row.get("pre_policy_score", row.get("prob_2_pre_regime") or row.get("prob_2_model") or 0.0) or 0.0)
        live_score = float(row.get("live_score", row.get("prob_2_rank") or row.get("prob_2") or 0.0) or 0.0)
        opp = float(row.get("opportunity_score", 0.0) or 0.0)
        risk = float(row.get("risk", 0.0) or 0.0)
        threshold_gap = float(row.get("distance_to_live_threshold", 0.0) or 0.0)
        return (
            band_order.get(str(row.get("pre_policy_score_band") or row.get("score_band") or "exploratory"), 1),
            visibility_order.get(str(row.get("visibility_band") or "below_visibility"), 1),
            -threshold_gap,
            pre_policy,
            liquidity_order.get(str(row.get("liquidity_tier") or "tier3"), 1),
            live_score,
            opp,
            -risk,
        )

    def _policy_math(self, *, factor: float, cap: float, threshold: float) -> dict:
        factor = float(factor or 0.0)
        cap = float(cap or 0.0)
        threshold = float(threshold or 0.0)
        max_reachable = min(cap, factor if factor > 0 else 0.0)
        required_pre_policy = (threshold / factor) if factor > 0 else None
        feasible_by_factor = bool(required_pre_policy is not None and required_pre_policy <= 1.0 + 1e-12)
        feasible_by_cap = threshold <= cap + 1e-12
        return {
            "max_reachable_post_policy": round(max_reachable, 4),
            "required_pre_policy": round(required_pre_policy, 4) if required_pre_policy is not None else None,
            "feasible_by_factor": feasible_by_factor,
            "feasible_by_cap": feasible_by_cap,
            "feasible": bool(feasible_by_factor and feasible_by_cap),
        }

    def _amber_min_threshold(self, liquidity_tier: str) -> float:
        return float(getattr(self.config, f"market_regime_amber_{str(liquidity_tier).lower()}_min_threshold", 0.60))

    def _amber_top_n(self, liquidity_tier: str) -> int:
        return max(1, int(getattr(self.config, f"market_regime_amber_{str(liquidity_tier).lower()}_top_n", 1) or 1))

    def _build_threshold_plan(self, *, regime_state: str, threshold_candidates: List[dict]) -> dict:
        state = str(regime_state or "green").lower()
        tiers = ("tier1", "tier2", "tier3")
        plan = {"mode": "absolute", "tiers": {}}
        relative_enabled = bool(getattr(self.config, "market_regime_amber_relative_threshold_enabled", True))
        for tier in tiers:
            policy = live_policy_for(state, tier, self.config)
            base_threshold = float(policy["threshold"])
            base_math = self._policy_math(factor=policy["factor"], cap=policy["cap"], threshold=base_threshold)
            tier_rows = [row for row in threshold_candidates if str(row.get("liquidity_tier") or "") == tier]
            scores = sorted((float(row.get("prob_2_rank", 0.0) or 0.0) for row in tier_rows), reverse=True)
            effective_threshold = base_threshold
            nth_score = None
            amber_floor = None
            top_n = None
            threshold_mode = "absolute"
            if state == "amber" and relative_enabled:
                amber_floor = float(self._amber_min_threshold(tier))
                top_n = int(self._amber_top_n(tier))
                if scores:
                    nth_idx = min(len(scores), top_n) - 1
                    nth_score = float(scores[nth_idx])
                effective_threshold = min(
                    base_threshold,
                    max(amber_floor, float(nth_score) if nth_score is not None else amber_floor),
                    float(base_math["max_reachable_post_policy"]),
                )
                threshold_mode = "amber_relative_top_n"
                plan["mode"] = threshold_mode
            effective_threshold = max(0.0, float(effective_threshold))
            effective_math = self._policy_math(factor=policy["factor"], cap=policy["cap"], threshold=effective_threshold)
            plan["tiers"][tier] = {
                "base_threshold": round(base_threshold, 4),
                "effective_threshold": round(effective_threshold, 4),
                "factor": round(float(policy["factor"]), 4),
                "cap": round(float(policy["cap"]), 4),
                "observed_candidates": len(scores),
                "top_score": round(float(scores[0]), 4) if scores else None,
                "nth_score": round(float(nth_score), 4) if nth_score is not None else None,
                "relative_floor": round(float(amber_floor), 4) if amber_floor is not None else None,
                "top_n": top_n,
                "mode": threshold_mode,
                "base_math": base_math,
                "effective_math": effective_math,
                "relaxed": bool(effective_threshold + 1e-12 < base_threshold),
            }
        return plan

    def _effective_threshold_for_row(self, row: dict, threshold_plan: dict) -> tuple[float, dict]:
        tier = str(row.get("liquidity_tier") or "tier3")
        tier_plan = dict(((threshold_plan or {}).get("tiers") or {}).get(tier) or {})
        effective_threshold = float(tier_plan.get("effective_threshold", row.get("live_threshold") or 0.0) or 0.0)
        return effective_threshold, tier_plan


    def _objective_semantics_contract(self) -> dict:
        now = time.time()
        cached = self._objective_semantics_cache if isinstance(self._objective_semantics_cache, dict) else None
        if cached is not None and (now - float(self._objective_semantics_cache_ts or 0.0)) <= 15.0:
            return dict(cached)
        contract = load_objective_semantics_contract(
            self.config.model_dir,
            live_threshold=effective_live_raw_threshold(self.config),
            stage1_selection_mode=str(getattr(self.config, "stage1_selection_mode", "") or ""),
        )
        self._objective_semantics_cache = dict(contract or {})
        self._objective_semantics_cache_ts = now
        return dict(contract or {})

    def _validated_floor(self, score_contract: dict) -> float:
        validated = [float(x) for x in (score_contract.get("validated_thresholds") or []) if x is not None]
        if validated:
            return round(min(validated), 4)
        fallback = score_contract.get("temporal_support_threshold")
        try:
            return round(float(fallback if fallback is not None else 0.60), 4)
        except Exception:
            return 0.60

    def _score_band(self, *, live_score: float, score_contract: dict) -> dict:
        validated_floor = self._validated_floor(score_contract)
        near_floor = min(validated_floor, float(getattr(self.config, "stage2_near_validated_floor", 0.45) or 0.45))
        score = float(live_score or 0.0)
        gap = max(0.0, validated_floor - score)
        if score >= validated_floor:
            band = "validated"
            label = "Validated band"
            priority = "validated_action_band"
        elif score >= near_floor:
            band = "near_validated"
            label = "Near validated"
            priority = "watch_closely"
        else:
            band = "exploratory"
            label = "Exploratory only"
            priority = "exploratory_only"
        objective_contract = self._objective_semantics_contract()
        objective_band = score_objective_band(
            live_score=score,
            contract=objective_contract,
            near_gap=float(getattr(self.config, "stage2_blocked_near_threshold_gap", 0.08) or 0.08),
        )
        payload = {
            "validated_floor": round(validated_floor, 4),
            "near_validated_floor": round(near_floor, 4),
            "distance_to_validated": round(gap, 4),
            "distance_to_validated_pct_points": round(gap * 100.0, 2),
            "score_band": band,
            "score_band_label": label,
            "monitor_priority": priority,
        }
        payload.update(dict(objective_band or {}))
        return payload

    def _visibility_band(self, *, live_score: float, live_threshold: float) -> dict:
        threshold = max(0.0, float(live_threshold or 0.0))
        score = float(live_score or 0.0)
        gap = max(0.0, threshold - score)
        near_gap = max(0.0, float(getattr(self.config, "stage2_blocked_near_threshold_gap", 0.08) or 0.08))
        if score >= threshold - 1e-12:
            band = "cleared_visible_threshold"
            label = "Cleared current threshold"
        elif gap <= near_gap + 1e-12:
            band = "near_visibility"
            label = "Near current threshold"
        else:
            band = "below_visibility"
            label = "Below current threshold"
        return {
            "distance_to_live_threshold": round(gap, 4),
            "distance_to_live_threshold_pct_points": round(gap * 100.0, 2),
            "visibility_band": band,
            "visibility_band_label": label,
            "near_visibility_gap": round(near_gap, 4),
        }


    def _score_diagnostics(self, *, visible_rows: List[dict], suppressed_rows: List[dict], informational_rows: List[dict], informational_overflow_rows: List[dict], score_contract: dict) -> dict:
        all_rows = list(visible_rows) + list(suppressed_rows) + list(informational_rows) + list(informational_overflow_rows)
        unique_rows = self._unique_rows_by_symbol(all_rows)

        def _values(rows: List[dict], key: str) -> list[float]:
            vals: list[float] = []
            for row in rows:
                value = row.get(key)
                if value in (None, ""):
                    continue
                try:
                    vals.append(float(value))
                except Exception:
                    continue
            return vals

        def _quantiles(vals: list[float]) -> dict:
            if not vals:
                return {"count": 0, "max": None, "p95": None, "median": None, "min": None}
            arr = np.asarray(vals, dtype=float)
            return {
                "count": int(arr.size),
                "min": round(float(np.quantile(arr, 0.0)), 4),
                "median": round(float(np.quantile(arr, 0.5)), 4),
                "p95": round(float(np.quantile(arr, 0.95)), 4),
                "max": round(float(np.quantile(arr, 1.0)), 4),
            }

        def _tier_summary(rows: List[dict]) -> dict:
            model_vals = _values(rows, "prob_2_model")
            pre_vals = _values(rows, "pre_policy_score")
            live_vals = _values(rows, "live_score")
            thresholds = [0.30, 0.35, 0.40, 0.45]
            return {
                "rows": len(rows),
                "model_score": _quantiles(model_vals),
                "pre_policy_score": _quantiles(pre_vals),
                "live_score": _quantiles(live_vals),
                "counts_above": {f"{t:.2f}": int(sum(1 for v in live_vals if v >= t)) for t in thresholds},
            }

        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        live_vals = _values(unique_rows, "live_score")
        pre_vals = _values(unique_rows, "pre_policy_score")
        model_vals = _values(unique_rows, "prob_2_model")
        validated_floor = self._validated_floor(score_contract)
        band_counts = []
        for threshold in thresholds:
            band_counts.append({
                "threshold": round(float(threshold), 2),
                "live_count": int(sum(1 for x in live_vals if x >= threshold)),
                "pre_policy_count": int(sum(1 for x in pre_vals if x >= threshold)),
                "model_count": int(sum(1 for x in model_vals if x >= threshold)),
            })
        capped_rows = [r for r in unique_rows if bool(r.get("was_capped"))]
        by_tier = {
            tier: _tier_summary([r for r in unique_rows if str(r.get("liquidity_tier") or "") == tier])
            for tier in ("tier1", "tier2", "tier3")
        }
        top_pretrim = []
        for row in sorted(
            unique_rows,
            key=lambda r: (float(r.get("pre_policy_score") or 0.0), float(r.get("live_score") or 0.0)),
            reverse=True,
        )[:15]:
            top_pretrim.append({
                "symbol": row.get("symbol"),
                "liquidity_tier": row.get("liquidity_tier"),
                "candidate_stage": row.get("candidate_stage"),
                "row_type": row.get("row_type"),
                "display_bucket": row.get("display_bucket"),
                "actionability_tier": row.get("actionability_tier"),
                "suppression_reason": row.get("suppression_reason"),
                "prob_2_model": row.get("prob_2_model"),
                "pre_policy_score": row.get("pre_policy_score"),
                "live_score": row.get("live_score"),
                "distance_to_validated": row.get("distance_to_validated"),
                "distance_to_live_threshold": row.get("distance_to_live_threshold"),
            })
        diagnostics = {
            "available": bool(unique_rows),
            "row_count": len(unique_rows),
            "row_count_total": len(all_rows),
            "duplicate_row_instances_removed": max(0, len(all_rows) - len(unique_rows)),
            "visible_count": len(visible_rows),
            "suppressed_count": len(suppressed_rows),
            "informational_count": len(informational_rows),
            "overflow_count": len(informational_overflow_rows),
            "validated_floor": round(float(validated_floor), 4),
            "guardrail_cap": round(float(self.config.tail_unvalidated_cap), 4),
            "guardrail_cap_below_validated_floor": bool(float(self.config.tail_unvalidated_cap) < float(validated_floor)),
            "live_score": _quantiles(live_vals),
            "pre_policy_score": _quantiles(pre_vals),
            "model_score": _quantiles(model_vals),
            "counts_above_thresholds": band_counts,
            "by_liquidity_tier": by_tier,
            "top_pretrim_candidates": top_pretrim,
            "scoring_funnel": {
                "stage2_visible": len(visible_rows),
                "stage2_suppressed": len(suppressed_rows),
                "stage2_informational_retained": len(informational_rows),
                "stage2_informational_overflow": len(informational_overflow_rows),
                "stage2_total_ranked": len(unique_rows),
            },
            "capped_rows": {
                "count": len(capped_rows),
                "share": round(float(len(capped_rows) / len(unique_rows)), 4) if unique_rows else 0.0,
                "max_live_score": round(max((float(r.get("live_score") or 0.0) for r in capped_rows), default=0.0), 4) if capped_rows else None,
                "max_pre_policy_score": round(max((float(r.get("pre_policy_score") or 0.0) for r in capped_rows), default=0.0), 4) if capped_rows else None,
            },
            "penalty_rows": {
                "panic_penalty_rows": int(sum(1 for r in all_rows if float(r.get("panic_penalty") or 0.0) > 0.0)),
                "sector_penalty_rows": int(sum(1 for r in all_rows if float(r.get("sector_penalty") or 0.0) > 0.0)),
                "binance_gap_penalty_rows": int(sum(1 for r in all_rows if float(r.get("binance_gap_penalty") or 0.0) > 0.0)),
                "binance_lead_penalty_rows": int(sum(1 for r in all_rows if float(r.get("binance_lead_penalty") or 0.0) > 0.0)),
            },
        }
        if diagnostics["available"]:
            max_live = diagnostics["live_score"]["max"]
            if max_live is not None and max_live < 0.45:
                headline = "Score-range starvation: current scan never reached 0.45 live score"
            elif max_live is not None and max_live < validated_floor:
                headline = f"Current scan produced no rows at or above validated floor {validated_floor:.2f}"
            else:
                headline = "Current scan reached the validated band or above"
        else:
            headline = "No scored rows available for current scan diagnostics"
        diagnostics["headline"] = headline
        return diagnostics

    def _candidate_quality_diagnostics(
        self,
        *,
        stage1_input_rows: Dict[str, dict],
        stage1_guardrails: Dict[str, dict],
        stage1_diags: Dict[str, dict],
        stage1_candidates: List[str],
        stage1_selection_meta: Dict[str, dict] | None = None,
        stage2_diags: Dict[str, dict],
        final_rows: List[dict],
    ) -> dict:
        selected_rank = {str(symbol): idx for idx, symbol in enumerate(list(stage1_candidates or []), start=1)}
        selection_sources = dict((stage1_selection_meta or {}).get("selected_sources") or {})
        final_rows = self._unique_rows_by_symbol(final_rows)
        final_by_symbol: Dict[str, dict] = {}
        for row in list(final_rows or []):
            symbol = str(row.get("symbol") or "")
            if symbol and symbol not in final_by_symbol:
                final_by_symbol[symbol] = row
        stage2_ready = set(str(s) for s in (stage2_diags or {}).keys())
        stage1_by_tier = {tier: {"feature_ready": 0, "blocked": 0, "selected": 0, "selected_not_scored": 0} for tier in ("tier1", "tier2", "tier3")}
        selectable_total = 0
        for symbol, row in (stage1_input_rows or {}).items():
            diag = (stage1_diags or {}).get(symbol) or {}
            guard = (stage1_guardrails or {}).get(symbol) or {}
            tier = str(classify_liquidity_tier(symbol, diag, self.config) or "tier3")
            bucket = stage1_by_tier.setdefault(tier, {"feature_ready": 0, "blocked": 0, "selected": 0, "selected_not_scored": 0})
            bucket["feature_ready"] += 1
            blocked = str(guard.get("block_code") or "") == "BLOCKED"
            if blocked:
                bucket["blocked"] += 1
            else:
                selectable_total += 1
            if symbol in selected_rank:
                bucket["selected"] += 1
                if symbol not in final_by_symbol:
                    bucket["selected_not_scored"] += 1
        for info in stage1_by_tier.values():
            denom = max(1, int(info.get("feature_ready") or 0) - int(info.get("blocked") or 0))
            info["selected_share"] = round(float(info.get("selected") or 0) / float(denom), 4)

        def _score_stats(rows: List[dict], key: str) -> dict:
            vals = []
            for row in rows:
                value = row.get(key)
                if value in (None, ""):
                    continue
                try:
                    vals.append(float(value))
                except Exception:
                    continue
            if not vals:
                return {"count": 0, "median": None, "max": None}
            arr = np.asarray(vals, dtype=float)
            return {"count": int(arr.size), "median": round(float(np.quantile(arr, 0.5)), 4), "max": round(float(np.quantile(arr, 1.0)), 4)}

        stage2_by_tier = {}
        for tier in ("tier1", "tier2", "tier3"):
            tier_rows = [r for r in list(final_rows or []) if str(r.get("liquidity_tier") or "") == tier]
            visible_rows = [r for r in tier_rows if str(r.get("row_type") or "") == "visible"]
            hidden_rows = [r for r in tier_rows if str(r.get("row_type") or "") != "visible"]
            live_vals = [float(r.get("live_score") or 0.0) for r in tier_rows if r.get("live_score") not in (None, "")]
            stage2_by_tier[tier] = {
                "scored": len(tier_rows),
                "visible": len(visible_rows),
                "hidden": len(hidden_rows),
                "model_score": _score_stats(tier_rows, "prob_2_model"),
                "pre_policy_score": _score_stats(tier_rows, "pre_policy_score"),
                "live_score": _score_stats(tier_rows, "live_score"),
                "count_ge_0_30": int(sum(1 for v in live_vals if v >= 0.30)),
                "count_ge_0_35": int(sum(1 for v in live_vals if v >= 0.35)),
                "count_ge_0_45": int(sum(1 for v in live_vals if v >= 0.45)),
            }

        trace_rows = []
        for symbol, row in sorted((stage1_input_rows or {}).items(), key=lambda item: (0 if item[0] in selected_rank else 1, int(selected_rank.get(item[0], 999999)), item[0])):
            diag = (stage1_diags or {}).get(symbol) or {}
            guard = (stage1_guardrails or {}).get(symbol) or {}
            final_row = final_by_symbol.get(symbol) or {}
            tier = str(classify_liquidity_tier(symbol, diag, self.config) or "tier3")
            trace_rows.append({
                "symbol": symbol,
                "liquidity_tier": tier,
                "stage1_selected": bool(symbol in selected_rank),
                "stage1_rank": selected_rank.get(symbol),
                "stage1_selection_source": selection_sources.get(symbol),
                "stage1_blocked": str(guard.get("block_code") or "") == "BLOCKED",
                "stage1_block_code": guard.get("block_code"),
                "stage2_fetched": bool(symbol in stage2_ready),
                "stage2_scored": bool(symbol in final_by_symbol),
                "final_row_type": final_row.get("row_type"),
                "final_display_bucket": final_row.get("display_bucket"),
                "final_suppression_reason": final_row.get("suppression_reason"),
                "final_actionability_tier": final_row.get("actionability_tier"),
                "candidate_rank_all": final_row.get("candidate_rank_all"),
                "pre_policy_rank": final_row.get("pre_policy_rank"),
                "prob_2_model": final_row.get("prob_2_model"),
                "pre_policy_score": final_row.get("pre_policy_score"),
                "live_score": final_row.get("live_score"),
            })

        stage2_max_live = max((float(r.get("live_score") or 0.0) for r in list(final_rows or []) if r.get("live_score") not in (None, "")), default=0.0)
        selected_share_all = round(float(len(stage1_candidates or [])) / float(max(1, selectable_total)), 4)
        pass_through_warning = bool(selected_share_all >= 0.9)
        top_visible_stage1_ranks = [selected_rank.get(str(r.get("symbol") or "")) for r in list(final_rows or []) if str(r.get("row_type") or "") == "visible"]
        top_visible_stage1_ranks = [int(r) for r in top_visible_stage1_ranks if r is not None][:10]
        if pass_through_warning:
            headline = "Stage1 shortlist is effectively pass-through: nearly every non-blocked candidate is reaching stage2"
        elif stage2_max_live < 0.35:
            headline = "Candidate-quality starvation: selected stage2 names stayed below 0.35 even before visibility trimming"
        else:
            headline = "Some selected stage2 names are approaching the lower research band"

        stage2_scored_unique = len(final_by_symbol)

        return {
            "available": bool(stage1_input_rows),
            "headline": headline,
            "stage1_feature_ready": len(stage1_input_rows or {}),
            "stage1_selectable": selectable_total,
            "stage1_selected": len(stage1_candidates or []),
            "stage1_selected_share": selected_share_all,
            "stage1_pass_through_warning": pass_through_warning,
            "stage1_selection_mode": (stage1_selection_meta or {}).get("selection_mode") or "primary_only",
            "stage1_primary_slots": int((stage1_selection_meta or {}).get("primary_slots") or 0),
            "stage1_recall_reserve_slots": int((stage1_selection_meta or {}).get("recall_reserve_slots") or 0),
            "stage1_selected_primary_count": int((stage1_selection_meta or {}).get("selected_primary_count") or 0),
            "stage1_selected_recall_reserve_count": int((stage1_selection_meta or {}).get("selected_recall_reserve_count") or 0),
            "stage1_selected_recall_promotion_count": int((stage1_selection_meta or {}).get("selected_recall_promotion_count") or 0),
            "stage1_selected_opportunity_model_count": int((stage1_selection_meta or {}).get("selected_opportunity_model_count") or 0),
            "stage1_selected_opportunity_reserve_count": int((stage1_selection_meta or {}).get("selected_opportunity_reserve_count") or 0),
            "stage1_selected_primary_backfill_count": int((stage1_selection_meta or {}).get("selected_primary_backfill_count") or 0),
            "configured_stage1_max_candidates": int(self.config.stage1_max_candidates),
            "stage2_scored": stage2_scored_unique,
            "selected_not_scored": max(0, len(stage1_candidates or []) - stage2_scored_unique),
            "top_visible_stage1_ranks": top_visible_stage1_ranks,
            "stage1_by_tier": stage1_by_tier,
            "stage2_by_tier": stage2_by_tier,
            "stage1_to_stage2_trace": trace_rows,
        }

    def _stage1_omission_stats(self, rows: List[dict] | None) -> dict:
        rows = list(rows or [])
        live_vals = []
        for row in rows:
            value = row.get("live_score")
            if value in (None, ""):
                continue
            try:
                live_vals.append(float(value))
            except Exception:
                continue
        top_rows = sorted(rows, key=lambda r: float(r.get("live_score") or 0.0), reverse=True)[:5]
        return {
            "rows": len(rows),
            "max_live_score": round(max(live_vals), 4) if live_vals else None,
            "count_ge_0_35": int(sum(1 for v in live_vals if v >= 0.35)),
            "count_ge_0_45": int(sum(1 for v in live_vals if v >= 0.45)),
            "count_ge_0_50": int(sum(1 for v in live_vals if v >= 0.50)),
            "count_ge_0_60": int(sum(1 for v in live_vals if v >= 0.60)),
            "top_symbols": [
                {
                    "symbol": r.get("symbol"),
                    "live_score": r.get("live_score"),
                    "pre_policy_score": r.get("pre_policy_score"),
                    "live_threshold": r.get("live_threshold"),
                    "stage1_primary_rank": r.get("stage1_primary_rank"),
                    "stage1_selection_source": r.get("stage1_selection_source"),
                }
                for r in top_rows
            ],
        }

    def _build_threshold_experiment_review(
        self,
        *,
        final_rows: List[dict],
        current_threshold: float,
        experiment_threshold: float = 0.28,
    ) -> dict:
        rows = [dict(r) for r in list(final_rows or [])]
        if not rows:
            return {
                "available": False,
                "headline": "No threshold experiment evidence this scan",
                "summary": "Run a complete scan before comparing the current live threshold with a 0.28 experiment.",
                "current_threshold": round(float(current_threshold), 4),
                "experiment_threshold": round(float(experiment_threshold), 4),
            }
        ranked = sorted(rows, key=lambda r: (float(r.get("live_score") or 0.0), str(r.get("symbol") or "")), reverse=True)
        baseline_visible = [r for r in ranked if float(r.get("live_score") or 0.0) >= float(current_threshold)]
        experiment_visible = [r for r in ranked if float(r.get("live_score") or 0.0) >= float(experiment_threshold)]
        added_rows = [r for r in experiment_visible if float(r.get("live_score") or 0.0) < float(current_threshold)]
        if len(experiment_visible) > len(baseline_visible):
            verdict = "controlled_threshold_experiment_supported"
            headline = "A 0.28 threshold experiment would widen the shortlist"
            summary = (
                f"Lowering the live threshold from {current_threshold:.2f} to {experiment_threshold:.2f} would increase visible rows "
                f"from {len(baseline_visible)} to {len(experiment_visible)} in this scan, adding {len(added_rows)} exploratory names without creating any validated candidates."
            )
        else:
            verdict = "threshold_experiment_low_incremental_gain"
            headline = "A 0.28 threshold experiment adds little in this scan"
            summary = (
                f"Lowering the live threshold from {current_threshold:.2f} to {experiment_threshold:.2f} would not materially change the visible shortlist in this scan."
            )
        added_band_counts = {
            "count_ge_0_28": int(sum(1 for r in added_rows if float(r.get("live_score") or 0.0) >= 0.28)),
            "count_ge_0_30": int(sum(1 for r in added_rows if float(r.get("live_score") or 0.0) >= 0.30)),
            "count_ge_0_35": int(sum(1 for r in added_rows if float(r.get("live_score") or 0.0) >= 0.35)),
            "count_ge_0_45": int(sum(1 for r in added_rows if float(r.get("live_score") or 0.0) >= 0.45)),
            "count_ge_0_50": int(sum(1 for r in added_rows if float(r.get("live_score") or 0.0) >= 0.50)),
        }
        return {
            "available": True,
            "headline": headline,
            "summary": summary,
            "verdict": verdict,
            "current_threshold": round(float(current_threshold), 4),
            "experiment_threshold": round(float(experiment_threshold), 4),
            "current_visible_count": int(len(baseline_visible)),
            "experiment_visible_count": int(len(experiment_visible)),
            "additional_visible_count": int(len(added_rows)),
            "current_top_symbols": [
                {
                    "symbol": r.get("symbol"),
                    "live_score": r.get("live_score"),
                    "pre_policy_score": r.get("pre_policy_score"),
                    "live_threshold": current_threshold,
                }
                for r in baseline_visible[:5]
            ],
            "added_symbols": [
                {
                    "symbol": r.get("symbol"),
                    "live_score": r.get("live_score"),
                    "pre_policy_score": r.get("pre_policy_score"),
                    "distance_to_experiment_threshold": round(max(0.0, float(r.get("live_score") or 0.0) - float(experiment_threshold)), 4),
                    "distance_to_current_threshold": round(max(0.0, float(current_threshold) - float(r.get("live_score") or 0.0)), 4),
                    "suppression_reason": r.get("suppression_reason"),
                    "actionability_tier": r.get("actionability_tier"),
                }
                for r in added_rows[:10]
            ],
            "added_band_counts": added_band_counts,
            "current_count_ge_0_45": int(sum(1 for r in baseline_visible if float(r.get("live_score") or 0.0) >= 0.45)),
            "experiment_count_ge_0_45": int(sum(1 for r in experiment_visible if float(r.get("live_score") or 0.0) >= 0.45)),
            "current_count_ge_0_50": int(sum(1 for r in baseline_visible if float(r.get("live_score") or 0.0) >= 0.50)),
            "experiment_count_ge_0_50": int(sum(1 for r in experiment_visible if float(r.get("live_score") or 0.0) >= 0.50)),
            "max_live_score": round(max(float(r.get("live_score") or 0.0) for r in ranked), 4),
        }

    def _build_stage1_omission_audit_summary(
        self,
        *,
        selected_rows: List[dict],
        omitted_rows: List[dict],
        omitted_total: int,
        omitted_audited: int,
        omitted_truncated: bool,
        stage1_selection_meta: Dict[str, dict] | None,
    ) -> dict:
        selected_stats = self._stage1_omission_stats(selected_rows)
        omitted_stats = self._stage1_omission_stats(omitted_rows)
        selected_max = float(selected_stats.get("max_live_score") or 0.0)
        omitted_max = float(omitted_stats.get("max_live_score") or 0.0)
        omitted_additional_ge_045 = max(0, int(omitted_stats.get("count_ge_0_45") or 0) - int(selected_stats.get("count_ge_0_45") or 0))
        omitted_additional_ge_050 = max(0, int(omitted_stats.get("count_ge_0_50") or 0) - int(selected_stats.get("count_ge_0_50") or 0))
        if omitted_audited <= 0:
            verdict = "no_nonblocked_omitted_names_to_audit"
            headline = "No Stage 1 omission audit candidates"
            summary = "All non-blocked Stage 1 names were selected for Stage 2, so this scan cannot diagnose Stage 1 omission."
        elif omitted_stats.get("rows", 0) <= 0:
            verdict = "insufficient_omitted_stage2_rows"
            headline = "Insufficient omitted-name evidence this scan"
            summary = "Non-selected Stage 1 names were shadow-fetched, but none produced usable Stage 2 rows. This scan does not yet distinguish Stage 1 omission from Stage 2 compression."
        elif omitted_additional_ge_050 > 0 or omitted_additional_ge_045 > 0 or omitted_max > (selected_max + 0.05):
            verdict = "stage1_omission_likely"
            headline = "Stage 1 omission likely contributed to the weak upper tail"
            summary = "Shadow-scored omitted Stage 1 names produced a meaningfully stronger Stage 2 upper tail than the selected set. Investigate Stage 1 selection before changing the Stage 2 threshold."
        elif selected_max < 0.45 and omitted_max < 0.45:
            verdict = "stage2_score_compression_likely"
            headline = "Stage 2 score compression looks like the primary bottleneck"
            summary = "Both selected and omitted Stage 1 names stayed below the near-validated band once Stage 2 scored them. The bottleneck looks more like Stage 2 upper-tail weakness than Stage 1 omission."
        else:
            verdict = "mixed_or_inconclusive"
            headline = "Mixed omission audit — both bottlenecks remain plausible"
            summary = "Shadow-scored omitted names did not clearly overturn the selected set, but they also did not rule Stage 1 omission out. Keep accumulating evidence before changing live logic."
        meta = stage1_selection_meta or {}
        return {
            "available": True,
            "headline": headline,
            "summary": summary,
            "verdict": verdict,
            "stage1_selection_mode": meta.get("selection_mode") or "primary_only",
            "stage1_primary_slots": int(meta.get("primary_slots") or 0),
            "stage1_effective_max_candidates": int(meta.get("effective_max_candidates") or 0),
            "omitted_nonblocked_total": int(omitted_total or 0),
            "omitted_nonblocked_audited": int(omitted_audited or 0),
            "omitted_audit_truncated": bool(omitted_truncated),
            "selected_stage2": selected_stats,
            "omitted_stage2": omitted_stats,
            "omitted_beats_selected_top_score": bool(omitted_max > selected_max),
            "omitted_additional_ge_0_45": omitted_additional_ge_045,
            "omitted_additional_ge_0_50": omitted_additional_ge_050,
        }

    def _score_omitted_stage2_row(
        self,
        *,
        symbol: str,
        row: dict,
        diag: dict,
        guard: dict,
        market_regime,
        btc_regime: str,
        bundle: ModelBundle | None,
        score_contract: dict,
        sector_penalty: float,
    ) -> dict | None:
        if str(guard.get("block_code") or "") == "BLOCKED":
            return None
        live_pipeline_mode = str(getattr(self.config, "live_pipeline_mode", "raw_threshold") or "raw_threshold").strip().lower()
        if live_pipeline_mode not in {"full", "raw_threshold"}:
            live_pipeline_mode = "raw_threshold"
        live_raw_threshold = effective_live_raw_threshold(self.config)
        is_panic = btc_regime == "BTC panic"
        threshold_boost = self.config.panic_threshold_boost if is_panic else 0.0
        liquidity_bucket = self._liquidity_bucket(diag)
        liquidity_tier = classify_liquidity_tier(symbol, diag, self.config)
        if bundle is not None:
            prob_model = float(bundle.predict_proba(pd.DataFrame([{k: row[k] for k in FEATURE_COLUMNS}]))[0])
        else:
            prob_model = heuristic_probability(row, guard, guardrail_cap=self.config.tail_unvalidated_cap)
        if live_pipeline_mode == "raw_threshold":
            prob_pre_regime = max(0.0, min(1.0, float(prob_model)))
            prob_adjusted = prob_pre_regime
            effective_threshold = float(live_raw_threshold)
            suppressed_by = None
        else:
            prob_adjusted, _ = apply_live_post_model_adjustments(
                prob_model,
                row,
                guard,
                is_panic=is_panic,
                threshold_boost=threshold_boost,
                sector_penalty=sector_penalty,
                guardrail_cap=self.config.tail_unvalidated_cap,
            )
            prob_pre_regime = float(prob_adjusted)
            live_policy = live_policy_for(market_regime.state, liquidity_tier, self.config)
            if live_policy.get("suppress"):
                suppressed_by = "regime"
                effective_threshold = float(live_policy.get("threshold") or 0.0)
            else:
                cooldown_blocked = False
                if market_regime.suppress_new_entries:
                    if liquidity_tier == "tier3":
                        cooldown_blocked = True
                    elif liquidity_tier == "tier2" and liquidity_bucket != "high":
                        cooldown_blocked = True
                if cooldown_blocked:
                    suppressed_by = "cooldown"
                    effective_threshold = float(live_policy.get("threshold") or 0.0)
                else:
                    suppressed_by = None
                    prob_adjusted = max(0.0, prob_adjusted * float(live_policy["factor"]))
                    prob_adjusted = min(prob_adjusted, float(live_policy["cap"]))
                    effective_threshold = float(live_policy.get("threshold") or 0.0)
        trust = self._apply_tail_trust(prob_adjusted, score_contract)
        score_band = self._score_band(live_score=trust["display_score"], score_contract=score_contract)
        threshold_cleared = suppressed_by is None and float(prob_adjusted) >= float(effective_threshold)
        return {
            "symbol": symbol,
            "prob_2_model": round(float(prob_model), 4),
            "pre_policy_score": round(float(prob_pre_regime), 4),
            "live_score": trust["display_score"],
            "live_threshold": round(float(effective_threshold), 4),
            "threshold_cleared": bool(threshold_cleared),
            "suppressed_by": suppressed_by or (None if threshold_cleared else "threshold"),
            "liquidity_tier": liquidity_tier,
            "score_band": score_band["score_band"],
            "distance_to_live_threshold": round(float(max(0.0, float(effective_threshold) - float(trust["display_score"]))), 4),
        }

    def _shadow_score_stage2_symbols(
        self,
        *,
        audit_symbols: List[str],
        stage2_seed_products: Dict[str, dict],
        btc_ctx: dict,
        eth_ctx: dict,
        btc_df: pd.DataFrame | None,
        market_regime,
        btc_regime: str,
        bundle: ModelBundle | None,
        score_contract: dict,
        sector_leader_rets: Dict[str, float],
        primary_ranks: Dict[str, int] | None = None,
        selection_sources: Dict[str, str] | None = None,
    ) -> List[dict]:
        rows: List[dict] = []
        primary_ranks = dict(primary_ranks or {})
        selection_sources = dict(selection_sources or {})
        symbols = [str(symbol) for symbol in list(audit_symbols or []) if str(symbol)]
        if not symbols:
            return rows
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(self._fetch_symbol_frame, symbol, self.config.stage2_lookback_5m_bars): symbol for symbol in symbols}
            for fut in as_completed(futures):
                symbol = futures[fut]
                try:
                    df = fut.result(timeout=self.config.http_timeout_seconds + 5)
                    history_bars = len(df)
                    observed_bars = int(df.attrs.get("observed_bars", int((df["volume"] > 0).sum()) if not df.empty else 0))
                    if history_bars < self.config.stage2_min_history_5m_bars or observed_bars < self.config.stage2_min_observed_5m_bars:
                        continue
                    feat = compute_live_features(symbol, df, btc_ctx=btc_ctx, eth_ctx=eth_ctx, btc_df=btc_df, cross_exchange=None)
                    product = stage2_seed_products.get(symbol, {})
                    diag = {**feat.diagnostics, "rolling_dollar_volume": float(product.get("rolling_dollar_volume", 0.0))}
                    guard = compute_guardrails(symbol, feat.feature_row, diag, feat.block_reason, self.state.model_metadata.get("pt2"), self.config)
                    row = self._score_omitted_stage2_row(
                        symbol=symbol,
                        row=feat.feature_row,
                        diag=diag,
                        guard=guard,
                        market_regime=market_regime,
                        btc_regime=btc_regime,
                        bundle=bundle,
                        score_contract=score_contract,
                        sector_penalty=self._get_sector_penalty(symbol, sector_leader_rets),
                    )
                    if row is None:
                        continue
                    row["stage1_primary_rank"] = primary_ranks.get(symbol)
                    row["stage1_selection_source"] = selection_sources.get(symbol)
                    rows.append(row)
                except Exception:
                    continue
        rows.sort(key=lambda r: (float(r.get("live_score") or 0.0), str(r.get("symbol") or "")), reverse=True)
        return rows

    def _build_stage1_selection_repair_review(
        self,
        *,
        current_mode: str,
        current_rows: List[dict],
        mode_rows: Dict[str, List[dict]],
        mode_meta: Dict[str, dict] | None = None,
    ) -> dict:
        current_rows = [dict(r) for r in list(current_rows or [])]
        if not current_rows:
            return {
                "available": False,
                "headline": "No stage1 selection repair review available yet",
                "summary": "Run a complete scan before comparing alternative Stage 1 selection modes.",
                "current_mode": str(current_mode or "primary_only"),
                "recommended_mode": None,
                "mode_rows": [],
            }
        current_mode = str(current_mode or "primary_only")
        current_stats = self._stage1_omission_stats(current_rows)
        current_symbols = {str(r.get("symbol") or "") for r in current_rows if str(r.get("symbol") or "")}
        mode_meta = dict(mode_meta or {})
        comparisons = []
        best = None
        best_key = None
        for mode, rows in list(mode_rows.items()):
            mode = str(mode or "")
            if not mode or mode == current_mode:
                continue
            stats = self._stage1_omission_stats(rows)
            promoted = [
                {
                    "symbol": r.get("symbol"),
                    "live_score": r.get("live_score"),
                    "pre_policy_score": r.get("pre_policy_score"),
                    "stage1_primary_rank": r.get("stage1_primary_rank"),
                    "stage1_selection_source": r.get("stage1_selection_source"),
                }
                for r in sorted(rows, key=lambda r: (float(r.get("live_score") or 0.0), str(r.get("symbol") or "")), reverse=True)
                if str(r.get("symbol") or "") not in current_symbols
            ][:5]
            row = {
                "mode": mode,
                "headline": f"{mode} shadow review",
                "summary": None,
                "rows": int(stats.get("rows") or 0),
                "max_live_score": stats.get("max_live_score"),
                "count_ge_0_35": int(stats.get("count_ge_0_35") or 0),
                "count_ge_0_45": int(stats.get("count_ge_0_45") or 0),
                "count_ge_0_50": int(stats.get("count_ge_0_50") or 0),
                "delta_ge_0_35": int(stats.get("count_ge_0_35") or 0) - int(current_stats.get("count_ge_0_35") or 0),
                "delta_ge_0_45": int(stats.get("count_ge_0_45") or 0) - int(current_stats.get("count_ge_0_45") or 0),
                "delta_ge_0_50": int(stats.get("count_ge_0_50") or 0) - int(current_stats.get("count_ge_0_50") or 0),
                "delta_max_live_score": round(float(stats.get("max_live_score") or 0.0) - float(current_stats.get("max_live_score") or 0.0), 4),
                "promoted_symbols": promoted,
                "selection_meta": mode_meta.get(mode) or {},
            }
            row["summary"] = (
                f"{mode} would produce {row['count_ge_0_35']} rows >= 0.35 and {row['count_ge_0_45']} rows >= 0.45, "
                f"vs {int(current_stats.get('count_ge_0_35') or 0)} and {int(current_stats.get('count_ge_0_45') or 0)} for {current_mode}."
            )
            comparisons.append(row)
            key = (
                row["count_ge_0_45"],
                row["count_ge_0_35"],
                float(row["max_live_score"] or 0.0),
                -len(promoted),
                mode,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = row
        comparisons.sort(key=lambda r: (r["count_ge_0_45"], r["count_ge_0_35"], float(r.get("max_live_score") or 0.0), r["mode"]), reverse=True)
        supported = False
        if best is not None:
            supported = (
                int(best.get("delta_ge_0_45") or 0) > 0
                or int(best.get("delta_ge_0_35") or 0) >= 2
                or float(best.get("delta_max_live_score") or 0.0) >= 0.05
            )
        if not comparisons:
            verdict = "no_alternative_modes_reviewed"
            headline = "No stage1 repair comparison available"
            summary = "No alternative Stage 1 modes produced a comparable shadow shortlist in this scan."
            recommended_mode = None
        elif supported and best is not None:
            verdict = "stage1_repair_mode_supported"
            headline = f"{best.get('mode')} looks like the best Stage 1 repair candidate"
            summary = (
                f"Relative to {current_mode}, {best.get('mode')} would lift the scan upper tail from "
                f"{float(current_stats.get('max_live_score') or 0.0):.4f} to {float(best.get('max_live_score') or 0.0):.4f} "
                f"and change rows >= 0.45 from {int(current_stats.get('count_ge_0_45') or 0)} to {int(best.get('count_ge_0_45') or 0)}. "
                f"Investigate this mode before lowering the Stage 2 threshold."
            )
            recommended_mode = best.get("mode")
        else:
            verdict = "no_clear_stage1_repair_mode"
            headline = "No alternative Stage 1 mode clearly beats the current mode yet"
            summary = (
                f"Alternative Stage 1 modes were shadow-compared against {current_mode}, but none clearly improved the upper tail enough "
                f"to justify a live mode change from this scan alone."
            )
            recommended_mode = None
        return {
            "available": True,
            "headline": headline,
            "summary": summary,
            "verdict": verdict,
            "current_mode": current_mode,
            "current_mode_stats": current_stats,
            "recommended_mode": recommended_mode,
            "recommended_mode_summary": best if supported else None,
            "mode_rows": comparisons,
        }

    def _run_stage1_selection_repair_review(
        self,
        *,
        stage1_input_rows: Dict[str, dict],
        stage1_guardrails: Dict[str, dict],
        stage1_candidates: List[str],
        stage1_selection_meta: Dict[str, dict] | None,
        stage2_seed_products: Dict[str, dict],
        btc_ctx: dict,
        eth_ctx: dict,
        btc_df: pd.DataFrame | None,
        market_regime,
        btc_regime: str,
        bundle: ModelBundle | None,
        score_contract: dict,
        sector_leader_rets: Dict[str, float],
        final_rows: List[dict],
    ) -> dict:
        current_mode = str((stage1_selection_meta or {}).get("selection_mode") or getattr(self.config, "stage1_selection_mode", "primary_only") or "primary_only")
        final_rows = [dict(r) for r in list(final_rows or [])]
        if not final_rows:
            return self._build_stage1_selection_repair_review(current_mode=current_mode, current_rows=[], mode_rows={}, mode_meta={})
        current_set = {str(s) for s in list(stage1_candidates or [])}
        current_rows = [dict(r) for r in final_rows if str(r.get("symbol") or "") in current_set]
        if not current_rows:
            current_rows = list(final_rows)
        opportunity_scores = dict((stage1_selection_meta or {}).get("opportunity_scores") or {})
        mode_names = [
            "primary_only",
            "hybrid_primary_plus_recall_reserve",
            "primary_plus_near_miss_recall_promotion",
            "stage1_opportunity_model",
            "primary_plus_opportunity_reserve",
        ]
        mode_to_symbols: Dict[str, List[str]] = {}
        mode_to_meta: Dict[str, dict] = {}
        for mode in mode_names:
            selected, meta = stage1_select(
                stage1_input_rows,
                stage1_guardrails,
                int(getattr(self.config, "stage1_max_candidates", 40) or 40),
                btc_regime=btc_regime,
                selection_mode=mode,
                recall_reserve_frac=float(getattr(self.config, "stage1_recall_reserve_frac", 0.25) or 0.25),
                recall_reserve_min=int(getattr(self.config, "stage1_recall_reserve_min", 6) or 6),
                recall_reserve_max=int(getattr(self.config, "stage1_recall_reserve_max", 12) or 12),
                promotion_overflow_window=int(getattr(self.config, "stage1_promotion_overflow_window", 20) or 20),
                opportunity_model_scores=opportunity_scores,
            )
            mode_to_symbols[mode] = list(selected)
            mode_to_meta[mode] = dict(meta or {})
        baseline_row_map = {str(r.get("symbol") or ""): dict(r) for r in current_rows}
        shadow_symbols = sorted({
            symbol
            for mode, symbols in mode_to_symbols.items()
            for symbol in symbols
            if symbol not in baseline_row_map
        })
        shadow_rows = self._shadow_score_stage2_symbols(
            audit_symbols=shadow_symbols,
            stage2_seed_products=stage2_seed_products,
            btc_ctx=btc_ctx,
            eth_ctx=eth_ctx,
            btc_df=btc_df,
            market_regime=market_regime,
            btc_regime=btc_regime,
            bundle=bundle,
            score_contract=score_contract,
            sector_leader_rets=sector_leader_rets,
            primary_ranks=(stage1_selection_meta or {}).get("primary_ranks") or {},
            selection_sources={},
        )
        shadow_row_map = {str(r.get("symbol") or ""): dict(r) for r in shadow_rows}
        mode_rows: Dict[str, List[dict]] = {}
        for mode, symbols in mode_to_symbols.items():
            rows: List[dict] = []
            selected_sources = dict((mode_to_meta.get(mode) or {}).get("selected_sources") or {})
            primary_ranks = dict((mode_to_meta.get(mode) or {}).get("primary_ranks") or {})
            for symbol in symbols:
                base = baseline_row_map.get(symbol) or shadow_row_map.get(symbol)
                if not base:
                    continue
                row = dict(base)
                row["stage1_primary_rank"] = primary_ranks.get(symbol)
                row["stage1_selection_source"] = selected_sources.get(symbol)
                rows.append(row)
            rows.sort(key=lambda r: (float(r.get("live_score") or 0.0), str(r.get("symbol") or "")), reverse=True)
            mode_rows[mode] = rows
        return self._build_stage1_selection_repair_review(
            current_mode=current_mode,
            current_rows=current_rows,
            mode_rows=mode_rows,
            mode_meta=mode_to_meta,
        )

    def _run_stage1_omission_audit(
        self,
        *,
        stage1_input_rows: Dict[str, dict],
        stage1_guardrails: Dict[str, dict],
        stage1_candidates: List[str],
        stage1_selection_meta: Dict[str, dict] | None,
        stage2_seed_products: Dict[str, dict],
        btc_ctx: dict,
        eth_ctx: dict,
        btc_df: pd.DataFrame | None,
        market_regime,
        btc_regime: str,
        bundle: ModelBundle | None,
        score_contract: dict,
        sector_leader_rets: Dict[str, float],
        final_rows: List[dict],
    ) -> dict:
        meta = stage1_selection_meta or {}
        primary_ranks = dict(meta.get("primary_ranks") or {})
        selected_set = set(str(s) for s in list(stage1_candidates or []))
        selected_rows = [dict(r) for r in list(final_rows or []) if str(r.get("symbol") or "") in selected_set]
        omitted_symbols = [
            symbol for symbol, _rank in sorted(primary_ranks.items(), key=lambda item: (int(item[1]), item[0]))
            if symbol not in selected_set and str((stage1_guardrails.get(symbol) or {}).get("block_code") or "") != "BLOCKED"
        ]
        audit_limit = 25
        audit_symbols = omitted_symbols[:audit_limit]
        if not audit_symbols:
            return self._build_stage1_omission_audit_summary(
                selected_rows=selected_rows,
                omitted_rows=[],
                omitted_total=len(omitted_symbols),
                omitted_audited=0,
                omitted_truncated=False,
                stage1_selection_meta=stage1_selection_meta,
            )
        omitted_rows = self._shadow_score_stage2_symbols(
            audit_symbols=audit_symbols,
            stage2_seed_products=stage2_seed_products,
            btc_ctx=btc_ctx,
            eth_ctx=eth_ctx,
            btc_df=btc_df,
            market_regime=market_regime,
            btc_regime=btc_regime,
            bundle=bundle,
            score_contract=score_contract,
            sector_leader_rets=sector_leader_rets,
            primary_ranks=primary_ranks,
            selection_sources={},
        )
        return self._build_stage1_omission_audit_summary(
            selected_rows=selected_rows,
            omitted_rows=omitted_rows,
            omitted_total=len(omitted_symbols),
            omitted_audited=len(audit_symbols),
            omitted_truncated=len(omitted_symbols) > len(audit_symbols),
            stage1_selection_meta=stage1_selection_meta,
        )

    def _effective_market_regime_actionability(self, market_regime, *, live_pipeline_mode: str) -> tuple[str | None, str | None]:
        raw_state = str(getattr(market_regime, "actionability_state", None) or "") or None
        if str(live_pipeline_mode or "").strip().lower() != "raw_threshold":
            return raw_state, None
        state = str(getattr(market_regime, "state", None) or "")
        if raw_state == "pending_blocked":
            return "advisory_pending", "raw-threshold mode keeps regime state informative while regime inputs are still catching up"
        if state == "green" and raw_state in {None, "", "normal"}:
            return "normal", None
        if state == "amber":
            return "advisory_only", "raw-threshold mode treats amber regime state as advisory and does not hide rows solely because of regime status"
        if state == "red":
            return "advisory_high_risk", "raw-threshold mode treats red regime state as high-risk advisory and does not hide rows solely because of regime status"
        if raw_state in {"blocked", "cooldown_restricted"}:
            return "advisory_only", "raw-threshold mode bypasses regime/cooldown suppression; regime remains advisory only"
        return raw_state, None

    def _build_decision_summary(self, *, visible_rows: List[dict], score_contract: dict, market_regime, hidden_watchlist_rows: int = 0, blocked_rows: List[dict] | None = None, effective_market_regime_actionability: str | None = None) -> dict:
        validated_floor = self._validated_floor(score_contract)
        near_floor = min(validated_floor, float(getattr(self.config, "stage2_near_validated_floor", 0.45) or 0.45))
        action_ready = [r for r in visible_rows if str(r.get("actionability_tier") or "") == "action_ready"]
        selective = [r for r in visible_rows if str(r.get("actionability_tier") or "") == "selective"]
        watchlist = [r for r in visible_rows if str(r.get("actionability_tier") or "") == "watchlist"]
        validated_rows = [r for r in visible_rows if str(r.get("score_band") or "") == "validated"]
        validated_selective_rows = [r for r in validated_rows if str(r.get("actionability_tier") or "") == "selective"]
        validated_watchlist_rows = [r for r in validated_rows if str(r.get("actionability_tier") or "") == "watchlist"]
        near_rows = [r for r in visible_rows if str(r.get("score_band") or "") == "near_validated"]
        exploratory_rows = [r for r in visible_rows if str(r.get("score_band") or "") == "exploratory"]
        objective_confirmed_rows = [r for r in visible_rows if str(r.get("objective_score_band") or "") in {"confirmed_shortlist", "strong_edge", "priority_edge", "elite_edge"}]
        strong_edge_rows = [r for r in visible_rows if str(r.get("objective_score_band") or "") == "strong_edge"]
        priority_edge_rows = [r for r in visible_rows if str(r.get("objective_score_band") or "") in {"priority_edge", "elite_edge"}]
        elite_edge_rows = [r for r in visible_rows if str(r.get("objective_score_band") or "") == "elite_edge"]
        top_focus_n = max(1, int(getattr(self.config, "stage2_decision_focus_top_n", 5) or 5))
        blocked_focus_n = max(1, int(getattr(self.config, "stage2_blocked_focus_top_n", 3) or 3))
        top_focus = [
            {
                "symbol": r.get("symbol"),
                "live_score": r.get("live_score"),
                "distance_to_validated": r.get("distance_to_validated"),
                "distance_to_live_threshold": r.get("distance_to_live_threshold"),
                "live_threshold": r.get("live_threshold"),
                "actionability_tier": r.get("actionability_tier"),
                "score_band": r.get("score_band"),
                "score_band_label": r.get("score_band_label"),
                "visibility_band": r.get("visibility_band"),
                "visibility_band_label": r.get("visibility_band_label"),
                "objective_score_band": r.get("objective_score_band"),
                "objective_score_band_label": r.get("objective_score_band_label"),
                "objective_quality_reference_rate": r.get("objective_quality_reference_rate"),
            }
            for r in visible_rows[:top_focus_n]
        ]
        hidden_watchlist_rows = max(0, int(hidden_watchlist_rows or 0))
        blocked_rows = list(blocked_rows or [])
        blocked_rows.sort(key=self._blocked_focus_sort_key, reverse=True)
        blocked_near_rows = [r for r in blocked_rows if str(r.get("pre_policy_score_band") or "") == "near_validated"]
        blocked_near_threshold_rows = [r for r in blocked_rows if str(r.get("visibility_band") or "") == "near_visibility"]
        blocked_exploratory_rows = [r for r in blocked_rows if str(r.get("pre_policy_score_band") or "") != "near_validated"]
        if blocked_near_rows:
            blocked_focus_source = blocked_near_rows
        elif blocked_near_threshold_rows:
            blocked_focus_source = blocked_near_threshold_rows
        else:
            blocked_focus_source = blocked_rows
        blocked_focus = [
            {
                "symbol": r.get("symbol"),
                "pre_policy_score": r.get("pre_policy_score"),
                "live_score": r.get("live_score"),
                "live_threshold": r.get("live_threshold"),
                "pre_policy_distance_to_validated": r.get("pre_policy_distance_to_validated"),
                "distance_to_validated": r.get("distance_to_validated"),
                "distance_to_live_threshold": r.get("distance_to_live_threshold"),
                "distance_to_live_threshold_pct_points": r.get("distance_to_live_threshold_pct_points"),
                "pre_policy_score_band": r.get("pre_policy_score_band"),
                "pre_policy_score_band_label": r.get("pre_policy_score_band_label"),
                "score_band": r.get("score_band"),
                "score_band_label": r.get("score_band_label"),
                "visibility_band": r.get("visibility_band"),
                "visibility_band_label": r.get("visibility_band_label"),
                "objective_score_band": r.get("objective_score_band"),
                "objective_score_band_label": r.get("objective_score_band_label"),
                "objective_quality_reference_rate": r.get("objective_quality_reference_rate"),
                "liquidity_tier": r.get("liquidity_tier"),
                "pre_policy_rank": r.get("pre_policy_rank") or r.get("candidate_rank_all"),
                "suppression_reason": r.get("suppression_reason"),
                "suppression_reason_detail": r.get("suppression_reason_detail") or r.get("policy_constraint_reason"),
            }
            for r in blocked_focus_source[:blocked_focus_n]
        ]
        cooldown_active = bool(getattr(market_regime, "cooldown_active", False))
        cooldown_until_utc = getattr(market_regime, "cooldown_until_utc", None)
        best_blocked_threshold_gap = min((float(r.get("distance_to_live_threshold") or 0.0) for r in blocked_rows), default=None)

        objective_contract = self._objective_semantics_contract()
        confirmed_floor = objective_contract.get("confirmed_shortlist_floor") if isinstance(objective_contract, dict) else None
        strong_floor = objective_contract.get("strong_edge_floor") if isinstance(objective_contract, dict) else None
        priority_floor = objective_contract.get("priority_edge_floor") if isinstance(objective_contract, dict) else None
        confirmed_quality = objective_contract.get("confirmed_shortlist_quality_reference") if isinstance(objective_contract, dict) else None

        if action_ready:
            headline = f"{len(action_ready)} validated candidate{'s' if len(action_ready) != 1 else ''} ready now"
            summary = f"Validated live scores reached at least {validated_floor:.2f}. Prioritize the action-ready shortlist; {len(selective)} additional selective rows remain cautionary."
        elif validated_rows:
            if objective_confirmed_rows and confirmed_floor is not None:
                if elite_edge_rows:
                    headline = f"{len(objective_confirmed_rows)} confirmed-shortlist row{'s' if len(objective_confirmed_rows) != 1 else ''} surfaced, including {len(elite_edge_rows)} elite-edge name{'s' if len(elite_edge_rows) != 1 else ''}"
                elif priority_edge_rows:
                    headline = f"{len(objective_confirmed_rows)} confirmed-shortlist row{'s' if len(objective_confirmed_rows) != 1 else ''} surfaced, including {len(priority_edge_rows)} priority-edge name{'s' if len(priority_edge_rows) != 1 else ''}"
                elif strong_edge_rows:
                    headline = f"{len(objective_confirmed_rows)} confirmed-shortlist row{'s' if len(objective_confirmed_rows) != 1 else ''} surfaced"
                else:
                    headline = f"{len(validated_rows)} validated-band row{'s' if len(validated_rows) != 1 else ''} surfaced"
            else:
                headline = f"{len(validated_rows)} validated-band row{'s' if len(validated_rows) != 1 else ''} surfaced, but none are action-ready"
            selective_count = len(validated_selective_rows)
            watchlist_count = len(validated_watchlist_rows)
            summary = (
                f"The scanner surfaced {len(validated_rows)} visible row{'s' if len(validated_rows) != 1 else ''} at or above the validated floor ({validated_floor:.2f}+), "
                f"but they remain advisory-only: {selective_count} selective and {watchlist_count} watchlist. "
                f"Treat them as validated-band monitoring rows, not direct action signals."
            )
            if objective_confirmed_rows and confirmed_floor is not None:
                summary += (
                    f" {len(objective_confirmed_rows)} visible row{'s' if len(objective_confirmed_rows) != 1 else ''} also cleared the confirmed shortlist floor "
                    f"({float(confirmed_floor):.2f}+)."
                )
                if strong_floor is not None:
                    summary += f" {len(strong_edge_rows)} reached the strong edge band ({float(strong_floor):.2f}+)."
                if priority_floor is not None:
                    summary += f" {len(priority_edge_rows)} reached the priority edge band ({float(priority_floor):.2f}+)."
                if elite_edge_rows:
                    summary += f" {len(elite_edge_rows)} reached the elite edge band."
            if hidden_watchlist_rows > 0:
                summary += f" {hidden_watchlist_rows} lower-priority watchlist rows were hidden from the visible shortlist and preserved in the review pack."
        elif selective:
            headline = f"No fully action-ready rows; {len(selective)} selective candidate{'s' if len(selective) != 1 else ''}"
            summary = f"This scan produced selective rows, but nothing cleared the full action-ready bar. Focus on the best {min(len(selective), top_focus_n)} names and keep policy constraints in view."
        elif near_rows:
            headline = f"No validated candidates; {len(near_rows)} near-band name{'s' if len(near_rows) != 1 else ''} worth monitoring"
            best_gap = min(float(r.get("distance_to_validated") or 0.0) for r in near_rows) if near_rows else 0.0
            summary = f"Nothing reached the validated band ({validated_floor:.2f}+), but the nearest shortlist names are within {best_gap * 100.0:.1f} percentage points of it. Treat the visible list as a monitoring queue, not a trading signal."
            if hidden_watchlist_rows > 0:
                summary += f" {hidden_watchlist_rows} lower-priority watchlist rows were hidden from the visible shortlist and preserved in the review pack."
        elif watchlist:
            if objective_confirmed_rows and confirmed_floor is not None:
                headline = f"No validated-tail candidates; {len(objective_confirmed_rows)} confirmed-shortlist name{'s' if len(objective_confirmed_rows) != 1 else ''} surfaced"
                summary = (
                    f"{len(objective_confirmed_rows)} visible row{'s' if len(objective_confirmed_rows) != 1 else ''} cleared the confirmed shortlist floor ({float(confirmed_floor):.2f}+)"
                )
                if confirmed_quality is not None:
                    summary += f", where current resolved visible quality is {float(confirmed_quality) * 100.0:.1f}%"
                summary += ". "
                if strong_floor is not None:
                    summary += f"{len(strong_edge_rows)} reached the strong edge band ({float(strong_floor):.2f}+). "
                if priority_floor is not None:
                    summary += f"{len(priority_edge_rows)} reached the priority edge band ({float(priority_floor):.2f}+). "
                summary += f"None reached the validated tail-probability band ({validated_floor:.2f}+), so use them as ranked decision-support names rather than tail-validated probabilities."
            else:
                headline = "No validated candidates this scan"
                summary = f"Visible names remain below the near-validated band ({near_floor:.2f}+). Treat the shortlist as exploratory only unless later scans improve."
            if blocked_near_threshold_rows and best_blocked_threshold_gap is not None:
                symbols = ", ".join(str(r.get("symbol")) for r in blocked_focus[:blocked_focus_n] if r.get("symbol"))
                summary += f" {len(blocked_near_threshold_rows)} blocked monitoring names sit within {best_blocked_threshold_gap * 100.0:.1f} percentage points of the current live threshold ({symbols or 'see blocked monitoring rows'})."
            if hidden_watchlist_rows > 0:
                summary += f" {hidden_watchlist_rows} lower-priority watchlist rows were hidden from the visible shortlist and preserved in the review pack."
        elif blocked_near_rows:
            headline = f"No visible candidates; {len(blocked_near_rows)} policy-blocked near-band name{'s' if len(blocked_near_rows) != 1 else ''}"
            best_gap = min(float(r.get("pre_policy_distance_to_validated") or 0.0) for r in blocked_near_rows) if blocked_near_rows else 0.0
            blocked_symbols = ", ".join(str(r.get("symbol")) for r in blocked_focus[:blocked_focus_n] if r.get("symbol"))
            summary = f"Live policy prevented any visible shortlist rows, but {blocked_symbols or 'the top blocked monitoring names'} were within {best_gap * 100.0:.1f} percentage points of the validated band before regime haircuts. Treat them as blocked monitoring rows only, not trade candidates."
            if cooldown_active and cooldown_until_utc:
                summary += f" Cooldown remains active until {cooldown_until_utc}."
        elif blocked_near_threshold_rows:
            headline = f"No visible candidates; {len(blocked_near_threshold_rows)} blocked name{'s' if len(blocked_near_threshold_rows) != 1 else ''} close to current threshold"
            blocked_symbols = ", ".join(str(r.get("symbol")) for r in blocked_focus[:blocked_focus_n] if r.get("symbol"))
            summary = f"No rows cleared live policy, but {blocked_symbols or 'the top blocked monitoring names'} sit within {float(best_blocked_threshold_gap or 0.0) * 100.0:.1f} percentage points of the current live threshold. Treat them as blocked monitoring rows only, not trade candidates."
            if cooldown_active and cooldown_until_utc:
                summary += f" Cooldown remains active until {cooldown_until_utc}."
        elif blocked_focus:
            strongest = blocked_focus[0]
            headline = "No visible candidates; top blocked monitoring names remain below tradeable range"
            summary = (
                f"The scanner did not surface any visible shortlist rows after live policy filters. "
                f"Top blocked monitoring row: {strongest.get('symbol') or '-'} at pre-policy {float(strongest.get('pre_policy_score') or 0.0):.2f}, "
                f"live {float(strongest.get('live_score') or 0.0):.2f}, threshold gap {float(strongest.get('distance_to_live_threshold') or 0.0) * 100.0:.1f} percentage points; "
                f"reason: {strongest.get('suppression_reason_detail') or strongest.get('suppression_reason') or '-'}."
            )
            if cooldown_active and cooldown_until_utc:
                summary += f" Cooldown remains active until {cooldown_until_utc}."
        else:
            headline = "No visible candidates this scan"
            summary = "The scanner did not surface any visible shortlist rows after live policy and output filters."
        return {
            "headline": headline,
            "summary": summary,
            "validated_floor": round(validated_floor, 4),
            "near_validated_floor": round(near_floor, 4),
            "action_ready_rows": len(action_ready),
            "selective_rows": len(selective),
            "watchlist_rows": len(watchlist),
            "validated_rows": len(validated_rows),
            "validated_selective_rows": len(validated_selective_rows),
            "validated_watchlist_rows": len(validated_watchlist_rows),
            "near_validated_rows": len(near_rows),
            "exploratory_rows": len(exploratory_rows),
            "objective_confirmed_rows": len(objective_confirmed_rows),
            "strong_edge_rows": len(strong_edge_rows),
            "priority_edge_rows": len(priority_edge_rows),
            "elite_edge_rows": len(elite_edge_rows),
            "objective_semantics_contract": objective_contract if isinstance(objective_contract, dict) else {},
            "hidden_watchlist_rows": hidden_watchlist_rows,
            "top_focus_symbols": top_focus,
            "no_validated_candidates": len(validated_rows) == 0,
            "market_regime_state": getattr(market_regime, "state", None),
            "market_regime_actionability": effective_market_regime_actionability if effective_market_regime_actionability is not None else getattr(market_regime, "actionability_state", None),
            "blocked_rows": len(blocked_rows),
            "blocked_near_validated_rows": len(blocked_near_rows),
            "blocked_near_threshold_rows": len(blocked_near_threshold_rows),
            "blocked_exploratory_rows": len(blocked_exploratory_rows),
            "best_blocked_threshold_gap": round(float(best_blocked_threshold_gap), 4) if best_blocked_threshold_gap is not None else None,
            "blocked_focus_symbols": blocked_focus,
            "blocked_focus_count": len(blocked_focus),
            "cooldown_active": cooldown_active,
            "cooldown_until_utc": cooldown_until_utc,
        }

    def _limit_visible_shortlist(self, rows: List[dict], *, effective_max: int, tracked_priority_symbols: List[str] | None = None) -> tuple[List[dict], List[dict], dict]:
        rows = list(rows or [])
        effective_max = max(0, int(effective_max or 0))
        if str(getattr(self.config, "live_selection_mode", "utility_constrained") or "utility_constrained").lower() == "utility_constrained":
            utility_tuning_override = load_active_utility_tuning_override(self.config.model_dir)
            override_source = str((utility_tuning_override or {}).get('source') or '')
            if utility_tuning_override:
                self.state.update_status(
                    live_utility_tuning_proof={
                        'active': override_source == 'utility_tuning_proof',
                        'proof_session_id': utility_tuning_override.get('proof_session_id') if override_source == 'utility_tuning_proof' else None,
                        'activated_at_utc': utility_tuning_override.get('activated_at_utc') if override_source == 'utility_tuning_proof' else None,
                        'expires_at_utc': utility_tuning_override.get('expires_at_utc') if override_source == 'utility_tuning_proof' else None,
                        'state_scope_key': utility_tuning_override.get('state_scope_key') if override_source == 'utility_tuning_proof' else None,
                        'utility_selection_engine_label': utility_tuning_override.get('utility_selection_engine_label') if override_source == 'utility_tuning_proof' else None,
                        'utility_expected_edge_weight': utility_tuning_override.get('utility_expected_edge_weight') if override_source == 'utility_tuning_proof' else None,
                        'utility_confidence_weight': utility_tuning_override.get('utility_confidence_weight') if override_source == 'utility_tuning_proof' else None,
                        'utility_probability_weight': utility_tuning_override.get('utility_probability_weight') if override_source == 'utility_tuning_proof' else None,
                        'utility_shortlist_target_max_names': utility_tuning_override.get('utility_shortlist_target_max_names') if override_source == 'utility_tuning_proof' else None,
                        'utility_shortlist_score_floor': utility_tuning_override.get('utility_shortlist_score_floor') if override_source == 'utility_tuning_proof' else None,
                        'utility_shortlist_score_dropoff': utility_tuning_override.get('utility_shortlist_score_dropoff') if override_source == 'utility_tuning_proof' else None,
                        'utility_confidence_floor': utility_tuning_override.get('utility_confidence_floor') if override_source == 'utility_tuning_proof' else None,
                        'utility_tier3_max_frac': utility_tuning_override.get('utility_tier3_max_frac') if override_source == 'utility_tuning_proof' else None,
                    },
                    live_utility_model_proof={
                        'active': override_source == 'utility_model_proof',
                        'proof_session_id': utility_tuning_override.get('proof_session_id') if override_source == 'utility_model_proof' else None,
                        'activated_at_utc': utility_tuning_override.get('activated_at_utc') if override_source == 'utility_model_proof' else None,
                        'expires_at_utc': utility_tuning_override.get('expires_at_utc') if override_source == 'utility_model_proof' else None,
                        'state_scope_key': utility_tuning_override.get('state_scope_key') if override_source == 'utility_model_proof' else None,
                        'utility_selection_engine_label': utility_tuning_override.get('utility_selection_engine_label') if override_source == 'utility_model_proof' else None,
                        'utility_expected_edge_weight': utility_tuning_override.get('utility_expected_edge_weight') if override_source == 'utility_model_proof' else None,
                        'utility_confidence_weight': utility_tuning_override.get('utility_confidence_weight') if override_source == 'utility_model_proof' else None,
                        'utility_probability_weight': utility_tuning_override.get('utility_probability_weight') if override_source == 'utility_model_proof' else None,
                        'utility_shortlist_target_max_names': utility_tuning_override.get('utility_shortlist_target_max_names') if override_source == 'utility_model_proof' else None,
                        'utility_shortlist_score_floor': utility_tuning_override.get('utility_shortlist_score_floor') if override_source == 'utility_model_proof' else None,
                        'utility_shortlist_score_dropoff': utility_tuning_override.get('utility_shortlist_score_dropoff') if override_source == 'utility_model_proof' else None,
                        'utility_confidence_floor': utility_tuning_override.get('utility_confidence_floor') if override_source == 'utility_model_proof' else None,
                        'utility_tier3_max_frac': utility_tuning_override.get('utility_tier3_max_frac') if override_source == 'utility_model_proof' else None,
                    },
                    live_utility_tuning_adoption={
                        'active': override_source == 'utility_tuning_adoption',
                        'adoption_session_id': utility_tuning_override.get('adoption_session_id') if override_source == 'utility_tuning_adoption' else None,
                        'adopted_at_utc': utility_tuning_override.get('adopted_at_utc') if override_source == 'utility_tuning_adoption' else None,
                        'state_scope_key': utility_tuning_override.get('state_scope_key') if override_source == 'utility_tuning_adoption' else None,
                        'utility_selection_engine_label': utility_tuning_override.get('utility_selection_engine_label') if override_source == 'utility_tuning_adoption' else None,
                        'utility_expected_edge_weight': utility_tuning_override.get('utility_expected_edge_weight') if override_source == 'utility_tuning_adoption' else None,
                        'utility_confidence_weight': utility_tuning_override.get('utility_confidence_weight') if override_source == 'utility_tuning_adoption' else None,
                        'utility_probability_weight': utility_tuning_override.get('utility_probability_weight') if override_source == 'utility_tuning_adoption' else None,
                        'utility_shortlist_target_max_names': utility_tuning_override.get('utility_shortlist_target_max_names') if override_source == 'utility_tuning_adoption' else None,
                        'utility_shortlist_score_floor': utility_tuning_override.get('utility_shortlist_score_floor') if override_source == 'utility_tuning_adoption' else None,
                        'utility_shortlist_score_dropoff': utility_tuning_override.get('utility_shortlist_score_dropoff') if override_source == 'utility_tuning_adoption' else None,
                        'utility_confidence_floor': utility_tuning_override.get('utility_confidence_floor') if override_source == 'utility_tuning_adoption' else None,
                        'utility_tier3_max_frac': utility_tuning_override.get('utility_tier3_max_frac') if override_source == 'utility_tuning_adoption' else None,
                    }
                )
            else:
                self.state.update_status(live_utility_tuning_proof={'active': False}, live_utility_model_proof={'active': False}, live_utility_tuning_adoption={'active': False}, live_utility_model_adoption={'active': False})
            utility_config = utility_config_with_runtime_override(self.config, utility_tuning_override)
            result = optimize_visible_shortlist(rows, effective_max=effective_max, config=utility_config, tracked_priority_symbols=tracked_priority_symbols)
            return result.visible_rows, result.trimmed_rows, result.meta

        tracked_set = {str(s) for s in (tracked_priority_symbols or []) if str(s)}
        pin_cap = max(0, int(getattr(self.config, "cooldown_followup_visible_pin_count", 5) or 5))

        def _pin(rows_in: List[dict]) -> List[dict]:
            if not tracked_set:
                return list(rows_in)
            pinned = [r for r in rows_in if str(r.get("symbol") or "") in tracked_set][:pin_cap]
            pinned_ids = {id(r) for r in pinned}
            others = [r for r in rows_in if id(r) not in pinned_ids]
            return pinned + others

        action_selective = [r for r in rows if str(r.get("actionability_tier") or "") in {"action_ready", "selective"}]
        watchlist = [r for r in rows if str(r.get("actionability_tier") or "") == "watchlist"]
        base_watchlist_cap = max(0, int(getattr(self.config, "stage2_watchlist_max_names", 12) or 12))
        watchlist_only_cap = max(0, int(getattr(self.config, "stage2_watchlist_only_max_names", base_watchlist_cap) or base_watchlist_cap))
        exploratory_only_cap = max(0, int(getattr(self.config, "stage2_watchlist_only_exploratory_max_names", 5) or 5))
        watchlist_cap = watchlist_only_cap if not action_selective else base_watchlist_cap
        kept_action = _pin(action_selective)[:effective_max] if effective_max else []
        remaining_slots = max(0, effective_max - len(kept_action))
        if action_selective:
            kept_watchlist = _pin(watchlist)[: min(remaining_slots, watchlist_cap)]
        else:
            near_watchlist = _pin([r for r in watchlist if str(r.get("score_band") or "") == "near_validated"])
            exploratory_watchlist = _pin([r for r in watchlist if str(r.get("score_band") or "") != "near_validated"])
            near_cap = min(remaining_slots, watchlist_cap)
            kept_near = near_watchlist[:near_cap]
            remaining_after_near = max(0, near_cap - len(kept_near))
            kept_exploratory = exploratory_watchlist[: min(remaining_after_near, exploratory_only_cap)]
            kept_watchlist = kept_near + kept_exploratory
        pre_cap_visible = kept_action + kept_watchlist
        max_symbol_share = 0.25
        absolute_symbol_cap = 2
        visible_symbol_counts: Dict[str, int] = {}
        visible: List[dict] = []
        concentration_trimmed: List[dict] = []
        for idx, row in enumerate(pre_cap_visible, start=1):
            symbol = str(row.get("symbol") or "")
            already = int(visible_symbol_counts.get(symbol, 0))
            share_cap = max(1, int(max(1, len(pre_cap_visible)) * max_symbol_share))
            effective_symbol_cap = max(1, min(absolute_symbol_cap, share_cap))
            top5_cap = 2 if idx <= 5 else effective_symbol_cap
            allowed = min(effective_symbol_cap, top5_cap)
            if already >= allowed:
                trimmed_row = dict(row)
                trimmed_row.setdefault("suppression_reason", "symbol_concentration")
                trimmed_row.setdefault("suppression_reason_detail", "trimmed to prevent one symbol from dominating the visible shortlist")
                concentration_trimmed.append(trimmed_row)
                continue
            visible_symbol_counts[symbol] = already + 1
            visible.append(row)
        kept_ids = {id(r) for r in visible}
        trimmed = [r for r in rows if id(r) not in kept_ids]
        if concentration_trimmed:
            trimmed.extend(concentration_trimmed)
        tracked_visible = [r for r in visible if str(r.get("symbol") or "") in tracked_set]
        meta = {
            "watchlist_cap_applied": watchlist_cap,
            "watchlist_visible": len([r for r in visible if str(r.get("actionability_tier") or "") == "watchlist"]),
            "watchlist_trimmed": max(0, len(watchlist) - len([r for r in visible if str(r.get("actionability_tier") or "") == "watchlist"])),
            "action_selective_visible": len([r for r in visible if str(r.get("actionability_tier") or "") in {"action_ready", "selective"}]),
            "near_watchlist_visible": sum(1 for r in visible if str(r.get("actionability_tier") or "") == "watchlist" and str(r.get("score_band") or "") == "near_validated"),
            "exploratory_watchlist_visible": sum(1 for r in visible if str(r.get("actionability_tier") or "") == "watchlist" and str(r.get("score_band") or "") != "near_validated"),
            "exploratory_watchlist_cap_applied": exploratory_only_cap if not action_selective else None,
            "tracked_visible_promoted": len(tracked_visible),
            "tracked_visible_symbols": [r.get("symbol") for r in tracked_visible if r.get("symbol")],
            "symbol_concentration_cap": {"max_share": max_symbol_share, "absolute_cap": absolute_symbol_cap},
            "symbol_concentration_trimmed": len(concentration_trimmed),
        }
        return visible, trimmed, meta


    def _with_publish_meta(self, snapshot, *, attempts: int, successes: int, failures: int, last_attempt: str | None, last_error: str | None, warning_reason: str | None = None):
        snapshot.partial_publish_attempts = int(attempts)
        snapshot.partial_publish_successes = int(successes)
        snapshot.partial_publish_failures = int(failures)
        snapshot.last_partial_publish_attempt_utc = last_attempt
        snapshot.last_partial_publish_error = last_error
        snapshot.regime_publish_warning = bool(warning_reason)
        snapshot.regime_publish_warning_reason = warning_reason
        if snapshot.state == "pending" and successes > 0 and not snapshot.regime_publish_warning:
            snapshot.regime_publish_warning = True
            snapshot.regime_publish_warning_reason = "computed_or_counted_but_not_applied"
        return snapshot

    def _score_contract(self) -> dict:
        meta = self.state.model_metadata.get("pt2") or {}
        status = self.state.get_status()
        repaired_meta, bundle = reconcile_runtime_metadata(
            meta,
            existing_status=status,
            min_count=self.config.tail_validation_min_count,
            min_wilson_lift=self.config.tail_validation_min_wilson_lift,
            min_precision_floor=self.config.tail_validation_min_precision_floor,
            unvalidated_tail_cap=self.config.tail_unvalidated_cap,
            scanner_contract_source="recomputed_runtime_adjusted",
            threshold_suppression_contract_source="recomputed_runtime_adjusted",
        )
        if repaired_meta != meta:
            self.state.set_model_metadata(repaired_meta)
        current_live = status.get("score_contract_live") or {}
        current_raw = status.get("score_contract_raw") or {}
        current_rec = status.get("score_reconciliation") or {}
        if current_live != bundle["score_contract_live"] or current_raw != bundle["score_contract_raw"] or current_rec != bundle["score_reconciliation"]:
            self.state.update_status(
                score_contract=bundle["score_contract"],
                score_contract_live=bundle["score_contract_live"],
                score_contract_raw=bundle["score_contract_raw"],
                score_reconciliation=bundle["score_reconciliation"],
            )
        contract = dict(bundle["score_contract"])
        if (bundle["score_contract_raw"].get("validated_thresholds") or []) and not (bundle["score_contract_live"].get("validated_thresholds") or []):
            notes = list(contract.get("notes") or [])
            extra = "Raw-model holdout tail exists, but adjusted-live score family has no validated >=0.60 tail."
            if extra not in notes:
                notes.append(extra)
            contract["notes"] = notes
        return contract

    def _assess_actionability(
        self,
        *,
        adjusted_score: float,
        trust: dict,
        score_contract: dict,
        market_regime,
        liquidity_tier: str,
        guard: dict,
        objective_band: dict | None = None,
        effective_market_regime_actionability: str | None = None,
    ) -> dict:
        temporal_state = str(score_contract.get("temporal_tail_state") or "")
        temporal_semantics = str(score_contract.get("temporal_tail_semantics") or "")
        tier = "watchlist"
        advisory_reasons: list[str] = []
        policy_constraints: list[str] = []
        evidence = ["tail_contract", "temporal_support", "market_regime", "liquidity", "risk_uncertainty", "score_band"]
        rank = 1
        actionability_type = "advisory_heuristic"

        objective_band = dict(objective_band or {})
        objective_label = str(objective_band.get("objective_score_band_label") or "")
        objective_code = str(objective_band.get("objective_score_band") or "")
        if str(trust.get("probability_semantics") or "") != "validated_tail_probability":
            validated_thresholds = [float(x) for x in (score_contract.get("validated_thresholds") or [])]
            if objective_code in {"confirmed_shortlist", "strong_edge", "priority_edge", "elite_edge"}:
                actionability_type = "objective_semantics_supported"
                advisory_reasons.append(f"below tail-validated probability band, but inside {objective_label.lower()} from replay/current shortlist evidence")
                if objective_code in {"priority_edge", "elite_edge"}:
                    tier = "selective"
                    rank = max(rank, 2)
            elif validated_thresholds:
                lowest_validated = min(validated_thresholds)
                advisory_reasons.append(f"row score fell below validated tail band after policy (<{lowest_validated:.2f})")
            else:
                advisory_reasons.append("score family has no validated tail for actioning")
        elif temporal_state == "validated_tail_temporally_supported":
            tier = "action_ready"
            rank = 3
            actionability_type = "advisory_contract_supported"
            advisory_reasons.append("validated tail has statistically supported reference-band persistence")
        elif temporal_state in {"validated_but_temporally_sparse", "validated_but_temporally_mixed", "validated_but_temporally_unobserved"}:
            tier = "selective"
            rank = 2
            advisory_reasons.append(str(score_contract.get("temporal_note") or "validated tail exists, but temporal support is thin"))
        elif temporal_state in {"temporal_support_unknown", "temporal_support_heuristic_only"}:
            tier = "selective"
            rank = 2
            advisory_reasons.append(str(score_contract.get("temporal_note") or "temporal support is advisory only"))
        else:
            advisory_reasons.append("use as ranked watchlist only")

        regime_actionability = str(effective_market_regime_actionability or getattr(market_regime, 'actionability_state', None) or "")
        if regime_actionability not in {"", "normal", "advisory_only", "advisory_pending"}:
            if tier == "action_ready":
                tier, rank = "selective", 2
            elif tier == "selective":
                tier, rank = "watchlist", 1
            policy_constraints.append(f"market regime is {regime_actionability}")
        elif regime_actionability in {"advisory_only", "advisory_pending"}:
            advisory_reasons.append(f"market regime is {regime_actionability}")

        if str(liquidity_tier) == "tier3":
            if tier == "action_ready":
                tier, rank = "selective", 2
            elif tier == "selective":
                tier, rank = "watchlist", 1
            advisory_reasons.append("tier3 liquidity")

        if float(guard.get("risk", 0.0) or 0.0) >= 0.55 or float(guard.get("uncertainty", 0.0) or 0.0) >= 0.60:
            if tier == "action_ready":
                tier, rank = "selective", 2
            elif tier == "selective":
                tier, rank = "watchlist", 1
            advisory_reasons.append("risk/uncertainty elevated")

        if float(adjusted_score) < 0.70 and tier == "action_ready":
            tier, rank = "selective", 2
            advisory_reasons.append("score below stronger action band")

        return {
            "actionability_tier": tier,
            "actionability_rank": rank,
            "actionability_type": actionability_type,
            "actionability_evidence": list(dict.fromkeys(evidence)),
            "actionability_reason": "; ".join(dict.fromkeys(advisory_reasons)) if advisory_reasons else "none",
            "policy_constraint_reason": "; ".join(dict.fromkeys(policy_constraints)) if policy_constraints else "none",
            "contract_truth_state": score_contract.get("tail_validation_state"),
            "contract_truth_semantics": trust.get("probability_semantics"),
            "temporal_tail_state": temporal_state,
            "temporal_tail_semantics": temporal_semantics,
        }

    def _apply_tail_trust(self, adjusted_score: float, score_contract: dict) -> dict:
        validated_thresholds = [float(x) for x in (score_contract.get("validated_thresholds") or [])]
        validated_set = {round(x, 2) for x in validated_thresholds}
        highest_validated = score_contract.get("highest_validated_threshold")
        unvalidated_cap = float(score_contract.get("unvalidated_tail_cap", self.config.tail_unvalidated_cap) or self.config.tail_unvalidated_cap)
        display_score = float(adjusted_score)
        opportunity_score = round(float(adjusted_score) * 100.0, 1)
        semantics = "calibrated_below_tail"
        trust_state = "sub_tail"
        note = None

        if adjusted_score >= 0.60:
            candidate_band = max((th for th in (0.80, 0.75, 0.70, 0.60) if adjusted_score >= th), default=0.60)
            raw_contract = (score_contract.get("raw_model_contract") or {}) if isinstance(score_contract, dict) else {}
            raw_validated = raw_contract.get("validated_thresholds") or []
            if not validated_thresholds:
                display_score = min(display_score, unvalidated_cap)
                semantics = "ranking_only"
                trust_state = "unvalidated_tail"
                if raw_validated:
                    note = f"adjusted-live tail unvalidated; raw-model validated through {max(raw_validated):.2f}"
                else:
                    note = "no validated >=0.60 tail; treat as ranking score"
            elif round(candidate_band, 2) in validated_set:
                semantics = "validated_tail_probability"
                trust_state = f"validated_tail_{int(round(candidate_band * 100))}"
            else:
                semantics = "tail_caution"
                trust_state = "above_validated_tail"
                if highest_validated is not None:
                    note = f"tail validated only through {float(highest_validated):.2f}"

        return {
            "display_score": round(float(display_score), 4),
            "ranking_score": round(float(adjusted_score), 4),
            "opportunity_score": opportunity_score,
            "probability_semantics": semantics,
            "tail_trust_state": trust_state,
            "tail_validated_threshold": highest_validated,
            "tail_trust_note": note,
        }

    def _build_preview_scores(
        self,
        *,
        feature_rows: Dict[str, dict],
        guardrails: Dict[str, dict],
        diags: Dict[str, dict],
        market_regime,
        btc_regime: str,
        bundle: ModelBundle | None,
        score_contract: dict,
        candidate_stage: str,
    ) -> List[dict]:
        if not self.config.rolling_candidates_enabled:
            return []
        if market_regime is None or market_regime.state == "pending":
            return []
        if len(feature_rows) < max(1, int(self.config.rolling_candidates_min_feature_rows)):
            return []

        preview_cap = max(1, int(self.config.rolling_candidates_max_names))
        preview_pool = max(preview_cap * 3, min(int(self.config.stage1_max_candidates), max(preview_cap * 3, 12)))
        candidate_symbols = stage1_rank(feature_rows, guardrails, preview_pool, btc_regime=btc_regime)
        sector_leader_rets = self._compute_sector_leader_rets(feature_rows) if candidate_stage.startswith("stage2") else {}
        model_meta = self.state.model_metadata.get("pt2") or {}
        active_model_hash = str(model_meta.get("model_fingerprint") or "untrained")
        live_pipeline_mode = str(getattr(self.config, "live_pipeline_mode", "raw_threshold") or "raw_threshold").strip().lower()
        if live_pipeline_mode not in {"full", "raw_threshold"}:
            live_pipeline_mode = "raw_threshold"
        live_raw_threshold = effective_live_raw_threshold(self.config)
        effective_market_regime_actionability, effective_market_regime_note = self._effective_market_regime_actionability(market_regime, live_pipeline_mode=live_pipeline_mode)
        is_panic = btc_regime == "BTC panic"
        threshold_boost = self.config.panic_threshold_boost if is_panic else 0.0

        rows: List[dict] = []
        for symbol in candidate_symbols:
            row = feature_rows[symbol]
            diag = diags[symbol]
            guard = guardrails[symbol]
            if guard.get("block_code") == "BLOCKED":
                continue

            if bundle is not None:
                prob_model = float(bundle.predict_proba(pd.DataFrame([{k: row[k] for k in FEATURE_COLUMNS}]))[0])
                pt2_label = "trained"
            else:
                prob_model = heuristic_probability(row, guard, guardrail_cap=self.config.tail_unvalidated_cap)
                pt2_label = "heuristic"

            sector_penalty = self._get_sector_penalty(symbol, sector_leader_rets) if candidate_stage.startswith("stage2") else 0.0
            liquidity_bucket = self._liquidity_bucket(diag)
            liquidity_tier = classify_liquidity_tier(symbol, diag, self.config)
            if live_pipeline_mode == "raw_threshold":
                adjustment_detail = {
                    "guardrail_capped": False,
                    "panic_penalty": 0.0,
                    "sector_penalty": 0.0,
                    "binance_gap_penalty": 0.0,
                    "binance_lead_penalty": 0.0,
                    "total_penalty": 0.0,
                }
                prob_pre_regime = float(max(0.0, min(1.0, prob_model)))
                prob_adjusted = prob_pre_regime
                live_policy = {"threshold": live_raw_threshold, "factor": 1.0, "cap": 1.0, "suppress": False}
            else:
                prob_adjusted, adjustment_detail = apply_live_post_model_adjustments(
                    prob_model, row, guard, is_panic=is_panic, threshold_boost=threshold_boost, sector_penalty=sector_penalty, guardrail_cap=self.config.tail_unvalidated_cap
                )

                prob_pre_regime = float(prob_adjusted)
                live_policy = live_policy_for(market_regime.state, liquidity_tier, self.config)
                if live_policy["suppress"]:
                    continue
                cooldown_blocked = False
                if market_regime.suppress_new_entries:
                    if liquidity_tier == "tier3":
                        cooldown_blocked = True
                    elif liquidity_tier == "tier2" and liquidity_bucket != "high":
                        cooldown_blocked = True
                if cooldown_blocked:
                    continue

                prob_adjusted = max(0.0, prob_adjusted * float(live_policy["factor"]))
                prob_adjusted = min(prob_adjusted, float(live_policy["cap"]))
                if prob_adjusted < float(live_policy["threshold"]):
                    continue

            trust = self._apply_tail_trust(prob_adjusted, score_contract)
            actionability = self._assess_actionability(
                adjusted_score=prob_adjusted,
                trust=trust,
                score_contract=score_contract,
                market_regime=market_regime,
                liquidity_tier=liquidity_tier,
                guard=guard,
                objective_band=self._score_band(live_score=trust["display_score"], score_contract=score_contract),
                effective_market_regime_actionability=effective_market_regime_actionability,
            )
            score_band = self._score_band(live_score=trust["display_score"], score_contract=score_contract)
            pre_policy_band = self._score_band(live_score=prob_pre_regime, score_contract=score_contract)
            reasons = self._build_reasons(row, guard)
            reasons.append(f"market regime: {market_regime.state}")
            reasons.append("rolling preview")
            reasons.append(actionability["actionability_reason"])
            if trust.get("tail_trust_note"):
                reasons.append(str(trust["tail_trust_note"]))

            rows.append({
                "symbol": symbol,
                "price": round(float(diag["latest_price"]), 8),
                "pt2": pt2_label,
                "prob_2_model": round(float(prob_model), 4),
                "prob_2_pre_regime": round(float(prob_pre_regime), 4),
                "pre_policy_score": round(float(prob_pre_regime), 4),
                "prob_2_rank": round(float(prob_adjusted), 4),
                "prob_2": trust["display_score"],
                "live_score": trust["display_score"],
                "validated_floor": score_band["validated_floor"],
                "near_validated_floor": score_band["near_validated_floor"],
                "pre_policy_validated_floor": pre_policy_band["validated_floor"],
                "pre_policy_near_validated_floor": pre_policy_band["near_validated_floor"],
                "pre_policy_distance_to_validated": pre_policy_band["distance_to_validated"],
                "pre_policy_distance_to_validated_pct_points": pre_policy_band["distance_to_validated_pct_points"],
                "pre_policy_score_band": pre_policy_band["score_band"],
                "pre_policy_score_band_label": pre_policy_band["score_band_label"],
                "distance_to_validated": score_band["distance_to_validated"],
                "distance_to_validated_pct_points": score_band["distance_to_validated_pct_points"],
                "score_band": score_band["score_band"],
                "score_band_label": score_band["score_band_label"],
                "monitor_priority": score_band["monitor_priority"],
                "objective_score_band": score_band.get("objective_score_band"),
                "objective_score_band_label": score_band.get("objective_score_band_label"),
                "objective_monitor_priority": score_band.get("objective_monitor_priority"),
                "objective_quality_reference_rate": score_band.get("objective_quality_reference_rate"),
                "objective_quality_reference_source": score_band.get("objective_quality_reference_source"),
                "objective_distance_to_confirmed_shortlist": score_band.get("objective_distance_to_confirmed_shortlist"),
                "objective_distance_to_confirmed_shortlist_pct_points": score_band.get("objective_distance_to_confirmed_shortlist_pct_points"),
                "objective_confirmed_shortlist_floor": score_band.get("objective_confirmed_shortlist_floor"),
                "objective_strong_edge_floor": score_band.get("objective_strong_edge_floor"),
                "objective_priority_edge_floor": score_band.get("objective_priority_edge_floor"),
                "objective_elite_edge_floor": score_band.get("objective_elite_edge_floor"),
                "opportunity_score": trust["opportunity_score"],
                "probability_semantics": trust["probability_semantics"],
                "tail_trust_state": trust["tail_trust_state"],
                "tail_validated_threshold": trust["tail_validated_threshold"],
                "tail_trust_note": trust["tail_trust_note"],
                "risk": guard["risk"],
                "risk_reasons": guard["risk_reasons"],
                "downside_risk": guard["downside_risk"],
                "uncertainty": guard["uncertainty"],
                "uncertainty_reasons": guard["uncertainty_reasons"],
                "btc_regime_context": btc_regime,
                "market_regime_state": market_regime.state,
                "headline_risk": market_regime.headline_risk,
                "market_regime_score": market_regime.score,
                "market_regime_reasons": list(market_regime.reasons),
                "market_regime_actionability": effective_market_regime_actionability if effective_market_regime_actionability is not None else market_regime.actionability_state,
                "market_regime_actionability_raw": market_regime.actionability_state,
                "market_regime_actionability_note": effective_market_regime_note,
                "cooldown_active": bool(market_regime.cooldown_active),
                "cooldown_until_utc": market_regime.cooldown_until_utc,
                "liquidity_tier": liquidity_tier,
                "actionability_tier": actionability["actionability_tier"],
                "actionability_rank": actionability["actionability_rank"],
                "actionability_type": actionability["actionability_type"],
                "actionability_evidence": actionability["actionability_evidence"],
                "actionability_reason": actionability["actionability_reason"],
                "policy_constraint_reason": actionability["policy_constraint_reason"],
                "contract_truth_state": actionability["contract_truth_state"],
                "contract_truth_semantics": actionability["contract_truth_semantics"],
                "temporal_tail_state": actionability["temporal_tail_state"],
                "temporal_tail_semantics": actionability["temporal_tail_semantics"],
                "live_threshold": round(float(live_policy["threshold"]), 4),
                "base_live_threshold": round(float(live_policy["threshold"]), 4),
                "threshold_policy_mode": "raw_threshold" if live_pipeline_mode == "raw_threshold" else "absolute",
                "threshold_math": ({"mode": "raw_threshold", "raw_threshold": round(float(live_raw_threshold), 4)} if live_pipeline_mode == "raw_threshold" else self._policy_math(factor=float(live_policy["factor"]), cap=float(live_policy["cap"]), threshold=float(live_policy["threshold"]))),
                "regime_haircut_factor": round(float(live_policy["factor"]), 4),
                "regime_cap": round(float(live_policy["cap"]), 4),
                **self._visibility_band(live_score=trust["display_score"], live_threshold=float(live_policy["threshold"])),
                "operator_override_active": bool(market_regime.override_state),
                "reasons": reasons,
                "block_code": guard["block_code"],
                "model_hash": active_model_hash,
                "app_version": APP_VERSION,
                "live_pipeline_mode": live_pipeline_mode,
                "was_capped": bool(adjustment_detail.get("guardrail_capped", False)),
                "panic_penalty": round(float(adjustment_detail.get("panic_penalty", 0.0) or 0.0), 4),
                "sector_penalty": round(float(adjustment_detail.get("sector_penalty", 0.0) or 0.0), 4),
                "binance_gap_penalty": round(float(adjustment_detail.get("binance_gap_penalty", 0.0) or 0.0), 4),
                "binance_lead_penalty": round(float(adjustment_detail.get("binance_lead_penalty", 0.0) or 0.0), 4),
                "post_model_total_penalty": round(float(adjustment_detail.get("total_penalty", 0.0) or 0.0), 4),
                "activity_bucket": self._activity_bucket(row),
                "liquidity_bucket": liquidity_bucket,
                "cohort_member": True,
                "cohort_mode": "rolling_preview",
                "candidate_stage": candidate_stage,
                "provisional": True,
                "deep_confirmed": candidate_stage.startswith("stage2"),
            })

        action_order = {"action_ready": 3, "selective": 2, "watchlist": 1}
        rows.sort(key=lambda x: (action_order.get(str(x.get("actionability_tier") or "watchlist"), 1), float(x.get("prob_2_rank", 0.0)), float(x.get("opportunity_score", 0.0))), reverse=True)
        rows = rows[:preview_cap]
        for idx, row in enumerate(rows, start=1):
            row["score_rank"] = idx
        return rows

    def _publish_running_snapshot(
        self,
        *,
        universe,
        requested: int,
        btc_regime: str,
        returned_light: int = 0,
        stage1_feature_ready: int = 0,
        stage1_candidates: int = 0,
        stage2_requested: int = 0,
        stage2_returned: int = 0,
        stage2_feature_ready: int = 0,
        skip_reasons: Counter | None = None,
        market_regime=None,
        blocked_stage1: int = 0,
        dropped_stage1_by_rank: int = 0,
        btc_ctx: dict | None = None,
        eth_ctx: dict | None = None,
        feature_rows: dict | None = None,
        publish_meta: dict | None = None,
        preview_scores: List[dict] | None = None,
        score_contract: dict | None = None,
    ) -> None:
        preview_scores = preview_scores if preview_scores is not None else []
        stage_summary = self._score_stage_summary(preview_scores)
        coverage = self._coverage_snapshot(
            universe,
            requested=requested,
            returned_light=returned_light,
            stage1_feature_ready=stage1_feature_ready,
            stage2_requested=stage2_requested,
            stage2_returned=stage2_returned,
            stage2_feature_ready=stage2_feature_ready,
            symbols_scored=stage_summary["visible_rows"],
            skip_reasons=skip_reasons,
            blocked_stage1=blocked_stage1,
            dropped_stage1_by_rank=dropped_stage1_by_rank,
        )
        coverage["symbols_previewed_count"] = stage_summary["preview_rows"]
        coverage["symbols_deep_confirmed_count"] = stage_summary["deep_confirmed_rows"]
        coverage["symbols_stage1_preview_count"] = stage_summary["stage1_preview_rows"]
        coverage["symbols_stage2_partial_count"] = stage_summary["stage2_partial_rows"]
        coverage["symbols_stage2_final_count"] = stage_summary["stage2_final_rows"]
        previous_regime = self.state.get_status().get("market_regime") or {}
        regime_snapshot = market_regime
        if regime_snapshot is None:
            readiness = assess_market_regime_readiness(self.config, btc_ctx, eth_ctx, feature_rows or {})
            regime_snapshot = pending_market_regime(
                previous=previous_regime,
                reason="regime evaluation pending",
                readiness=readiness,
                publish_meta=publish_meta,
            )
        if regime_snapshot.state == "pending" and int(getattr(regime_snapshot, "partial_publish_successes", 0) or 0) > 0:
            regime_snapshot.regime_publish_warning = True
            regime_snapshot.regime_publish_warning_reason = regime_snapshot.regime_publish_warning_reason or "computed_or_counted_but_not_applied"
        if preview_scores is not None:
            self.state.set_scores(preview_scores)
        tail_counts = {
            "above_0_60": sum(1 for s in preview_scores if float(s.get("prob_2", 0.0) or 0.0) >= 0.60),
            "above_0_70": sum(1 for s in preview_scores if float(s.get("prob_2", 0.0) or 0.0) >= 0.70),
            "above_0_75": sum(1 for s in preview_scores if float(s.get("prob_2", 0.0) or 0.0) >= 0.75),
            "above_0_80": sum(1 for s in preview_scores if float(s.get("prob_2", 0.0) or 0.0) >= 0.80),
        }
        stage_counts = {
            "stage1_candidates": stage1_candidates,
            "visible_rows": stage_summary["visible_rows"],
            "preview_rows": stage_summary["preview_rows"],
            "deep_confirmed_rows": stage_summary["deep_confirmed_rows"],
            "stage1_preview_rows": stage_summary["stage1_preview_rows"],
            "stage2_partial_rows": stage_summary["stage2_partial_rows"],
            "stage2_final_rows": stage_summary["stage2_final_rows"],
            "stage2_scored": stage_summary["stage2_scored"],
        }
        self.state.update_status(
            universe=universe.diagnostics,
            coverage=coverage,
            guardrails=self._guardrail_snapshot(blocked_stage1=blocked_stage1),
            stage_counts=stage_counts,
            tail_counts=tail_counts,
            regime_context=btc_regime,
            market_regime=regime_snapshot.as_dict(),
            score_contract=score_contract or {},
            score_contract_live=(score_contract or {}).get("live_contract", score_contract or {}),
            score_contract_raw=(score_contract or {}).get("raw_model_contract", {}),
            score_reconciliation=(score_contract or {}).get("score_reconciliation", {}),
            live_universe_mode_requested=self.config.live_universe_mode,
            live_universe_mode_effective=universe.diagnostics.get("selection_mode", "dynamic"),
            scan_result_scope="partial",
            scan_result_generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def start_background_threads(self) -> None:
        logger.info("startup_complete")
        if self.review_packs is not None:
            self.review_packs.start_background_threads()
        recovered_followup = self._recover_persisted_followup(startup=True)
        if self.config.startup_scan and not recovered_followup:
            self.trigger_scan("startup")
        if not self.config.disable_scheduler:
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="scan-scheduler")
            self._scheduler_thread.start()

    def stop_background_threads(self) -> None:
        self._stop.set()
        self._cancel_followup_scan(reason="shutdown")
        if self.review_packs is not None:
            self.review_packs.stop_background_threads()

    def trigger_scan(self, trigger: str) -> bool:
        if self.state.scan_lock.locked():
            logger.info("scan_trigger_ignored trigger=%s reason=already_running", trigger)
            return False
        t = threading.Thread(target=self._run_scan, args=(trigger,), daemon=True, name=f"scan-{trigger}")
        self._current_thread = t
        t.start()
        return True

    def _scheduler_loop(self) -> None:
        while not self._stop.wait(self.config.scan_interval_minutes * 60):
            if self._recover_persisted_followup(startup=False):
                continue
            self.trigger_scan("scheduler")

    def _run_scan(self, trigger: str) -> None:
        if not self.state.scan_lock.acquire(blocking=False):
            return
        try:
            logger.info("scan_start trigger=%s", trigger)
            if trigger not in {"cooldown_followup", "cooldown_followup_confirmation"}:
                self._cancel_followup_scan(reason="superseded_by_scan")
            self.state.scan_started(f"scan started by {trigger}")
            # v2.6.1: resolve any pending paper-trade outcomes before scanning
            if self.paper_trade is not None:
                try:
                    resolved = self.paper_trade.resolve_pending()
                    if resolved:
                        logger.info("paper_trade_resolved_before_scan count=%d", resolved)
                except Exception as exc:
                    logger.warning("paper_trade_resolve_failed error=%s", exc)
            if self.review_packs is not None:
                try:
                    resolved_reviews = self._resolve_review_runs_catchup(phase="before_scan")
                    if resolved_reviews:
                        logger.info("review_runs_resolved_before_scan count=%d", resolved_reviews)
                except Exception as exc:
                    logger.warning("review_runs_resolve_before_scan_failed error=%s", exc)
            artifacts = self._build_scan(trigger=trigger)
            self.state.set_score_views(artifacts.scores, artifacts.informational_rows)
            self.state.set_coverage(artifacts.coverage)
            self.state.update_status(**artifacts.status_updates)
            self.state.scan_finished("scan complete", phase="complete")
            self._maybe_schedule_followup_scan(trigger=trigger)
            # v2.6.1: log predictions via PaperTradeService
            if self.paper_trade is not None and self.config.paper_trade_log_enabled:
                self.paper_trade.log_predictions(artifacts.scores)
            elif self.config.paper_trade_log_enabled:
                self._log_paper_trade(artifacts.scores)
            if self.review_packs is not None:
                current_status = self.state.get_status()
                try:
                    self.review_packs.record_scan(
                        status=current_status,
                        visible_rows=artifacts.scores,
                        suppressed_rows=artifacts.suppressed_rows,
                        informational_rows=artifacts.informational_rows,
                        overflow_rows=artifacts.informational_overflow_rows,
                        trigger_source=trigger,
                    )
                except Exception as exc:
                    logger.warning("review_pack_record_failed trigger=%s error=%s", trigger, exc)
                if self.model_output_distribution_service is not None:
                    try:
                        self.model_output_distribution_service.record_scan(
                            status=current_status,
                            visible_rows=artifacts.scores,
                            suppressed_rows=artifacts.suppressed_rows,
                            informational_rows=artifacts.informational_rows,
                            overflow_rows=artifacts.informational_overflow_rows,
                            trigger_source=trigger,
                        )
                    except Exception as exc:
                        logger.warning("model_output_distribution_record_failed trigger=%s error=%s", trigger, exc)
                try:
                    resolved_reviews = self._resolve_review_runs_catchup(phase="after_scan")
                    if resolved_reviews:
                        logger.info("review_runs_resolved_after_scan count=%d", resolved_reviews)
                except Exception as exc:
                    logger.warning("review_runs_resolve_after_scan_failed error=%s", exc)
            if self.shadow_selection_comparison_service is not None:
                try:
                    current_status = self.state.get_status()
                    shadow_summary = self.shadow_selection_comparison_service.record_scan(
                        status=current_status,
                        live_rows=artifacts.scores,
                        trimmed_visible_rows=artifacts.trimmed_visible_rows,
                        effective_max=int(((current_status.get("decision_summary") or {}).get("effective_max") or self.config.stage2_max_names) or self.config.stage2_max_names),
                        tracked_priority_symbols=[str(r.get("symbol") or "") for r in artifacts.scores if bool(r.get("tracked_followup_symbol"))],
                        trigger_source=trigger,
                    )
                    self.state.update_status(shadow_selection_comparison=shadow_summary)
                except Exception as exc:
                    logger.warning("shadow_selection_comparison_record_failed trigger=%s error=%s", trigger, exc)
            if self.semantics_shadow_comparison_service is not None:
                try:
                    current_status = self.state.get_status()
                    semantics_shadow_summary = self.semantics_shadow_comparison_service.record_scan(
                        status=current_status,
                        live_rows=artifacts.scores,
                        trimmed_visible_rows=artifacts.trimmed_visible_rows,
                        suppressed_rows=artifacts.suppressed_rows,
                        trigger_source=trigger,
                    )
                    self.state.update_status(semantics_shadow_comparison=semantics_shadow_summary)
                except Exception as exc:
                    logger.warning("semantics_shadow_comparison_record_failed trigger=%s error=%s", trigger, exc)
            logger.info("scan_complete trigger=%s scored=%s", trigger, len(artifacts.scores))
        except Exception as exc:
            logger.exception("scan_failed trigger=%s error=%s", trigger, exc)
            self.state.scan_finished(f"scan failed: {type(exc).__name__}: {exc}", phase="failed")
        finally:
            self.state.scan_lock.release()

    def _build_scan(self, trigger: str = "manual") -> ScanArtifacts:
        health = self.client.health().as_dict()
        self.state.update_status(data_source=health)
        self.state.scan_progress("health_check", "coinbase health checked")

        logger.info("discover_universe_start")
        products = self.client.list_products()
        currencies = self.client.list_currencies()
        volume_map = self.client.get_volume_summary()
        locked_symbols = self._locked_live_cohort()
        universe = UniverseBuilder(self.config).build(
            products,
            currencies,
            volume_map,
            locked_symbols=locked_symbols,
            selection_label=self._selection_label(locked_symbols),
        )
        logger.info("discover_universe_end eligible=%s selected=%s", len(universe.eligible), len(universe.selected_for_fetch))
        self.state.scan_progress("discover_universe", universe.diagnostics["summary"], symbols_total=len(universe.selected_for_fetch))

        requested = len(universe.selected_for_fetch)
        score_contract = self._score_contract()
        live_candidate_override = load_active_live_candidate_override(self.config.model_dir)
        active_model_path = str((live_candidate_override.get('model_bundle_path_override') or self.config.model_path_pt2))
        bundle = ModelBundle.load(active_model_path)
        override_source = str((live_candidate_override or {}).get('source') or '')
        live_candidate_proof_status = {
            'active': override_source == 'live_candidate_proof',
            'proof_session_id': live_candidate_override.get('proof_session_id') if override_source == 'live_candidate_proof' else None,
            'model_bundle_path_override': live_candidate_override.get('model_bundle_path_override') if override_source == 'live_candidate_proof' else None,
            'model_bundle_label_override': live_candidate_override.get('model_bundle_label_override') if override_source == 'live_candidate_proof' else None,
            'stage1_selection_mode_override': live_candidate_override.get('stage1_selection_mode_override') if override_source == 'live_candidate_proof' else None,
            'stage1_max_candidates_override': live_candidate_override.get('stage1_max_candidates_override') if override_source == 'live_candidate_proof' else None,
            'activated_at_utc': live_candidate_override.get('activated_at_utc') if override_source == 'live_candidate_proof' else None,
            'expires_at_utc': live_candidate_override.get('expires_at_utc') if override_source == 'live_candidate_proof' else None,
            'state_scope_key': live_candidate_override.get('state_scope_key') if override_source == 'live_candidate_proof' else None,
        }
        live_utility_model_proof_status = {
            'active': override_source == 'utility_model_proof',
            'proof_session_id': live_candidate_override.get('proof_session_id') if override_source == 'utility_model_proof' else None,
            'model_bundle_path_override': live_candidate_override.get('model_bundle_path_override') if override_source == 'utility_model_proof' else None,
            'model_bundle_label_override': live_candidate_override.get('model_bundle_label_override') if override_source == 'utility_model_proof' else None,
            'activated_at_utc': live_candidate_override.get('activated_at_utc') if override_source == 'utility_model_proof' else None,
            'expires_at_utc': live_candidate_override.get('expires_at_utc') if override_source == 'utility_model_proof' else None,
            'state_scope_key': live_candidate_override.get('state_scope_key') if override_source == 'utility_model_proof' else None,
        }
        live_candidate_adoption_status = {
            'active': override_source == 'live_candidate_adoption',
            'adoption_session_id': live_candidate_override.get('adoption_session_id') if override_source == 'live_candidate_adoption' else None,
            'model_bundle_path_override': live_candidate_override.get('model_bundle_path_override') if override_source == 'live_candidate_adoption' else None,
            'model_bundle_label_override': live_candidate_override.get('model_bundle_label_override') if override_source == 'live_candidate_adoption' else None,
            'stage1_selection_mode_override': live_candidate_override.get('stage1_selection_mode_override') if override_source == 'live_candidate_adoption' else None,
            'stage1_max_candidates_override': live_candidate_override.get('stage1_max_candidates_override') if override_source == 'live_candidate_adoption' else None,
            'live_raw_threshold_override': live_candidate_override.get('live_raw_threshold_override') if override_source == 'live_candidate_adoption' else None,
            'adopted_at_utc': live_candidate_override.get('adopted_at_utc') if override_source == 'live_candidate_adoption' else None,
            'state_scope_key': live_candidate_override.get('state_scope_key') if override_source == 'live_candidate_adoption' else None,
        }
        live_utility_tuning_adoption_status = {
            'active': override_source == 'utility_tuning_adoption',
            'adoption_session_id': live_candidate_override.get('adoption_session_id') if override_source == 'utility_tuning_adoption' else None,
            'adopted_at_utc': live_candidate_override.get('adopted_at_utc') if override_source == 'utility_tuning_adoption' else None,
            'state_scope_key': live_candidate_override.get('state_scope_key') if override_source == 'utility_tuning_adoption' else None,
            'utility_selection_engine_label': live_candidate_override.get('utility_selection_engine_label') if override_source == 'utility_tuning_adoption' else None,
            'utility_expected_edge_weight': live_candidate_override.get('utility_expected_edge_weight') if override_source == 'utility_tuning_adoption' else None,
            'utility_confidence_weight': live_candidate_override.get('utility_confidence_weight') if override_source == 'utility_tuning_adoption' else None,
            'utility_probability_weight': live_candidate_override.get('utility_probability_weight') if override_source == 'utility_tuning_adoption' else None,
            'utility_shortlist_target_max_names': live_candidate_override.get('utility_shortlist_target_max_names') if override_source == 'utility_tuning_adoption' else None,
            'utility_shortlist_score_floor': live_candidate_override.get('utility_shortlist_score_floor') if override_source == 'utility_tuning_adoption' else None,
            'utility_shortlist_score_dropoff': live_candidate_override.get('utility_shortlist_score_dropoff') if override_source == 'utility_tuning_adoption' else None,
            'utility_confidence_floor': live_candidate_override.get('utility_confidence_floor') if override_source == 'utility_tuning_adoption' else None,
            'utility_tier3_max_frac': live_candidate_override.get('utility_tier3_max_frac') if override_source == 'utility_tuning_adoption' else None,
        }
        live_utility_model_adoption_status = {
            'active': override_source == 'utility_model_adoption',
            'adoption_session_id': live_candidate_override.get('adoption_session_id') if override_source == 'utility_model_adoption' else None,
            'adopted_at_utc': live_candidate_override.get('adopted_at_utc') if override_source == 'utility_model_adoption' else None,
            'state_scope_key': live_candidate_override.get('state_scope_key') if override_source == 'utility_model_adoption' else None,
            'utility_selection_engine_label': live_candidate_override.get('utility_selection_engine_label') if override_source == 'utility_model_adoption' else None,
            'utility_expected_edge_weight': live_candidate_override.get('utility_expected_edge_weight') if override_source == 'utility_model_adoption' else None,
            'utility_confidence_weight': live_candidate_override.get('utility_confidence_weight') if override_source == 'utility_model_adoption' else None,
            'utility_probability_weight': live_candidate_override.get('utility_probability_weight') if override_source == 'utility_model_adoption' else None,
            'utility_shortlist_target_max_names': live_candidate_override.get('utility_shortlist_target_max_names') if override_source == 'utility_model_adoption' else None,
            'utility_shortlist_score_floor': live_candidate_override.get('utility_shortlist_score_floor') if override_source == 'utility_model_adoption' else None,
            'utility_shortlist_score_dropoff': live_candidate_override.get('utility_shortlist_score_dropoff') if override_source == 'utility_model_adoption' else None,
            'utility_confidence_floor': live_candidate_override.get('utility_confidence_floor') if override_source == 'utility_model_adoption' else None,
            'utility_tier3_max_frac': live_candidate_override.get('utility_tier3_max_frac') if override_source == 'utility_model_adoption' else None,
        }
        self.state.update_status(
            live_candidate_proof=live_candidate_proof_status,
            live_utility_model_proof=live_utility_model_proof_status,
            live_candidate_adoption=live_candidate_adoption_status,
            live_utility_tuning_adoption=live_utility_tuning_adoption_status,
            live_utility_model_adoption=live_utility_model_adoption_status,
        )
        prior_blocked_context = self._active_followup_context()
        tracked_followup_symbols = list(prior_blocked_context.get("tracked_symbols") or [])
        followup_reserve_meta = {
            "triggered": False,
            "requested_symbols": tracked_followup_symbols,
            "eligible_symbols": [],
            "missing_symbols": [],
            "blocked_symbols": [],
            "already_present_symbols": [],
            "injected_symbols": [],
            "reserve_count": max(0, int(getattr(self.config, "cooldown_followup_stage1_reserve_count", 5) or 5)),
        }
        stage1_input_rows: Dict[str, dict] = {}
        stage1_guardrails: Dict[str, dict] = {}
        stage1_diags: Dict[str, dict] = {}
        stage2_seed_products: Dict[str, dict] = {p["id"]: p for p in universe.selected_for_fetch}
        skip_reasons = Counter()
        returned_light = 0
        stage1_feature_ready = 0
        partial_regime_publish_min = max(1, int(self.config.market_regime_partial_min_feature_rows))
        partial_regime_publish_every = max(1, int(self.config.market_regime_partial_publish_every))
        next_partial_regime_at = partial_regime_publish_min
        next_stage1_snapshot_at = min(max(1, int(self.config.rolling_candidates_publish_every)), requested) if requested else 0

        btc_light_ctx, eth_light_ctx = self._fetch_context(self.config.stage1_light_calendar_5m_bars)
        # v2.6.0: determine BTC regime early for regime-aware gating
        btc_regime = self._btc_regime_label(btc_light_ctx)
        partial_publish_attempts = 0
        partial_publish_successes = 0
        partial_publish_failures = 0
        last_partial_publish_attempt_utc = None
        last_partial_publish_error = None
        partial_regime_published = False
        current_market_regime = pending_market_regime(
            previous=self.state.get_status().get("market_regime") or {},
            reason="regime evaluation pending",
            readiness=assess_market_regime_readiness(self.config, btc_light_ctx, eth_light_ctx, stage1_input_rows),
            publish_meta={
                "partial_publish_attempts": partial_publish_attempts,
                "partial_publish_successes": partial_publish_successes,
                "partial_publish_failures": partial_publish_failures,
                "last_partial_publish_attempt_utc": last_partial_publish_attempt_utc,
                "last_partial_publish_error": last_partial_publish_error,
            },
        )
        self._publish_running_snapshot(
            universe=universe,
            requested=requested,
            btc_regime=btc_regime,
            btc_ctx=btc_light_ctx,
            eth_ctx=eth_light_ctx,
            feature_rows=stage1_input_rows,
            market_regime=current_market_regime,
            publish_meta={
                "partial_publish_attempts": partial_publish_attempts,
                "partial_publish_successes": partial_publish_successes,
                "partial_publish_failures": partial_publish_failures,
                "last_partial_publish_attempt_utc": last_partial_publish_attempt_utc,
                "last_partial_publish_error": last_partial_publish_error,
            },
            returned_light=returned_light,
            stage1_feature_ready=stage1_feature_ready,
            preview_scores=[],
            score_contract=score_contract,
        )

        # v2.6.0: fetch BTC candle frame for btc_corr feature
        btc_candle_df = None
        try:
            btc_candle_df = self.client.get_candles("BTC-USD", self.config.stage1_light_calendar_5m_bars)
        except Exception:
            pass

        logger.info(
            "market_fetch_start requested=%s calendar_bars=%s feature_bars=%s btc_regime=%s",
            requested,
            self.config.stage1_light_calendar_5m_bars,
            self.config.stage1_light_feature_5m_bars,
            btc_regime,
        )
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {
                pool.submit(self._fetch_symbol_frame, product["id"], self.config.stage1_light_calendar_5m_bars): product
                for product in universe.selected_for_fetch
            }
            for idx, fut in enumerate(as_completed(futures), start=1):
                product = futures[fut]
                symbol = product["id"]
                logger.info("per_symbol_fetch_start stage=light symbol=%s", symbol)
                try:
                    df = fut.result(timeout=self.config.http_timeout_seconds + 5)
                    returned_light += 1
                    feature_df = self._prepare_feature_frame(df, self.config.stage1_light_feature_5m_bars)
                    history_bars = len(feature_df)
                    observed_bars = int(feature_df.attrs.get("observed_bars", int((feature_df["volume"] > 0).sum()) if not feature_df.empty else 0))
                    if history_bars < self.config.stage1_min_history_5m_bars:
                        skip_reasons["stage1_insufficient_history"] += 1
                        self.state.scan_progress("market_fetch", f"skip {symbol}: stage1 insufficient history", inc_done=True, inc_skipped=True)
                        continue
                    if observed_bars < self.config.stage1_min_observed_5m_bars:
                        skip_reasons["stage1_insufficient_observed"] += 1
                        self.state.scan_progress("market_fetch", f"skip {symbol}: stage1 sparse prints", inc_done=True, inc_skipped=True)
                        continue
                    feat = compute_live_features(symbol, feature_df, btc_ctx=btc_light_ctx, eth_ctx=eth_light_ctx, btc_df=btc_candle_df)
                    diag = {**feat.diagnostics, "rolling_dollar_volume": float(product.get("rolling_dollar_volume", 0.0))}
                    guard = compute_guardrails(symbol, feat.feature_row, diag, feat.block_reason, self.state.model_metadata.get("pt2"), self.config)
                    stage1_input_rows[symbol] = feat.feature_row
                    stage1_diags[symbol] = diag
                    stage1_guardrails[symbol] = guard
                    stage1_feature_ready += 1
                    self.state.scan_progress("market_fetch", f"fetched {symbol} ({idx}/{requested})", inc_done=True)
                except Exception as exc:
                    skip_reasons["stage1_fetch_failed"] += 1
                    self.state.scan_progress("market_fetch", f"failed {symbol}: {exc}", inc_done=True, inc_failed=True)
                    logger.warning("per_symbol_fetch_fail stage=light symbol=%s error=%s", symbol, exc)

                readiness = assess_market_regime_readiness(self.config, btc_light_ctx, eth_light_ctx, stage1_input_rows)
                should_publish_partial = False
                if requested and idx >= next_stage1_snapshot_at:
                    should_publish_partial = True
                    next_stage1_snapshot_at += max(1, int(self.config.rolling_candidates_publish_every))
                if readiness.get("partial_regime_eligible") and (not partial_regime_published or stage1_feature_ready >= next_partial_regime_at):
                    partial_publish_attempts += 1
                    last_partial_publish_attempt_utc = datetime.now(timezone.utc).isoformat()
                    try:
                        computed_market_regime = build_market_regime(
                            self.config,
                            btc_light_ctx,
                            eth_light_ctx,
                            stage1_input_rows,
                            previous=(current_market_regime.as_dict() if current_market_regime is not None else (self.state.get_status().get("market_regime") or {})),
                            readiness=readiness,
                            publish_meta={
                                "partial_publish_attempts": partial_publish_attempts,
                                "partial_publish_successes": partial_publish_successes,
                                "partial_publish_failures": partial_publish_failures,
                                "last_partial_publish_attempt_utc": last_partial_publish_attempt_utc,
                                "last_partial_publish_error": None,
                            },
                        )
                        partial_publish_successes += 1
                        partial_regime_published = True
                        last_partial_publish_error = None
                        current_market_regime = mark_market_regime_applied(
                            computed_market_regime,
                            previous=(current_market_regime.as_dict() if current_market_regime is not None else (self.state.get_status().get("market_regime") or {})),
                            applied_at_utc=last_partial_publish_attempt_utc,
                        )
                        current_market_regime = self._with_publish_meta(
                            current_market_regime,
                            attempts=partial_publish_attempts,
                            successes=partial_publish_successes,
                            failures=partial_publish_failures,
                            last_attempt=last_partial_publish_attempt_utc,
                            last_error=None,
                        )
                    except Exception as exc:
                        partial_publish_failures += 1
                        last_partial_publish_error = f"{type(exc).__name__}: {exc}"
                        current_market_regime = pending_market_regime(
                            previous=(current_market_regime.as_dict() if current_market_regime is not None else (self.state.get_status().get("market_regime") or {})),
                            reason="regime eligible but unpublished",
                            readiness=readiness,
                            publish_meta={
                                "partial_publish_attempts": partial_publish_attempts,
                                "partial_publish_successes": partial_publish_successes,
                                "partial_publish_failures": partial_publish_failures,
                                "last_partial_publish_attempt_utc": last_partial_publish_attempt_utc,
                                "last_partial_publish_error": last_partial_publish_error,
                                "regime_publish_warning": True,
                                "regime_publish_warning_reason": "eligible_but_unpublished",
                            },
                        )
                    should_publish_partial = True
                    next_partial_regime_at = max(next_partial_regime_at + partial_regime_publish_every, stage1_feature_ready + partial_regime_publish_every)
                elif should_publish_partial:
                    if readiness.get("partial_regime_eligible") and partial_regime_published and current_market_regime is not None and current_market_regime.state != "pending":
                        current_market_regime.readiness = readiness
                        current_market_regime.partial_regime_eligible = bool(readiness.get("partial_regime_eligible"))
                        current_market_regime = self._with_publish_meta(
                            current_market_regime,
                            attempts=partial_publish_attempts,
                            successes=partial_publish_successes,
                            failures=partial_publish_failures,
                            last_attempt=last_partial_publish_attempt_utc,
                            last_error=last_partial_publish_error,
                        )
                    else:
                        warning = bool(readiness.get("partial_regime_eligible") and not partial_regime_published)
                        current_market_regime = pending_market_regime(
                            previous=(current_market_regime.as_dict() if current_market_regime is not None else (self.state.get_status().get("market_regime") or {})),
                            reason="regime evaluation pending" if not warning else "regime eligible but unpublished",
                            readiness=readiness,
                            publish_meta={
                                "partial_publish_attempts": partial_publish_attempts,
                                "partial_publish_successes": partial_publish_successes,
                                "partial_publish_failures": partial_publish_failures,
                                "last_partial_publish_attempt_utc": last_partial_publish_attempt_utc,
                                "last_partial_publish_error": last_partial_publish_error,
                                "regime_publish_warning": warning,
                                "regime_publish_warning_reason": "eligible_but_unpublished" if warning else None,
                            },
                        )
                if should_publish_partial:
                    blocked_stage1_partial = sum(1 for g in stage1_guardrails.values() if g["block_code"] == "BLOCKED")
                    preview_scores = self._build_preview_scores(
                        feature_rows=stage1_input_rows,
                        guardrails=stage1_guardrails,
                        diags=stage1_diags,
                        market_regime=current_market_regime,
                        btc_regime=btc_regime,
                        bundle=bundle,
                        score_contract=score_contract,
                        candidate_stage="stage1_preview",
                    )
                    self._publish_running_snapshot(
                        universe=universe,
                        requested=requested,
                        btc_regime=btc_regime,
                        btc_ctx=btc_light_ctx,
                        eth_ctx=eth_light_ctx,
                        feature_rows=stage1_input_rows,
                        returned_light=returned_light,
                        stage1_feature_ready=stage1_feature_ready,
                        stage1_candidates=len(preview_scores),
                        skip_reasons=skip_reasons,
                        market_regime=current_market_regime,
                        blocked_stage1=blocked_stage1_partial,
                        publish_meta={
                            "partial_publish_attempts": partial_publish_attempts,
                            "partial_publish_successes": partial_publish_successes,
                            "partial_publish_failures": partial_publish_failures,
                            "last_partial_publish_attempt_utc": last_partial_publish_attempt_utc,
                            "last_partial_publish_error": last_partial_publish_error,
                        },
                        preview_scores=preview_scores,
                        score_contract=score_contract,
                    )
                time.sleep(self.config.request_pause_seconds)

        final_readiness = assess_market_regime_readiness(self.config, btc_light_ctx, eth_light_ctx, stage1_input_rows)
        market_regime = build_market_regime(
            self.config,
            btc_light_ctx,
            eth_light_ctx,
            stage1_input_rows,
            previous=(current_market_regime.as_dict() if current_market_regime is not None else (self.state.get_status().get("market_regime") or {})),
            readiness=final_readiness,
            publish_meta={
                "partial_publish_attempts": partial_publish_attempts,
                "partial_publish_successes": partial_publish_successes,
                "partial_publish_failures": partial_publish_failures,
                "last_partial_publish_attempt_utc": last_partial_publish_attempt_utc,
                "last_partial_publish_error": last_partial_publish_error,
            },
        )
        configured_stage1_max_candidates = int(live_candidate_override.get('stage1_max_candidates_override') or self.config.stage1_max_candidates)
        regime_candidate_cap = configured_stage1_max_candidates
        if market_regime.state == "amber":
            regime_candidate_cap = min(regime_candidate_cap, max(12, int(configured_stage1_max_candidates * 0.75)))
        elif market_regime.state == "red":
            regime_candidate_cap = min(regime_candidate_cap, max(8, int(configured_stage1_max_candidates * 0.45)))

        blocked_stage1 = sum(1 for g in stage1_guardrails.values() if g["block_code"] == "BLOCKED")
        opportunity_scores = {}
        selection_mode = str(live_candidate_override.get('stage1_selection_mode_override') or getattr(self.config, "stage1_selection_mode", "primary_only") or "primary_only")
        if getattr(self, "stage1_opportunity", None) is not None:
            try:
                opportunity_scores = self.stage1_opportunity.score_feature_rows(stage1_input_rows, stage1_guardrails)
            except Exception:
                opportunity_scores = {}
        stage1_candidates, stage1_selection_meta = stage1_select(
            stage1_input_rows,
            stage1_guardrails,
            regime_candidate_cap,
            btc_regime=btc_regime,
            selection_mode=selection_mode,
            recall_reserve_frac=float(getattr(self.config, "stage1_recall_reserve_frac", 0.25) or 0.25),
            recall_reserve_min=int(getattr(self.config, "stage1_recall_reserve_min", 6) or 6),
            recall_reserve_max=int(getattr(self.config, "stage1_recall_reserve_max", 12) or 12),
            promotion_overflow_window=int(getattr(self.config, "stage1_promotion_overflow_window", 20) or 20),
            opportunity_model_scores=opportunity_scores,
        )
        if tracked_followup_symbols:
            stage1_candidates, followup_reserve_meta = self._apply_followup_candidate_reserve(
                stage1_candidates,
                stage1_input_rows,
                stage1_guardrails,
                tracked_followup_symbols,
            )
            for symbol in list((followup_reserve_meta or {}).get("injected_symbols") or []):
                stage1_selection_meta.setdefault("selected_sources", {})[str(symbol)] = "followup_reserve"
        stage1_dropped_by_rank = max(0, len(stage1_input_rows) - blocked_stage1 - len(stage1_candidates))
        self.state.scan_progress(
            "stage1_select",
            f"stage1 shortlisted {len(stage1_candidates)} of {len(stage1_input_rows)} (btc={btc_regime}; market={market_regime.state})",
            symbols_total=requested,
        )
        stage1_preview_scores = self._build_preview_scores(
            feature_rows=stage1_input_rows,
            guardrails=stage1_guardrails,
            diags=stage1_diags,
            market_regime=market_regime,
            btc_regime=btc_regime,
            bundle=bundle,
            score_contract=score_contract,
            candidate_stage="stage1_preview",
        )
        self._publish_running_snapshot(
            universe=universe,
            requested=requested,
            btc_regime=btc_regime,
            returned_light=returned_light,
            stage1_feature_ready=stage1_feature_ready,
            stage1_candidates=len(stage1_candidates),
            stage2_requested=len(stage1_candidates),
            skip_reasons=skip_reasons,
            market_regime=market_regime,
            blocked_stage1=blocked_stage1,
            dropped_stage1_by_rank=stage1_dropped_by_rank,
            preview_scores=stage1_preview_scores,
            score_contract=score_contract,
        )

        btc_deep_ctx, eth_deep_ctx = self._fetch_context(self.config.stage2_lookback_5m_bars)
        # v2.6.0: fetch deep BTC frame for btc_corr
        btc_deep_df = None
        try:
            btc_deep_df = self.client.get_candles("BTC-USD", self.config.stage2_lookback_5m_bars)
        except Exception:
            pass

        # v2.6.1: fetch cross-exchange signals from Binance for Stage 2 candidates
        cross_exchange_signals: Dict[str, dict] = {}
        if not self.config.demo_mode:
            try:
                cross_exchange_signals = self.binance.get_cross_exchange_signals(stage1_candidates)
                if cross_exchange_signals:
                    logger.info("binance_cross_exchange_fetched symbols=%d", len(cross_exchange_signals))
            except Exception as exc:
                logger.warning("binance_cross_exchange_failed error=%s", exc)

        stage2_rows: Dict[str, dict] = {}
        stage2_guardrails: Dict[str, dict] = {}
        stage2_diags: Dict[str, dict] = {}
        stage2_feature_ready = 0
        stage2_returned = 0
        next_stage2_snapshot_at = 10
        logger.info("stage2_fetch_start requested=%s lookback_bars=%s", len(stage1_candidates), self.config.stage2_lookback_5m_bars)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(self._fetch_symbol_frame, symbol, self.config.stage2_lookback_5m_bars): symbol for symbol in stage1_candidates}
            for fut in as_completed(futures):
                symbol = futures[fut]
                try:
                    df = fut.result(timeout=self.config.http_timeout_seconds + 5)
                    stage2_returned += 1
                    history_bars = len(df)
                    observed_bars = int(df.attrs.get("observed_bars", int((df["volume"] > 0).sum()) if not df.empty else 0))
                    if history_bars < self.config.stage2_min_history_5m_bars:
                        skip_reasons["stage2_insufficient_history"] += 1
                        self.state.scan_progress("stage2_fetch", f"skip {symbol}: deep history", inc_skipped=True)
                        continue
                    if observed_bars < self.config.stage2_min_observed_5m_bars:
                        skip_reasons["stage2_insufficient_observed"] += 1
                        self.state.scan_progress("stage2_fetch", f"skip {symbol}: deep sparse prints", inc_skipped=True)
                        continue
                    # v2.6.1: pass cross-exchange signals for this symbol
                    cx = cross_exchange_signals.get(symbol)
                    feat = compute_live_features(symbol, df, btc_ctx=btc_deep_ctx, eth_ctx=eth_deep_ctx, btc_df=btc_deep_df, cross_exchange=cx)
                    product = stage2_seed_products.get(symbol, {})
                    diag = {**feat.diagnostics, "rolling_dollar_volume": float(product.get("rolling_dollar_volume", 0.0))}
                    guard = compute_guardrails(symbol, feat.feature_row, diag, feat.block_reason, self.state.model_metadata.get("pt2"), self.config)
                    stage2_rows[symbol] = feat.feature_row
                    stage2_diags[symbol] = diag
                    stage2_guardrails[symbol] = guard
                    stage2_feature_ready += 1
                    self.state.scan_progress("stage2_fetch", f"deep fetched {symbol}")
                except Exception as exc:
                    skip_reasons["stage2_fetch_failed"] += 1
                    self.state.scan_progress("stage2_fetch", f"failed {symbol}: {exc}", inc_failed=True)
                    logger.warning("per_symbol_fetch_fail stage=deep symbol=%s error=%s", symbol, exc)
                if stage2_returned >= next_stage2_snapshot_at or stage2_feature_ready >= next_stage2_snapshot_at:
                    stage2_preview_scores = self._build_preview_scores(
                        feature_rows=stage2_rows if stage2_rows else stage1_input_rows,
                        guardrails=stage2_guardrails if stage2_rows else stage1_guardrails,
                        diags=stage2_diags if stage2_rows else stage1_diags,
                        market_regime=market_regime,
                        btc_regime=btc_regime,
                        bundle=bundle,
                        score_contract=score_contract,
                        candidate_stage="stage2_partial" if stage2_rows else "stage1_preview",
                    )
                    self._publish_running_snapshot(
                        universe=universe,
                        requested=requested,
                        btc_regime=btc_regime,
                        btc_ctx=btc_light_ctx,
                        eth_ctx=eth_light_ctx,
                        feature_rows=stage1_input_rows,
                        returned_light=returned_light,
                        stage1_feature_ready=stage1_feature_ready,
                        stage1_candidates=len(stage1_candidates),
                        stage2_requested=len(stage1_candidates),
                        stage2_returned=stage2_returned,
                        stage2_feature_ready=stage2_feature_ready,
                        skip_reasons=skip_reasons,
                        market_regime=market_regime,
                        blocked_stage1=blocked_stage1,
                        dropped_stage1_by_rank=stage1_dropped_by_rank,
                        publish_meta={
                            "partial_publish_attempts": partial_publish_attempts,
                            "partial_publish_successes": partial_publish_successes,
                            "partial_publish_failures": partial_publish_failures,
                            "last_partial_publish_attempt_utc": last_partial_publish_attempt_utc,
                            "last_partial_publish_error": last_partial_publish_error,
                        },
                        preview_scores=stage2_preview_scores,
                        score_contract=score_contract,
                    )
                    next_stage2_snapshot_at += 10
                time.sleep(self.config.request_pause_seconds)

        self.state.scan_progress("evaluate", f"evaluating {len(stage2_rows)} deep candidates")

        is_panic = btc_regime == "BTC panic"
        threshold_boost = self.config.panic_threshold_boost if is_panic else 0.0

        # v2.6.1: compute sector leader returns for cross-asset contagion check
        sector_leader_rets = self._compute_sector_leader_rets(stage2_rows)

        raw_scores = []
        suppressed_rows = []
        threshold_candidates = []
        threshold_plan = {"mode": "absolute", "tiers": {}}
        dropped_stage2_blocked = 0
        capped = 0
        event_risk = 0
        suppressed_regime = 0
        suppressed_threshold = 0
        suppressed_cooldown = 0
        model_meta = self.state.model_metadata.get("pt2") or {}
        active_model_hash = str(model_meta.get("model_fingerprint") or "untrained")
        live_pipeline_mode = str(getattr(self.config, "live_pipeline_mode", "raw_threshold") or "raw_threshold").strip().lower()
        if live_pipeline_mode not in {"full", "raw_threshold"}:
            live_pipeline_mode = "raw_threshold"
        live_raw_threshold = effective_live_raw_threshold(self.config)
        effective_market_regime_actionability, effective_market_regime_note = self._effective_market_regime_actionability(market_regime, live_pipeline_mode=live_pipeline_mode)
        for symbol, row in stage2_rows.items():
            guard = stage2_guardrails[symbol]
            if guard["block_code"] == "BLOCKED":
                dropped_stage2_blocked += 1
                skip_reasons["stage2_blocked"] += 1
                continue
            if guard["block_code"] == "EVENT_RISK":
                event_risk += 1
            if guard["capped"]:
                capped += 1

            if bundle is not None:
                prob_model = float(bundle.predict_proba(pd.DataFrame([{k: row[k] for k in FEATURE_COLUMNS}]))[0])
                pt2_label = "trained"
            else:
                prob_model = heuristic_probability(row, guard, guardrail_cap=self.config.tail_unvalidated_cap)
                pt2_label = "heuristic"

            sector_penalty = self._get_sector_penalty(symbol, sector_leader_rets)
            liquidity_bucket = self._liquidity_bucket(stage2_diags[symbol])
            liquidity_tier = classify_liquidity_tier(symbol, stage2_diags[symbol], self.config)
            if live_pipeline_mode == "raw_threshold":
                adjustment_detail = {
                    "guardrail_capped": False,
                    "panic_penalty": 0.0,
                    "sector_penalty": 0.0,
                    "binance_gap_penalty": 0.0,
                    "binance_lead_penalty": 0.0,
                    "total_penalty": 0.0,
                }
                prob_pre_regime = max(0.0, min(1.0, float(prob_model)))
                prob_adjusted = prob_pre_regime
                live_policy = {"threshold": live_raw_threshold, "factor": 1.0, "cap": 1.0, "suppress": False}
                suppress_reason = None
                cooldown_blocked = False
            else:
                prob_adjusted, adjustment_detail = apply_live_post_model_adjustments(
                    prob_model, row, guard, is_panic=is_panic, threshold_boost=threshold_boost, sector_penalty=sector_penalty, guardrail_cap=self.config.tail_unvalidated_cap
                )

                prob_pre_regime = prob_adjusted
                live_policy = live_policy_for(market_regime.state, liquidity_tier, self.config)
                suppress_reason = None
                if live_policy["suppress"]:
                    suppress_reason = "regime"

                cooldown_blocked = False
                if market_regime.suppress_new_entries:
                    if liquidity_tier == "tier3":
                        cooldown_blocked = True
                    elif liquidity_tier == "tier2" and liquidity_bucket != "high":
                        cooldown_blocked = True

                prob_adjusted = max(0.0, prob_adjusted * float(live_policy["factor"]))
                prob_adjusted = min(prob_adjusted, float(live_policy["cap"]))

            trust = self._apply_tail_trust(prob_adjusted, score_contract)
            actionability = self._assess_actionability(
                adjusted_score=prob_adjusted,
                trust=trust,
                score_contract=score_contract,
                market_regime=market_regime,
                liquidity_tier=liquidity_tier,
                guard=guard,
                objective_band=self._score_band(live_score=trust["display_score"], score_contract=score_contract),
                effective_market_regime_actionability=effective_market_regime_actionability,
            )
            score_band = self._score_band(live_score=trust["display_score"], score_contract=score_contract)
            pre_policy_band = self._score_band(live_score=prob_pre_regime, score_contract=score_contract)
            reasons = self._build_reasons(row, guard)
            reasons.append(f"market regime: {market_regime.state}")
            if float(live_policy["factor"]) < 0.999:
                reasons.append(f"event-risk haircut x{float(live_policy['factor']):.2f}")
            if float(live_policy["cap"]) < 0.99:
                reasons.append(f"live cap {float(live_policy['cap']):.2f}")
            if market_regime.cooldown_active:
                reasons.append("cooldown active")
            if market_regime.override_state:
                reasons.append("operator override")
            reasons.append(actionability["actionability_reason"])
            if trust.get("tail_trust_note"):
                reasons.append(str(trust["tail_trust_note"]))

            row_payload = {
                "symbol": symbol,
                "price": round(float(stage2_diags[symbol]["latest_price"]), 8),
                "pt2": pt2_label,
                "prob_2_model": round(float(prob_model), 4),
                "prob_2_pre_regime": round(float(prob_pre_regime), 4),
                "pre_policy_score": round(float(prob_pre_regime), 4),
                "prob_2_rank": round(float(prob_adjusted), 4),
                "prob_2": trust["display_score"],
                "live_score": trust["display_score"],
                "validated_floor": score_band["validated_floor"],
                "near_validated_floor": score_band["near_validated_floor"],
                "pre_policy_validated_floor": pre_policy_band["validated_floor"],
                "pre_policy_near_validated_floor": pre_policy_band["near_validated_floor"],
                "pre_policy_distance_to_validated": pre_policy_band["distance_to_validated"],
                "pre_policy_distance_to_validated_pct_points": pre_policy_band["distance_to_validated_pct_points"],
                "pre_policy_score_band": pre_policy_band["score_band"],
                "pre_policy_score_band_label": pre_policy_band["score_band_label"],
                "distance_to_validated": score_band["distance_to_validated"],
                "distance_to_validated_pct_points": score_band["distance_to_validated_pct_points"],
                "score_band": score_band["score_band"],
                "score_band_label": score_band["score_band_label"],
                "monitor_priority": score_band["monitor_priority"],
                "objective_score_band": score_band.get("objective_score_band"),
                "objective_score_band_label": score_band.get("objective_score_band_label"),
                "objective_monitor_priority": score_band.get("objective_monitor_priority"),
                "objective_quality_reference_rate": score_band.get("objective_quality_reference_rate"),
                "objective_quality_reference_source": score_band.get("objective_quality_reference_source"),
                "objective_distance_to_confirmed_shortlist": score_band.get("objective_distance_to_confirmed_shortlist"),
                "objective_distance_to_confirmed_shortlist_pct_points": score_band.get("objective_distance_to_confirmed_shortlist_pct_points"),
                "objective_confirmed_shortlist_floor": score_band.get("objective_confirmed_shortlist_floor"),
                "objective_strong_edge_floor": score_band.get("objective_strong_edge_floor"),
                "objective_priority_edge_floor": score_band.get("objective_priority_edge_floor"),
                "objective_elite_edge_floor": score_band.get("objective_elite_edge_floor"),
                "opportunity_score": trust["opportunity_score"],
                "probability_semantics": trust["probability_semantics"],
                "tail_trust_state": trust["tail_trust_state"],
                "tail_validated_threshold": trust["tail_validated_threshold"],
                "tail_trust_note": trust["tail_trust_note"],
                "risk": guard["risk"],
                "risk_reasons": guard["risk_reasons"],
                "downside_risk": guard["downside_risk"],
                "uncertainty": guard["uncertainty"],
                "uncertainty_reasons": guard["uncertainty_reasons"],
                "btc_regime_context": btc_regime,
                "market_regime_state": market_regime.state,
                "headline_risk": market_regime.headline_risk,
                "market_regime_score": market_regime.score,
                "market_regime_reasons": list(market_regime.reasons),
                "market_regime_actionability": effective_market_regime_actionability if effective_market_regime_actionability is not None else market_regime.actionability_state,
                "market_regime_actionability_raw": market_regime.actionability_state,
                "market_regime_actionability_note": effective_market_regime_note,
                "cooldown_active": bool(market_regime.cooldown_active),
                "cooldown_until_utc": market_regime.cooldown_until_utc,
                "liquidity_tier": liquidity_tier,
                "actionability_tier": actionability["actionability_tier"],
                "actionability_rank": actionability["actionability_rank"],
                "actionability_type": actionability["actionability_type"],
                "actionability_evidence": actionability["actionability_evidence"],
                "actionability_reason": actionability["actionability_reason"],
                "policy_constraint_reason": actionability["policy_constraint_reason"],
                "contract_truth_state": actionability["contract_truth_state"],
                "contract_truth_semantics": actionability["contract_truth_semantics"],
                "temporal_tail_state": actionability["temporal_tail_state"],
                "temporal_tail_semantics": actionability["temporal_tail_semantics"],
                "live_threshold": round(float(live_policy["threshold"]), 4),
                "base_live_threshold": round(float(live_policy["threshold"]), 4),
                "threshold_policy_mode": "raw_threshold" if live_pipeline_mode == "raw_threshold" else "absolute",
                "threshold_math": ({"mode": "raw_threshold", "raw_threshold": round(float(live_raw_threshold), 4)} if live_pipeline_mode == "raw_threshold" else self._policy_math(factor=float(live_policy["factor"]), cap=float(live_policy["cap"]), threshold=float(live_policy["threshold"]))),
                "regime_haircut_factor": round(float(live_policy["factor"]), 4),
                "regime_cap": round(float(live_policy["cap"]), 4),
                **self._visibility_band(live_score=trust["display_score"], live_threshold=float(live_policy["threshold"])),
                "operator_override_active": bool(market_regime.override_state),
                "reasons": reasons,
                "block_code": guard["block_code"],
                "model_hash": active_model_hash,
                "app_version": APP_VERSION,
                "was_capped": bool(adjustment_detail.get("guardrail_capped", False)),
                "panic_penalty": round(float(adjustment_detail.get("panic_penalty", 0.0) or 0.0), 4),
                "sector_penalty": round(float(adjustment_detail.get("sector_penalty", 0.0) or 0.0), 4),
                "binance_gap_penalty": round(float(adjustment_detail.get("binance_gap_penalty", 0.0) or 0.0), 4),
                "binance_lead_penalty": round(float(adjustment_detail.get("binance_lead_penalty", 0.0) or 0.0), 4),
                "post_model_total_penalty": round(float(adjustment_detail.get("total_penalty", 0.0) or 0.0), 4),
                "activity_bucket": self._activity_bucket(row),
                "liquidity_bucket": liquidity_bucket,
                "cohort_member": True,
                "cohort_mode": universe.diagnostics.get("selection_mode", "dynamic"),
                "candidate_stage": "stage2_final",
                "provisional": False,
                "deep_confirmed": True,
                "row_type": "candidate",
                "tracked_followup_symbol": bool(str(row.get("symbol") or "") in set(tracked_followup_symbols or [])),
                "suppression_reason": None,
                "suppression_reason_detail": None,
                "display_bucket": "candidate",
                "informational_only": False,
                "is_actionable_now": True,
            }

            if live_pipeline_mode != "raw_threshold" and suppress_reason == "regime":
                suppressed_regime += 1
                skip_reasons["stage2_regime_suppressed"] += 1
                row_payload["suppression_reason"] = "regime"
                row_payload["suppression_reason_detail"] = row_payload.get("policy_constraint_reason") or "blocked by live market regime policy"
                row_payload["display_bucket"] = "informational_suppressed"
                row_payload["informational_only"] = True
                row_payload["is_actionable_now"] = False
                row_payload["row_type"] = "suppressed"
                suppressed_rows.append(row_payload)
                continue
            if cooldown_blocked:
                suppressed_cooldown += 1
                skip_reasons["stage2_cooldown_suppressed"] += 1
                row_payload["suppression_reason"] = "cooldown"
                row_payload["suppression_reason_detail"] = row_payload.get("policy_constraint_reason") or "blocked by active cooldown"
                row_payload["display_bucket"] = "informational_suppressed"
                row_payload["informational_only"] = True
                row_payload["is_actionable_now"] = False
                row_payload["row_type"] = "suppressed"
                suppressed_rows.append(row_payload)
                continue
            threshold_candidates.append(row_payload)

        threshold_plan = self._build_threshold_plan(regime_state=market_regime.state, threshold_candidates=threshold_candidates) if live_pipeline_mode != "raw_threshold" else {}
        for row_payload in threshold_candidates:
            if live_pipeline_mode == "raw_threshold":
                effective_threshold = live_raw_threshold
                tier_plan = {"mode": "raw_threshold", "effective_math": {"raw_threshold": round(float(live_raw_threshold), 4)}}
            else:
                effective_threshold, tier_plan = self._effective_threshold_for_row(row_payload, threshold_plan)
            row_payload["base_live_threshold"] = round(float(row_payload.get("base_live_threshold", row_payload.get("live_threshold") or 0.0) or 0.0), 4)
            row_payload["live_threshold"] = round(float(effective_threshold), 4)
            row_payload["threshold_policy_mode"] = str(tier_plan.get("mode") or "absolute")
            row_payload["threshold_math"] = dict(tier_plan.get("effective_math") or row_payload.get("threshold_math") or {})
            row_payload.update(self._visibility_band(live_score=float(row_payload.get("live_score", row_payload.get("prob_2_rank") or 0.0) or 0.0), live_threshold=float(effective_threshold)))
            threshold_blocked = float(row_payload.get("prob_2_rank", 0.0) or 0.0) < float(effective_threshold)
            if threshold_blocked:
                suppressed_threshold += 1
                skip_reasons["stage2_threshold_suppressed"] += 1
                row_payload["suppression_reason"] = "threshold"
                row_payload["suppression_reason_detail"] = row_payload.get("policy_constraint_reason") or "below current live threshold"
                row_payload["display_bucket"] = "informational_suppressed"
                row_payload["informational_only"] = True
                row_payload["is_actionable_now"] = False
                row_payload["row_type"] = "suppressed"
                suppressed_rows.append(row_payload)
                continue
            row_payload["row_type"] = "visible"
            raw_scores.append(row_payload)

        rankable_rows = list(raw_scores) + list(suppressed_rows)
        rankable_rows.sort(key=self._informational_sort_key, reverse=True)
        for idx, score in enumerate(rankable_rows, start=1):
            score["candidate_rank_all"] = idx
            score["would_be_rank"] = idx
            score["pre_policy_rank"] = idx

        raw_scores = [row for row in rankable_rows if not bool(row.get("informational_only"))]
        if str(getattr(self.config, "live_selection_mode", "utility_constrained") or "utility_constrained").lower() == "utility_constrained":
            annotate_rows_for_utility(raw_scores, self.config)
        raw_scores.sort(key=self._row_sort_key, reverse=True)
        for idx, score in enumerate(raw_scores, start=1):
            score["score_rank"] = idx
            score["display_bucket"] = "actionable"
            score["informational_only"] = False
            score["is_actionable_now"] = True
            score["row_type"] = "visible"
            score["tracked_followup_visible"] = bool(score.get("tracked_followup_symbol"))

        effective_max = self.config.stage2_panic_max_names if is_panic else self.config.stage2_max_names
        if market_regime.state == "amber":
            effective_max = min(effective_max, max(6, int(self.config.stage2_max_names * 0.65)))
        elif market_regime.state == "red":
            effective_max = min(effective_max, max(2, int(self.config.stage2_max_names * 0.20)))
        scores, trimmed_visible_rows, shortlist_meta = self._limit_visible_shortlist(raw_scores, effective_max=effective_max, tracked_priority_symbols=tracked_followup_symbols)
        tracked_visible_rank = 0
        for idx, score in enumerate(scores, start=1):
            score["score_rank"] = idx
            score["tracked_followup_visible"] = bool(score.get("tracked_followup_symbol"))
            if score.get("tracked_followup_visible"):
                tracked_visible_rank += 1
                score["tracked_followup_visible_rank"] = tracked_visible_rank
        dropped_by_output_cap = len(trimmed_visible_rows)
        informational_rows: List[dict] = list(suppressed_rows)
        if self.config.informational_rankings_enabled and self.config.informational_include_display_trimmed and dropped_by_output_cap > 0:
            for score in trimmed_visible_rows:
                trimmed = dict(score)
                trimmed["suppression_reason"] = "display_trim"
                trimmed["suppression_reason_detail"] = (
                    "watchlist candidate trimmed to keep the visible shortlist focused"
                    if str(trimmed.get("actionability_tier") or "") == "watchlist"
                    else "ranked candidate trimmed by output cap"
                )
                trimmed["display_bucket"] = "informational_suppressed"
                trimmed["informational_only"] = True
                trimmed["is_actionable_now"] = False
                informational_rows.append(trimmed)
        informational_overflow_rows: List[dict] = []
        if self.config.informational_rankings_enabled:
            informational_rows.sort(key=self._informational_sort_key, reverse=True)
            informational_cap = max(1, int(self.config.informational_rankings_max_names))
            informational_overflow_rows = informational_rows[informational_cap:]
            informational_rows = informational_rows[:informational_cap]
            for idx, row in enumerate(informational_rows, start=1):
                row["informational_rank"] = idx
                row["pre_policy_rank"] = row.get("pre_policy_rank") or row.get("candidate_rank_all") or idx
                row["display_bucket"] = "informational_suppressed"
                row["informational_only"] = True
                row["is_actionable_now"] = False
                row["row_type"] = "informational"
                row["review_bucket"] = "informational_retained"
            base_rank = len(informational_rows)
            for idx, row in enumerate(informational_overflow_rows, start=1):
                row["informational_rank"] = base_rank + idx
                row["pre_policy_rank"] = row.get("pre_policy_rank") or row.get("candidate_rank_all") or (base_rank + idx)
                row["display_bucket"] = "informational_overflow"
                row["informational_only"] = True
                row["is_actionable_now"] = False
                row["row_type"] = "overflow"
                row["review_bucket"] = "informational_overflow"
                row.setdefault("suppression_reason", "display_trim")
                row.setdefault("suppression_reason_detail", "trimmed row preserved only in the review pack because the informational cap was reached")
        else:
            informational_rows = []

        tail_counts = {
            "above_0_60": sum(1 for s in scores if s["prob_2"] >= 0.60),
            "above_0_70": sum(1 for s in scores if s["prob_2"] >= 0.70),
            "above_0_75": sum(1 for s in scores if s["prob_2"] >= 0.75),
            "above_0_80": sum(1 for s in scores if s["prob_2"] >= 0.80),
        }
        total_blocked = blocked_stage1 + dropped_stage2_blocked
        guardrail_stats = {
            "blocked": total_blocked,
            "blocked_stage1": blocked_stage1,
            "blocked_stage2": dropped_stage2_blocked,
            "event_risk": event_risk,
            "probability_capped": capped,
            "capped": capped,
            "suppressed_regime": suppressed_regime,
            "suppressed_threshold": suppressed_threshold,
            "suppressed_cooldown": suppressed_cooldown,
        }
        stage_summary = self._score_stage_summary(scores)
        informational_summary = self._score_stage_summary(informational_rows)
        decision_summary = self._build_decision_summary(
            visible_rows=scores,
            score_contract=score_contract,
            market_regime=market_regime,
            hidden_watchlist_rows=dropped_by_output_cap,
            blocked_rows=suppressed_rows,
            effective_market_regime_actionability=effective_market_regime_actionability,
        )
        blocked_monitoring_context = self._build_blocked_monitoring_context(
            trigger=trigger,
            market_regime=market_regime,
            blocked_rows=suppressed_rows,
            decision_summary=decision_summary,
            effective_market_regime_actionability=effective_market_regime_actionability,
        )
        all_ranked_rows = self._unique_rows_by_symbol(list(scores) + list(suppressed_rows) + list(trimmed_visible_rows))
        threshold_experiment_review = self._build_threshold_experiment_review(
            final_rows=all_ranked_rows,
            current_threshold=effective_live_raw_threshold(self.config),
            experiment_threshold=0.28,
        )
        stage1_omission_audit = self._run_stage1_omission_audit(
            stage1_input_rows=stage1_input_rows,
            stage1_guardrails=stage1_guardrails,
            stage1_candidates=stage1_candidates,
            stage1_selection_meta=stage1_selection_meta,
            stage2_seed_products=stage2_seed_products,
            btc_ctx=btc_deep_ctx,
            eth_ctx=eth_deep_ctx,
            btc_df=btc_deep_df,
            market_regime=market_regime,
            btc_regime=btc_regime,
            bundle=bundle,
            score_contract=score_contract,
            sector_leader_rets=sector_leader_rets,
            final_rows=all_ranked_rows,
        )
        stage1_selection_repair_review = self._run_stage1_selection_repair_review(
            stage1_input_rows=stage1_input_rows,
            stage1_guardrails=stage1_guardrails,
            stage1_candidates=stage1_candidates,
            stage1_selection_meta=stage1_selection_meta,
            stage2_seed_products=stage2_seed_products,
            btc_ctx=btc_deep_ctx,
            eth_ctx=eth_deep_ctx,
            btc_df=btc_deep_df,
            market_regime=market_regime,
            btc_regime=btc_regime,
            bundle=bundle,
            score_contract=score_contract,
            sector_leader_rets=sector_leader_rets,
            final_rows=all_ranked_rows,
        )
        all_stage_summary = self._score_stage_summary(all_ranked_rows)
        followup_comparison = self._build_followup_comparison(
            trigger=trigger,
            prior_context=prior_blocked_context,
            current_rows=all_ranked_rows,
        )
        decision_summary = self._apply_followup_comparison_to_decision_summary(decision_summary, followup_comparison)
        active_campaign = self._active_cooldown_campaign()
        if active_campaign:
            decision_summary["cooldown_campaign_run_count"] = int(active_campaign.get("merged_from_runs") or 0)
            decision_summary["cooldown_campaign_unique_symbols"] = int(active_campaign.get("merged_unique_symbols") or 0)
        else:
            decision_summary.setdefault("cooldown_campaign_run_count", 0)
            decision_summary.setdefault("cooldown_campaign_unique_symbols", 0)
        coverage = {
            "universe_count": len(universe.eligible),
            "cohort_mode": universe.diagnostics.get("selection_mode", "dynamic"),
            "trained_cohort_requested_count": int(universe.diagnostics.get("trained_cohort_requested_count", 0) or 0),
            "trained_cohort_available_count": int(universe.diagnostics.get("trained_cohort_available_count", 0) or 0),
            "trained_cohort_missing_count": int(universe.diagnostics.get("trained_cohort_missing_count", 0) or 0),
            "symbols_requested_count": requested,
            "symbols_returned_with_bars_count": returned_light,
            "symbols_with_sufficient_bars_count": stage1_feature_ready,
            "symbols_scored_count": len(all_ranked_rows),
            "symbols_previewed_count": all_stage_summary["preview_rows"],
            "symbols_deep_confirmed_count": sum(1 for r in all_ranked_rows if bool(r.get("deep_confirmed"))),
            "symbols_stage1_preview_count": all_stage_summary["stage1_preview_rows"],
            "symbols_stage2_partial_count": all_stage_summary["stage2_partial_rows"],
            "symbols_stage2_final_count": sum(1 for r in all_ranked_rows if str(r.get("candidate_stage") or "") == "stage2_final"),
            "stage1_feature_ready_count": stage1_feature_ready,
            "stage2_fetch_requested_count": len(stage1_candidates),
            "stage2_fetch_returned_count": stage2_returned,
            "stage2_feature_ready_count": stage2_feature_ready,
            "dropped_stage1_insufficient_history": skip_reasons.get("stage1_insufficient_history", 0),
            "dropped_stage1_insufficient_observed": skip_reasons.get("stage1_insufficient_observed", 0),
            "dropped_stage1_fetch_failed": skip_reasons.get("stage1_fetch_failed", 0),
            "dropped_stage1_blocked": blocked_stage1,
            "dropped_stage1_by_rank": stage1_dropped_by_rank,
            "dropped_stage2_insufficient_history": skip_reasons.get("stage2_insufficient_history", 0),
            "dropped_stage2_insufficient_observed": skip_reasons.get("stage2_insufficient_observed", 0),
            "dropped_stage2_fetch_failed": skip_reasons.get("stage2_fetch_failed", 0),
            "dropped_stage2_blocked": dropped_stage2_blocked,
            "dropped_stage2_regime_suppressed": suppressed_regime,
            "dropped_stage2_threshold_suppressed": suppressed_threshold,
            "dropped_stage2_cooldown_suppressed": suppressed_cooldown,
            "dropped_stage2_display_trimmed": dropped_by_output_cap,
            "dropped_stage2_output_cap": dropped_by_output_cap,
            "top_skip_reasons": [{"reason": k, "count": int(v)} for k, v in skip_reasons.most_common(10)],
            "followup_reserved_symbols": len(followup_reserve_meta.get("injected_symbols") or []),
            "followup_reserved_existing_symbols": len(followup_reserve_meta.get("already_present_symbols") or []),
            "tracked_visible_promoted": int(shortlist_meta.get("tracked_visible_promoted") or 0),
            "tracked_visible_symbols": list(shortlist_meta.get("tracked_visible_symbols") or []),
            "selection_engine": str(shortlist_meta.get("selection_engine") or getattr(self.config, "live_selection_mode", "legacy")),
            "dynamic_shortlist_score_floor": shortlist_meta.get("dynamic_score_floor"),
            "utility_confidence_floor": shortlist_meta.get("confidence_floor"),
            "utility_visible_cap": shortlist_meta.get("visible_cap"),
        }
        health = self.client.health().as_dict()
        status_updates = {
            "data_source": health,
            "coverage": coverage,
            "profiles": {
                "training_profile_available": bool((self.state.model_metadata.get("pt2") or {}).get("feature_mean")),
                "activity_profile_available": True,
            },
            "guardrails": guardrail_stats,
            "stage_counts": {
                "stage1_candidates": len(stage1_candidates),
                "visible_rows": stage_summary["visible_rows"],
                "informational_rows": informational_summary["visible_rows"],
                "informational_regime_rows": informational_summary["informational_regime_rows"],
                "informational_cooldown_rows": informational_summary["informational_cooldown_rows"],
                "informational_threshold_rows": informational_summary["informational_threshold_rows"],
                "informational_display_trim_rows": informational_summary["informational_display_trim_rows"],
                "informational_overflow_rows": len(informational_overflow_rows),
                "preview_rows": all_stage_summary["preview_rows"],
                "deep_confirmed_rows": sum(1 for r in all_ranked_rows if bool(r.get("deep_confirmed"))),
                "stage1_preview_rows": all_stage_summary["stage1_preview_rows"],
                "stage2_partial_rows": sum(1 for r in all_ranked_rows if str(r.get("candidate_stage") or "") == "stage2_partial"),
                "stage2_final_rows": sum(1 for r in all_ranked_rows if str(r.get("candidate_stage") or "") == "stage2_final"),
                "stage2_scored": len(all_ranked_rows),
                "stage2_visible_after_trim": stage_summary["visible_rows"],
                "stage2_hidden_after_trim": max(0, len(all_ranked_rows) - stage_summary["visible_rows"]),
                "action_ready_rows": stage_summary["action_ready_rows"],
                "selective_rows": stage_summary["selective_rows"],
                "watchlist_rows": stage_summary["watchlist_rows"],
            },
            "actionability_summary": {
                "action_ready_rows": stage_summary["action_ready_rows"],
                "selective_rows": stage_summary["selective_rows"],
                "watchlist_rows": stage_summary["watchlist_rows"],
                "near_validated_rows": decision_summary["near_validated_rows"],
                "exploratory_rows": decision_summary["exploratory_rows"],
                "actionability_type": "advisory_heuristic",
                "tail_validation_state": score_contract.get("tail_validation_state"),
                "temporal_tail_state": score_contract.get("temporal_tail_state"),
                "temporal_tail_semantics": score_contract.get("temporal_tail_semantics"),
                "temporal_support_basis": score_contract.get("temporal_support_basis"),
                "market_regime_actionability": market_regime.actionability_state,
            },
            "suppression_summary": {
                "threshold_suppressed_rows": suppressed_threshold,
                "regime_suppressed_rows": suppressed_regime,
                "cooldown_suppressed_rows": suppressed_cooldown,
                "display_trimmed_rows": dropped_by_output_cap,
                "visible_rows": stage_summary["visible_rows"],
                "informational_rows": informational_summary["visible_rows"],
                "informational_regime_rows": informational_summary["informational_regime_rows"],
                "informational_cooldown_rows": informational_summary["informational_cooldown_rows"],
                "informational_threshold_rows": informational_summary["informational_threshold_rows"],
                "informational_display_trim_rows": informational_summary["informational_display_trim_rows"],
                "informational_overflow_rows": len(informational_overflow_rows),
                "action_ready_rows": stage_summary["action_ready_rows"],
                "selective_rows": stage_summary["selective_rows"],
                "watchlist_rows": stage_summary["watchlist_rows"],
            },
            "informational_rankings_summary": {
                "enabled": bool(self.config.informational_rankings_enabled),
                "rows": informational_summary["visible_rows"],
                "overflow_rows": len(informational_overflow_rows),
                "max_names": self.config.informational_rankings_max_names,
                "regime_rows": informational_summary["informational_regime_rows"],
                "cooldown_rows": informational_summary["informational_cooldown_rows"],
                "threshold_rows": informational_summary["informational_threshold_rows"],
                "display_trim_rows": informational_summary["informational_display_trim_rows"],
                "label": "Informational Rankings (Not Surfaced)" if live_pipeline_mode == "raw_threshold" else "Informational Rankings (Blocked by Live Policy)",
                "advisory_only": True,
            },
            "tail_counts": tail_counts,
            "model": {"pt2": build_model_status_summary(self.state.model_metadata.get("pt2") or {"trained": False, "path": self.config.model_path_pt2})},
            "regime_context": btc_regime,
            "market_regime": {
                **market_regime.as_dict(),
                "effective_actionability_state": effective_market_regime_actionability if effective_market_regime_actionability is not None else market_regime.actionability_state,
                "raw_actionability_state": market_regime.actionability_state,
                "effective_actionability_note": effective_market_regime_note,
            },
            "decision_summary": decision_summary,
            "blocked_monitoring_context": blocked_monitoring_context,
            "cooldown_campaign": self.state.get_status().get("cooldown_campaign") or self._empty_cooldown_campaign(),
            "followup_comparison": followup_comparison,
            "threshold_policy": {**threshold_plan, "dimension": "liquidity_tier", "note": "threshold tiers are liquidity tiers, not score bands"},
            "score_contract": score_contract,
            "score_contract_live": score_contract.get("live_contract", score_contract),
            "score_contract_raw": score_contract.get("raw_model_contract", {}),
            "score_reconciliation": score_contract.get("score_reconciliation", {}),
            "live_universe_mode_requested": self.config.live_universe_mode,
            "live_universe_mode_effective": universe.diagnostics.get("selection_mode", "dynamic"),
            "candidate_quality": self._candidate_quality_diagnostics(
                stage1_input_rows=stage1_input_rows,
                stage1_guardrails=stage1_guardrails,
                stage1_diags=stage1_diags,
                stage1_candidates=stage1_candidates,
                stage1_selection_meta=stage1_selection_meta,
                stage2_diags=stage2_diags,
                final_rows=all_ranked_rows,
            ),
            "stage1_omission_audit": stage1_omission_audit,
            "stage1_selection_repair_review": stage1_selection_repair_review,
            "threshold_experiment_review": threshold_experiment_review,
            "score_diagnostics": self._score_diagnostics(
                visible_rows=scores,
                suppressed_rows=suppressed_rows,
                informational_rows=informational_rows,
                informational_overflow_rows=informational_overflow_rows,
                score_contract=score_contract,
            ),
        }
        return ScanArtifacts(
            scores=scores,
            informational_rows=informational_rows,
            informational_overflow_rows=informational_overflow_rows,
            coverage=coverage,
            status_updates=status_updates,
            suppressed_rows=suppressed_rows,
            trimmed_visible_rows=trimmed_visible_rows,
        )

    def _prepare_feature_frame(self, df: pd.DataFrame, feature_bars: int) -> pd.DataFrame:
        feature_df = df.tail(feature_bars).copy().reset_index(drop=True)
        feature_df.attrs["observed_bars"] = int((feature_df["volume"] > 0).sum()) if not feature_df.empty else 0
        feature_df.attrs["raw_bars"] = int(df.attrs.get("raw_bars", len(df)))
        feature_df.attrs["filled_bars"] = int(len(feature_df) - feature_df.attrs["observed_bars"])
        return feature_df

    def _fetch_context(self, lookback_bars: int) -> Tuple[dict, dict]:
        btc_df = None
        eth_df = None
        try:
            btc_df = self.client.get_candles("BTC-USD", lookback_bars)
        except Exception as exc:
            logger.warning("context_fetch_failed symbol=BTC-USD error=%s", exc)
        try:
            eth_df = self.client.get_candles("ETH-USD", lookback_bars)
        except Exception as exc:
            logger.warning("context_fetch_failed symbol=ETH-USD error=%s", exc)
        return self._make_ctx(btc_df), self._make_ctx(eth_df)

    def _fetch_symbol_frame(self, symbol: str, lookback_bars: int) -> pd.DataFrame:
        return self.client.get_candles(symbol, lookback_bars)

    def _make_ctx(self, df: pd.DataFrame | None) -> dict:
        if df is None or df.empty:
            return {"ret_15m": 0.0, "ret_1h": 0.0, "ret_4h": 0.0, "ret_24h": 0.0, "rv_1h": 0.0, "rv_4h": 0.0, "rv_24h": 0.0, "rv_ratio_1h_24h": 1.0, "_ready": False}
        close = df["close"].astype(float)
        rets = close.pct_change().fillna(0.0)
        ret_15m = float(close.iloc[-1] / close.iloc[-4] - 1.0) if len(close) > 3 and close.iloc[-4] else 0.0
        ret_1h = float(close.iloc[-1] / close.iloc[-13] - 1.0) if len(close) > 12 and close.iloc[-13] else 0.0
        ret_4h = float(close.iloc[-1] / close.iloc[-49] - 1.0) if len(close) > 48 and close.iloc[-49] else 0.0
        ret_24h = float(close.iloc[-1] / close.iloc[-289] - 1.0) if len(close) > 288 and close.iloc[-289] else 0.0
        rv_1h = float(rets.tail(12).std() * (12 ** 0.5)) if len(rets) >= 12 else 0.0
        rv_4h = float(rets.tail(48).std() * (48 ** 0.5)) if len(rets) >= 48 else 0.0
        rv_24h = float(rets.tail(288).std() * (288 ** 0.5)) if len(rets) >= 288 else 0.0
        rv_ratio = float(rv_1h / rv_24h) if rv_24h > 0 else 1.0
        return {"ret_15m": ret_15m, "ret_1h": ret_1h, "ret_4h": ret_4h, "ret_24h": ret_24h, "rv_1h": rv_1h, "rv_4h": rv_4h, "rv_24h": rv_24h, "rv_ratio_1h_24h": rv_ratio}

    def _btc_regime_label(self, btc_ctx: dict) -> str:
        ret_1h = btc_ctx.get("ret_1h", 0.0)
        ret_24h = btc_ctx.get("ret_24h", 0.0)
        if ret_1h <= self.config.btc_panic_threshold:
            return "BTC panic"
        if ret_24h > 0.03 and ret_1h > 0:
            return "BTC strong"
        if ret_24h < -0.04:
            return "BTC weak"
        return "BTC mixed"

    def _activity_bucket(self, row: dict) -> str:
        density = float(row.get("nonzero_volume_rate_24h", 0.0) or 0.0)
        rvol = float(row.get("rvol_1h", 0.0) or 0.0)
        if density >= 0.75 and rvol >= 1.0:
            return "high"
        if density >= 0.45:
            return "medium"
        return "low"

    def _liquidity_bucket(self, diag: dict) -> str:
        dv = float(diag.get("rolling_dollar_volume", 0.0) or 0.0)
        if dv >= 20_000_000:
            return "high"
        if dv >= 2_000_000:
            return "medium"
        return "low"

    def _build_reasons(self, row: dict, guardrail: dict) -> List[str]:
        reasons: List[str] = []
        if row["ret_60m"] > 0.015:
            reasons.append("1h momentum")
        if row["ret_24h"] > 0.03:
            reasons.append("24h trend")
        if row["asset_vs_btc_1h"] > 0.01:
            reasons.append("outperforming BTC")
        if row["adx_proxy"] > 20:
            reasons.append("trend strength")
        if row["rvol_1h"] > 1.2:
            reasons.append("activity pickup")
        if row["dist_24h_high"] > -0.03:
            reasons.append("near local highs")
        if row.get("momentum_persistence_1h", 0.0) > 0.15:
            reasons.append("persistent momentum")
        if row.get("up_volume_ratio_1h", 0.5) > 0.65:
            reasons.append("buying pressure")
        if row.get("btc_corr_24h", 0.0) < 0.3:
            reasons.append("low BTC correlation")
        # v2.6.1 blindspot warnings
        if row.get("move_vs_atr_ratio", 0.0) > 5.0:
            reasons.append("extended move (>5x ATR)")
        if row.get("volume_concentration", 1.0) > 5.0:
            reasons.append("concentrated volume")
        if row.get("volume_acceleration", 1.0) < 0.5:
            reasons.append("fading volume")
        if row.get("btc_recovery_from_trough", 288) < 18 and row.get("btc_trough_depth", 0) > 0.03:
            reasons.append("possible dead cat bounce")
        if row.get("spread_cost_proxy", 0.0) > 0.06:
            reasons.append("high execution cost")
        # v2.6.1 cross-exchange signals
        if row.get("binance_price_gap", 0.0) < -0.005:
            reasons.append("Binance price lower (divergence)")
        if row.get("binance_lead_1h", 0.0) < -0.02:
            reasons.append("Binance 1h weaker")
        if row.get("binance_lead_15m", 0.0) > 0.01 and row.get("binance_price_gap", 0.0) > 0.003:
            reasons.append("Binance leading higher")
        if row["history_bars_ratio_24h"] < 0.75:
            reasons.append("short live history")
        if guardrail["capped"]:
            reasons.append("probability capped by risk")
        if not reasons:
            reasons.append("mixed setup")
        return reasons

    # ── v2.6.1 sector leader logic ──────────────────────────────────────

    # Known crypto sector groups. Leader is the first symbol.
    SECTOR_GROUPS = {
        "solana_eco": ["SOL-USD", "RAY-USD", "JTO-USD", "JUP-USD", "ORCA-USD", "BONK-USD", "WIF-USD", "PYTH-USD", "MSOL-USD"],
        "ethereum_l2": ["ETH-USD", "OP-USD", "ARB-USD", "MATIC-USD", "ZK-USD", "ZORA-USD"],
        "defi_blue": ["AAVE-USD", "UNI-USD", "COMP-USD", "MKR-USD", "SNX-USD", "CRV-USD"],
        "ai_tokens": ["FET-USD", "RENDER-USD", "TAO-USD", "NEAR-USD", "AKT-USD"],
        "memecoins": ["DOGE-USD", "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FARTCOIN-USD", "POPCAT-USD", "MOG-USD"],
        "l1_alts": ["ADA-USD", "AVAX-USD", "DOT-USD", "ATOM-USD", "SUI-USD", "APT-USD", "SEI-USD", "NEAR-USD", "ICP-USD"],
    }

    # Reverse lookup: symbol -> list of sectors it belongs to
    _SYMBOL_TO_SECTORS: Dict[str, List[str]] | None = None

    @classmethod
    def _get_symbol_sectors(cls) -> Dict[str, List[str]]:
        if cls._SYMBOL_TO_SECTORS is None:
            mapping: Dict[str, List[str]] = {}
            for sector, members in cls.SECTOR_GROUPS.items():
                for sym in members:
                    if sym not in mapping:
                        mapping[sym] = []
                    mapping[sym].append(sector)
            cls._SYMBOL_TO_SECTORS = mapping
        return cls._SYMBOL_TO_SECTORS

    def _compute_sector_leader_rets(self, stage2_rows: Dict[str, dict]) -> Dict[str, float]:
        """Compute the 1h return of each sector's leader from Stage 2 features."""
        leader_rets: Dict[str, float] = {}
        for sector, members in self.SECTOR_GROUPS.items():
            leader = members[0]  # first member is the leader
            if leader in stage2_rows:
                leader_rets[sector] = stage2_rows[leader].get("ret_60m", 0.0)
        return leader_rets

    def _get_sector_penalty(self, symbol: str, sector_leader_rets: Dict[str, float]) -> float:
        """If any of this symbol's sector leaders is down >3% in 1h, apply a penalty.

        The penalty is proportional to the leader's drawdown:
        leader at -3% → 0.03 penalty, leader at -6% → 0.06, etc.
        """
        sectors = self._get_symbol_sectors().get(symbol, [])
        if not sectors:
            return 0.0
        worst_penalty = 0.0
        for sector in sectors:
            leader_ret = sector_leader_rets.get(sector, 0.0)
            if leader_ret < -0.03:
                penalty = min(0.10, abs(leader_ret) - 0.01)  # starts at 0.02 for -3%, caps at 0.10
                worst_penalty = max(worst_penalty, penalty)
        return worst_penalty

    def _log_paper_trade(self, scores: List[dict]) -> None:
        """v2.6.0: append predictions to paper trade log for forward validation."""
        try:
            log_path = Path(self.config.paper_trade_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()
            with open(log_path, "a", encoding="utf-8") as f:
                for s in scores:
                    entry = {
                        "logged_at_utc": now,
                        "symbol": s["symbol"],
                        "price": s["price"],
                        "prob_2": s["prob_2"],
                        "prob_2_pre_regime": s.get("prob_2_pre_regime"),
                        "risk": s["risk"],
                        "block_code": s["block_code"],
                        "btc_regime": s["btc_regime_context"],
                        "market_regime_state": s.get("market_regime_state"),
                        "headline_risk": s.get("headline_risk"),
                        "liquidity_tier": s.get("liquidity_tier"),
                        "live_threshold": s.get("live_threshold"),
                        "regime_haircut_factor": s.get("regime_haircut_factor"),
                        "regime_cap": s.get("regime_cap"),
                        "cooldown_active": s.get("cooldown_active"),
                        "operator_override_active": s.get("operator_override_active"),
                        "pt2": s["pt2"],
                    }
                    f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.warning("paper_trade_log_failed error=%s", exc)
