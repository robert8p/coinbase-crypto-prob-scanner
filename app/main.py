from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .coinbase_client import CoinbaseClient
from .config import AppConfig
from .demo_data import STABLES
from .modeling import (
    ModelBundle,
    build_model_status_summary,
    compact_score_contract,
    compact_score_reconciliation,
    reconcile_runtime_metadata,
)
from .paper_trade import PaperTradeService
from .replay import HistoricalReplayService
from .stage1_opportunity import Stage1OpportunityService
from .model_audit import ModelAuditService
from .raw_score_baseline import RawScoreBaselineService
from .benchmark import BenchmarkLabService
from .historical_decision_lab import HistoricalDecisionLabService
from .utility_selection_lab import UtilitySelectionLabService
from .utility_policy_search_lab import UtilityPolicySearchLabService
from .shadow_selection_comparison import ShadowSelectionComparisonService
from .shadow_selection_outcome_review import ShadowSelectionOutcomeReviewService
from .utility_tuning_lab import UtilityTuningLabService
from .utility_model_lab import UtilityModelLabService
from .utility_model_proof import UtilityModelProofService
from .utility_model_proof_review import UtilityModelProofReviewService
from .utility_model_adoption import UtilityModelAdoptionService
from .utility_model_adoption_review import UtilityModelAdoptionReviewService
from .utility_tuning_proof import UtilityTuningProofService
from .utility_tuning_proof_review import UtilityTuningProofReviewService
from .utility_tuning_adoption import UtilityTuningAdoptionService
from .utility_tuning_adoption_review import UtilityTuningAdoptionReviewService
from .live_candidate_adoption_review import LiveCandidateAdoptionReviewService
from .misranking import MisrankingDiagnosticService
from .threshold_boundary import ThresholdBoundaryReviewService
from .cooldown_shortlist import CooldownShortlistReviewService
from .stage2_retrain_review import Stage2RetrainReviewService
from .fresh_retrain_audit import FreshRetrainAuditService
from .challenger_comparison import OfflineChallengerComparisonService
from .stage1_policy_lab import Stage1PolicyLabService
from .next_live_candidate_lab import NextLiveCandidateLabService
from .live_candidate_proof import LiveCandidateProofService
from .live_candidate_proof_review import LiveCandidateProofReviewService
from .live_candidate_adoption import LiveCandidateAdoptionService
from .live_candidate_adoption_review import LiveCandidateAdoptionReviewService
from .decision_checkpoint import DecisionCheckpointService
from .decision_branch_automation import DecisionBranchAutomationService
from .review_runs import ReviewPackService
from .scanner import ScannerService
from .state import AppState
from .training import TrainingService
from .evidence_automation import EvidenceAutomationService
from .model_output_distribution import ModelOutputDistributionService
from .utility_operator_automation import UtilityOperatorAutomationService
from .version import APP_VERSION
from .runtime_scope import initialize_runtime_scope

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

config = AppConfig()
initialize_runtime_scope(config.model_dir, app_version=APP_VERSION, force=True)
state = AppState(config)
client = CoinbaseClient(config)
paper_trade = PaperTradeService(config, client)
review_packs = ReviewPackService(config, client)
shadow_selection_comparison = ShadowSelectionComparisonService(config, review_packs=review_packs)
scanner = ScannerService(config, state, client, paper_trade=paper_trade, review_packs=review_packs, shadow_selection_comparison_service=shadow_selection_comparison)
trainer = TrainingService(config, state, client)
replay = HistoricalReplayService(config, client, scanner, review_packs)
stage1_opportunity = Stage1OpportunityService(config)
scanner.stage1_opportunity = stage1_opportunity
model_output_distribution = ModelOutputDistributionService(config)
scanner.model_output_distribution_service = model_output_distribution
model_audit = ModelAuditService(config)
raw_score_baseline = RawScoreBaselineService(config, replay)
benchmark_lab = BenchmarkLabService(config, replay, review_packs, model_audit)
misranking_diagnostic = MisrankingDiagnosticService(config, review_packs)
threshold_boundary_review = ThresholdBoundaryReviewService(config, review_packs)
cooldown_shortlist_review = CooldownShortlistReviewService(config, review_packs)
stage2_retrain_review = Stage2RetrainReviewService(config, review_packs, state)
historical_decision_lab = HistoricalDecisionLabService(config, replay, benchmark_lab, review_packs, stage2_retrain_review)
utility_selection_lab = UtilitySelectionLabService(config, replay, review_packs)
utility_policy_search_lab = UtilityPolicySearchLabService(config, replay, review_packs)
shadow_selection_outcome_review = ShadowSelectionOutcomeReviewService(config, review_packs, shadow_selection_comparison, utility_policy_search_lab)
utility_tuning_lab = UtilityTuningLabService(config, replay, review_packs)
utility_model_lab = UtilityModelLabService(config, client)
utility_model_proof = UtilityModelProofService(config, review_packs)
utility_model_proof_review = UtilityModelProofReviewService(config, review_packs)
utility_model_adoption = UtilityModelAdoptionService(config)
utility_model_adoption_review = UtilityModelAdoptionReviewService(config, review_packs)
utility_tuning_proof = UtilityTuningProofService(config, review_packs)
utility_tuning_proof_review = UtilityTuningProofReviewService(config, review_packs)
utility_tuning_adoption = UtilityTuningAdoptionService(config)
utility_tuning_adoption_review = UtilityTuningAdoptionReviewService(config, review_packs)
live_candidate_adoption_review = LiveCandidateAdoptionReviewService(config, review_packs)
stage1_policy_lab = Stage1PolicyLabService(config, replay, review_packs)
next_live_candidate_lab = NextLiveCandidateLabService(config, replay, review_packs)
live_candidate_proof = LiveCandidateProofService(config, review_packs)
live_candidate_proof_review = LiveCandidateProofReviewService(config, review_packs)
live_candidate_adoption = LiveCandidateAdoptionService(config)
live_candidate_adoption_review = LiveCandidateAdoptionReviewService(config, review_packs)
fresh_retrain_audit = FreshRetrainAuditService(config, state, client, review_packs)
offline_challenger_comparison = OfflineChallengerComparisonService(config, client, review_packs)
decision_checkpoint = DecisionCheckpointService(config, review_packs)
decision_branch_automation = DecisionBranchAutomationService(config, decision_checkpoint, fresh_retrain_audit_service=fresh_retrain_audit)
evidence_automation = EvidenceAutomationService(
    config,
    state,
    review_packs,
    decision_checkpoint,
    decision_branch_automation,
    misranking_diagnostic,
    threshold_boundary_review,
    cooldown_shortlist_review,
    stage2_retrain_review,
    trainer,
    model_output_distribution,
)
review_packs.post_evaluation_callback = evidence_automation.handle_post_maturity
utility_operator_automation = UtilityOperatorAutomationService(
    config,
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
)



def _dict_to_text(payload: dict) -> str:
    def _render(value, indent=0):
        prefix = " " * indent
        if isinstance(value, dict):
            lines = []
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    nested = _render(v, indent + 2)
                    if nested:
                        lines.append(nested)
                else:
                    shown = '-' if v is None else v
                    lines.append(f"{prefix}{k}: {shown}")
            return "\n".join(lines)
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    nested = _render(item, indent + 2)
                    if nested:
                        lines.append(nested)
                else:
                    shown = '-' if item is None else item
                    lines.append(f"{prefix}- {shown}")
            return "\n".join(lines)
        return f"{prefix}{'-' if value is None else value}"

    if not isinstance(payload, dict):
        return _render(payload)
    return _render(payload)
def _is_stablecoin_pair(symbol: str) -> bool:
    base = str(symbol).split("-", 1)[0].upper()
    return base in STABLES


def _reconcile_loaded_metadata(meta: dict) -> tuple[dict, dict, dict, dict]:
    repaired, bundle = reconcile_runtime_metadata(
        meta,
        existing_status=state.get_status(),
        min_count=config.tail_validation_min_count,
        min_wilson_lift=config.tail_validation_min_wilson_lift,
        min_precision_floor=config.tail_validation_min_precision_floor,
        unvalidated_tail_cap=config.tail_unvalidated_cap,
        ran_on_startup=True,
        ran_on_model_load=True,
        scanner_contract_source="recomputed_runtime_adjusted",
        threshold_suppression_contract_source="recomputed_runtime_adjusted",
    )
    return repaired, bundle["score_contract_raw"], bundle["score_contract_live"], bundle["score_reconciliation"]


def _refresh_runtime_contracts(*, reason: str = "api") -> dict:
    meta = (state.model_metadata or {}).get("pt2") or {}
    status = state.get_status()
    repaired, bundle = reconcile_runtime_metadata(
        meta,
        existing_status=status,
        min_count=config.tail_validation_min_count,
        min_wilson_lift=config.tail_validation_min_wilson_lift,
        min_precision_floor=config.tail_validation_min_precision_floor,
        unvalidated_tail_cap=config.tail_unvalidated_cap,
        scanner_contract_source="recomputed_runtime_adjusted",
        threshold_suppression_contract_source="recomputed_runtime_adjusted",
    )
    if repaired != meta:
        state.set_model_metadata(repaired)
    current_live = status.get("score_contract_live") or {}
    current_raw = status.get("score_contract_raw") or {}
    current_rec = status.get("score_reconciliation") or {}
    if current_live != bundle["score_contract_live"] or current_raw != bundle["score_contract_raw"] or current_rec != bundle["score_reconciliation"]:
        state.update_status(
            score_contract=bundle["score_contract"],
            score_contract_live=bundle["score_contract_live"],
            score_contract_raw=bundle["score_contract_raw"],
            score_reconciliation=bundle["score_reconciliation"],
        )
    refreshed = state.get_status()
    refreshed["score_contract"] = bundle["score_contract"]
    refreshed["score_contract_live"] = bundle["score_contract_live"]
    refreshed["score_contract_raw"] = bundle["score_contract_raw"]
    refreshed["score_reconciliation"] = bundle["score_reconciliation"]
    refreshed["model"] = {"pt2": build_model_status_summary(repaired)}
    return refreshed


loaded = ModelBundle.load(config.model_path_pt2)
if loaded is not None:
    meta = {"trained": True, "path": config.model_path_pt2, **loaded.metadata}
    meta, raw_contract, live_contract, reconciliation = _reconcile_loaded_metadata(meta)
    state.set_model_metadata(meta)
    state.update_status(
        score_contract={**dict(live_contract), "raw_model_contract": dict(raw_contract), "live_contract": dict(live_contract), "score_reconciliation": dict(reconciliation)},
        score_contract_live=live_contract,
        score_contract_raw=raw_contract,
        score_reconciliation=reconciliation,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    scanner.start_background_threads()
    evidence_automation.start_background_threads()
    utility_operator_automation.start_background_threads()
    yield
    utility_operator_automation.stop_background_threads()
    evidence_automation.stop_background_threads()
    scanner.stop_background_threads()


app = FastAPI(title="Coinbase Crypto Prob Scanner", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def require_admin(x_admin_password: str | None = Header(default=None)):
    if x_admin_password != config.admin_password:
        raise HTTPException(status_code=401, detail="unauthorized")


def require_admin_or_query(admin_password: str | None = Query(default=None), x_admin_password: str | None = Header(default=None)):
    if (x_admin_password or admin_password) != config.admin_password:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"demo_mode": config.demo_mode, "app_version": APP_VERSION})


@app.head("/")
def index_head():
    return JSONResponse({"ok": True})




def _public_status_payload(status: dict) -> dict:
    public = dict(status or {})
    pt2 = ((public.get("model") or {}).get("pt2") or {})
    public["model"] = {"pt2": build_model_status_summary(pt2)}
    compact_live = compact_score_contract(public.get("score_contract_live") or public.get("score_contract") or {})
    compact_raw = compact_score_contract(public.get("score_contract_raw") or {})
    compact_rec = compact_score_reconciliation(public.get("score_reconciliation") or {})
    public.setdefault("live_pipeline_mode", getattr(config, "live_pipeline_mode", "raw_threshold"))
    public["score_contract"] = compact_live
    public["score_contract_live"] = compact_live
    public["score_contract_raw"] = compact_raw
    public["score_reconciliation"] = compact_rec
    public["suppression_summary"] = public.get("suppression_summary") or {
        "threshold_suppressed_rows": int(((public.get("coverage") or {}).get("dropped_stage2_threshold_suppressed", 0)) or 0),
        "regime_suppressed_rows": int(((public.get("coverage") or {}).get("dropped_stage2_regime_suppressed", 0)) or 0),
        "cooldown_suppressed_rows": int(((public.get("coverage") or {}).get("dropped_stage2_cooldown_suppressed", 0)) or 0),
        "display_trimmed_rows": int(((public.get("coverage") or {}).get("dropped_stage2_display_trimmed", 0)) or 0),
        "visible_rows": int(((public.get("stage_counts") or {}).get("visible_rows", 0)) or 0),
        "informational_rows": int(((public.get("stage_counts") or {}).get("informational_rows", 0)) or 0),
        "informational_regime_rows": int(((public.get("stage_counts") or {}).get("informational_regime_rows", 0)) or 0),
        "informational_cooldown_rows": int(((public.get("stage_counts") or {}).get("informational_cooldown_rows", 0)) or 0),
        "informational_threshold_rows": int(((public.get("stage_counts") or {}).get("informational_threshold_rows", 0)) or 0),
        "informational_display_trim_rows": int(((public.get("stage_counts") or {}).get("informational_display_trim_rows", 0)) or 0),
        "informational_overflow_rows": int(((public.get("stage_counts") or {}).get("informational_overflow_rows", 0)) or 0),
        "action_ready_rows": int(((public.get("stage_counts") or {}).get("action_ready_rows", 0)) or 0),
        "selective_rows": int(((public.get("stage_counts") or {}).get("selective_rows", 0)) or 0),
        "watchlist_rows": int(((public.get("stage_counts") or {}).get("watchlist_rows", 0)) or 0),
    }
    public["score_diagnostics"] = public.get("score_diagnostics") or {"available": False, "row_count": 0, "counts_above_thresholds": []}
    public["informational_rankings_summary"] = public.get("informational_rankings_summary") or {
        "enabled": bool(config.informational_rankings_enabled),
        "rows": int(((public.get("suppression_summary") or {}).get("informational_rows", 0)) or 0),
        "max_names": int(config.informational_rankings_max_names),
        "regime_rows": int(((public.get("suppression_summary") or {}).get("informational_regime_rows", 0)) or 0),
        "cooldown_rows": int(((public.get("suppression_summary") or {}).get("informational_cooldown_rows", 0)) or 0),
        "threshold_rows": int(((public.get("suppression_summary") or {}).get("informational_threshold_rows", 0)) or 0),
        "display_trim_rows": int(((public.get("suppression_summary") or {}).get("informational_display_trim_rows", 0)) or 0),
        "overflow_rows": int(((public.get("suppression_summary") or {}).get("informational_overflow_rows", 0)) or 0),
        "label": "Informational Rankings (Blocked by Live Policy)",
        "advisory_only": True,
    }
    public["follow_up_scan"] = public.get("follow_up_scan") or {
        "scheduled": False,
        "reason": None,
        "trigger": None,
        "run_after_utc": None,
        "tracked_symbols": [],
        "tracked_count": 0,
        "source_scan_finished_utc": None,
    }
    public["blocked_monitoring_context"] = public.get("blocked_monitoring_context") or {
        "context_active": False,
        "tracked_symbols": [],
        "tracked_rows": [],
        "tracked_count": 0,
        "source_run_finished_utc": None,
    }
    public["followup_comparison"] = public.get("followup_comparison") or {
        "available": False,
        "tracked_count": 0,
        "visible_now_count": 0,
        "still_blocked_count": 0,
        "missing_count": 0,
        "tracked_visible_rows": [],
        "tracked_visible_symbols": [],
        "top_changes": [],
    }
    public["live_candidate_proof"] = public.get("live_candidate_proof") or (live_candidate_proof.latest_summary() or {})
    public["live_candidate_proof_review"] = public.get("live_candidate_proof_review") or (live_candidate_proof_review.latest_summary() or {})
    public["live_candidate_adoption"] = public.get("live_candidate_adoption") or (live_candidate_adoption.latest_summary() or {})
    public["live_candidate_adoption_review"] = public.get("live_candidate_adoption_review") or (live_candidate_adoption_review.latest_summary() or {})
    public["utility_operator_automation"] = public.get("utility_operator_automation") or (utility_operator_automation.latest_summary() or {})
    public["configured_live_selection_mode"] = str(public.get("configured_live_selection_mode") or getattr(config, "live_selection_mode", "legacy") or "legacy")
    public["effective_live_selection_mode"] = str(public.get("effective_live_selection_mode") or public.get("selection_engine") or public["configured_live_selection_mode"])
    public["effective_live_selection_engine"] = str(public.get("effective_live_selection_engine") or public.get("selection_engine") or public["effective_live_selection_mode"])
    public["selection_engine"] = public["effective_live_selection_engine"]
    public["decision_summary"] = public.get("decision_summary") or {
        "headline": "No scan verdict available yet",
        "summary": "Run a scan to generate a shortlist verdict.",
        "validated_floor": min((compact_live.get("validated_thresholds") or [0.60])) if (compact_live.get("validated_thresholds") or [0.60]) else 0.60,
        "near_validated_floor": float(getattr(config, "stage2_near_validated_floor", 0.45) or 0.45),
        "action_ready_rows": int(((public.get("suppression_summary") or {}).get("action_ready_rows", 0)) or 0),
        "selective_rows": int(((public.get("suppression_summary") or {}).get("selective_rows", 0)) or 0),
        "watchlist_rows": int(((public.get("suppression_summary") or {}).get("watchlist_rows", 0)) or 0),
        "near_validated_rows": 0,
        "exploratory_rows": 0,
        "top_focus_symbols": [],
        "no_validated_candidates": True,
        "blocked_rows": 0,
        "blocked_near_validated_rows": 0,
        "blocked_near_threshold_rows": 0,
        "blocked_exploratory_rows": 0,
        "best_blocked_threshold_gap": None,
        "blocked_focus_symbols": [],
        "blocked_focus_count": 0,
        "tracked_visible_count": 0,
        "tracked_visible_symbols": [],
        "cooldown_active": False,
        "cooldown_until_utc": None,
    }
    public["follow_up_scan"] = public.get("follow_up_scan") or {
        "scheduled": False,
        "reason": None,
        "run_after_utc": None,
        "trigger": None,
        "triggered_at_utc": None,
        "max_wait_minutes": int(getattr(config, "cooldown_followup_scan_max_wait_minutes", 12) or 12),
    }
    return public


def _decision_rule_checkpoint(summary: dict | None = None) -> dict:
    return decision_checkpoint.build_summary(summary=summary)

def _decision_branch_summary(checkpoint_summary: dict | None = None) -> dict:
    return decision_branch_automation.build_summary(checkpoint_summary=checkpoint_summary)

def _universe_mode_payload(status: dict) -> dict:
    requested = status.get("live_universe_mode_requested") or config.live_universe_mode
    effective = status.get("live_universe_mode_effective") or (status.get("universe") or {}).get("selection_mode") or (status.get("coverage") or {}).get("cohort_mode")
    if not effective:
        pt2 = (status.get("model") or {}).get("pt2") or {}
        cohort_size = int(pt2.get("trained_cohort_size", 0) or 0)
        if str(requested).lower() == "trained_cohort":
            effective = "trained_cohort_pending" if cohort_size > 0 else "dynamic_fallback"
        else:
            effective = "dynamic"
    return {
        "requested": requested,
        "effective": effective,
    }


@app.get("/health")
def health():
    status = _refresh_runtime_contracts(reason="health")
    pt2 = (status.get("model") or {}).get("pt2") or {}
    universe_mode = _universe_mode_payload(status)
    public = _public_status_payload(status)
    return {
        "ok": True,
        "version": APP_VERSION,
        "app_mode": "demo" if config.demo_mode else "live",
        "data_source_status": public["data_source"]["message"],
        "scan": public["scan"],
        "model": public["model"],
        "regime_context": public.get("regime_context", "unknown"),
        "market_regime": public.get("market_regime") or {},
        "score_contract": public.get("score_contract") or {},
        "score_contract_live": public.get("score_contract_live") or {},
        "score_contract_raw": public.get("score_contract_raw") or {},
        "score_reconciliation": public.get("score_reconciliation") or {},
        "live_universe_mode": universe_mode["effective"],
        "live_universe_mode_requested": universe_mode["requested"],
        "live_universe_mode_effective": universe_mode["effective"],
        "configured_live_selection_mode": str(getattr(config, "live_selection_mode", "legacy") or "legacy"),
        "effective_live_selection_mode": str(public.get("effective_live_selection_mode") or public.get("selection_engine") or getattr(config, "live_selection_mode", "legacy") or "legacy"),
        "effective_live_selection_engine": str(public.get("effective_live_selection_engine") or public.get("selection_engine") or getattr(config, "live_selection_mode", "legacy") or "legacy"),
        "decision_branch_automation": decision_branch_automation.build_summary(),
        "cohort": {
            "selection_mode": (status.get("universe") or {}).get("selection_mode", "dynamic"),
            "trained_cohort_size": int(pt2.get("trained_cohort_size", 0) or 0),
            "trained_cohort_hash": pt2.get("trained_cohort_hash"),
        },
        "target": {
            "move_pct": config.target_move_pct,
            "horizon_minutes": config.target_horizon_minutes,
            "quality_max_mae": config.quality_max_mae,
            "quality_min_end_ret": config.quality_min_end_ret,
        },
    }


def _build_status_snapshot(*, reason: str = "api_status") -> dict:
    status = _refresh_runtime_contracts(reason=reason)
    pt2 = (status.get("model") or {}).get("pt2") or {}
    universe_mode = _universe_mode_payload(status)
    status["target"] = {
        "move_pct": config.target_move_pct,
        "horizon_minutes": config.target_horizon_minutes,
        "quality_max_mae": config.quality_max_mae,
        "quality_min_end_ret": config.quality_min_end_ret,
    }
    status["live_universe_mode"] = universe_mode["effective"]
    status["live_universe_mode_requested"] = universe_mode["requested"]
    status["live_universe_mode_effective"] = universe_mode["effective"]
    status["market_regime"] = status.get("market_regime") or {}
    status["score_contract"] = status.get("score_contract_live") or status.get("score_contract") or {}
    status["actionability_summary"] = status.get("actionability_summary") or {
        "action_ready_rows": 0,
        "selective_rows": 0,
        "watchlist_rows": 0,
        "actionability_type": "advisory_heuristic",
        "tail_validation_state": None,
        "temporal_tail_state": None,
        "temporal_tail_semantics": None,
        "temporal_support_basis": None,
        "market_regime_actionability": None,
    }
    try:
        status["decision_checkpoint"] = decision_checkpoint.build_summary()
    except FileNotFoundError:
        status["decision_checkpoint"] = decision_checkpoint.build_summary(summary={"evidence": {"visible_rows": 0, "visible_quality_hit_rate": None, "non_visible_quality_hit_rate": None}})
    status["decision_branch_automation"] = decision_branch_automation.build_summary(checkpoint_summary=status["decision_checkpoint"])
    status["model_output_distribution"] = model_output_distribution.latest_summary()
    status["cohort"] = {
        "selection_mode": (status.get("universe") or {}).get("selection_mode", "dynamic"),
        "trained_cohort_size": int(pt2.get("trained_cohort_size", 0) or 0),
        "trained_cohort_hash": pt2.get("trained_cohort_hash"),
    }
    return _public_status_payload(status)




@app.get("/health.txt")
def health_txt():
    status = _refresh_runtime_contracts(reason="health_txt")
    pt2 = (status.get("model") or {}).get("pt2") or {}
    universe_mode = _universe_mode_payload(status)
    public = _public_status_payload(status)
    try:
        branch_summary = decision_branch_automation.build_summary()
    except Exception as exc:
        branch_summary = {
            "status": "unavailable",
            "headline": "Decision branch automation summary unavailable",
            "summary": str(exc),
        }
    payload = {
        "ok": True,
        "version": APP_VERSION,
        "app_mode": "demo" if config.demo_mode else "live",
        "data_source_status": public["data_source"]["message"],
        "scan": public["scan"],
        "model": public["model"],
        "regime_context": public.get("regime_context", "unknown"),
        "market_regime": public.get("market_regime") or {},
        "score_contract": public.get("score_contract") or {},
        "score_contract_live": public.get("score_contract_live") or {},
        "score_contract_raw": public.get("score_contract_raw") or {},
        "score_reconciliation": public.get("score_reconciliation") or {},
        "live_universe_mode": universe_mode["effective"],
        "live_universe_mode_requested": universe_mode["requested"],
        "live_universe_mode_effective": universe_mode["effective"],
        "configured_live_selection_mode": str(getattr(config, "live_selection_mode", "legacy") or "legacy"),
        "effective_live_selection_mode": str(public.get("effective_live_selection_mode") or public.get("selection_engine") or getattr(config, "live_selection_mode", "legacy") or "legacy"),
        "effective_live_selection_engine": str(public.get("effective_live_selection_engine") or public.get("selection_engine") or getattr(config, "live_selection_mode", "legacy") or "legacy"),
        "decision_branch_automation": branch_summary,
        "cohort": {
            "selection_mode": (status.get("universe") or {}).get("selection_mode", "dynamic"),
            "trained_cohort_size": int(pt2.get("trained_cohort_size", 0) or 0),
            "trained_cohort_hash": pt2.get("trained_cohort_hash"),
        },
        "target": {
            "move_pct": config.target_move_pct,
            "horizon_minutes": config.target_horizon_minutes,
            "quality_max_mae": config.quality_max_mae,
            "quality_min_end_ret": config.quality_min_end_ret,
        },
    }
    filename = f"health_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%z')}.txt"
    return PlainTextResponse(
        content=json.dumps(payload, indent=2, sort_keys=True, default=str),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
@app.get("/api/status")
def api_status():
    return _build_status_snapshot(reason="api_status")


@app.get("/api/status.txt")
def api_status_txt():
    snapshot = _build_status_snapshot(reason="api_status_txt")
    scan_finished = (((snapshot.get("scan") or {}).get("finished_at_utc") or "").replace(":", "-") or "latest")
    filename = f"api_status_{scan_finished}.txt"
    body = json.dumps(snapshot, indent=2, sort_keys=True, default=str)
    return PlainTextResponse(
        content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/scores")
def api_scores(actionable_only: bool = Query(default=False), informational_only: bool = Query(default=False)):
    actionable = state.get_scores()
    informational = state.get_informational_scores()
    if actionable_only and informational_only:
        raise HTTPException(status_code=400, detail="choose either actionable_only or informational_only")
    if actionable_only:
        return {"rows": actionable, "count": len(actionable), "bucket": "actionable"}
    if informational_only:
        return {"rows": informational, "count": len(informational), "bucket": "informational"}
    return {
        "rows": actionable,
        "count": len(actionable),
        "bucket": "actionable",
        "informational_rows": informational,
        "informational_count": len(informational),
        "informational_bucket": "informational",
    }

@app.get("/api/scores/informational")
def api_scores_informational():
    rows = state.get_informational_scores()
    return {"rows": rows, "count": len(rows), "bucket": "informational"}


@app.post("/train")
def train(_: None = Depends(require_admin)):
    return evidence_automation.start_training()


@app.get("/api/training/status")
def training_status():
    return state.get_training()


@app.get("/api/training/orchestration/status")
def training_orchestration_status():
    return evidence_automation.refresh_training_only(reason="api_training_orchestration_status")


@app.get("/api/training/orchestration/status.txt")
def training_orchestration_status_txt():
    payload = evidence_automation.refresh_training_only(reason="api_training_orchestration_status_txt")
    filename = f"training_orchestration_{(payload.get('generated_at_utc') or 'latest').replace(':', '-')}.txt"
    return PlainTextResponse(
        content=json.dumps(payload, indent=2, sort_keys=True, default=str),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/debug/coverage")
def debug_coverage():
    return state.get_coverage()

@app.get("/api/debug/market-regime")
def debug_market_regime():
    status = _refresh_runtime_contracts(reason="debug_market_regime")
    return status.get("market_regime") or {}


@app.get("/api/paper-trade/summary")
def paper_trade_summary():
    """Forward validation accuracy computed from resolved predictions."""
    return paper_trade.get_summary()




@app.get("/api/reliability-lab")
def reliability_lab():
    """Live reliability dashboard for the active target + current model only."""
    return paper_trade.get_reliability_lab()


@app.get("/api/paper-trade/recent")
def paper_trade_recent(limit: int = 50):
    """Most recent resolved predictions with actual outcomes."""
    rows = paper_trade.get_recent(limit=min(limit, 200))
    return {"rows": rows, "count": len(rows)}


@app.get("/api/paper-trade/counts")
def paper_trade_counts():
    """Quick status: predictions logged, resolved, pending."""
    return paper_trade.get_counts()


@app.get("/api/debug/score-contract")
def debug_score_contract():
    status = _refresh_runtime_contracts(reason="debug_score_contract")
    return status.get("score_contract") or {}


@app.get("/api/debug/reconciliation")
def debug_reconciliation():
    status = _refresh_runtime_contracts(reason="debug_reconciliation")
    return status.get("score_reconciliation") or {}






@app.get("/api/policy-audit")
def api_policy_audit(hours: int = Query(default=24, ge=1, le=24*30)):
    return review_packs.get_policy_audit(hours=hours)


@app.get("/api/policy-audit/run/{run_id}")
def api_policy_audit_run(run_id: str):
    run = review_packs.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run.get("policy_audit") or {}

@app.get("/api/runs")
def api_runs(limit: int = Query(default=20, ge=1, le=100)):
    runs = review_packs.get_runs(limit=limit)
    return {"runs": runs, "count": len(runs)}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    run = review_packs.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/api/runs/{run_id}/download")
def api_run_download(run_id: str, evaluated: bool = Query(default=True)):
    pack = review_packs.get_pack_for_run(run_id, evaluated=evaluated)
    if not pack or not pack.exists():
        raise HTTPException(status_code=404, detail="review pack not found")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/reviews/latest-scan.zip")
def api_latest_scan_pack():
    pack = review_packs.latest_scan_link if review_packs.latest_scan_link.exists() else None
    if not pack:
        latest_runs = review_packs.get_runs(limit=1)
        if not latest_runs:
            raise HTTPException(status_code=404, detail="no scan review packs yet")
        pack = review_packs.get_pack_for_run(latest_runs[0]["run_id"], evaluated=False)
    return FileResponse(str(pack), media_type="application/zip", filename=Path(pack).name)


@app.get("/api/reviews/latest-evaluated.zip")
def api_latest_evaluated_pack():
    pack = review_packs.latest_eval_link if review_packs.latest_eval_link.exists() else None
    if not pack:
        latest_runs = [r for r in review_packs.get_runs(limit=20) if r.get("evaluation_complete")]
        if not latest_runs:
            raise HTTPException(status_code=404, detail="no evaluated review packs yet")
        pack = review_packs.get_pack_for_run(latest_runs[0]["run_id"], evaluated=True)
    return FileResponse(str(pack), media_type="application/zip", filename=Path(pack).name)




@app.post("/api/replay/run")
def run_replay(
    start_utc: str | None = Query(default=None),
    end_utc: str | None = Query(default=None),
    hours: int = Query(default=None),
    step_minutes: int = Query(default=None),
    max_scans: int = Query(default=None),
    max_symbols: int = Query(default=None),
    pipeline_mode: str = Query(default="full"),
    raw_threshold: float = Query(default=0.30),
    _=Depends(require_admin),
):
    if not config.replay_enabled:
        raise HTTPException(status_code=404, detail="historical replay disabled")
    try:
        result = replay.run(
            start_utc=start_utc,
            end_utc=end_utc,
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            pipeline_mode=pipeline_mode,
            raw_threshold=raw_threshold,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/replay/latest-summary")
def replay_latest_summary(_=Depends(require_admin)):
    summary = replay.latest_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="no replay summary available")
    return summary


@app.get("/api/replay/latest.zip")
def replay_latest_zip(_=Depends(require_admin)):
    pack = replay.latest_pack()
    if pack is None or not pack.exists():
        raise HTTPException(status_code=404, detail="no replay pack available")
    return FileResponse(path=str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/stage1-opportunity/build-from-replay")
def build_stage1_opportunity_from_replay(_=Depends(require_admin)):
    try:
        summary = stage1_opportunity.build_from_latest_replay_pack()
        return {"ok": True, "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/stage1-opportunity/summary")
def stage1_opportunity_summary(_=Depends(require_admin)):
    summary = stage1_opportunity.latest_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="no stage1 opportunity scorer summary available")
    return summary


@app.post("/api/model-audit/build-from-replay")
def build_model_audit_from_replay(_=Depends(require_admin)):
    try:
        summary = model_audit.build_from_latest_replay_pack()
        return {"ok": True, "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/model-audit/summary")
def model_audit_summary(_=Depends(require_admin)):
    summary = model_audit.latest_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="no model audit summary available")
    return summary





@app.post("/api/reviews/raw-score-baseline/run")
def run_raw_score_baseline(
    hours: int = Query(default=168, ge=1, le=24 * 30),
    step_minutes: int = Query(default=120, ge=5, le=24 * 60),
    max_scans: int = Query(default=84, ge=1, le=500),
    max_symbols: int = Query(default=100, ge=1, le=1000),
    _=Depends(require_admin),
):
    if not config.replay_enabled:
        raise HTTPException(status_code=404, detail="historical replay disabled")
    try:
        return raw_score_baseline.run(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/reviews/raw-score-baseline/summary")
def raw_score_baseline_summary():
    summary = raw_score_baseline.latest_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="no raw score baseline summary available")
    return summary


@app.get("/api/reviews/raw-score-baseline/summary.txt")
def raw_score_baseline_summary_txt():
    payload = raw_score_baseline.latest_summary()
    if not payload:
        raise HTTPException(status_code=404, detail="no raw score baseline summary available")
    generated = ((payload.get("generated_at_utc") or payload.get("app_version") or "latest").replace(":", "-"))
    filename = f"raw_score_baseline_{generated}.txt"
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    return PlainTextResponse(
        content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/reviews/raw-score-baseline/latest-pack.zip")
def raw_score_baseline_latest_pack(_=Depends(require_admin_or_query)):
    pack = raw_score_baseline.latest_pack()
    if not pack:
        raise HTTPException(status_code=404, detail="no raw score baseline pack available")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)

@app.post("/api/benchmark/run-threshold-sweep")
def run_benchmark_threshold_sweep(
    hours: int = Query(default=96, ge=1, le=24 * 30),
    step_minutes: int = Query(default=120, ge=5, le=24 * 60),
    max_scans: int = Query(default=48, ge=1, le=500),
    max_symbols: int = Query(default=100, ge=1, le=1000),
    thresholds: str = Query(default="0.25,0.30,0.35,0.40"),
    _=Depends(require_admin),
):
    if not config.replay_enabled:
        raise HTTPException(status_code=404, detail="historical replay disabled")
    try:
        summary = benchmark_lab.run_threshold_sweep(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            thresholds=thresholds,
        )
        return {"ok": True, "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/benchmark/summary")
def benchmark_summary(_=Depends(require_admin)):
    summary = benchmark_lab.latest_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="no benchmark summary available")
    return summary



@app.get("/api/benchmark/latest-pack.zip")
def benchmark_latest_pack(_=Depends(require_admin_or_query)):
    try:
        pack = benchmark_lab.build_benchmark_pack()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/benchmark/latest-classification-pack.zip")
def benchmark_latest_classification_pack(_=Depends(require_admin_or_query)):
    try:
        pack = benchmark_lab.build_symbol_classification_pack()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/historical-decision-lab/run")
def run_historical_decision_lab(
    hours: int = Query(default=168, ge=1, le=24 * 30),
    step_minutes: int = Query(default=120, ge=5, le=24 * 60),
    max_scans: int = Query(default=84, ge=1, le=500),
    max_symbols: int = Query(default=100, ge=1, le=1000),
    thresholds: str = Query(default="0.30,0.35,0.40"),
    _=Depends(require_admin),
):
    if not config.replay_enabled:
        raise HTTPException(status_code=404, detail="historical replay disabled")
    try:
        summary = historical_decision_lab.run(
            hours=hours,
            step_minutes=step_minutes,
            max_scans=max_scans,
            max_symbols=max_symbols,
            thresholds=thresholds,
        )
        return {"ok": True, "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/historical-decision-lab/summary")
def historical_decision_lab_summary(_=Depends(require_admin)):
    summary = historical_decision_lab.latest_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="no historical decision lab summary available")
    return summary


@app.get("/api/historical-decision-lab/latest-pack.zip")
def historical_decision_lab_latest_pack(_=Depends(require_admin_or_query)):
    pack = historical_decision_lab.latest_pack()
    if not pack:
        raise HTTPException(status_code=404, detail="no historical decision lab pack available")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)



@app.get("/api/reviews/decision-checkpoint")
def api_decision_checkpoint_summary():
    try:
        return decision_checkpoint.build_summary()
    except FileNotFoundError:
        return decision_checkpoint.build_summary(summary={"evidence": {"visible_rows": 0, "visible_quality_hit_rate": None, "non_visible_quality_hit_rate": None}})


@app.post("/api/reviews/decision-checkpoint/ack")
def api_decision_checkpoint_ack(_: None = Depends(require_admin)):
    return decision_checkpoint.acknowledge()


@app.get("/api/reviews/decision-branch")
def api_decision_branch_summary():
    checkpoint = decision_checkpoint.latest_summary() or decision_checkpoint.build_summary(summary={"evidence": {"visible_rows": 0, "visible_quality_hit_rate": None, "non_visible_quality_hit_rate": None}})
    return decision_branch_automation.build_summary(checkpoint_summary=checkpoint)


@app.post("/api/reviews/decision-branch/auto-execute")
def api_decision_branch_set_auto_execute(enabled: bool = Query(...), _: None = Depends(require_admin)):
    return decision_branch_automation.set_auto_execute_enabled(bool(enabled))


@app.post("/api/reviews/decision-branch/execute")
def api_decision_branch_execute(_: None = Depends(require_admin)):
    return decision_branch_automation.execute_now()


@app.post("/api/reviews/decision-branch/clear-override")
def api_decision_branch_clear_override(_: None = Depends(require_admin)):
    decision_branch_automation.clear_active_override()
    checkpoint = decision_checkpoint.latest_summary() or decision_checkpoint.build_summary(summary={"evidence": {"visible_rows": 0, "visible_quality_hit_rate": None, "non_visible_quality_hit_rate": None}})
    return decision_branch_automation.build_summary(checkpoint_summary=checkpoint)


@app.post("/api/reviews/decision-branch/ack")
def api_decision_branch_ack(_: None = Depends(require_admin)):
    return decision_branch_automation.acknowledge()

@app.post("/api/utility-selection-lab/run")
def api_utility_selection_lab_run(
    hours: int = Form(168),
    step_minutes: int = Form(120),
    max_scans: int = Form(84),
    max_symbols: int = Form(100),
    _: None = Depends(require_admin),
):
    return utility_selection_lab.run(hours=hours, step_minutes=step_minutes, max_scans=max_scans, max_symbols=max_symbols)


@app.get("/api/utility-selection-lab/summary")
def api_utility_selection_lab_summary():
    return utility_selection_lab.latest_summary()


@app.get("/api/utility-selection-lab/summary.txt")
def api_utility_selection_lab_summary_txt():
    payload = utility_selection_lab.latest_summary()
    generated = str(payload.get("generated_at_utc") or "latest").replace(":", "").replace("-", "")
    filename = f"utility_selection_lab_summary_{generated}.txt"
    body = "\n".join([
        f"Headline: {payload.get('headline')}",
        f"Summary: {payload.get('summary')}",
        "",
        str(payload.get('decision_memo_markdown') or ''),
    ])
    return PlainTextResponse(content=body, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-selection-lab/latest-pack.zip")
def api_utility_selection_lab_pack():
    pack = utility_selection_lab.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility selection lab pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/utility-policy-search/run")
def api_utility_policy_search_run(
    hours: int = Form(168),
    step_minutes: int = Form(120),
    max_scans: int = Form(84),
    max_symbols: int = Form(100),
    _: None = Depends(require_admin),
):
    return utility_policy_search_lab.start_run(hours=hours, step_minutes=step_minutes, max_scans=max_scans, max_symbols=max_symbols)


@app.get("/api/utility-policy-search/status")
def api_utility_policy_search_status():
    return utility_policy_search_lab.latest_status()


@app.get("/api/utility-policy-search/summary")
def api_utility_policy_search_summary():
    return utility_policy_search_lab.latest_summary()


@app.get("/api/utility-policy-search/status.txt")
def api_utility_policy_search_status_txt():
    payload = utility_policy_search_lab.latest_status()
    generated = str(payload.get("updated_at_utc") or "latest").replace(":", "").replace("-", "")
    filename = f"utility_policy_search_status_{generated}.txt"
    body = "\n".join([
        f"Headline: {payload.get('headline')}",
        f"Summary: {payload.get('summary')}",
        f"Status: {payload.get('status')}",
        f"Phase: {payload.get('phase')}",
        f"Progress: {payload.get('progress_pct')}",
        f"Started: {payload.get('started_at_utc')}",
        f"Updated: {payload.get('updated_at_utc')}",
        f"Error: {payload.get('last_error')}",
    ])
    return PlainTextResponse(content=body, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-policy-search/summary.txt")
def api_utility_policy_search_summary_txt():
    payload = utility_policy_search_lab.latest_summary()
    generated = str(payload.get("generated_at_utc") or "latest").replace(":", "").replace("-", "")
    filename = f"utility_policy_search_summary_{generated}.txt"
    body = "\n".join([
        f"Headline: {payload.get('headline')}",
        f"Summary: {payload.get('summary')}",
        "",
        str(payload.get('decision_memo_markdown') or ''),
    ])
    return PlainTextResponse(content=body, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-policy-search/latest-pack.zip")
def api_utility_policy_search_pack():
    pack = utility_policy_search_lab.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility policy search pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/shadow-selection-comparison/summary")
def api_shadow_selection_comparison_summary():
    return shadow_selection_comparison.latest_summary()


@app.get("/api/shadow-selection-comparison/summary.txt")
def api_shadow_selection_comparison_summary_txt():
    payload = shadow_selection_comparison.latest_summary()
    generated = str(payload.get("generated_at_utc") or payload.get("updated_at_utc") or "latest").replace(":", "").replace("-", "")
    filename = f"shadow_selection_comparison_summary_{generated}.txt"
    body = "\n".join([
        f"Headline: {payload.get('headline')}",
        f"Summary: {payload.get('summary')}",
        f"Status: {payload.get('status')}",
        f"Live engine: {payload.get('effective_live_selection_engine') or '-'}",
        f"Policy: {((payload.get('challenger_policy') or {}).get('policy_name')) or '-'}",
        "",
        json.dumps(payload, indent=2, sort_keys=True),
    ])
    return PlainTextResponse(content=body, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/shadow-selection-comparison/latest-pack.zip")
def api_shadow_selection_comparison_pack():
    pack = shadow_selection_comparison.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No shadow selection comparison pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/shadow-selection-outcome-review/summary")
def api_shadow_selection_outcome_review_summary():
    return shadow_selection_outcome_review.latest_summary()


@app.get("/api/shadow-selection-outcome-review/summary.txt")
def api_shadow_selection_outcome_review_summary_txt():
    payload = shadow_selection_outcome_review.latest_summary()
    generated = str(payload.get("generated_at_utc") or "latest").replace(":", "").replace("-", "")
    filename = f"shadow_selection_outcome_review_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/shadow-selection-outcome-review/latest-pack.zip")
def api_shadow_selection_outcome_review_pack():
    pack = shadow_selection_outcome_review.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No shadow selection outcome review pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/utility-model-lab/run")
def api_utility_model_lab_run(
    max_symbols: int = Form(0),
    visible_cap: int = Form(0),
    _: None = Depends(require_admin),
):
    return utility_model_lab.run(
        max_symbols=(int(max_symbols) if int(max_symbols or 0) > 0 else None),
        visible_cap=(int(visible_cap) if int(visible_cap or 0) > 0 else None),
    )


@app.get("/api/utility-model-lab/summary")
def api_utility_model_lab_summary():
    return utility_model_lab.latest_summary()


@app.get("/api/utility-model-lab/summary.txt")
def api_utility_model_lab_summary_txt():
    payload = utility_model_lab.latest_summary()
    generated = str(payload.get('generated_at_utc') or 'latest').replace(':', '').replace('-', '')
    filename = f"utility_model_lab_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-model-lab/latest-pack.zip")
def api_utility_model_lab_pack():
    pack = utility_model_lab.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility model lab pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/utility-model-proof/activate")
def api_utility_model_proof_activate(
    proof_hours: int = Form(24),
    _: None = Depends(require_admin),
):
    return utility_model_proof.activate(proof_hours=proof_hours)


@app.post("/api/utility-model-proof/clear")
def api_utility_model_proof_clear(_: None = Depends(require_admin)):
    return utility_model_proof.clear()


@app.get("/api/utility-model-proof/summary")
def api_utility_model_proof_summary():
    return utility_model_proof.build_summary(reason='api_summary')


@app.get("/api/utility-model-proof/summary.txt")
def api_utility_model_proof_summary_txt():
    payload = utility_model_proof.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"utility_model_proof_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-model-proof/latest-pack.zip")
def api_utility_model_proof_pack():
    utility_model_proof.build_summary(reason='api_pack')
    pack = utility_model_proof.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility model proof pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/utility-model-proof-review/summary")
def api_utility_model_proof_review_summary():
    return utility_model_proof_review.build_summary(reason='api_summary')


@app.get("/api/utility-model-proof-review/summary.txt")
def api_utility_model_proof_review_summary_txt():
    payload = utility_model_proof_review.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"utility_model_proof_review_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-model-proof-review/latest-pack.zip")
def api_utility_model_proof_review_pack():
    utility_model_proof_review.build_summary(reason='api_pack')
    pack = utility_model_proof_review.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility model proof review pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/utility-tuning-lab/run")
def api_utility_tuning_lab_run(
    hours: int = Form(168),
    step_minutes: int = Form(120),
    max_scans: int = Form(84),
    max_symbols: int = Form(100),
    _: None = Depends(require_admin),
):
    return utility_tuning_lab.run(hours=hours, step_minutes=step_minutes, max_scans=max_scans, max_symbols=max_symbols)


@app.get("/api/utility-tuning-lab/summary")
def api_utility_tuning_lab_summary():
    return utility_tuning_lab.latest_summary()


@app.get("/api/utility-tuning-lab/summary.txt")
def api_utility_tuning_lab_summary_txt():
    payload = utility_tuning_lab.latest_summary()
    generated = str(payload.get("generated_at_utc") or "latest").replace(":", "").replace("-", "")
    filename = f"utility_tuning_lab_summary_{generated}.txt"
    body = "\n".join([
        f"Headline: {payload.get('headline')}",
        f"Summary: {payload.get('summary')}",
        "",
        str(payload.get('decision_memo_markdown') or ''),
    ])
    return PlainTextResponse(content=body, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-tuning-lab/latest-pack.zip")
def api_utility_tuning_lab_pack():
    pack = utility_tuning_lab.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility tuning lab pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/utility-tuning-proof/activate")
def api_utility_tuning_proof_activate(
    proof_hours: int = Form(24),
    _: None = Depends(require_admin),
):
    return utility_tuning_proof.activate(proof_hours=proof_hours)


@app.post("/api/utility-tuning-proof/clear")
def api_utility_tuning_proof_clear(_: None = Depends(require_admin)):
    return utility_tuning_proof.clear()


@app.get("/api/utility-tuning-proof/summary")
def api_utility_tuning_proof_summary():
    return utility_tuning_proof.build_summary(reason='api_summary')


@app.get("/api/utility-tuning-proof/summary.txt")
def api_utility_tuning_proof_summary_txt():
    payload = utility_tuning_proof.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"utility_tuning_proof_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-tuning-proof/latest-pack.zip")
def api_utility_tuning_proof_pack():
    utility_tuning_proof.build_summary(reason='api_pack')
    pack = utility_tuning_proof.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility tuning proof pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/utility-tuning-proof-review/summary")
def api_utility_tuning_proof_review_summary():
    return utility_tuning_proof_review.build_summary(reason='api_summary')


@app.get("/api/utility-tuning-proof-review/summary.txt")
def api_utility_tuning_proof_review_summary_txt():
    payload = utility_tuning_proof_review.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"utility_tuning_proof_review_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-tuning-proof-review/latest-pack.zip")
def api_utility_tuning_proof_review_pack():
    utility_tuning_proof_review.build_summary(reason='api_pack')
    pack = utility_tuning_proof_review.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility tuning proof review pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/utility-tuning-adoption/activate")
def api_utility_tuning_adoption_activate(_: None = Depends(require_admin)):
    return utility_tuning_adoption.activate()


@app.post("/api/utility-tuning-adoption/clear")
def api_utility_tuning_adoption_clear(_: None = Depends(require_admin)):
    return utility_tuning_adoption.clear()


@app.get("/api/utility-tuning-adoption/summary")
def api_utility_tuning_adoption_summary():
    return utility_tuning_adoption.build_summary(reason='api_summary')


@app.get("/api/utility-tuning-adoption/summary.txt")
def api_utility_tuning_adoption_summary_txt():
    payload = utility_tuning_adoption.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"utility_tuning_adoption_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-tuning-adoption/latest-pack.zip")
def api_utility_tuning_adoption_pack():
    utility_tuning_adoption.build_summary(reason='api_pack')
    pack = utility_tuning_adoption.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility tuning adoption pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/model-output-distribution")
def api_model_output_distribution():
    return model_output_distribution.latest_summary()


@app.get("/api/model-output-distribution.txt")
def api_model_output_distribution_txt():
    payload = model_output_distribution.latest_summary()
    generated = ((payload.get("generated_at_utc") or payload.get("app_version") or "latest").replace(":", "-"))
    filename = f"model_output_distribution_{generated}.txt"
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    return PlainTextResponse(
        content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/automation/status")
def api_automation_status():
    return evidence_automation.latest_status() or evidence_automation.refresh(reason="api_automation_status", event_type="manual_refresh")


@app.post("/api/automation/refresh")
def api_automation_refresh(_: None = Depends(require_admin)):
    return evidence_automation.refresh(reason="admin_refresh", event_type="manual_refresh")


@app.get("/api/reviews/post-maturity-bundle.zip")
def api_post_maturity_bundle():
    if not evidence_automation.bundle_path.exists():
        evidence_automation.refresh(reason="api_post_maturity_bundle", event_type="manual_refresh")
    if not evidence_automation.bundle_path.exists():
        raise HTTPException(status_code=404, detail="no automated review bundle available yet")
    return FileResponse(str(evidence_automation.bundle_path), media_type="application/zip", filename=evidence_automation.bundle_path.name)


@app.get("/api/reviews/diagnostic-battery")
def api_diagnostic_battery():
    if not evidence_automation.diag_json_path.exists():
        evidence_automation.refresh(reason="api_diagnostic_battery", event_type="manual_refresh")
    return evidence_automation.latest_diagnostic_battery()


@app.get("/api/reviews/diagnostic-battery.txt")
def api_diagnostic_battery_txt():
    if not evidence_automation.diag_txt_path.exists():
        evidence_automation.refresh(reason="api_diagnostic_battery_txt", event_type="manual_refresh")
    if not evidence_automation.diag_txt_path.exists():
        raise HTTPException(status_code=404, detail="no diagnostic battery available yet")
    return FileResponse(str(evidence_automation.diag_txt_path), media_type="text/plain", filename=evidence_automation.diag_txt_path.name)


@app.get("/api/reviews/current-version.zip")
def api_current_version_pack(include_evaluated: bool = Query(default=True)):
    try:
        pack = review_packs.build_current_version_pack(include_evaluated=include_evaluated)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


def _build_current_version_summary_payload() -> dict:
    try:
        summary = review_packs.get_current_version_summary()
        summary["model_output_distribution"] = model_output_distribution.latest_summary()
        checkpoint = _decision_rule_checkpoint(summary)
        summary["decision_rule_checkpoint"] = checkpoint
        summary["decision_branch_automation"] = _decision_branch_summary(checkpoint)
        return summary
    except FileNotFoundError:
        return {
            "app_version": APP_VERSION,
            "generated_at_utc": None,
            "deployed_since_utc": None,
            "scan_pack_count": 0,
            "evaluated_pack_count": 0,
            "total_visible_rows": 0,
            "total_suppressed_rows": 0,
            "regime_breakdown": [],
            "scan_score_diagnostics": {"available": False, "scan_count": 0, "counts_above_thresholds": []},
            "evidence": {
                "available": False,
                "resolved_rows": 0,
                "visible_rows": 0,
                "non_visible_rows": 0,
                "visible_quality_hit_rate": None,
                "non_visible_quality_hit_rate": None,
                "visible_raw_hit_rate": None,
                "non_visible_raw_hit_rate": None,
                "visible_avg_end_ret": None,
                "non_visible_avg_end_ret": None,
                "visible_avg_mae": None,
                "non_visible_avg_mae": None,
                "display_trim_quality_hits": 0,
                "threshold_quality_hits": 0,
                "regime_quality_hits": 0,
                "cooldown_quality_hits": 0,
                "score_range": {
                    "max_live_score": None,
                    "p95_live_score": None,
                    "median_live_score": None,
                    "max_pre_policy_score": None,
                    "p95_pre_policy_score": None,
                    "median_pre_policy_score": None,
                },
                "threshold_bands": [],
                "validated_bands_dormant": False,
                "headline": "No review packs yet for this deployed version",
                "summary": "Run a scan and wait for review-pack persistence before judging deployment score ranges.",
            },
            "model_output_distribution": model_output_distribution.latest_summary(),
            "decision_rule_checkpoint": _decision_rule_checkpoint({"evidence": {"visible_rows": 0, "visible_quality_hit_rate": None, "non_visible_quality_hit_rate": None}}),
            "decision_branch_automation": _decision_branch_summary(_decision_rule_checkpoint({"evidence": {"visible_rows": 0, "visible_quality_hit_rate": None, "non_visible_quality_hit_rate": None}})),
        }


@app.get("/api/reviews/current-version-summary")
def api_current_version_summary():
    return _build_current_version_summary_payload()


@app.get("/api/reviews/current-version-summary.txt")
def api_current_version_summary_txt():
    summary = _build_current_version_summary_payload()
    generated = ((summary.get("generated_at_utc") or summary.get("app_version") or "latest").replace(":", "-"))
    filename = f"current_version_summary_{generated}.txt"
    body = json.dumps(summary, indent=2, sort_keys=True, default=str)
    return PlainTextResponse(
        content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )




@app.get("/api/reviews/misranking-diagnostic")
def api_misranking_diagnostic_summary():
    try:
        return misranking_diagnostic.build_summary()
    except FileNotFoundError:
        return {
            "app_version": APP_VERSION,
            "generated_at_utc": None,
            "source": "current_version",
            "headline": "No review packs yet for misranking diagnosis",
            "summary": "Wait for evaluated evidence before trying to diagnose hidden winners and surfaced disappointments.",
            "evidence_snapshot": {
                "resolved_rows": 0,
                "visible_rows": 0,
                "non_visible_rows": 0,
                "visible_quality_hit_rate": None,
                "non_visible_quality_hit_rate": None,
                "visible_avg_end_ret": None,
                "non_visible_avg_end_ret": None,
                "visible_avg_mae": None,
                "non_visible_avg_mae": None,
            },
            "verdict": {
                "visible_beating_hidden": False,
                "green_regime_issue_detected": False,
                "hidden_winner_count": 0,
                "surfaced_disappointment_count": 0,
                "dominant_bottleneck": "no_evidence_yet",
                "dominant_bottleneck_reason": "No evaluated packs are available yet.",
                "recommended_action": "wait_for_evaluated_evidence",
                "recommended_action_reason": "The misranking tranche should only run once resolved evidence exists.",
            },
            "regime_diagnostics": {"available": False, "rows": []},
            "symbol_diagnostics": {
                "available": False,
                "rows": [],
                "hidden_winners": [],
                "surfaced_disappointments": [],
                "correctly_surfaced_strong": [],
                "correctly_hidden_weak": [],
                "too_sparse_to_judge": [],
            },
            "decision_memo_markdown": "# Misranking diagnostic\n\nNo evaluated evidence yet.",
            "notes": ["Run the diagnostic again once evaluated packs exist."],
        }


@app.get("/api/reviews/misranking-diagnostic.zip")
def api_misranking_diagnostic_pack():
    try:
        pack = misranking_diagnostic.build_pack()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)




@app.get("/api/reviews/threshold-boundary-review")
def api_threshold_boundary_review_summary():
    try:
        return threshold_boundary_review.build_summary()
    except FileNotFoundError:
        return {
            "app_version": APP_VERSION,
            "generated_at_utc": None,
            "source": "current_version_no_evaluated_run",
            "headline": "No evaluated run yet for threshold-boundary review",
            "summary": "Wait for an evaluated run before judging whether the shortlist boundary is blocking too many good names.",
            "run_snapshot": {"available": False},
            "threshold_boundary": {
                "available": False,
                "gap_buckets": [],
                "scenario_thresholds": [],
                "top_false_suppressions": [],
                "repeated_near_threshold_symbols": [],
            },
            "verdict": {
                "threshold_boundary_problem_detected": False,
                "dominant_bottleneck": "no_evaluated_run_yet",
                "dominant_bottleneck_reason": "No evaluated run is available yet for threshold-boundary review.",
                "recommended_action": "keep_live_path_unchanged_wait_for_evaluated_run",
                "recommended_action_reason": "Do not move the threshold boundary before at least one run resolves.",
            },
            "decision_memo_markdown": "# Threshold-boundary review\n\nNo evaluated run yet.",
            "notes": ["Run the review again once at least one evaluated run exists."],
        }


@app.get("/api/reviews/threshold-boundary-review.zip")
def api_threshold_boundary_review_pack():
    try:
        pack = threshold_boundary_review.build_pack()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)

@app.get("/api/reviews/cooldown-shortlist-review")
def api_cooldown_shortlist_review_summary():
    try:
        return cooldown_shortlist_review.build_summary()
    except FileNotFoundError:
        return {
            "app_version": APP_VERSION,
            "generated_at_utc": None,
            "source": "current_version_no_cooldown_evaluated_run",
            "headline": "No cooldown-restricted evaluated run yet for cooldown shortlist review",
            "summary": "Wait for a cooldown-restricted evaluated run before diagnosing amber/cooldown shortlist quality.",
            "run_snapshot": {"available": False},
            "cooldown_shortlist": {
                "available": False,
                "near_threshold_hidden_rows": [],
                "repeated_surfaced_weak_symbols": [],
                "repeated_surfaced_strong_symbols": [],
                "rolling_recent_cooldown": {"run_count": 0, "visible": {"count": 0}, "hidden": {"count": 0}},
            },
            "verdict": {
                "cooldown_shortlist_problem_detected": False,
                "dominant_bottleneck": "no_cooldown_restricted_evidence_yet",
                "dominant_bottleneck_reason": "No cooldown-restricted evaluated run is available yet.",
                "recommended_action": "keep_live_path_unchanged_wait_for_cooldown_evidence",
                "recommended_action_reason": "Do not change cooldown shortlist behavior until evaluated evidence exists.",
            },
            "decision_memo_markdown": "# Cooldown-restricted shortlist review\n\nNo evaluated cooldown-restricted evidence yet.",
            "notes": ["Run the review again once a cooldown-restricted evaluated pack exists."],
        }


@app.get("/api/reviews/cooldown-shortlist-review.zip")
def api_cooldown_shortlist_review_pack():
    try:
        pack = cooldown_shortlist_review.build_pack()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)

@app.post("/api/utility-model-adoption/activate")
def api_utility_model_adoption_activate(_: None = Depends(require_admin)):
    return utility_model_adoption.activate()


@app.post("/api/utility-model-adoption/clear")
def api_utility_model_adoption_clear(_: None = Depends(require_admin)):
    return utility_model_adoption.clear()


@app.get("/api/utility-model-adoption/summary")
def api_utility_model_adoption_summary():
    return utility_model_adoption.build_summary(reason='api_summary')


@app.get("/api/utility-model-adoption/summary.txt")
def api_utility_model_adoption_summary_txt():
    payload = utility_model_adoption.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"utility_model_adoption_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-model-adoption/latest-pack.zip")
def api_utility_model_adoption_pack():
    utility_model_adoption.build_summary(reason='api_pack')
    pack = utility_model_adoption.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No utility model adoption pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/reviews/fresh-retrain-audit/run")
def api_fresh_retrain_audit_run(_: None = Depends(require_admin)):
    return fresh_retrain_audit.start_run()


@app.post("/api/reviews/fresh-retrain-audit/stop")
def api_fresh_retrain_audit_stop(_: None = Depends(require_admin)):
    return fresh_retrain_audit.stop_run()


@app.get("/api/reviews/fresh-retrain-audit/summary")
def api_fresh_retrain_audit_summary():
    return fresh_retrain_audit.latest_summary()


@app.get("/api/reviews/fresh-retrain-audit/summary.txt")
def api_fresh_retrain_audit_summary_txt():
    payload = fresh_retrain_audit.latest_summary()
    generated = ((payload.get("generated_at_utc") or payload.get("app_version") or "latest").replace(":", "-"))
    filename = f"fresh_retrain_audit_summary_{generated}.txt"
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    return PlainTextResponse(
        content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/reviews/fresh-retrain-audit/latest-pack.zip")
def api_fresh_retrain_audit_pack():
    pack = fresh_retrain_audit.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No fresh retrain audit pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)




@app.post("/api/reviews/challenger-comparison/run")
def api_challenger_comparison_run(_: None = Depends(require_admin)):
    return offline_challenger_comparison.start_run()


@app.post("/api/reviews/challenger-comparison/stop")
def api_challenger_comparison_stop(_: None = Depends(require_admin)):
    return offline_challenger_comparison.stop_run()


@app.get("/api/reviews/challenger-comparison/summary")
def api_challenger_comparison_summary():
    return offline_challenger_comparison.latest_summary()


@app.get("/api/reviews/challenger-comparison/summary.txt")
def api_challenger_comparison_summary_txt():
    payload = offline_challenger_comparison.latest_summary()
    generated = ((payload.get("generated_at_utc") or payload.get("app_version") or "latest").replace(":", "-"))
    filename = f"challenger_comparison_summary_{generated}.txt"
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    return PlainTextResponse(
        content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/reviews/challenger-comparison/latest-pack.zip")
def api_challenger_comparison_pack():
    pack = offline_challenger_comparison.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No challenger comparison pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/reviews/stage1-policy-lab/run")
def api_stage1_policy_lab_run(
    hours: int = Form(168),
    step_minutes: int = Form(120),
    max_scans: int = Form(84),
    max_symbols: int = Form(100),
    _: None = Depends(require_admin),
):
    return stage1_policy_lab.run(hours=hours, step_minutes=step_minutes, max_scans=max_scans, max_symbols=max_symbols)


@app.get("/api/reviews/stage1-policy-lab/summary")
def api_stage1_policy_lab_summary():
    return stage1_policy_lab.latest_summary()


@app.get("/api/reviews/stage1-policy-lab/summary.txt")
def api_stage1_policy_lab_summary_txt():
    payload = stage1_policy_lab.latest_summary()
    generated = str(payload.get("generated_at_utc") or "latest").replace(":", "").replace("-", "")
    filename = f"stage1_policy_lab_summary_{generated}.txt"
    body = "\n".join([
        f"Headline: {payload.get('headline')}",
        f"Summary: {payload.get('summary')}",
        "",
        str(payload.get('decision_memo_markdown') or ""),
    ])
    return PlainTextResponse(
        content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/reviews/stage1-policy-lab/latest-pack.zip")
def api_stage1_policy_lab_pack():
    pack = stage1_policy_lab.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No stage1 policy lab pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/reviews/next-live-candidate-lab/run")
def api_next_live_candidate_lab_run(
    hours: int = Form(168),
    step_minutes: int = Form(120),
    max_scans: int = Form(84),
    max_symbols: int = Form(100),
    _: None = Depends(require_admin),
):
    return next_live_candidate_lab.run(hours=hours, step_minutes=step_minutes, max_scans=max_scans, max_symbols=max_symbols)


@app.get("/api/reviews/next-live-candidate-lab/summary")
def api_next_live_candidate_lab_summary():
    return next_live_candidate_lab.latest_summary()


@app.get("/api/reviews/next-live-candidate-lab/summary.txt")
def api_next_live_candidate_lab_summary_txt():
    payload = next_live_candidate_lab.latest_summary()
    generated = str(payload.get("generated_at_utc") or "latest").replace(":", "").replace("-", "")
    filename = f"next_live_candidate_lab_summary_{generated}.txt"
    body = "\n".join([
        f"Headline: {payload.get('headline')}",
        f"Summary: {payload.get('summary')}",
        "",
        str(payload.get('decision_memo_markdown') or ''),
    ])
    return PlainTextResponse(
        content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/reviews/next-live-candidate-lab/latest-pack.zip")
def api_next_live_candidate_lab_pack():
    pack = next_live_candidate_lab.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No next live candidate lab pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)



@app.post("/api/reviews/live-candidate-proof/activate")
def api_live_candidate_proof_activate(
    proof_hours: int = Form(24),
    _: None = Depends(require_admin),
):
    return live_candidate_proof.activate(proof_hours=proof_hours)


@app.post("/api/reviews/live-candidate-proof/clear")
def api_live_candidate_proof_clear(_: None = Depends(require_admin)):
    return live_candidate_proof.clear()


@app.get("/api/reviews/live-candidate-proof/summary")
def api_live_candidate_proof_summary():
    return live_candidate_proof.build_summary(reason='api_summary')


@app.get("/api/reviews/live-candidate-proof/summary.txt")
def api_live_candidate_proof_summary_txt():
    payload = live_candidate_proof.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"live_candidate_proof_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/reviews/live-candidate-proof/latest-pack.zip")
def api_live_candidate_proof_pack():
    live_candidate_proof.build_summary(reason='api_pack')
    pack = live_candidate_proof.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No live candidate proof pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)



@app.get("/api/reviews/live-candidate-proof-review/summary")
def api_live_candidate_proof_review_summary():
    return live_candidate_proof_review.build_summary(reason='api_summary')


@app.get("/api/reviews/live-candidate-proof-review/summary.txt")
def api_live_candidate_proof_review_summary_txt():
    payload = live_candidate_proof_review.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"live_candidate_proof_review_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/reviews/live-candidate-proof-review/latest-pack.zip")
def api_live_candidate_proof_review_pack():
    live_candidate_proof_review.build_summary(reason='api_pack')
    pack = live_candidate_proof_review.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No live candidate proof review pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.post("/api/reviews/live-candidate-adoption/activate")
def api_live_candidate_adoption_activate(_: None = Depends(require_admin)):
    return live_candidate_adoption.activate()


@app.post("/api/reviews/live-candidate-adoption/clear")
def api_live_candidate_adoption_clear(_: None = Depends(require_admin)):
    return live_candidate_adoption.clear()


@app.get("/api/reviews/live-candidate-adoption/summary")
def api_live_candidate_adoption_summary():
    return live_candidate_adoption.build_summary(reason='api_summary')


@app.get("/api/reviews/live-candidate-adoption/summary.txt")
def api_live_candidate_adoption_summary_txt():
    payload = live_candidate_adoption.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"live_candidate_adoption_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/reviews/live-candidate-adoption/latest-pack.zip")
def api_live_candidate_adoption_pack():
    live_candidate_adoption.build_summary(reason='api_pack')
    pack = live_candidate_adoption.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No live candidate adoption pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/reviews/live-candidate-adoption-review/summary")
def api_live_candidate_adoption_review_summary():
    return live_candidate_adoption_review.build_summary(reason='api_summary')


@app.get("/api/reviews/live-candidate-adoption-review/summary.txt")
def api_live_candidate_adoption_review_summary_txt():
    payload = live_candidate_adoption_review.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"live_candidate_adoption_review_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/reviews/live-candidate-adoption-review/latest-pack.zip")
def api_live_candidate_adoption_review_pack():
    live_candidate_adoption_review.build_summary(reason='api_pack')
    pack = live_candidate_adoption_review.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No live candidate adoption review pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/reviews/live-candidate-adoption-review/summary")
def api_live_candidate_adoption_review_summary():
    return live_candidate_adoption_review.build_summary(reason='api_summary')


@app.get("/api/reviews/live-candidate-adoption-review/summary.txt")
def api_live_candidate_adoption_review_summary_txt():
    payload = live_candidate_adoption_review.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or '').replace(':', '-').replace('+', '_') or 'latest'
    filename = f"live_candidate_adoption_review_summary_{generated}.txt"
    return PlainTextResponse(_dict_to_text(payload), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/reviews/live-candidate-adoption-review/latest-pack.zip")
def api_live_candidate_adoption_review_pack():
    live_candidate_adoption_review.build_summary(reason='api_pack')
    pack = live_candidate_adoption_review.latest_pack()
    if pack is None:
        raise HTTPException(status_code=404, detail="No live candidate adoption review pack available yet")
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/utility-model-adoption-review/summary")
def api_utility_model_adoption_review_summary():
    return utility_model_adoption_review.build_summary(reason='api_summary')


@app.get("/api/utility-model-adoption-review/summary.txt")
def api_utility_model_adoption_review_summary_txt():
    payload = utility_model_adoption_review.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or 'latest').replace(':', '').replace('-', '')
    filename = f"utility_model_adoption_review_summary_{generated}.txt"
    return PlainTextResponse(json.dumps(payload, indent=2, default=str), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-model-adoption-review/latest-pack.zip")
def api_utility_model_adoption_review_pack():
    utility_model_adoption_review.build_summary(reason='api_pack')
    pack = utility_model_adoption_review.latest_pack()
    if not pack:
        raise HTTPException(status_code=404, detail='No utility-model adoption review pack available yet')
    return FileResponse(pack, media_type='application/zip', filename='utility_model_adoption_review_pack.zip')


@app.get("/api/utility-tuning-adoption-review/summary")
def api_utility_tuning_adoption_review_summary():
    return utility_tuning_adoption_review.build_summary(reason='api_summary')


@app.get("/api/utility-tuning-adoption-review/summary.txt")
def api_utility_tuning_adoption_review_summary_txt():
    payload = utility_tuning_adoption_review.build_summary(reason='api_summary_txt')
    generated = str(payload.get('generated_at_utc') or 'latest').replace(':', '').replace('-', '')
    filename = f"utility_tuning_adoption_review_summary_{generated}.txt"
    return PlainTextResponse(json.dumps(payload, indent=2, default=str), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-tuning-adoption-review/latest-pack.zip")
def api_utility_tuning_adoption_review_pack():
    utility_tuning_adoption_review.build_summary(reason='api_pack')
    pack = utility_tuning_adoption_review.latest_pack()
    if not pack:
        raise HTTPException(status_code=404, detail='No utility-tuning adoption review pack available yet')
    return FileResponse(pack, media_type='application/zip', filename='utility_tuning_adoption_review_pack.zip')


@app.post("/api/utility-operator-automation/start")
def api_utility_operator_automation_start(_: None = Depends(require_admin)):
    return utility_operator_automation.start()


@app.post("/api/utility-operator-automation/stop")
def api_utility_operator_automation_stop(_: None = Depends(require_admin)):
    return utility_operator_automation.stop()


@app.get("/api/utility-operator-automation/status")
def api_utility_operator_automation_status():
    return utility_operator_automation.refresh(reason='api_status')


@app.get("/api/utility-operator-automation/status.txt")
def api_utility_operator_automation_status_txt():
    payload = utility_operator_automation.refresh(reason='api_status_txt')
    generated = str(payload.get('generated_at_utc') or 'latest').replace(':', '').replace('-', '')
    filename = f"utility_operator_automation_status_{generated}.txt"
    return PlainTextResponse(json.dumps(payload, indent=2, default=str), headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/utility-operator-automation/latest-pack.zip")
def api_utility_operator_automation_pack():
    utility_operator_automation.refresh(reason='api_pack')
    pack = utility_operator_automation.latest_pack()
    if not pack:
        raise HTTPException(status_code=404, detail='No utility operator automation pack available yet')
    return FileResponse(pack, media_type='application/zip', filename='utility_operator_automation_pack.zip')


@app.get("/api/reviews/stage2-retrain-review")
def api_stage2_retrain_review_summary():
    try:
        return stage2_retrain_review.build_summary()
    except FileNotFoundError:
        return {
            "available": False,
            "headline": "No Stage 2 retrain review available",
            "summary": "Build review packs first so the app can judge whether a shadow Stage 2 retrain is justified.",
            "verdict": "no_review_evidence_yet",
        }


@app.get("/api/reviews/stage2-retrain-review.zip")
def api_stage2_retrain_review_pack():
    try:
        pack = stage2_retrain_review.build_pack()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/debug/stage2-retrain-review")
def api_debug_stage2_retrain_review():
    return stage2_retrain_review.build_summary()


@app.get("/api/reviews/last-24h.zip")
def api_last_24h_pack():
    pack = review_packs.build_aggregate_pack(hours=24)
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)


@app.get("/api/reviews/last-7d.zip")
def api_last_7d_pack():
    pack = review_packs.build_aggregate_pack(hours=24 * 7)
    return FileResponse(str(pack), media_type="application/zip", filename=pack.name)

@app.get("/api/debug/model-metadata")
def debug_model_metadata():
    _refresh_runtime_contracts(reason="debug_model_metadata")
    return {"pt2": state.model_metadata.get("pt2") or {}}


@app.get("/api/debug/stage1-omission-audit")
def api_debug_stage1_omission_audit():
    status = state.get_status()
    return status.get("stage1_omission_audit") or {"available": False}


@app.get("/api/debug/stage1-selection-repair")
def api_debug_stage1_selection_repair():
    status = state.get_status()
    return status.get("stage1_selection_repair_review") or {"available": False}


@app.get("/api/debug/threshold-experiment")
def api_debug_threshold_experiment():
    status = state.get_status()
    return status.get("threshold_experiment_review") or {"available": False}
