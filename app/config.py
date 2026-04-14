from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Set


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_csv(name: str, default: str) -> List[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class AppConfig:
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "/var/data/model"))
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "changeme"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    demo_mode: bool = field(default_factory=lambda: _get_bool("DEMO_MODE", True))
    disable_scheduler: bool = field(default_factory=lambda: _get_bool("DISABLE_SCHEDULER", False))
    startup_scan: bool = field(default_factory=lambda: _get_bool("STARTUP_SCAN", True))
    scan_interval_minutes: int = field(default_factory=lambda: _get_int("SCAN_INTERVAL_MINUTES", 15))

    http_timeout_seconds: float = field(default_factory=lambda: _get_float("HTTP_TIMEOUT_SECONDS", 12.0))
    request_pause_seconds: float = field(default_factory=lambda: _get_float("REQUEST_PAUSE_SECONDS", 0.05))
    max_workers: int = field(default_factory=lambda: max(1, _get_int("MAX_WORKERS", 2)))

    coinbase_exchange_base_url: str = field(default_factory=lambda: os.getenv("COINBASE_EXCHANGE_BASE_URL", "https://api.exchange.coinbase.com"))
    coinbase_advanced_base_url: str = field(default_factory=lambda: os.getenv("COINBASE_ADVANCED_BASE_URL", "https://api.coinbase.com/api/v3/brokerage/market"))
    preferred_coinbase_runtime: str = field(default_factory=lambda: os.getenv("PREFERRED_COINBASE_RUNTIME", "exchange_public"))

    universe_policy: str = field(default_factory=lambda: os.getenv("UNIVERSE_POLICY", "full_eligible"))
    live_universe_mode: str = field(default_factory=lambda: os.getenv("LIVE_UNIVERSE_MODE", "trained_cohort"))
    universe_top_n: int = field(default_factory=lambda: _get_int("UNIVERSE_TOP_N", 120))
    universe_max_products: int = field(default_factory=lambda: _get_int("UNIVERSE_MAX_PRODUCTS", 0))
    universe_quotes: List[str] = field(default_factory=lambda: _get_csv("UNIVERSE_QUOTES", "USD,USDC"))
    stablecoin_exclusion_enabled: bool = field(default_factory=lambda: _get_bool("EXCLUDE_STABLECOINS", True))
    exclude_view_only: bool = field(default_factory=lambda: _get_bool("EXCLUDE_VIEW_ONLY", True))
    exclusion_list: Set[str] = field(default_factory=lambda: set(_get_csv("EXCLUSION_LIST", "")))

    universe_min_listing_age_days: int = field(default_factory=lambda: _get_int("UNIVERSE_MIN_LISTING_AGE_DAYS", 21))
    universe_min_history_5m_bars: int = field(default_factory=lambda: _get_int("UNIVERSE_MIN_HISTORY_5M_BARS", 288))
    universe_min_observed_5m_bars: int = field(default_factory=lambda: _get_int("UNIVERSE_MIN_OBSERVED_5M_BARS", 36))
    universe_min_24h_dollar_volume_usd: float = field(default_factory=lambda: _get_float("UNIVERSE_MIN_24H_DOLLAR_VOLUME_USD", 500_000.0))
    universe_min_activity_rate: float = field(default_factory=lambda: _get_float("UNIVERSE_MIN_ACTIVITY_RATE", 0.40))
    universe_max_illiquidity_proxy: float = field(default_factory=lambda: _get_float("UNIVERSE_MAX_ILLIQUIDITY_PROXY", 0.06))

    stage1_light_calendar_5m_bars: int = field(default_factory=lambda: _get_int("STAGE1_LIGHT_CALENDAR_5M_BARS", 864))
    stage1_light_feature_5m_bars: int = field(default_factory=lambda: _get_int("STAGE1_LIGHT_FEATURE_5M_BARS", 300))
    stage1_min_history_5m_bars: int = field(default_factory=lambda: _get_int("STAGE1_MIN_HISTORY_5M_BARS", 96))
    stage1_min_observed_5m_bars: int = field(default_factory=lambda: _get_int("STAGE1_MIN_OBSERVED_5M_BARS", 18))
    stage1_max_candidates: int = field(default_factory=lambda: _get_int("STAGE1_MAX_CANDIDATES", 40))
    stage1_panic_max_candidates: int = field(default_factory=lambda: _get_int("STAGE1_PANIC_MAX_CANDIDATES", 40))
    stage1_selection_mode: str = field(default_factory=lambda: os.getenv("STAGE1_SELECTION_MODE", "stage1_opportunity_model"))
    stage1_recall_reserve_frac: float = field(default_factory=lambda: _get_float("STAGE1_RECALL_RESERVE_FRAC", 0.25))
    stage1_recall_reserve_min: int = field(default_factory=lambda: _get_int("STAGE1_RECALL_RESERVE_MIN", 6))
    stage1_recall_reserve_max: int = field(default_factory=lambda: _get_int("STAGE1_RECALL_RESERVE_MAX", 12))
    stage1_promotion_overflow_window: int = field(default_factory=lambda: _get_int("STAGE1_PROMOTION_OVERFLOW_WINDOW", 20))

    stage2_lookback_5m_bars: int = field(default_factory=lambda: _get_int("STAGE2_LOOKBACK_5M_BARS", 2400))
    stage2_min_history_5m_bars: int = field(default_factory=lambda: _get_int("STAGE2_MIN_HISTORY_5M_BARS", 288))
    stage2_min_observed_5m_bars: int = field(default_factory=lambda: _get_int("STAGE2_MIN_OBSERVED_5M_BARS", 48))
    stage2_max_names: int = field(default_factory=lambda: _get_int("STAGE2_MAX_NAMES", 50))
    stage2_panic_max_names: int = field(default_factory=lambda: _get_int("STAGE2_PANIC_MAX_NAMES", 15))
    stage2_watchlist_max_names: int = field(default_factory=lambda: _get_int("STAGE2_WATCHLIST_MAX_NAMES", 12))
    stage2_watchlist_only_max_names: int = field(default_factory=lambda: _get_int("STAGE2_WATCHLIST_ONLY_MAX_NAMES", 12))
    stage2_watchlist_only_exploratory_max_names: int = field(default_factory=lambda: _get_int("STAGE2_WATCHLIST_ONLY_EXPLORATORY_MAX_NAMES", 5))
    stage2_blocked_focus_top_n: int = field(default_factory=lambda: _get_int("STAGE2_BLOCKED_FOCUS_TOP_N", 3))
    stage2_blocked_near_threshold_gap: float = field(default_factory=lambda: _get_float("STAGE2_BLOCKED_NEAR_THRESHOLD_GAP", 0.08))
    stage2_decision_focus_top_n: int = field(default_factory=lambda: _get_int("STAGE2_DECISION_FOCUS_TOP_N", 5))
    stage2_near_validated_floor: float = field(default_factory=lambda: _get_float("STAGE2_NEAR_VALIDATED_FLOOR", 0.45))
    stage2_min_dollar_volume_hard: float = field(default_factory=lambda: _get_float("STAGE2_MIN_DOLLAR_VOLUME_HARD", 100_000.0))
    stage2_min_dollar_volume_soft: float = field(default_factory=lambda: _get_float("STAGE2_MIN_DOLLAR_VOLUME_SOFT", 250_000.0))

    target_move_pct: float = field(default_factory=lambda: _get_float("TARGET_MOVE_PCT", 0.02))
    target_horizon_minutes: int = field(default_factory=lambda: _get_int("TARGET_HORIZON_MINUTES", 240))
    quality_max_mae: float = field(default_factory=lambda: _get_float("QUALITY_MAX_MAE", -0.020))
    quality_min_end_ret: float = field(default_factory=lambda: _get_float("QUALITY_MIN_END_RET", -0.008))

    downside_cap: float = field(default_factory=lambda: _get_float("DOWNSIDE_CAP", 0.78))
    uncertainty_cap: float = field(default_factory=lambda: _get_float("UNCERTAINTY_CAP", 0.72))
    event_risk_cap: float = field(default_factory=lambda: _get_float("EVENT_RISK_CAP", 0.70))
    btc_panic_threshold: float = field(default_factory=lambda: _get_float("BTC_PANIC_THRESHOLD", -0.025))
    panic_threshold_boost: float = field(default_factory=lambda: _get_float("PANIC_THRESHOLD_BOOST", 0.10))

    train_lookback_days: int = field(default_factory=lambda: _get_int("TRAIN_LOOKBACK_DAYS", 120))
    train_max_symbols: int = field(default_factory=lambda: _get_int("TRAIN_MAX_SYMBOLS", 90))
    train_feature_warmup_5m_bars: int = field(default_factory=lambda: _get_int("TRAIN_FEATURE_WARMUP_5M_BARS", 288))
    train_sample_every_n_bars: int = field(default_factory=lambda: _get_int("TRAIN_SAMPLE_EVERY_N_BARS", 3))

    paper_trade_log_enabled: bool = field(default_factory=lambda: _get_bool("PAPER_TRADE_LOG_ENABLED", True))
    demo_fail_symbols: Set[str] = field(default_factory=lambda: set(_get_csv("DEMO_FAIL_SYMBOLS", "")))

    # v4.0.0 market regime engine
    market_regime_override: str = field(default_factory=lambda: os.getenv("MARKET_REGIME_OVERRIDE", ""))
    market_regime_override_note: str = field(default_factory=lambda: os.getenv("MARKET_REGIME_OVERRIDE_NOTE", ""))
    market_regime_cooldown_minutes: int = field(default_factory=lambda: _get_int("MARKET_REGIME_COOLDOWN_MINUTES", 60))
    cooldown_followup_scan_enabled: bool = field(default_factory=lambda: _get_bool("COOLDOWN_FOLLOWUP_SCAN_ENABLED", True))
    cooldown_followup_scan_max_wait_minutes: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_SCAN_MAX_WAIT_MINUTES", 120))
    cooldown_followup_scan_delay_seconds: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_SCAN_DELAY_SECONDS", 10))
    cooldown_followup_scan_min_blocked_rows: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_SCAN_MIN_BLOCKED_ROWS", 3))
    cooldown_followup_track_top_n: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_TRACK_TOP_N", 5))
    cooldown_followup_stage1_reserve_count: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_STAGE1_RESERVE_COUNT", 5))
    cooldown_followup_context_max_age_minutes: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_CONTEXT_MAX_AGE_MINUTES", 180))
    cooldown_followup_comparison_top_n: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_COMPARISON_TOP_N", 5))
    cooldown_followup_visible_pin_count: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_VISIBLE_PIN_COUNT", 5))
    cooldown_followup_confirmation_enabled: bool = field(default_factory=lambda: _get_bool("COOLDOWN_FOLLOWUP_CONFIRMATION_ENABLED", True))
    cooldown_followup_confirmation_delay_minutes: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_CONFIRMATION_DELAY_MINUTES", 10))
    cooldown_followup_confirmation_min_improved_live: int = field(default_factory=lambda: _get_int("COOLDOWN_FOLLOWUP_CONFIRMATION_MIN_IMPROVED_LIVE", 1))
    cooldown_campaign_max_tracked_symbols: int = field(default_factory=lambda: _get_int("COOLDOWN_CAMPAIGN_MAX_TRACKED_SYMBOLS", 12))
    cooldown_campaign_max_source_runs: int = field(default_factory=lambda: _get_int("COOLDOWN_CAMPAIGN_MAX_SOURCE_RUNS", 8))

    market_regime_red_btc_15m_shock: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_BTC_15M_SHOCK", 0.020))
    market_regime_red_btc_1h_move: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_BTC_1H_MOVE", 0.025))
    market_regime_red_eth_1h_move: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_ETH_1H_MOVE", 0.030))
    market_regime_red_btc_vol_ratio: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_BTC_VOL_RATIO", 1.80))
    market_regime_red_eth_vol_ratio: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_ETH_VOL_RATIO", 1.80))
    market_regime_red_breadth_neg_15m: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_BREADTH_NEG_15M", 0.70))
    market_regime_red_breadth_neg_1h: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_BREADTH_NEG_1H", 0.65))
    market_regime_red_score: int = field(default_factory=lambda: _get_int("MARKET_REGIME_RED_SCORE", 3))
    market_regime_red_total_score: int = field(default_factory=lambda: _get_int("MARKET_REGIME_RED_TOTAL_SCORE", 5))

    market_regime_amber_btc_15m_move: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_BTC_15M_MOVE", 0.010))
    market_regime_amber_btc_1h_move: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_BTC_1H_MOVE", 0.015))
    market_regime_amber_eth_1h_move: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_ETH_1H_MOVE", 0.018))
    market_regime_amber_btc_vol_ratio: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_BTC_VOL_RATIO", 1.35))
    market_regime_amber_eth_vol_ratio: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_ETH_VOL_RATIO", 1.35))
    market_regime_amber_breadth_neg_15m: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_BREADTH_NEG_15M", 0.58))
    market_regime_amber_breadth_neg_1h: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_BREADTH_NEG_1H", 0.55))
    market_regime_amber_score: int = field(default_factory=lambda: _get_int("MARKET_REGIME_AMBER_SCORE", 1))
    market_regime_amber_total_score: int = field(default_factory=lambda: _get_int("MARKET_REGIME_AMBER_TOTAL_SCORE", 2))

    market_regime_partial_min_feature_rows: int = field(default_factory=lambda: _get_int("MARKET_REGIME_PARTIAL_MIN_FEATURE_ROWS", 80))
    market_regime_partial_publish_every: int = field(default_factory=lambda: _get_int("MARKET_REGIME_PARTIAL_PUBLISH_EVERY", 40))

    market_regime_liquid_major_symbols: Set[str] = field(default_factory=lambda: set(_get_csv("MARKET_REGIME_LIQUID_MAJOR_SYMBOLS", "SOL-USD,XRP-USD,LINK-USD,LTC-USD,DOGE-USD,ADA-USD,AVAX-USD,SUI-USD")))
    market_regime_tier2_volume_floor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_TIER2_VOLUME_FLOOR", 5_000_000.0))

    market_regime_green_tier1_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER1_FACTOR", 1.00))
    market_regime_green_tier1_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER1_CAP", 0.95))
    market_regime_green_tier1_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER1_THRESHOLD", 0.00))
    market_regime_green_tier1_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_GREEN_TIER1_SUPPRESS", False))
    market_regime_green_tier2_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER2_FACTOR", 1.00))
    market_regime_green_tier2_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER2_CAP", 0.95))
    market_regime_green_tier2_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER2_THRESHOLD", 0.00))
    market_regime_green_tier2_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_GREEN_TIER2_SUPPRESS", False))
    market_regime_green_tier3_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER3_FACTOR", 1.00))
    market_regime_green_tier3_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER3_CAP", 0.95))
    market_regime_green_tier3_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_GREEN_TIER3_THRESHOLD", 0.00))
    market_regime_green_tier3_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_GREEN_TIER3_SUPPRESS", False))

    market_regime_amber_tier1_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER1_FACTOR", 0.85))
    market_regime_amber_tier1_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER1_CAP", 0.88))
    market_regime_amber_tier1_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER1_THRESHOLD", 0.74))
    market_regime_amber_tier1_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_AMBER_TIER1_SUPPRESS", False))
    market_regime_amber_tier2_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER2_FACTOR", 0.75))
    market_regime_amber_tier2_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER2_CAP", 0.80))
    market_regime_amber_tier2_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER2_THRESHOLD", 0.76))
    market_regime_amber_tier2_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_AMBER_TIER2_SUPPRESS", False))
    market_regime_amber_tier3_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER3_FACTOR", 0.60))
    market_regime_amber_tier3_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER3_CAP", 0.72))
    market_regime_amber_tier3_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER3_THRESHOLD", 0.82))
    market_regime_amber_tier3_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_AMBER_TIER3_SUPPRESS", False))
    market_regime_amber_relative_threshold_enabled: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_AMBER_RELATIVE_THRESHOLD_ENABLED", True))
    market_regime_amber_tier1_min_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER1_MIN_THRESHOLD", 0.42))
    market_regime_amber_tier2_min_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER2_MIN_THRESHOLD", 0.36))
    market_regime_amber_tier3_min_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_AMBER_TIER3_MIN_THRESHOLD", 0.30))
    market_regime_amber_tier1_top_n: int = field(default_factory=lambda: _get_int("MARKET_REGIME_AMBER_TIER1_TOP_N", 6))
    market_regime_amber_tier2_top_n: int = field(default_factory=lambda: _get_int("MARKET_REGIME_AMBER_TIER2_TOP_N", 4))
    market_regime_amber_tier3_top_n: int = field(default_factory=lambda: _get_int("MARKET_REGIME_AMBER_TIER3_TOP_N", 2))

    market_regime_red_tier1_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER1_FACTOR", 0.65))
    market_regime_red_tier1_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER1_CAP", 0.65))
    market_regime_red_tier1_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER1_THRESHOLD", 0.82))
    market_regime_red_tier1_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_RED_TIER1_SUPPRESS", False))
    market_regime_red_tier2_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER2_FACTOR", 0.55))
    market_regime_red_tier2_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER2_CAP", 0.50))
    market_regime_red_tier2_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER2_THRESHOLD", 0.84))
    market_regime_red_tier2_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_RED_TIER2_SUPPRESS", False))
    market_regime_red_tier3_factor: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER3_FACTOR", 0.00))
    market_regime_red_tier3_cap: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER3_CAP", 0.00))
    market_regime_red_tier3_threshold: float = field(default_factory=lambda: _get_float("MARKET_REGIME_RED_TIER3_THRESHOLD", 1.00))
    market_regime_red_tier3_suppress: bool = field(default_factory=lambda: _get_bool("MARKET_REGIME_RED_TIER3_SUPPRESS", True))



    # v4.1.0 rolling previews + tail trust
    rolling_candidates_enabled: bool = field(default_factory=lambda: _get_bool("ROLLING_CANDIDATES_ENABLED", True))
    rolling_candidates_min_feature_rows: int = field(default_factory=lambda: _get_int("ROLLING_CANDIDATES_MIN_FEATURE_ROWS", 40))
    rolling_candidates_publish_every: int = field(default_factory=lambda: _get_int("ROLLING_CANDIDATES_PUBLISH_EVERY", 25))
    rolling_candidates_max_names: int = field(default_factory=lambda: _get_int("ROLLING_CANDIDATES_MAX_NAMES", 10))

    tail_validation_min_count: int = field(default_factory=lambda: _get_int("TAIL_VALIDATION_MIN_COUNT", 25))
    tail_validation_min_wilson_lift: float = field(default_factory=lambda: _get_float("TAIL_VALIDATION_MIN_WILSON_LIFT", 1.10))
    tail_validation_min_precision_floor: float = field(default_factory=lambda: _get_float("TAIL_VALIDATION_MIN_PRECISION_FLOOR", 0.18))
    tail_unvalidated_cap: float = field(default_factory=lambda: _get_float("TAIL_UNVALIDATED_CAP", 0.65))

    # v4.10.0 live pipeline simplification
    live_pipeline_mode: str = field(default_factory=lambda: os.getenv("LIVE_PIPELINE_MODE", "raw_threshold").strip().lower() or "raw_threshold")
    live_raw_threshold: float = field(default_factory=lambda: _get_float("LIVE_RAW_THRESHOLD", 0.30))
    live_selection_mode: str = field(default_factory=lambda: os.getenv("LIVE_SELECTION_MODE", "legacy").strip().lower() or "utility_constrained")
    utility_shortlist_target_max_names: int = field(default_factory=lambda: _get_int("UTILITY_SHORTLIST_TARGET_MAX_NAMES", 8))
    utility_shortlist_score_floor: float = field(default_factory=lambda: _get_float("UTILITY_SHORTLIST_SCORE_FLOOR", 0.52))
    utility_shortlist_score_dropoff: float = field(default_factory=lambda: _get_float("UTILITY_SHORTLIST_SCORE_DROPOFF", 0.16))
    utility_confidence_floor: float = field(default_factory=lambda: _get_float("UTILITY_CONFIDENCE_FLOOR", 0.28))
    utility_tier3_max_frac: float = field(default_factory=lambda: _get_float("UTILITY_TIER3_MAX_FRAC", 0.25))
    utility_pinned_visible_cap: int = field(default_factory=lambda: _get_int("UTILITY_PINNED_VISIBLE_CAP", 2))
    utility_tracked_symbol_floor_relaxation: float = field(default_factory=lambda: _get_float("UTILITY_TRACKED_SYMBOL_FLOOR_RELAXATION", 0.04))
    utility_tracked_symbol_confidence_relaxation: float = field(default_factory=lambda: _get_float("UTILITY_TRACKED_SYMBOL_CONFIDENCE_RELAXATION", 0.04))
    utility_expected_edge_weight: float = field(default_factory=lambda: _get_float("UTILITY_EXPECTED_EDGE_WEIGHT", 0.52))
    utility_confidence_weight: float = field(default_factory=lambda: _get_float("UTILITY_CONFIDENCE_WEIGHT", 0.30))
    utility_probability_weight: float = field(default_factory=lambda: _get_float("UTILITY_PROBABILITY_WEIGHT", 0.18))
    utility_scan_readiness_floor: float = field(default_factory=lambda: _get_float("UTILITY_SCAN_READINESS_FLOOR", 0.57))
    utility_pairwise_margin_floor: float = field(default_factory=lambda: _get_float("UTILITY_PAIRWISE_MARGIN_FLOOR", 0.055))
    utility_pairwise_margin_soft_floor: float = field(default_factory=lambda: _get_float("UTILITY_PAIRWISE_MARGIN_SOFT_FLOOR", 0.03))
    utility_multi_name_relaxation: float = field(default_factory=lambda: _get_float("UTILITY_MULTI_NAME_RELAXATION", 0.06))
    utility_strong_support_count_min: int = field(default_factory=lambda: _get_int("UTILITY_STRONG_SUPPORT_COUNT_MIN", 2))
    utility_moderate_support_count_min: int = field(default_factory=lambda: _get_int("UTILITY_MODERATE_SUPPORT_COUNT_MIN", 2))
    utility_strong_top_live_floor: float = field(default_factory=lambda: _get_float("UTILITY_STRONG_TOP_LIVE_FLOOR", 0.42))
    utility_moderate_top_live_floor: float = field(default_factory=lambda: _get_float("UTILITY_MODERATE_TOP_LIVE_FLOOR", 0.34))
    utility_weak_top_live_floor: float = field(default_factory=lambda: _get_float("UTILITY_WEAK_TOP_LIVE_FLOOR", 0.28))

    # v4.2.0 automated review packs
    review_packs_enabled: bool = field(default_factory=lambda: _get_bool("REVIEW_PACKS_ENABLED", True))
    review_evaluator_enabled: bool = field(default_factory=lambda: _get_bool("REVIEW_EVALUATOR_ENABLED", True))
    review_evaluate_interval_minutes: int = field(default_factory=lambda: _get_int("REVIEW_EVALUATE_INTERVAL_MINUTES", 5))
    review_outcome_buffer_minutes: int = field(default_factory=lambda: _get_int("REVIEW_OUTCOME_BUFFER_MINUTES", 5))
    review_retention_days: int = field(default_factory=lambda: _get_int("REVIEW_RETENTION_DAYS", 30))
    review_max_runs_in_aggregate: int = field(default_factory=lambda: _get_int("REVIEW_MAX_RUNS_IN_AGGREGATE", 50))
    review_resolve_batch_runs: int = field(default_factory=lambda: _get_int("REVIEW_RESOLVE_BATCH_RUNS", 50))
    review_resolve_max_loops: int = field(default_factory=lambda: _get_int("REVIEW_RESOLVE_MAX_LOOPS", 4))
    review_pack_recent_resolved_limit: int = field(default_factory=lambda: _get_int("REVIEW_PACK_RECENT_RESOLVED_LIMIT", 250))
    review_pack_recent_resolved_lookback_days: int = field(default_factory=lambda: _get_int("REVIEW_PACK_RECENT_RESOLVED_LOOKBACK_DAYS", 30))

    # v4.11.29 evidence automation + safe orchestration
    automation_enabled: bool = field(default_factory=lambda: _get_bool("AUTOMATION_ENABLED", True))
    automation_safe_branch_ack_enabled: bool = field(default_factory=lambda: _get_bool("AUTOMATION_SAFE_BRANCH_ACK_ENABLED", True))
    automation_review_bundle_enabled: bool = field(default_factory=lambda: _get_bool("AUTOMATION_REVIEW_BUNDLE_ENABLED", True))
    automation_diagnostic_battery_enabled: bool = field(default_factory=lambda: _get_bool("AUTOMATION_DIAGNOSTIC_BATTERY_ENABLED", True))
    automation_training_orchestration_enabled: bool = field(default_factory=lambda: _get_bool("AUTOMATION_TRAINING_ORCHESTRATION_ENABLED", True))
    automation_training_poll_seconds: int = field(default_factory=lambda: _get_int("AUTOMATION_TRAINING_POLL_SECONDS", 30))

    # v4.2.1 informational suppressed rankings
    informational_rankings_enabled: bool = field(default_factory=lambda: _get_bool("INFORMATIONAL_RANKINGS_ENABLED", True))
    informational_rankings_max_names: int = field(default_factory=lambda: _get_int("INFORMATIONAL_RANKINGS_MAX_NAMES", 25))
    informational_include_display_trimmed: bool = field(default_factory=lambda: _get_bool("INFORMATIONAL_INCLUDE_DISPLAY_TRIMMED", True))

    # v4.6.0 historical live-emulation replay
    replay_enabled: bool = field(default_factory=lambda: _get_bool("REPLAY_ENABLED", True))
    replay_default_hours: int = field(default_factory=lambda: _get_int("REPLAY_DEFAULT_HOURS", 24))
    replay_default_step_minutes: int = field(default_factory=lambda: _get_int("REPLAY_DEFAULT_STEP_MINUTES", 60))
    replay_max_scans: int = field(default_factory=lambda: _get_int("REPLAY_MAX_SCANS", 24))
    replay_max_symbols: int = field(default_factory=lambda: _get_int("REPLAY_MAX_SYMBOLS", 100))
    replay_prefetch_max_workers: int = field(default_factory=lambda: _get_int("REPLAY_PREFETCH_MAX_WORKERS", 2))

    @property
    def candles_per_horizon(self) -> int:
        return max(1, self.target_horizon_minutes // 5)

    @property
    def model_path_pt2(self) -> str:
        return os.path.join(self.model_dir, "pt2")

    @property
    def paper_trade_log_path(self) -> str:
        return os.path.join(self.model_dir, "paper_trade_log.jsonl")
