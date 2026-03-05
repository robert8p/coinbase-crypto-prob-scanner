import os
from dataclasses import dataclass
from typing import List

def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1","true","yes","y","on")

def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def _csv(v: str | None) -> List[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]

@dataclass(frozen=True)
class Settings:
    horizon_minutes: int = 240
    horizon_mode: str = "FIXED"
    scan_interval_minutes: int = 30
    min_bars_5m: int = 7
    timezone: str = "UTC"
    benchmark_symbol: str = "BTC-USD"

    crypto_universe: List[str] | None = None
    quote_allowlist: List[str] = None  # type: ignore
    exclude_stablecoin_base: bool = True
    stablecoin_bases: List[str] = None  # type: ignore
    universe_max: int = 250
    universe_refresh_minutes: int = 360
    min_24h_dollar_volume: float = 0.0

    coinbase_base_url: str = "https://api.exchange.coinbase.com"
    coinbase_max_rps: float = 5.0
    coinbase_max_inflight: int = 5
    max_candle_staleness_minutes: int = 60

    admin_password: str | None = None
    train_lookback_days: int = 60
    train_max_products: int = 0
    calib_min_bucket_samples: int = 200
    enet_c_values: List[float] = None  # type: ignore
    enet_l1_values: List[float] = None  # type: ignore
    prior_alpha_values: List[float] = None  # type: ignore

    model_dir: str = "./runtime/model"

    demo_mode: bool = False
    disable_scheduler: bool = False
    debug_password: str | None = None

    run_scan_on_startup: bool = True
    startup_scan_delay_seconds: int = 2

    auto_train_on_startup: bool = True
    auto_train_max_products: int = 20
    auto_train_lookback_days: int = 14
    auto_train_cooldown_minutes: int = 720

    tod_rvol_lookback_days: int = 20
    tod_rvol_min_days: int = 8

    liq_rolling_bars: int = 12
    liq_dvol_min_usd: float = 2_000_000.0
    liq_range_pct_max: float = 0.012
    liq_wick_atr_max: float = 0.8

def load_settings() -> Settings:
    quote_allow = [x.upper() for x in _csv(os.getenv("QUOTE_ALLOWLIST","USD,USDT,USDC"))]
    stable_bases = [x.upper() for x in _csv(os.getenv("STABLECOIN_BASES","USDT,USDC,DAI,TUSD,USDP,GUSD,FRAX,LUSD,EURC,BUSD"))]
    uni_raw = os.getenv("CRYPTO_UNIVERSE")
    uni = [x.upper() for x in _csv(uni_raw)] if uni_raw else None

    enet_c = [float(x) for x in _csv(os.getenv("ENET_C_VALUES","0.5,1.0"))]
    enet_l1 = [float(x) for x in _csv(os.getenv("ENET_L1_VALUES","0.0,0.5"))]
    prior_a = [float(x) for x in _csv(os.getenv("PRIOR_ALPHA_VALUES","0.6,0.7,0.8,0.9"))]

    return Settings(
        horizon_minutes=_int("HORIZON_MINUTES",240),
        horizon_mode=os.getenv("HORIZON_MODE","FIXED").strip().upper(),
        scan_interval_minutes=_int("SCAN_INTERVAL_MINUTES",30),
        min_bars_5m=_int("MIN_BARS_5M",7),
        timezone=os.getenv("TIMEZONE","UTC"),
        benchmark_symbol=os.getenv("BENCHMARK_SYMBOL","BTC-USD").strip().upper(),

        crypto_universe=uni,
        quote_allowlist=quote_allow,
        exclude_stablecoin_base=_bool("EXCLUDE_STABLECOIN_BASE", True),
        stablecoin_bases=stable_bases,
        universe_max=_int("UNIVERSE_MAX",250),
        universe_refresh_minutes=_int("UNIVERSE_REFRESH_MINUTES",360),
        min_24h_dollar_volume=_float("MIN_24H_DOLLAR_VOLUME",0.0),

        coinbase_base_url=os.getenv("COINBASE_BASE_URL","https://api.exchange.coinbase.com").strip(),
        coinbase_max_rps=_float("COINBASE_MAX_RPS",5.0),
        coinbase_max_inflight=_int("COINBASE_MAX_INFLIGHT",5),
        max_candle_staleness_minutes=_int("MAX_CANDLE_STALENESS_MINUTES",60),

        admin_password=os.getenv("ADMIN_PASSWORD"),
        train_lookback_days=_int("TRAIN_LOOKBACK_DAYS",60),
        train_max_products=_int("TRAIN_MAX_PRODUCTS",0),
        calib_min_bucket_samples=_int("CALIB_MIN_BUCKET_SAMPLES",200),
        enet_c_values=enet_c,
        enet_l1_values=enet_l1,
        prior_alpha_values=prior_a,

        model_dir=os.getenv("MODEL_DIR","./runtime/model"),

        demo_mode=_bool("DEMO_MODE",False),
        disable_scheduler=_bool("DISABLE_SCHEDULER",False),
        debug_password=os.getenv("DEBUG_PASSWORD"),

        run_scan_on_startup=_bool("RUN_SCAN_ON_STARTUP", True),
        startup_scan_delay_seconds=_int("STARTUP_SCAN_DELAY_SECONDS", 2),

        auto_train_on_startup=_bool("AUTO_TRAIN_ON_STARTUP", True),
        auto_train_max_products=_int("AUTO_TRAIN_MAX_PRODUCTS", 20),
        auto_train_lookback_days=_int("AUTO_TRAIN_LOOKBACK_DAYS", 14),
        auto_train_cooldown_minutes=_int("AUTO_TRAIN_COOLDOWN_MINUTES", 720),

        tod_rvol_lookback_days=_int("TOD_RVOL_LOOKBACK_DAYS",20),
        tod_rvol_min_days=_int("TOD_RVOL_MIN_DAYS",8),

        liq_rolling_bars=_int("LIQ_ROLLING_BARS",12),
        liq_dvol_min_usd=_float("LIQ_DVOL_MIN_USD",2_000_000.0),
        liq_range_pct_max=_float("LIQ_RANGE_PCT_MAX",0.012),
        liq_wick_atr_max=_float("LIQ_WICK_ATR_MAX",0.8),
    )
