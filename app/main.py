import datetime as dt
import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from .config import load_settings, Settings
from .coinbase_client import CoinbaseClient
from .universe import UniverseManager
from .scheduler import ScanState, scan_once, try_start_scheduler
from .training import TrainingManager, PAUSE_FILE
from .storage import read_json, atomic_write_json, ensure_dir

def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _models_missing(cfg: Settings) -> bool:
    p1 = Path(cfg.model_dir) / "pt1" / "bundle.joblib"
    p2 = Path(cfg.model_dir) / "pt2" / "bundle.joblib"
    return (not p1.exists()) or (not p2.exists())

cfg: Settings = load_settings()
templates = Jinja2Templates(directory="app/templates")

app = FastAPI(title="Coinbase Crypto Prob Scanner (3% / 5%)")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

state = ScanState()
universe_mgr = UniverseManager(cfg)
training_mgr = TrainingManager(cfg)
cb_client: CoinbaseClient | None = None

@app.on_event("startup")
async def startup_event():
    global cfg, universe_mgr, training_mgr, cb_client
    cfg = load_settings()
    universe_mgr = UniverseManager(cfg)
    training_mgr = TrainingManager(cfg)

    cb_client = CoinbaseClient(cfg.coinbase_base_url, cfg.coinbase_max_rps, cfg.coinbase_max_inflight, demo_mode=cfg.demo_mode)
    await cb_client.__aenter__()

    try_start_scheduler(cfg, cb_client, universe_mgr, state)

    async def _warm_start_scan():
        try:
            await asyncio.sleep(max(0, int(cfg.startup_scan_delay_seconds)))
            await scan_once(cfg, cb_client, universe_mgr, state)
        except Exception as e:
            state.last_error = f"startup_scan_error:{type(e).__name__}: {e}"

    if cfg.run_scan_on_startup and (not cfg.disable_scheduler):
        asyncio.create_task(_warm_start_scan())

    async def _auto_train():
        try:
            if not cfg.auto_train_on_startup:
                return
            if not cfg.admin_password:
                return
            if not _models_missing(cfg):
                return

            guard_path = Path(cfg.model_dir) / "auto_train_guard_v2.json"
            now = dt.datetime.now(dt.timezone.utc)
            if guard_path.exists():
                try:
                    import json
                    j = json.loads(guard_path.read_text(encoding="utf-8"))
                    last = dt.datetime.fromisoformat(j.get("last_attempt_utc"))
                    age_min = (now - last).total_seconds()/60.0
                    if age_min < float(cfg.auto_train_cooldown_minutes):
                        return
                except Exception:
                    pass

            await asyncio.sleep(5)
            products, _ = await universe_mgr.resolve_universe(cb_client)

            try:
                import json
                guard_path.parent.mkdir(parents=True, exist_ok=True)
                guard_path.write_text(json.dumps({"last_attempt_utc": now.isoformat()}), encoding="utf-8")
            except Exception:
                pass

            training_mgr.start(
                cb_client,
                products,
                max_products=int(cfg.auto_train_max_products),
                lookback_days=int(cfg.auto_train_lookback_days),
            )
        except Exception as e:
            state.last_error = f"auto_train_error:{type(e).__name__}: {e}"

    asyncio.create_task(_auto_train())

@app.on_event("shutdown")
async def shutdown_event():
    global cb_client
    if cb_client is not None:
        await cb_client.__aexit__(None, None, None)
        cb_client = None

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/api/status")
async def api_status():
    global cb_client, cfg
    cb_status = cb_client.status() if cb_client is not None else None
    pause = (Path(cfg.model_dir) / PAUSE_FILE).exists()
    # Multi-worker safety: read last scan snapshot from disk (written by the scheduler-holding worker)
    meta_path = Path(cfg.model_dir) / "last_scores_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else None
    disk_coverage = (meta.get("coverage") if isinstance(meta, dict) else None) or None
    disk_model = (meta.get("model") if isinstance(meta, dict) else None) or None
    if (not state.coverage) and disk_coverage:
        state.coverage = disk_coverage
    if (not state.model_notes) and disk_model:
        state.model_notes = disk_model
    # Training status persisted by training worker
    tpath = Path(cfg.model_dir) / "training_status.json"
    tstat = read_json(tpath) if tpath.exists() else None
    training_view = tstat if isinstance(tstat, dict) else training_mgr.status

    return {
        "now_utc": _now_utc(),
        "config": {
            "scan_interval_minutes": cfg.scan_interval_minutes,
            "horizon_minutes": cfg.horizon_minutes,
            "horizon_mode": cfg.horizon_mode,
            "benchmark_symbol": cfg.benchmark_symbol,
            "targets": {"prob_1": "3%", "prob_2": "5%"},
            "quote_allowlist": cfg.quote_allowlist,
            "exclude_stablecoin_base": cfg.exclude_stablecoin_base,
            "universe_max": cfg.universe_max,
            "universe_refresh_minutes": cfg.universe_refresh_minutes,
            "max_candle_staleness_minutes": cfg.max_candle_staleness_minutes,
            "demo_mode": cfg.demo_mode,
            "disable_scheduler": cfg.disable_scheduler,
            "coinbase_max_rps": cfg.coinbase_max_rps,
            "coinbase_max_inflight": cfg.coinbase_max_inflight,
            "model_dir": cfg.model_dir,
            "auto_train": {
                "enabled": cfg.auto_train_on_startup,
                "max_products": cfg.auto_train_max_products,
                "lookback_days": cfg.auto_train_lookback_days,
                "cooldown_minutes": cfg.auto_train_cooldown_minutes,
            },
        },
        "coinbase": {
            "ok": cb_status.ok if cb_status else False,
            "message": cb_status.message if cb_status else "not_initialized",
            "base_url": cfg.coinbase_base_url,
            "last_request_utc": cb_status.last_request_utc if cb_status else None,
            "last_error": cb_status.last_error if cb_status else None,
            "rate_limit": {
                "last_429_utc": cb_client.state.last_429_utc if cb_client else None,
                "last_backoff_seconds": cb_client.state.last_backoff_seconds if cb_client else None,
                "backoff_count": cb_client.state.backoff_count if cb_client else 0,
                "recent_errors": cb_client.state.recent_errors if cb_client else 0,
            },
        },
        "universe": (state.coverage.get("universe_meta") if state.coverage else None),
        "model": state.model_notes,
        "training": training_view,
        "scan": {"running": state.scan_running, "paused_for_training": pause},
        "coverage": state.coverage or {
            "universe_count": 0,
            "products_requested_count": 0,
            "products_returned_with_candles_count": 0,
            "products_with_sufficient_candles_count": 0,
            "products_scored_count": 0,
            "top_skip_reasons": {},
            "last_run_utc": state.last_run_utc,
            "last_candle_timestamp": state.last_candle_timestamp,
        },
        "last_error": state.last_error,
    }

@app.get("/api/scores")
async def api_scores():
    # Multi-worker safety: if this worker has no in-memory rows, read from disk snapshot
    if not state.rows:
        spath = Path(cfg.model_dir) / "last_scores.json"
        snap = read_json(spath) if spath.exists() else None
        if isinstance(snap, dict) and isinstance(snap.get("rows"), list):
            return {"now_utc": _now_utc(), "last_run_utc": snap.get("last_run_utc"), "rows": snap.get("rows")}
    return {"now_utc": _now_utc(), "last_run_utc": state.last_run_utc, "rows": state.rows or []}

@app.post("/train")
async def train(body: Dict[str, Any]):
    global cb_client, cfg
    if cfg.admin_password is None or cfg.admin_password.strip() == "":
        raise HTTPException(status_code=400, detail="ADMIN_PASSWORD not set; cannot train.")
    pw = str(body.get("password","")).strip()
    if pw != cfg.admin_password and (cfg.debug_password is None or pw != cfg.debug_password):
        raise HTTPException(status_code=401, detail="Invalid password.")
    if cb_client is None:
        raise HTTPException(status_code=500, detail="Coinbase client not initialized.")
    products, meta = await universe_mgr.resolve_universe(cb_client)
    if training_mgr.is_running():
        return {"ok": True, "message": "training already running", "status": training_mgr.status}
    training_mgr.start(cb_client, products)
    return {"ok": True, "message": "training started", "status": training_mgr.status, "universe_meta": meta}

@app.get("/api/training/status")
async def training_status():
    tpath = Path(cfg.model_dir) / "training_status.json"
    tstat = read_json(tpath) if tpath.exists() else None
    return tstat if isinstance(tstat, dict) else training_mgr.status

@app.get("/api/debug/coverage")
async def debug_coverage(password: str):
    ok_pw = (cfg.debug_password and password == cfg.debug_password) or (cfg.admin_password and password == cfg.admin_password)
    if not ok_pw:
        raise HTTPException(status_code=401, detail="Invalid password.")
    return {"now_utc": _now_utc(), "last_run_utc": state.last_run_utc, "skipped": state.debug_skipped(limit=200)}
