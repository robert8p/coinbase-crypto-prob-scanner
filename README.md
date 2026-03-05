# Coinbase Crypto Prob Scanner (3% / 5%)

FastAPI web app that scans Coinbase Exchange spot products every **30 minutes** (aligned to **:00 / :30 UTC**) and shows:

- **Prob 3%**: probability the asset reaches **+3%** from scan time within the forward horizon
- **Prob 5%**: probability the asset reaches **+5%** from scan time within the forward horizon

No trade execution. No backtesting. Scanner + training only.

## Key “no-manual-steps” behavior (this build)
- If `pt1/pt2` models are missing **and** `ADMIN_PASSWORD` is set, the app will **auto-train once on startup** (rate-limit safe defaults).
- While training is running, live scanning is **automatically paused** (no need to flip `DISABLE_SCHEDULER`).
- After training completes, the app automatically resumes scanning and switches to `using: model`.

You can still start training manually from the UI, but you do **not** need to toggle any env vars.

## Probability definition (exact)
For each product at scan time:
- `P0` = current price at scan time (latest completed 5m candle close aligned to the scan)
- `H_future` = maximum future **1-minute HIGH** from scan time until `(scan time + HORIZON_MINUTES)`
- `Y=1` if `H_future >= (1+X)*P0` else `0`, where `X` is `0.03` or `0.05`

## Deploy on Render (single Web Service)
- Docker Web Service
- Health check path: `/health`
- Persistent disk mounted at `/var/data`
- Set `MODEL_DIR=/var/data/model`

### Minimal env vars (recommended)
```
ADMIN_PASSWORD=change-me-now
MODEL_DIR=/var/data/model

DEMO_MODE=false
DISABLE_SCHEDULER=0

SCAN_INTERVAL_MINUTES=30
HORIZON_MINUTES=240
HORIZON_MODE=FIXED

QUOTE_ALLOWLIST=USD,USDT,USDC
EXCLUDE_STABLECOIN_BASE=true
UNIVERSE_MAX=0  # 0 = all eligible products (no cap)
UNIVERSE_REFRESH_MINUTES=360

COINBASE_MAX_RPS=5
COINBASE_MAX_INFLIGHT=5
MAX_CANDLE_STALENESS_MINUTES=60
```

### Optional warm-start (prevents initial zero coverage until first :00/:30 tick)
- `RUN_SCAN_ON_STARTUP` (default `true`)
- `STARTUP_SCAN_DELAY_SECONDS` (default `2`)

### Optional auto-train controls (defaults are safe)
- `AUTO_TRAIN_ON_STARTUP` (default `true`)
- `AUTO_TRAIN_MAX_PRODUCTS` (default `20`)
- `AUTO_TRAIN_LOOKBACK_DAYS` (default `14`)
- `AUTO_TRAIN_COOLDOWN_MINUTES` (default `60`)  # prevents repeated retries


## Render health-check safety
CPU-heavy model fitting runs in a worker thread so `/health` stays responsive during training.


## Universe size
- Set `UNIVERSE_MAX=0` to scan **all** eligible products (no cap).


## Training size
- `TRAIN_MAX_PRODUCTS=0` means **no cap** (train on all resolved products).
- `AUTO_TRAIN_MAX_PRODUCTS` controls only the startup auto-train (default 20).


## Render uptime during training
This build defaults to **2 uvicorn workers** (set `UVICORN_WORKERS` to change). One worker can stay responsive for `/health` while the other performs CPU-heavy training. Scheduler and training are protected with disk locks so only one runs.
