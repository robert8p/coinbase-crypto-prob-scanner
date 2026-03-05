# Coinbase Crypto Prob Scanner (3% / 5%)

FastAPI web app that scans Coinbase Exchange spot products every **30 minutes** (aligned to **:00 / :30 UTC**) and shows:

- **Prob 3%**: probability the asset reaches **+3%** from scan time within the forward horizon
- **Prob 5%**: probability the asset reaches **+5%** from scan time within the forward horizon

No trade execution. No backtesting. Scanner + training only.

## Probability definition (exact)
For each product at scan time:
- `P0` = current price at scan time (latest completed 5m candle close aligned to the scan)
- `H_future` = maximum future **1-minute HIGH** from scan time until `(scan time + HORIZON_MINUTES)`
- `Y=1` if `H_future >= (1+X)*P0` else `0`, where `X` is `0.03` or `0.05`

## Render quickstart
- Docker Web Service
- Health check path: `/health`
- Persistent disk mounted at `/var/data`
- Set `MODEL_DIR=/var/data/model`

### Debug bundle
```
DEMO_MODE=true
DISABLE_SCHEDULER=1
ADMIN_PASSWORD=change-me-now
MODEL_DIR=/var/data/model
```

### Live bundle
```
DEMO_MODE=false
DISABLE_SCHEDULER=0
ADMIN_PASSWORD=change-me-now
MODEL_DIR=/var/data/model

SCAN_INTERVAL_MINUTES=30
HORIZON_MINUTES=240
HORIZON_MODE=FIXED

QUOTE_ALLOWLIST=USD,USDT,USDC
EXCLUDE_STABLECOIN_BASE=true
UNIVERSE_MAX=250
UNIVERSE_REFRESH_MINUTES=360

COINBASE_MAX_RPS=5
COINBASE_MAX_INFLIGHT=5
MAX_CANDLE_STALENESS_MINUTES=60
```

### Optional warm-start (prevents initial zero coverage until first :00/:30 tick)
- `RUN_SCAN_ON_STARTUP` (default `true`)
- `STARTUP_SCAN_DELAY_SECONDS` (default `2`)
