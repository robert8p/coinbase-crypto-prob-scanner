# Coinbase Crypto Prob Scanner (3% / 5%)

Production-ready **FastAPI** web app that scans Coinbase Exchange spot products every **30 minutes** (aligned to **:00 / :30 UTC**) and displays:

- **Prob 3%**: probability the asset reaches **+3%** from scan time within the forward horizon
- **Prob 5%**: probability the asset reaches **+5%** from scan time within the forward horizon

**No trade execution. No backtesting.** Scanner + training only.

---

## Probability definition (exact)

For each product at scan time:

- `P0` = current price at scan time (**latest completed 5m candle close** aligned to the scan)
- `H_future` = maximum future **1-minute HIGH** from scan time until `(scan time + HORIZON_MINUTES)`
- `Y = 1` if `H_future >= (1+X)*P0` else `0`, where `X` is **0.03** or **0.05**

Displayed values are **true probabilities (0–1)**, not ranks.

---

## Deploy on Render (Step-by-Step)

### 1) GitHub (no CLI)

1. Download the zip and unzip it on your computer.
2. Create a new GitHub repository (private recommended).
3. GitHub web UI → **Add files → Upload files**.
4. Upload the **contents** of the unzipped folder (repo root should contain `app/`, `Dockerfile`, `requirements.txt`, `README.md`).
5. Commit changes.

### 2) Render (single Web Service)

1. Render → **New → Web Service** → connect your GitHub repo
2. Runtime: **Docker**
3. Health Check Path: `/health`
4. Add **Persistent Disk**:
   - Mount path: `/var/data`
5. Add env var:
   - `MODEL_DIR=/var/data/model`
6. The Docker command runs **one worker**: `uvicorn ... --workers 1` (required so the scheduler is single-instance).

### 3) Phase 1: Debug-first deployment

Set these env vars first:

**Debug bundle (copy/paste):**
```
DEMO_MODE=true
DISABLE_SCHEDULER=1
ADMIN_PASSWORD=change-me-now
MODEL_DIR=/var/data/model
```

What “good” looks like:

- `/health` → `200 OK`
- `/api/status` → JSON with keys: `coinbase`, `model`, `training`, `coverage`, `config`
- `/api/scores` → JSON (rows may be empty, but must not error)
- `/` → dashboard loads (no 500)

### 4) Phase 2: Live scanning

Switch:
- `DEMO_MODE=false`
- `DISABLE_SCHEDULER=0`

**Live bundle (copy/paste):**
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

Confirm scheduler alignment:
- Status panel `last_run_utc` updates on `:00` and `:30` UTC (or every `SCAN_INTERVAL_MINUTES`).
- Coverage line shows “Scored X / Universe Y”. If X << Y, see skip reasons.

### 5) Coverage diagnostics

Dashboard shows skip reason counts when coverage is low.
Protected endpoint (max 200 rows):

`/api/debug/coverage?password=<ADMIN_PASSWORD>`

### 6) Training (one-click)

1. Open `/` (dashboard).
2. Enter `ADMIN_PASSWORD` in Training.
3. Click **Start training (3% & 5%)**.
4. UI polls `/api/training/status`.

Artifacts written under `MODEL_DIR` (persistent disk on Render):

- `MODEL_DIR/pt1/bundle.joblib`  (trained for **+3%**)
- `MODEL_DIR/pt2/bundle.joblib`  (trained for **+5%**)
- `MODEL_DIR/volume_profiles/*.json`
- `MODEL_DIR/cache/...`

### 7) Rate limit safety

If `/api/status` shows rate limiting / backoffs:
- Reduce `UNIVERSE_MAX` (e.g., 250 → 150)
- Lower `COINBASE_MAX_RPS` (e.g., 5 → 3)
- Lower `COINBASE_MAX_INFLIGHT` (e.g., 5 → 2)

---

## Environment variables (all env-only)

### Core
- `HORIZON_MINUTES` (default `240`)
- `HORIZON_MODE` (default `FIXED`) values: `FIXED`, `UTC_DAY_END`
- `SCAN_INTERVAL_MINUTES` (default `30`)
- `MIN_BARS_5M` (default `7`)
- `TIMEZONE` (default `UTC`)
- `BENCHMARK_SYMBOL` (default `BTC-USD`)

### Universe
- `CRYPTO_UNIVERSE` (optional explicit list: `BTC-USD,ETH-USD,...`)
- `QUOTE_ALLOWLIST` (default `USD,USDT,USDC`)
- `EXCLUDE_STABLECOIN_BASE` (default `true`)
- `STABLECOIN_BASES` (default `USDT,USDC,DAI,TUSD,USDP,GUSD,FRAX,LUSD,EURC,BUSD`)
- `UNIVERSE_MAX` (default `250`)
- `UNIVERSE_REFRESH_MINUTES` (default `360`)
- `MIN_24H_DOLLAR_VOLUME` (default `0`)

### Coinbase rate limiting
- `COINBASE_BASE_URL` (default `https://api.exchange.coinbase.com`)
- `COINBASE_MAX_RPS` (default `5`)
- `COINBASE_MAX_INFLIGHT` (default `5`)
- `MAX_CANDLE_STALENESS_MINUTES` (default `60`)

### Training
- `ADMIN_PASSWORD` (required to train)
- `TRAIN_LOOKBACK_DAYS` (default `60`)
- `TRAIN_MAX_PRODUCTS` (default `0` means “all in universe”; for practical runtimes you may set e.g. `50`)
- `CALIB_MIN_BUCKET_SAMPLES` (default `200`)
- `ENET_C_VALUES` (default `0.5,1.0`)
- `ENET_L1_VALUES` (default `0.0,0.5`)
- `PRIOR_ALPHA_VALUES` (default `0.6,0.7,0.8,0.9`)

### Storage
- `MODEL_DIR` (default `./runtime/model`)

### Debug
- `DEMO_MODE` (default `false`)
- `DISABLE_SCHEDULER` (default `0`)
- `DEBUG_PASSWORD` (optional; also accepted by `/api/debug/coverage`)
