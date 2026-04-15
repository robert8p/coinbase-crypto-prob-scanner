# Coinbase Crypto Prob Scanner v4.20.6

A FastAPI decision-support scanner for Coinbase-tradeable crypto. It identifies names with the strongest evidence of achieving a **quality +2.0% move within 240 minutes**, applies live-policy controls for regime stress, cooldowns, and actionability, then persists evidence packs for honest post-hoc review. It also exposes an automated Control Ledger factual export and ledger-input pack so a separate review/governor/build chat system can ingest current app truth without guessing.

**This is a research/advisory scanner, not a trade execution system.** It produces a shortlist of candidates for human review. It does not place orders.

## What the app does

Every 15 minutes (configurable), the scanner:

1. Checks Coinbase health and builds a filtered universe of tradeable products.
2. Performs a light **Stage 1** fetch across the universe to compute features and rank candidates.
3. Deep-fetches the top **Stage 2** candidates with richer history.
4. Scores each candidate using a trained model (or heuristic fallback if untrained).
5. Applies post-model adjustments: guardrail caps, panic penalties, Binance divergence penalties, sector contagion penalties.
6. Applies regime policy: market regime state (green/amber/red) × liquidity tier (tier1/tier2/tier3) determines factor, cap, and threshold.
7. Separates rows into visible, suppressed (regime/cooldown/threshold), informational, and overflow buckets.
8. Persists a review pack (SQLite + ZIP artifact) for every scan.
9. After the 4-hour horizon elapses, resolves each row against actual outcomes.
10. Aggregates version-level evidence summaries for deployment-window analysis.

## Scoring pipeline

A candidate's score passes through these stages (each can only reduce the score):

```
raw model output  →  guardrail cap (0.65)  →  panic penalty (−0.10)
  →  sector penalty (≤0.10)  →  Binance gap penalty (≤0.08)
  →  Binance lead penalty (≤0.06)  →  regime factor (×0.60–1.0)
  →  regime cap (0.00–0.95)  →  live_score
```

The **live_score** is what the operator sees. Rows must clear the regime-specific live threshold to be visible.

### Score semantics

The app distinguishes:

| Field | Meaning |
|-------|---------|
| `prob_2_model` | Raw calibrated model probability |
| `pre_policy_score` | After post-model adjustments, before regime policy |
| `live_score` | After all adjustments — what the operator sees |
| `validated_floor` | Whether this score is in a statistically validated band |
| `actionability_tier` | action_ready / selective / watchlist — based on validation + temporal support |

**Validated tail**: if holdout metrics show precision ≥ event_rate × 1.1 with ≥25 samples and Wilson lower bound passes, scores in that band are "validated" and treated as probabilities rather than ranking hints.

## Architecture

### Deployment

Single Docker container on Render. One Uvicorn worker. Background threads for the scan scheduler, review evaluator, and cooldown follow-up recovery run inside the same web process.

```
┌──────────────────────────────────────┐
│  FastAPI web process (single worker) │
│                                      │
│  ├─ HTTP routes (main.py)            │
│  ├─ Scan scheduler thread            │
│  ├─ Review evaluator thread          │
│  └─ Follow-up scheduler threads      │
│                                      │
│  Persistent disk: /var/data          │
│  ├─ model/pt2 (trained model)        │
│  ├─ model/review_packs.db (SQLite)   │
│  └─ model/paper_trade_log.jsonl      │
└──────────────────────────────────────┘
```

### Backend modules

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `scanner.py` | ~2,800 | Primary scan orchestration, scoring, follow-up campaigns |
| `review_runs.py` | ~1,300 | Review pack persistence, outcome resolution, evidence aggregation |
| `modeling.py` | ~1,200 | Model bundle, score contracts, tail trust, training metrics |
| `paper_trade.py` | ~750 | Forward-validation persistence and summary |
| `features.py` | ~750 | Feature engineering, stage-1 ranking, guardrails |
| `main.py` | ~540 | FastAPI routes, status shaping, app lifespan |
| `regime.py` | ~490 | Market regime engine, liquidity tiers, live policy lookup |
| `state.py` | ~470 | Mutable in-memory application state |
| `config.py` | ~240 | Environment-driven configuration (130+ knobs) |
| `universe.py` | ~220 | Eligible/selected product filtering |
| `coinbase_client.py` | ~350 | Coinbase market data adapter |
| `live_scoring.py` | ~110 | Post-model penalty calculations |
| `binance_client.py` | ~140 | Cross-exchange signal fetching |

### Frontend

Single-page operator UI: `templates/index.html` + `static/app.js`. Server renders shell via Jinja2, JavaScript fetches JSON endpoints to populate data panels. No frontend framework.

### Data flow

```
Coinbase REST API
    ↓
Universe builder → Stage 1 fetch → Feature computation → Stage 1 rank
    ↓
Stage 2 deep fetch → Model scoring → Post-model adjustments
    ↓
Regime policy → Threshold plan → Visibility classification
    ↓
┌──────────────┬──────────────┬─────────────────┐
│ Visible rows │ Suppressed   │ Informational   │
│ (actionable) │ (blocked)    │ (overflow)      │
└──────┬───────┴──────┬───────┴────────┬────────┘
       ↓              ↓                ↓
   Homepage UI    Review pack DB   Evidence chain
       ↓              ↓
   JSON API       Outcome resolution (after 4h)
                      ↓
              Current-version summary
```

## Market regime engine

The regime engine scores market stress from BTC/ETH price moves, volatility ratios, and market breadth.

| State | Meaning | Effect |
|-------|---------|--------|
| **green** | Normal conditions | Scores pass through; thresholds at 0 |
| **amber** | Elevated stress | Factors 0.60–0.85; caps 0.72–0.88; thresholds 0.74–0.82 |
| **red** | High stress / shock | Factors 0.00–0.65; tier3 fully suppressed |

Within each regime state, three **liquidity tiers** receive different treatment:

- **Tier 1**: BTC-USD, ETH-USD
- **Tier 2**: Named liquid majors (SOL, XRP, LINK, etc.) or >$5M 24h volume
- **Tier 3**: Everything else

After a regime transitions from red/amber back to green, a **cooldown period** suppresses new entries for lower-tier assets.

## Configuration

All configuration is via environment variables. Key groups:

**Runtime**: `DEMO_MODE`, `DISABLE_SCHEDULER`, `SCAN_INTERVAL_MINUTES`, `LOG_LEVEL`

**Target**: `TARGET_MOVE_PCT` (0.02), `TARGET_HORIZON_MINUTES` (240), `QUALITY_MAX_MAE` (-0.020), `QUALITY_MIN_END_RET` (-0.008)

**Regime policy**: `MARKET_REGIME_{state}_{tier}_{param}` where state ∈ {GREEN, AMBER, RED}, tier ∈ {TIER1, TIER2, TIER3}, param ∈ {FACTOR, CAP, THRESHOLD, SUPPRESS}

**Guardrails**: `DOWNSIDE_CAP` (0.78), `UNCERTAINTY_CAP` (0.72), `BTC_PANIC_THRESHOLD` (-0.025), `PANIC_THRESHOLD_BOOST` (0.10)

**Validation**: `TAIL_VALIDATION_MIN_COUNT` (25), `TAIL_VALIDATION_MIN_WILSON_LIFT` (1.10), `TAIL_UNVALIDATED_CAP` (0.65)

See `.env.example` for the complete list with defaults.

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Operator dashboard (HTML) |
| `/health` | GET | Health check with model/scan/regime status |
| `/api/status` | GET | Full system status including regime, coverage, score contract |
| `/api/scores` | GET | Current visible scores (optional `?actionable_only=true`) |
| `/api/scores/informational` | GET | Suppressed/informational rows |
| `/api/debug/score-contract` | GET | Current score contract details |
| `/api/debug/market-regime` | GET | Regime state, metrics, reasons |
| `/api/policy-audit` | GET | Policy audit: suppression patterns, false-positive/negative rates |
| `/api/paper-trade/summary` | GET | Forward-validation summary |
| `/api/reviews/runs` | GET | Recent review pack runs |
| `/api/reviews/current-version-summary` | GET | Deployment-window evidence summary |
| `/api/control-ledger/facts` | GET | Auto-generated factual Control Ledger JSON |
| `/api/control-ledger/facts.txt` | GET | Auto-generated factual Control Ledger TXT |
| `/api/control-ledger/release-manifest` | GET | Current tranche/release manifest JSON |
| `/api/control-ledger/ledger-input-pack.zip` | GET | Packaged factual inputs for the four-chat workflow |
| `/api/reviews/post-maturity-bundle.zip` | GET | Latest automated post-maturity evidence bundle |
| `/api/reviews/diagnostic-battery` | GET | Latest automated diagnostic verdict JSON |
| `/api/reviews/diagnostic-battery.txt` | GET | Latest automated diagnostic verdict TXT |
| `/api/automation/status` | GET | Control-plane automation status |
| `/api/training/orchestration/status` | GET | Training orchestration status (no auto-promotion) |
| `/api/model-output-distribution` | GET | Rolling raw model-output distribution diagnostic summary |
| `/api/reviews/current-version.zip` | GET | Downloadable evidence bundle |
| `/api/replay/run` | POST | Historical live-emulation replay run (admin) |
| `/api/replay/latest-summary` | GET | Latest replay summary (admin) |
| `/api/replay/latest.zip` | GET | Latest replay evidence pack (admin) |
| `/train` | POST | Trigger model retraining (requires admin password) |

## Running locally

```bash
# Clone and install
pip install -r requirements.txt

# Demo mode (no live API calls, synthetic data)
DEMO_MODE=true python -m uvicorn app.main:app --port 8000

# Live mode
DEMO_MODE=false MODEL_DIR=/var/data/model python -m uvicorn app.main:app --port 8000
```

After startup, the factual Control Ledger exports are available at `/api/control-ledger/facts`, `/api/control-ledger/facts.txt`, and `/api/control-ledger/ledger-input-pack.zip`.

## Docker

```bash
docker build -t crypto-scanner .
docker run -p 8000:8000 -v /var/data:/var/data -e DEMO_MODE=false crypto-scanner
```

## Running tests

```bash
# Copy tests/ and pytest.ini into the project root
pytest
```

## Known issues (v4.6.3)

**Score-range starvation**: The guardrail cap (0.65) is structurally below the validated floor (0.60), meaning any capped row can never reach the validated band. Combined with additive penalty stacking and regime multiplicative haircuts, live scores rarely approach the bands where the model has proven precision. See `SCORE_RANGE_DIAGNOSIS.md` for the full analysis and recommended fixes.

**Module size**: `scanner.py` (2,800 lines) and `review_runs.py` (1,300 lines) are too large. A refactoring plan to split them into focused sub-packages is provided in `REFACTORING_PLAN.md`.

**No automated test suite**: The project previously relied on `validate_repo.py` for smoke testing. A pytest-based suite is now provided covering config, scoring, regime policy, and pipeline behavior.

**Single-process architecture**: Scan orchestration, review resolution, and web serving share one process. This is acceptable for the current scale but limits reliability.

**Config sprawl**: 130+ environment variables with no categorization or deprecation policy. The 45 regime policy fields alone encode a 3D matrix (state × tier × param) as flat names.

## Changelog

See `CHANGELOG.md` for the full version history.


## Historical replay

The v4.8.1 replay engine reuses the current trained cohort, model, stage1/stage2 logic, regime engine, and visibility rules to emulate historical scans on Coinbase 5-minute data. It is designed to answer two questions faster than waiting for live evaluated packs alone:

- do surfaced rows beat the hidden remainder?
- is stage1 missing too many later-good opportunities?

Replay outputs are written to a downloadable ZIP via `/api/replay/latest.zip` and include surfaced-row evidence, a counterfactual stage1 opportunity audit, threshold-band summaries, and concentration/repeatability summaries.

The v4.8.0 app defaults stage1 back to **primary_only** selection and adds replay-side promotion audits so alternate stage1 widening rules can be judged before changing live selection again. Hybrid stage1 modes remain available for controlled experiments.

## Stage1 opportunity scorer

The v4.9.0 app adds a dedicated **Stage1 Opportunity Scorer** built from replay-labeled stage1 rows. It is designed to improve upstream stage1 ranking by learning which stage1 feature profiles are more likely to become later **quality** opportunities, using the replay counterfactual rows as training data.

Important constraints:

- it is a **stage1 opportunity-ranking aid**, not a calibrated live probability model
- it should be judged by replay evidence, not by how sophisticated it sounds
- live stage1 remains **`primary_only` by default** until scorer-based selection modes prove they outperform the baseline

You can build the scorer from the latest replay pack through either:

- the homepage **Stage1 Opportunity Scorer** panel
- `POST /api/stage1-opportunity/build-from-replay`

And inspect the latest saved scorer summary through either:

- the homepage summary loader
- `GET /api/stage1-opportunity/summary`


## Replay ablation and model audit

The v4.9.1 tranche adds two more decision-critical evaluation tools:

- **Replay pipeline ablation**
  - replay runs now accept `pipeline_mode=full` or `pipeline_mode=raw_threshold`
  - the replay summary includes a pipeline-ablation comparison so you can see whether the simpler raw-threshold path outperforms or underperforms the full post-model pipeline
  - the homepage replay controls expose both the pipeline mode and the raw-threshold input

- **Replay-based model audit**
  - builds from the latest replay pack
  - evaluates `prob_2_model` ordering quality using resolved replay rows
  - reports AUC, Brier score, calibration deciles, tail precision, and per-symbol summaries
  - available through:
    - the homepage **Model Audit** panel
    - `POST /api/model-audit/build-from-replay`
    - `GET /api/model-audit/summary`

These tools are meant to confront a hard question directly: is the current model/pipeline actually helping the scanner surface a materially better shortlist, or just adding complexity around weak signal?


## Live pipeline simplification

The v4.10.0 tranche responds directly to replay evidence that the full live pipeline was hurting shortlist quality. The app now defaults to a simpler live path:

- `LIVE_PIPELINE_MODE=raw_threshold`
- `LIVE_RAW_THRESHOLD=0.30`

In that default mode, the live shortlist is driven primarily by raw model ordering plus the fixed raw threshold, instead of the full regime/cooldown/post-model suppression stack. The older full pipeline is still available for controlled comparison, but it is no longer the default path.

This is an explicit simplification tranche: the goal is to test whether a less-governed shortlist is more useful than an over-constrained one.
