# Changelog

## v4.20.6
- Added control-ledger automation endpoints: `/api/control-ledger/facts`, `/api/control-ledger/facts.txt`, `/api/control-ledger/release-manifest`, `/api/control-ledger/release-manifest.txt`, and `/api/control-ledger/ledger-input-pack.zip`.
- Added a generated factual Control Ledger export and downloadable ledger-input pack so the four-chat operating system can ingest a clean factual base without auto-writing strategic judgment.
- Surfaced control-ledger downloads in `/health`, `/api/status`, `/api/reviews/current-version-summary`, and on the homepage operator-share buttons.
- Added repo-root `release_manifest.json` to carry tranche-level release metadata for the build/review/governor loop.

## 4.20.3 - 2026-04-15
- Fixed a scan-failing NameError in blocked monitoring context wiring introduced in 4.20.2.
- Threaded effective regime actionability into blocked monitoring context generation.
- Suppressed stale prior-version model-output distribution payloads from current-version summary endpoints.


## v4.20.2
- Fixed raw-threshold effective regime actionability propagation into row assessment and summaries.
- Added objective decision-support card to the current-version UI.
- Updated decision-summary headlines to foreground confirmed/strong/priority/elite objective bands when present.

## v4.20.0
- Objective-aligned score semantics calibration tranche.
- Added confirmed-shortlist / strong-edge / priority-edge / elite-edge bands from raw baseline evidence.
- Preserved live selection logic while fixing overly pessimistic row semantics and summaries.
- Review-pack CSVs now include objective semantics fields.

## v4.19.0

- Pivot away from another shortlist-policy loop and add a replay-backed Raw Score Baseline Lab.
- Capture every resolved stage-2 rankable replay row before display trimming so raw-model ranking can be judged directly.
- Report raw-model score quantiles, dead-upper-tail verdict, top-1%/5%/10% quality, per-scan top-1/top-3/top-5 quality, base rate, and compression diagnostics.
- Add homepage controls plus summary TXT and pack ZIP outputs for the new baseline evidence.
- Keep live path unchanged.

## v4.18.1

- narrowed controlled shadow comparison to the single supported offline winner instead of a broad top-N challenger set
- added explicit live-path-unchanged and primary-shadow-candidate metadata to shadow comparison and outcome review summaries
- kept legacy live behavior unchanged while making the supported winner the clean shadow candidate for proof gathering

## v4.18.0
- Expand Utility Policy Search into a broader offline challenger-family search over materially different policy families.
- Add family-level results, supported-family tracking, and richer offline-only decision memo output.
- Add pool-construction and regime-cap variants so unsupported families can fail offline before any future live/shadow testing.

## v4.17.3
- Enforced the rule that live/shadow testing is blocked unless a challenger has already beaten legacy on historic/offline data.
- Shadow Selection Comparison now returns `blocked_offline_gate` instead of recording unsupported challengers.
- Shadow Selection Outcome Review now returns `blocked_offline_gate` when no offline-supported challenger exists.

## v4.17.2
- Fix active concurrent shadow challenger maturity tracking and outcome review state reporting.

## v4.16.7
- Fixed shadow comparison live incumbent detection to use resolved runtime/config selection state, preventing false `live_engine_not_legacy` skips when legacy is live.

## v4.15.1
- Added Utility Policy Search status endpoint, background start, polling UI, progress bar, and ready/error states.
- Added utility policy search status TXT download and automatic status refresh on page load.

## v4.15.0
- Added utility policy search lab to rank multiple shortlist challengers against legacy on the same replay frame.
- Added UI controls and download endpoints for policy-search summary TXT and pack ZIP.
- This tranche is intended to stop one-off heuristic thrash and identify the single best next live shortlist candidate.

## v4.14.3
- Restore utility selection behavior closer to v4.13.8 while preserving slotted-config compatibility.
- Keep Health TXT download support.

## v4.14.1
- added /health.txt download endpoint and homepage button
- widened green-scan companion logic in utility_constrained_v5 to improve pairwise scan wins without broadening stressed-regime lists

## v4.13.9
- Fix utility_tuning_lab runtime error: vars() argument must have __dict__ attribute.
- Make utility config merging compatible with slotted dataclass configs.

## v4.13.8 — adaptive utility shortlist engine

- Replaced hard scan-readiness suppression with adaptive strong/moderate/weak scan modes.
- Allows small trusted shortlists on moderate scans while preserving no-shortlist behavior on blocked scans.
- Utility shortlist engine now identifies as `utility_constrained_v3`.

## v4.13.7 — utility selection engine rebuild

- Rebuilt utility shortlist gating around scan readiness and dominance margin.
- Weak scans now default to no shortlist instead of weak forced singleton picks.
- Strong close top candidates can now expand the shortlist to 2–3 names to improve scan-level pairwise wins.

## v4.13.5 — autopilot first-run unblock

- utility selection lab now tolerates missing current-version review packs on a fresh deploy.
- autopilot now checks the correct utility-selection pass verdict.
- automation summary messaging is clearer for blocked/error states.

## v4.13.4 — staged autopilot offline gating

- Utility autopilot now runs offline labs in stages and pins each fresh session snapshot immediately.
- It stops at the first failed utility-selection gate instead of running deeper branches unnecessarily.
- Fixed utility_model_lab invocation in autopilot.

## v4.13.3 — strict autopilot freshness enforcement

- Added session-pinned offline lab snapshots for automation.
- Redacted stale latest summaries from automation status until freshness is proven.
- Tightened the first gate so automation proceeds only on an explicit utility-engine offline win.

## v4.13.2 — automation freshness and first-gate hard stop

- Utility operator automation now requires session-fresh offline lab summaries before branch selection.
- Added a hard stop when utility selection returns legacy_engine_preferred_offline.
- Automation UI now shows freshness flags and the selection-gate verdict.

## v4.13.1 — utility operator automation frontend hotfix

- Added missing escapeHtml helper used by the Utility Operator Automation renderer and other summary panels.
- Fixed homepage runtime failure when starting Utility Autopilot.

## v4.13.0 — one-click utility operator automation

- Added a background utility operator automation state machine to run the current gated utility workflow end to end.
- Added utility-model and utility-tuning adoption review backends so automation can reach keep-versus-rollback conclusions instead of stopping at adoption.
- Added one-click UI controls and automation status/pack endpoints.

## v4.12.9 — multipart startup hotfix

- Added `python-multipart` to requirements so FastAPI `Form(...)` routes can load at startup.
- No scoring, shortlist, proof, adoption, or model logic changed.

## v4.12.8 — controlled utility model adoption gate

- Added evidence-gated adoption for the utility-model challenger with rollback preserved.
- Scanner and shortlist override layers now recognize utility-model adoption overrides.
- Added summary/TXT/pack endpoints and UI controls for controlled utility-model adoption.

## v4.12.7 — controlled utility-model proof

- Added bounded live proof and isolated proof review for the utility-model challenger.
- Runtime overrides can now carry the exact challenger model bundle while preserving current utility shortlist semantics.
- Scanner status now records distinct `live_utility_model_proof` metadata for review-pack attribution.

## v4.12.6 — utility model challenger lab

- Added an offline utility-aligned model challenger lab.
- Compares incumbent event-model ranking against a first utility-trained regressor on the same holdout frame.
- Saves the challenger artifact for future proof work.

## v4.12.5 — controlled utility adoption review

- Added isolated adoption-window review so the adopted tuned utility path can produce a keep-versus-rollback verdict.
- Scanner status now records distinct `live_utility_tuning_adoption` metadata for review-pack attribution.
- Clearing controlled utility adoption now preserves session context for later review.

## v4.12.4 — controlled utility tuning adoption gate

- Added a controlled adoption gate for the tuned utility bundle after isolated live proof success.
- Scanner now supports both proof-window overrides and adopted-bundle overrides for utility settings.
- Added rollback-aware activation and clear endpoints for the tuned utility bundle.

## v4.12.2 — utility tuning lab

- Added an offline Utility Tuning Lab that searches utility-score weights and shortlist gates on the same replay frame.
- Produces a recommended parameter bundle only when a tuning candidate clearly beats the current live utility settings offline.

## v4.12.1 — utility selection lab

- Added an offline Utility Selection Lab that compares the new utility-constrained shortlist engine against the legacy ranked-cap shortlist on the same replay frame.
- Added a reusable legacy shortlist helper so utility selection can be evaluated against the previous shortlist semantics without changing the live scanner.
- Added summary/TXT/pack endpoints and homepage controls for utility selection validation.

## v4.12.0 — utility-constrained shortlist redesign

- Added utility-based shortlist annotation and constrained visible selection.
- Default live selection now uses utility_constrained mode.
- The operator UI now surfaces decision score and confidence cues for visible and informational rows.

## v4.11.50 — controlled live adoption review

- Added isolated adoption-window review so the adopted live path can produce a keep-versus-rollback verdict.
- Scanner status now records distinct `live_candidate_adoption` metadata for review-pack attribution.
- Clearing controlled adoption now preserves session context for later review.

## v4.11.48 — isolated live proof review

- Added retained proof-session persistence so a controlled live proof can be reviewed after activation or clearing.
- Added live proof review service that filters proof-attributable runs instead of relying on contaminated current-version aggregates.
- Added summary/TXT/pack endpoints and homepage controls for controlled live proof review.
- Proof activation now stamps a proof session id and stores an activation-time baseline current-version summary for later comparison context.


## v4.11.47 — controlled live candidate proof harness

- Added a controlled live candidate proof harness that activates one exact next-live candidate for a scoped proof window instead of making an open-ended live change.
- Added runtime model + Stage 1 overrides for live scans, sourced only from the controlled proof harness and bounded by deployment scope + expiry.
- Added summary, TXT, and pack endpoints plus homepage controls for activation, clearing, and monitoring.
- Status now carries live-candidate-proof context so operators can see whether a proof window is active.

## v4.11.46 — next live candidate lab

- Added an offline Next Live Candidate Lab that compares model × Stage 1 combinations on the same replay frame and recommends at most one exact next live candidate.
- Historical replay now supports temporary model-bundle overrides so incumbent and shadow model candidates can be scored on the same frame without touching live artifacts.
- Added Next Live Candidate Lab summary/TXT/pack endpoints and homepage controls.

## v4.11.45 — Stage 1 Policy Lab

- Added an offline Stage 1 Policy Lab that compares the current live-style Stage 1 policy against looser candidates on a common replay frame.
- Replay now supports Stage 1 selection-mode and max-candidate overrides for offline policy testing.
- Added Stage 1 Policy Lab summary/TXT/pack endpoints and homepage controls.

## v4.11.44 — scan-level shortlist utility rebuild

- Rebuilt model selection to prioritize adjusted scan-level shortlist utility before generic AUC/Brier tie-breakers.
- Added per-scan visible-vs-hidden gap, per-scan win rate, top-of-scan quality, and shortlist-width metrics to holdout evaluation.
- Rebuilt offline challenger comparison verdicting around scan-level shortlist usefulness, with AUC/Brier demoted to secondary context.
- Updated challenger-comparison UI and memo surfaces to show scan-level utility instead of mainly generic model metrics.

## 4.11.42 - retrain/challenger terminal-state truth fix
- Persist terminal progress for completed/cancelled/failed retrain and challenger runs
- Normalize frontend async state to avoid contradictory running/finished rendering
- Show started/finished timestamps in retrain and challenger UI panels

## 4.11.41 - retrain/challenger stop controls and stale-run healing
- adds stop/cancel endpoints for fresh retrain audit and offline challenger comparison
- adds UI stop buttons for both background jobs
- marks stale running summaries as interrupted after restart or thread loss
- preserves non-promoting behavior and live-path logic

## 4.11.40 - retrain and challenger comparison repair tranche
- fixed the challenger comparison runtime bug by passing the required target move percentage when building the shared evaluation frame
- pinned fresh retrain audit summaries to a falsified source checkpoint instead of mixing a fresh deployment scope into the branch narrative
- persisted a source incumbent model copy alongside the shadow candidate so offline challenger comparison uses the intended incumbent artifact
- upgraded challenger comparison to include a threshold-shortlist proxy comparison on the shared holdout, alongside AUC/Brier/tail/concentration metrics
- kept live Stage 1 mode, threshold, Stage 2 semantics, and live promotion behavior unchanged

## 4.11.39 - offline challenger comparison tranche
- added a non-promoting offline challenger comparison service that scores the incumbent live pt2 bundle versus the latest shadow retrain candidate on a shared offline evaluation frame
- added challenger comparison endpoints, UI card, progress polling, summary TXT export, and comparison pack download
- kept live Stage 1 mode, threshold, Stage 2 scoring semantics, and live promotion behavior unchanged

## 4.11.38
- Added fresh retrain audit progress stages, symbol counters, and heartbeat updates while the non-promoting shadow branch runs.
- Added automatic 5-second polling in the UI while the fresh retrain audit is running so long jobs no longer appear stuck.
- Running summaries now include the current live-path snapshot immediately instead of placeholder dashes.

## v4.11.34
- Add a one-run offline historical decision lab that replays the current threshold, runs the threshold sweep, and packages a future-decision memo without touching live scoring.
- Add `/api/historical-decision-lab/run`, `/api/historical-decision-lab/summary`, and `/api/historical-decision-lab/latest-pack.zip`.
- Surface a new Historical Decision Lab card on the homepage and wire benchmark download buttons.
- Keep Stage 1 mode, live threshold, Stage 2 semantics, regime logic, and model promotion behavior unchanged.

## v4.11.33
- Fix frontend syntax error in model output distribution card so status page can render.

## v4.11.31
- Fixed startup NameError in evidence automation diagnostic battery by defining model_output before use.
- No live-scoring, threshold, Stage 1, Stage 2, regime, or retrain behavior changes.

## v4.11.31
- Add read-only model output distribution diagnostic with per-scan snapshots, rolling summary, and endpoint `/api/model-output-distribution`.
- Surface the diagnostic on the status page and include it in automated post-maturity bundles.
- Keep live Stage 1 mode, threshold, scoring semantics, regime logic, and promotion behavior unchanged.

## v4.11.29 - evidence automation and safe branch orchestration
- added a control-plane-only `EvidenceAutomationService` to automate checkpoint/branch refresh, safe no-op acknowledgement, post-maturity review bundle generation, diagnostic verdict generation, and training orchestration artifacts
- added `/api/reviews/post-maturity-bundle.zip`, `/api/reviews/diagnostic-battery`, `/api/reviews/diagnostic-battery.txt`, `/api/automation/status`, and training orchestration status endpoints
- wired post-maturity automation to the evaluated-pack completion path so fresh evidence automatically produces a bundled artifact and consolidated verdict without changing live scoring semantics
- kept Stage 1 mode, threshold, Stage 2 scoring, penalty/cap behavior, retrain promotion behavior, regime policy, and visible shortlist logic unchanged

## v4.11.28 - Stage 1 opportunity-model switch
- switched the live default `STAGE1_SELECTION_MODE` from `primary_plus_near_miss_recall_promotion` to `stage1_opportunity_model`
- kept `LIVE_RAW_THRESHOLD=0.35`, Stage 2 scoring, penalty/cap logic, branch truth logic, and retraining behavior unchanged
- aligned the live tranche with Decision Tree V2 after the current-version starvation branch showed Stage 1 omission was the best-supported next fix
- preserves the narrow intervention ordering: Stage 1 repair first, then fresh evidence, then reassess threshold / Stage 2 / retrain only if the new window justifies it

## v4.11.27 - decision-state truth reconciliation
- Fixed checkpoint outcome truth so the canonical branch verdict follows current deployment-window evidence rather than a stale first-trigger latch.
- Hardened confirmation to require both the legacy 15% floor and visible quality superiority over the hidden remainder.
- Fixed branch automation to prefer the canonical current-evidence verdict when a stale checkpoint file contains a contradictory outcome.
- Added regression tests covering stale falsified-vs-confirmed splits and canonical branch behavior.

## v4.11.26 - confirmed-branch truth fix
- fixed decision-branch automation so a confirmed checkpoint no longer recommends or auto-applies the 0.28 threshold experiment
- aligned the branch summary with the checkpoint rule: keep the live path unchanged and continue evidence accrual after confirmation
- added a clear-override recommendation when a current-scope 0.28 override conflicts with a confirmed checkpoint
- preserved live Stage 1 mode, live threshold, model bundle, and scan logic unchanged


## v4.11.14 - Stage 2 retrain review tranche
- Added Stage 2 retrain review service and pack to judge whether a shadow recency retrain is the next justified lever.
- Added `/api/reviews/stage2-retrain-review`, `/api/reviews/stage2-retrain-review.zip`, and `/api/debug/stage2-retrain-review`.
- Kept live Stage 1 mode and live threshold unchanged; no live model switch is performed by this tranche.
## v4.11.13 - controlled live switch to stage1_opportunity_model

- switched the deploy-time live Stage 1 mode in `render.yaml` from `primary_only` to `stage1_opportunity_model`
- updated `.env.example` to match the controlled live Stage 1 switch
- updated decision-checkpoint narrative fields so the app describes the live experiment truthfully after the mode switch
- held `LIVE_RAW_THRESHOLD` at `0.35` and left scanner / model / regime logic unchanged

## v4.11.12 — Stage1 selection-repair review tranche

- added `stage1_selection_repair_review` to shadow-compare current-scan Stage 1 modes against the live `primary_only` baseline using the same Stage 2 scoring context
- added `/api/debug/stage1-selection-repair` for operator review of Stage 1 repair candidates before changing live selection mode
- surfaced `stage1_selection_repair_review_latest` in current-version summaries and persisted the review into review packs
- kept `LIVE_RAW_THRESHOLD`, `STAGE1_SELECTION_MODE`, checkpoint logic, branch logic, and model behavior unchanged while identifying the best next Stage 1 repair candidate mode


## v4.11.11 — Threshold experiment review tranche

- added `threshold_experiment_review` to compare the current live threshold with a shadow `0.28` threshold on the same ranked rows
- added `/api/debug/threshold-experiment` for operator review of the controlled threshold experiment before changing live settings
- surfaced `threshold_experiment_review_latest` in current-version summaries and review packs
- kept `STAGE1_SELECTION_MODE`, `LIVE_RAW_THRESHOLD`, checkpoint logic, and model behavior unchanged

## v4.11.10 — Stage1 omission audit scan-finalization hotfix
- Fixed a scan-time `UnboundLocalError` in the Stage 1 omission audit path by moving omission-audit execution until after `all_ranked_rows` is built.
- Preserved v4.11.9 live threshold, Stage 1 mode, checkpoint, and branch semantics unchanged.

## v4.11.9
- add Stage 1 omission audit with shadow Stage 2 scoring for non-blocked omitted names
- expose omission-vs-compression verdict in live status and review packs
- add `/api/debug/stage1-omission-audit` endpoint
- include omission audit in current-version summary and scan/evaluated review ZIPs
- UI: add Stage 1 omission audit card on the homepage

## v4.11.8
- threshold-truth audit hardening: review/diagnostic services now fall back to the current effective live raw threshold instead of a hard-coded 0.35 when row-level threshold metadata is absent.
- added scanner regression coverage for stale decision-branch overrides to ensure preview rows use the current-scope threshold, not a prior override.
- preserves v4.11.7 runtime-scope threshold fix and does not change scanner selection, model, regime, or checkpoint logic.


## v4.11.7
- Added a runtime-scope file for the current deployment window so threshold overrides are evaluated against one authoritative scope from startup onward.
- Unified effective-threshold truth across scanner suppression and decision-branch state by making both consume the same runtime-scope-aware threshold helper.
- Updated checkpoint and branch summaries to use the current runtime scope, eliminating null deployment scope in current-window decision state.
## v4.11.6 — Override-reset + button-action hotfix
- Fixed decision-branch stale override handling for pre-scope legacy override files.
- Legacy decision-branch overrides with no scope metadata are now ignored for current deployment-window semantics.
- Improved operator clear-override writes so cleared state is scoped to the active deployment window.
- Preserved scanner, checkpoint rules, Stage 1 mode, and live experiment logic.


## v4.11.5 — State-truth hotfix
- Scoped decision-checkpoint state to the current deployment window and app version.
- Scoped decision-branch acknowledgement and execution metadata to the current deployment window and app version.
- Prevented stale decision-branch runtime overrides from prior deployments from being treated as active for the current deployment.
- Preserved current scanner and experiment logic while making checkpoint/branch semantics truthful again.

## v4.11.4 — Decision branch acknowledgement + execution logging hotfix
- fixed decision-branch acknowledgement persistence so operator acknowledgement survives summary rebuilds
- fixed decision-branch notification clearing so `branch_notification_pending` stays false after acknowledgement for the active checkpoint trigger
- fixed execution logging for auto-applied threshold experiments and added backfill of last execution metadata for already-active overrides
- kept the live experiment unchanged: no threshold-policy logic changes, no Stage 1 changes, no model changes

## v4.11.3 — Scanner hotfix for decision-branch automation

- fixed a runtime scanner failure caused by `scanner.py` using `effective_live_raw_threshold(...)` without importing it
- preserved all v4.11.2 decision-checkpoint and decision-branch automation behavior unchanged
- no live policy change, no threshold change, no retrain, and no diagnostic-surface expansion


## v4.11.2 — Decision-branch automation with UI controls

- added `app/decision_branch_automation.py` to automate supported post-checkpoint branch actions and persist branch state
- added public endpoint `/api/reviews/decision-branch`
- added admin endpoints for:
  - toggling auto-execute of supported actions
  - executing the supported branch now
  - clearing the active runtime threshold override
  - acknowledging the branch notification
- added runtime threshold override support so the confirmed branch can apply the `0.28` threshold experiment in-app without a manual config edit
- surfaced decision-branch state, override status, and operator controls in the homepage UI and current-version summary
- kept unsupported branches honest: falsification still surfaces an explicit manual retrain + concentration-audit requirement rather than pretending it is safely self-executable

## v4.11.1 — Decision checkpoint automation tranche

- added `app/decision_checkpoint.py` to automate the 30 resolved-visible-row decision checkpoint inside the app
- added public endpoint `/api/reviews/decision-checkpoint` and admin acknowledge endpoint `/api/reviews/decision-checkpoint/ack`
- surfaced automated decision-checkpoint state in `/api/status`, current-version summary, and homepage/status UI
- persisted triggered checkpoint state to `MODEL_DIR/decision_checkpoint_summary.json` so the app records the first decision outcome instead of relying on manual monitoring
- added explicit note that future decision points and follow-on actions should default to automation rather than manual watching

## v4.11.0 — Stage1 revert + decision rule commit

- reverted live Stage1 selection to `primary_only` in `render.yaml` so the unvalidated Stage1 Opportunity Scorer is no longer the default live path
- added `DECISION_RULE.md` with explicit confirmation / falsification criteria and triggered next actions
- surfaced the decision checkpoint on the homepage current-version evidence panel, including current Stage1 mode, visible vs non-visible quality hit rate, resolved visible rows, and rows remaining to the 30-row decision point
- kept the tranche narrow: no new diagnostic service, no threshold change, no retrain, and no new modules


## v4.10.8 — Cooldown-restricted shortlist-quality diagnostic tranche
- Added `app/cooldown_shortlist.py` with a dedicated cooldown-restricted shortlist-quality review service.
- Added public endpoints:
  - `/api/reviews/cooldown-shortlist-review`
  - `/api/reviews/cooldown-shortlist-review.zip`
- New review diagnoses whether surfaced names in cooldown-restricted runs are underperforming the hidden remainder, highlights near-threshold better hidden names, and flags repeated surfaced weak symbols across recent cooldown-restricted evaluated runs.
- Keeps the live path unchanged; this tranche is diagnostic only and exists to test whether amber/cooldown shortlist quality is the next bottleneck.
- Version bumped to `4.10.8`.

## v4.10.7 — Threshold-boundary review tranche

- added a dedicated threshold-boundary review service and ZIP export so the app can diagnose whether the 0.35 shortlist boundary is overblocking good names without changing the live path
- added public review endpoints for threshold-boundary summary JSON and downloadable threshold-boundary ZIP pack
- added scenario support for near-threshold promotion analysis, gap-bucket analysis, and repeated near-threshold symbol tracking
- kept the tranche narrow: no threshold move, no stage1 redesign, no retraining, and no architecture changes

## v4.10.6 — Misranking endpoint fallback fix

- fixed the misranking diagnostic endpoint so it falls back to the most recent mature app-version evidence when the freshly deployed version has scans but no resolved rows yet
- made the evidence source explicit via `source`, `evidence_source_app_version`, `fallback_used`, and `fallback_reason` in the summary output
- added an honest no-evidence diagnostic path when neither the deployed version nor a recent prior version has resolved evidence yet
- added `misranking_diagnostic_manifest.json` to the ZIP bundle
- added regression coverage for mature fallback and no-evidence behavior

## v4.10.5
- added a dedicated post-maturity misranking diagnostic service and review-pack export so the app can diagnose hidden winners, surfaced disappointments, and green-regime shortlist failures without disturbing the live tranche
- added public review endpoints for misranking summary JSON and downloadable misranking ZIP pack
- kept the tranche narrow: no threshold, stage1, retraining, or architecture changes; this patch only adds evidence to explain shortlist-boundary mistakes

## v4.10.4
- fixed Benchmark Lab export population so threshold rows now pull the real replay comparison metrics from replay summaries instead of exporting blank comparison columns
- fixed benchmark recommendation quality by reading the correct replay evidence fields and counterfactual recall fields
- changed symbol classification export behavior so the classification pack always downloads cleanly; when live evaluated evidence is not ready it now falls back to benchmark replay classification when available, or exports an explicit unavailable-yet artifact instead of hard-failing
- kept the tranche narrow: no ranking-logic changes, only benchmark/export correctness fixes

## v4.10.2
- added Benchmark Lab with automated in-app raw-threshold sweep comparison
- added automated symbol classifications: repeat winners, repeat disappointments, hidden outperformers, visible underperformers
- added benchmark summary endpoints and homepage panel for parallel offline analysis while live evidence accumulates

## v4.10.1 — Regime truthfulness and regime-sliced evidence tranche

- repaired regime semantics under `LIVE_PIPELINE_MODE=raw_threshold` so status, decision summaries, and newly persisted review runs no longer imply that amber/red regimes are visibility-blocking when rows can still surface; raw regime state is preserved alongside an effective advisory actionability label
- added current-version **regime-sliced evaluated evidence** with visible-vs-hidden quality hit rates, raw hit rates, average end return, and average MAE broken out by regime/actionability bucket
- added current-version **threshold-band by regime** exports so it is obvious where the simplified path is working or failing across green / amber / red conditions
- updated the homepage current-version summary panel to show regime semantics notes, evaluated evidence by regime, and threshold-band performance by regime
- added regression tests covering raw-threshold regime semantics and regime evidence aggregation

## v4.10.0 — Pipeline simplification tranche

- promoted a simplified live pipeline path with `LIVE_PIPELINE_MODE=raw_threshold` as the new default and `LIVE_RAW_THRESHOLD=0.30` as the default live visibility threshold
- in raw-threshold mode, the live shortlist is driven by raw model output plus the fixed threshold instead of the full regime/cooldown/post-model suppression stack
- kept the full pipeline available as an explicit mode for controlled comparison rather than as the default path
- aligned rolling preview logic with the simplified live pipeline so previews do not disagree with the final shortlist path
- updated the replay UI defaults to make raw-threshold comparison the primary replay candidate path
- added config coverage for the new live pipeline defaults

## v4.9.1 — Live validation / model audit / replay ablation tranche

- added a replay-based **model audit** service with admin endpoints to evaluate `prob_2_model` ordering quality from the latest replay pack, including AUC, Brier score, calibration deciles, tail precision, and per-symbol resolved summaries
- added replay **pipeline ablation** support with `pipeline_mode` (`full` or `raw_threshold`) and a replay summary comparison showing whether the simpler raw-threshold path outperforms or underperforms the full post-model pipeline
- exposed the new replay ablation controls and model-audit build/load actions in the homepage UI so they can be run without manual API calls
- added a defensive visible-shortlist symbol-concentration safeguard so duplicate-symbol leakage cannot dominate the surfaced shortlist if it ever appears
- added regression tests for model audit generation, pipeline ablation summary, and visible-shortlist concentration trimming

## v4.9.0 — Dedicated stage1 opportunity scorer tranche

- added a dedicated **Stage1 Opportunity Scorer** that can be built from the latest replay pack's replay-labeled stage1 rows and persisted under `MODEL_DIR`
- added scorer-based stage1 selection modes for controlled testing: `stage1_opportunity_model` and `primary_plus_opportunity_reserve`
- kept live behavior conservative by leaving `STAGE1_SELECTION_MODE=primary_only` as the default until replay evidence proves a scorer-based mode is better
- extended replay counterfactual rows and promotion-audit comparisons with opportunity-model rank/score metadata and scorer-based selection-mode comparisons
- added admin endpoints and homepage controls to build the stage1 opportunity scorer from the latest replay pack and inspect its validation summary
- added regression tests for opportunity-score-driven stage1 selection modes

## v4.8.1 — Live scan candle-alignment hotfix

- fixed a live-scanning bug where Stage 1 candle windows could be anchored to a non-5-minute timestamp, causing every symbol to fail with `stage1_insufficient_history` despite candles being returned
- aligned live candle range start/end times to the latest completed 5-minute boundary before regularization
- added a regression test covering unaligned live end-times

## v4.8.0 — Stage1 promotion audit tranche

- added configurable `STAGE1_SELECTION_MODE` with `primary_only` as the new default after replay evidence showed the prior widening logic did not improve ranking quality
- kept hybrid stage1 modes available for controlled experiments and replay audits
- added replay-side stage1 evidence outputs:
  - primary-rank bucket summary
  - recall-rank bucket summary
  - feature-delta summary for selected hits vs missed hits
  - promotion-audit comparison across candidate stage1 selection rules
- enriched replay counterfactual rows with stage1 feature snapshots and selection-source metadata

## v4.7.0 — Stage1 recall repair tranche

- replaced stage1 primary-only selection with a hybrid **primary + recall reserve** shortlist
- preserves most stage1 slots for the existing primary ranking while reserving a configurable share for recall-biased names
- added stage1 diagnostics for selection mode, primary slots, recall reserve slots, and selection source per symbol
- propagated stage1 selection source into replay counterfactual rows and review trace exports
- intended to improve **stage1 quality recall** after replay evidence showed too many later quality opportunities were being missed upstream

## 4.6.3

- made the historical replay controls visually explicit with stronger field captions instead of relying on subtle label styling
- added a replay progress panel with an indeterminate progress bar and elapsed-time indicator while replay is running

## 4.6.2

- fixed the historical replay homepage controls by adding explicit field labels and a local admin-password input inside the replay section
- added helper text explaining the replay parameters so the UI no longer relies on guesswork

## 4.6.1

- added homepage UI controls for historical replay so replay runs, replay summary loading, and replay pack download no longer require manual API calls
- wired replay actions to the existing admin-password workflow on the homepage

## 4.6.0

- added a historical live-emulation replay engine that reuses the current model, locked cohort, stage1/stage2 selection logic, market-regime engine, and visibility rules on historical Coinbase 5-minute data
- added a counterfactual opportunity audit to measure whether stage1 is missing later quality opportunities
- added admin replay endpoints: `/api/replay/run`, `/api/replay/latest-summary`, and `/api/replay/latest.zip`
- added historical as-of candle helpers in the Coinbase client for timestamp-anchored replay
- added regression tests for replay counterfactual recall, replay pack contents, and timestamp-anchored historical range fetches

## 4.5.3

- Added current-version evidence outputs for symbol repeatability and outlier concentration so review packs can distinguish broadly repeatable signal from one-symbol spikes.
- Fixed current-version pack aggregation queries to include `review_status_path`, restoring scan score diagnostics, candidate-quality summaries, and cohort symbol summaries inside the version bundle.
- Exported `current_version_symbol_repeatability.csv` and `current_version_outlier_concentration.json` in the version pack.
- Added regression tests for symbol repeatability aggregation and outlier-concentration detection.

## 4.5.2

- Fixed decision-summary truth semantics so validated-band visible rows are no longer mislabeled as "no validated candidates" when they remain advisory-only watchlist rows.
- Added explicit `validated_rows`, `validated_selective_rows`, and `validated_watchlist_rows` to `decision_summary`, and made `no_validated_candidates` depend on actual validated-band presence rather than actionability tier.
- Added regression tests for validated-band watchlist truthfulness.

## 4.5.1

- Fixed stage2 accounting truth by de-duplicating symbol-level diagnostics, funnel counts, and candidate-quality summaries that were previously inflated by informational review rows.
- Corrected `score_diagnostics` to report unique ranked symbols, added `row_count_total` and `duplicate_row_instances_removed`, and removed duplicate names from `top_pretrim_candidates`.
- Corrected stage2 scored/final coverage counts and hidden-after-trim counts to reflect unique ranked symbols rather than duplicated review row instances.
- Added regression tests for stage2 accounting de-duplication.

## 4.5.0

- Fixed status snapshot integrity by archiving the last completed scan separately and clearing result-only sections when a new scan starts.
- Prevented partial running snapshots from inheriting stale completed-scan decision, score-diagnostic, and candidate-quality fields.
- Added explicit last-completed scan metadata to the status payload and surfaced it in the homepage status panel.

## 4.4.9

- Lowered the default `STAGE1_MAX_CANDIDATES` from 80 to 40 so stage1 selection is less likely to behave as a pass-through.
- Added pass-through detection, selected-share metrics, and top visible stage1 ranks to current-scan candidate-quality diagnostics.
- Added current-version cohort symbol summary to surface dead-weight names in the locked live cohort.
- Updated status and deployment-reality panels to show stage1 shortlist discipline and cohort deadweight more directly.

## 4.4.8

- Added candidate-quality diagnostics that trace stage1 selection into stage2 scored rows, visibility outcomes, and liquidity-tier contribution.
- Exported `candidate_quality.json`, `stage1_to_stage2_trace.csv`, and `candidate_quality_by_tier.csv` in run packs.
- Added current-version candidate-quality aggregation to make it easier to judge whether weakness starts in stage1 selection, the locked cohort, or stage2 scoring.
- Updated the homepage and version summary panels to surface candidate-quality diagnostics directly.

## 4.4.7

- Fixed stage-2 telemetry so scored counts reflect all ranked stage-2 rows, not just visible rows after trimming.
- Added deeper current-scan score diagnostics: scoring funnel, liquidity-tier breakdown, and top pre-trim candidates.
- Exported `top_pretrim_candidates.csv` in review packs to diagnose whether the app is finding stronger candidates and then hiding them.
- Updated homepage status cards to show current-scan score generation and trim behavior directly.

## 4.4.6.1

- Fixed homepage JavaScript regression where the deployment-reality panel referenced `summary` in the wrong renderer and threw `summary is not defined`.
- Restored loading of the homepage and current-version diagnostics panel.

## 4.4.6

- Raised the unvalidated-tail / guardrail cap from 0.59 to 0.65 to remove the structural contradiction with the 0.60 validated floor.
- Made the cap configurable through `TAIL_UNVALIDATED_CAP` and wired that value through live scoring, holdout simulation, and heuristic fallback scoring.
- Added per-scan score diagnostics to runtime status and review packs (`score_diagnostics.json`) so score-range starvation can be seen on each run, not only after aggregating deployment windows.
- Added `current_version_scan_score_diagnostics.csv` to the current-version export and surfaced the counts on the homepage.
- Added a first pytest suite and refreshed README / `.env.example` to better match the actual app.

## 4.4.5

- Added current-version evidence summary generation and API output so the deployment's actual score range, threshold-band counts, visible/non-visible hit rates, and regime breakdown are visible without unpacking ZIPs by hand.
- Included `current_version_evidence_summary.json`, `current_version_threshold_bands.csv`, and `current_version_regime_breakdown.csv` inside the current-version review bundle.
- Added a homepage "Deployment reality" panel fed from `/api/reviews/current-version-summary` to show whether validated bands are dormant in the active deployment window.

## 4.4.4

- Fixed version-truth skew in regenerated review packs so summary and manifest metadata use the run's stored app version rather than the currently running code version.
- Updated repo validation to exercise `/api/reviews/current-version.zip`, force a due demo review resolution, and confirm evaluated packs are included once outcomes resolve.
- Refreshed docs and validator references so the package does not still claim 4.4.2 after 4.4.3/4.4.4 changes.

## 4.4.3

- Added one-click current-version review bundle export via `/api/reviews/current-version.zip`, including evaluated packs when available.
- Preserved blocked-to-follow-up continuity and rolling evidence artifacts for version-level review.

- Consolidates repeated blocked scans during the same cooldown window into a single cooldown campaign.
- Merges tracked blocked names across scans and carries the combined set into the expiry follow-up.
- Exports cooldown_campaign.json and campaign counts in manifests/summary.

## 4.3.9
- fixed cooldown follow-up comparison so previously blocked names that become visible are counted and summarized correctly
- pinned tracked follow-up names in the visible shortlist when they graduate after cooldown
- added `tracked_visible_rows.csv` plus tracked-visible status fields for clearer handoff evidence

# Changelog

## 4.3.7
- added a blocked-to-follow-up handoff so the strongest blocked names are persisted as tracked context and automatically compared on the cooldown follow-up scan
- reserved Stage 2 recheck slots for tracked blocked names during cooldown follow-up scans so they are less likely to disappear purely because of ranking caps
- added `blocked_monitoring_context`, `followup_comparison`, and follow-up comparison exports (`follow_up_scan.json`, `blocked_monitoring_context.json`, `followup_comparison.json`, `followup_changes.csv`)
- updated the UI and review-pack summaries to show whether previously blocked names became visible, stayed blocked, or moved closer to the live threshold

## 4.3.6
- added threshold-proximity telemetry (`distance_to_live_threshold`, `visibility_band`) so blocked rows are ranked by how close they are to becoming visible, not just by residual score order
- improved blocked-monitoring verdicts to explain closeness to the current live threshold as well as distance to the validated band
- added cooldown follow-up scan scheduling so temporary cooldown states can trigger an automatic recheck shortly after expiry without manual intervention
- expanded blocked monitoring UI/export fields with threshold, threshold gap, and visibility-band context

## 4.3.4
- preserved informational and overflow evidence rows in review packs instead of silently dropping them when the informational cap was hit
- added `informational_overflow_rows.csv` and `pending_outcomes.csv` to scan/evaluated packs for clearer evidence accounting
- tightened weak-scan visible shortlists so watchlist-only runs emphasize near-band rows and cap exploratory spillover
- clarified scan/report accounting with overflow row counts in status summaries and manifests

## v4.3.2

- Added a decision-verdict layer so weak scans explain whether they are validated, near-validated, or exploratory only.
- Tightened visible shortlist construction so oversized watchlists are trimmed into a smaller monitoring queue while the overflow remains available in informational rankings and review packs.
- Added row-level score-band fields and distance-to-validated telemetry.
- Clarified threshold-policy diagnostics so liquidity tiers are identified as policy tiers, not score bands.
- Kept informational export ordering stable so missing ranks remain null rather than being treated as rank 0.

## v4.1.1 (2026-03-18) — Rolling Preview Telemetry Cleanup

- Aligned stage telemetry with the new rolling-preview pipeline so `stage2_scored` no longer counts stage-1 preview rows.
- Added explicit preview/confirmation counters to coverage and stage status: `visible_rows`, `preview_rows`, `deep_confirmed_rows`, `stage1_preview_rows`, `stage2_partial_rows`, and `stage2_final_rows`.
- Clarified the dashboard so running scans distinguish visible preview rows from deep-confirmed rows and show friendlier stage labels.
- Kept backward-compatible legacy fields while making the newer telemetry the source of truth.

## v4.1.1 (2026-03-18) — Regime Apply-vs-Compute Fix

- Fixed the core partial-publication bug where a successfully computed market regime could be overwritten by a later pending heartbeat during the same running scan.
- Successful partial publication now preserves the last applied non-pending regime across later running snapshots until a newer computed regime replaces it.
- Added explicit computed-vs-applied regime lifecycle metadata: `last_computed_*`, `last_applied_*`, `computed_snapshot_version`, and `applied_snapshot_version`.
- Added an inconsistency warning path for the exact broken state this release prevents: partial publish counted as successful while the exposed public regime remains pending.
- Updated UI/debug surfacing so running scans show computed vs applied regime state directly.

## v4.0.3 (2026-03-18) — Live Regime Publication + Coherent Running Status

- Running scans now reset stale scores, coverage, guardrails, and tail counters at scan start rather than mixing new progress with the previous completed snapshot.
- The market-regime engine now publishes mid-scan from partial stage-1 coverage once enough feature-ready symbols exist, instead of staying pending until the very end of the light-fetch loop.
- `/api/status` and `/api/debug/coverage` now stay aligned during a running scan because partial coverage is written through the same status path.
- `updated_at_utc` now advances on scan lifecycle updates, and `scan_result_scope` / `scan_result_generated_at_utc` distinguish partial from final results.
- Version bumped to 4.1.0 across app, UI, docs, paper-trade logging, and validation harness.

## v4.0.1 (2026-03-18) — Event-Risk Market Regime Engine

- Added a live **market regime engine** with green/amber/red states, headline-risk classification, metrics, reasons, and cooldown persistence.
- Added **tier-aware live policy** for BTC/ETH, liquid majors, and smaller alts with configurable haircut, cap, threshold, and suppression rules by regime state.
- Added explicit **suppression accounting** for regime suppression, cooldown suppression, and threshold suppression in status + coverage outputs.
- Added **prob_2_pre_regime**, `market_regime_state`, `headline_risk`, `liquidity_tier`, live threshold, haircut factor, cap, and override/cooldown fields to scored rows.
- Updated the UI and API so live scans show raw model score, pre-regime score, adjusted score, and market-regime context clearly.
- Bumped app versioning, docs, and smoke-tested startup scan in demo mode.

## v3.0.0 (2026-03-17) — Trained Cohort Lock + Reliability Lab

- Added **trained cohort locking**: training now selects a deterministic top-liquidity cohort, saves it in model metadata, and live scans use that same cohort by default.
- Added `/api/reliability-lab` with live honesty metrics for the active target + current model only.
- Added score-bucket calibration, top-tail truth tables, rolling drift windows, and evidence strength.
- Logged capped state, activity bucket, liquidity bucket, and score rank for each prediction.
- Fixed the scan progress mismatch where `symbols_total` could exceed the actual requested universe.
- Version bumped to 3.0.0 across app, UI, docs, user-agents, and validation harness.

# Changelog

## v2.8.0 (2026-03-17) — Accuracy Hardening

Four categories of changes: statistical rigor, horizon optimization, dynamic configuration, and UI improvements.

### New in v2.8.0 beyond v2.7.0

- **Model-fingerprint-aware forward validation**: paper-trade summary now isolates the currently deployed target *and* model fingerprint.
- **Row-level target resolution**: each prediction stores its own move/horizon/quality thresholds and is resolved against those exact settings later.
- **Adjusted-score-first selection and readiness**: candidate selection and live-threshold recommendation now prefer the deployed adjusted score path when those metrics are available.
- **Upper-tail diagnostics added**: score quantiles, top-bucket lift, and `dead_upper_tail` are now persisted in training metadata to expose non-actionable models quickly.
- **Cadence alignment restored**: Render training sampling now matches 15-minute live scanning (`TRAIN_SAMPLE_EVERY_N_BARS=3`).

### Breaking changes

- **Default horizon changed**: 120 → 240 minutes. Retrain required.
- **Quality thresholds loosened**: MAE -1.5% → -2.0%, end_ret -0.5% → -0.8% (proportional to doubled horizon). Retrain required.
- **Embargo widened**: 150 → 270 minutes (horizon + 30 min buffer). Retrain required.
- Training output metadata schema changed (new fields: wilson bounds, temporal stability, adjusted metrics).

### Wilson confidence-bound model selection

- Model selection now uses Wilson score interval lower bounds (90% confidence) instead of raw precision.
- With 20 samples at 75% precision, raw says 0.75 but Wilson lower bound says 0.56. The selector now uses 0.56.
- Prevents over-trusting a thin lucky tail. Small but statistically strong tails are favored over large but noisy ones.
- Wilson bounds also reported in training output: `wilson_lower_0_60`, `wilson_lower_0_70`, etc.
- Readiness gate updated: requires Wilson lower bound >= 0.50 (was raw precision >= 0.65).

### Temporal stability evaluation

- After selecting the best model, the test set is split into 3 time windows.
- Per-window AUC, precision, and event rate are computed and reported.
- Training output includes `temporal_stability_model` and `temporal_stability_adjusted` with per-window and aggregate metrics.
- Readiness gate includes temporal stability: if worst-window AUC < 0.52, `high_confidence_ready` is forced to `false`.
- This answers "does the model hold up across different time periods?" — the key question for regime-sensitive crypto.

### Adjusted-score offline evaluation

- The scanner's post-model overlays (guardrail caps, panic suppression) are now simulated on the test set during training.
- Training output includes `adjusted_*` metrics (e.g., `adjusted_auc_holdout`, `adjusted_precision_at_0_60`) alongside raw model metrics.
- This closes the gap between "what offline metrics report" and "what the user actually sees."
- Binance and sector penalties cannot be simulated historically (cross-asset, no backfill) — documented as a known limitation.

### Horizon optimization

- Default horizon extended from 120 to 240 minutes. More time to touch +2% while momentum features still have predictive power.
- Quality thresholds proportionally loosened for longer path exposure: MAE -1.5% → -2.0%, end_ret -0.5% → -0.8%.
- Stage 2 lookback increased from 2200 to 2400 bars.
- Expected quality event rate increases from ~12% to ~20-25%, enabling meaningful probability spread above 0.60.

### Dynamic configuration

- All hardcoded references to 120-minute horizon removed from code, UI, and documentation.
- Target move, horizon, and quality thresholds displayed in UI status panel (Target card).
- Health and status API endpoints now include `target` config block.
- Forward validation comments and messages are horizon-agnostic.

### UI improvements

- Status panel shows live target configuration (move %, horizon, MAE threshold, end_ret threshold).
- Score table shows both Adjusted and Model probability columns.
- Forward validation panel uses dynamic horizon description.

## v2.6.1 (2026-03-17) — Validation Integrity & Blindspot Features

### Breaking changes

- **Feature set changed**: 60 features (was 53 in v2.6.0). 7 blindspot features added. Binance cross-exchange signals removed from FEATURE_COLUMNS (now post-model overlays). Retrain required.

### Validation integrity (addresses external review findings)

- **Purged time split with 150-minute embargo**: train/validation/test splits now have a gap equal to the prediction horizon + buffer. Samples in the embargo window are dropped. Eliminates horizon-overlap leakage in offline metrics.
- **Separate validation and test sets**: model selection uses validation set; final reported metrics come from untouched test set. Reported AUC/precision are no longer optimistic from selection bias.
- **Forward validation cooldown dedup**: each symbol logged at most once per horizon window. Prevents correlated predictions from inflating forward precision.
- **Episode-level precision**: forward validation summary reports deduplicated episode precision alongside raw precision. Episode precision is the trustworthy number.
- **BTC panic as hard readiness gate**: `high_confidence_ready` blocked if BTC panic challenge precision < 0.30.
- **Stale backlog resolution**: paper trade resolver fetches candles from oldest pending prediction, not a fixed offset. Survives app downtime.

### Dual probability output

- Every score now reports `prob_2_model` (raw calibrated model output, matches offline metrics) and `prob_2` (risk-adjusted confidence score after caps, regime suppression, sector/Binance penalties).
- UI shows both columns: "Adjusted" and "Model".
- Forward validation tracks and reports precision for both.
- Documentation updated: `prob_2` is described as a risk-adjusted confidence score, not a calibrated probability.

### Blindspot features (7 new)

- `btc_recovery_from_trough`: bars since BTC's 24h low — detects dead cat bounces
- `btc_trough_depth`: BTC's 24h high-to-low range — severity of recent crash
- `move_vs_atr_ratio`: |ret_24h| / atr_pct — detects exhaustion after extended runs
- `volume_concentration`: max candle volume / mean volume — detects whale-driven illusions
- `volume_acceleration`: recent volume / prior volume — detects fading momentum
- `spread_cost_proxy`: execution cost estimate — flags untradeable spreads
- `session_bucket`: Asia/Europe/US/overnight — enables session-aware learning

### Binance cross-exchange signals (post-model overlay)

- `binance_lead_15m`, `binance_lead_1h`, `binance_price_gap` computed from Binance public API
- Applied as post-model adjustment, NOT as learned features (no historical Binance data in training)
- Graceful degradation if Binance unreachable (features default to 0.0)
- ~35 symbol mappings for major Coinbase-listed assets

### Sector contagion (post-model overlay)

- 6 sector groups defined: Solana ecosystem, Ethereum L2, DeFi blue chips, AI tokens, memecoins, L1 alts
- If a sector leader drops >3% in 1h, all sector members receive a probability penalty
- Applied post-model, not learned

### Model improvements

- Calibration spread check: isotonic calibration skipped if it collapses prediction spread below 0.04 std
- Memory cleanup: only best model candidate retained during training (gc.collect after each)
- LightGBM feature consistency: always passes numpy arrays to avoid feature-name mismatch
- Diagnostic metadata: pred_spread_std, pred_spread_range, raw_spread_std in training output

### Operational

- Version consistently v2.6.1 across main.py, HTML, user-agent strings
- Numpy RuntimeWarning suppression for zero-variance correlation computations
- `btc_led_panic` hard block threshold raised to BTC -5% (was -3%); normal panic handled by EVENT_RISK + regime gating

## v2.6.0 (2026-03-16) — Quality-Conditioned Rebuild

Implements findings from the v2.5.0 technical review. Three fundamental changes.

### Breaking changes

- **Training label changed**: `y` is now quality-conditioned (touched +2% AND MAE > -1.5% AND end_ret > -0.5%). Models trained on v2.5.0 are incompatible — retrain required after deploy.
- **Feature set changed**: 3 features removed, 5 added. Persisted models from v2.5.0 will fail to load due to feature mismatch — retrain required.
- **`high_confidence_ready` is now much stricter**: requires AUC > 0.58, count >= 20, precision >= 0.65. Models that passed in v2.5.0 will likely fail in v2.6.0 until properly trained.

### Label and objective

- Primary target is quality-conditioned: touch +2% with drawdown discipline and positive drift
- Raw touch preserved as diagnostic (`y_raw_touch`) but never used for training
- Sample weights: 0.0 for MAE < -8%, 0.30 for MAE < -5%, 0.50 for MAE < -3%, 1.10 bonus for quality touches
- Expected quality event rate: 25-35% (was ~78% in v2.5.0)

### Model

- LightGBM is the primary model family (8 hyperparameter configurations)
- Logistic regression retained as fallback (2 configurations)
- Graceful degradation if lightgbm not installed
- Early stopping on calibration set (patience 50)
- Model type exposed in metadata and UI

### Features

- Removed: `drawdown_24h`, `drawdown_7d`, `weekend_flag`
- Added: `btc_corr_24h`, `momentum_persistence_1h`, `rv_ratio_1h_24h`, `up_volume_ratio_1h`, `time_since_impulse`

### Regime gating

- BTC panic suppresses all probabilities by configurable boost (default 0.10)
- Stage 1 candidates halved during panic
- Stage 2 output capped at 15 during panic (vs 50 normally)
- Stage 1 ranking penalizes assets lagging BTC during panic/weak
- Regime exposed as top-level `regime_context` in status

### Stage 2

- Hard dollar-volume floor ($100K): assets below are blocked
- Soft dollar-volume penalty ($100K-$250K): +0.10 uncertainty

### Model selection

- AUC gate: models with AUC < 0.58 deprioritized regardless of precision
- Selection score now includes AUC gate as first element

### Readiness

- `high_confidence_ready` requires: AUC > 0.58, count >= 20, precision >= 0.65, challenge precision >= 0.40
- Returns `readiness_details` dict with explanation

### Validation

- New challenge sets: BTC-panic-only, low-activity-only
- Paper trade logging to JSONL for forward validation
- `/api/paper-trade/summary` endpoint

### Operational

- `regime_context` as top-level status field
- Health endpoint includes `version` and `regime_context`
- UI: regime badges with color coding, model type display, quality metrics in training panel
- `libgomp1` added to Dockerfile for LightGBM OpenMP support


## v4.1.2
- crash-safe runtime-state recovery after restart
- automatic SQLite schema repair for `model_hash` and related paper-trade columns
- lower-memory training path with float32 feature frames
- named-feature inference to eliminate LightGBM feature-name warnings

## 4.11.35
- Fix Historical Decision Lab frontend JavaScript property access so run/load/download buttons work.

## 4.11.37

- add direct homepage download/open buttons for operator share items: status, current-version summary, model-output distribution, post-maturity bundle, and fresh retrain audit summary/pack
- add text export endpoints for model-output distribution and fresh retrain audit summary
- keep live logic unchanged; this is a UI/export ergonomics tranche only

## 4.11.36
- Add a supported post-falsification non-promoting branch that builds a fresh shadow retrain candidate plus symbol concentration audit pack.
- Wire decision-branch execute-now to start the safe shadow retrain audit instead of stopping at a manual-only note.
- Add UI + endpoints for loading and downloading the fresh retrain audit artifacts.


## v4.14.0
- Regime-aware utility selection v4 with cluster-safe multi-name gating
- Dynamic singleton/two-name/three-name policy based on regime and top-cluster quality
- Utility selection lab now reports the active engine label dynamically

## v4.15.3
- Reverted LIVE_SELECTION_MODE default to legacy in config and render.yaml.
- Kept Utility Policy Search as the challenger discovery path while protecting live with the incumbent legacy shortlist.


## v4.20.4
- Fix fresh retrain audit source-resolution so it only binds to the current deployment falsification scope.
- Clear stale audit artifacts at run start and prevent failed runs from surfacing old shadow candidate payloads as current truth.
- Gate latest-pack download on a genuinely completed current-version audit run.
