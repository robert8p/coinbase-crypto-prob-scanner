function escapeHtml(value) {
  const s = String(value ?? '');
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
  return (Number(v) * 100).toFixed(1) + '%';
}
function fmtNum(v, d) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
  return Number(v).toFixed(d === undefined ? 2 : d);
}
function formatNumber(v, d) {
  return fmtNum(v, d === undefined ? 4 : d);
}
function safeJoin(v) {
  if (!v) return '-';
  if (Array.isArray(v)) return v.join(', ');
  return String(v);
}
function riskClass(v) {
  const n = Number(v || 0);
  if (n < 0.20) return 'risk-low';
  if (n < 0.40) return 'risk-mid';
  return 'risk-high';
}
function regimeClass(state) {
  const s = String(state || '').toLowerCase();
  if (s === 'red') return 'regime-red';
  if (s === 'amber') return 'regime-amber';
  if (s === 'pending') return 'regime-pending';
  return 'regime-green';
}


const replayUiState = {
  running: false,
  startedAtMs: null,
  timer: null,
};

const uiState = {
  actionableRows: [],
  informationalRows: [],
  sort: {
    actionable: { key: 'rank', direction: 'asc' },
    informational: { key: 'rank', direction: 'asc' },
  },
  decisionBranchSummary: null,
};

function rowRank(row, informational) {
  return informational
    ? (row.informational_rank ?? row.pre_policy_rank ?? row.candidate_rank_all ?? row.would_be_rank ?? Number.POSITIVE_INFINITY)
    : (row.score_rank ?? Number.POSITIVE_INFINITY);
}

function rowValue(row, key, informational) {
  switch (key) {
    case 'rank':
      return rowRank(row, informational);
    case 'price':
      return Number(row.price ?? Number.NaN);
    case 'live_score':
      return Number((row.live_score ?? row.prob_2) ?? Number.NaN);
    case 'decision_score':
      return Number((row.utility_decision_score ?? row.opportunity_score) ?? Number.NaN);
    case 'opportunity_score':
      return Number(row.opportunity_score ?? Number.NaN);
    case 'candidate_stage':
      return String(stageLabel(row.candidate_stage));
    case 'probability_semantics':
      return String(row.probability_semantics || row.contract_truth_semantics || '');
    case 'pre_regime':
      return Number(row.prob_2_pre_regime ?? Number.NaN);
    case 'model_score':
      return Number(row.prob_2_model ?? Number.NaN);
    case 'live_threshold':
      return Number(row.live_threshold ?? Number.NaN);
    case 'market_regime_state':
      return String(row.market_regime_state || '');
    case 'liquidity_tier':
      return String(row.liquidity_tier || '');
    case 'risk':
      return Number(row.risk ?? Number.NaN);
    case 'reason_summary':
      return String((row.suppression_reason_detail || row.policy_constraint_reason || row.actionability_reason || safeJoin(row.reasons) || ''));
    case 'symbol':
    default:
      return String(row.symbol || '');
  }
}

function compareValues(a, b) {
  const aNum = Number(a);
  const bNum = Number(b);
  const aNumValid = Number.isFinite(aNum);
  const bNumValid = Number.isFinite(bNum);
  if (aNumValid || bNumValid) {
    if (!aNumValid && !bNumValid) return 0;
    if (!aNumValid) return 1;
    if (!bNumValid) return -1;
    if (aNum < bNum) return -1;
    if (aNum > bNum) return 1;
    return 0;
  }
  const aStr = String(a || '').toLowerCase();
  const bStr = String(b || '').toLowerCase();
  if (aStr < bStr) return -1;
  if (aStr > bStr) return 1;
  return 0;
}

function sortRows(rows, tableKey) {
  const informational = tableKey === 'informational';
  const sort = uiState.sort[tableKey] || { key: 'rank', direction: 'asc' };
  const dir = sort.direction === 'desc' ? -1 : 1;
  return [...rows].sort(function(left, right) {
    const primary = compareValues(rowValue(left, sort.key, informational), rowValue(right, sort.key, informational));
    if (primary !== 0) return primary * dir;
    return compareValues(rowRank(left, informational), rowRank(right, informational));
  });
}

function updateSortIndicators() {
  document.querySelectorAll('th.sortable').forEach(function(th) {
    const tableKey = th.getAttribute('data-table');
    const key = th.getAttribute('data-sort-key');
    const state = uiState.sort[tableKey];
    const indicator = th.querySelector('.sort-indicator');
    const active = state && state.key === key;
    th.classList.toggle('active', !!active);
    if (!indicator) return;
    indicator.textContent = active ? (state.direction === 'asc' ? '▲' : '▼') : '↕';
  });
}

function attachSortHandlers() {
  document.querySelectorAll('th.sortable').forEach(function(th) {
    if (th.dataset.boundSort === 'true') return;
    th.dataset.boundSort = 'true';
    th.addEventListener('click', function() {
      const tableKey = th.getAttribute('data-table');
      const key = th.getAttribute('data-sort-key');
      const current = uiState.sort[tableKey] || { key: 'rank', direction: 'asc' };
      uiState.sort[tableKey] = {
        key,
        direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc',
      };
      renderScores({ rows: uiState.actionableRows, informational_rows: uiState.informationalRows });
    });
  });
  updateSortIndicators();
}

function stageLabel(stage) {
  const s = String(stage || '');
  if (s === 'stage1_preview') return 'Stage 1 preview';
  if (s === 'stage2_partial') return 'Stage 2 partial';
  if (s === 'stage2_final') return 'Stage 2 final';
  return s || '-';
}
function scopeLabel(row) {
  if (row.deep_confirmed) return 'confirmed';
  if (row.provisional) return 'preview';
  return 'final';
}

async function getJson(url, opts) {
  const r = await fetch(url, opts || {});
  if (!r.ok) {
    let detail = '';
    const contentType = r.headers.get('content-type') || '';
    try {
      if (contentType.includes('application/json')) {
        const payload = await r.json();
        detail = payload.detail || payload.message || payload.error || JSON.stringify(payload);
      } else {
        detail = (await r.text() || '').trim();
      }
    } catch (_) {
      detail = '';
    }
    const suffix = detail ? (': ' + detail) : '';
    throw new Error(url + ' failed: ' + r.status + suffix);
  }
  return await r.json();
}

function renderStatus(status, scores) {
  const regime = status.market_regime || {};
  const scan = status.scan || {};
  const cov = status.coverage || {};
  const guard = status.guardrails || {};
  const target = status.target || {};
  const contract = status.score_contract || {};
  const decision = status.decision_summary || {};
  const decisionCheckpoint = status.decision_checkpoint || {};
  const followUp = status.follow_up_scan || {};
  const blockedContext = status.blocked_monitoring_context || {};
  const cooldownCampaign = status.cooldown_campaign || {};
  const followupComparison = status.followup_comparison || {};
  const scoreDiag = status.score_diagnostics || {};
  const scanFunnel = scoreDiag.scoring_funnel || {};
  const tierDiag = scoreDiag.by_liquidity_tier || {};
  const candidateQuality = status.candidate_quality || {};
  const stage1OmissionAudit = status.stage1_omission_audit || {};
  const candidateTier = candidateQuality.stage2_by_tier || {};
  const modelOutput = status.model_output_distribution || {};
  const stage1Tier = candidateQuality.stage1_by_tier || {};
  const currentPartial = !!(scan.running && status.scan_result_scope === 'partial');
  const lastCompleted = status.last_completed_scan_result || {};
  const banner = document.getElementById('scanBanner');
  banner.className = 'banner ' + regimeClass(regime.state);
  banner.innerHTML =
    '<strong>Scan:</strong> ' + (scan.phase || '-') +
    ' &nbsp;|&nbsp; <strong>Market regime:</strong> ' + (regime.state || '-') +
    ' &nbsp;|&nbsp; <strong>Headline risk:</strong> ' + (regime.headline_risk || '-') +
    ' &nbsp;|&nbsp; <strong>Actionability:</strong> ' + (regime.actionability_state || '-') +
    ' &nbsp;|&nbsp; <strong>BTC context:</strong> ' + (status.regime_context || '-') +
    ' &nbsp;|&nbsp; <strong>Rows:</strong> ' + (scores.count || 0) +
    ' &nbsp;|&nbsp; <strong>Scope:</strong> ' + (status.scan_result_scope || '-') +
    ' &nbsp;|&nbsp; <strong>Verdict:</strong> ' + (decision.followup_headline || decision.headline || '-') +
    ' &nbsp;|&nbsp; <strong>Decision checkpoint:</strong> ' + (decisionCheckpoint.decision_checkpoint_outcome || decisionCheckpoint.current_outcome || decisionCheckpoint.status || '-') + (currentPartial ? ' &nbsp;|&nbsp; <strong>Last completed:</strong> ' + (lastCompleted.scan_finished_at_utc || '-') : '');

  const reasons = safeJoin(regime.reasons);
  const metrics = regime.metrics || {};
  const modelOutputHeadline = modelOutput.headline || '-';
  const modelOutputTail = modelOutput.average_upper_tail_counts_per_scan || {};
  const modelOutputCard =
    '<div class="card"><h3>Model output distribution</h3>' +
      '<p><strong>Headline:</strong> ' + modelOutputHeadline + '</p>' +
      '<p><strong>Scans in window:</strong> ' + (modelOutput.scans_in_window || 0) + '</p>' +
      '<p><strong>Avg ge 0.45 / scan:</strong> ' + fmtNum(modelOutputTail.ge_0_45, 3) + '</p>' +
      '<p><strong>Avg ge 0.50 / scan:</strong> ' + fmtNum(modelOutputTail.ge_0_50, 3) + '</p>' +
      '<p><strong>Frac zero ge 0.45 scans:</strong> ' + fmtPct(modelOutput['fraction_of_scans_with_zero_ge_0.45_rows']) + '</p>' +
      '<p><strong>Max score in window:</strong> ' + fmtNum(modelOutput.max_score_seen_in_window, 4) + '</p>' +
    '</div>';
  document.getElementById('modelOutputDistributionPanel').innerHTML = '<div class="grid">' + modelOutputCard + '</div>';

  document.getElementById('statusPanel').innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Current scan state</h3>' +
        '<p><strong>Running:</strong> ' + (scan.running ? 'true' : 'false') + '</p>' +
        '<p><strong>Phase:</strong> ' + (scan.phase || '-') + '</p>' +
        '<p><strong>Scope:</strong> ' + (status.scan_result_scope || '-') + '</p>' +
        '<p><strong>Result sections refer to:</strong> ' + (currentPartial ? 'current partial scan only; last completed shown separately below' : 'current scan') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Last completed scan</h3>' +
        '<p><strong>Available:</strong> ' + (lastCompleted.available ? 'true' : 'false') + '</p>' +
        '<p><strong>Finished:</strong> ' + (lastCompleted.scan_finished_at_utc || '-') + '</p>' +
        '<p><strong>Generated:</strong> ' + (lastCompleted.scan_result_generated_at_utc || '-') + '</p>' +
        '<p><strong>Headline:</strong> ' + (((lastCompleted.decision_summary || {}).headline) || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Market regime</h3>' +
        '<p><strong>State:</strong> ' + (regime.state || '-') + '</p>' +
        '<p><strong>Headline risk:</strong> ' + (regime.headline_risk || '-') + '</p>' +
        '<p><strong>Score:</strong> ' + fmtNum(regime.score, 0) + '</p>' +
        '<p><strong>Cooldown:</strong> ' + (regime.cooldown_active ? 'active' : 'inactive') + '</p>' +
        '<p><strong>Until:</strong> ' + (regime.cooldown_until_utc || '-') + '</p>' +
        '<p class="small"><strong>Reasons:</strong> ' + reasons + '</p>' +
      '</div>' +
      '<div class="card"><h3>Signals policy</h3>' +
        '<p><strong>Suppress new entries:</strong> ' + (regime.suppress_new_entries ? 'true' : 'false') + '</p>' +
        '<p><strong>Actionability:</strong> ' + (regime.actionability_state || '-') + '</p>' +
        '<p><strong>Shock triggered:</strong> ' + (regime.shock_triggered ? 'true' : 'false') + '</p>' +
        '<p><strong>Override:</strong> ' + (regime.override_state || '-') + '</p>' +
        '<p><strong>Determined:</strong> ' + (regime.determined_at_utc || '-') + '</p>' +
        '<p><strong>Source:</strong> ' + (regime.source || '-') + '</p>' +
        '<p><strong>Eligible:</strong> ' + (regime.partial_regime_eligible ? 'true' : 'false') + '</p>' +
        '<p><strong>Attempts:</strong> ' + (regime.partial_publish_attempts || 0) + ' / ok=' + (regime.partial_publish_successes || 0) + ' / fail=' + (regime.partial_publish_failures || 0) + '</p>' +
        '<p><strong>Last attempt:</strong> ' + (regime.last_partial_publish_attempt_utc || '-') + '</p>' +
        '<p><strong>Computed:</strong> ' + (regime.last_computed_state || '-') + ' @ ' + (regime.last_computed_at_utc || '-') + '</p>' +
        '<p><strong>Applied:</strong> ' + (regime.last_applied_state || '-') + ' @ ' + (regime.last_applied_at_utc || '-') + '</p>' +
        '<p><strong>Snapshot versions:</strong> computed=' + (regime.computed_snapshot_version || 0) + ' / applied=' + (regime.applied_snapshot_version || 0) + '</p>' +
        '<p><strong>Warning:</strong> ' + (regime.regime_publish_warning ? (regime.regime_publish_warning_reason || 'true') : 'none') + '</p>' +
        '<p><strong>Scope:</strong> ' + (status.scan_result_scope || '-') + '</p>' +
        '<p><strong>Generated:</strong> ' + (status.scan_result_generated_at_utc || '-') + '</p>' +
        '<p class="small"><strong>Metrics:</strong> btc15m=' + fmtPct(metrics.btc_abs_15m) + ', btc1h=' + fmtPct(metrics.btc_abs_1h) + ', eth1h=' + fmtPct(metrics.eth_abs_1h) + '</p>' +
        '<p class="small"><strong>Waiting on:</strong> ' + safeJoin((regime.readiness || {}).waiting_on) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Coverage</h3>' +
        '<p><strong>Requested:</strong> ' + (cov.symbols_requested_count || 0) + '</p>' +
        '<p><strong>Stage1 ready:</strong> ' + (cov.stage1_feature_ready_count || 0) + '</p>' +
        '<p><strong>Visible rows:</strong> ' + ((status.stage_counts || {}).visible_rows || cov.symbols_scored_count || 0) + '</p>' +
        '<p><strong>Preview rows:</strong> ' + ((status.stage_counts || {}).preview_rows || cov.symbols_previewed_count || 0) + '</p>' +
        '<p><strong>Confirmed rows:</strong> ' + ((status.stage_counts || {}).deep_confirmed_rows || cov.symbols_deep_confirmed_count || 0) + '</p>' +
        '<p><strong>Stage2 ready:</strong> ' + (cov.stage2_feature_ready_count || 0) + '</p>' +
        '<p><strong>Follow-up reserved:</strong> ' + (cov.followup_reserved_symbols || 0) + ' <span class="small">(already present ' + (cov.followup_reserved_existing_symbols || 0) + ')</span></p>' +
        '<p><strong>Cohort mode:</strong> ' + (cov.cohort_mode || '-') + '</p>' +
        '<p><strong>Universe mode:</strong> ' + (status.live_universe_mode_effective || status.live_universe_mode || '-') + ' <span class="small">(req: ' + (status.live_universe_mode_requested || '-') + ')</span></p>' +
        '<p><strong>Configured live selection:</strong> ' + (status.configured_live_selection_mode || '-') + '</p>' +
        '<p><strong>Effective live selection:</strong> ' + (status.effective_live_selection_mode || '-') + '</p>' +
        '<p><strong>Selection engine:</strong> ' + (status.effective_live_selection_engine || status.selection_engine || '-') + '</p>' +
        '<p><strong>Updated:</strong> ' + (status.updated_at_utc || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Decision checkpoint</h3>' +
        '<p><strong>Outcome:</strong> ' + (decisionCheckpoint.decision_checkpoint_outcome || decisionCheckpoint.current_outcome || '-') + '</p>' +
        '<p><strong>Resolved visible rows:</strong> ' + (decisionCheckpoint.resolved_visible_rows || 0) + ' / target=' + (decisionCheckpoint.decision_target_visible_rows || 30) + '</p>' +
        '<p><strong>Visible q-hit:</strong> ' + fmtPct(decisionCheckpoint.current_visible_quality_hit_rate) + '</p>' +
        '<p><strong>Non-visible q-hit:</strong> ' + fmtPct(decisionCheckpoint.current_non_visible_quality_hit_rate) + '</p>' +
        '<p><strong>Rows remaining:</strong> ' + (decisionCheckpoint.rows_remaining_until_decision || 0) + '</p>' +
        '<p><strong>Triggered:</strong> ' + (decisionCheckpoint.triggered ? 'true' : 'false') + '</p>' +
        '<p><strong>Triggered at:</strong> ' + (decisionCheckpoint.decision_checkpoint_triggered_at_utc || '-') + '</p>' +
        '<p class="small"><strong>Summary:</strong> ' + (decisionCheckpoint.summary || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Blocked → Follow-up</h3>' +
        '<p><strong>Scheduled:</strong> ' + (followUp.scheduled ? 'true' : 'false') + '</p>' +
        '<p><strong>Sequence:</strong> ' + (followUp.sequence || '-') + '</p>' +
        '<p><strong>Reason:</strong> ' + (followUp.reason || '-') + '</p>' +
        '<p><strong>Run after:</strong> ' + (followUp.run_after_utc || '-') + '</p>' +
        '<p><strong>Tracked names:</strong> ' + (followUp.tracked_count || blockedContext.tracked_count || 0) + '</p>' +
        '<p><strong>Tracked symbols:</strong> ' + safeJoin(followUp.tracked_symbols || blockedContext.tracked_symbols) + '</p>' +
        '<p><strong>Comparison:</strong> ' + (followupComparison.available ? 'available' : 'pending') + '</p>' +
        '<p><strong>Visible now:</strong> ' + (followupComparison.visible_now_count || 0) + '</p>' +
        '<p><strong>Tracked visible symbols:</strong> ' + safeJoin(followupComparison.tracked_visible_symbols || []) + '</p>' +
        '<p><strong>Still blocked:</strong> ' + (followupComparison.still_blocked_count || 0) + '</p>' +
        '<p><strong>Near visibility now:</strong> ' + (followupComparison.near_visibility_now_count || 0) + '</p>' +
        '<p class="small"><strong>Follow-up verdict:</strong> ' + (decision.followup_headline || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Stage telemetry</h3>' +
        '<p><strong>Stage1 candidates:</strong> ' + ((status.stage_counts || {}).stage1_candidates || 0) + '</p>' +
        '<p><strong>Stage1 selection mode:</strong> ' + (status.stage1_selection_mode || '-') + '</p>' +
        '<p><strong>Stage2 feature-ready:</strong> ' + ((status.coverage || {}).stage2_feature_ready_count || 0) + '</p>' +
        '<p><strong>Stage2 scored total:</strong> ' + ((status.stage_counts || {}).stage2_scored || 0) + '</p>' +
        '<p><strong>Visible after trim:</strong> ' + ((status.stage_counts || {}).stage2_visible_after_trim || 0) + '</p>' +
        '<p><strong>Hidden after trim:</strong> ' + ((status.stage_counts || {}).stage2_hidden_after_trim || 0) + '</p>' +
        '<p><strong>Stage2 partial rows:</strong> ' + ((status.stage_counts || {}).stage2_partial_rows || 0) + '</p>' +
        '<p><strong>Stage2 final rows:</strong> ' + ((status.stage_counts || {}).stage2_final_rows || 0) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Current scan score diagnostics</h3>' +
        '<p><strong>Headline:</strong> ' + (scoreDiag.headline || '-') + '</p>' +
        '<p><strong>Model max / p95 / median:</strong> ' + fmtNum(((scoreDiag.model_score || {}).max), 4) + ' / ' + fmtNum(((scoreDiag.model_score || {}).p95), 4) + ' / ' + fmtNum(((scoreDiag.model_score || {}).median), 4) + '</p>' +
        '<p><strong>Pre-policy max / p95 / median:</strong> ' + fmtNum(((scoreDiag.pre_policy_score || {}).max), 4) + ' / ' + fmtNum(((scoreDiag.pre_policy_score || {}).p95), 4) + ' / ' + fmtNum(((scoreDiag.pre_policy_score || {}).median), 4) + '</p>' +
        '<p><strong>Live max / p95 / median:</strong> ' + fmtNum(((scoreDiag.live_score || {}).max), 4) + ' / ' + fmtNum(((scoreDiag.live_score || {}).p95), 4) + ' / ' + fmtNum(((scoreDiag.live_score || {}).median), 4) + '</p>' +
        '<p><strong>Guardrail cap:</strong> ' + fmtPct(scoreDiag.guardrail_cap) + ' <span class="small">below floor=' + (scoreDiag.guardrail_cap_below_validated_floor ? 'true' : 'false') + '</span></p>' +
        '<p><strong>Funnel:</strong> ranked ' + (scanFunnel.stage2_total_ranked || 0) + ' / visible ' + (scanFunnel.stage2_visible || 0) + ' / informational ' + (scanFunnel.stage2_informational_retained || 0) + ' / overflow ' + (scanFunnel.stage2_informational_overflow || 0) + '</p>' +
      '</div>' +
      '<div class="card"><h3>By liquidity tier</h3>' +
        '<p><strong>Tier1 live max:</strong> ' + fmtNum((((tierDiag.tier1 || {}).live_score || {}).max), 4) + ' <span class="small">rows=' + (((tierDiag.tier1 || {}).rows) || 0) + '</span></p>' +
        '<p><strong>Tier2 live max:</strong> ' + fmtNum((((tierDiag.tier2 || {}).live_score || {}).max), 4) + ' <span class="small">rows=' + (((tierDiag.tier2 || {}).rows) || 0) + '</span></p>' +
        '<p><strong>Tier3 live max:</strong> ' + fmtNum((((tierDiag.tier3 || {}).live_score || {}).max), 4) + ' <span class="small">rows=' + (((tierDiag.tier3 || {}).rows) || 0) + '</span></p>' +
        '<p><strong>Tier1 ≥0.35:</strong> ' + ((((tierDiag.tier1 || {}).counts_above || {})['0.35']) || 0) + '</p>' +
        '<p><strong>Tier2 ≥0.35:</strong> ' + ((((tierDiag.tier2 || {}).counts_above || {})['0.35']) || 0) + '</p>' +
        '<p><strong>Tier3 ≥0.35:</strong> ' + ((((tierDiag.tier3 || {}).counts_above || {})['0.35']) || 0) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Candidate quality</h3>' +
        '<p><strong>Headline:</strong> ' + (candidateQuality.headline || '-') + '</p>' +
        '<p><strong>Stage1 ready / selectable / selected:</strong> ' + (candidateQuality.stage1_feature_ready || 0) + ' / ' + (candidateQuality.stage1_selectable || 0) + ' / ' + (candidateQuality.stage1_selected || 0) + '</p>' +
        '<p><strong>Selected share:</strong> ' + fmtPct(candidateQuality.stage1_selected_share) + ' <span class="small">cap=' + (candidateQuality.configured_stage1_max_candidates || 0) + '</span></p>' +
        '<p><strong>Pass-through warning:</strong> ' + (candidateQuality.stage1_pass_through_warning ? 'true' : 'false') + '</p>' +
        '<p><strong>Top visible stage1 ranks:</strong> ' + safeJoin(candidateQuality.top_visible_stage1_ranks || []) + '</p>' +
        '<p><strong>Selected not scored:</strong> ' + (candidateQuality.selected_not_scored || 0) + '</p>' +
        '<p><strong>Tier1 selected share:</strong> ' + fmtPct((stage1Tier.tier1 || {}).selected_share) + ' <span class="small">live max=' + fmtNum((((candidateTier.tier1 || {}).live_score || {}).max), 4) + '</span></p>' +
        '<p><strong>Tier2 selected share:</strong> ' + fmtPct((stage1Tier.tier2 || {}).selected_share) + ' <span class="small">live max=' + fmtNum((((candidateTier.tier2 || {}).live_score || {}).max), 4) + '</span></p>' +
        '<p><strong>Tier3 selected share:</strong> ' + fmtPct((stage1Tier.tier3 || {}).selected_share) + ' <span class="small">live max=' + fmtNum((((candidateTier.tier3 || {}).live_score || {}).max), 4) + '</span></p>' +
      '</div>' +
      '<div class="card"><h3>Stage1 omission audit</h3>' +
        '<p><strong>Headline:</strong> ' + (stage1OmissionAudit.headline || '-') + '</p>' +
        '<p><strong>Verdict:</strong> ' + (stage1OmissionAudit.verdict || '-') + '</p>' +
        '<p><strong>Omitted audited / total:</strong> ' + (stage1OmissionAudit.omitted_nonblocked_audited || 0) + ' / ' + (stage1OmissionAudit.omitted_nonblocked_total || 0) + '</p>' +
        '<p><strong>Selected max / omitted max:</strong> ' + fmtNum(((stage1OmissionAudit.selected_stage2 || {}).max_live_score), 4) + ' / ' + fmtNum(((stage1OmissionAudit.omitted_stage2 || {}).max_live_score), 4) + '</p>' +
        '<p><strong>Selected ≥0.45 / omitted ≥0.45:</strong> ' + (((stage1OmissionAudit.selected_stage2 || {}).count_ge_0_45) || 0) + ' / ' + (((stage1OmissionAudit.omitted_stage2 || {}).count_ge_0_45) || 0) + '</p>' +
        '<p><strong>Selected ≥0.50 / omitted ≥0.50:</strong> ' + (((stage1OmissionAudit.selected_stage2 || {}).count_ge_0_50) || 0) + ' / ' + (((stage1OmissionAudit.omitted_stage2 || {}).count_ge_0_50) || 0) + '</p>' +
        '<p class="small">' + (stage1OmissionAudit.summary || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Guardrails</h3>' +
        '<p><strong>Blocked:</strong> ' + (guard.blocked || 0) + '</p>' +
        '<p><strong>Event risk:</strong> ' + (guard.event_risk || 0) + '</p>' +
        '<p><strong>Probability capped:</strong> ' + (guard.probability_capped || guard.capped || 0) + '</p>' +
        '<p><strong>Suppressed regime:</strong> ' + (guard.suppressed_regime || 0) + '</p>' +
        '<p><strong>Suppressed threshold:</strong> ' + (guard.suppressed_threshold || 0) + '</p>' +
        '<p><strong>Suppressed cooldown:</strong> ' + (guard.suppressed_cooldown || 0) + '</p>' +
        '<p><strong>Display trimmed:</strong> ' + (cov.dropped_stage2_display_trimmed || cov.dropped_stage2_output_cap || 0) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Decision verdict</h3>' +
        '<p><strong>Headline:</strong> ' + (decision.headline || '-') + '</p>' +
        '<p class="small">' + (decision.summary || '-') + '</p>' +
        '<p><strong>Validated floor:</strong> ' + fmtPct(decision.validated_floor) + '</p>' +
        '<p><strong>Near-validated floor:</strong> ' + fmtPct(decision.near_validated_floor) + '</p>' +
        '<p><strong>Near / exploratory:</strong> ' + (decision.near_validated_rows || 0) + ' / ' + (decision.exploratory_rows || 0) + '</p>' +
        '<p><strong>Hidden watchlist rows:</strong> ' + (decision.hidden_watchlist_rows || 0) + '</p>' +
        '<p><strong>Blocked near-band / near-threshold:</strong> ' + (decision.blocked_near_validated_rows || 0) + ' / ' + (decision.blocked_near_threshold_rows || 0) + '</p>' +
        '<p><strong>Best blocked threshold gap:</strong> ' + (decision.best_blocked_threshold_gap === null || decision.best_blocked_threshold_gap === undefined ? '-' : fmtNum((decision.best_blocked_threshold_gap || 0) * 100, 2) + ' pp') + '</p>' +
        '<p><strong>Focus:</strong> ' + safeJoin((decision.top_focus_symbols || []).map(function(r) { return r.symbol; })) + '</p>' +
        '<p><strong>Blocked focus:</strong> ' + safeJoin((decision.blocked_focus_symbols || []).map(function(r) { return r.symbol; })) + '</p>' +
        '<p><strong>Cooldown until:</strong> ' + (decision.cooldown_until_utc || regime.cooldown_until_utc || '-') + '</p>' +
        '<p><strong>Follow-up scan:</strong> ' + (followUp.run_after_utc || '-') + ' <span class="small">' + (followUp.reason || '') + '</span></p>' +
      '</div>' +
      '<div class="card"><h3>Suppression vs advice</h3>' +
        '<p><strong>Visible rows:</strong> ' + (((status.suppression_summary || {}).visible_rows) || 0) + '</p>' +
        '<p><strong>Threshold suppressed:</strong> ' + (((status.suppression_summary || {}).threshold_suppressed_rows) || 0) + '</p>' +
        '<p><strong>Regime suppressed:</strong> ' + (((status.suppression_summary || {}).regime_suppressed_rows) || 0) + '</p>' +
        '<p><strong>Cooldown suppressed:</strong> ' + (((status.suppression_summary || {}).cooldown_suppressed_rows) || 0) + '</p>' +
        '<p><strong>Trimmed:</strong> ' + (((status.suppression_summary || {}).display_trimmed_rows) || 0) + '</p>' +
        '<p><strong>Informational rows:</strong> ' + (((status.suppression_summary || {}).informational_rows) || 0) + ' <span class="small">(regime ' + (((status.suppression_summary || {}).informational_regime_rows) || 0) + ', cooldown ' + (((status.suppression_summary || {}).informational_cooldown_rows) || 0) + ', threshold ' + (((status.suppression_summary || {}).informational_threshold_rows) || 0) + ', trim ' + (((status.suppression_summary || {}).informational_display_trim_rows) || 0) + ', overflow ' + (((status.suppression_summary || {}).informational_overflow_rows) || 0) + ')</span></p>' +
        '<p><strong>Action-ready / selective / watchlist:</strong> ' + (((status.suppression_summary || {}).action_ready_rows) || 0) + ' / ' + (((status.suppression_summary || {}).selective_rows) || 0) + ' / ' + (((status.suppression_summary || {}).watchlist_rows) || 0) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Score contract</h3>' +
        '<p><strong>Default semantics:</strong> ' + (contract.probability_semantics_default || '-') + '</p>' +
        '<p><strong>Tail state:</strong> ' + (contract.tail_validation_state || '-') + '</p>' +
        '<p><strong>Temporal state:</strong> ' + (contract.temporal_tail_state || '-') + '</p>' +
        '<p><strong>Temporal basis:</strong> ' + (contract.temporal_support_basis || '-') + '</p>' +
        '<p><strong>Validated thresholds:</strong> ' + safeJoin(contract.validated_thresholds) + '</p>' +
        '<p><strong>Highest validated:</strong> ' + (contract.highest_validated_threshold || '-') + '</p>' +
        '<p><strong>Unvalidated cap:</strong> ' + fmtPct(contract.unvalidated_tail_cap) + '</p>' +
        '<p class="small"><strong>Notes:</strong> ' + safeJoin(contract.notes) + '</p>' +
        '<p class="small"><strong>Temporal note:</strong> ' + (contract.temporal_note || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Target</h3>' +
        '<p><strong>Move:</strong> +' + ((target.move_pct || 0.02) * 100).toFixed(1) + '%</p>' +
        '<p><strong>Horizon:</strong> ' + (target.horizon_minutes || '-') + ' min</p>' +
        '<p><strong>Max MAE:</strong> ' + fmtPct(target.quality_max_mae) + '</p>' +
        '<p><strong>Min end ret:</strong> ' + fmtPct(target.quality_min_end_ret) + '</p>' +
      '</div>' +
    '</div>';
}

function scoreRowHtml(row, informational) {
  const stage = stageLabel(row.candidate_stage);
  const scope = scopeLabel(row);
  const rankValue = informational ? (row.informational_rank ?? row.pre_policy_rank ?? row.candidate_rank_all ?? row.would_be_rank) : (row.score_rank);
  const rank = rankValue === null || rankValue === undefined ? '-' : rankValue;
  const gapText = row.distance_to_validated_pct_points === null || row.distance_to_validated_pct_points === undefined ? '-' : fmtNum(row.distance_to_validated_pct_points, 2) + ' pp';
  const preGapText = row.pre_policy_distance_to_validated_pct_points === null || row.pre_policy_distance_to_validated_pct_points === undefined ? '-' : fmtNum(row.pre_policy_distance_to_validated_pct_points, 2) + ' pp';
  const utilityNote = (row.utility_decision_score === undefined && row.utility_confidence === undefined)
    ? ''
    : '<br><span class="small">decision ' + fmtNum(row.utility_decision_score, 3) + ' / conf ' + fmtPct(row.utility_confidence) + '</span><br><span class="small">edge ' + fmtNum(row.utility_expected_edge, 3) + ' / rank ' + (row.utility_rank || '-') + '</span>' ;
  const objectiveNote = row.objective_score_band_label
    ? '<br><span class="small">Edge: ' + row.objective_score_band_label + (row.objective_quality_reference_rate === null || row.objective_quality_reference_rate === undefined ? '' : ' / ref ' + fmtPct(row.objective_quality_reference_rate)) + '</span>'
    : '';
  const bucketNote = informational
    ? '<span class="small">Suppressed: ' + (row.suppression_reason || '-') + '</span><br><span class="small">Detail: ' + (row.suppression_reason_detail || row.policy_constraint_reason || '-') + '</span><br><span class="small">Pre-policy rank: ' + (row.pre_policy_rank || row.candidate_rank_all || '-') + '</span><br><span class="small">Pre-policy band: ' + (row.pre_policy_score_band_label || '-') + ' / gap ' + preGapText + '</span><br><span class="small">Tail band: ' + (row.score_band_label || '-') + ' / gap ' + gapText + '</span>' + objectiveNote + utilityNote
    : '<span class="small">' + (row.actionability_tier || '-') + '</span><br><span class="small">Tail band: ' + (row.score_band_label || '-') + ' / gap ' + gapText + '</span>' + objectiveNote + utilityNote;
  return '<tr>' +
    '<td>' + rank + '</td>' +
    '<td>' + row.symbol + '</td>' +
    '<td>' + fmtNum(row.price, 6) + '</td>' +
    '<td><strong>' + fmtPct(row.live_score || row.prob_2) + '</strong> <span class="pill' + (row.pt2 === 'trained' ? ' pill-lgbm' : '') + '">' + row.pt2 + '</span></td>' +
    '<td>' + fmtNum((row.utility_decision_score ?? row.opportunity_score), row.utility_decision_score === undefined ? 1 : 3) + '</td>' +
    '<td>' + stage + ' <span class="small">(' + scope + ')</span><br>' + bucketNote + '</td>' +
    '<td>' + (row.probability_semantics || row.contract_truth_semantics || '-') + '<br><span class="small">' + (row.objective_score_band_label || '-') + '</span><br><span class="small">' + (row.actionability_type || '-') + '</span></td>' +
    '<td>' + fmtPct(row.prob_2_pre_regime) + '</td>' +
    '<td>' + fmtPct(row.prob_2_model) + '</td>' +
    '<td>' + fmtPct(row.live_threshold) + '</td>' +
    '<td>' + (row.market_regime_state || '-') + '</td>' +
    '<td>' + (row.liquidity_tier || '-') + '</td>' +
    '<td class="' + riskClass(row.risk) + '">' + fmtPct(row.risk) + '</td>' +
    '<td>' + safeJoin(row.reasons) + '<br><span class="small">Advice: ' + (row.actionability_reason || '-') + '</span><br><span class="small">Policy: ' + (row.policy_constraint_reason || '-') + '</span><br><span class="small">Monitor: ' + (row.monitor_priority || '-') + '</span></td>' +
  '</tr>';
}

function blockedFocusRowHtml(row) {
  if (!row) return '';
  const preGap = row.pre_policy_distance_to_validated === null || row.pre_policy_distance_to_validated === undefined ? '-' : fmtNum((row.pre_policy_distance_to_validated || 0) * 100, 2) + ' pp';
  const liveGap = row.distance_to_validated === null || row.distance_to_validated === undefined ? '-' : fmtNum((row.distance_to_validated || 0) * 100, 2) + ' pp';
  const thresholdGap = row.distance_to_live_threshold === null || row.distance_to_live_threshold === undefined ? '-' : fmtNum((row.distance_to_live_threshold || 0) * 100, 2) + ' pp';
  return '<tr>' +
    '<td>' + (row.pre_policy_rank || '-') + '</td>' +
    '<td>' + (row.symbol || '-') + '</td>' +
    '<td><strong>' + fmtPct(row.pre_policy_score) + '</strong><br><span class="small">live ' + fmtPct(row.live_score) + ' / threshold ' + fmtPct(row.live_threshold) + '</span></td>' +
    '<td>' + (row.liquidity_tier || '-') + '</td>' +
    '<td>' + (row.pre_policy_score_band_label || '-') + '<br><span class="small">pre-gap ' + preGap + ' / live-gap ' + liveGap + '</span></td>' +
    '<td>' + (row.visibility_band_label || '-') + '<br><span class="small">threshold gap ' + thresholdGap + '</span></td>' +
    '<td>' + (row.suppression_reason || '-') + '<br><span class="small">' + (row.suppression_reason_detail || '-') + '</span></td>' +
  '</tr>';
}

function renderScores(status, scores) {
  uiState.actionableRows = scores.rows || [];
  uiState.informationalRows = scores.informational_rows || [];
  const rows = sortRows(uiState.actionableRows, 'actionable');
  const informational = sortRows(uiState.informationalRows, 'informational');
  const tbody = document.getElementById('scoreRows');
  const infoBody = document.getElementById('informationalScoreRows');
  const infoCount = document.getElementById('informationalCount');
  tbody.innerHTML = rows.length
    ? rows.map(function(row) { return scoreRowHtml(row, false); }).join('')
    : '<tr><td colspan="14" class="muted">No visible shortlist rows</td></tr>';
  if (infoBody) {
    infoBody.innerHTML = informational.length
      ? informational.map(function(row) { return scoreRowHtml(row, true); }).join('')
      : '<tr><td colspan="14" class="muted">No informational suppressed rows</td></tr>';
  }
  if (infoCount) {
    infoCount.textContent = String(informational.length || 0);
  }
  const blocked = (((status || {}).decision_summary || {}).blocked_focus_symbols) || [];
  const blockedBody = document.getElementById('blockedFocusRows');
  const blockedCount = document.getElementById('blockedFocusCount');
  if (blockedBody) {
    blockedBody.innerHTML = blocked.length
      ? blocked.map(function(row) { return blockedFocusRowHtml(row); }).join('')
      : '<tr><td colspan="7" class="muted">No blocked monitoring rows in focus</td></tr>';
  }
  if (blockedCount) {
    blockedCount.textContent = String(blocked.length || 0);
  }
  updateSortIndicators();
}


function followupChangeRowHtml(row) {
  const priorLive = row.prior_live_score === null || row.prior_live_score === undefined ? '-' : fmtNum(row.prior_live_score, 4);
  const currentLive = row.current_live_score === null || row.current_live_score === undefined ? '-' : fmtNum(row.current_live_score, 4);
  const deltaLive = row.delta_live_score === null || row.delta_live_score === undefined ? '-' : fmtNum(row.delta_live_score, 4);
  let outcome = 'Still blocked';
  if (row.became_visible) outcome = 'Now visible';
  else if (row.missing_current) outcome = 'Not present';
  else if (row.current_visibility_band === 'near_visibility') outcome = 'Near visibility';
  return '<tr>' +
    '<td class="mono">' + (row.symbol || '-') + '</td>' +
    '<td>live ' + priorLive + '<br><span class="small">band ' + (row.prior_score_band || '-') + ' / threshold ' + (row.prior_live_threshold ?? '-') + '</span></td>' +
    '<td>live ' + currentLive + '<br><span class="small">' + ((row.current_row_type || '-') + ' / ' + (row.current_actionability_tier || row.current_visibility_band || '-')) + '</span></td>' +
    '<td>' + deltaLive + '</td>' +
    '<td>' + outcome + '</td>' +
  '</tr>';
}

function renderFollowup(status) {
  const panel = document.getElementById('followupPanel');
  const body = document.getElementById('followupChangesRows');
  const countEl = document.getElementById('followupChangesCount');
  if (!panel || !body) return;
  const followUp = status.follow_up_scan || {};
  const blockedContext = status.blocked_monitoring_context || {};
  const cmp = status.followup_comparison || {};
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Schedule</h3>' +
        '<p><strong>Scheduled:</strong> ' + (followUp.scheduled ? 'true' : 'false') + '</p>' +
        '<p><strong>Reason:</strong> ' + (followUp.reason || '-') + '</p>' +
        '<p><strong>Run after:</strong> ' + (followUp.run_after_utc || '-') + '</p>' +
        '<p><strong>Source scan:</strong> ' + (followUp.source_scan_finished_utc || blockedContext.source_run_finished_utc || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Tracked blocked names</h3>' +
        '<p><strong>Count:</strong> ' + (blockedContext.tracked_count || 0) + '</p>' +
        '<p><strong>Symbols:</strong> ' + safeJoin(blockedContext.tracked_symbols) + '</p>' +
        '<p><strong>Source regime:</strong> ' + (blockedContext.market_regime_state || '-') + ' <span class="small">(' + (blockedContext.market_regime_actionability || '-') + ')</span></p>' +
      '</div>' +
      '<div class="card"><h3>Follow-up comparison</h3>' +
        '<p><strong>Available:</strong> ' + (cmp.available ? 'true' : 'false') + '</p>' +
        '<p><strong>Visible now:</strong> ' + (cmp.visible_now_count || 0) + '</p>' +
        '<p><strong>Tracked visible:</strong> ' + safeJoin(cmp.tracked_visible_symbols || []) + '</p>' +
        '<p><strong>Still blocked:</strong> ' + (cmp.still_blocked_count || 0) + '</p>' +
        '<p><strong>Near visibility now:</strong> ' + (cmp.near_visibility_now_count || 0) + '</p>' +
        '<p><strong>Improved live:</strong> ' + (cmp.improved_live_count || 0) + '</p>' +
        '<p class="small"><strong>Verdict:</strong> ' + (((status.decision_summary || {}).followup_headline) || '-') + '</p>' +
      '</div>' +
    '</div>' +
    '<p class="small">' + ((((status.decision_summary || {}).followup_summary) || 'The comparison appears after a scheduled cooldown follow-up scan runs.')) + '</p>';
  const changes = (cmp.top_changes || []);
  body.innerHTML = changes.length ? changes.map(followupChangeRowHtml).join('') : '<tr><td colspan="5" class="muted">No follow-up comparison rows yet</td></tr>';
  if (countEl) countEl.textContent = String(changes.length || 0);
}

function renderTraining(t) {
  const m = t || {};
  document.getElementById('trainingPanel').innerHTML =
    '<p><strong>Running:</strong> ' + (m.running ? 'true' : 'false') + '</p>' +
    '<p><strong>Status:</strong> ' + (m.message || '-') + '</p>' +
    '<p><strong>Started:</strong> ' + (m.started_at_utc || '-') + '</p>' +
    '<p><strong>Finished:</strong> ' + (m.finished_at_utc || '-') + '</p>';
}

function renderValidation(v) {
  const panel = document.getElementById('validationPanel');
  if (!v || !v.ok) {
    panel.innerHTML = '<p class="muted">' + ((v && v.message) || 'No data yet') + '</p>';
    return;
  }
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Coverage</h3><p><strong>Predictions:</strong> ' + (v.total_predictions || 0) + '</p><p><strong>Resolved:</strong> ' + (v.total_resolved || 0) + '</p><p><strong>Pending:</strong> ' + (v.total_pending || 0) + '</p></div>' +
      '<div class="card"><h3>Overall</h3><p><strong>Quality touch:</strong> ' + fmtPct((v.overall || {}).quality_touch_rate) + '</p><p><strong>Raw touch:</strong> ' + fmtPct((v.overall || {}).raw_touch_rate) + '</p><p><strong>Avg MAE:</strong> ' + fmtPct((v.overall || {}).avg_mae) + '</p><p><strong>Avg end ret:</strong> ' + fmtPct((v.overall || {}).avg_end_ret) + '</p></div>' +
    '</div>';
}

function renderReliability(lab) {
  const panel = document.getElementById('reliabilityPanel');
  if (!lab || !lab.ok) {
    panel.innerHTML = '<p class="muted">' + ((lab && lab.message) || 'No data yet') + '</p>';
    return;
  }
  const h = lab.headline || {};
  const g = lab.reliability_gate || {};
  const e = lab.evidence || {};
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Gate</h3><p><strong>Status:</strong> ' + (g.status || '-') + '</p><p>' + (g.message || '-') + '</p></div>' +
      '<div class="card"><h3>Honesty</h3><p><strong>Live confidence honesty:</strong> ' + fmtPct(h.live_confidence_honesty) + '</p><p><strong>Avg abs gap:</strong> ' + fmtPct(h.avg_abs_calibration_gap) + '</p><p><strong>Quality touch:</strong> ' + fmtPct(h.quality_touch_rate) + '</p></div>' +
      '<div class="card"><h3>Evidence</h3><p><strong>Overall:</strong> ' + ((e.overall || {}).level || '-') + '</p><p><strong>Resolved:</strong> ' + (((e.overall || {}).resolved_predictions) || 0) + '</p><p><strong>High confidence:</strong> ' + ((e.high_confidence || {}).level || '-') + '</p></div>' +
    '</div>';
}


function auditBucketCard(title, bucket) {
  const b = bucket || {};
  return '<div class="card"><h3>' + title + '</h3>' +
    '<p><strong>Count:</strong> ' + (b.count || 0) + '</p>' +
    '<p><strong>Quality hit:</strong> ' + fmtPct(b.quality_hit_rate) + '</p>' +
    '<p><strong>Raw hit:</strong> ' + fmtPct(b.raw_hit_rate) + '</p>' +
    '<p><strong>Avg end ret:</strong> ' + fmtPct(b.avg_end_ret) + '</p>' +
    '<p><strong>Avg MAE:</strong> ' + fmtPct(b.avg_mae) + '</p>' +
  '</div>';
}

function renderPolicyAudit(day24, day7) {
  const panel = document.getElementById('policyAuditPanel');
  if (!panel) return;
  if (!day24 || !day7) {
    panel.innerHTML = '<p class="muted">Policy audit unavailable.</p>';
    return;
  }
  function reasonLine(audit, key) {
    const b = ((audit || {}).suppressed_by_reason || {})[key] || {};
    return key + ': ' + (b.count || 0) + ' / qhit=' + fmtPct(b.quality_hit_rate) + ' / avgEnd=' + fmtPct(b.avg_end_ret);
  }
  function regimeRows(audit) {
    const rows = audit.regime_breakdown || {};
    const entries = Object.keys(rows).sort().map(function(k) {
      const r = rows[k] || {};
      const vis = r.visible || {};
      const sup = r.suppressed || {};
      return '<tr><td>' + k + '</td><td>' + (r.total || 0) + '</td><td>' + (vis.count || 0) + '</td><td>' + fmtPct(vis.quality_hit_rate) + '</td><td>' + (sup.count || 0) + '</td><td>' + fmtPct(sup.quality_hit_rate) + '</td></tr>';
    }).join('');
    return '<table><thead><tr><th>Regime</th><th>Total</th><th>Visible</th><th>Visible q-hit</th><th>Suppressed</th><th>Suppressed q-hit</th></tr></thead><tbody>' + (entries || '<tr><td colspan="6" class="muted">No evaluated regime data yet</td></tr>') + '</tbody></table>';
  }
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Last 24h</h3>' +
        '<p><strong>Runs:</strong> ' + (day24.run_count || 0) + ' / completed=' + (day24.completed_run_count || 0) + '</p>' +
        '<p><strong>Evaluated rows:</strong> ' + (day24.evaluated_rows || 0) + '</p>' +
        '<p><strong>False suppressions (quality/raw):</strong> ' + (day24.false_suppressions_quality_count || 0) + ' / ' + (day24.false_suppressions_raw_count || 0) + '</p>' +
        '<p><strong>Bad visible rows:</strong> ' + (day24.bad_visible_rows_count || 0) + '</p>' +
        '<p><strong>Overblock gap:</strong> ' + fmtPct(day24.policy_overblock_gap_quality) + '</p>' +
        '<p><strong>Protection gap:</strong> ' + fmtPct(day24.policy_protection_gap_end_ret) + '</p>' +
        '<p class="small"><strong>Suppression reasons:</strong><br>' + reasonLine(day24, 'regime') + '<br>' + reasonLine(day24, 'cooldown') + '<br>' + reasonLine(day24, 'threshold') + '<br>' + reasonLine(day24, 'display_trim') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Last 7d</h3>' +
        '<p><strong>Runs:</strong> ' + (day7.run_count || 0) + ' / completed=' + (day7.completed_run_count || 0) + '</p>' +
        '<p><strong>Evaluated rows:</strong> ' + (day7.evaluated_rows || 0) + '</p>' +
        '<p><strong>False suppressions (quality/raw):</strong> ' + (day7.false_suppressions_quality_count || 0) + ' / ' + (day7.false_suppressions_raw_count || 0) + '</p>' +
        '<p><strong>Bad visible rows:</strong> ' + (day7.bad_visible_rows_count || 0) + '</p>' +
        '<p><strong>Overblock gap:</strong> ' + fmtPct(day7.policy_overblock_gap_quality) + '</p>' +
        '<p><strong>Protection gap:</strong> ' + fmtPct(day7.policy_protection_gap_end_ret) + '</p>' +
        '<p class="small"><strong>Suppression reasons:</strong><br>' + reasonLine(day7, 'regime') + '<br>' + reasonLine(day7, 'cooldown') + '<br>' + reasonLine(day7, 'threshold') + '<br>' + reasonLine(day7, 'display_trim') + '</p>' +
      '</div>' +
      auditBucketCard('Visible rows (24h)', day24.visible) +
      auditBucketCard('Suppressed rows (24h)', day24.suppressed) +
    '</div>' +
    '<h3 style="margin-top:16px;">Regime effectiveness (24h)</h3>' + regimeRows(day24);
}


function setReplayProgress(running, text) {
  const progressText = document.getElementById('replayProgressText');
  const elapsedText = document.getElementById('replayElapsedText');
  const bar = document.getElementById('replayProgressBar');
  const runBtn = document.getElementById('runReplayButton');
  const loadBtn = document.getElementById('loadReplaySummaryButton');
  const downloadBtn = document.getElementById('downloadReplayPackButton');
  replayUiState.running = !!running;
  if (progressText) progressText.textContent = text || (running ? 'Running replay…' : 'Idle');
  if (bar) bar.classList.toggle('running', !!running);
  if (runBtn) runBtn.disabled = !!running;
  if (loadBtn) loadBtn.disabled = !!running;
  if (downloadBtn) downloadBtn.disabled = !!running;
  if (replayUiState.timer) {
    clearInterval(replayUiState.timer);
    replayUiState.timer = null;
  }
  if (running) {
    replayUiState.startedAtMs = Date.now();
    if (elapsedText) elapsedText.textContent = 'Replay is running. This indicator is time-based, not server-side percentage progress.';
    replayUiState.timer = setInterval(function() {
      const secs = Math.max(0, Math.floor((Date.now() - replayUiState.startedAtMs) / 1000));
      const mins = Math.floor(secs / 60);
      const rem = secs % 60;
      if (elapsedText) elapsedText.textContent = 'Elapsed: ' + mins + 'm ' + rem + 's — waiting for replay completion.';
    }, 1000);
  } else {
    replayUiState.startedAtMs = null;
    if (elapsedText) elapsedText.textContent = 'No replay currently running.';
  }
}

function renderReplaySummary(summary) {
  const panel = document.getElementById('replayPanel');
  if (!panel) return;
  if (!summary || !summary.available) {
    panel.innerHTML = '<p class="muted">Replay summary unavailable.</p>';
    return;
  }
  const windowInfo = summary.window || {};
  const visible = summary.visible_bucket || {};
  const nonVisible = summary.non_visible_bucket || {};
  const counter = summary.counterfactual || {};
  const outlier = summary.outlier_concentration || {};
  const topScans = (summary.top_scans || []).slice(0, 5).map(function(row) {
    return '<tr><td>' + (row.as_of_utc || '-') + '</td><td>' + (row.market_regime_state || '-') + '</td><td>' + (row.visible_rows || 0) + '</td><td>' + fmtNum(row.live_max, 4) + '</td><td>' + (row.validated_rows || 0) + '</td><td>' + (row.decision_headline || '-') + '</td></tr>';
  }).join('');
  const missedRows = (counter.top_missed_quality_rows || []).slice(0, 5).map(function(row) {
    return '<tr><td>' + (row.symbol || '-') + '</td><td>' + fmtNum(row.stage1_rank_all, 0) + '</td><td>' + fmtPct(row.end_ret) + '</td><td>' + fmtPct(row.mfe) + '</td><td>' + fmtPct(row.mae) + '</td></tr>';
  }).join('');
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Replay window</h3>' +
        '<p><strong>Headline:</strong> ' + (summary.headline || '-') + '</p>' +
        '<p><strong>Start:</strong> ' + (windowInfo.start_utc || '-') + '</p>' +
        '<p><strong>End:</strong> ' + (windowInfo.end_utc || '-') + '</p>' +
        '<p><strong>Scans:</strong> ' + (windowInfo.scan_count || 0) + '</p>' +
        '<p><strong>Step minutes:</strong> ' + (windowInfo.scan_step_minutes || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Visible vs hidden</h3>' +
        '<p><strong>Visible quality hit:</strong> ' + fmtPct(visible.quality_hit_rate) + '</p>' +
        '<p><strong>Visible raw hit:</strong> ' + fmtPct(visible.raw_hit_rate) + '</p>' +
        '<p><strong>Non-visible quality hit:</strong> ' + fmtPct(nonVisible.quality_hit_rate) + '</p>' +
        '<p><strong>Non-visible raw hit:</strong> ' + fmtPct(nonVisible.raw_hit_rate) + '</p>' +
        '<p><strong>Visible avg end ret:</strong> ' + fmtPct(visible.avg_end_ret) + '</p>' +
        '<p><strong>Non-visible avg end ret:</strong> ' + fmtPct(nonVisible.avg_end_ret) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Stage1 counterfactual</h3>' +
        '<p><strong>Headline:</strong> ' + (counter.headline || '-') + '</p>' +
        '<p><strong>Stage1 quality recall:</strong> ' + fmtPct(counter.stage1_quality_recall) + '</p>' +
        '<p><strong>Selectable quality opps:</strong> ' + (counter.selectable_quality_opportunities || 0) + '</p>' +
        '<p><strong>Selected quality opps:</strong> ' + (counter.selected_quality_opportunities || 0) + '</p>' +
        '<p><strong>Missed quality opps:</strong> ' + (counter.missed_quality_opportunities || 0) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Concentration</h3>' +
        '<p><strong>Headline:</strong> ' + (outlier.headline || '-') + '</p>' +
        '<p><strong>Summary:</strong> ' + (outlier.summary || '-') + '</p>' +
      '</div>' +
    '</div>' +
    '<h3 style="margin-top:16px;">Pipeline ablation</h3>' +
    '<table><thead><tr><th>Mode</th><th>Visible q-hit</th><th>Visible raw hit</th><th>Visible avg end ret</th><th>Stage1 recall</th></tr></thead><tbody>' + (((summary.pipeline_ablation || {}).rows || []).map(function(row) { return '<tr><td>' + (row.mode || '-') + '</td><td>' + fmtPct(row.visible_quality_hit_rate) + '</td><td>' + fmtPct(row.visible_raw_hit_rate) + '</td><td>' + fmtPct(row.visible_avg_end_ret) + '</td><td>' + fmtPct(row.stage1_quality_recall) + '</td></tr>'; }).join('') || '<tr><td colspan="5" class="muted">No pipeline ablation comparison in this replay summary.</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Top replay scans</h3>' +
    '<table><thead><tr><th>As of</th><th>Regime</th><th>Visible</th><th>Live max</th><th>Validated rows</th><th>Headline</th></tr></thead><tbody>' + (topScans || '<tr><td colspan="6" class="muted">No replay scans yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Top missed quality rows</h3>' +
    '<table><thead><tr><th>Symbol</th><th>Stage1 rank</th><th>End ret</th><th>MFE</th><th>MAE</th></tr></thead><tbody>' + (missedRows || '<tr><td colspan="5" class="muted">No missed quality rows in this replay</td></tr>') + '</tbody></table>';
}

function adminPasswordOrThrow() {
  const replayPw = (document.getElementById('replayAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = replayPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

async function fetchBlob(url, opts) {
  const r = await fetch(url, opts || {});
  if (!r.ok) throw new Error(url + ' failed: ' + r.status);
  return await r.blob();
}

async function runReplay() {
  const msg = document.getElementById('replayMessage');
  msg.textContent = 'Starting replay…';
  try {
    const pw = adminPasswordOrThrow();
    const hours = Number(document.getElementById('replayHours').value || 24);
    const step = Number(document.getElementById('replayStepMinutes').value || 60);
    const maxScans = Number(document.getElementById('replayMaxScans').value || 24);
    const maxSymbols = Number(document.getElementById('replayMaxSymbols').value || 100);
    const pipelineMode = (document.getElementById('replayPipelineMode') || {}).value || 'full';
    const rawThreshold = Number((document.getElementById('replayRawThreshold') || {}).value || 0.30);
    const params = new URLSearchParams({
      hours: String(hours),
      step_minutes: String(step),
      max_scans: String(maxScans),
      max_symbols: String(maxSymbols),
      pipeline_mode: String(pipelineMode),
      raw_threshold: String(rawThreshold),
    });
    setReplayProgress(true, 'Running replay…');
    const out = await getJson('/api/replay/run?' + params.toString(), { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderReplaySummary(out.summary || null);
    msg.textContent = 'Replay complete';
    setReplayProgress(false, 'Replay complete');
  } catch (e) {
    msg.textContent = e.message;
    setReplayProgress(false, 'Replay failed');
  }
}

async function loadReplaySummary() {
  const msg = document.getElementById('replayMessage');
  msg.textContent = 'Loading replay summary…';
  setReplayProgress(false, 'Loading latest replay summary');
  try {
    const pw = adminPasswordOrThrow();
    const summary = await getJson('/api/replay/latest-summary', { headers: { 'X-Admin-Password': pw } });
    renderReplaySummary(summary);
    msg.textContent = 'Replay summary loaded';
  } catch (e) {
    msg.textContent = e.message;
  }
}

async function downloadReplayPack() {
  const msg = document.getElementById('replayMessage');
  msg.textContent = 'Downloading replay pack…';
  setReplayProgress(false, 'Preparing replay pack download');
  try {
    const pw = adminPasswordOrThrow();
    const blob = await fetchBlob('/api/replay/latest.zip', { headers: { 'X-Admin-Password': pw } });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'replay_pack.zip';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    msg.textContent = 'Replay pack downloaded';
  } catch (e) {
    msg.textContent = e.message;
  }
}


function stage1OpportunityPasswordOrThrow() {
  const localPw = (document.getElementById('stage1OpportunityAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderStage1OpportunitySummary(summary) {
  const panel = document.getElementById('stage1OpportunityPanel');
  if (!panel) return;
  if (!summary || !summary.available) {
    panel.innerHTML = '<p class="muted">Stage1 opportunity scorer summary unavailable.</p>';
    return;
  }
  const positiveRows = (summary.top_positive_weights || []).map(function(row) {
    return '<tr><td>' + (row.feature || '-') + '</td><td>' + fmtNum(row.weight, 4) + '</td></tr>';
  }).join('');
  const negativeRows = (summary.top_negative_weights || []).map(function(row) {
    return '<tr><td>' + (row.feature || '-') + '</td><td>' + fmtNum(row.weight, 4) + '</td></tr>';
  }).join('');
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Scorer summary</h3>' +
        '<p><strong>Trained at:</strong> ' + (summary.trained_at_utc || '-') + '</p>' +
        '<p><strong>Source replay pack:</strong> ' + (summary.source_replay_pack || '-') + '</p>' +
        '<p><strong>Rows:</strong> ' + (summary.row_count_all || 0) + '</p>' +
        '<p><strong>Validation AUC:</strong> ' + fmtNum(summary.auc_validation, 4) + '</p>' +
        '<p><strong>Validation Brier:</strong> ' + fmtNum(summary.brier_validation, 4) + '</p>' +
        '<p><strong>Positive rate:</strong> ' + fmtPct(summary.positive_rate_all) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Supported stage1 modes</h3><p>' + ((summary.selection_modes_supported || []).join(', ') || '-') + '</p></div>' +
    '</div>' +
    '<h3 style="margin-top:16px;">Top positive weights</h3>' +
    '<table><thead><tr><th>Feature</th><th>Weight</th></tr></thead><tbody>' + (positiveRows || '<tr><td colspan="2" class="muted">No positive weights</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Top negative weights</h3>' +
    '<table><thead><tr><th>Feature</th><th>Weight</th></tr></thead><tbody>' + (negativeRows || '<tr><td colspan="2" class="muted">No negative weights</td></tr>') + '</tbody></table>';
}

async function buildStage1OpportunityFromReplay() {
  const msg = document.getElementById('stage1OpportunityMessage');
  msg.textContent = 'Building stage1 opportunity scorer…';
  try {
    const pw = stage1OpportunityPasswordOrThrow();
    const out = await getJson('/api/stage1-opportunity/build-from-replay', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderStage1OpportunitySummary(out.summary || null);
    msg.textContent = 'Stage1 opportunity scorer built';
  } catch (e) {
    msg.textContent = e.message;
  }
}

async function loadStage1OpportunitySummary() {
  const msg = document.getElementById('stage1OpportunityMessage');
  msg.textContent = 'Loading stage1 opportunity scorer summary…';
  try {
    const pw = stage1OpportunityPasswordOrThrow();
    const summary = await getJson('/api/stage1-opportunity/summary', { headers: { 'X-Admin-Password': pw } });
    renderStage1OpportunitySummary(summary);
    msg.textContent = 'Stage1 opportunity scorer summary loaded';
  } catch (e) {
    msg.textContent = e.message;
  }
}

function modelAuditPasswordOrThrow() {
  const localPw = (document.getElementById('modelAuditAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderModelAuditSummary(summary) {
  const panel = document.getElementById('modelAuditPanel');
  if (!panel) return;
  if (!summary || !summary.available) {
    panel.innerHTML = '<p class="muted">Model audit summary unavailable.</p>';
    return;
  }
  const calRows = (summary.calibration_deciles || []).map(function(row) {
    return '<tr><td>' + (row.bucket || '-') + '</td><td>' + (row.count || 0) + '</td><td>' + fmtPct(row.predicted_mean) + '</td><td>' + fmtPct(row.actual_rate) + '</td><td>' + fmtNum(row.min_score, 4) + '</td><td>' + fmtNum(row.max_score, 4) + '</td></tr>';
  }).join('');
  const symbolRows = (summary.per_symbol_rows || []).slice(0, 10).map(function(row) {
    return '<tr><td>' + (row.symbol || '-') + '</td><td>' + (row.resolved_rows || 0) + '</td><td>' + fmtPct(row.quality_hit_rate) + '</td><td>' + fmtPct(row.raw_hit_rate) + '</td><td>' + fmtPct(row.avg_end_ret) + '</td><td>' + fmtNum(row.max_prob_2_model, 4) + '</td></tr>';
  }).join('');
  const tail = summary.tail_precision || {};
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Model audit</h3>' +
        '<p><strong>Generated at:</strong> ' + (summary.generated_at_utc || '-') + '</p>' +
        '<p><strong>Source replay pack:</strong> ' + (summary.source_replay_pack || '-') + '</p>' +
        '<p><strong>Resolved rows:</strong> ' + (summary.resolved_row_count || 0) + '</p>' +
        '<p><strong>Positive rate:</strong> ' + fmtPct(summary.positive_rate) + '</p>' +
        '<p><strong>AUC:</strong> ' + fmtNum(summary.auc, 4) + '</p>' +
        '<p><strong>Brier:</strong> ' + fmtNum(summary.brier, 4) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Tail precision</h3>' +
        '<p><strong>Top 10% precision:</strong> ' + fmtPct(((tail.top_10pct || {}).precision)) + ' <span class="small">(' + (((tail.top_10pct || {}).count) || 0) + ' rows)</span></p>' +
        '<p><strong>Top 5% precision:</strong> ' + fmtPct(((tail.top_5pct || {}).precision)) + ' <span class="small">(' + (((tail.top_5pct || {}).count) || 0) + ' rows)</span></p>' +
        '<p><strong>Replay headline:</strong> ' + (summary.source_replay_headline || '-') + '</p>' +
      '</div>' +
    '</div>' +
    '<h3 style="margin-top:16px;">Calibration deciles</h3>' +
    '<table><thead><tr><th>Bucket</th><th>Count</th><th>Pred mean</th><th>Actual rate</th><th>Min</th><th>Max</th></tr></thead><tbody>' + (calRows || '<tr><td colspan="6" class="muted">No calibration rows</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Top symbols (>=10 resolved rows)</h3>' +
    '<table><thead><tr><th>Symbol</th><th>Rows</th><th>Q-hit</th><th>Raw hit</th><th>Avg end ret</th><th>Max model</th></tr></thead><tbody>' + (symbolRows || '<tr><td colspan="6" class="muted">No symbol rows</td></tr>') + '</tbody></table>';
}

async function buildModelAuditFromReplay() {
  const msg = document.getElementById('modelAuditMessage');
  msg.textContent = 'Building model audit…';
  try {
    const pw = modelAuditPasswordOrThrow();
    const out = await getJson('/api/model-audit/build-from-replay', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderModelAuditSummary(out.summary || null);
    msg.textContent = 'Model audit built';
  } catch (e) {
    msg.textContent = e.message;
  }
}

async function loadModelAuditSummary() {
  const msg = document.getElementById('modelAuditMessage');
  msg.textContent = 'Loading model audit summary…';
  try {
    const pw = modelAuditPasswordOrThrow();
    const summary = await getJson('/api/model-audit/summary', { headers: { 'X-Admin-Password': pw } });
    renderModelAuditSummary(summary);
    msg.textContent = 'Model audit summary loaded';
  } catch (e) {
    msg.textContent = e.message;
  }
}



function rawScoreBaselinePasswordOrThrow() {
  const localPw = (document.getElementById('rawScoreBaselineAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderRawScoreBaselineSummary(summary) {
  const panel = document.getElementById('rawScoreBaselinePanel');
  if (!panel) return;
  if (!summary || !summary.available) {
    panel.innerHTML = '<p class="muted">Raw score baseline summary unavailable.</p>';
    return;
  }
  const raw = (summary.raw_model_score_distribution || {});
  const rawQ = raw.score_quantiles || {};
  const preQ = ((summary.pre_policy_score_distribution || {}).score_quantiles) || {};
  const liveQ = ((summary.live_score_distribution || {}).score_quantiles) || {};
  const topPct = raw.top_bucket_quality_rate || {};
  const lifts = raw.top_bucket_lift || {};
  const scanTop = summary.scan_topk_quality || {};
  const diag = summary.diagnosis || {};
  const comp = summary.compression_summary || {};
  const topRows = ['top_1pct', 'top_5pct', 'top_10pct'].map(function(key) {
    const row = topPct[key] || {};
    return '<tr><td>' + key + '</td><td>' + (row.count || 0) + '</td><td>' + fmtPct(row.quality_rate) + '</td><td>' + fmtNum(lifts[key], 3) + 'x</td><td>' + fmtNum(row.score_min, 4) + '</td><td>' + fmtNum(row.score_max, 4) + '</td></tr>';
  }).join('');
  const scanRows = ['top_1', 'top_3', 'top_5'].map(function(key) {
    const row = scanTop[key] || {};
    return '<tr><td>' + key + '</td><td>' + (row.scan_count || 0) + '</td><td>' + fmtPct(row.mean_quality_rate) + '</td><td>' + fmtNum(row.lift_vs_base, 3) + 'x</td><td>' + fmtPct(row.share_of_scans_with_hit) + '</td></tr>';
  }).join('');
  const quantileRows = ['q50', 'q75', 'q90', 'q95', 'q99', 'max'].map(function(key) {
    return '<tr><td>' + key + '</td><td>' + fmtNum(rawQ[key], 4) + '</td><td>' + fmtNum(preQ[key], 4) + '</td><td>' + fmtNum(liveQ[key], 4) + '</td></tr>';
  }).join('');
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Baseline verdict</h3>' +
        '<p><strong>Headline:</strong> ' + (summary.headline || '-') + '</p>' +
        '<p><strong>Primary blocker:</strong> ' + (diag.primary_blocker || '-') + '</p>' +
        '<p><strong>Ranking strength:</strong> ' + (diag.ranking_strength || '-') + '</p>' +
        '<p><strong>Tail state:</strong> ' + (diag.tail_state || '-') + '</p>' +
        '<p><strong>Compression significant:</strong> ' + (diag.compression_significant ? 'true' : 'false') + '</p>' +
        '<p class="small"><strong>Recommended next tranche:</strong> ' + (diag.recommended_next_tranche || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Resolved scope</h3>' +
        '<p><strong>Resolved rows:</strong> ' + (summary.resolved_row_count || 0) + '</p>' +
        '<p><strong>Scans:</strong> ' + (summary.scan_count || 0) + '</p>' +
        '<p><strong>Symbols:</strong> ' + (summary.symbol_count || 0) + '</p>' +
        '<p><strong>Base quality rate:</strong> ' + fmtPct(summary.base_quality_rate) + '</p>' +
        '<p><strong>Source replay headline:</strong> ' + (((summary.source_replay || {}).headline) || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Raw upper tail</h3>' +
        '<p><strong>Dead upper tail:</strong> ' + (raw.dead_upper_tail ? 'true' : 'false') + '</p>' +
        '<p><strong>q90 / q95 / q99:</strong> ' + fmtNum(rawQ.q90, 4) + ' / ' + fmtNum(rawQ.q95, 4) + ' / ' + fmtNum(rawQ.q99, 4) + '</p>' +
        '<p><strong>Max:</strong> ' + fmtNum(rawQ.max, 4) + '</p>' +
        '<p><strong>Top 10% lift:</strong> ' + fmtNum(lifts.top_10pct, 3) + 'x</p>' +
        '<p><strong>Top 5% lift:</strong> ' + fmtNum(lifts.top_5pct, 3) + 'x</p>' +
        '<p><strong>Top 1% lift:</strong> ' + fmtNum(lifts.top_1pct, 3) + 'x</p>' +
      '</div>' +
      '<div class="card"><h3>Compression check</h3>' +
        '<p><strong>Raw-live q99 gap:</strong> ' + fmtNum(comp.raw_minus_live_q99, 4) + '</p>' +
        '<p><strong>Raw-live max gap:</strong> ' + fmtNum(comp.raw_minus_live_max, 4) + '</p>' +
        '<p><strong>Avg post-model penalty:</strong> ' + fmtNum(comp.average_post_model_total_penalty, 4) + '</p>' +
        '<p><strong>Penalized rows:</strong> ' + fmtPct(comp.penalized_row_fraction) + '</p>' +
        '<p><strong>Capped rows:</strong> ' + fmtPct(comp.capped_row_fraction) + '</p>' +
      '</div>' +
    '</div>' +
    '<h3 style="margin-top:16px;">Raw top-percentile quality</h3>' +
    '<table><thead><tr><th>Bucket</th><th>Count</th><th>Quality rate</th><th>Lift vs base</th><th>Min score</th><th>Max score</th></tr></thead><tbody>' + (topRows || '<tr><td colspan="6" class="muted">No percentile rows</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Per-scan unconditional top-k by raw model</h3>' +
    '<table><thead><tr><th>Bucket</th><th>Scans</th><th>Mean quality</th><th>Lift vs base</th><th>Share of scans with hit</th></tr></thead><tbody>' + (scanRows || '<tr><td colspan="5" class="muted">No scan top-k rows</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Score quantiles</h3>' +
    '<table><thead><tr><th>Quantile</th><th>Raw model</th><th>Pre-policy</th><th>Live</th></tr></thead><tbody>' + (quantileRows || '<tr><td colspan="4" class="muted">No quantile rows</td></tr>') + '</tbody></table>';
}

async function runRawScoreBaseline() {
  const msg = document.getElementById('rawScoreBaselineMessage');
  const runBtn = document.getElementById('runRawScoreBaselineButton');
  const loadBtn = document.getElementById('loadRawScoreBaselineSummaryButton');
  const packBtn = document.getElementById('downloadRawScoreBaselinePackButton');
  if (msg) msg.textContent = 'Running raw score baseline…';
  if (runBtn) runBtn.disabled = true;
  if (loadBtn) loadBtn.disabled = true;
  if (packBtn) packBtn.disabled = true;
  try {
    const pw = rawScoreBaselinePasswordOrThrow();
    const form = new FormData();
    form.append('hours', String(Number((document.getElementById('rawScoreBaselineHours') || {}).value || 168)));
    form.append('step_minutes', String(Number((document.getElementById('rawScoreBaselineStepMinutes') || {}).value || 120)));
    form.append('max_scans', String(Number((document.getElementById('rawScoreBaselineMaxScans') || {}).value || 84)));
    form.append('max_symbols', String(Number((document.getElementById('rawScoreBaselineMaxSymbols') || {}).value || 100)));
    const res = await fetch('/api/reviews/raw-score-baseline/run', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: form });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderRawScoreBaselineSummary(out.summary || null);
    if (msg) msg.textContent = 'Raw score baseline completed';
  } catch (err) {
    if (msg) msg.textContent = 'Could not run raw score baseline: ' + err.message;
  } finally {
    if (runBtn) runBtn.disabled = false;
    if (loadBtn) loadBtn.disabled = false;
    if (packBtn) packBtn.disabled = false;
  }
}

async function loadRawScoreBaselineSummary(silent) {
  const msg = document.getElementById('rawScoreBaselineMessage');
  if (!silent && msg) msg.textContent = 'Loading raw score baseline summary…';
  try {
    const res = await fetch('/api/reviews/raw-score-baseline/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderRawScoreBaselineSummary(out);
    if (!silent && msg) msg.textContent = 'Raw score baseline summary loaded';
  } catch (err) {
    if (!silent && msg) msg.textContent = 'Could not load raw score baseline summary: ' + err.message;
  }
}

function downloadRawScoreBaselinePack() {
  const msg = document.getElementById('rawScoreBaselineMessage');
  try {
    const pw = rawScoreBaselinePasswordOrThrow();
    if (msg) msg.textContent = 'Opening raw score baseline pack…';
    const url = '/api/reviews/raw-score-baseline/latest-pack.zip?admin_password=' + encodeURIComponent(pw);
    window.open(url, '_blank');
  } catch (err) {
    if (msg) msg.textContent = 'Could not open raw score baseline pack: ' + err.message;
  }
}



function decisionBranchPasswordOrThrow() {
  const localPw = (document.getElementById('decisionBranchAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function setDecisionBranchButtonsBusy(isBusy) {
  const ids = [
    'toggleDecisionBranchAutoExecuteButton',
    'executeDecisionBranchButton',
    'clearDecisionBranchOverrideButton',
    'ackDecisionBranchButton'
  ];
  ids.forEach(function(id) {
    const el = document.getElementById(id);
    if (el) el.disabled = Boolean(isBusy) || el.disabled;
  });
}

function renderDecisionBranchAutomation(summary) {
  const panel = document.getElementById('decisionBranchPanel');
  const msg = document.getElementById('decisionBranchMessage');
  if (!panel) return;
  uiState.decisionBranchSummary = summary || null;
  if (!summary) {
    panel.innerHTML = '<p class="muted">Decision branch automation unavailable.</p>';
    return;
  }
  const action = summary.branch_action || {};
  const overrides = summary.runtime_overrides || {};
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Branch state</h3>' +
        '<p><strong>Checkpoint outcome:</strong> ' + (summary.checkpoint_outcome || '-') + '</p>' +
        '<p><strong>Status:</strong> ' + (action.status || '-') + '</p>' +
        '<p><strong>Headline:</strong> ' + (action.headline || '-') + '</p>' +
        '<p class="small"><strong>Summary:</strong> ' + (action.summary || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Execution settings</h3>' +
        '<p><strong>Auto-execute supported actions:</strong> ' + (summary.auto_execute_supported_actions_enabled ? 'enabled' : 'disabled') + '</p>' +
        '<p><strong>Supported action:</strong> ' + (action.next_action_label || '-') + '</p>' +
        '<p><strong>Manual required:</strong> ' + (action.manual_required ? 'true' : 'false') + '</p>' +
        '<p><strong>Can execute now:</strong> ' + (action.can_execute_now ? 'true' : 'false') + '</p>' +
        '<p><strong>Pending notification:</strong> ' + (summary.branch_notification_pending ? 'true' : 'false') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Runtime override</h3>' +
        '<p><strong>Configured threshold:</strong> ' + fmtPct(summary.configured_live_raw_threshold) + '</p>' +
        '<p><strong>Effective threshold:</strong> ' + fmtPct(summary.effective_live_raw_threshold) + '</p>' +
        '<p><strong>Override active:</strong> ' + (overrides.threshold_experiment_active ? 'true' : 'false') + '</p>' +
        '<p><strong>Override source:</strong> ' + (overrides.override_source || '-') + '</p>' +
        '<p><strong>Applied:</strong> ' + (overrides.override_applied_at_utc || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Last execution</h3>' +
        '<p><strong>Action:</strong> ' + (summary.last_execution_action || '-') + '</p>' +
        '<p><strong>Result:</strong> ' + (summary.last_execution_result || '-') + '</p>' +
        '<p><strong>At:</strong> ' + (summary.last_execution_at_utc || '-') + '</p>' +
        '<p class="small"><strong>Manual follow-up:</strong> ' + (summary.manual_follow_up_note || '-') + '</p>' +
      '</div>' +
    '</div>';
  const toggleBtn = document.getElementById('toggleDecisionBranchAutoExecuteButton');
  const execBtn = document.getElementById('executeDecisionBranchButton');
  const clearBtn = document.getElementById('clearDecisionBranchOverrideButton');
  const ackBtn = document.getElementById('ackDecisionBranchButton');
  if (toggleBtn) {
    toggleBtn.textContent = summary.auto_execute_supported_actions_enabled ? 'Disable auto-execute' : 'Enable auto-execute';
    toggleBtn.title = 'Turn automatic post-checkpoint actions on or off.';
  }
  if (execBtn) {
    execBtn.disabled = !action.can_execute_now;
    execBtn.title = action.can_execute_now ? 'Execute the currently supported branch action now.' : 'No executable branch action is available right now.';
  }
  if (clearBtn) {
    clearBtn.disabled = !action.can_clear_override;
    clearBtn.title = action.can_clear_override ? 'Clear the active runtime override and return to the configured threshold.' : 'No active runtime override to clear.';
  }
  if (ackBtn) {
    ackBtn.disabled = !action.can_acknowledge;
    ackBtn.title = action.can_acknowledge ? 'Acknowledge this branch state and clear the pending notification.' : 'No branch state is available to acknowledge.';
  }
  if (msg && !msg.textContent) msg.textContent = '';
}

async function toggleDecisionBranchAutoExecute() {
  const msg = document.getElementById('decisionBranchMessage');
  msg.textContent = 'Updating auto-execute setting…';
  setDecisionBranchButtonsBusy(true);
  try {
    const pw = decisionBranchPasswordOrThrow();
    const current = uiState.decisionBranchSummary || {};
    const enabled = !Boolean(current.auto_execute_supported_actions_enabled);
    await getJson('/api/reviews/decision-branch/auto-execute?enabled=' + encodeURIComponent(String(enabled)), {
      method: 'POST',
      headers: { 'X-Admin-Password': pw }
    });
    msg.textContent = 'Auto-execute setting updated';
    await refreshAll();
  } catch (e) {
    msg.textContent = e.message;
  } finally {
    await refreshAll();
  }
}

async function executeDecisionBranchNow() {
  const msg = document.getElementById('decisionBranchMessage');
  msg.textContent = 'Executing supported branch action…';
  setDecisionBranchButtonsBusy(true);
  try {
    const pw = decisionBranchPasswordOrThrow();
    await getJson('/api/reviews/decision-branch/execute', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    msg.textContent = 'Decision branch action processed';
    await refreshAll();
  } catch (e) {
    msg.textContent = e.message;
  } finally {
    await refreshAll();
  }
}

async function clearDecisionBranchOverride() {
  const msg = document.getElementById('decisionBranchMessage');
  msg.textContent = 'Clearing active runtime override…';
  setDecisionBranchButtonsBusy(true);
  try {
    const pw = decisionBranchPasswordOrThrow();
    await getJson('/api/reviews/decision-branch/clear-override', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    msg.textContent = 'Active runtime override cleared';
    await refreshAll();
  } catch (e) {
    msg.textContent = e.message;
  } finally {
    await refreshAll();
  }
}

async function acknowledgeDecisionBranch() {
  const msg = document.getElementById('decisionBranchMessage');
  msg.textContent = 'Acknowledging decision branch…';
  setDecisionBranchButtonsBusy(true);
  try {
    const pw = decisionBranchPasswordOrThrow();
    await getJson('/api/reviews/decision-branch/ack', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    msg.textContent = 'Decision branch acknowledged';
    await refreshAll();
  } catch (e) {
    msg.textContent = e.message;
  } finally {
    await refreshAll();
  }
}

function benchmarkPasswordOrThrow() {
  const localPw = (document.getElementById('benchmarkAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderBenchmarkSummary(summary) {
  const panel = document.getElementById('benchmarkPanel');
  if (!panel) return;
  if (!summary || !(summary.rows || []).length) {
    panel.innerHTML = '<p class="muted">No benchmark summary available yet.</p>';
    return;
  }
  const rec = summary.recommendation || {};
  const benchmarkRows = (summary.rows || []).map(function(row) {
    return '<tr><td>' + fmtNum(row.threshold, 2) + '</td><td>' + fmtPct(row.visible_quality_hit_rate) + '</td><td>' + fmtPct(row.non_visible_quality_hit_rate) + '</td><td>' + fmtPct(row.visible_avg_end_ret) + '</td><td>' + (row.visible_rows || 0) + '</td><td>' + fmtPct(row.stage1_quality_recall) + '</td><td>' + (row.top_symbol_at_0_45 || '-') + '</td><td>' + fmtPct(row.top_symbol_share_at_0_45) + '</td></tr>';
  }).join('');
  const classes = summary.symbol_classification || {};
  function classificationRows(items, hitField) {
    return (items || []).map(function(row) {
      const hitKey = hitField || 'quality_hit_rate';
      return '<tr><td>' + (row.symbol || '-') + '</td><td>' + (row.resolved_rows || 0) + '</td><td>' + (row.visible_rows || 0) + '</td><td>' + (row.non_visible_rows || 0) + '</td><td>' + fmtPct(row[hitKey]) + '</td><td>' + fmtPct(row.avg_end_ret) + '</td><td>' + fmtNum(row.max_live_score, 4) + '</td></tr>';
    }).join('');
  }
  const liveSummary = summary.live_current_version || {};
  const liveEvidence = liveSummary.evidence || {};
  const modelAudit = summary.model_audit || {};
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Benchmark sweep</h3>' +
        '<p><strong>Pipeline mode:</strong> ' + (summary.pipeline_mode || '-') + '</p>' +
        '<p><strong>Window:</strong> ' + (summary.hours || 0) + 'h / every ' + (summary.step_minutes || 0) + 'm</p>' +
        '<p><strong>Thresholds:</strong> ' + ((summary.thresholds || []).map(function(v){ return fmtNum(v, 2); }).join(', ') || '-') + '</p>' +
        '<p><strong>Recommended threshold:</strong> ' + (rec.recommended_threshold === null || rec.recommended_threshold === undefined ? '-' : fmtNum(rec.recommended_threshold, 2)) + '</p>' +
        '<p class="small"><strong>Reason:</strong> ' + (rec.reason || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Current live slice</h3>' +
        '<p><strong>Evaluated packs:</strong> ' + (liveSummary.evaluated_pack_count || 0) + '</p>' +
        '<p><strong>Visible q-hit:</strong> ' + fmtPct(liveEvidence.visible_quality_hit_rate) + '</p>' +
        '<p><strong>Hidden q-hit:</strong> ' + fmtPct(liveEvidence.non_visible_quality_hit_rate) + '</p>' +
        '<p><strong>Visible avg end ret:</strong> ' + fmtPct(liveEvidence.visible_avg_end_ret) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Model audit snapshot</h3>' +
        '<p><strong>AUC:</strong> ' + fmtNum(modelAudit.auc, 4) + '</p>' +
        '<p><strong>Brier:</strong> ' + fmtNum(modelAudit.brier, 4) + '</p>' +
        '<p><strong>Top 10% precision:</strong> ' + fmtPct((((modelAudit.tail_precision || {}).top_10pct || {}).precision)) + '</p>' +
        '<p><strong>Top 5% precision:</strong> ' + fmtPct((((modelAudit.tail_precision || {}).top_5pct || {}).precision)) + '</p>' +
      '</div>' +
    '</div>' +
    '<h3 style="margin-top:16px;">Threshold comparison</h3>' +
    '<table><thead><tr><th>Threshold</th><th>Visible q-hit</th><th>Hidden q-hit</th><th>Visible avg end ret</th><th>Visible rows</th><th>Stage1 recall</th><th>Top >=0.45 symbol</th><th>Top share</th></tr></thead><tbody>' + (benchmarkRows || '<tr><td colspan="8" class="muted">No benchmark rows yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Repeat winners</h3>' +
    '<table><thead><tr><th>Symbol</th><th>Resolved</th><th>Visible</th><th>Hidden</th><th>Q-hit</th><th>Avg end ret</th><th>Max live</th></tr></thead><tbody>' + (classificationRows(classes.repeat_winners) || '<tr><td colspan="7" class="muted">No repeat winners yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Hidden outperformers</h3>' +
    '<table><thead><tr><th>Symbol</th><th>Resolved</th><th>Visible</th><th>Hidden</th><th>Hidden q-hit</th><th>Avg end ret</th><th>Max live</th></tr></thead><tbody>' + (classificationRows(classes.hidden_outperformers, 'non_visible_quality_hit_rate') || '<tr><td colspan="7" class="muted">No hidden outperformers yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Repeat disappointments</h3>' +
    '<table><thead><tr><th>Symbol</th><th>Resolved</th><th>Visible</th><th>Hidden</th><th>Q-hit</th><th>Avg end ret</th><th>Max live</th></tr></thead><tbody>' + (classificationRows(classes.repeat_disappointments) || '<tr><td colspan="7" class="muted">No repeat disappointments yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Visible underperformers</h3>' +
    '<table><thead><tr><th>Symbol</th><th>Resolved</th><th>Visible</th><th>Hidden</th><th>Visible q-hit</th><th>Avg end ret</th><th>Max live</th></tr></thead><tbody>' + (classificationRows(classes.visible_underperformers, 'visible_quality_hit_rate') || '<tr><td colspan="7" class="muted">No visible underperformers yet</td></tr>') + '</tbody></table>';
}

async function runBenchmarkSweep() {
  const msg = document.getElementById('benchmarkMessage');
  msg.textContent = 'Running benchmark threshold sweep…';
  try {
    const pw = benchmarkPasswordOrThrow();
    const hours = Number(document.getElementById('benchmarkHours').value || 96);
    const step = Number(document.getElementById('benchmarkStepMinutes').value || 120);
    const maxScans = Number(document.getElementById('benchmarkMaxScans').value || 48);
    const maxSymbols = Number(document.getElementById('benchmarkMaxSymbols').value || 100);
    const thresholds = (document.getElementById('benchmarkThresholds').value || '0.25,0.30,0.35,0.40');
    const params = new URLSearchParams({
      hours: String(hours),
      step_minutes: String(step),
      max_scans: String(maxScans),
      max_symbols: String(maxSymbols),
      thresholds: thresholds,
    });
    const out = await getJson('/api/benchmark/run-threshold-sweep?' + params.toString(), { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderBenchmarkSummary(out.summary || null);
    msg.textContent = 'Benchmark threshold sweep finished';
  } catch (e) {
    msg.textContent = e.message;
  }
}



function downloadBenchmarkPack() {
  const msg = document.getElementById('benchmarkMessage');
  try {
    const pw = benchmarkPasswordOrThrow();
    const url = '/api/benchmark/latest-pack.zip?admin_password=' + encodeURIComponent(pw);
    window.open(url, '_blank');
    msg.textContent = 'Benchmark pack download started';
  } catch (e) {
    msg.textContent = e.message;
  }
}

function downloadClassificationPack() {
  const msg = document.getElementById('benchmarkMessage');
  try {
    const pw = benchmarkPasswordOrThrow();
    const url = '/api/benchmark/latest-classification-pack.zip?admin_password=' + encodeURIComponent(pw);
    window.open(url, '_blank');
    msg.textContent = 'Classification pack download started';
  } catch (e) {
    msg.textContent = e.message;
  }
}

async function loadBenchmarkSummary() {
  const msg = document.getElementById('benchmarkMessage');
  msg.textContent = 'Loading benchmark summary…';
  try {
    const pw = benchmarkPasswordOrThrow();
    const summary = await getJson('/api/benchmark/summary', { headers: { 'X-Admin-Password': pw } });
    renderBenchmarkSummary(summary);
    msg.textContent = 'Benchmark summary loaded';
  } catch (e) {
    msg.textContent = e.message;
  }
}

function utilityModelLabPasswordOrThrow() {
  const localPw = (document.getElementById('utilityModelLabAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderUtilityModelLabSummary(summary) {
  const panel = document.getElementById('utilityModelLabPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility model lab summary loaded yet.</p>';
    return;
  }
  const inc = summary.incumbent_shortlist_metrics || {};
  const cand = summary.utility_model_shortlist_metrics || {};
  const deltas = summary.best_vs_incumbent_deltas || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Incumbent utility / quality gap / win rate:</strong> ' + fmtNum(inc.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(inc.visible_hidden_quality_gap) + ' / ' + fmtPct(inc.scan_pairwise_win_rate) + '</p>' +
    '<p><strong>Challenger utility / quality gap / win rate:</strong> ' + fmtNum(cand.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(cand.visible_hidden_quality_gap) + ' / ' + fmtPct(cand.scan_pairwise_win_rate) + '</p>' +
    '<p><strong>Utility / utility-gap / quality-gap delta:</strong> ' + fmtNum(deltas.scan_shortlist_utility_score_delta, 4) + ' / ' + fmtNum(deltas.visible_hidden_utility_gap_delta, 4) + ' / ' + fmtPct(deltas.visible_hidden_quality_gap_delta) + '</p>' +
    '<p><strong>Pairwise win delta / top-1 delta:</strong> ' + fmtPct(deltas.scan_pairwise_win_rate_delta) + ' / ' + fmtPct(deltas.scan_top1_quality_delta) + '</p>';
}

async function runUtilityModelLab() {
  const msg = document.getElementById('utilityModelLabMessage');
  if (msg) msg.textContent = 'Running utility model lab…';
  try {
    const pw = utilityModelLabPasswordOrThrow();
    const fd = new FormData();
    fd.append('max_symbols', String(Number(document.getElementById('utilityModelLabMaxSymbols').value || 0)));
    fd.append('visible_cap', String(Number(document.getElementById('utilityModelLabVisibleCap').value || 0)));
    const res = await fetch('/api/utility-model-lab/run', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelLabSummary(out);
    if (msg) msg.textContent = 'Utility model lab completed';
  } catch (err) {
    if (msg) msg.textContent = 'Could not run utility model lab: ' + err.message;
  }
}

async function loadUtilityModelLabSummary(silent = false) {
  const msg = document.getElementById('utilityModelLabMessage');
  if (!silent && msg) msg.textContent = 'Loading utility model lab summary…';
  try {
    const res = await fetch('/api/utility-model-lab/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelLabSummary(out);
    if (!silent && msg) msg.textContent = 'Utility model lab summary loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility model lab summary: ' + err.message;
  }
}

function downloadUtilityModelLabPack() {
  const msg = document.getElementById('utilityModelLabMessage');
  if (msg) msg.textContent = 'Opening utility model lab pack…';
  window.open('/api/utility-model-lab/latest-pack.zip', '_blank');
}

function utilityTuningLabPasswordOrThrow() {
  const localPw = (document.getElementById('utilityTuningLabAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderUtilityTuningLabSummary(summary) {
  const panel = document.getElementById('utilityTuningLabPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility tuning lab summary loaded yet.</p>';
    return;
  }
  const best = summary.best_candidate || {};
  const baseline = summary.baseline_candidate || {};
  const deltas = summary.best_vs_baseline_deltas || {};
  const overrides = summary.recommended_env_overrides || {};
  const overrideText = Object.keys(overrides).length ? Object.entries(overrides).map(([k,v]) => `${k}=${v}`).join('<br>') : '-';
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Baseline / best candidate:</strong> ' + escapeHtml((baseline.engine_label || '-') + ' / ' + (best.engine_label || '-')) + '</p>' +
    '<p><strong>Utility / gap / win-rate delta:</strong> ' + fmtNum(deltas.scan_shortlist_utility_score_delta, 4) + ' / ' + fmtPct(deltas.scan_shortlist_mean_gap_delta) + ' / ' + fmtPct(deltas.scan_shortlist_pairwise_win_rate_delta) + '</p>' +
    '<p><strong>Top-1 delta / avg visible rows delta:</strong> ' + fmtPct(deltas.scan_shortlist_top1_visible_quality_delta) + ' / ' + fmtNum(deltas.scan_shortlist_avg_visible_rows_per_scan_delta, 2) + '</p>' +
    '<p><strong>Recommended env overrides:</strong><br>' + overrideText + '</p>';
}

async function runUtilityTuningLab() {
  const msg = document.getElementById('utilityTuningLabMessage');
  if (msg) msg.textContent = 'Running utility tuning lab…';
  try {
    const pw = utilityTuningLabPasswordOrThrow();
    const fd = new FormData();
    fd.append('hours', String(Number(document.getElementById('utilityTuningLabHours').value || 168)));
    fd.append('step_minutes', String(Number(document.getElementById('utilityTuningLabStepMinutes').value || 120)));
    fd.append('max_scans', String(Number(document.getElementById('utilityTuningLabMaxScans').value || 84)));
    fd.append('max_symbols', String(Number(document.getElementById('utilityTuningLabMaxSymbols').value || 100)));
    const res = await fetch('/api/utility-tuning-lab/run', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningLabSummary(out);
    if (msg) msg.textContent = 'Utility tuning lab completed';
  } catch (err) {
    if (msg) msg.textContent = 'Utility tuning lab failed: ' + err.message;
  }
}

async function loadUtilityTuningLabSummary(silent = false) {
  const msg = document.getElementById('utilityTuningLabMessage');
  if (!silent && msg) msg.textContent = 'Loading utility tuning lab summary…';
  try {
    const res = await fetch('/api/utility-tuning-lab/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningLabSummary(out);
    if (!silent && msg) msg.textContent = 'Utility tuning lab summary loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility tuning lab summary: ' + err.message;
  }
}

function downloadUtilityTuningLabPack() {
  const msg = document.getElementById('utilityTuningLabMessage');
  if (msg) msg.textContent = 'Opening utility tuning lab pack…';
  window.open('/api/utility-tuning-lab/latest-pack.zip', '_blank');
}

function utilitySelectionLabPasswordOrThrow() {
  const localPw = (document.getElementById('utilitySelectionLabAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderUtilitySelectionLabSummary(summary) {
  const panel = document.getElementById('utilitySelectionLabPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility selection lab summary loaded yet.</p>';
    return;
  }
  const utilityEngine = summary.utility_engine || {};
  const legacyEngine = summary.legacy_engine || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Utility engine score / gap / pairwise:</strong> ' + fmtNum(utilityEngine.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utilityEngine.scan_shortlist_mean_gap) + ' / ' + fmtPct(utilityEngine.scan_shortlist_pairwise_win_rate) + '</p>' +
    '<p><strong>Legacy engine score / gap / pairwise:</strong> ' + fmtNum(legacyEngine.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(legacyEngine.scan_shortlist_mean_gap) + ' / ' + fmtPct(legacyEngine.scan_shortlist_pairwise_win_rate) + '</p>' +
    '<p><strong>Utility vs legacy top-1 visible quality:</strong> ' + fmtPct(utilityEngine.scan_shortlist_top1_visible_quality) + ' / ' + fmtPct(legacyEngine.scan_shortlist_top1_visible_quality) + '</p>' +
    '<p><strong>Utility vs legacy avg visible rows/scan:</strong> ' + fmtNum(utilityEngine.scan_shortlist_avg_visible_rows_per_scan, 2) + ' / ' + fmtNum(legacyEngine.scan_shortlist_avg_visible_rows_per_scan, 2) + '</p>';
}

async function runUtilitySelectionLab() {
  const msg = document.getElementById('utilitySelectionLabMessage');
  if (msg) msg.textContent = 'Running utility selection lab…';
  try {
    const pw = utilitySelectionLabPasswordOrThrow();
    const fd = new FormData();
    fd.append('hours', String(Number(document.getElementById('utilitySelectionLabHours').value || 168)));
    fd.append('step_minutes', String(Number(document.getElementById('utilitySelectionLabStepMinutes').value || 120)));
    fd.append('max_scans', String(Number(document.getElementById('utilitySelectionLabMaxScans').value || 84)));
    fd.append('max_symbols', String(Number(document.getElementById('utilitySelectionLabMaxSymbols').value || 100)));
    const res = await fetch('/api/utility-selection-lab/run', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilitySelectionLabSummary(out);
    if (msg) msg.textContent = 'Utility selection lab completed';
  } catch (err) {
    if (msg) msg.textContent = 'Utility selection lab failed: ' + err.message;
  }
}

async function loadUtilitySelectionLabSummary() {
  const msg = document.getElementById('utilitySelectionLabMessage');
  if (msg) msg.textContent = 'Loading utility selection lab summary…';
  try {
    const res = await fetch('/api/utility-selection-lab/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilitySelectionLabSummary(out);
    if (msg) msg.textContent = 'Utility selection lab summary loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility selection lab summary: ' + err.message;
  }
}

function downloadUtilitySelectionLabPack() {
  const msg = document.getElementById('utilitySelectionLabMessage');
  if (msg) msg.textContent = 'Opening utility selection lab pack…';
  window.open('/api/utility-selection-lab/latest-pack.zip', '_blank');
}

function historicalDecisionLabPasswordOrThrow() {
  const localPw = (document.getElementById('historicalDecisionLabAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderHistoricalDecisionLabSummary(summary) {
  const panel = document.getElementById('historicalDecisionLabPanel');
  if (!panel) return;
  if (!summary || !summary.available) {
    panel.innerHTML = '<p class="muted">No historical decision lab summary available yet.</p>';
    return;
  }
  const live = summary.current_live_evidence || {};
  const conc = summary.upper_tail_concentration || {};
  const c45 = conc.threshold_0_45 || {};
  const c60 = conc.threshold_0_60 || {};
  const mod = summary.model_output_distribution || {};
  const retrain = conc.future_retrain_spec || {};
  const rec = summary.benchmark_recommendation || {};
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Live action</h3>' +
        '<p><strong>Action:</strong> ' + (summary.live_action || '-') + '</p>' +
        '<p><strong>Why:</strong> ' + (summary.live_action_reason || '-') + '</p>' +
        '<p><strong>Live threshold:</strong> ' + fmtNum(summary.live_current_threshold, 2) + '</p>' +
        '<p><strong>Visible q-hit:</strong> ' + fmtPct(live.visible_quality_hit_rate) + '</p>' +
        '<p><strong>Hidden q-hit:</strong> ' + fmtPct(live.non_visible_quality_hit_rate) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Offline sweep</h3>' +
        '<p><strong>Recommended threshold:</strong> ' + (rec.recommended_threshold === null || rec.recommended_threshold === undefined ? '-' : fmtNum(rec.recommended_threshold, 2)) + '</p>' +
        '<p><strong>Delta vs live:</strong> ' + fmtNum(summary.benchmark_threshold_delta_vs_live, 4) + '</p>' +
        '<p class="small"><strong>Reason:</strong> ' + (rec.reason || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Upper-tail concentration</h3>' +
        '<p><strong>&ge;0.45 top symbol:</strong> ' + (c45.top_symbol || '-') + '</p>' +
        '<p><strong>&ge;0.45 top share:</strong> ' + fmtPct(c45.top_symbol_share) + '</p>' +
        '<p><strong>&ge;0.60 top symbol:</strong> ' + (c60.top_symbol || '-') + '</p>' +
        '<p><strong>&ge;0.60 top share:</strong> ' + fmtPct(c60.top_symbol_share) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Replay tail density</h3>' +
        '<p><strong>Headline:</strong> ' + (mod.headline || '-') + '</p>' +
        '<p><strong>Scans:</strong> ' + (mod.scans_in_window || 0) + '</p>' +
        '<p><strong>Avg &ge;0.45 / scan:</strong> ' + fmtNum(((mod.average_upper_tail_counts_per_scan || {})['ge_0.45']), 3) + '</p>' +
        '<p><strong>Frac zero &ge;0.45 scans:</strong> ' + fmtPct(mod['fraction_of_scans_with_zero_ge_0.45_rows']) + '</p>' +
      '</div>' +
    '</div>' +
    '<div class="card" style="margin-top:14px;">' +
      '<p><strong>Future retrain-spec note:</strong> ' + (retrain.reason || '-') + '</p>' +
      '<p><strong>Include symbol concentration controls:</strong> ' + ((retrain.future_retrain_spec_should_include_symbol_concentration_controls) ? 'yes' : 'no') + '</p>' +
    '</div>';
}

async function runHistoricalDecisionLab() {
  const msg = document.getElementById('historicalDecisionLabMessage');
  msg.textContent = 'Running historical decision lab…';
  try {
    const pw = historicalDecisionLabPasswordOrThrow();
    const hours = Number(document.getElementById('historicalDecisionLabHours').value || 168);
    const step = Number(document.getElementById('historicalDecisionLabStepMinutes').value || 120);
    const maxScans = Number(document.getElementById('historicalDecisionLabMaxScans').value || 84);
    const maxSymbols = Number(document.getElementById('historicalDecisionLabMaxSymbols').value || 100);
    const thresholds = (document.getElementById('historicalDecisionLabThresholds').value || '0.30,0.35,0.40');
    const params = new URLSearchParams({
      hours: String(hours),
      step_minutes: String(step),
      max_scans: String(maxScans),
      max_symbols: String(maxSymbols),
      thresholds: thresholds,
    });
    const out = await getJson('/api/historical-decision-lab/run?' + params.toString(), { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderHistoricalDecisionLabSummary(out.summary || null);
    msg.textContent = 'Historical decision lab finished';
  } catch (e) {
    msg.textContent = e.message;
  }
}

async function loadHistoricalDecisionLabSummary() {
  const msg = document.getElementById('historicalDecisionLabMessage');
  msg.textContent = 'Loading historical decision lab summary…';
  try {
    const pw = historicalDecisionLabPasswordOrThrow();
    const summary = await getJson('/api/historical-decision-lab/summary', { headers: { 'X-Admin-Password': pw } });
    renderHistoricalDecisionLabSummary(summary);
    msg.textContent = 'Historical decision lab summary loaded';
  } catch (e) {
    msg.textContent = e.message;
  }
}

function downloadHistoricalDecisionLabPack() {
  const msg = document.getElementById('historicalDecisionLabMessage');
  try {
    const pw = historicalDecisionLabPasswordOrThrow();
    const url = '/api/historical-decision-lab/latest-pack.zip?admin_password=' + encodeURIComponent(pw);
    window.open(url, '_blank');
    msg.textContent = 'Historical decision lab pack download started';
  } catch (e) {
    msg.textContent = e.message;
  }
}


function stage1PolicyLabPasswordOrThrow() {
  const localPw = (document.getElementById('stage1PolicyLabAdminPassword') || {}).value || '';
  const sharedPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || sharedPw;
  if (!pw) throw new Error('Enter admin password first');
  return pw;
}

function renderStage1PolicyLabSummary(summary) {
  const panel = document.getElementById('stage1PolicyLabPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No Stage 1 policy lab summary available yet.</p>';
    return;
  }
  const rows = summary.policy_rows || [];
  const best = summary.best_policy || {};
  const baseline = summary.baseline_policy || {};
  const deltas = summary.best_vs_baseline_deltas || {};
  let table = '<table><thead><tr><th>Policy</th><th>Utility</th><th>Gap</th><th>Win rate</th><th>Top1</th><th>Top3</th><th>Avg vis/scan</th><th>Vis Q</th><th>Recall</th></tr></thead><tbody>';
  rows.forEach((row) => {
    table += '<tr>' +
      '<td>' + escapeHtml(row.label || '-') + '</td>' +
      '<td>' + fmtNum(row.scan_shortlist_utility_score, 4) + '</td>' +
      '<td>' + fmtPct(row.scan_shortlist_mean_gap) + '</td>' +
      '<td>' + fmtPct(row.scan_shortlist_pairwise_win_rate) + '</td>' +
      '<td>' + fmtPct(row.scan_shortlist_top1_visible_quality) + '</td>' +
      '<td>' + fmtPct(row.scan_shortlist_top3_visible_quality) + '</td>' +
      '<td>' + fmtNum(row.scan_shortlist_avg_visible_rows_per_scan, 2) + '</td>' +
      '<td>' + fmtPct(row.visible_quality_hit_rate) + '</td>' +
      '<td>' + fmtPct(row.stage1_quality_recall) + '</td>' +
      '</tr>';
  });
  table += '</tbody></table>';
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Reason:</strong> ' + escapeHtml(summary.recommended_action_reason || '-') + '</p>' +
    '<p><strong>Live baseline:</strong> ' + escapeHtml((baseline.label || ((summary.live_baseline || {}).stage1_selection_mode || '-') + '@' + ((summary.live_baseline || {}).stage1_max_candidates || '-'))) + '</p>' +
    '<p><strong>Best policy:</strong> ' + escapeHtml(best.label || '-') + '</p>' +
    '<p><strong>Utility / gap / win-rate delta:</strong> ' + fmtNum(deltas.scan_shortlist_utility_score_delta, 4) + ' / ' + fmtPct(deltas.scan_shortlist_mean_gap_delta) + ' / ' + fmtPct(deltas.scan_shortlist_pairwise_win_rate_delta) + '</p>' +
    '<p><strong>Avg visible rows/scan delta:</strong> ' + fmtNum(deltas.scan_shortlist_avg_visible_rows_per_scan_delta, 2) + '</p>' +
    table;
}

async function runStage1PolicyLab() {
  const msg = document.getElementById('stage1PolicyLabMessage');
  if (msg) msg.textContent = 'Running Stage 1 policy lab…';
  try {
    const pw = stage1PolicyLabPasswordOrThrow();
    const fd = new FormData();
    fd.append('hours', String(Number(document.getElementById('stage1PolicyLabHours').value || 168)));
    fd.append('step_minutes', String(Number(document.getElementById('stage1PolicyLabStepMinutes').value || 120)));
    fd.append('max_scans', String(Number(document.getElementById('stage1PolicyLabMaxScans').value || 84)));
    fd.append('max_symbols', String(Number(document.getElementById('stage1PolicyLabMaxSymbols').value || 100)));
    const res = await fetch('/api/reviews/stage1-policy-lab/run', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderStage1PolicyLabSummary(out);
    if (msg) msg.textContent = 'Stage 1 policy lab completed';
  } catch (err) {
    if (msg) msg.textContent = 'Stage 1 policy lab failed: ' + err.message;
  }
}

async function loadStage1PolicyLabSummary(silent = false) {
  const msg = document.getElementById('stage1PolicyLabMessage');
  if (!silent && msg) msg.textContent = 'Loading Stage 1 policy lab summary…';
  try {
    const res = await fetch('/api/reviews/stage1-policy-lab/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderStage1PolicyLabSummary(out);
    if (!silent && msg) msg.textContent = 'Stage 1 policy lab summary loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load Stage 1 policy lab summary: ' + err.message;
  }
}

function downloadStage1PolicyLabPack() {
  const msg = document.getElementById('stage1PolicyLabMessage');
  if (msg) msg.textContent = 'Opening Stage 1 policy lab pack…';
  window.open('/api/reviews/stage1-policy-lab/latest-pack.zip', '_blank');
}


function nextLiveCandidateLabPasswordOrThrow() {
  const localPw = (document.getElementById('nextLiveCandidateLabAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderNextLiveCandidateLabSummary(summary) {
  const panel = document.getElementById('nextLiveCandidateLabPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No next live candidate lab summary loaded yet.</p>';
    return;
  }
  const rows = Array.isArray(summary.candidate_rows) ? summary.candidate_rows : [];
  const best = summary.best_combo || {};
  const baseline = summary.baseline_combo || {};
  const deltas = summary.best_vs_baseline_deltas || {};
  let table = '<table><thead><tr><th>Combo</th><th>Utility</th><th>Gap</th><th>Win rate</th><th>Top1</th><th>Avg vis/scan</th></tr></thead><tbody>';
  rows.slice(0, 8).forEach((row) => {
    table += '<tr>' +
      '<td>' + escapeHtml(row.label || '-') + '</td>' +
      '<td>' + fmtNum(row.scan_shortlist_utility_score, 4) + '</td>' +
      '<td>' + fmtPct(row.scan_shortlist_mean_gap) + '</td>' +
      '<td>' + fmtPct(row.scan_shortlist_pairwise_win_rate) + '</td>' +
      '<td>' + fmtPct(row.scan_shortlist_top1_visible_quality) + '</td>' +
      '<td>' + fmtNum(row.scan_shortlist_avg_visible_rows_per_scan, 2) + '</td>' +
      '</tr>';
  });
  table += '</tbody></table>';
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Reason:</strong> ' + escapeHtml(summary.recommended_action_reason || '-') + '</p>' +
    '<p><strong>Baseline:</strong> ' + escapeHtml((baseline.label || 'incumbent_live_pt2__' + (((summary.live_baseline || {}).stage1_selection_mode || '-') + '@' + ((summary.live_baseline || {}).stage1_max_candidates || '-')))) + '</p>' +
    '<p><strong>Best combo:</strong> ' + escapeHtml(best.label || '-') + '</p>' +
    '<p><strong>Utility / gap / win-rate delta:</strong> ' + fmtNum(deltas.scan_shortlist_utility_score_delta, 4) + ' / ' + fmtPct(deltas.scan_shortlist_mean_gap_delta) + ' / ' + fmtPct(deltas.scan_shortlist_pairwise_win_rate_delta) + '</p>' +
    '<p><strong>Top1 delta / avg visible rows delta:</strong> ' + fmtPct(deltas.scan_shortlist_top1_visible_quality_delta) + ' / ' + fmtNum(deltas.scan_shortlist_avg_visible_rows_per_scan_delta, 2) + '</p>' +
    table;
}

async function runNextLiveCandidateLab() {
  const msg = document.getElementById('nextLiveCandidateLabMessage');
  if (msg) msg.textContent = 'Running next live candidate lab…';
  try {
    const pw = nextLiveCandidateLabPasswordOrThrow();
    const fd = new FormData();
    fd.append('hours', String(Number(document.getElementById('nextLiveCandidateLabHours').value || 168)));
    fd.append('step_minutes', String(Number(document.getElementById('nextLiveCandidateLabStepMinutes').value || 120)));
    fd.append('max_scans', String(Number(document.getElementById('nextLiveCandidateLabMaxScans').value || 84)));
    fd.append('max_symbols', String(Number(document.getElementById('nextLiveCandidateLabMaxSymbols').value || 100)));
    const res = await fetch('/api/reviews/next-live-candidate-lab/run', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderNextLiveCandidateLabSummary(out);
    if (msg) msg.textContent = 'Next live candidate lab completed';
  } catch (err) {
    if (msg) msg.textContent = 'Next live candidate lab failed: ' + err.message;
  }
}

async function loadNextLiveCandidateLabSummary(silent = false) {
  const msg = document.getElementById('nextLiveCandidateLabMessage');
  if (!silent && msg) msg.textContent = 'Loading next live candidate lab summary…';
  try {
    const res = await fetch('/api/reviews/next-live-candidate-lab/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderNextLiveCandidateLabSummary(out);
    if (!silent && msg) msg.textContent = 'Next live candidate lab summary loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load next live candidate lab summary: ' + err.message;
  }
}

function downloadNextLiveCandidateLabPack() {
  const msg = document.getElementById('nextLiveCandidateLabMessage');
  if (msg) msg.textContent = 'Opening next live candidate lab pack…';
  window.open('/api/reviews/next-live-candidate-lab/latest-pack.zip', '_blank');
}


function liveCandidateProofPasswordOrThrow() {
  const localPw = (document.getElementById('liveCandidateProofAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderLiveCandidateProofSummary(summary) {
  const panel = document.getElementById('liveCandidateProofPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No controlled live proof summary loaded yet.</p>';
    return;
  }
  const proof = summary.proof_window || {};
  const candidate = summary.recommended_candidate || {};
  const evidence = summary.current_scope_evidence || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Proof active:</strong> ' + escapeHtml(String(!!proof.active)) + '</p>' +
    '<p><strong>Activated / expires / remaining minutes:</strong> ' + escapeHtml(proof.activated_at_utc || '-') + ' / ' + escapeHtml(proof.expires_at_utc || '-') + ' / ' + fmtNum(proof.remaining_minutes, 0) + '</p>' +
    '<p><strong>Candidate model / Stage 1:</strong> ' + escapeHtml((candidate.model_source || candidate.model_kind || '-') + ' / ' + (candidate.stage1_selection_mode || '-') + ' @ ' + (candidate.stage1_max_candidates || '-')) + '</p>' +
    '<p><strong>Current-scope resolved rows / visible rows:</strong> ' + fmtNum(evidence.resolved_rows, 0) + ' / ' + fmtNum(evidence.visible_rows, 0) + '</p>' +
    '<p><strong>Current-scope visible vs hidden quality:</strong> ' + fmtPct(evidence.visible_quality_hit_rate) + ' / ' + fmtPct(evidence.non_visible_quality_hit_rate) + '</p>';
}

async function activateLiveCandidateProof() {
  const msg = document.getElementById('liveCandidateProofMessage');
  if (msg) msg.textContent = 'Activating controlled live proof window…';
  try {
    const pw = liveCandidateProofPasswordOrThrow();
    const fd = new FormData();
    fd.append('proof_hours', String(Number(document.getElementById('liveCandidateProofHours').value || 24)));
    const res = await fetch('/api/reviews/live-candidate-proof/activate', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderLiveCandidateProofSummary(out);
    if (msg) msg.textContent = 'Controlled live proof window activated';
  } catch (err) {
    if (msg) msg.textContent = 'Could not activate controlled live proof window: ' + err.message;
  }
}

async function clearLiveCandidateProof() {
  const msg = document.getElementById('liveCandidateProofMessage');
  if (msg) msg.textContent = 'Clearing controlled live proof window…';
  try {
    const pw = liveCandidateProofPasswordOrThrow();
    const res = await fetch('/api/reviews/live-candidate-proof/clear', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderLiveCandidateProofSummary(out);
    if (msg) msg.textContent = 'Controlled live proof window cleared';
  } catch (err) {
    if (msg) msg.textContent = 'Could not clear controlled live proof window: ' + err.message;
  }
}

async function loadLiveCandidateProofSummary(silent = false) {
  const msg = document.getElementById('liveCandidateProofMessage');
  if (!silent && msg) msg.textContent = 'Loading controlled live proof summary…';
  try {
    const res = await fetch('/api/reviews/live-candidate-proof/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderLiveCandidateProofSummary(out);
    if (!silent && msg) msg.textContent = 'Controlled live proof summary loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load controlled live proof summary: ' + err.message;
  }
}

function downloadLiveCandidateProofPack() {
  const msg = document.getElementById('liveCandidateProofMessage');
  if (msg) msg.textContent = 'Opening controlled live proof pack…';
  window.open('/api/reviews/live-candidate-proof/latest-pack.zip', '_blank');
}


function renderLiveCandidateProofReviewSummary(summary) {
  const panel = document.getElementById('liveCandidateProofReviewPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No controlled live proof review loaded yet.</p>';
    return;
  }
  const session = summary.proof_session || {};
  const runs = summary.proof_runs || {};
  const evidence = summary.proof_evidence || {};
  const utility = summary.scan_shortlist_utility || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Session id / candidate:</strong> ' + escapeHtml((session.proof_session_id || '-') + ' / ' + (summary.candidate_label || '-')) + '</p>' +
    '<p><strong>Matching runs / evaluated runs:</strong> ' + fmtNum(runs.matching_runs, 0) + ' / ' + fmtNum(runs.evaluated_runs, 0) + '</p>' +
    '<p><strong>Resolved rows / visible rows / hidden rows:</strong> ' + fmtNum(evidence.resolved_rows, 0) + ' / ' + fmtNum(evidence.visible_rows, 0) + ' / ' + fmtNum(evidence.hidden_rows, 0) + '</p>' +
    '<p><strong>Visible vs hidden quality:</strong> ' + fmtPct(evidence.visible_quality_hit_rate) + ' / ' + fmtPct(evidence.hidden_quality_hit_rate) + ' (gap ' + fmtPct(evidence.visible_hidden_gap) + ')</p>' +
    '<p><strong>Scan utility / pairwise win / top-1:</strong> ' + fmtNum(utility.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utility.scan_shortlist_pairwise_win_rate) + ' / ' + fmtPct(utility.scan_shortlist_top1_visible_quality) + '</p>';
}

async function loadLiveCandidateProofReviewSummary(silent = false) {
  const msg = document.getElementById('liveCandidateProofReviewMessage');
  if (!silent && msg) msg.textContent = 'Loading controlled live proof review…';
  try {
    const res = await fetch('/api/reviews/live-candidate-proof-review/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderLiveCandidateProofReviewSummary(out);
    if (!silent && msg) msg.textContent = 'Controlled live proof review loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load controlled live proof review: ' + err.message;
  }
}

function downloadLiveCandidateProofReviewPack() {
  const msg = document.getElementById('liveCandidateProofReviewMessage');
  if (msg) msg.textContent = 'Opening controlled live proof review pack…';
  window.open('/api/reviews/live-candidate-proof-review/latest-pack.zip', '_blank');
}


function liveCandidateAdoptionPasswordOrThrow() {
  const localPw = (document.getElementById('liveCandidateAdoptionAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderLiveCandidateAdoptionSummary(summary) {
  const panel = document.getElementById('liveCandidateAdoptionPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No controlled live candidate adoption summary loaded yet.</p>';
    return;
  }
  const candidate = summary.candidate || {};
  const proof = summary.proof_review || {};
  const evidence = proof.proof_evidence || {};
  const utility = proof.scan_shortlist_utility || {};
  const active = summary.active_adoption || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Candidate model / Stage 1 / threshold:</strong> ' + escapeHtml((candidate.model_source || candidate.model_kind || '-') + ' / ' + (candidate.stage1_selection_mode || '-') + ' @ ' + (candidate.stage1_max_candidates || '-') + ' / ' + (candidate.raw_threshold || '-')) + '</p>' +
    '<p><strong>Proof verdict / visible rows / gap:</strong> ' + escapeHtml((proof.verdict || '-')) + ' / ' + fmtNum(evidence.visible_rows, 0) + ' / ' + fmtPct(evidence.visible_hidden_gap) + '</p>' +
    '<p><strong>Proof utility / pairwise win:</strong> ' + fmtNum(utility.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utility.scan_shortlist_pairwise_win_rate) + '</p>' +
    '<p><strong>Adoption active / adopted at:</strong> ' + escapeHtml(String(!!active.active)) + ' / ' + escapeHtml(active.adopted_at_utc || '-') + '</p>';
}

async function activateLiveCandidateAdoption() {
  const msg = document.getElementById('liveCandidateAdoptionMessage');
  if (msg) msg.textContent = 'Activating controlled live candidate adoption…';
  try {
    const pw = liveCandidateAdoptionPasswordOrThrow();
    const res = await fetch('/api/reviews/live-candidate-adoption/activate', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderLiveCandidateAdoptionSummary(out);
    if (msg) msg.textContent = 'Controlled live candidate adoption activated';
  } catch (err) {
    if (msg) msg.textContent = 'Could not activate controlled live candidate adoption: ' + err.message;
  }
}

async function clearLiveCandidateAdoption() {
  const msg = document.getElementById('liveCandidateAdoptionMessage');
  if (msg) msg.textContent = 'Clearing controlled live candidate adoption…';
  try {
    const pw = liveCandidateAdoptionPasswordOrThrow();
    const res = await fetch('/api/reviews/live-candidate-adoption/clear', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderLiveCandidateAdoptionSummary(out);
    if (msg) msg.textContent = 'Controlled live candidate adoption cleared';
  } catch (err) {
    if (msg) msg.textContent = 'Could not clear controlled live candidate adoption: ' + err.message;
  }
}

async function loadLiveCandidateAdoptionSummary(silent = false) {
  const msg = document.getElementById('liveCandidateAdoptionMessage');
  if (!silent && msg) msg.textContent = 'Loading controlled live candidate adoption…';
  try {
    const res = await fetch('/api/reviews/live-candidate-adoption/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderLiveCandidateAdoptionSummary(out);
    if (!silent && msg) msg.textContent = 'Controlled live candidate adoption loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load controlled live candidate adoption: ' + err.message;
  }
}

function downloadLiveCandidateAdoptionPack() {
  const msg = document.getElementById('liveCandidateAdoptionMessage');
  if (msg) msg.textContent = 'Opening controlled live candidate adoption pack…';
  window.open('/api/reviews/live-candidate-adoption/latest-pack.zip', '_blank');
}

function renderLiveCandidateAdoptionReviewSummary(summary) {
  const panel = document.getElementById('liveCandidateAdoptionReviewPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No controlled live candidate adoption review loaded yet.</p>';
    return;
  }
  const session = summary.adoption_session || {};
  const runs = summary.adoption_runs || {};
  const evidence = summary.adoption_evidence || {};
  const utility = summary.scan_shortlist_utility || {};
  const deltas = summary.deltas_vs_activation_baseline || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Adoption session / candidate:</strong> ' + escapeHtml((session.adoption_session_id || '-') + ' / ' + (summary.candidate_label || '-')) + '</p>' +
    '<p><strong>Matching runs / evaluated runs:</strong> ' + fmtNum(runs.matching_runs, 0) + ' / ' + fmtNum(runs.evaluated_runs, 0) + '</p>' +
    '<p><strong>Resolved rows / visible rows / hidden rows:</strong> ' + fmtNum(evidence.resolved_rows, 0) + ' / ' + fmtNum(evidence.visible_rows, 0) + ' / ' + fmtNum(evidence.hidden_rows, 0) + '</p>' +
    '<p><strong>Visible vs hidden quality:</strong> ' + fmtPct(evidence.visible_quality_hit_rate) + ' / ' + fmtPct(evidence.hidden_quality_hit_rate) + ' (gap ' + fmtPct(evidence.visible_hidden_gap) + ')</p>' +
    '<p><strong>Scan utility / pairwise win / top-1:</strong> ' + fmtNum(utility.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utility.scan_shortlist_pairwise_win_rate) + ' / ' + fmtPct(utility.scan_shortlist_top1_visible_quality) + '</p>' +
    '<p><strong>Gap delta vs activation baseline:</strong> ' + fmtPct(deltas.visible_hidden_gap_delta_vs_activation) + '</p>';
}

async function loadLiveCandidateAdoptionReviewSummary(silent = false) {
  const msg = document.getElementById('liveCandidateAdoptionReviewMessage');
  if (!silent && msg) msg.textContent = 'Loading controlled live candidate adoption review…';
  try {
    const res = await fetch('/api/utility-tuning-adoption-review/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderLiveCandidateAdoptionReviewSummary(out);
    if (!silent && msg) msg.textContent = 'Controlled live candidate adoption review loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load controlled live candidate adoption review: ' + err.message;
  }
}

function downloadLiveCandidateAdoptionReviewPack() {
  const msg = document.getElementById('liveCandidateAdoptionReviewMessage');
  if (msg) msg.textContent = 'Opening controlled live candidate adoption review pack…';
  window.open('/api/utility-tuning-adoption-review/latest-pack.zip', '_blank');
}

function utilityTuningProofPasswordOrThrow() {
  const localPw = (document.getElementById('utilityTuningProofAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderUtilityTuningProofSummary(summary) {
  const panel = document.getElementById('utilityTuningProofPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility tuning proof summary loaded yet.</p>';
    return;
  }
  const proof = summary.proof_window || {};
  const params = summary.recommended_params || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Proof active:</strong> ' + escapeHtml(String(!!proof.active)) + '</p>' +
    '<p><strong>Activated / expires / remaining minutes:</strong> ' + escapeHtml(proof.activated_at_utc || '-') + ' / ' + escapeHtml(proof.expires_at_utc || '-') + ' / ' + fmtNum(proof.remaining_minutes, 0) + '</p>' +
    '<p><strong>Target max names / score floor / confidence floor:</strong> ' + fmtNum(params.utility_shortlist_target_max_names, 0) + ' / ' + fmtNum(params.utility_shortlist_score_floor, 3) + ' / ' + fmtNum(params.utility_confidence_floor, 3) + '</p>' +
    '<p><strong>Edge / confidence / probability weights:</strong> ' + fmtNum(params.utility_expected_edge_weight, 2) + ' / ' + fmtNum(params.utility_confidence_weight, 2) + ' / ' + fmtNum(params.utility_probability_weight, 2) + '</p>';
}

async function activateUtilityTuningProof() {
  const msg = document.getElementById('utilityTuningProofMessage');
  if (msg) msg.textContent = 'Activating utility tuning proof window…';
  try {
    const pw = utilityTuningProofPasswordOrThrow();
    const fd = new FormData();
    fd.append('proof_hours', String(Number(document.getElementById('utilityTuningProofHours').value || 24)));
    const res = await fetch('/api/utility-tuning-proof/activate', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningProofSummary(out);
    if (msg) msg.textContent = 'Utility tuning proof window activated';
  } catch (err) {
    if (msg) msg.textContent = 'Could not activate utility tuning proof window: ' + err.message;
  }
}

async function clearUtilityTuningProof() {
  const msg = document.getElementById('utilityTuningProofMessage');
  if (msg) msg.textContent = 'Clearing utility tuning proof window…';
  try {
    const pw = utilityTuningProofPasswordOrThrow();
    const res = await fetch('/api/utility-tuning-proof/clear', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningProofSummary(out);
    if (msg) msg.textContent = 'Utility tuning proof window cleared';
  } catch (err) {
    if (msg) msg.textContent = 'Could not clear utility tuning proof window: ' + err.message;
  }
}

async function loadUtilityTuningProofSummary(silent = false) {
  const msg = document.getElementById('utilityTuningProofMessage');
  if (!silent && msg) msg.textContent = 'Loading utility tuning proof summary…';
  try {
    const res = await fetch('/api/utility-tuning-proof/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningProofSummary(out);
    if (!silent && msg) msg.textContent = 'Utility tuning proof summary loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility tuning proof summary: ' + err.message;
  }
}

function downloadUtilityTuningProofPack() {
  const msg = document.getElementById('utilityTuningProofMessage');
  if (msg) msg.textContent = 'Opening utility tuning proof pack…';
  window.open('/api/utility-tuning-proof/latest-pack.zip', '_blank');
}

function renderUtilityTuningProofReviewSummary(summary) {
  const panel = document.getElementById('utilityTuningProofReviewPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility tuning proof review loaded yet.</p>';
    return;
  }
  const proof = summary.proof_evidence || {};
  const utility = summary.scan_shortlist_utility || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Resolved rows / visible rows / hidden rows:</strong> ' + fmtNum(proof.resolved_rows, 0) + ' / ' + fmtNum(proof.visible_rows, 0) + ' / ' + fmtNum(proof.hidden_rows, 0) + '</p>' +
    '<p><strong>Visible vs hidden quality:</strong> ' + fmtPct(proof.visible_quality_hit_rate) + ' / ' + fmtPct(proof.hidden_quality_hit_rate) + ' (gap ' + fmtPct(proof.visible_hidden_gap) + ')</p>' +
    '<p><strong>Utility score / pairwise win rate:</strong> ' + fmtNum(utility.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utility.scan_shortlist_pairwise_win_rate) + '</p>';
}

async function loadUtilityTuningProofReviewSummary(silent = false) {
  const msg = document.getElementById('utilityTuningProofReviewMessage');
  if (!silent && msg) msg.textContent = 'Loading utility tuning proof review…';
  try {
    const res = await fetch('/api/utility-tuning-proof-review/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningProofReviewSummary(out);
    if (!silent && msg) msg.textContent = 'Utility tuning proof review loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility tuning proof review: ' + err.message;
  }
}

function downloadUtilityTuningProofReviewPack() {
  const msg = document.getElementById('utilityTuningProofReviewMessage');
  if (msg) msg.textContent = 'Opening utility tuning proof review pack…';
  window.open('/api/utility-tuning-proof-review/latest-pack.zip', '_blank');
}

function utilityTuningAdoptionPasswordOrThrow() {
  const localPw = (document.getElementById('utilityTuningAdoptionAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderUtilityTuningAdoptionSummary(summary) {
  const panel = document.getElementById('utilityTuningAdoptionPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility tuning adoption summary loaded yet.</p>';
    return;
  }
  const params = summary.candidate_params || {};
  const proof = summary.proof_review || {};
  const evidence = proof.proof_evidence || {};
  const utility = proof.scan_shortlist_utility || {};
  const active = summary.active_adoption || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Target max names / score floor / confidence floor:</strong> ' + fmtNum(params.utility_shortlist_target_max_names, 0) + ' / ' + fmtNum(params.utility_shortlist_score_floor, 3) + ' / ' + fmtNum(params.utility_confidence_floor, 3) + '</p>' +
    '<p><strong>Edge / confidence / probability weights:</strong> ' + fmtNum(params.utility_expected_edge_weight, 2) + ' / ' + fmtNum(params.utility_confidence_weight, 2) + ' / ' + fmtNum(params.utility_probability_weight, 2) + '</p>' +
    '<p><strong>Proof verdict / visible rows / gap:</strong> ' + escapeHtml((proof.verdict || '-')) + ' / ' + fmtNum(evidence.visible_rows, 0) + ' / ' + fmtPct(evidence.visible_hidden_gap) + '</p>' +
    '<p><strong>Proof utility / pairwise win:</strong> ' + fmtNum(utility.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utility.scan_shortlist_pairwise_win_rate) + '</p>' +
    '<p><strong>Adoption active / adopted at:</strong> ' + escapeHtml(String(!!active.active)) + ' / ' + escapeHtml(active.adopted_at_utc || '-') + '</p>';
}

async function activateUtilityTuningAdoption() {
  const msg = document.getElementById('utilityTuningAdoptionMessage');
  if (msg) msg.textContent = 'Activating utility tuning adoption…';
  try {
    const pw = utilityTuningAdoptionPasswordOrThrow();
    const res = await fetch('/api/utility-tuning-adoption/activate', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningAdoptionSummary(out);
    if (msg) msg.textContent = 'Utility tuning adoption activated';
  } catch (err) {
    if (msg) msg.textContent = 'Could not activate utility tuning adoption: ' + err.message;
  }
}

async function clearUtilityTuningAdoption() {
  const msg = document.getElementById('utilityTuningAdoptionMessage');
  if (msg) msg.textContent = 'Clearing utility tuning adoption…';
  try {
    const pw = utilityTuningAdoptionPasswordOrThrow();
    const res = await fetch('/api/utility-tuning-adoption/clear', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningAdoptionSummary(out);
    if (msg) msg.textContent = 'Utility tuning adoption cleared';
  } catch (err) {
    if (msg) msg.textContent = 'Could not clear utility tuning adoption: ' + err.message;
  }
}

async function loadUtilityTuningAdoptionSummary(silent = false) {
  const msg = document.getElementById('utilityTuningAdoptionMessage');
  if (!silent && msg) msg.textContent = 'Loading utility tuning adoption…';
  try {
    const res = await fetch('/api/utility-tuning-adoption/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningAdoptionSummary(out);
    if (!silent && msg) msg.textContent = 'Utility tuning adoption loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility tuning adoption: ' + err.message;
  }
}

function downloadUtilityTuningAdoptionPack() {
  const msg = document.getElementById('utilityTuningAdoptionMessage');
  if (msg) msg.textContent = 'Opening utility tuning adoption pack…';
  window.open('/api/utility-tuning-adoption/latest-pack.zip', '_blank');
}

function renderUtilityTuningAdoptionReviewSummary(summary) {
  const panel = document.getElementById('utilityTuningAdoptionReviewPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility tuning adoption review loaded yet.</p>';
    return;
  }
  const session = summary.adoption_session || {};
  const runs = summary.adoption_runs || {};
  const evidence = summary.adoption_evidence || {};
  const utility = summary.scan_shortlist_utility || {};
  const deltas = summary.deltas_vs_activation_baseline || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Adoption session / candidate:</strong> ' + escapeHtml((session.adoption_session_id || '-') + ' / ' + (summary.candidate_label || '-')) + '</p>' +
    '<p><strong>Matching runs / evaluated runs:</strong> ' + fmtNum(runs.matching_runs, 0) + ' / ' + fmtNum(runs.evaluated_runs, 0) + '</p>' +
    '<p><strong>Resolved rows / visible rows / hidden rows:</strong> ' + fmtNum(evidence.resolved_rows, 0) + ' / ' + fmtNum(evidence.visible_rows, 0) + ' / ' + fmtNum(evidence.hidden_rows, 0) + '</p>' +
    '<p><strong>Visible vs hidden quality:</strong> ' + fmtPct(evidence.visible_quality_hit_rate) + ' / ' + fmtPct(evidence.hidden_quality_hit_rate) + ' (gap ' + fmtPct(evidence.visible_hidden_gap) + ')</p>' +
    '<p><strong>Scan utility / pairwise win / top-1:</strong> ' + fmtNum(utility.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utility.scan_shortlist_pairwise_win_rate) + ' / ' + fmtPct(utility.scan_shortlist_top1_visible_quality) + '</p>' +
    '<p><strong>Gap delta vs activation baseline:</strong> ' + fmtPct(deltas.visible_hidden_gap_delta_vs_activation) + '</p>';
}

async function loadUtilityTuningAdoptionReviewSummary(silent = false) {
  const msg = document.getElementById('utilityTuningAdoptionReviewMessage');
  if (!silent && msg) msg.textContent = 'Loading utility tuning adoption review…';
  try {
    const res = await fetch('/api/utility-tuning-adoption-review/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityTuningAdoptionReviewSummary(out);
    if (!silent && msg) msg.textContent = 'Utility tuning adoption review loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility tuning adoption review: ' + err.message;
  }
}

function downloadUtilityTuningAdoptionReviewPack() {
  const msg = document.getElementById('utilityTuningAdoptionReviewMessage');
  if (msg) msg.textContent = 'Opening utility tuning adoption review pack…';
  window.open('/api/utility-tuning-adoption-review/latest-pack.zip', '_blank');
}

function utilityModelProofPasswordOrThrow() {
  const localPw = (document.getElementById('utilityModelProofAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderUtilityModelProofSummary(summary) {
  const panel = document.getElementById('utilityModelProofPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility model proof summary loaded yet.</p>';
    return;
  }
  const proof = summary.proof_window || {};
  const candidate = summary.candidate_model || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Candidate path:</strong> ' + escapeHtml(candidate.path || '-') + '</p>' +
    '<p><strong>Proof active / activated / expires:</strong> ' + escapeHtml(String(!!proof.active)) + ' / ' + escapeHtml(proof.activated_at_utc || '-') + ' / ' + escapeHtml(proof.expires_at_utc || '-') + '</p>';
}

async function activateUtilityModelProof() {
  const msg = document.getElementById('utilityModelProofMessage');
  if (msg) msg.textContent = 'Activating utility model proof…';
  try {
    const pw = utilityModelProofPasswordOrThrow();
    const fd = new FormData();
    fd.append('proof_hours', String(Number(document.getElementById('utilityModelProofHours').value || 24)));
    const res = await fetch('/api/utility-model-proof/activate', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelProofSummary(out);
    if (msg) msg.textContent = 'Utility model proof activated';
  } catch (err) {
    if (msg) msg.textContent = 'Could not activate utility model proof: ' + err.message;
  }
}

async function clearUtilityModelProof() {
  const msg = document.getElementById('utilityModelProofMessage');
  if (msg) msg.textContent = 'Clearing utility model proof…';
  try {
    const pw = utilityModelProofPasswordOrThrow();
    const res = await fetch('/api/utility-model-proof/clear', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelProofSummary(out);
    if (msg) msg.textContent = 'Utility model proof cleared';
  } catch (err) {
    if (msg) msg.textContent = 'Could not clear utility model proof: ' + err.message;
  }
}

async function loadUtilityModelProofSummary(silent = false) {
  const msg = document.getElementById('utilityModelProofMessage');
  if (!silent && msg) msg.textContent = 'Loading utility model proof…';
  try {
    const res = await fetch('/api/utility-model-proof/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelProofSummary(out);
    if (!silent && msg) msg.textContent = 'Utility model proof loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility model proof: ' + err.message;
  }
}

function downloadUtilityModelProofPack() {
  const msg = document.getElementById('utilityModelProofMessage');
  if (msg) msg.textContent = 'Opening utility model proof pack…';
  window.open('/api/utility-model-proof/latest-pack.zip', '_blank');
}

function setUtilityModelProofReviewDownloadsEnabled(enabled) {
  const btn = document.getElementById('downloadUtilityModelProofReviewPackButton');
  if (btn) btn.disabled = !enabled;
}

function renderUtilityModelProofReviewSummary(summary) {
  const panel = document.getElementById('utilityModelProofReviewPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    setUtilityModelProofReviewDownloadsEnabled(false);
    panel.innerHTML = '<p class="muted">No utility model proof review loaded yet.</p>';
    return;
  }
  const runs = summary.proof_runs || {};
  const evidence = summary.proof_evidence || {};
  const utility = summary.scan_shortlist_utility || {};
  const status = summary.status || 'unknown';
  const packReady = !!summary.pack_ready;
  setUtilityModelProofReviewDownloadsEnabled(packReady);
  panel.innerHTML =
    '<p><strong>Status:</strong> ' + escapeHtml(status) + (packReady ? ' · pack ready' : ' · pack not ready') + '</p>' +
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Updated:</strong> ' + escapeHtml(summary.generated_at_utc || '-') + '</p>' +
    '<p><strong>Matching runs / evaluated runs:</strong> ' + fmtNum(runs.matching_runs, 0) + ' / ' + fmtNum(runs.evaluated_runs, 0) + '</p>' +
    '<p><strong>Resolved rows / visible rows / hidden rows:</strong> ' + fmtNum(evidence.resolved_rows, 0) + ' / ' + fmtNum(evidence.visible_rows, 0) + ' / ' + fmtNum(evidence.hidden_rows, 0) + '</p>' +
    '<p><strong>Visible vs hidden quality:</strong> ' + fmtPct(evidence.visible_quality_hit_rate) + ' / ' + fmtPct(evidence.hidden_quality_hit_rate) + ' (gap ' + fmtPct(evidence.visible_hidden_gap) + ')</p>' +
    '<p><strong>Scan utility / pairwise win / top-1:</strong> ' + fmtNum(utility.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utility.scan_shortlist_pairwise_win_rate) + ' / ' + fmtPct(utility.scan_shortlist_top1_visible_quality) + '</p>';
}

async function loadUtilityModelProofReviewSummary(silent = false) {
  const msg = document.getElementById('utilityModelProofReviewMessage');
  if (!silent && msg) msg.textContent = 'Loading utility model proof review…';
  try {
    const res = await fetch('/api/utility-model-proof-review/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelProofReviewSummary(out);
    if (!silent && msg) msg.textContent = 'Utility model proof review loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility model proof review: ' + err.message;
  }
}

function downloadUtilityModelProofReviewPack() {
  const msg = document.getElementById('utilityModelProofReviewMessage');
  const btn = document.getElementById('downloadUtilityModelProofReviewPackButton');
  if (btn && btn.disabled) {
    if (msg) msg.textContent = 'No proof review pack is ready yet.';
    return;
  }
  if (msg) msg.textContent = 'Opening utility model proof review pack…';
  window.open('/api/utility-model-proof-review/latest-pack.zip', '_blank');
}

function utilityModelAdoptionPasswordOrThrow() {
  const localPw = (document.getElementById('utilityModelAdoptionAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderUtilityModelAdoptionSummary(summary) {
  const panel = document.getElementById('utilityModelAdoptionPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility model adoption summary loaded yet.</p>';
    return;
  }
  const cand = summary.candidate_model || {};
  const proof = summary.proof_review || {};
  const evidence = proof.proof_evidence || {};
  const utility = proof.scan_shortlist_utility || {};
  const active = summary.active_adoption || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Recommended action:</strong> ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Candidate model:</strong> ' + escapeHtml((cand.label || '-') + ' / ' + (cand.path || '-')) + '</p>' +
    '<p><strong>Proof verdict / visible rows / gap:</strong> ' + escapeHtml((proof.verdict || '-')) + ' / ' + fmtNum(evidence.visible_rows, 0) + ' / ' + fmtPct(evidence.visible_hidden_gap) + '</p>' +
    '<p><strong>Proof utility / pairwise win:</strong> ' + fmtNum(utility.scan_shortlist_utility_score, 4) + ' / ' + fmtPct(utility.scan_shortlist_pairwise_win_rate) + '</p>' +
    '<p><strong>Adoption active / adopted at:</strong> ' + escapeHtml(String(!!active.active)) + ' / ' + escapeHtml(active.adopted_at_utc || '-') + '</p>';
}

async function activateUtilityModelAdoption() {
  const msg = document.getElementById('utilityModelAdoptionMessage');
  if (msg) msg.textContent = 'Activating utility model adoption…';
  try {
    const pw = utilityModelAdoptionPasswordOrThrow();
    const res = await fetch('/api/utility-model-adoption/activate', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelAdoptionSummary(out);
    if (msg) msg.textContent = 'Utility model adoption activated';
  } catch (err) {
    if (msg) msg.textContent = 'Could not activate utility model adoption: ' + err.message;
  }
}

async function clearUtilityModelAdoption() {
  const msg = document.getElementById('utilityModelAdoptionMessage');
  if (msg) msg.textContent = 'Clearing utility model adoption…';
  try {
    const pw = utilityModelAdoptionPasswordOrThrow();
    const res = await fetch('/api/utility-model-adoption/clear', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelAdoptionSummary(out);
    if (msg) msg.textContent = 'Utility model adoption cleared';
  } catch (err) {
    if (msg) msg.textContent = 'Could not clear utility model adoption: ' + err.message;
  }
}

async function loadUtilityModelAdoptionSummary(silent = false) {
  const msg = document.getElementById('utilityModelAdoptionMessage');
  if (!silent && msg) msg.textContent = 'Loading utility model adoption…';
  try {
    const res = await fetch('/api/utility-model-adoption/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityModelAdoptionSummary(out);
    if (!silent && msg) msg.textContent = 'Utility model adoption loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility model adoption: ' + err.message;
  }
}

function downloadUtilityModelAdoptionPack() {
  const msg = document.getElementById('utilityModelAdoptionMessage');
  if (msg) msg.textContent = 'Opening utility model adoption pack…';
  window.open('/api/utility-model-adoption/latest-pack.zip', '_blank');
}

function utilityOperatorAutomationPasswordOrThrow() {
  const localPw = (document.getElementById('utilityOperatorAutomationAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

function renderUtilityOperatorAutomationStatus(summary) {
  const panel = document.getElementById('utilityOperatorAutomationPanel');
  if (!panel) return;
  if (!summary || Object.keys(summary).length === 0) {
    panel.innerHTML = '<p class="muted">No utility operator automation status loaded yet.</p>';
    return;
  }
  const state = summary.state || {};
  const modelLab = summary.utility_model_lab || {};
  const tuningLab = summary.utility_tuning_lab || {};
  const modelProof = summary.utility_model_proof_review || {};
  const tuningProof = summary.utility_tuning_proof_review || {};
  const modelAdopt = summary.utility_model_adoption_review || {};
  const tuningAdopt = summary.utility_tuning_adoption_review || {};
  const freshness = summary.offline_lab_freshness || {};
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Summary:</strong> ' + escapeHtml(summary.summary || '-') + '</p>' +
    '<p><strong>App version:</strong> ' + escapeHtml(summary.app_version || '-') + '</p>' +
    '<p><strong>Session / active / phase:</strong> ' + escapeHtml((state.session_id || '-')) + ' / ' + escapeHtml(String(!!state.active)) + ' / ' + escapeHtml(state.phase || '-') + '</p>' +
    '<p><strong>Selected branch / attempted:</strong> ' + escapeHtml((state.selected_branch || '-')) + ' / ' + escapeHtml((state.attempted_branches || []).join(', ') || '-') + '</p>' +
    '<p><strong>Selection gate verdict:</strong> ' + escapeHtml(state.selection_gate_verdict || '-') + '</p>' +
    '<p><strong>Freshness (selection / tuning / model):</strong> ' + escapeHtml(String(!!freshness.utility_selection_lab)) + ' / ' + escapeHtml(String(!!freshness.utility_tuning_lab)) + ' / ' + escapeHtml(String(!!freshness.utility_model_lab)) + '</p>' +
    '<p><strong>Last action / last error:</strong> ' + escapeHtml(state.last_action || '-') + ' / ' + escapeHtml(state.last_error || '-') + '</p>' +
    '<p><strong>Offline verdicts (model / tuning):</strong> ' + escapeHtml(modelLab.verdict || '-') + ' / ' + escapeHtml(tuningLab.verdict || '-') + '</p>' +
    '<p><strong>Proof review verdicts (model / tuning):</strong> ' + escapeHtml(modelProof.verdict || '-') + ' / ' + escapeHtml(tuningProof.verdict || '-') + '</p>' +
    '<p><strong>Adoption review verdicts (model / tuning):</strong> ' + escapeHtml(modelAdopt.verdict || '-') + ' / ' + escapeHtml(tuningAdopt.verdict || '-') + '</p>';
}

async function startUtilityOperatorAutomation() {
  const msg = document.getElementById('utilityOperatorAutomationMessage');
  if (msg) msg.textContent = 'Starting utility operator automation…';
  try {
    const pw = utilityOperatorAutomationPasswordOrThrow();
    const res = await fetch('/api/utility-operator-automation/start', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityOperatorAutomationStatus(out);
    if (msg) msg.textContent = 'Utility operator automation started';
  } catch (err) {
    if (msg) msg.textContent = 'Could not start utility operator automation: ' + err.message;
  }
}

async function stopUtilityOperatorAutomation() {
  const msg = document.getElementById('utilityOperatorAutomationMessage');
  if (msg) msg.textContent = 'Stopping utility operator automation…';
  try {
    const pw = utilityOperatorAutomationPasswordOrThrow();
    const res = await fetch('/api/utility-operator-automation/stop', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityOperatorAutomationStatus(out);
    if (msg) msg.textContent = 'Utility operator automation stopped';
  } catch (err) {
    if (msg) msg.textContent = 'Could not stop utility operator automation: ' + err.message;
  }
}

async function loadUtilityOperatorAutomationStatus(silent = false) {
  const msg = document.getElementById('utilityOperatorAutomationMessage');
  if (!silent && msg) msg.textContent = 'Loading utility operator automation status…';
  try {
    const res = await fetch('/api/utility-operator-automation/status');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityOperatorAutomationStatus(out);
    if (!silent && msg) msg.textContent = 'Utility operator automation status loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load utility operator automation status: ' + err.message;
  }
}

function downloadUtilityOperatorAutomationPack() {
  const msg = document.getElementById('utilityOperatorAutomationMessage');
  if (msg) msg.textContent = 'Opening utility operator automation pack…';
  window.open('/api/utility-operator-automation/latest-pack.zip', '_blank');
}

function utilityPolicySearchPasswordOrThrow() {
  const localPw = (document.getElementById('utilityPolicySearchAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

let utilityPolicySearchPollTimer = null;

function stopUtilityPolicySearchPolling() {
  if (utilityPolicySearchPollTimer) {
    clearTimeout(utilityPolicySearchPollTimer);
    utilityPolicySearchPollTimer = null;
  }
}

function scheduleUtilityPolicySearchPolling(delayMs) {
  stopUtilityPolicySearchPolling();
  utilityPolicySearchPollTimer = setTimeout(function() {
    loadUtilityPolicySearchStatus(true);
  }, delayMs || 3000);
}

function setUtilityPolicySearchDownloadsEnabled(enabled) {
  const packBtn = document.getElementById('downloadUtilityPolicySearchPackButton');
  const summaryTxtBtn = document.getElementById('downloadUtilityPolicySearchSummaryTxtButton');
  const summaryTxtLink = document.getElementById('downloadUtilityPolicySearchSummaryTxtLink');
  if (packBtn) packBtn.disabled = !enabled;
  if (summaryTxtBtn) summaryTxtBtn.disabled = !enabled;
  if (summaryTxtLink) summaryTxtLink.setAttribute('aria-disabled', enabled ? 'false' : 'true');
}

function renderUtilityPolicySearchStatus(status) {
  const panel = document.getElementById('utilityPolicySearchStatusPanel');
  if (!panel) return;
  if (!status || !status.available) {
    panel.innerHTML = '<p class="muted">No offline family search status loaded yet.</p>';
    setUtilityPolicySearchDownloadsEnabled(false);
    return;
  }
  const active = !!status.active;
  const progressPct = Number(status.progress_pct || 0);
  const currentPolicy = status.current_policy || {};
  const badge = active ? 'Running' : (status.status === 'error' ? 'Error' : (status.status === 'completed' ? 'Ready' : 'Idle'));
  const badgeColor = active ? '#1d4ed8' : (status.status === 'error' ? '#b91c1c' : (status.status === 'completed' ? '#166534' : '#4b5563'));
  panel.innerHTML =
    '<div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">' +
      '<span style="display:inline-block; padding:4px 10px; border-radius:999px; background:' + badgeColor + '; color:#fff; font-size:12px; font-weight:600;">' + escapeHtml(badge) + '</span>' +
      '<span><strong>Phase:</strong> ' + escapeHtml(status.phase || '-') + '</span>' +
      '<span><strong>Progress:</strong> ' + escapeHtml(String(progressPct)) + '%</span>' +
    '</div>' +
    '<div style="height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin-bottom:8px;">' +
      '<div style="height:10px; width:' + escapeHtml(String(Math.max(0, Math.min(100, progressPct)))) + '%; background:#2563eb;"></div>' +
    '</div>' +
    '<p><strong>Headline:</strong> ' + escapeHtml(status.headline || '-') + '</p>' +
    '<p><strong>Summary:</strong> ' + escapeHtml(status.summary || '-') + '</p>' +
    '<p><strong>Started / updated:</strong> ' + escapeHtml(status.started_at_utc || '-') + ' / ' + escapeHtml(status.updated_at_utc || '-') + '</p>' +
    '<p><strong>Step:</strong> ' + escapeHtml(String(status.current_step || '-')) + ' of ' + escapeHtml(String(status.total_steps || '-')) + '</p>' +
    '<p><strong>Current policy:</strong> ' + escapeHtml(currentPolicy.policy_name || currentPolicy.policy_id || '-') + '</p>' +
    '<p><strong>Last error:</strong> ' + escapeHtml(status.last_error || '-') + '</p>';
  setUtilityPolicySearchDownloadsEnabled(!!status.result_ready);
}

function renderUtilityPolicySearchSummary(summary) {
  const panel = document.getElementById('utilityPolicySearchPanel');
  if (!panel) return;
  if (!summary || !summary.available) {
    panel.innerHTML = '<p class="muted">No offline family search summary available yet.</p>';
    return;
  }
  const winner = summary.winner || {};
  const ranked = summary.ranked_policies || [];
  const families = summary.family_results || [];
  const rows = ranked.slice(0, 5).map((item, idx) => '<tr>' +
    '<td>' + String(idx + 1) + '</td>' +
    '<td>' + escapeHtml(item.policy_name || item.engine_label || '-') + '</td>' +
    '<td>' + escapeHtml(item.family_name || item.family_id || '-') + '</td>' +
    '<td>' + escapeHtml(item.support_level || '-') + '</td>' +
    '<td>' + escapeHtml(formatNumber(item.scan_shortlist_utility_score)) + '</td>' +
    '<td>' + escapeHtml(formatNumber(item.scan_shortlist_pairwise_win_rate)) + '</td>' +
    '<td>' + escapeHtml(formatNumber(item.scan_shortlist_mean_gap)) + '</td>' +
    '<td>' + escapeHtml(formatNumber(item.scan_shortlist_avg_visible_rows_per_scan)) + '</td>' +
    '</tr>').join('');
  const familyRows = families.slice(0, 5).map((item, idx) => {
    const famWinner = item.family_winner || {};
    return '<tr>' +
      '<td>' + String(idx + 1) + '</td>' +
      '<td>' + escapeHtml(item.family_name || item.family_id || '-') + '</td>' +
      '<td>' + escapeHtml(item.family_support_level || '-') + '</td>' +
      '<td>' + escapeHtml(famWinner.policy_name || famWinner.engine_label || '-') + '</td>' +
      '<td>' + escapeHtml(formatNumber(famWinner.scan_shortlist_utility_score)) + '</td>' +
      '<td>' + escapeHtml(formatNumber(famWinner.scan_shortlist_pairwise_win_rate)) + '</td>' +
      '</tr>';
  }).join('');
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Summary:</strong> ' + escapeHtml(summary.summary || '-') + '</p>' +
    '<p><strong>Verdict / action:</strong> ' + escapeHtml(summary.verdict || '-') + ' / ' + escapeHtml(summary.recommended_action || '-') + '</p>' +
    '<p><strong>Winner:</strong> ' + escapeHtml(winner.policy_name || winner.engine_label || '-') + '</p>' +
    '<p><strong>Winner family / support / utility / pairwise:</strong> ' + escapeHtml(winner.family_name || winner.family_id || '-') + ' / ' + escapeHtml(winner.support_level || '-') + ' / ' + escapeHtml(formatNumber(winner.scan_shortlist_utility_score)) + ' / ' + escapeHtml(formatNumber(winner.scan_shortlist_pairwise_win_rate)) + '</p>' +
    '<p><strong>Top families:</strong></p>' +
    '<div class="table-wrap"><table><thead><tr><th>#</th><th>Family</th><th>Family support</th><th>Best policy</th><th>Utility</th><th>Pairwise</th></tr></thead><tbody>' + familyRows + '</tbody></table></div>' +
    '<p><strong>Top policies:</strong></p>' +
    '<div class="table-wrap"><table><thead><tr><th>#</th><th>Policy</th><th>Family</th><th>Support</th><th>Utility</th><th>Pairwise</th><th>Gap</th><th>Avg visible</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
}

async function loadUtilityPolicySearchStatus(silent) {
  const msg = document.getElementById('utilityPolicySearchMessage');
  if (!silent && msg) msg.textContent = 'Loading offline family search status…';
  try {
    const res = await fetch('/api/utility-policy-search/status');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityPolicySearchStatus(out);
    if (out && out.active) {
      if (msg) msg.textContent = 'Offline family search is running';
      scheduleUtilityPolicySearchPolling(3000);
    } else {
      stopUtilityPolicySearchPolling();
      if (!silent && msg) msg.textContent = out && out.status === 'completed' ? 'Offline family search status loaded — results ready' : 'Offline family search status loaded';
      if (out && out.result_ready) {
        loadUtilityPolicySearchSummary(true);
      }
    }
  } catch (err) {
    stopUtilityPolicySearchPolling();
    if (msg) msg.textContent = 'Could not load offline family search status: ' + err.message;
  }
}

async function runUtilityPolicySearch() {
  const msg = document.getElementById('utilityPolicySearchMessage');
  if (msg) msg.textContent = 'Starting offline family search…';
  try {
    const pw = utilityPolicySearchPasswordOrThrow();
    const form = new FormData();
    form.append('hours', String(Number((document.getElementById('utilityPolicySearchHours') || {}).value || 168)));
    form.append('step_minutes', String(Number((document.getElementById('utilityPolicySearchStepMinutes') || {}).value || 120)));
    form.append('max_scans', String(Number((document.getElementById('utilityPolicySearchMaxScans') || {}).value || 84)));
    form.append('max_symbols', String(Number((document.getElementById('utilityPolicySearchMaxSymbols') || {}).value || 100)));
    const res = await fetch('/api/utility-policy-search/run', { method: 'POST', headers: { 'X-Admin-Password': pw }, body: form });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityPolicySearchStatus(out);
    if (msg) msg.textContent = 'Offline family search started';
    scheduleUtilityPolicySearchPolling(1500);
  } catch (err) {
    stopUtilityPolicySearchPolling();
    if (msg) msg.textContent = 'Could not run offline family search: ' + err.message;
  }
}

async function loadUtilityPolicySearchSummary(silent) {
  const msg = document.getElementById('utilityPolicySearchMessage');
  if (!silent && msg) msg.textContent = 'Loading offline family search summary…';
  try {
    const res = await fetch('/api/utility-policy-search/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderUtilityPolicySearchSummary(out);
    if (!silent && msg) msg.textContent = 'Offline family search summary loaded';
  } catch (err) {
    if (msg) msg.textContent = 'Could not load offline family search summary: ' + err.message;
  }
}

function downloadUtilityPolicySearchPack() {
  const msg = document.getElementById('utilityPolicySearchMessage');
  if (msg) msg.textContent = 'Opening offline family search pack…';
  window.open('/api/utility-policy-search/latest-pack.zip', '_blank');
}


const shadowSelectionComparisonUiState = {
  summary: null,
};

function setShadowSelectionComparisonDownloadsEnabled(enabled, reason) {
  const packBtn = document.getElementById('downloadShadowSelectionComparisonPackButton');
  const hint = document.getElementById('shadowSelectionComparisonPackHint');
  const why = reason || 'Pack unavailable until a recorded shadow comparison exists.';
  if (packBtn) {
    packBtn.disabled = !enabled;
    packBtn.title = enabled ? 'Download the latest recorded shadow comparison pack' : why;
    packBtn.textContent = enabled ? 'Download Shadow Comparison Pack' : 'Shadow Comparison Pack Not Ready';
  }
  if (hint) {
    hint.textContent = enabled ? 'Pack ready: download available.' : why;
  }
}

function shadowComparisonBadge(status) {
  const s = String(status || '').toLowerCase();
  if (s === 'recorded' || s === 'completed' || s === 'ready') return { label: 'Ready', color: '#166534' };
  if (s === 'skipped') return { label: 'Skipped', color: '#92400e' };
  if (s === 'error' || s === 'failed') return { label: 'Error', color: '#b91c1c' };
  return { label: 'Waiting', color: '#1d4ed8' };
}

function renderShadowSelectionComparisonSummary(summary) {
  const panel = document.getElementById('shadowSelectionComparisonPanel');
  if (!panel) return;
  shadowSelectionComparisonUiState.summary = summary || null;
  if (!summary || !summary.available) {
    panel.innerHTML = '<div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">' +
      '<span style="display:inline-block; padding:4px 10px; border-radius:999px; background:#1d4ed8; color:#fff; font-size:12px; font-weight:600;">Waiting</span>' +
      '<span><strong>Status:</strong> waiting</span>' +
      '</div>' +
      '<p><strong>Headline:</strong> No shadow comparison summary available yet</p>' +
      '<p><strong>Summary:</strong> The first comparison will appear after a completed scan when legacy is live and challenger policies are available.</p>';
    setShadowSelectionComparisonDownloadsEnabled(false, 'Pack unavailable until the first recorded comparison exists.');
    return;
  }
  const incumbent = summary.incumbent || {};
  const primaryChallenger = summary.challenger || {};
  const comparison = summary.comparison || {};
  const trailing = summary.trailing_24h || {};
  const badge = shadowComparisonBadge(summary.status);
  const generatedAt = summary.generated_at_utc || summary.updated_at_utc || '-';
  const liveEngine = summary.effective_live_selection_engine || summary.effective_live_selection_mode || summary.configured_live_selection_mode || '-';
  const packReady = !!summary.pack_available;
  const packReason = packReady ? '' : (summary.skip_reason ? ('Pack unavailable: ' + summary.skip_reason.replace(/_/g, ' ')) : 'Pack unavailable until a recorded comparison exists.');
  const challengerRecords = Array.isArray(summary.challenger_records) ? summary.challenger_records : [];
  const activeCount = Number(summary.active_challenger_count || challengerRecords.length || 0);
  const challengerHtml = challengerRecords.length
    ? '<div style="margin-top:10px;"><strong>Active challengers on this scan:</strong><ul style="margin:6px 0 0 18px;">' +
        challengerRecords.map(function(item) {
          return '<li><strong>' + escapeHtml(item.policy_name || item.policy_id || '-') + '</strong>' +
            ' — count ' + escapeHtml(String(item.visible_count || 0)) +
            ', overlap ' + escapeHtml(String(item.overlap_count || 0)) +
            ', density delta ' + escapeHtml(formatNumber(item.density_delta)) +
            ', top symbols: ' + escapeHtml((item.top_symbols || []).join(', ') || '-') +
            '</li>';
        }).join('') +
      '</ul></div>'
    : '';
  const trailingCounts = Array.isArray(trailing.challenger_counts) && trailing.challenger_counts.length
    ? trailing.challenger_counts.map(function(item) { return escapeHtml(item.policy_name + ' (' + item.comparisons + ')'); }).join(', ')
    : '-';
  panel.innerHTML =
    '<div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">' +
      '<span style="display:inline-block; padding:4px 10px; border-radius:999px; background:' + badge.color + '; color:#fff; font-size:12px; font-weight:600;">' + escapeHtml(badge.label) + '</span>' +
      '<span><strong>Status:</strong> ' + escapeHtml(summary.status || '-') + '</span>' +
      '<span><strong>Updated:</strong> ' + escapeHtml(generatedAt) + '</span>' +
      '<span><strong>Pack ready:</strong> ' + escapeHtml(packReady ? 'yes' : 'no') + '</span>' +
    '</div>' +
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Summary:</strong> ' + escapeHtml(summary.summary || '-') + '</p>' +
    '<p><strong>Live engine:</strong> ' + escapeHtml(liveEngine) + ' <span class="muted">| active challengers:</span> ' + escapeHtml(String(activeCount)) + '</p>' +
    '<p><strong>Primary challenger snapshot:</strong> ' + escapeHtml(((summary.challenger_policy || {}).policy_name) || primaryChallenger.engine || '-') +
      ' <span class="muted">| counts (incumbent vs challenger):</span> ' + escapeHtml(String(incumbent.visible_count || 0)) + ' / ' + escapeHtml(String(primaryChallenger.visible_count || 0)) +
      ' <span class="muted">| overlap:</span> ' + escapeHtml(String(comparison.overlap_count || 0)) + '</p>' +
    '<p><strong>Trailing 24h comparisons:</strong> ' + escapeHtml(String(trailing.comparisons || 0)) +
      ' <span class="muted">| avg overlap vs incumbent:</span> ' + escapeHtml(formatNumber(trailing.avg_overlap_ratio_vs_incumbent)) +
      ' <span class="muted">| avg density delta:</span> ' + escapeHtml(formatNumber(trailing.avg_density_delta)) + '</p>' +
    '<p><strong>Trailing 24h challenger counts:</strong> ' + trailingCounts + '</p>' +
    challengerHtml +
    ((summary.skip_reason || '') ? '<p><strong>Skip reason:</strong> ' + escapeHtml(summary.skip_reason) + '</p>' : '');
  setShadowSelectionComparisonDownloadsEnabled(packReady, packReason);
}

async function loadShadowSelectionComparisonSummary(silent) {
  const msg = document.getElementById('shadowSelectionComparisonMessage');
  if (!silent && msg) msg.textContent = 'Loading shadow comparison summary…';
  try {
    const res = await fetch('/api/shadow-selection-comparison/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderShadowSelectionComparisonSummary(out);
    if (!silent && msg) msg.textContent = 'Shadow comparison summary loaded';
    if (silent && msg) msg.textContent = out && out.status ? ('Shadow comparison ' + out.status) : '';
  } catch (err) {
    setShadowSelectionComparisonDownloadsEnabled(false, 'Pack unavailable until the first recorded comparison exists.');
    if (msg) msg.textContent = 'Could not load shadow comparison summary: ' + err.message;
  }
}

async function downloadShadowSelectionComparisonPack() {
  const msg = document.getElementById('shadowSelectionComparisonMessage');
  let summary = shadowSelectionComparisonUiState.summary || {};
  if (!summary.available) {
    try {
      const res = await fetch('/api/shadow-selection-comparison/summary');
      if (!res.ok) throw new Error(await res.text());
      summary = await res.json();
      renderShadowSelectionComparisonSummary(summary);
    } catch (err) {
      if (msg) msg.textContent = 'Could not load shadow comparison summary before download: ' + err.message;
      return;
    }
  }
  if (!summary.pack_available) {
    if (msg) {
      const status = summary.status || 'waiting';
      const reason = summary.skip_reason ? (' (' + summary.skip_reason + ')') : '';
      msg.textContent = 'No shadow comparison pack is ready yet. Current state: ' + status + reason;
    }
    return;
  }
  if (msg) msg.textContent = 'Downloading shadow comparison pack…';
  window.location.href = '/api/shadow-selection-comparison/latest-pack.zip';
}


const shadowSelectionOutcomeReviewUiState = { summary: null };

function setShadowSelectionOutcomeReviewDownloadsEnabled(enabled, reason) {
  const packBtn = document.getElementById('downloadShadowSelectionOutcomeReviewPackButton');
  const hint = document.getElementById('shadowSelectionOutcomeReviewPackHint');
  const why = reason || 'Pack unavailable until at least one recorded shadow comparison has matured.';
  if (packBtn) {
    packBtn.disabled = !enabled;
    packBtn.title = enabled ? 'Download the latest matured shadow outcome review pack' : why;
    packBtn.textContent = enabled ? 'Download Shadow Outcome Review Pack' : 'Shadow Outcome Review Pack Not Ready';
  }
  if (hint) hint.textContent = enabled ? 'Pack ready: download available.' : why;
}

function shadowSelectionOutcomeBadge(status) {
  const s = String(status || '').toLowerCase();
  if (s === 'reviewed' || s === 'ready' || s === 'completed') return { label: 'Ready', color: '#166534' };
  if (s === 'error' || s === 'failed') return { label: 'Error', color: '#b91c1c' };
  return { label: 'Waiting', color: '#1d4ed8' };
}

function renderShadowSelectionOutcomeReview(summary) {
  const panel = document.getElementById('shadowSelectionOutcomeReviewPanel');
  if (!panel) return;
  shadowSelectionOutcomeReviewUiState.summary = summary || null;
  if (!summary || !summary.available) {
    panel.innerHTML = '<p><strong>Headline:</strong> No shadow outcome review available yet</p><p><strong>Summary:</strong> Waiting for matured shadow comparisons.</p>';
    setShadowSelectionOutcomeReviewDownloadsEnabled(false, 'Pack unavailable until at least one recorded shadow comparison has matured.');
    return;
  }
  const badge = shadowSelectionOutcomeBadge(summary.status);
  const incumbent = summary.incumbent || {};
  const challenger = summary.challenger || {};
  const results = Array.isArray(summary.challenger_results) ? summary.challenger_results : [];
  const retired = Array.isArray(summary.retired_challengers) ? summary.retired_challengers : [];
  const leaderboardHtml = results.length
    ? '<div style="margin-top:10px;"><strong>Challenger leaderboard:</strong><ul style="margin:6px 0 0 18px;">' +
        results.map(function(item) {
          return '<li><strong>' + escapeHtml(item.policy_name || item.engine || '-') + '</strong>' +
            ' — utility Δ ' + escapeHtml(formatNumber(item.utility_score_delta_vs_legacy)) +
            ', pairwise Δ ' + escapeHtml(formatNumber(item.pairwise_delta_vs_legacy)) +
            ', matured ' + escapeHtml(String(item.matured_comparisons || 0)) +
            ', retire recommended: ' + escapeHtml(item.retire_recommended ? 'yes' : 'no') +
            '</li>';
        }).join('') +
      '</ul></div>'
    : '';
  const retiredText = retired.length
    ? retired.map(function(item) { return escapeHtml(item.policy_name || item.engine || '-'); }).join(', ')
    : '-';
  panel.innerHTML =
    '<div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">' +
    '<span style="display:inline-block; padding:4px 10px; border-radius:999px; background:' + badge.color + '; color:#fff; font-size:12px; font-weight:600;">' + escapeHtml(badge.label) + '</span>' +
    '<span><strong>Status:</strong> ' + escapeHtml(summary.status || '-') + '</span>' +
    '<span><strong>Updated:</strong> ' + escapeHtml(summary.generated_at_utc || '-') + '</span>' +
    '<span><strong>Pack ready:</strong> ' + escapeHtml(summary.pack_available ? 'yes' : 'no') + '</span>' +
    '</div>' +
    '<p><strong>Headline:</strong> ' + escapeHtml(summary.headline || '-') + '</p>' +
    '<p><strong>Summary:</strong> ' + escapeHtml(summary.summary || '-') + '</p>' +
    '<p><strong>Verdict:</strong> ' + escapeHtml(summary.verdict || '-') + '</p>' +
    '<p><strong>Matured comparisons / waiting:</strong> ' + escapeHtml(String(summary.matured_comparisons || 0)) + ' / ' + escapeHtml(String(summary.waiting_for_maturity || 0)) + '</p>' +
    '<p><strong>Top incumbent utility / pairwise / gap:</strong> ' + escapeHtml(formatNumber(incumbent.scan_shortlist_utility_score)) + ' / ' + escapeHtml(formatNumber(incumbent.scan_shortlist_pairwise_win_rate)) + ' / ' + escapeHtml(formatNumber(incumbent.scan_shortlist_mean_gap)) + '</p>' +
    '<p><strong>Top challenger utility / pairwise / gap:</strong> ' + escapeHtml(formatNumber(challenger.scan_shortlist_utility_score)) + ' / ' + escapeHtml(formatNumber(challenger.scan_shortlist_pairwise_win_rate)) + ' / ' + escapeHtml(formatNumber(challenger.scan_shortlist_mean_gap)) + '</p>' +
    '<p><strong>Retired challengers:</strong> ' + retiredText + '</p>' +
    leaderboardHtml;
  setShadowSelectionOutcomeReviewDownloadsEnabled(!!summary.pack_available, 'Pack unavailable until at least one recorded shadow comparison has matured.');
}

async function loadShadowSelectionOutcomeReview(silent) {
  const msg = document.getElementById('shadowSelectionOutcomeReviewMessage');
  if (!silent && msg) msg.textContent = 'Loading shadow outcome review…';
  try {
    const res = await fetch('/api/shadow-selection-outcome-review/summary');
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    renderShadowSelectionOutcomeReview(out);
    if (msg) msg.textContent = (!silent ? 'Shadow outcome review loaded' : ('Shadow outcome review ' + (out.status || '')));
  } catch (err) {
    setShadowSelectionOutcomeReviewDownloadsEnabled(false, 'Pack unavailable until at least one recorded shadow comparison has matured.');
    if (msg) msg.textContent = 'Could not load shadow outcome review: ' + err.message;
  }
}

async function downloadShadowSelectionOutcomeReviewPack() {
  const msg = document.getElementById('shadowSelectionOutcomeReviewMessage');
  let summary = shadowSelectionOutcomeReviewUiState.summary || {};
  if (!summary.available) {
    try {
      const res = await fetch('/api/shadow-selection-outcome-review/summary');
      if (!res.ok) throw new Error(await res.text());
      summary = await res.json();
      renderShadowSelectionOutcomeReview(summary);
    } catch (err) {
      if (msg) msg.textContent = 'Could not load shadow outcome review before download: ' + err.message;
      return;
    }
  }
  if (!summary.pack_available) {
    if (msg) msg.textContent = 'No shadow outcome review pack is ready yet. Current state: ' + (summary.status || 'waiting');
    return;
  }
  if (msg) msg.textContent = 'Downloading shadow outcome review pack…';
  window.location.href = '/api/shadow-selection-outcome-review/latest-pack.zip';
}

function freshRetrainAuditPasswordOrThrow() {
  const localPw = (document.getElementById('freshRetrainAuditAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}


let freshRetrainAuditPollTimer = null;

function stopFreshRetrainAuditPolling() {
  if (freshRetrainAuditPollTimer) {
    clearTimeout(freshRetrainAuditPollTimer);
    freshRetrainAuditPollTimer = null;
  }
}

function scheduleFreshRetrainAuditPolling() {
  stopFreshRetrainAuditPolling();
  freshRetrainAuditPollTimer = setTimeout(function() {
    loadFreshRetrainAuditSummary(true);
  }, 5000);
}

function normalizeAsyncRunState(summary) {
  const rawStatus = String((summary && summary.status) || '').trim().toLowerCase();
  const hasFinishedAt = !!(summary && summary.finished_at_utc);
  let status = rawStatus || ((summary && summary.running) ? 'running' : 'unknown');
  if (hasFinishedAt && (status === 'running' || status === 'stopping' || status === 'unknown')) {
    status = 'completed';
  }
  const terminalStatuses = new Set(['completed', 'cancelled', 'failed', 'interrupted']);
  const terminal = hasFinishedAt || terminalStatuses.has(status);
  const running = !terminal && (status === 'running' || status === 'stopping' || !!(summary && summary.running));
  return { status, running, terminal, hasFinishedAt };
}

function normalizeAsyncProgress(summary, state, finishedDetailText) {
  const base = Object.assign({
    stage: state.status || 'unknown',
    detail: (summary && summary.summary) || '-',
    fraction: 0,
    completed_symbols: 0,
    total_symbols: 0,
    current_symbol: null,
  }, (summary && summary.progress) || {});
  let fraction = Number(base.fraction || 0);
  if (!Number.isFinite(fraction)) fraction = 0;
  fraction = Math.max(0, Math.min(1, fraction));
  let completed = Number(base.completed_symbols || 0);
  if (!Number.isFinite(completed)) completed = 0;
  let total = Number(base.total_symbols || 0);
  if (!Number.isFinite(total)) total = 0;
  let stage = String(base.stage || state.status || 'unknown');
  let detail = String(base.detail || (summary && summary.summary) || '-');
  let currentSymbol = base.current_symbol || null;
  if (state.terminal) {
    if (state.status === 'completed') {
      fraction = 1;
      if (total > 0 && completed < total) completed = total;
      stage = 'completed';
      detail = finishedDetailText;
      currentSymbol = null;
    } else if (!detail || detail === '-') {
      detail = (summary && summary.summary) || state.status;
    }
  }
  return {
    stage,
    detail,
    fraction,
    completed_symbols: Math.max(0, Math.trunc(completed)),
    total_symbols: Math.max(0, Math.trunc(total)),
    current_symbol: currentSymbol,
  };
}

function renderFreshRetrainAuditSummary(summary) {
  const panel = document.getElementById('freshRetrainAuditPanel');
  if (!panel) return;
  if (!summary) {
    panel.innerHTML = '<p class="muted">No fresh retrain audit summary available yet.</p>';
    return;
  }
  const live = summary.current_live_path || {};
  const spec = summary.shadow_training_spec || {};
  const shadow = summary.shadow_model_result || {};
  const trainConc = summary.training_symbol_concentration || {};
  const qualityConc = summary.quality_symbol_concentration || {};
  const live045 = ((summary.live_outlier_concentration || {}).threshold_0_45) || {};
  const retrainNote = summary.future_retrain_spec_note || {};
  const state = normalizeAsyncRunState(summary);
  const progress = normalizeAsyncProgress(summary, state, 'Run finished. Use the summary and pack links below.');
  const fraction = Math.max(0, Math.min(1, Number(progress.fraction || 0)));
  const pct = Math.round(fraction * 100);
  const running = state.running;
  const heartbeat = summary.last_heartbeat_at_utc || summary.generated_at_utc || '-';
  panel.innerHTML =
    '<div class="card" style="margin-bottom:14px;">' +
      '<p><strong>Progress:</strong> ' + pct + '%</p>' +
      '<div class="progress-shell"><div class="progress-bar" style="width:' + pct + '%"></div></div>' +
      '<p><strong>Stage:</strong> ' + (progress.stage || summary.status || '-') + '</p>' +
      '<p><strong>Detail:</strong> ' + (progress.detail || summary.summary || '-') + '</p>' +
      '<p><strong>Symbols processed:</strong> ' + (progress.completed_symbols || 0) + ' / ' + (progress.total_symbols || 0) + '</p>' +
      '<p><strong>Current symbol:</strong> ' + (progress.current_symbol || '-') + '</p>' +
      '<p><strong>Heartbeat:</strong> ' + heartbeat + '</p>' +
      '<p><strong>Stop requested:</strong> ' + ((summary.stop_requested) ? 'true' : 'false') + '</p>' +
      '<p class="small muted">' + (running ? 'Auto-refreshing every 5 seconds while this branch is running.' : 'Run finished. Use the summary and pack links below.') + '</p>' +
    '</div>' +
    '<div class="grid">' +
      '<div class="card"><h3>Branch state</h3>' +
        '<p><strong>Status:</strong> ' + (state.status || summary.status || '-') + '</p>' +
        '<p><strong>Started:</strong> ' + (summary.started_at_utc || '-') + '</p>' +
        '<p><strong>Finished:</strong> ' + (summary.finished_at_utc || '-') + '</p>' +
        '<p><strong>Headline:</strong> ' + (summary.headline || '-') + '</p>' +
        '<p><strong>Verdict:</strong> ' + (summary.verdict || '-') + '</p>' +
        '<p class="small"><strong>Summary:</strong> ' + (summary.summary || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Current live path</h3>' +
        '<p><strong>Checkpoint outcome:</strong> ' + (live.decision_checkpoint_outcome || '-') + '</p>' +
        '<p><strong>Visible q-hit:</strong> ' + fmtPct(live.visible_quality_hit_rate) + '</p>' +
        '<p><strong>Non-visible q-hit:</strong> ' + fmtPct(live.non_visible_quality_hit_rate) + '</p>' +
        '<p><strong>Resolved visible rows:</strong> ' + (live.resolved_visible_rows || 0) + '</p>' +
        '<p><strong>Stage 1 mode:</strong> ' + (live.stage1_selection_mode || '-') + '</p>' +
        '<p><strong>Threshold:</strong> ' + fmtPct(live.live_raw_threshold) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Shadow model</h3>' +
        '<p><strong>Type:</strong> ' + (shadow.model_type || '-') + '</p>' +
        '<p><strong>Adjusted AUC:</strong> ' + fmtNum(shadow.adjusted_auc_holdout, 4) + '</p>' +
        '<p><strong>Adjusted Brier:</strong> ' + fmtNum(shadow.adjusted_brier_holdout, 4) + '</p>' +
        '<p><strong>Rows:</strong> ' + (shadow.training_rows || 0) + '</p>' +
        '<p><strong>Quality event rate:</strong> ' + fmtPct(shadow.quality_event_rate) + '</p>' +
        '<p><strong>Trained:</strong> ' + (shadow.trained_at_utc || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Concentration audit</h3>' +
        '<p><strong>Training top symbol:</strong> ' + (trainConc.top_symbol || '-') + ' (' + fmtPct(trainConc.top_symbol_share) + ')</p>' +
        '<p><strong>Quality top symbol:</strong> ' + (qualityConc.top_symbol || '-') + ' (' + fmtPct(qualityConc.top_symbol_share) + ')</p>' +
        '<p><strong>Live >=0.45 top symbol:</strong> ' + (live045.top_symbol || '-') + ' (' + fmtPct(live045.top_symbol_share) + ')</p>' +
        '<p><strong>Include concentration controls:</strong> ' + (retrainNote.include_symbol_concentration_controls ? 'yes' : 'no') + '</p>' +
        '<p class="small"><strong>Reason:</strong> ' + (retrainNote.reason || '-') + '</p>' +
      '</div>' +
    '</div>' +
    '<div class="card" style="margin-top:14px;">' +
      '<p><strong>Training spec:</strong> ' + (spec.label || '-') + ' | days=' + (spec.train_lookback_days || '-') + ' | max symbols=' + (spec.train_max_symbols || '-') + ' | sample every=' + (spec.train_sample_every_n_bars || '-') + '</p>' +
      '<p><strong>Promotion blocked:</strong> ' + ((summary.live_promotion_blocked) ? 'true' : 'false') + '</p>' +
      '<p><strong>Pack path:</strong> ' + (((summary.artifact_paths || {}).pack_path) || '-') + '</p>' +
    '</div>';
  if (running) {
    scheduleFreshRetrainAuditPolling();
  } else {
    stopFreshRetrainAuditPolling();
  }
}

async function runFreshRetrainAudit() {
  const msg = document.getElementById('freshRetrainAuditMessage');
  msg.textContent = 'Starting fresh retrain audit…';
  stopFreshRetrainAuditPolling();
  try {
    const pw = freshRetrainAuditPasswordOrThrow();
    const out = await getJson('/api/reviews/fresh-retrain-audit/run', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderFreshRetrainAuditSummary(out);
    msg.textContent = (out && out.running) ? 'Fresh retrain audit started — auto-refresh is on' : 'Fresh retrain audit request processed';
  } catch (e) {
    msg.textContent = e.message;
  }
}

async function loadFreshRetrainAuditSummary(silent) {
  const msg = document.getElementById('freshRetrainAuditMessage');
  if (!silent) msg.textContent = 'Loading fresh retrain audit summary…';
  try {
    const pw = freshRetrainAuditPasswordOrThrow();
    const summary = await getJson('/api/reviews/fresh-retrain-audit/summary', { headers: { 'X-Admin-Password': pw } });
    renderFreshRetrainAuditSummary(summary);
    if (!silent) msg.textContent = summary && summary.running ? 'Fresh retrain audit summary loaded — still running' : 'Fresh retrain audit summary loaded';
  } catch (e) {
    if (!silent) msg.textContent = e.message;
    stopFreshRetrainAuditPolling();
  }
}

async function stopFreshRetrainAudit() {
  const msg = document.getElementById('freshRetrainAuditMessage');
  msg.textContent = 'Stopping fresh retrain audit…';
  try {
    const pw = freshRetrainAuditPasswordOrThrow();
    const out = await getJson('/api/reviews/fresh-retrain-audit/stop', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderFreshRetrainAuditSummary(out);
    msg.textContent = out && (out.status === 'stopping' || out.running) ? 'Fresh retrain audit stop requested' : 'Fresh retrain audit stop request processed';
    scheduleFreshRetrainAuditPolling();
  } catch (e) {
    msg.textContent = e.message;
  }
}

function downloadFreshRetrainAuditPack() {
  const msg = document.getElementById('freshRetrainAuditMessage');
  try {
    const pw = freshRetrainAuditPasswordOrThrow();
    const url = '/api/reviews/fresh-retrain-audit/latest-pack.zip?admin_password=' + encodeURIComponent(pw);
    window.open(url, '_blank');
    msg.textContent = 'Fresh retrain audit pack download started';
  } catch (e) {
    msg.textContent = e.message;
  }
}


function challengerComparisonPasswordOrThrow() {
  const localPw = (document.getElementById('challengerComparisonAdminPassword') || {}).value || '';
  const auditPw = (document.getElementById('freshRetrainAuditAdminPassword') || {}).value || '';
  const globalPw = (document.getElementById('adminPassword') || {}).value || '';
  const pw = localPw || auditPw || globalPw;
  if (!pw) throw new Error('enter admin password first');
  return pw;
}

let challengerComparisonPollTimer = null;

function stopChallengerComparisonPolling() {
  if (challengerComparisonPollTimer) {
    clearTimeout(challengerComparisonPollTimer);
    challengerComparisonPollTimer = null;
  }
}

function scheduleChallengerComparisonPolling() {
  stopChallengerComparisonPolling();
  challengerComparisonPollTimer = setTimeout(function() {
    loadChallengerComparisonSummary(true);
  }, 5000);
}

function renderChallengerComparisonSummary(summary) {
  const panel = document.getElementById('challengerComparisonPanel');
  if (!panel) return;
  if (!summary) {
    panel.innerHTML = '<p class="muted">No challenger comparison summary available yet.</p>';
    return;
  }
  const live = summary.current_live_path || {};
  const incumbent = summary.incumbent_model || {};
  const challenger = summary.challenger_model || {};
  const comparison = summary.comparison || {};
  const deltas = comparison.deltas || {};
  const evalMeta = summary.shared_evaluation_frame || {};
  const state = normalizeAsyncRunState(summary);
  const progress = normalizeAsyncProgress(summary, state, 'Run finished. Use the summary and pack links above.');
  const fraction = Math.max(0, Math.min(1, Number(progress.fraction || 0)));
  const pct = Math.round(fraction * 100);
  const running = state.running;
  const heartbeat = summary.last_heartbeat_at_utc || summary.generated_at_utc || '-';
  const incumbent45 = ((incumbent.threshold_stats_adjusted || {})['0.45']) || {};
  const challenger45 = ((challenger.threshold_stats_adjusted || {})['0.45']) || {};
  const incumbent50 = ((incumbent.threshold_stats_adjusted || {})['0.50']) || {};
  const challenger50 = ((challenger.threshold_stats_adjusted || {})['0.50']) || {};
  const incumbentConc45 = (((incumbent.concentration || {})['0.45']) || {});
  const challengerConc45 = (((challenger.concentration || {})['0.45']) || {});
  const incumbentScan = (incumbent.scan_shortlist_utility || {});
  const challengerScan = (challenger.scan_shortlist_utility || {});
  panel.innerHTML =
    '<div class="card" style="margin-bottom:14px;">' +
      '<p><strong>Progress:</strong> ' + pct + '%</p>' +
      '<div class="progress-shell"><div class="progress-bar" style="width:' + pct + '%"></div></div>' +
      '<p><strong>Stage:</strong> ' + (progress.stage || summary.status || '-') + '</p>' +
      '<p><strong>Detail:</strong> ' + (progress.detail || summary.summary || '-') + '</p>' +
      '<p><strong>Symbols processed:</strong> ' + (progress.completed_symbols || 0) + ' / ' + (progress.total_symbols || 0) + '</p>' +
      '<p><strong>Current symbol:</strong> ' + (progress.current_symbol || '-') + '</p>' +
      '<p><strong>Heartbeat:</strong> ' + heartbeat + '</p>' +
      '<p><strong>Stop requested:</strong> ' + ((summary.stop_requested) ? 'true' : 'false') + '</p>' +
      '<p class="small muted">' + (running ? 'Auto-refreshing every 5 seconds while this comparison is running.' : 'Run finished. Use the summary and pack links above.') + '</p>' +
    '</div>' +
    '<div class="grid">' +
      '<div class="card"><h3>Branch context</h3>' +
        '<p><strong>Status:</strong> ' + (summary.status || '-') + '</p>' +
        '<p><strong>Verdict:</strong> ' + (summary.verdict || '-') + '</p>' +
        '<p><strong>Recommended action:</strong> ' + (summary.recommended_action || '-') + '</p>' +
        '<p><strong>Checkpoint outcome:</strong> ' + (live.decision_checkpoint_outcome || '-') + '</p>' +
        '<p><strong>Visible q-hit:</strong> ' + fmtPct(live.visible_quality_hit_rate) + '</p>' +
        '<p><strong>Non-visible q-hit:</strong> ' + fmtPct(live.non_visible_quality_hit_rate) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Shared eval frame</h3>' +
        '<p><strong>Rows test:</strong> ' + (evalMeta.rows_test || 0) + '</p>' +
        '<p><strong>Rows all:</strong> ' + (evalMeta.rows_all || 0) + '</p>' +
        '<p><strong>Symbols used:</strong> ' + (evalMeta.symbols_used_count || 0) + '</p>' +
        '<p><strong>Lookback days:</strong> ' + (evalMeta.train_lookback_days || '-') + '</p>' +
        '<p><strong>Sample every:</strong> ' + (evalMeta.sample_every_n_bars || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Incumbent</h3>' +
        '<p><strong>Adjusted AUC:</strong> ' + fmtNum(incumbent.adjusted_auc_holdout, 4) + '</p>' +
        '<p><strong>Adjusted Brier:</strong> ' + fmtNum(incumbent.adjusted_brier_holdout, 4) + '</p>' +
        '<p><strong>>=0.45 count / precision:</strong> ' + (incumbent45.count || 0) + ' / ' + fmtPct(incumbent45.precision) + '</p>' +
        '<p><strong>>=0.50 count / precision:</strong> ' + (incumbent50.count || 0) + ' / ' + fmtPct(incumbent50.precision) + '</p>' +
        '<p><strong>>=0.45 top symbol:</strong> ' + (incumbentConc45.top_symbol || '-') + ' (' + fmtPct(incumbentConc45.top_symbol_share) + ')</p>' +
        '<p><strong>Scan utility / gap:</strong> ' + fmtNum(incumbentScan.utility_score, 4) + ' / ' + fmtPct(incumbentScan.mean_gap) + '</p>' +
        '<p><strong>Top-1 / Top-3 scan quality:</strong> ' + fmtPct(incumbentScan.top1_mean_quality) + ' / ' + fmtPct(incumbentScan.top3_mean_quality) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Challenger</h3>' +
        '<p><strong>Adjusted AUC:</strong> ' + fmtNum(challenger.adjusted_auc_holdout, 4) + '</p>' +
        '<p><strong>Adjusted Brier:</strong> ' + fmtNum(challenger.adjusted_brier_holdout, 4) + '</p>' +
        '<p><strong>>=0.45 count / precision:</strong> ' + (challenger45.count || 0) + ' / ' + fmtPct(challenger45.precision) + '</p>' +
        '<p><strong>>=0.50 count / precision:</strong> ' + (challenger50.count || 0) + ' / ' + fmtPct(challenger50.precision) + '</p>' +
        '<p><strong>>=0.45 top symbol:</strong> ' + (challengerConc45.top_symbol || '-') + ' (' + fmtPct(challengerConc45.top_symbol_share) + ')</p>' +
        '<p><strong>Scan utility / gap:</strong> ' + fmtNum(challengerScan.utility_score, 4) + ' / ' + fmtPct(challengerScan.mean_gap) + '</p>' +
        '<p><strong>Top-1 / Top-3 scan quality:</strong> ' + fmtPct(challengerScan.top1_mean_quality) + ' / ' + fmtPct(challengerScan.top3_mean_quality) + '</p>' +
      '</div>' +
    '</div>' +
    '<div class="card" style="margin-top:14px;">' +
      '<p><strong>Summary:</strong> ' + (comparison.summary || summary.summary || '-') + '</p>' +
      '<p><strong>Scan utility delta:</strong> ' + fmtNum(deltas.scan_shortlist_utility_score_delta, 4) + '</p>' +
      '<p><strong>Per-scan quality-gap delta:</strong> ' + fmtPct(deltas.scan_shortlist_mean_gap_delta) + '</p>' +
      '<p><strong>Per-scan win-rate delta:</strong> ' + fmtPct(deltas.scan_shortlist_pairwise_win_rate_delta) + '</p>' +
      '<p><strong>Top-of-scan quality delta:</strong> ' + fmtPct(deltas.scan_shortlist_top1_mean_quality_delta) + '</p>' +
      '<p><strong>Top-3 scan quality delta:</strong> ' + fmtPct(deltas.scan_shortlist_top3_mean_quality_delta) + '</p>' +
      '<p><strong>Avg visible rows/scan delta:</strong> ' + fmtNum(deltas.scan_shortlist_avg_visible_rows_per_scan_delta, 2) + '</p>' +
      '<p><strong>Adjusted AUC delta:</strong> ' + fmtNum(deltas.adjusted_auc_delta, 4) + '</p>' +
      '<p><strong>Adjusted Brier delta:</strong> ' + fmtNum(deltas.adjusted_brier_delta, 4) + ' <span class="small">(negative is better for challenger)</span></p>' +
      '<p><strong>>=0.45 precision delta:</strong> ' + fmtPct(deltas.precision_ge_0_45_delta) + '</p>' +
      '<p><strong>Concentration OK:</strong> ' + (comparison.concentration_ok ? 'true' : 'false') + '</p>' +
    '</div>';
  if (running) {
    scheduleChallengerComparisonPolling();
  } else {
    stopChallengerComparisonPolling();
  }
}

async function runChallengerComparison() {
  const msg = document.getElementById('challengerComparisonMessage');
  msg.textContent = 'Starting offline challenger comparison…';
  stopChallengerComparisonPolling();
  try {
    const pw = challengerComparisonPasswordOrThrow();
    const out = await getJson('/api/reviews/challenger-comparison/run', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderChallengerComparisonSummary(out);
    msg.textContent = (out && out.running) ? 'Offline challenger comparison started — auto-refresh is on' : 'Offline challenger comparison request processed';
  } catch (e) {
    msg.textContent = e.message;
  }
}

async function loadChallengerComparisonSummary(silent) {
  const msg = document.getElementById('challengerComparisonMessage');
  if (!silent) msg.textContent = 'Loading challenger comparison summary…';
  try {
    const pw = challengerComparisonPasswordOrThrow();
    const summary = await getJson('/api/reviews/challenger-comparison/summary', { headers: { 'X-Admin-Password': pw } });
    renderChallengerComparisonSummary(summary);
    if (!silent) msg.textContent = summary && summary.running ? 'Challenger comparison summary loaded — still running' : 'Challenger comparison summary loaded';
  } catch (e) {
    if (!silent) msg.textContent = e.message;
    stopChallengerComparisonPolling();
  }
}

async function stopChallengerComparison() {
  const msg = document.getElementById('challengerComparisonMessage');
  msg.textContent = 'Stopping offline challenger comparison…';
  try {
    const pw = challengerComparisonPasswordOrThrow();
    const out = await getJson('/api/reviews/challenger-comparison/stop', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    renderChallengerComparisonSummary(out);
    msg.textContent = out && (out.status === 'stopping' || out.running) ? 'Challenger comparison stop requested' : 'Challenger comparison stop request processed';
    scheduleChallengerComparisonPolling();
  } catch (e) {
    msg.textContent = e.message;
  }
}

function downloadChallengerComparisonPack() {
  const msg = document.getElementById('challengerComparisonMessage');
  try {
    const pw = challengerComparisonPasswordOrThrow();
    const url = '/api/reviews/challenger-comparison/latest-pack.zip?admin_password=' + encodeURIComponent(pw);
    window.open(url, '_blank');
    msg.textContent = 'Challenger comparison pack download started';
  } catch (e) {
    msg.textContent = e.message;
  }
}

function renderCurrentVersionSummary(summary) {
  const panel = document.getElementById('currentVersionSummaryPanel');
  if (!panel) return;
  if (!summary) {
    panel.innerHTML = '<p class="muted">Current-version evidence summary unavailable.</p>';
    return;
  }
  const evidence = summary.evidence || {};
  const range = evidence.score_range || {};
  const decisionRule = summary.decision_checkpoint || summary.decision_rule_checkpoint || {};
  const decisionBranch = summary.decision_branch_automation || {};
  const decisionSummary = summary.decision_summary || {};
  const regimeRows = (summary.regime_breakdown || []).map(function(row) {
    return '<tr><td>' + (row.market_regime_state || '-') + '</td><td>' + (row.market_regime_actionability || '-') + '</td><td>' + (row.run_count || 0) + '</td><td>' + (row.evaluated_run_count || 0) + '</td><td>' + (row.visible_rows || 0) + '</td><td>' + (row.suppressed_rows || 0) + '</td></tr>';
  }).join('');
  const regimeEvidence = summary.regime_evidence || {};
  const regimeEvidenceRows = (regimeEvidence.rows || []).map(function(row) {
    return '<tr><td>' + (row.market_regime_state || '-') + '</td><td>' + (row.market_regime_actionability || '-') + '</td><td>' + (row.resolved_rows || 0) + '</td><td>' + fmtPct(row.visible_quality_hit_rate) + '</td><td>' + fmtPct(row.non_visible_quality_hit_rate) + '</td><td>' + fmtPct(row.visible_avg_end_ret) + '</td><td>' + fmtPct(row.non_visible_avg_end_ret) + '</td></tr>';
  }).join('');
  const regimeThresholdRows = (regimeEvidence.threshold_rows || []).map(function(row) {
    return '<tr><td>' + (row.market_regime_state || '-') + '</td><td>' + (row.market_regime_actionability || '-') + '</td><td>&ge; ' + fmtNum(row.threshold, 2) + '</td><td>' + (row.count || 0) + '</td><td>' + (row.visible_count || 0) + '</td><td>' + (row.non_visible_count || 0) + '</td><td>' + fmtPct(row.quality_hit_rate) + '</td></tr>';
  }).join('');
  const bandRows = (evidence.threshold_bands || []).map(function(row) {
    return '<tr><td>&ge; ' + fmtNum(row.threshold, 2) + '</td><td>' + (row.count || 0) + '</td><td>' + (row.visible_count || 0) + '</td><td>' + (row.non_visible_count || 0) + '</td><td>' + fmtPct(row.quality_hit_rate) + '</td><td>' + fmtPct(row.raw_hit_rate) + '</td></tr>';
  }).join('');
  const scanBandRows = ((summary.scan_score_diagnostics || {}).counts_above_thresholds || []).map(function(row) {
    return '<tr><td>&ge; ' + fmtNum(row.threshold, 2) + '</td><td>' + (row.model_count || 0) + '</td><td>' + (row.pre_policy_count || 0) + '</td><td>' + (row.live_count || 0) + '</td></tr>';
  }).join('');
  const candidateQualityRows = ((summary.candidate_quality || {}).tiers || []).map(function(row) {
    return '<tr><td>' + (row.liquidity_tier || '-') + '</td><td>' + (row.stage1_selected || 0) + '</td><td>' + (row.stage2_scored || 0) + '</td><td>' + (row.stage2_visible || 0) + '</td><td>' + (row.stage2_hidden || 0) + '</td><td>' + fmtNum(row.max_live_score, 4) + '</td><td>' + (row.stage2_count_ge_0_35 || 0) + '</td></tr>';
  }).join('');
  const cohortRows = ((summary.cohort_symbols || {}).rows || []).slice(0, 10).map(function(row) {
    return '<tr><td>' + (row.symbol || '-') + '</td><td>' + (row.liquidity_tier || '-') + '</td><td>' + (row.selected_scans || 0) + '</td><td>' + (row.visible_scans || 0) + '</td><td>' + fmtNum(row.max_live_score, 4) + '</td><td>' + (row.count_ge_0_35 || 0) + '</td></tr>';
  }).join('');
  panel.innerHTML =
    '<div class="grid">' +
      '<div class="card"><h3>Version bundle</h3>' +
        '<p><strong>Version:</strong> ' + (summary.app_version || '-') + '</p>' +
        '<p><strong>Deployed since:</strong> ' + (summary.deployed_since_utc || '-') + '</p>' +
        '<p><strong>Scan packs:</strong> ' + (summary.scan_pack_count || 0) + '</p>' +
        '<p><strong>Evaluated packs:</strong> ' + (summary.evaluated_pack_count || 0) + '</p>' +
        '<p><strong>Headline:</strong> ' + (evidence.headline || '-') + '</p>' +
        '<p class="small"><strong>Summary:</strong> ' + (evidence.summary || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Score range since deployment</h3>' +
        '<p><strong>Max live:</strong> ' + fmtNum(range.max_live_score, 4) + '</p>' +
        '<p><strong>P95 live:</strong> ' + fmtNum(range.p95_live_score, 4) + '</p>' +
        '<p><strong>Median live:</strong> ' + fmtNum(range.median_live_score, 4) + '</p>' +
        '<p><strong>Max pre-policy:</strong> ' + fmtNum(range.max_pre_policy_score, 4) + '</p>' +
        '<p><strong>P95 pre-policy:</strong> ' + fmtNum(range.p95_pre_policy_score, 4) + '</p>' +
        '<p><strong>Median pre-policy:</strong> ' + fmtNum(range.median_pre_policy_score, 4) + '</p>' +
      '</div>' +
      '<div class="card"><h3>Resolved evidence</h3>' +
        '<p><strong>Resolved rows:</strong> ' + (evidence.resolved_rows || 0) + '</p>' +
        '<p><strong>Visible rows:</strong> ' + (evidence.visible_rows || 0) + ' / q-hit=' + fmtPct(evidence.visible_quality_hit_rate) + '</p>' +
        '<p><strong>Non-visible rows:</strong> ' + (evidence.non_visible_rows || 0) + ' / q-hit=' + fmtPct(evidence.non_visible_quality_hit_rate) + '</p>' +
        '<p><strong>Display-trim q-hits:</strong> ' + (evidence.display_trim_quality_hits || 0) + '</p>' +
        '<p><strong>Threshold q-hits:</strong> ' + (evidence.threshold_quality_hits || 0) + '</p>' +
        '<p><strong>Validated bands dormant:</strong> ' + (evidence.validated_bands_dormant ? 'true' : 'false') + '</p>' +
        '<p class="small"><strong>Regime semantics:</strong> ' + (summary.regime_semantics_note || 'Regime labels are being applied directly as stored.') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Objective decision support</h3>' +
        '<p><strong>Headline:</strong> ' + (decisionSummary.headline || '-') + '</p>' +
        '<p><strong>Confirmed shortlist:</strong> ' + (decisionSummary.objective_confirmed_rows || 0) + '</p>' +
        '<p><strong>Strong / priority / elite:</strong> ' + (decisionSummary.strong_edge_rows || 0) + ' / ' + (decisionSummary.priority_edge_rows || 0) + ' / ' + (decisionSummary.elite_edge_rows || 0) + '</p>' +
        '<p><strong>Blocked near threshold:</strong> ' + (decisionSummary.blocked_near_threshold_rows || 0) + '</p>' +
        '<p><strong>Effective regime actionability:</strong> ' + (decisionSummary.market_regime_actionability || '-') + '</p>' +
        '<p class="small"><strong>Summary:</strong> ' + (decisionSummary.summary || '-') + '</p>' +
      '</div>' +
      '<div class="card"><h3>Decision checkpoint</h3>' +
        '<p><strong>Stage1 mode:</strong> ' + (decisionRule.stage1_selection_mode || '-') + '</p>' +
        '<p><strong>Resolved visible rows:</strong> ' + (decisionRule.resolved_visible_rows || 0) + ' / target=' + (decisionRule.decision_target_visible_rows || 30) + '</p>' +
        '<p><strong>Visible q-hit:</strong> ' + fmtPct(decisionRule.current_visible_quality_hit_rate) + '</p>' +
        '<p><strong>Non-visible q-hit:</strong> ' + fmtPct(decisionRule.current_non_visible_quality_hit_rate) + '</p>' +
        '<p><strong>Rows remaining:</strong> ' + (decisionRule.rows_remaining_until_decision || 0) + '</p>' +
        '<p><strong>Status:</strong> ' + (decisionRule.decision_checkpoint_outcome || decisionRule.current_outcome || decisionRule.status || '-') + '</p>' +
        '<p><strong>Triggered at:</strong> ' + (decisionRule.decision_checkpoint_triggered_at_utc || '-') + '</p>' +
        '<p class="small"><strong>Rule:</strong> confirm at >=15% visible q-hit across 30+ resolved visible rows; falsify if visible q-hit falls below non-visible at 30+ rows.</p>' +
      '</div>' +
      '<div class="card"><h3>Decision branch automation</h3>' +
        '<p><strong>Branch status:</strong> ' + (((decisionBranch.branch_action || {}).status) || '-') + '</p>' +
        '<p><strong>Checkpoint outcome:</strong> ' + (decisionBranch.checkpoint_outcome || '-') + '</p>' +
        '<p><strong>Auto-execute:</strong> ' + (decisionBranch.auto_execute_supported_actions_enabled ? 'enabled' : 'disabled') + '</p>' +
        '<p><strong>Configured / effective threshold:</strong> ' + fmtPct(decisionBranch.configured_live_raw_threshold) + ' / ' + fmtPct(decisionBranch.effective_live_raw_threshold) + '</p>' +
        '<p><strong>Next action:</strong> ' + (((decisionBranch.branch_action || {}).next_action_label) || '-') + '</p>' +
        '<p><strong>Override active:</strong> ' + ((((decisionBranch.runtime_overrides || {}).threshold_experiment_active)) ? 'true' : 'false') + '</p>' +
        '<p><strong>Last execution:</strong> ' + (decisionBranch.last_execution_result || '-') + '</p>' +
      '</div>' +
    '</div>' +
    '<h3 style="margin-top:16px;">Per-scan score diagnostics</h3>' +
    '<table><thead><tr><th>Band</th><th>Model count</th><th>Pre-policy count</th><th>Live count</th></tr></thead><tbody>' + (scanBandRows || '<tr><td colspan="4" class="muted">No per-scan diagnostics yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Threshold bands</h3>' +
    '<table><thead><tr><th>Band</th><th>Count</th><th>Visible</th><th>Non-visible</th><th>Quality hit</th><th>Raw hit</th></tr></thead><tbody>' + (bandRows || '<tr><td colspan="6" class="muted">No resolved threshold evidence yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Candidate quality by tier</h3>' +
    '<table><thead><tr><th>Tier</th><th>Stage1 selected</th><th>Stage2 scored</th><th>Visible</th><th>Hidden</th><th>Max live</th><th>&ge;0.35</th></tr></thead><tbody>' + (candidateQualityRows || '<tr><td colspan="7" class="muted">No candidate-quality diagnostics yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Cohort symbol summary</h3>' +
    '<table><thead><tr><th>Symbol</th><th>Tier</th><th>Selected scans</th><th>Visible scans</th><th>Max live</th><th>&ge;0.35</th></tr></thead><tbody>' + (cohortRows || '<tr><td colspan="6" class="muted">No cohort symbol summary yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Regime breakdown</h3>' +
    '<table><thead><tr><th>Regime</th><th>Actionability</th><th>Runs</th><th>Evaluated</th><th>Visible</th><th>Suppressed</th></tr></thead><tbody>' + (regimeRows || '<tr><td colspan="6" class="muted">No runs yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Evaluated evidence by regime</h3>' +
    '<p class="small muted">This is the key slice for checking whether the simplified path works in green but degrades in amber or red.</p>' +
    '<table><thead><tr><th>Regime</th><th>Actionability</th><th>Resolved</th><th>Visible q-hit</th><th>Hidden q-hit</th><th>Visible avg end ret</th><th>Hidden avg end ret</th></tr></thead><tbody>' + (regimeEvidenceRows || '<tr><td colspan="7" class="muted">No regime-sliced evaluated evidence yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Threshold bands by regime</h3>' +
    '<table><thead><tr><th>Regime</th><th>Actionability</th><th>Band</th><th>Count</th><th>Visible</th><th>Hidden</th><th>Q-hit</th></tr></thead><tbody>' + (regimeThresholdRows || '<tr><td colspan="7" class="muted">No regime threshold evidence yet</td></tr>') + '</tbody></table>';
}

function renderReviewRuns(payload) {
  const panel = document.getElementById('reviewRunsPanel');
  if (!panel) return;
  const runs = (payload && payload.runs) || [];
  if (!runs.length) {
    panel.innerHTML = '<p class="muted">No review runs yet. They will appear automatically after scans complete.</p>';
    return;
  }
  panel.innerHTML = '<table><thead><tr><th>Run</th><th>Finished</th><th>Regime</th><th>Visible</th><th>Suppressed</th><th>Evaluated</th><th>Downloads</th></tr></thead><tbody>' +
    runs.map(function(run) {
      return '<tr>' +
        '<td class="mono">' + run.run_id + '</td>' +
        '<td>' + (run.scan_finished_utc || '-') + '</td>' +
        '<td>' + (run.market_regime_state || '-') + ' <span class="small">(' + (run.market_regime_actionability || '-') + ')</span></td>' +
        '<td>' + (run.visible_rows_count || 0) + '</td>' +
        '<td>' + (run.suppressed_rows_count || 0) + '</td>' +
        '<td>' + (run.evaluation_complete ? 'yes' : 'pending') + '</td>' +
        '<td><a href="/api/runs/' + run.run_id + '/download?evaluated=false">scan</a>' +
        (run.evaluation_complete ? ' | <a href="/api/runs/' + run.run_id + '/download?evaluated=true">evaluated</a>' : '') + '</td>' +
      '</tr>';
    }).join('') + '</tbody></table>';
}

function renderSemanticsComparisonSummary(payload) {
  const panel = document.getElementById('semanticsComparisonPanel');
  if (!panel) return;
  if (!payload || payload.available === false) {
    panel.innerHTML = '<p class="muted">' + escapeHtml((payload && payload.summary) || 'No semantics comparison summary loaded yet.') + '</p>';
    return;
  }
  const rows = (payload.comparison_table || []).map(function(row) {
    return '<tr>' +
      '<td>' + escapeHtml(row.path_label || row.path_name || '-') + '</td>' +
      '<td>' + fmtNum(row.visible_quality_hit_rate, 3) + '</td>' +
      '<td>' + fmtNum(row.hidden_quality_hit_rate, 3) + '</td>' +
      '<td>' + (row.visible_count ?? '-') + '</td>' +
      '<td>' + (row.hidden_count ?? '-') + '</td>' +
      '<td>' + fmtNum(row.top_3_quality_rate, 3) + '</td>' +
      '<td>' + fmtNum(row.mean_shortlist_size, 2) + '</td>' +
      '</tr>';
  }).join('');
  const best = payload.best_path_now || {};
  const effects = (payload.obvious_effects || []).map(function(item) { return '<li>' + escapeHtml(item) + '</li>'; }).join('');
  panel.innerHTML =
    '<p><strong>Headline:</strong> ' + escapeHtml(payload.headline || '-') + '</p>' +
    '<p><strong>Summary:</strong> ' + escapeHtml(payload.summary || '-') + '</p>' +
    '<p><strong>Best path now:</strong> ' + escapeHtml(best.path_label || best.path_name || '-') + '</p>' +
    '<table><thead><tr><th>Path</th><th>Visible Q-hit</th><th>Hidden Q-hit</th><th>Visible rows</th><th>Hidden rows</th><th>Top-3 Q-hit</th><th>Mean shortlist</th></tr></thead><tbody>' + (rows || '<tr><td colspan="7" class="muted">No comparison rows yet</td></tr>') + '</tbody></table>' +
    '<h3 style="margin-top:16px;">Code-truth note</h3>' +
    '<p>' + escapeHtml(((payload.code_truth_note || {}).summary) || '-') + '</p>' +
    '<h3 style="margin-top:16px;">Scope note</h3>' +
    '<p>' + escapeHtml(((payload.scope_note || {}).summary) || '-') + '</p>' +
    '<h3 style="margin-top:16px;">Obvious effects</h3>' +
    '<ul>' + (effects || '<li class="muted">No obvious effects recorded yet.</li>') + '</ul>';
}

async function runSemanticsComparison() {
  const pw = (document.getElementById('semanticsComparisonAdminPassword') || {}).value || '';
  const hours = Number((document.getElementById('semanticsComparisonHours') || {}).value || 168);
  const stepMinutes = Number((document.getElementById('semanticsComparisonStepMinutes') || {}).value || 120);
  const maxScans = Number((document.getElementById('semanticsComparisonMaxScans') || {}).value || 84);
  const maxSymbols = Number((document.getElementById('semanticsComparisonMaxSymbols') || {}).value || 100);
  const msg = document.getElementById('semanticsComparisonMessage');
  if (msg) msg.textContent = 'Running semantics comparison…';
  try {
    const params = new URLSearchParams({
      hours: String(hours),
      step_minutes: String(stepMinutes),
      max_scans: String(maxScans),
      max_symbols: String(maxSymbols),
    });
    const out = await getJson('/api/semantics-comparison/run?' + params.toString(), { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (msg) msg.textContent = (out.summary || {}).headline || 'Semantics comparison completed.';
    renderSemanticsComparisonSummary(out.summary || out);
  } catch (e) {
    if (msg) msg.textContent = e.message;
  }
}

async function loadSemanticsComparisonSummary(silent) {
  const msg = document.getElementById('semanticsComparisonMessage');
  try {
    const payload = await getJson('/api/semantics-comparison/latest-summary');
    renderSemanticsComparisonSummary(payload);
    if (msg && !silent) msg.textContent = payload.headline || 'Loaded semantics comparison summary.';
  } catch (e) {
    if (msg && !silent) msg.textContent = e.message;
  }
}

function downloadSemanticsComparisonPack() {
  window.location.href = '/api/semantics-comparison/latest-pack.zip';
}

async function refreshAll() {
  attachSortHandlers();
  try {
    const [status, scores, training, validation, reliability, runs, policy24, policy7d, currentVersionSummary, decisionBranchSummary] = await Promise.all([
      getJson('/api/status'),
      getJson('/api/scores'),
      getJson('/api/training/status'),
      getJson('/api/paper-trade/summary'),
      getJson('/api/reliability-lab'),
      getJson('/api/runs?limit=10'),
      getJson('/api/policy-audit?hours=24'),
      getJson('/api/policy-audit?hours=168'),
      getJson('/api/reviews/current-version-summary'),
      getJson('/api/reviews/decision-branch')
    ]);
    renderStatus(status, scores);
    renderScores(status, scores);
    renderFollowup(status);
    renderTraining(training);
    renderValidation(validation);
    renderReliability(reliability);
    renderCurrentVersionSummary(currentVersionSummary);
    renderDecisionBranchAutomation(decisionBranchSummary);
    renderReviewRuns(runs);
    renderPolicyAudit(policy24, policy7d);
    loadShadowSelectionComparisonSummary(true);
  } catch (e) {
    document.getElementById('scanBanner').textContent = 'Load failed: ' + e.message;
  }
}

async function startTraining() {
  const pw = document.getElementById('adminPassword').value;
  const msg = document.getElementById('trainMessage');
  msg.textContent = 'Starting…';
  try {
    const out = await getJson('/train', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    msg.textContent = out.message || 'Started';
    setTimeout(refreshAll, 500);
  } catch (e) {
    msg.textContent = e.message;
  }
}

document.getElementById('trainButton').addEventListener('click', startTraining);
document.getElementById('runReplayButton').addEventListener('click', runReplay);
document.getElementById('loadReplaySummaryButton').addEventListener('click', loadReplaySummary);
document.getElementById('downloadReplayPackButton').addEventListener('click', downloadReplayPack);
attachSortHandlers();
refreshAll();
setInterval(refreshAll, 10000);

document.getElementById('buildStage1OpportunityButton').addEventListener('click', buildStage1OpportunityFromReplay);
document.getElementById('loadStage1OpportunitySummaryButton').addEventListener('click', loadStage1OpportunitySummary);
document.getElementById('buildModelAuditButton').addEventListener('click', buildModelAuditFromReplay);
document.getElementById('loadModelAuditSummaryButton').addEventListener('click', loadModelAuditSummary);
document.getElementById('runBenchmarkButton').addEventListener('click', runBenchmarkSweep);
document.getElementById('loadBenchmarkSummaryButton').addEventListener('click', loadBenchmarkSummary);
const downloadBenchmarkPackButton = document.getElementById('downloadBenchmarkPackButton');
const downloadClassificationPackButton = document.getElementById('downloadClassificationPackButton');
if (downloadBenchmarkPackButton) downloadBenchmarkPackButton.addEventListener('click', downloadBenchmarkPack);
if (downloadClassificationPackButton) downloadClassificationPackButton.addEventListener('click', downloadClassificationPack);
const runRawScoreBaselineButton = document.getElementById('runRawScoreBaselineButton');
const loadRawScoreBaselineSummaryButton = document.getElementById('loadRawScoreBaselineSummaryButton');
const downloadRawScoreBaselinePackButton = document.getElementById('downloadRawScoreBaselinePackButton');
if (runRawScoreBaselineButton) runRawScoreBaselineButton.addEventListener('click', runRawScoreBaseline);
if (loadRawScoreBaselineSummaryButton) loadRawScoreBaselineSummaryButton.addEventListener('click', function() { loadRawScoreBaselineSummary(false); });
if (downloadRawScoreBaselinePackButton) downloadRawScoreBaselinePackButton.addEventListener('click', downloadRawScoreBaselinePack);
loadRawScoreBaselineSummary(true);
const runUtilityPolicySearchButton = document.getElementById('runUtilityPolicySearchButton');
const loadUtilityPolicySearchStatusButton = document.getElementById('loadUtilityPolicySearchStatusButton');
const loadUtilityPolicySearchSummaryButton = document.getElementById('loadUtilityPolicySearchSummaryButton');
const downloadUtilityPolicySearchPackButton = document.getElementById('downloadUtilityPolicySearchPackButton');
if (runUtilityPolicySearchButton) runUtilityPolicySearchButton.addEventListener('click', runUtilityPolicySearch);
if (loadUtilityPolicySearchStatusButton) loadUtilityPolicySearchStatusButton.addEventListener('click', function() { loadUtilityPolicySearchStatus(false); });
if (loadUtilityPolicySearchSummaryButton) loadUtilityPolicySearchSummaryButton.addEventListener('click', function() { loadUtilityPolicySearchSummary(false); });
if (downloadUtilityPolicySearchPackButton) downloadUtilityPolicySearchPackButton.addEventListener('click', downloadUtilityPolicySearchPack);
loadUtilityPolicySearchStatus(true);

const loadShadowSelectionComparisonSummaryButton = document.getElementById('loadShadowSelectionComparisonSummaryButton');
const downloadShadowSelectionComparisonPackButton = document.getElementById('downloadShadowSelectionComparisonPackButton');
setShadowSelectionComparisonDownloadsEnabled(false);
if (loadShadowSelectionComparisonSummaryButton) loadShadowSelectionComparisonSummaryButton.addEventListener('click', function() { loadShadowSelectionComparisonSummary(false); });
if (downloadShadowSelectionComparisonPackButton) downloadShadowSelectionComparisonPackButton.addEventListener('click', downloadShadowSelectionComparisonPack);
loadShadowSelectionComparisonSummary(true);

const loadShadowSelectionOutcomeReviewButton = document.getElementById('loadShadowSelectionOutcomeReviewButton');
const downloadShadowSelectionOutcomeReviewPackButton = document.getElementById('downloadShadowSelectionOutcomeReviewPackButton');
setShadowSelectionOutcomeReviewDownloadsEnabled(false);
if (loadShadowSelectionOutcomeReviewButton) loadShadowSelectionOutcomeReviewButton.addEventListener('click', function() { loadShadowSelectionOutcomeReview(false); });
if (downloadShadowSelectionOutcomeReviewPackButton) downloadShadowSelectionOutcomeReviewPackButton.addEventListener('click', downloadShadowSelectionOutcomeReviewPack);
loadShadowSelectionOutcomeReview(true);

const runUtilitySelectionLabButton = document.getElementById('runUtilitySelectionLabButton');
const loadUtilitySelectionLabSummaryButton = document.getElementById('loadUtilitySelectionLabSummaryButton');
const downloadUtilitySelectionLabPackButton = document.getElementById('downloadUtilitySelectionLabPackButton');
if (runUtilitySelectionLabButton) runUtilitySelectionLabButton.addEventListener('click', runUtilitySelectionLab);
if (loadUtilitySelectionLabSummaryButton) loadUtilitySelectionLabSummaryButton.addEventListener('click', loadUtilitySelectionLabSummary);
if (downloadUtilitySelectionLabPackButton) downloadUtilitySelectionLabPackButton.addEventListener('click', downloadUtilitySelectionLabPack);
const runUtilityModelLabButton = document.getElementById('runUtilityModelLabButton');
const loadUtilityModelLabSummaryButton = document.getElementById('loadUtilityModelLabSummaryButton');
const downloadUtilityModelLabPackButton = document.getElementById('downloadUtilityModelLabPackButton');
if (runUtilityModelLabButton) runUtilityModelLabButton.addEventListener('click', runUtilityModelLab);
if (loadUtilityModelLabSummaryButton) loadUtilityModelLabSummaryButton.addEventListener('click', loadUtilityModelLabSummary);
if (downloadUtilityModelLabPackButton) downloadUtilityModelLabPackButton.addEventListener('click', downloadUtilityModelLabPack);
const activateUtilityModelProofButton = document.getElementById('activateUtilityModelProofButton');
const clearUtilityModelProofButton = document.getElementById('clearUtilityModelProofButton');
const loadUtilityModelProofSummaryButton = document.getElementById('loadUtilityModelProofSummaryButton');
const downloadUtilityModelProofPackButton = document.getElementById('downloadUtilityModelProofPackButton');
const loadUtilityModelProofReviewSummaryButton = document.getElementById('loadUtilityModelProofReviewSummaryButton');
const downloadUtilityModelProofReviewPackButton = document.getElementById('downloadUtilityModelProofReviewPackButton');
setUtilityModelProofReviewDownloadsEnabled(false);
if (activateUtilityModelProofButton) activateUtilityModelProofButton.addEventListener('click', activateUtilityModelProof);
if (clearUtilityModelProofButton) clearUtilityModelProofButton.addEventListener('click', clearUtilityModelProof);
if (loadUtilityModelProofSummaryButton) loadUtilityModelProofSummaryButton.addEventListener('click', loadUtilityModelProofSummary);
if (downloadUtilityModelProofPackButton) downloadUtilityModelProofPackButton.addEventListener('click', downloadUtilityModelProofPack);
if (loadUtilityModelProofReviewSummaryButton) loadUtilityModelProofReviewSummaryButton.addEventListener('click', loadUtilityModelProofReviewSummary);
if (downloadUtilityModelProofReviewPackButton) downloadUtilityModelProofReviewPackButton.addEventListener('click', downloadUtilityModelProofReviewPack);
loadUtilityModelProofReviewSummary(true);
const activateUtilityModelAdoptionButton = document.getElementById('activateUtilityModelAdoptionButton');
const clearUtilityModelAdoptionButton = document.getElementById('clearUtilityModelAdoptionButton');
const loadUtilityModelAdoptionSummaryButton = document.getElementById('loadUtilityModelAdoptionSummaryButton');
const downloadUtilityModelAdoptionPackButton = document.getElementById('downloadUtilityModelAdoptionPackButton');
if (activateUtilityModelAdoptionButton) activateUtilityModelAdoptionButton.addEventListener('click', activateUtilityModelAdoption);
if (clearUtilityModelAdoptionButton) clearUtilityModelAdoptionButton.addEventListener('click', clearUtilityModelAdoption);
if (loadUtilityModelAdoptionSummaryButton) loadUtilityModelAdoptionSummaryButton.addEventListener('click', loadUtilityModelAdoptionSummary);
if (downloadUtilityModelAdoptionPackButton) downloadUtilityModelAdoptionPackButton.addEventListener('click', downloadUtilityModelAdoptionPack);
const runUtilityTuningLabButton = document.getElementById('runUtilityTuningLabButton');
const loadUtilityTuningLabSummaryButton = document.getElementById('loadUtilityTuningLabSummaryButton');
const downloadUtilityTuningLabPackButton = document.getElementById('downloadUtilityTuningLabPackButton');
if (runUtilityTuningLabButton) runUtilityTuningLabButton.addEventListener('click', runUtilityTuningLab);
if (loadUtilityTuningLabSummaryButton) loadUtilityTuningLabSummaryButton.addEventListener('click', loadUtilityTuningLabSummary);
if (downloadUtilityTuningLabPackButton) downloadUtilityTuningLabPackButton.addEventListener('click', downloadUtilityTuningLabPack);
const activateUtilityTuningProofButton = document.getElementById('activateUtilityTuningProofButton');
const clearUtilityTuningProofButton = document.getElementById('clearUtilityTuningProofButton');
const loadUtilityTuningProofSummaryButton = document.getElementById('loadUtilityTuningProofSummaryButton');
const downloadUtilityTuningProofPackButton = document.getElementById('downloadUtilityTuningProofPackButton');
if (activateUtilityTuningProofButton) activateUtilityTuningProofButton.addEventListener('click', activateUtilityTuningProof);
if (clearUtilityTuningProofButton) clearUtilityTuningProofButton.addEventListener('click', clearUtilityTuningProof);
if (loadUtilityTuningProofSummaryButton) loadUtilityTuningProofSummaryButton.addEventListener('click', loadUtilityTuningProofSummary);
if (downloadUtilityTuningProofPackButton) downloadUtilityTuningProofPackButton.addEventListener('click', downloadUtilityTuningProofPack);
const loadUtilityTuningProofReviewSummaryButton = document.getElementById('loadUtilityTuningProofReviewSummaryButton');
const downloadUtilityTuningProofReviewPackButton = document.getElementById('downloadUtilityTuningProofReviewPackButton');
if (loadUtilityTuningProofReviewSummaryButton) loadUtilityTuningProofReviewSummaryButton.addEventListener('click', loadUtilityTuningProofReviewSummary);
if (downloadUtilityTuningProofReviewPackButton) downloadUtilityTuningProofReviewPackButton.addEventListener('click', downloadUtilityTuningProofReviewPack);
const activateUtilityTuningAdoptionButton = document.getElementById('activateUtilityTuningAdoptionButton');
const clearUtilityTuningAdoptionButton = document.getElementById('clearUtilityTuningAdoptionButton');
const loadUtilityTuningAdoptionSummaryButton = document.getElementById('loadUtilityTuningAdoptionSummaryButton');
const downloadUtilityTuningAdoptionPackButton = document.getElementById('downloadUtilityTuningAdoptionPackButton');
if (activateUtilityTuningAdoptionButton) activateUtilityTuningAdoptionButton.addEventListener('click', activateUtilityTuningAdoption);
if (clearUtilityTuningAdoptionButton) clearUtilityTuningAdoptionButton.addEventListener('click', clearUtilityTuningAdoption);
if (loadUtilityTuningAdoptionSummaryButton) loadUtilityTuningAdoptionSummaryButton.addEventListener('click', loadUtilityTuningAdoptionSummary);
if (downloadUtilityTuningAdoptionPackButton) downloadUtilityTuningAdoptionPackButton.addEventListener('click', downloadUtilityTuningAdoptionPack);
const loadUtilityTuningAdoptionReviewSummaryButton = document.getElementById('loadUtilityTuningAdoptionReviewSummaryButton');
const downloadUtilityTuningAdoptionReviewPackButton = document.getElementById('downloadUtilityTuningAdoptionReviewPackButton');
if (loadUtilityTuningAdoptionReviewSummaryButton) loadUtilityTuningAdoptionReviewSummaryButton.addEventListener('click', loadUtilityTuningAdoptionReviewSummary);
if (downloadUtilityTuningAdoptionReviewPackButton) downloadUtilityTuningAdoptionReviewPackButton.addEventListener('click', downloadUtilityTuningAdoptionReviewPack);

const historicalDecisionLabRunButton = document.getElementById('runHistoricalDecisionLabButton');
const historicalDecisionLabLoadButton = document.getElementById('loadHistoricalDecisionLabSummaryButton');
const historicalDecisionLabDownloadButton = document.getElementById('downloadHistoricalDecisionLabPackButton');
if (historicalDecisionLabRunButton) historicalDecisionLabRunButton.addEventListener('click', runHistoricalDecisionLab);
if (historicalDecisionLabLoadButton) historicalDecisionLabLoadButton.addEventListener('click', loadHistoricalDecisionLabSummary);
if (historicalDecisionLabDownloadButton) historicalDecisionLabDownloadButton.addEventListener('click', downloadHistoricalDecisionLabPack);
const stage1PolicyLabRunButton = document.getElementById('runStage1PolicyLabButton');
const stage1PolicyLabLoadButton = document.getElementById('loadStage1PolicyLabSummaryButton');
const stage1PolicyLabDownloadButton = document.getElementById('downloadStage1PolicyLabPackButton');
if (stage1PolicyLabRunButton) stage1PolicyLabRunButton.addEventListener('click', runStage1PolicyLab);
if (stage1PolicyLabLoadButton) stage1PolicyLabLoadButton.addEventListener('click', loadStage1PolicyLabSummary);
if (stage1PolicyLabDownloadButton) stage1PolicyLabDownloadButton.addEventListener('click', downloadStage1PolicyLabPack);
const nextLiveCandidateLabRunButton = document.getElementById('runNextLiveCandidateLabButton');
const nextLiveCandidateLabLoadButton = document.getElementById('loadNextLiveCandidateLabSummaryButton');
const nextLiveCandidateLabDownloadButton = document.getElementById('downloadNextLiveCandidateLabPackButton');
if (nextLiveCandidateLabRunButton) nextLiveCandidateLabRunButton.addEventListener('click', runNextLiveCandidateLab);
if (nextLiveCandidateLabLoadButton) nextLiveCandidateLabLoadButton.addEventListener('click', loadNextLiveCandidateLabSummary);
if (nextLiveCandidateLabDownloadButton) nextLiveCandidateLabDownloadButton.addEventListener('click', downloadNextLiveCandidateLabPack);
const activateLiveCandidateProofButton = document.getElementById('activateLiveCandidateProofButton');
const clearLiveCandidateProofButton = document.getElementById('clearLiveCandidateProofButton');
const loadLiveCandidateProofSummaryButton = document.getElementById('loadLiveCandidateProofSummaryButton');
const downloadLiveCandidateProofPackButton = document.getElementById('downloadLiveCandidateProofPackButton');
if (activateLiveCandidateProofButton) activateLiveCandidateProofButton.addEventListener('click', activateLiveCandidateProof);
if (clearLiveCandidateProofButton) clearLiveCandidateProofButton.addEventListener('click', clearLiveCandidateProof);
if (loadLiveCandidateProofSummaryButton) loadLiveCandidateProofSummaryButton.addEventListener('click', loadLiveCandidateProofSummary);
if (downloadLiveCandidateProofPackButton) downloadLiveCandidateProofPackButton.addEventListener('click', downloadLiveCandidateProofPack);
const loadLiveCandidateProofReviewSummaryButton = document.getElementById('loadLiveCandidateProofReviewSummaryButton');
const downloadLiveCandidateProofReviewPackButton = document.getElementById('downloadLiveCandidateProofReviewPackButton');
if (loadLiveCandidateProofReviewSummaryButton) loadLiveCandidateProofReviewSummaryButton.addEventListener('click', loadLiveCandidateProofReviewSummary);
if (downloadLiveCandidateProofReviewPackButton) downloadLiveCandidateProofReviewPackButton.addEventListener('click', downloadLiveCandidateProofReviewPack);

const activateLiveCandidateAdoptionButton = document.getElementById('activateLiveCandidateAdoptionButton');
const clearLiveCandidateAdoptionButton = document.getElementById('clearLiveCandidateAdoptionButton');
const loadLiveCandidateAdoptionSummaryButton = document.getElementById('loadLiveCandidateAdoptionSummaryButton');
const downloadLiveCandidateAdoptionPackButton = document.getElementById('downloadLiveCandidateAdoptionPackButton');
if (activateLiveCandidateAdoptionButton) activateLiveCandidateAdoptionButton.addEventListener('click', activateLiveCandidateAdoption);
if (clearLiveCandidateAdoptionButton) clearLiveCandidateAdoptionButton.addEventListener('click', clearLiveCandidateAdoption);
if (loadLiveCandidateAdoptionSummaryButton) loadLiveCandidateAdoptionSummaryButton.addEventListener('click', loadLiveCandidateAdoptionSummary);
if (downloadLiveCandidateAdoptionPackButton) downloadLiveCandidateAdoptionPackButton.addEventListener('click', downloadLiveCandidateAdoptionPack);
const loadLiveCandidateAdoptionReviewSummaryButton = document.getElementById('loadLiveCandidateAdoptionReviewSummaryButton');
const downloadLiveCandidateAdoptionReviewPackButton = document.getElementById('downloadLiveCandidateAdoptionReviewPackButton');
if (loadLiveCandidateAdoptionReviewSummaryButton) loadLiveCandidateAdoptionReviewSummaryButton.addEventListener('click', loadLiveCandidateAdoptionReviewSummary);
if (downloadLiveCandidateAdoptionReviewPackButton) downloadLiveCandidateAdoptionReviewPackButton.addEventListener('click', downloadLiveCandidateAdoptionReviewPack);
const startUtilityOperatorAutomationButton = document.getElementById('startUtilityOperatorAutomationButton');
const stopUtilityOperatorAutomationButton = document.getElementById('stopUtilityOperatorAutomationButton');
const loadUtilityOperatorAutomationStatusButton = document.getElementById('loadUtilityOperatorAutomationStatusButton');
const downloadUtilityOperatorAutomationPackButton = document.getElementById('downloadUtilityOperatorAutomationPackButton');
if (startUtilityOperatorAutomationButton) startUtilityOperatorAutomationButton.addEventListener('click', startUtilityOperatorAutomation);
if (stopUtilityOperatorAutomationButton) stopUtilityOperatorAutomationButton.addEventListener('click', stopUtilityOperatorAutomation);
if (loadUtilityOperatorAutomationStatusButton) loadUtilityOperatorAutomationStatusButton.addEventListener('click', loadUtilityOperatorAutomationStatus);
if (downloadUtilityOperatorAutomationPackButton) downloadUtilityOperatorAutomationPackButton.addEventListener('click', downloadUtilityOperatorAutomationPack);
const freshRetrainAuditRunButton = document.getElementById('runFreshRetrainAuditButton');
const freshRetrainAuditStopButton = document.getElementById('stopFreshRetrainAuditButton');
const freshRetrainAuditLoadButton = document.getElementById('loadFreshRetrainAuditSummaryButton');
const freshRetrainAuditDownloadButton = document.getElementById('downloadFreshRetrainAuditPackButton');
if (freshRetrainAuditRunButton) freshRetrainAuditRunButton.addEventListener('click', runFreshRetrainAudit);
if (freshRetrainAuditStopButton) freshRetrainAuditStopButton.addEventListener('click', stopFreshRetrainAudit);
if (freshRetrainAuditLoadButton) freshRetrainAuditLoadButton.addEventListener('click', loadFreshRetrainAuditSummary);
if (freshRetrainAuditDownloadButton) freshRetrainAuditDownloadButton.addEventListener('click', downloadFreshRetrainAuditPack);
const challengerComparisonRunButton = document.getElementById('runChallengerComparisonButton');
const challengerComparisonStopButton = document.getElementById('stopChallengerComparisonButton');
const challengerComparisonLoadButton = document.getElementById('loadChallengerComparisonSummaryButton');
const challengerComparisonDownloadButton = document.getElementById('downloadChallengerComparisonPackButton');
if (challengerComparisonRunButton) challengerComparisonRunButton.addEventListener('click', runChallengerComparison);
if (challengerComparisonStopButton) challengerComparisonStopButton.addEventListener('click', stopChallengerComparison);
if (challengerComparisonLoadButton) challengerComparisonLoadButton.addEventListener('click', loadChallengerComparisonSummary);
if (challengerComparisonDownloadButton) challengerComparisonDownloadButton.addEventListener('click', downloadChallengerComparisonPack);
const decisionBranchToggleButton = document.getElementById('toggleDecisionBranchAutoExecuteButton');
const decisionBranchExecuteButton = document.getElementById('executeDecisionBranchButton');
const decisionBranchClearButton = document.getElementById('clearDecisionBranchOverrideButton');
const decisionBranchAckButton = document.getElementById('ackDecisionBranchButton');
if (decisionBranchToggleButton) decisionBranchToggleButton.addEventListener('click', toggleDecisionBranchAutoExecute);
if (decisionBranchExecuteButton) decisionBranchExecuteButton.addEventListener('click', executeDecisionBranchNow);
if (decisionBranchClearButton) decisionBranchClearButton.addEventListener('click', clearDecisionBranchOverride);
if (decisionBranchAckButton) decisionBranchAckButton.addEventListener('click', acknowledgeDecisionBranch);
const runSemanticsComparisonButton = document.getElementById('runSemanticsComparisonButton');
const loadSemanticsComparisonSummaryButton = document.getElementById('loadSemanticsComparisonSummaryButton');
const downloadSemanticsComparisonPackButton = document.getElementById('downloadSemanticsComparisonPackButton');
if (runSemanticsComparisonButton) runSemanticsComparisonButton.addEventListener('click', runSemanticsComparison);
if (loadSemanticsComparisonSummaryButton) loadSemanticsComparisonSummaryButton.addEventListener('click', function() { loadSemanticsComparisonSummary(false); });
if (downloadSemanticsComparisonPackButton) downloadSemanticsComparisonPackButton.addEventListener('click', downloadSemanticsComparisonPack);
loadSemanticsComparisonSummary(true);
