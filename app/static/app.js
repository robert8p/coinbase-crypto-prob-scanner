const fmt = (x, digits=8) => {
  if (x === null || x === undefined) return "—";
  if (typeof x !== "number" || !isFinite(x)) return "—";
  const d = Math.abs(x) < 1 ? 8 : 4;
  return x.toFixed(Math.min(digits, d));
};

const pct = (x) => {
  if (x === null || x === undefined) return "—";
  if (typeof x !== "number" || !isFinite(x)) return "—";
  return (x * 100).toFixed(1) + "%";
};

let rows = [];
let sortKey = "prob_2";
let sortDir = "desc";

function setText(id, txt) { const el = document.getElementById(id); if (el) el.textContent = txt; }

function buildQuoteOptions(rows) {
  const q = new Set();
  rows.forEach(r => { if (r.quote) q.add(r.quote); });
  const sel = document.getElementById("quoteFilter");
  const current = sel.value;
  const opts = ["", ...Array.from(q).sort()];
  sel.innerHTML = opts.map(v => `<option value="${v}">${v==="" ? "Any" : v}</option>`).join("");
  sel.value = current || "";
}

function passesFilters(r) {
  const minProb = parseFloat(document.getElementById("minProb").value || "0");
  const pMin = parseFloat(document.getElementById("priceMin").value || "0");
  const pMaxRaw = document.getElementById("priceMax").value;
  const pMax = pMaxRaw === "" ? Infinity : parseFloat(pMaxRaw);
  const quote = document.getElementById("quoteFilter").value;
  const risk = document.getElementById("riskFilter").value;
  const showStale = document.getElementById("showStale").checked;

  if (!showStale && r.included === false) return false;
  if (quote && r.quote !== quote) return false;
  if (risk && r.risk !== risk) return false;

  const price = (typeof r.price === "number") ? r.price : 0;
  if (price < pMin) return false;
  if (price > pMax) return false;

  const p2 = (typeof r.prob_2 === "number") ? r.prob_2 : -1;
  return p2 >= minProb;
}

function renderTable() {
  const tbody = document.querySelector("#scoresTable tbody");
  const filtered = rows.filter(passesFilters);
  filtered.sort((a, b) => {
    const na = (typeof a[sortKey] === "number") ? a[sortKey] : -Infinity;
    const nb = (typeof b[sortKey] === "number") ? b[sortKey] : -Infinity;
    if (na === nb) return (a.display_symbol || "").localeCompare(b.display_symbol || "");
    return sortDir === "asc" ? (na - nb) : (nb - na);
  });

  tbody.innerHTML = filtered.map(r => {
    const badge = r.included === false ? `<span class="badge badge-warn">SKIPPED</span>` : "";
    return `
      <tr class="${r.included===false ? "row-muted" : ""}">
        <td><div class="prod">${r.display_symbol || r.product}${badge}</div><div class="small dim">${r.prob_2_source || ""}</div></td>
        <td class="num">${fmt(r.price)}</td>
        <td class="num">${fmt(r.vwap)}</td>
        <td>${r.risk || "—"} <div class="small dim">${r.risk_reasons || ""}</div></td>
        <td class="num">${r.prob_1 === null ? "—" : pct(r.prob_1)}</td>
        <td class="num">${r.prob_2 === null ? "—" : pct(r.prob_2)}</td>
        <td>${r.quote || "—"}</td>
        <td>${r.category || "—"}</td>
        <td class="small">${r.reasons || r.skip_reason || ""}</td>
        <td class="small">${r.last_candle_time ? r.last_candle_time.replace("T"," ").replace("+00:00","Z") : "—"}</td>
      </tr>`;
  }).join("");

  setText("tableMeta", `Showing ${filtered.length} rows (of ${rows.length}) • Sorted by ${sortKey} ${sortDir}`);
}

async function fetchStatus() {
  const r = await fetch("/api/status");
  const s = await r.json();
  const cfg = s.config || {};
  setText("horizonLabel", `${cfg.horizon_minutes || "—"} min (${cfg.horizon_mode || "—"})`);
  setText("modePill", cfg.demo_mode ? "DEMO_MODE" : "LIVE");
  document.getElementById("modePill").className = "pill " + (cfg.demo_mode ? "pill-warn" : "pill-ok");

  const cb = s.coinbase || {};
  setText("coinbaseStatus", `${cb.message || "—"}${cb.last_error ? " • " + cb.last_error : ""}`);

  const rl = cb.rate_limit || {};
  setText("rateLimitStatus", `429: ${rl.last_429_utc || "—"} • backoffs: ${rl.backoff_count || 0} • last: ${rl.last_backoff_seconds || "—"}s`);

  const cov = s.coverage || {};
  setText("lastRun", cov.last_run_utc || "—");
  setText("lastCandle", cov.last_candle_timestamp || "—");
  setText("coverageLine", `Scored ${cov.products_scored_count || 0} / Universe ${cov.universe_count || 0}`);

  const model = s.model || {};
  setText("modelStatus", `${model.using || "—"} • pt1:${model.pt1?.reason || "—"} • pt2:${model.pt2?.reason || "—"}`);

  const skip = cov.top_skip_reasons || {};
  const keys = Object.keys(skip);
  setText("skipReasons", keys.length ? "Skip reasons: " + keys.slice(0, 10).map(k => `${k}(${skip[k]})`).join(" • ") : "Skip reasons: —");

  const scan = s.scan || {};
  const pause = scan.paused_for_training ? " (paused for training)" : "";
  setText("lastError", s.last_error ? `Last error: ${s.last_error}` : (scan.running ? "Scan running..." : `OK${pause}`));

  const t = s.training || {};
  if (t.running) setText("trainStatus", `TRAINING... ${t.progress || ""}`);
  else if (t.last_error) setText("trainStatus", `Training error: ${t.last_error}`);
  else if (t.last_result) setText("trainStatus", `Training done • pt1 brier: ${t.last_result.pt1_metrics?.brier_val ?? "—"} • pt2 brier: ${t.last_result.pt2_metrics?.brier_val ?? "—"}`);
  else setText("trainStatus", "Not training (heuristic fallback used until trained).");
}

async function fetchScores() {
  const r = await fetch("/api/scores");
  const s = await r.json();
  rows = s.rows || [];
  buildQuoteOptions(rows);
  renderTable();
}

async function startTraining() {
  const pw = document.getElementById("adminPassword").value || "";
  const r = await fetch("/train", { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({password: pw}) });
  if (!r.ok) {
    const e = await r.json().catch(()=>({detail:"Unknown error"}));
    alert(`Training failed: ${e.detail || r.statusText}`);
    return;
  }
  setText("trainStatus", "Training started...");
}

function wire() {
  document.getElementById("refreshBtn").addEventListener("click", async () => { await fetchStatus(); await fetchScores(); });
  ["minProb","priceMin","priceMax","quoteFilter","riskFilter","showStale"].forEach(id => {
    document.getElementById(id).addEventListener("input", renderTable);
    document.getElementById(id).addEventListener("change", renderTable);
  });
  document.getElementById("trainBtn").addEventListener("click", startTraining);
  document.querySelectorAll("#scoresTable th").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.k;
      if (!k) return;
      if (sortKey === k) sortDir = (sortDir === "desc" ? "asc" : "desc");
      else { sortKey = k; sortDir = (k.startsWith("prob") ? "desc" : "asc"); }
      renderTable();
    });
  });
}

wire();
(async () => {
  await fetchStatus();
  await fetchScores();
  setInterval(fetchStatus, 10000);
  setInterval(fetchScores, 30000);
})();
