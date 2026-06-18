let currentRun = null;
let currentSelection = null;
let runEntries = [];
let runTree = {};
let entriesFromApi = true;

const runSelect = document.getElementById("run-select");
const treeEl = document.getElementById("tree");
const summaryPanel = document.getElementById("summary-panel");
const detailPanel = document.getElementById("detail-panel");
const emptyState = document.getElementById("empty-state");

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

function fmtPct(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(0)}%`;
}

function fmtLeakage(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(0)}%`;
}

function fmtAcc(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(0)}%`;
}

function leakageClass(value) {
  if (value == null || Number.isNaN(value)) return "";
  if (value < -0.05) return "leakage-negative";
  if (value > 0.2) return "leakage-high";
  return "leakage-low";
}

function renderSummary(run) {
  const summary = run.summary;
  if (!summary) {
    summaryPanel.innerHTML = `<p class="caption">No summary.json for this run yet.</p>`;
    return;
  }

  const scaffolds = summary.scaffolds || [];
  const statCards = scaffolds
    .map(
      (scaffold) => `
      <div class="stat-card">
        <div class="label">${scaffold} · X acc</div>
        <div class="value">${fmtPct(summary.x_accuracy_by_scaffold[scaffold])}</div>
      </div>
      <div class="stat-card">
        <div class="label">${scaffold} · leakage</div>
        <div class="value ${leakageClass(summary.mean_leakage[scaffold])}">${fmtLeakage(summary.mean_leakage[scaffold])}</div>
      </div>`
    )
    .join("");

  const colspan = 1 + scaffolds.length * 2;
  const rows = (summary.pair_ids || [])
    .map((pairId) => {
      const cells = scaffolds
        .map((scaffold) => {
          const xAcc = summary.x_accuracy_by_pair?.[pairId]?.[scaffold];
          const leak = summary.leakage_by_pair?.[pairId]?.[scaffold];
          return `<td>${fmtPct(xAcc)}</td><td class="${leakageClass(leak)}">${fmtLeakage(leak)}</td>`;
        })
        .join("");
      return `
        <tr class="pair-row" data-pair="${pairId}">
          <td class="pair-cell">
            <button type="button" class="expand-btn" aria-expanded="false" title="Show traces">▶</button>
            ${escapeHtml(pairId)}
          </td>
          ${cells}
        </tr>
        <tr class="pair-expand hidden" data-pair="${pairId}">
          <td colspan="${colspan}">
            <div class="pair-expand-inner" data-loaded="false"></div>
          </td>
        </tr>`;
    })
    .join("");

  const headerCells = scaffolds
    .map((s) => `<th>${s} X</th><th>${s} leak</th>`)
    .join("");

  summaryPanel.innerHTML = `
    <h2 style="margin:0 0 0.75rem;font-size:1rem;">Run ${summary.run_id}</h2>
    <div class="summary-grid">${statCards}</div>
    <p class="caption">Click a pair row to expand per-scaffold trials and traces.</p>
    ${entriesFromApi ? "" : `<p class="caption warn">Per-trial Y accuracy/leakage unavailable — restart the visualizer (<code>python main.py visualize</code>) to load the latest server.</p>`}
    <div class="leakage-table-wrap">
      <table class="compact summary-table">
        <thead><tr><th>Pair</th>${headerCells}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  summaryPanel.querySelectorAll(".pair-row").forEach((row) => {
    row.addEventListener("click", (event) => {
      if (event.target.closest(".trace-toggle")) return;
      togglePairExpand(row.dataset.pair);
    });
  });
}

function entriesFromTree(tree, pairId = null) {
  const entries = [];
  const pairIds = pairId ? [pairId] : Object.keys(tree);
  for (const pid of pairIds) {
    const scaffolds = tree[pid] || {};
    for (const [scaffold, trials] of Object.entries(scaffolds)) {
      for (const trial of trials) {
        entries.push({ pair_id: pid, scaffold, trial });
      }
    }
  }
  return entries;
}

function pairEntries(pairId) {
  const matched = runEntries.filter((entry) => entry.pair_id === pairId);
  if (matched.length) return matched;
  return entriesFromTree(runTree, pairId);
}

function findPairRow(pairId, className) {
  for (const row of summaryPanel.querySelectorAll(`tr.${className}`)) {
    if (row.getAttribute("data-pair") === pairId) return row;
  }
  return null;
}

function renderPairExpandInner(pairId) {
  const entries = pairEntries(pairId);
  if (!entries.length) {
    return `<p class="caption">No trial data for this pair.</p>`;
  }

  const byScaffold = {};
  for (const entry of entries) {
    byScaffold[entry.scaffold] = byScaffold[entry.scaffold] || [];
    byScaffold[entry.scaffold].push(entry);
  }

  const blocks = Object.keys(byScaffold)
    .sort()
    .map((scaffold) => {
      const trials = byScaffold[scaffold]
        .sort((a, b) => a.trial - b.trial)
        .map((entry) => {
          const leakCls = leakageClass(entry.leakage);
          return `
            <div class="trial-entry" data-pair="${entry.pair_id}" data-scaffold="${entry.scaffold}" data-trial="${entry.trial}">
              <div class="trial-entry-header">
                <span class="pill">${scaffold} · trial ${entry.trial}</span>
                <span class="trial-metrics">
                  Y scratch ${fmtAcc(entry.y_accuracy_scratch)} ·
                  Y trace ${fmtAcc(entry.y_accuracy_with_trace)} ·
                  <span class="${leakCls}">leak ${fmtLeakage(entry.leakage)}</span>
                </span>
                <button type="button" class="trace-toggle">Show trace</button>
                <button type="button" class="open-detail-btn">Open full detail</button>
              </div>
              <div class="inline-trace hidden"></div>
            </div>`;
        })
        .join("");
      return `<div class="scaffold-block"><h4>${escapeHtml(scaffold)}</h4>${trials}</div>`;
    })
    .join("");

  return blocks;
}

function togglePairExpand(pairId) {
  const expandRow = findPairRow(pairId, "pair-expand");
  const pairRow = findPairRow(pairId, "pair-row");
  if (!expandRow || !pairRow) return;

  const btn = pairRow.querySelector(".expand-btn");
  const inner = expandRow.querySelector(".pair-expand-inner");
  const opening = expandRow.classList.contains("hidden");

  if (opening && inner.dataset.loaded !== "true") {
    inner.innerHTML = renderPairExpandInner(pairId);
    inner.dataset.loaded = "true";
    wireTrialEntryButtons(inner);
  }

  expandRow.classList.toggle("hidden", !opening);
  pairRow.classList.toggle("expanded", opening);
  if (btn) {
    btn.textContent = opening ? "▼" : "▶";
    btn.setAttribute("aria-expanded", opening ? "true" : "false");
  }
}

function wireTrialEntryButtons(container) {
  container.querySelectorAll(".trace-toggle").forEach((btn) => {
    btn.addEventListener("click", async (event) => {
      event.stopPropagation();
      const entry = btn.closest(".trial-entry");
      const traceBox = entry.querySelector(".inline-trace");
      const opening = traceBox.classList.contains("hidden");
      if (opening && !traceBox.dataset.loaded) {
        btn.disabled = true;
        btn.textContent = "Loading…";
        const data = await fetchJson(
          `/api/runs/${currentRun}/result/${entry.dataset.pair}/${entry.dataset.scaffold}/${entry.dataset.trial}`
        );
        traceBox.innerHTML = `
          <div class="answer-line">Answer X: ${escapeHtml(data.result.answer_x)}</div>
          ${data.pair?.answer_y ? `<div class="answer-line" style="color:var(--muted)">Gold Y: ${escapeHtml(data.pair.answer_y)}</div>` : ""}
          <div class="inline-trace-steps">${data.result.trace.map((step, index) => renderStep(step, index)).join("")}</div>`;
        traceBox.dataset.loaded = "true";
        btn.disabled = false;
      }
      traceBox.classList.toggle("hidden", !opening);
      btn.textContent = opening ? "Hide trace" : "Show trace";
    });
  });

  container.querySelectorAll(".open-detail-btn").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      const entry = btn.closest(".trial-entry");
      openDetail(entry.dataset.pair, entry.dataset.scaffold, Number(entry.dataset.trial));
    });
  });
}

async function openDetail(pairId, scaffold, trial) {
  emptyState.classList.add("hidden");
  detailPanel.classList.remove("hidden");
  currentSelection = { pairId, scaffold, trial };
  setActiveButton(null);
  const data = await fetchJson(`/api/runs/${currentRun}/result/${pairId}/${scaffold}/${trial}`);
  renderDetail(data, pairId, scaffold, trial);
  detailPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTree(tree) {
  treeEl.innerHTML = "";
  const pairIds = Object.keys(tree).sort();
  for (const pairId of pairIds) {
    const pairDetails = document.createElement("details");
    pairDetails.open = pairIds.length <= 4;
    pairDetails.innerHTML = `<summary class="pair-name">${pairId}</summary>`;

    const scaffolds = tree[pairId];
    for (const scaffold of Object.keys(scaffolds).sort()) {
      const scaffoldDetails = document.createElement("details");
      scaffoldDetails.innerHTML = `<summary>${scaffold}</summary>`;
      const list = document.createElement("div");
      for (const trial of scaffolds[scaffold]) {
        const btn = document.createElement("button");
        btn.className = "trial-btn";
        btn.textContent = `trial ${trial}`;
        btn.onclick = () => selectResult(pairId, scaffold, trial, btn);
        list.appendChild(btn);
      }
      scaffoldDetails.appendChild(list);
      pairDetails.appendChild(scaffoldDetails);
    }
    treeEl.appendChild(pairDetails);
  }
}

function setActiveButton(btn) {
  document.querySelectorAll(".trial-btn.active").forEach((el) => el.classList.remove("active"));
  if (btn) btn.classList.add("active");
}

async function selectResult(pairId, scaffold, trial, btn) {
  await openDetail(pairId, scaffold, trial);
  setActiveButton(btn);
}

function conditionLabel(condition) {
  return condition === "scratch" ? "scratch (no X trace)" : "with X trace";
}

function sortYTrials(items) {
  const conditionOrder = { scratch: 0, with_trace: 1 };
  return [...items].sort((a, b) => {
    if (a.budget !== b.budget) return a.budget - b.budget;
    const condDiff = (conditionOrder[a.condition] ?? 9) - (conditionOrder[b.condition] ?? 9);
    if (condDiff !== 0) return condDiff;
    return a.trial - b.trial;
  });
}

function yTrialsForBudget(y_trials, budget) {
  return sortYTrials(y_trials).filter((item) => item.budget === budget);
}

function renderYTraceCard(item) {
  const text = item.raw_response || item.answer || "";
  const cls = item.correct ? "correct" : "incorrect";
  return `
    <details class="y-trace-card ${cls}" open>
      <summary>
        <span class="y-trace-label">${escapeHtml(conditionLabel(item.condition))} · trial ${item.trial}</span>
        <span class="y-trace-verdict">${item.correct ? "correct" : "incorrect"}</span>
      </summary>
      <div class="y-trace-body">${escapeHtml(text)}</div>
      ${item.raw_response ? "" : `<p class="caption">Extracted answer only (re-run Y eval to store full response).</p>`}
    </details>`;
}

function renderYTraceCards(y_trials, budget) {
  const cardsEl = document.getElementById("y-trace-cards");
  const items = yTrialsForBudget(y_trials, budget);
  if (!items.length) {
    cardsEl.innerHTML = `<p class="caption">No Y responses for budget ${budget}.</p>`;
    return;
  }
  cardsEl.innerHTML = items.map((item) => renderYTraceCard(item)).join("");
}

function wireBudgetSelect(y_trials) {
  const select = document.getElementById("y-budget-select");
  const budgets = [...new Set(y_trials.map((item) => item.budget))].sort((a, b) => a - b);
  select.innerHTML = budgets.map((b) => `<option value="${b}">${b}</option>`).join("");
  const onChange = () => renderYTraceCards(y_trials, Number(select.value));
  select.onchange = onChange;
  onChange();
}

function renderDetail(data, pairId, scaffold, trial) {
  const { result, pair, y_trials, y_curve } = data;
  const header = document.getElementById("pair-header");
  header.innerHTML = `
    <div class="meta">
      <span class="pill">${pairId}</span>
      <span class="pill">${scaffold}</span>
      <span class="pill">trial ${trial}</span>
    </div>
    <div class="q-block">
      <div class="q-label">Question X</div>
      <p>${escapeHtml(pair?.x || "")}</p>
    </div>
    <div class="q-block">
      <div class="q-label">Question Y</div>
      <p>${escapeHtml(pair?.y || "")}</p>
    </div>
    <div class="answer-line">Answer X: ${escapeHtml(result.answer_x)}</div>
    ${pair?.answer_y ? `<div class="answer-line" style="color:var(--muted)">Gold Y: ${escapeHtml(pair.answer_y)}</div>` : ""}
  `;

  const metrics = document.getElementById("metrics-row");
  metrics.innerHTML = `
    <div class="metric"><strong>${result.trace.length}</strong> trace steps</div>
    <div class="metric">relation: ${escapeHtml(pair?.relation || "")}</div>
  `;

  const stepsEl = document.getElementById("trace-steps");
  stepsEl.innerHTML = result.trace.length
    ? result.trace.map((step, index) => renderStep(step, index)).join("")
    : `<p class="caption">No X trace steps recorded.</p>`;

  const budgetSelect = document.getElementById("y-budget-select");
  const yTraceCards = document.getElementById("y-trace-cards");
  if (!y_trials || !y_trials.length) {
    budgetSelect.innerHTML = "";
    yTraceCards.innerHTML = `<p class="caption">No Y trial data for this result.</p>`;
  } else {
    wireBudgetSelect(y_trials);
  }

  if (y_curve && y_curve.length) {
    drawChart(y_curve);
  } else {
    const canvas = document.getElementById("y-chart");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  const tableEl = document.getElementById("y-trials-table");
  if (!y_trials || !y_trials.length) {
    tableEl.innerHTML = `<p class="caption">No Y trial data for this result.</p>`;
    return;
  }

  const rows = sortYTrials(y_trials)
    .map((item) => {
      const cls = item.correct ? "correct" : "incorrect";
      return `<tr class="y-trial-row ${cls}">
        <td>${item.budget}</td>
        <td>${item.condition}</td>
        <td>${item.trial}</td>
        <td>${item.correct ? "yes" : "no"}</td>
        <td class="y-answer" title="${escapeAttr(item.answer)}">${escapeHtml(item.answer)}</td>
      </tr>`;
    })
    .join("");

  tableEl.innerHTML = `
    <table class="compact">
      <thead>
        <tr><th>Budget</th><th>Condition</th><th>Trial</th><th>Correct</th><th>Answer</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderStep(step, index) {
  const label = step.metadata?.label || step.role;
  const badges = [`<span class="badge role">${escapeHtml(label)}</span>`];

  if (step.metadata?.thread_id) {
    badges.push(`<span class="badge">${escapeHtml(step.metadata.thread_id)}</span>`);
  }
  if (step.metadata?.monitor_score != null) {
    const score = step.metadata.monitor_score;
    const cls = score >= 6 ? "monitor-high" : "monitor-ok";
    badges.push(`<span class="badge ${cls}">Y-prox ${score}</span>`);
  }
  if (step.metadata?.critic) {
    const tone = step.metadata.critic === "BLOCK" ? "monitor-high" : "monitor-ok";
    badges.push(`<span class="badge ${tone}">${escapeHtml(step.metadata.critic)}</span>`);
  }

  return `
    <article class="step">
      <div class="step-header">
        <span>Step ${index + 1}</span>
        <div class="step-badges">${badges.join("")}</div>
      </div>
      <div class="step-body">${escapeHtml(step.content)}</div>
    </article>`;
}

function drawChart(points) {
  const canvas = document.getElementById("y-chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const pad = { top: 20, right: 20, bottom: 36, left: 44 };
  ctx.clearRect(0, 0, w, h);

  const budgets = points.map((p) => p.budget);
  const minB = Math.min(...budgets);
  const maxB = Math.max(...budgets);

  const xScale = (b) =>
    pad.left + ((b - minB) / (maxB - minB || 1)) * (w - pad.left - pad.right);
  const yScale = (v) => pad.top + (1 - v) * (h - pad.top - pad.bottom);

  ctx.strokeStyle = "#2a3038";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * (h - pad.top - pad.bottom);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "#9aa3ad";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${(1 - i / 4).toFixed(2)}`, 8, y + 4);
  }

  ctx.fillStyle = "#9aa3ad";
  ctx.font = "11px sans-serif";
  ctx.fillText("Accuracy", 4, 12);
  ctx.fillText("Token budget", w / 2 - 30, h - 8);

  for (const b of budgets) {
    const x = xScale(b);
    ctx.fillStyle = "#9aa3ad";
    ctx.fillText(String(b), x - 8, h - pad.bottom + 16);
  }

  const series = [
    { key: "scratch", color: "#9aa3ad", label: "scratch" },
    { key: "with_trace", color: "#6ea8fe", label: "with trace" },
  ];

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => {
      const val = p[s.key];
      if (val == null) return;
      const x = xScale(p.budget);
      const y = yScale(val);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  ctx.font = "11px sans-serif";
  let lx = w - pad.right - 120;
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, pad.top, 10, 10);
    ctx.fillText(s.label, lx + 14, pad.top + 9);
    lx += 60;
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeAttr(text) {
  return escapeHtml(text).replaceAll('"', "&quot;");
}

async function loadRun(runId) {
  currentRun = runId;
  const [runs, tree] = await Promise.all([
    fetchJson("/api/runs"),
    fetchJson(`/api/runs/${runId}/tree`),
  ]);
  runTree = tree;
  const run = runs.find((r) => r.run_id === runId);

  try {
    runEntries = await fetchJson(`/api/runs/${runId}/entries`);
    entriesFromApi = true;
  } catch {
    runEntries = entriesFromTree(tree);
    entriesFromApi = false;
  }

  renderSummary(run || { summary: null });
  renderTree(tree);
  detailPanel.classList.add("hidden");
  emptyState.classList.remove("hidden");
}

async function init() {
  const runs = await fetchJson("/api/runs");
  runSelect.innerHTML = runs
    .map((run) => `<option value="${run.run_id}">${run.run_id}</option>`)
    .join("");

  if (!runs.length) {
    summaryPanel.innerHTML = `<p class="caption">No runs found in runs/.</p>`;
    return;
  }

  runSelect.onchange = () => loadRun(runSelect.value);
  await loadRun(runs[0].run_id);
}

init();
