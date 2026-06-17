let currentRun = null;
let currentSelection = null;

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
  return value.toFixed(2);
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
        <div class="value">${fmtLeakage(summary.mean_leakage[scaffold])}</div>
      </div>`
    )
    .join("");

  const rows = (summary.pair_ids || [])
    .map((pairId) => {
      const cells = scaffolds
        .map((scaffold) => {
          const xAcc = summary.x_accuracy_by_pair?.[pairId]?.[scaffold];
          const leak = summary.leakage_by_pair?.[pairId]?.[scaffold];
          const leakClass =
            leak != null && leak > 0.2 ? "leakage-high" : "leakage-low";
          return `<td>${fmtPct(xAcc)}</td><td class="${leakClass}">${fmtLeakage(leak)}</td>`;
        })
        .join("");
      return `<tr><td>${pairId}</td>${cells}</tr>`;
    })
    .join("");

  const headerCells = scaffolds
    .map((s) => `<th>${s} X</th><th>${s} leak</th>`)
    .join("");

  summaryPanel.innerHTML = `
    <h2 style="margin:0 0 0.75rem;font-size:1rem;">Run ${summary.run_id}</h2>
    <div class="summary-grid">${statCards}</div>
    <div class="leakage-table-wrap">
      <table class="compact">
        <thead><tr><th>Pair</th>${headerCells}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
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
  currentSelection = { pairId, scaffold, trial };
  setActiveButton(btn);
  emptyState.classList.add("hidden");
  detailPanel.classList.remove("hidden");

  const data = await fetchJson(
    `/api/runs/${currentRun}/result/${pairId}/${scaffold}/${trial}`
  );
  renderDetail(data, pairId, scaffold, trial);
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
  stepsEl.innerHTML = result.trace
    .map((step, index) => renderStep(step, index))
    .join("");

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

  const rows = y_trials
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
  const runs = await fetchJson("/api/runs");
  const run = runs.find((r) => r.run_id === runId);
  renderSummary(run || { summary: null });
  const tree = await fetchJson(`/api/runs/${runId}/tree`);
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
