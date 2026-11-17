/* InsurAutoML dashboard: polls the local JSON API and renders the page. */
"use strict";

const POLL_STATE_MS = 1500;
const POLL_MAP_MS = 4000;
// no new event for this long while trials should be running: run interrupted?
const STALE_S = 180;

const STEP_LABELS = {
  encoder: "Encoder",
  imputer: "Imputer",
  balancing: "Balancing",
  scaling: "Scaling",
  feature_selection: "Feature selection",
  model: "Model",
  complete_prep: "Complete prep",
  missing_prep: "Missing prep",
};

// warm = better (score 1: best loss rank), cold = worse
const SCALE = [
  [0, "#2c7bb6"],
  [0.25, "#abd9e9"],
  [0.5, "#ffffbf"],
  [0.75, "#fdae61"],
  [1, "#d7191c"],
];

const view = {
  data: null,
  runId: null,
  emphasis: "model",
  layout: null,
  layoutEmphasis: null,
  scores: null,
  sortKey: "number",
  sortAsc: true,
  signature: null,
  tableSignature: null,
  selection: null,
  methodStep: null,
};

/* ------------------------------------------------------------- helpers */

const $ = (id) => document.getElementById(id);
const hasPlotly = () => typeof window.Plotly !== "undefined";

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "–";
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 60) return seconds + "s";
  const m = Math.floor(seconds / 60), s = seconds % 60;
  if (m < 60) return m + "m " + String(s).padStart(2, "0") + "s";
  const h = Math.floor(m / 60);
  return h + "h " + String(m % 60).padStart(2, "0") + "m";
}

function fmtLoss(value) {
  if (value === null || value === undefined) return "–";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(3);
  return value.toFixed(4);
}

function fmtValue(value) {
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(2);
    return String(Number(value.toPrecision(3)));
  }
  return String(value);
}

function stepLabel(step) {
  return STEP_LABELS[step] || step.replace(/_/g, " ");
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function scaleColor(score) {
  if (score === null || score === undefined) return cssVar("--idle");
  for (let i = 1; i < SCALE.length; i++) {
    if (score <= SCALE[i][0]) {
      const [a, ca] = SCALE[i - 1], [b, cb] = SCALE[i];
      const t = (score - a) / (b - a);
      const mix = (k) => Math.round(
        parseInt(ca.substr(k, 2), 16) * (1 - t) + parseInt(cb.substr(k, 2), 16) * t
      );
      return "rgb(" + mix(1) + "," + mix(3) + "," + mix(5) + ")";
    }
  }
  return SCALE[SCALE.length - 1][1];
}

function plotLayout(extra) {
  const text = cssVar("--text"), border = cssVar("--border");
  const axis = { gridcolor: border, zerolinecolor: border, linecolor: border, automargin: true };
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: text, size: 12, family: "system-ui, sans-serif" },
    margin: { l: 52, r: 12, t: 8, b: 40 },
    xaxis: Object.assign({}, axis),
    yaxis: Object.assign({}, axis),
    hoverlabel: { font: { size: 12 } },
    showlegend: false,
  }, extra || {});
}

const PLOT_CONFIG = { displaylogo: false, responsive: true };

function steps() {
  const data = view.data;
  if (data && data.map_steps && data.map_steps.length) return data.map_steps;
  const found = [];
  for (const trial of (data ? data.trials : [])) {
    for (const step of Object.keys(trial.config || {})) {
      if (!found.includes(step)) found.push(step);
    }
  }
  return found;
}

function modelOf(trial) {
  const model = (trial.config || {}).model;
  return model && model.method ? model.method : "–";
}

function paramsText(hyperparameters) {
  return Object.entries(hyperparameters || {})
    .map(([key, value]) => key + "=" + fmtValue(value))
    .join(", ");
}

function pipelineText(trial) {
  return steps()
    .filter((step) => (trial.config || {})[step])
    .map((step) => {
      const setting = trial.config[step];
      const params = paramsText(setting.hyperparameters);
      return stepLabel(step) + ": " + setting.method + (params ? " (" + params + ")" : "");
    })
    .join(" · ");
}

function fittedTrials() {
  return (view.data ? view.data.trials : []).filter(
    (t) => t.training_status === "FITTED" && t.loss !== null && t.loss !== undefined
  );
}

/* ------------------------------------------------------------- polling */

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(response.status + " " + response.statusText);
  return response.json();
}

async function pollState() {
  try {
    const data = await getJSON("/api/state");
    const runId = [data.path, data.run.start, data.run.model_name].join("|");
    if (runId !== view.runId) {
      // another run: start over
      view.runId = runId;
      view.layout = null;
      view.layoutEmphasis = null;
      view.scores = null;
      view.signature = null;
      view.tableSignature = null;
      view.selection = null;
      $("selection").hidden = true;
    }
    view.data = data;
    render();
  } catch (err) {
    setBadge("disconnected", "stale");
  }
  setTimeout(pollState, POLL_STATE_MS);
}

async function pollMap() {
  try {
    await updateMap();
  } catch (err) {
    $("map-status").textContent = "Map not available: " + err.message;
  }
  setTimeout(pollMap, POLL_MAP_MS);
}

/* -------------------------------------------------------------- header */

function setBadge(text, cls) {
  const badge = $("run-status");
  badge.textContent = text;
  badge.className = "badge" + (cls ? " " + cls : "");
}

function renderHeader() {
  const { run, progress } = view.data;
  $("run-name").textContent = run.model_name || "Optimization";
  document.title = (run.model_name ? run.model_name + " · " : "") + "InsurAutoML Dashboard";

  const items = [
    ["task", run.task_mode],
    ["objective", run.objective],
    ["search", run.search_algo],
    ["scheduler", run.search_scheduler],
    ["ensemble", run.ensemble_strategy ? run.ensemble_strategy + (run.n_estimators ? " × " + run.n_estimators : "") : null],
    ["time limit / step", run.timeout_per_trial ? fmtDuration(run.timeout_per_trial) : null],
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");
  $("run-meta").innerHTML = items
    .map(([key, value]) => "<span>" + escapeHtml(key) + " <b>" + escapeHtml(value) + "</b></span>")
    .join("");

  if (progress.finished) {
    setBadge("finished", "finished");
  } else if (!run.start) {
    setBadge("waiting for the run…");
  } else if (
    view.data.last_time && view.data.server_time - view.data.last_time > STALE_S
  ) {
    setBadge("no updates for " + fmtDuration(view.data.server_time - view.data.last_time), "stale");
  } else {
    setBadge("running");
  }
}

/* ------------------------------------------------------------ progress */

function renderProgress() {
  const { progress, best } = view.data;
  const timeout = progress.timeout;
  const timeShare = timeout ? Math.min(1, progress.elapsed / timeout) : 0;
  $("time-bar").style.width = (timeShare * 100).toFixed(1) + "%";
  $("time-text").textContent = fmtDuration(progress.elapsed) + (timeout ? " / " + fmtDuration(timeout) : "");

  const total = progress.trials_total;
  const trialShare = total ? Math.min(1, progress.trials_finished / total) : 0;
  $("trial-bar").style.width = (trialShare * 100).toFixed(1) + "%";
  $("trial-text").textContent = progress.trials_finished + (total ? " / " + total : "") + " finished";

  $("eta").textContent = progress.finished
    ? "done"
    : progress.eta === null ? "estimating…"
      : progress.eta < 1 ? "finishing…" : "~" + fmtDuration(progress.eta);
  $("round").textContent = progress.rounds > 1
    ? Math.max(progress.round, 1) + " / " + progress.rounds
    : "1 / 1";
  $("best-loss").textContent = best ? fmtLoss(best.loss) : "–";

  const counts = progress.counts;
  const chips = [
    ["running", counts.RUNNING, "--accent"],
    ["fitted", counts.FITTED, "--ok"],
    ["timed out", counts.TIMEOUT, "--warn"],
    ["failed", counts.ERROR, "--bad"],
    ["stopped", counts.TERMINATED, "--idle"],
  ];
  $("counts").innerHTML = chips
    .map(([label, count, color]) =>
      '<span class="chip"><span class="dot" style="background:var(' + color + ')"></span>' +
      escapeHtml(count) + " " + escapeHtml(label) + "</span>")
    .join("");
}

/* -------------------------------------------------------- best/running */

function renderPipeline(trial) {
  return steps()
    .filter((step) => (trial.config || {})[step])
    .map((step) => {
      const setting = trial.config[step];
      const params = paramsText(setting.hyperparameters);
      return '<div class="step"><span class="step-name">' + escapeHtml(stepLabel(step)) +
        '</span><span><div class="method">' + escapeHtml(setting.method) + "</div>" +
        (params ? '<div class="params">' + escapeHtml(params) + "</div>" : "") +
        "</span></div>";
    })
    .join("");
}

function renderBest() {
  const best = view.data.best;
  const element = $("best");
  if (!best) {
    element.className = "pipeline muted";
    element.textContent = "No fitted trial yet.";
    $("best-meta").textContent = "";
    return;
  }
  element.className = "pipeline";
  element.innerHTML = renderPipeline(best) || '<span class="muted">No settings recorded.</span>';
  $("best-meta").textContent = "loss " + fmtLoss(best.loss) + " · trial #" + best.number +
    (view.data.progress.rounds > 1 ? " · round " + (best.round + 1) : "");
}

function renderRunning() {
  const running = view.data.running;
  $("running-count").textContent = running.length ? String(running.length) : "";
  const element = $("running");
  if (!running.length) {
    element.className = "running muted";
    element.textContent = view.data.progress.finished ? "The optimization finished." : "No trial running.";
    return;
  }
  element.className = "running";
  element.innerHTML = running
    .map((trial) => {
      const elapsed = Math.max(0, view.data.server_time - trial.start);
      const progress = trial.iterations
        ? trial.iterations + (trial.iterations === 1 ? " step" : " steps") +
          (trial.loss !== null && trial.loss !== undefined ? " · loss " + fmtLoss(trial.loss) : "")
        : "first step";
      return '<div class="run-item"><div class="row"><span><b>#' + trial.number + "</b> " +
        escapeHtml(modelOf(trial)) + '</span><span class="muted">' + fmtDuration(elapsed) +
        "</span></div>" + '<div class="params">' + escapeHtml(progress) + "</div></div>";
    })
    .join("");
}

/* -------------------------------------------------------------- charts */

function renderCharts() {
  if (!hasPlotly()) return;
  renderLossChart();
  renderMethodChart();
  renderDurationChart();
}

function renderLossChart() {
  const fitted = fittedTrials().slice().sort((a, b) => a.number - b.number);
  const positive = fitted.every((t) => t.loss > 0);
  const logBox = $("log-scale");
  logBox.disabled = !positive;
  logBox.parentElement.title = positive ? "" : "Log scale needs positive losses";
  const traces = [{
    type: "scatter",
    mode: "markers",
    name: "trials",
    x: fitted.map((t) => t.number),
    y: fitted.map((t) => t.loss),
    text: fitted.map((t) => escapeHtml(modelOf(t))),
    marker: { color: cssVar("--accent"), size: 8, opacity: 0.8 },
    hovertemplate: "#%{x} %{text}<br>loss %{y:.4g}<extra></extra>",
  }];
  // best so far
  let best = Infinity;
  const steps = { x: [], y: [] };
  for (const trial of fitted) {
    best = Math.min(best, trial.loss);
    steps.x.push(trial.number);
    steps.y.push(best);
  }
  traces.push({
    type: "scatter",
    mode: "lines",
    name: "best so far",
    x: steps.x,
    y: steps.y,
    line: { shape: "hv", color: cssVar("--text"), width: 1.5 },
    hovertemplate: "best so far %{y:.4g}<extra></extra>",
  });
  const layout = plotLayout({
    xaxis: Object.assign(plotLayout().xaxis, { title: { text: "trial" } }),
    yaxis: Object.assign(plotLayout().yaxis, {
      title: { text: "loss (lower is better)" },
      type: positive && logBox.checked ? "log" : "linear",
    }),
    uirevision: "loss",
  });
  Plotly.react("loss-chart", traces, layout, PLOT_CONFIG);
}

function renderMethodChart() {
  const select = $("method-step");
  const available = steps();
  if (select.options.length !== available.length) {
    select.innerHTML = available
      .map((step) => '<option value="' + escapeHtml(step) + '">' + escapeHtml(stepLabel(step)) + "</option>")
      .join("");
    select.value = available.includes(view.methodStep) ? view.methodStep : (available.includes("model") ? "model" : available[0] || "");
  }
  const step = select.value;
  view.methodStep = step;
  const groups = {};
  for (const trial of fittedTrials()) {
    const setting = (trial.config || {})[step];
    const method = setting ? setting.method : "–";
    (groups[method] = groups[method] || []).push(trial.loss);
  }
  const median = (values) => {
    const s = values.slice().sort((a, b) => a - b);
    return s[Math.floor((s.length - 1) / 2)];
  };
  const order = Object.keys(groups).sort((a, b) => median(groups[a]) - median(groups[b]));
  const traces = order.map((method) => ({
    type: "box",
    orientation: "h",
    name: method,
    x: groups[method],
    boxpoints: "all",
    jitter: 0.4,
    pointpos: 0,
    marker: { size: 5, color: cssVar("--accent"), opacity: 0.7 },
    line: { color: cssVar("--muted"), width: 1 },
    fillcolor: "rgba(0,0,0,0)",
    hovertemplate: escapeHtml(method) + "<br>loss %{x:.4g}<extra></extra>",
  }));
  // best method on top
  traces.reverse();
  const height = Math.max(260, 22 * traces.length + 70);
  $("method-chart").style.height = height + "px";
  const layout = plotLayout({
    height,
    xaxis: Object.assign(plotLayout().xaxis, { title: { text: "loss" } }),
    yaxis: Object.assign(plotLayout().yaxis, { tickfont: { size: 11 } }),
    margin: { l: 8, r: 12, t: 8, b: 40 },
  });
  Plotly.react("method-chart", traces, layout, PLOT_CONFIG);
}

function renderDurationChart() {
  const trials = view.data.trials.filter((t) => t.duration !== null && t.duration !== undefined);
  const limit = view.data.run.timeout_per_trial;
  const durations = trials.map((t) => t.duration).sort((a, b) => a - b);
  const median = durations.length ? durations[Math.floor((durations.length - 1) / 2)] : null;
  $("duration-meta").textContent = durations.length
    ? "median " + fmtDuration(median) + (limit ? " · time limit per step " + fmtDuration(limit) : "")
    : "";
  const statusColor = { FITTED: "--ok", TIMEOUT: "--warn", ERROR: "--bad", TERMINATED: "--idle" };
  const traces = Object.entries(statusColor).map(([status, color]) => ({
    type: "histogram",
    name: status.toLowerCase(),
    x: trials.filter((t) => t.status === status).map((t) => t.duration),
    marker: { color: cssVar(color) },
    opacity: 0.85,
    hovertemplate: "%{x}s: %{y} trials<extra>" + status.toLowerCase() + "</extra>",
  })).filter((trace) => trace.x.length);
  const layout = plotLayout({
    barmode: "stack",
    xaxis: Object.assign(plotLayout().xaxis, { title: { text: "seconds" } }),
    yaxis: Object.assign(plotLayout().yaxis, { title: { text: "trials" } }),
  });
  Plotly.react("duration-chart", traces, layout, PLOT_CONFIG);
}

/* --------------------------------------------------------------- table */

function sortValue(trial, key) {
  if (key === "model") return modelOf(trial);
  const value = trial[key];
  return value === null || value === undefined ? Infinity : value;
}

function renderTable() {
  const filter = $("filter").value.trim().toLowerCase();
  const signature = JSON.stringify([view.signature, filter, view.sortKey, view.sortAsc]);
  if (signature === view.tableSignature) return;
  view.tableSignature = signature;

  const bestKey = view.data.best ? view.data.best.key : null;
  const multiRound = view.data.progress.rounds > 1;
  let trials = view.data.trials.slice();
  trials.sort((a, b) => {
    const x = sortValue(a, view.sortKey), y = sortValue(b, view.sortKey);
    const order = x < y ? -1 : x > y ? 1 : 0;
    return view.sortAsc ? order : -order;
  });
  const rows = [];
  for (const trial of trials) {
    const pipeline = pipelineText(trial);
    const haystack = [trial.status, modelOf(trial), pipeline, "#" + trial.number].join(" ").toLowerCase();
    if (filter && !haystack.includes(filter)) continue;
    const error = trial.error
      ? '<details><summary>error</summary><pre class="error">' + escapeHtml(trial.error) + "</pre></details>"
      : "";
    rows.push(
      "<tr" + (trial.key === bestKey ? ' class="best"' : "") + ">" +
      '<td class="num">' + trial.number + "</td>" +
      '<td class="num">' + (multiRound ? trial.round + 1 : 1) + "</td>" +
      '<td><span class="status s-' + escapeHtml(trial.status) + '">' + escapeHtml(trial.status.toLowerCase()) + "</span>" + error + "</td>" +
      '<td class="num">' + fmtLoss(trial.loss) + "</td>" +
      '<td class="num">' + (trial.duration !== null && trial.duration !== undefined
        ? fmtDuration(trial.duration)
        : fmtDuration(view.data.server_time - trial.start)) + "</td>" +
      "<td>" + escapeHtml(modelOf(trial)) + "</td>" +
      '<td class="pipe">' + escapeHtml(pipeline) + "</td></tr>"
    );
  }
  $("trials").querySelector("tbody").innerHTML = rows.join("") ||
    '<tr><td colspan="7" class="muted">No trials' + (filter ? " match the filter" : " yet") + ".</td></tr>";
  for (const th of $("trials").querySelectorAll("th[data-sort]")) {
    th.classList.toggle("sorted", th.dataset.sort === view.sortKey);
    th.classList.toggle("asc", th.dataset.sort === view.sortKey && view.sortAsc);
  }
}

/* ----------------------------------------------------------------- map */

function renderEmphasis() {
  const select = $("emphasis");
  const available = view.data.map_steps || [];
  const options = available.map((step) => step).join(",");
  if (select.dataset.options === options) return;
  select.dataset.options = options;
  select.innerHTML = available
    .map((step) => '<option value="' + escapeHtml(step) + '">' + escapeHtml(stepLabel(step)) + "</option>")
    .join("");
  if (!available.includes(view.emphasis)) view.emphasis = available.includes("model") ? "model" : available[0];
  select.value = view.emphasis || "";
}

async function updateMap() {
  if (!view.data) return;
  const status = $("map-status");
  if (!(view.data.map_steps || []).length) {
    status.textContent = view.data.run.start
      ? "No search space recorded for this run: the map is not available."
      : "Waiting for the search space…";
    return;
  }
  const emphasis = view.emphasis;
  if (!view.layout || view.layoutEmphasis !== emphasis) {
    const result = await getJSON("/api/map?emphasis=" + encodeURIComponent(emphasis));
    if (result.status !== "ready") {
      status.textContent = result.status === "building"
        ? "Building the map of the search space…"
        : "Map not available" + (result.error ? ": " + result.error : ".");
      return;
    }
    if (emphasis !== view.emphasis) return;
    view.layout = result.layout;
    view.layoutEmphasis = emphasis;
  }
  const scores = await getJSON("/api/map/scores?emphasis=" + encodeURIComponent(emphasis));
  if (scores.status !== "ready" || emphasis !== view.emphasis) return;
  view.scores = scores;
  const scored = Object.keys(scores.scores).length;
  status.textContent = scored
    ? scored + " fitted trials on the map · " + view.layout.x.length + " sampled settings"
    : "The map is colored once trials are fitted.";
  drawMap();
}

function drawMap() {
  if (!hasPlotly() || !view.layout || !view.scores) return;
  const layout = view.layout, scores = view.scores;
  const idle = cssVar("--idle");
  const explored = { x: [], y: [], c: [], o: [] };
  const unexplored = { x: [], y: [] };
  for (let i = 0; i < layout.x.length; i++) {
    const score = scores.score[i], confidence = scores.confidence[i];
    if (score !== null && confidence > 0.05) {
      explored.x.push(layout.x[i]);
      explored.y.push(layout.y[i]);
      explored.c.push(score);
      explored.o.push(0.1 + 0.55 * confidence);
    } else {
      unexplored.x.push(layout.x[i]);
      unexplored.y.push(layout.y[i]);
    }
  }
  const trials = view.data.trials.filter((t) => scores.positions[t.key]);
  const group = (filter) => trials.filter(filter);
  const fitted = group((t) => scores.scores[t.key] !== undefined);
  const running = group((t) => t.status === "RUNNING");
  const other = group((t) => t.status !== "RUNNING" && scores.scores[t.key] === undefined);
  const best = view.data.best && scores.positions[view.data.best.key] ? [view.data.best] : [];

  // trial number only: details are in the trial table (a large box hides the map)
  const hover = (t) => "#" + t.number;
  const xy = (list, axis) => list.map((t) => scores.positions[t.key][axis]);
  const card = cssVar("--card");

  const traces = [
    {
      type: "scatter", mode: "markers", name: "unexplored",
      x: unexplored.x, y: unexplored.y,
      marker: { size: 5, color: idle, opacity: 0.16 },
      hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "markers", name: "estimated",
      x: explored.x, y: explored.y,
      marker: { size: 6, color: explored.c, colorscale: SCALE, cmin: 0, cmax: 1, opacity: explored.o },
      hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "markers", name: "failed / timed out",
      x: xy(other, 0), y: xy(other, 1), customdata: other.map((t) => t.key),
      text: other.map(hover), hovertemplate: "%{text}<extra></extra>",
      marker: { symbol: "x", size: 8, color: idle },
    },
    {
      type: "scatter", mode: "markers", name: "fitted",
      x: xy(fitted, 0), y: xy(fitted, 1), customdata: fitted.map((t) => t.key),
      text: fitted.map(hover), hovertemplate: "%{text}<extra></extra>",
      marker: {
        size: 11, color: fitted.map((t) => scores.scores[t.key]),
        colorscale: SCALE, cmin: 0, cmax: 1,
        line: { color: cssVar("--text"), width: 1 },
        colorbar: {
          title: { text: "rank", side: "right" }, thickness: 10, len: 0.6,
          tickvals: [0, 1], ticktext: ["worse", "better"], outlinewidth: 0,
        },
      },
    },
    {
      type: "scatter", mode: "markers", name: "running",
      x: xy(running, 0), y: xy(running, 1), customdata: running.map((t) => t.key),
      text: running.map(hover), hovertemplate: "%{text}<extra></extra>",
      marker: { symbol: "circle-open", size: 15, color: cssVar("--accent"), line: { width: 2 } },
    },
    {
      type: "scatter", mode: "markers", name: "best",
      x: xy(best, 0), y: xy(best, 1), customdata: best.map((t) => t.key),
      text: best.map(hover), hovertemplate: "%{text}<extra>best</extra>",
      marker: { symbol: "star", size: 22, color: "#d7191c", line: { color: cssVar("--text"), width: 1.5 } },
    },
  ];
  const annotations = layout.labels.map((label) => ({
    x: label.x, y: label.y, text: escapeHtml(label.text), showarrow: false,
    font: { size: 11, color: cssVar("--muted") },
    // labels are drawn above the data: no background hiding the trials
    opacity: 0.9,
  }));
  const hidden = { visible: true, showgrid: false, zeroline: false, showticklabels: false, ticks: "", showline: false };
  Plotly.react("map", traces, plotLayout({
    xaxis: hidden, yaxis: hidden, annotations,
    margin: { l: 8, r: 8, t: 8, b: 8 },
    dragmode: "lasso",
    uirevision: view.layoutEmphasis,
    hovermode: "closest",
  }), Object.assign({}, PLOT_CONFIG, { modeBarButtonsToRemove: ["autoScale2d"] }));
}

function bindMapSelection() {
  const map = $("map");
  map.on("plotly_selected", (event) => {
    const keys = new Set();
    for (const point of (event && event.points) || []) {
      if (point.customdata) keys.add(point.customdata);
    }
    view.selection = keys.size ? keys : null;
    renderSelection();
  });
  map.on("plotly_deselect", () => {
    view.selection = null;
    renderSelection();
  });
}

function renderSelection() {
  const box = $("selection");
  if (!view.selection || !view.data) {
    box.hidden = true;
    return;
  }
  const trials = view.data.trials.filter((t) => view.selection.has(t.key));
  const scores = view.scores ? view.scores.scores : {};
  const scored = trials.filter((t) => scores[t.key] !== undefined).map((t) => scores[t.key]).sort((a, b) => a - b);
  const lines = [];
  let head = "<b>" + trials.length + " trials selected</b>";
  if (scored.length) {
    const median = scored[Math.floor((scored.length - 1) / 2)];
    const best = trials.filter((t) => scores[t.key] !== undefined).reduce((a, b) => (a.loss <= b.loss ? a : b));
    head += " · median rank top " + Math.max(1, Math.round((1 - median) * 100)) + "% · best loss " + fmtLoss(best.loss);
  }
  lines.push(head);
  for (const step of steps()) {
    const counts = {};
    for (const trial of trials) {
      const setting = (trial.config || {})[step];
      if (setting) counts[setting.method] = (counts[setting.method] || 0) + 1;
    }
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    if (!entries.length) continue;
    const [method, count] = entries[0];
    const share = count / trials.length;
    let text = escapeHtml(stepLabel(step)) + ": <b>" + escapeHtml(method) + "</b> " + Math.round(share * 100) + "%";
    if (entries.length > 1) text += ' <span class="muted">(+' + (entries.length - 1) + " other)</span>";
    // hyperparameters of the most common method
    const values = {};
    for (const trial of trials) {
      const setting = (trial.config || {})[step];
      if (!setting || setting.method !== method) continue;
      for (const [key, value] of Object.entries(setting.hyperparameters || {})) {
        (values[key] = values[key] || []).push(value);
      }
    }
    const params = Object.entries(values).map(([key, list]) => {
      if (list.every((v) => typeof v === "number")) {
        const lo = Math.min(...list), hi = Math.max(...list);
        return key + " " + (lo === hi ? fmtValue(lo) : fmtValue(lo) + "–" + fmtValue(hi));
      }
      const freq = {};
      for (const v of list) freq[String(v)] = (freq[String(v)] || 0) + 1;
      const [top, n] = Object.entries(freq).sort((a, b) => b[1] - a[1])[0];
      return key + " = " + top + (n < list.length ? " (" + Math.round((n / list.length) * 100) + "%)" : "");
    });
    if (params.length) text += ' <span class="muted">· ' + escapeHtml(params.join(" · ")) + "</span>";
    lines.push("<div>" + text + "</div>");
  }
  box.innerHTML = lines.join("");
  box.hidden = false;
}

/* --------------------------------------------------------------- render */

function render() {
  const data = view.data;
  renderHeader();
  renderProgress();
  renderBest();
  renderRunning();
  renderEmphasis();

  // charts and table only when the trials changed
  const signature = JSON.stringify(data.trials.map((t) => [t.key, t.status, t.loss, t.duration]));
  if (signature !== view.signature) {
    view.signature = signature;
    renderCharts();
    if (view.selection) renderSelection();
  }
  renderTable();
  $("footer").textContent = "Event log: " + data.path;
}

function init() {
  if (!hasPlotly()) {
    $("plotly-missing").hidden = false;
    for (const id of ["map", "loss-chart", "method-chart", "duration-chart"]) {
      $(id).innerHTML = '<p class="muted">Chart not available (plotly not installed).</p>';
      $(id).style.height = "auto";
    }
  } else {
    // empty plots, so the map selection can be bound
    const hidden = { visible: false };
    Plotly.newPlot("map", [], plotLayout({ xaxis: hidden, yaxis: hidden }), PLOT_CONFIG)
      .then(bindMapSelection);
  }
  $("emphasis").addEventListener("change", (event) => {
    view.emphasis = event.target.value;
    view.selection = null;
    $("selection").hidden = true;
    $("map-status").textContent = "Building the map of the search space…";
    updateMap().catch(() => {});
  });
  $("method-step").addEventListener("change", () => view.data && hasPlotly() && renderMethodChart());
  $("log-scale").addEventListener("change", () => view.data && hasPlotly() && renderLossChart());
  $("filter").addEventListener("input", () => view.data && renderTable());
  for (const th of document.querySelectorAll("#trials th[data-sort]")) {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      view.sortAsc = view.sortKey === key ? !view.sortAsc : true;
      view.sortKey = key;
      if (view.data) renderTable();
    });
  }
  // follow the system theme in the charts
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      view.signature = null;
      if (view.data) { render(); drawMap(); }
    });
  }
  pollState();
  pollMap();
}

document.addEventListener("DOMContentLoaded", init);
