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

const THEMES = ["auto", "light", "dark"];
const THEME_LABELS = { auto: "Auto", light: "Light", dark: "Dark" };

// status: color + icon + label (color is never the only cue)
const STATUS = {
  RUNNING: { label: "running", color: "--accent",
    icon: '<circle cx="7" cy="7" r="5.25"/><path d="M7 4.3V7l1.9 1.3"/>' },
  FITTED: { label: "fitted", color: "--good",
    icon: '<circle cx="7" cy="7" r="5.25"/><path d="M4.6 7.1l1.7 1.7 3.2-3.4"/>' },
  TIMEOUT: { label: "timed out", color: "--serious",
    icon: '<circle cx="7" cy="7" r="5.25"/><path d="M7 4v3.4M7 9.6v.1"/>' },
  ERROR: { label: "failed", color: "--critical",
    icon: '<circle cx="7" cy="7" r="5.25"/><path d="M5 5l4 4M9 5L5 9"/>' },
  TERMINATED: { label: "stopped", color: "--muted",
    icon: '<circle cx="7" cy="7" r="5.25"/><path d="M5.2 7h3.6"/>' },
};

function statusIcon(status) {
  const spec = STATUS[status] || STATUS.TERMINATED;
  return '<svg class="status-icon" viewBox="0 0 14 14" aria-hidden="true" style="stroke:var(' +
    spec.color + ')">' + spec.icon + "</svg>";
}

// map colors: diverging pair, worse (score 0, cool) - neutral - better (score 1, warm)
function divergingScale() {
  return [0, 1, 2, 3, 4].map((i) => [i / 4, cssVar("--div-" + i)]);
}

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


function plotLayout(extra) {
  const grid = cssVar("--grid"), axisColor = cssVar("--axis");
  const axis = {
    gridcolor: grid,
    gridwidth: 1,
    zeroline: false,
    linecolor: axisColor,
    showline: true,
    ticks: "",
    tickfont: { color: cssVar("--muted"), size: 11 },
    title: { font: { color: cssVar("--muted"), size: 11.5 } },
    automargin: true,
  };
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: cssVar("--ink-2"), size: 12, family: cssVar("--font") },
    margin: { l: 52, r: 14, t: 6, b: 40 },
    xaxis: Object.assign({}, axis),
    yaxis: Object.assign({}, axis),
    hoverlabel: {
      bgcolor: cssVar("--surface"),
      bordercolor: cssVar("--axis"),
      font: { color: cssVar("--ink"), size: 12, family: cssVar("--font") },
    },
    modebar: { bgcolor: "rgba(0,0,0,0)", color: cssVar("--muted"), activecolor: cssVar("--accent") },
    showlegend: false,
  }, extra || {});
}

function axisTitle(text) {
  return { text, font: { color: cssVar("--muted"), size: 11.5 } };
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
    setBadge("disconnected", "is-stale");
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
  badge.querySelector(".pill-text").textContent = text;
  badge.className = "pill" + (cls ? " " + cls : "");
}

function renderHeader() {
  const { run, progress } = view.data;
  $("run-name").textContent = run.model_name || "Optimization";
  document.title = (run.model_name ? run.model_name + " · " : "") + "InsurAutoML Dashboard";

  const items = [
    ["Task", run.task_mode],
    ["Objective", run.objective],
    ["Search", run.search_algo],
    ["Scheduler", run.search_scheduler],
    ["Ensemble", run.ensemble_strategy ? run.ensemble_strategy + (run.n_estimators ? " × " + run.n_estimators : "") : null],
    ["Time limit per step", run.timeout_per_trial ? fmtDuration(run.timeout_per_trial) : null],
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");
  $("run-meta").innerHTML = items
    .map(([key, value]) => "<div><dt>" + escapeHtml(key) + "</dt><dd>" + escapeHtml(value) + "</dd></div>")
    .join("");

  if (progress.finished) {
    setBadge("finished", "is-finished");
  } else if (!run.start) {
    setBadge("waiting for the run…");
  } else if (
    progress.stale ||
    (view.data.last_time && view.data.server_time - view.data.last_time > STALE_S)
  ) {
    setBadge("no updates for " + fmtDuration(view.data.server_time - view.data.last_time), "is-stale");
  } else {
    setBadge("running", "is-running");
  }
}

/* ------------------------------------------------------------ progress */

function renderProgress() {
  const { progress, best } = view.data;
  const timeout = progress.timeout;
  const timeShare = timeout ? Math.min(1, progress.elapsed / timeout) : 0;
  $("time-bar").style.width = (timeShare * 100).toFixed(1) + "%";
  $("time-text").innerHTML = escapeHtml(fmtDuration(progress.elapsed)) +
    (timeout ? '<span class="unit"> / ' + escapeHtml(fmtDuration(timeout)) + "</span>" : "");

  const total = progress.trials_total;
  const trialShare = total ? Math.min(1, progress.trials_finished / total) : 0;
  $("trial-bar").style.width = (trialShare * 100).toFixed(1) + "%";
  $("trial-text").innerHTML = escapeHtml(progress.trials_finished) +
    (total ? '<span class="unit"> / ' + escapeHtml(total) + "</span>" : "");

  $("eta").textContent = progress.finished
    ? "Done"
    : progress.eta === null ? "Estimating…"
      : progress.eta < 1 ? "Finishing…" : "~" + fmtDuration(progress.eta);
  $("eta-sub").textContent = progress.finished
    ? "in " + fmtDuration(progress.wall_time)
    : progress.time_left !== null && progress.time_left !== undefined
      ? fmtDuration(progress.time_left) + " of budget left" : "";
  $("round").innerHTML = escapeHtml(Math.max(progress.round, 1)) +
    '<span class="unit"> / ' + escapeHtml(Math.max(progress.rounds, 1)) + "</span>";
  $("round-sub").textContent = view.data.run.ensemble_strategy
    ? (progress.rounds > 1 ? view.data.run.ensemble_strategy + " rounds" : view.data.run.ensemble_strategy)
    : "";
  $("best-loss").textContent = best ? fmtLoss(best.loss) : "–";
  $("best-sub").textContent = best
    ? "trial #" + best.number + (progress.rounds > 1 ? " · round " + (best.round + 1) : "")
    : "no fitted trial yet";

  const counts = progress.counts;
  $("counts").innerHTML = ["RUNNING", "FITTED", "TIMEOUT", "ERROR", "TERMINATED"]
    .map((status) => {
      const count = counts[status] || 0;
      return '<span class="status-chip' + (count ? "" : " zero") + '">' + statusIcon(status) +
        "<b>" + escapeHtml(count) + "</b>" + escapeHtml(STATUS[status].label) + "</span>";
    })
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
    element.className = "pipeline empty";
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
  const badge = $("running-count");
  badge.hidden = !running.length;
  badge.textContent = String(running.length);
  const element = $("running");
  if (!running.length) {
    element.className = "running empty";
    element.textContent = view.data.progress.finished ? "The optimization finished." : "No trial running.";
    return;
  }
  element.className = "running";
  element.innerHTML = running
    .map((trial) => {
      const elapsed = Math.max(0, view.data.progress.now - trial.start);
      const progress = trial.iterations
        ? trial.iterations + (trial.iterations === 1 ? " step" : " steps") +
          (trial.loss !== null && trial.loss !== undefined ? " · loss " + fmtLoss(trial.loss) : "")
        : "first step";
      return '<div class="run-item"><div class="row"><span><span class="trial-no">#' + trial.number +
        "</span><b>" + escapeHtml(modelOf(trial)) + '</b></span><span class="elapsed">' +
        fmtDuration(elapsed) + '</span></div><div class="params">' + escapeHtml(progress) + "</div></div>";
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
  const surface = cssVar("--surface");
  // best so far (step line)
  let best = Infinity;
  const line = { x: [], y: [] };
  for (const trial of fitted) {
    best = Math.min(best, trial.loss);
    line.x.push(trial.number);
    line.y.push(best);
  }
  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: "best so far",
      x: line.x,
      y: line.y,
      line: { shape: "hv", color: cssVar("--series-2"), width: 2 },
      hovertemplate: "best so far %{y:.4g}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "fitted trial",
      x: fitted.map((t) => t.number),
      y: fitted.map((t) => t.loss),
      text: fitted.map((t) => escapeHtml(modelOf(t))),
      marker: { color: cssVar("--accent"), size: 8, line: { color: surface, width: 2 } },
      hovertemplate: "<b>#%{x}</b> %{text}<br>loss %{y:.4g}<extra></extra>",
    },
  ];
  // direct label at the end of the best-so-far line
  const annotations = line.x.length ? [{
    x: line.x[line.x.length - 1], y: line.y[line.y.length - 1],
    text: "best " + fmtLoss(line.y[line.y.length - 1]),
    showarrow: false, xanchor: "right", yanchor: "bottom", yshift: 4,
    font: { size: 11, color: cssVar("--ink-2") },
  }] : [];
  const base = plotLayout();
  const layout = plotLayout({
    xaxis: Object.assign(base.xaxis, { title: axisTitle("trial") }),
    yaxis: Object.assign(base.yaxis, {
      title: axisTitle("loss"),
      type: positive && logBox.checked ? "log" : "linear",
    }),
    annotations,
    uirevision: "loss",
  });
  Plotly.react("loss-chart", traces, layout, PLOT_CONFIG);
}

function renderMethodChart() {
  const select = $("method-step");
  const available = steps();
  if (select.dataset.options !== available.join(",")) {
    select.dataset.options = available.join(",");
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
    marker: { size: 6, color: cssVar("--accent"), opacity: 0.75, line: { color: cssVar("--surface"), width: 1 } },
    line: { color: cssVar("--accent"), width: 1.5 },
    fillcolor: cssVar("--accent-wash"),
    hovertemplate: escapeHtml(method) + "<br>loss %{x:.4g}<extra></extra>",
  }));
  // best method on top
  traces.reverse();
  const height = Math.max(260, 22 * traces.length + 70);
  $("method-chart").style.height = height + "px";
  const base = plotLayout();
  const layout = plotLayout({
    height,
    xaxis: Object.assign(base.xaxis, { title: axisTitle("loss") }),
    yaxis: Object.assign(base.yaxis, {
      tickfont: { size: 11.5, color: cssVar("--ink-2") }, showline: false, showgrid: false,
    }),
    margin: { l: 8, r: 14, t: 6, b: 40 },
  });
  Plotly.react("method-chart", traces, layout, PLOT_CONFIG);
}

function renderDurationChart() {
  const trials = view.data.trials.filter((t) => t.duration !== null && t.duration !== undefined);
  const limit = view.data.run.timeout_per_trial;
  const durations = trials.map((t) => t.duration).sort((a, b) => a - b);
  const median = durations.length ? durations[Math.floor((durations.length - 1) / 2)] : null;
  $("duration-meta").textContent = durations.length
    ? durations.length + " finished · median " + fmtDuration(median) +
      (limit ? " · limit " + fmtDuration(limit) + " per step" : "")
    : "Finished trials";
  const traces = [{
    type: "histogram",
    name: "trials",
    x: durations,
    nbinsx: Math.min(24, Math.max(6, Math.ceil(Math.sqrt(durations.length || 1)) * 2)),
    marker: { color: cssVar("--accent"), line: { color: cssVar("--surface"), width: 2 } },
    hovertemplate: "%{x}<br>%{y} trials<extra></extra>",
  }];
  const shapes = median !== null ? [{
    type: "line", x0: median, x1: median, yref: "paper", y0: 0, y1: 1,
    line: { color: cssVar("--ink-2"), width: 1.5, dash: "dot" },
  }] : [];
  const annotations = median !== null ? [{
    x: median, yref: "paper", y: 1, text: "median " + fmtDuration(median), showarrow: false,
    xanchor: "center", yanchor: "bottom", yshift: 2, font: { size: 11, color: cssVar("--ink-2") },
  }] : [];
  const base = plotLayout();
  const layout = plotLayout({
    bargap: 0.04,
    margin: { l: 52, r: 14, t: 22, b: 40 },
    shapes,
    annotations,
    xaxis: Object.assign(base.xaxis, { title: axisTitle("seconds") }),
    yaxis: Object.assign(base.yaxis, { title: axisTitle("trials") }),
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
  // running trials: their time changes on every update
  const clock = view.data.running.length ? Math.floor(view.data.progress.now) : null;
  const signature = JSON.stringify([view.signature, filter, view.sortKey, view.sortAsc, clock]);
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
    const status = STATUS[trial.status] || STATUS.TERMINATED;
    rows.push(
      "<tr" + (trial.key === bestKey ? ' class="best"' : "") + ">" +
      '<td class="num">' + trial.number + "</td>" +
      '<td class="num">' + (multiRound ? trial.round + 1 : 1) + "</td>" +
      '<td><span class="status-pill">' + statusIcon(trial.status) + escapeHtml(status.label) + "</span>" +
        (trial.key === bestKey ? '<span class="best-tag">best</span>' : "") + error + "</td>" +
      '<td class="num">' + fmtLoss(trial.loss) + "</td>" +
      '<td class="num">' + (trial.duration !== null && trial.duration !== undefined
        ? fmtDuration(trial.duration)
        : fmtDuration(view.data.progress.now - trial.start)) + "</td>" +
      '<td class="model">' + escapeHtml(modelOf(trial)) + "</td>" +
      '<td class="pipe">' + escapeHtml(pipeline) + "</td></tr>"
    );
  }
  $("trials").querySelector("tbody").innerHTML = rows.join("") ||
    '<tr><td colspan="7" class="empty">No trials' + (filter ? " match the filter" : " yet") + ".</td></tr>";
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
  const idle = cssVar("--muted"), scale = divergingScale(), surface = cssVar("--surface");
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

  const traces = [
    {
      type: "scatter", mode: "markers", name: "unexplored",
      x: unexplored.x, y: unexplored.y,
      marker: { size: 5, color: cssVar("--axis"), opacity: 0.45 },
      hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "markers", name: "estimated",
      x: explored.x, y: explored.y,
      marker: { size: 6, color: explored.c, colorscale: scale, cmin: 0, cmax: 1, opacity: explored.o },
      hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "markers", name: "failed / timed out",
      x: xy(other, 0), y: xy(other, 1), customdata: other.map((t) => t.key),
      text: other.map(hover), hovertemplate: "%{text}<extra></extra>",
      marker: { symbol: "x-thin", size: 8, color: idle, line: { color: idle, width: 2 } },
    },
    {
      type: "scatter", mode: "markers", name: "fitted",
      x: xy(fitted, 0), y: xy(fitted, 1), customdata: fitted.map((t) => t.key),
      text: fitted.map(hover), hovertemplate: "%{text}<extra></extra>",
      marker: {
        size: 11, color: fitted.map((t) => scores.scores[t.key]),
        colorscale: scale, cmin: 0, cmax: 1,
        line: { color: surface, width: 2 },
      },
    },
    {
      type: "scatter", mode: "markers", name: "running",
      x: xy(running, 0), y: xy(running, 1), customdata: running.map((t) => t.key),
      text: running.map(hover), hovertemplate: "%{text}<extra></extra>",
      marker: { symbol: "circle-open", size: 16, color: cssVar("--accent"), line: { width: 2.5 } },
    },
    {
      type: "scatter", mode: "markers", name: "best",
      x: xy(best, 0), y: xy(best, 1), customdata: best.map((t) => t.key),
      text: best.map(hover), hovertemplate: "%{text}<extra>best</extra>",
      marker: { symbol: "star", size: 22, color: cssVar("--div-4"), line: { color: cssVar("--ink"), width: 1.5 } },
    },
  ];
  const annotations = layout.labels.map((label) => ({
    x: label.x, y: label.y, text: escapeHtml(label.text), showarrow: false,
    font: { size: 11, color: cssVar("--ink-2") },
    // labels are drawn above the data: no background hiding the trials
    opacity: 0.85,
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

function currentTheme() {
  try {
    return localStorage.getItem("insurautoml-theme") || "auto";
  } catch (err) {
    return "auto";
  }
}

function applyTheme(theme) {
  if (theme === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $("theme-label").textContent = THEME_LABELS[theme];
  $("theme-toggle").setAttribute("aria-label", "Color theme: " + THEME_LABELS[theme].toLowerCase());
  try {
    localStorage.setItem("insurautoml-theme", theme);
  } catch (err) {
    // no storage (private window): the theme is not remembered
  }
}

function redraw() {
  view.signature = null;
  view.tableSignature = null;
  if (view.data) { render(); drawMap(); }
}

function init() {
  applyTheme(currentTheme());
  $("theme-toggle").addEventListener("click", () => {
    const theme = THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length];
    applyTheme(theme);
    redraw();
  });
  if (!hasPlotly()) {
    $("plotly-missing").hidden = false;
    for (const id of ["map", "loss-chart", "method-chart", "duration-chart"]) {
      $(id).innerHTML = '<p class="empty">Chart not available (plotly not installed).</p>';
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
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  }
  pollState();
  pollMap();
}

document.addEventListener("DOMContentLoaded", init);
