"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  model: null,
  file: null,
  result: null,
  view: { start: 0, width: 20 },
  gain: 1,
  trace: null,
  traceKey: "",
  current: -1,
  busy: false,
};

/* ---------- formatage ---------- */

const nf1 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
const nf0 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });

function clock(seconds, decimals = 0) {
  const sign = seconds < 0 ? "−" : "";
  const s = Math.abs(seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  const sec = decimals ? rest.toFixed(decimals).padStart(3 + decimals, "0") : String(Math.floor(rest)).padStart(2, "0");
  return `${sign}${m}:${sec}`;
}

function duration(seconds) {
  if (seconds < 60) return `${nf1.format(seconds)} s`;
  const m = Math.floor(seconds / 60);
  return `${m} min ${String(Math.round(seconds - m * 60)).padStart(2, "0")} s`;
}

function bytes(n) {
  if (n < 1024 * 1024) return `${nf0.format(n / 1024)} Kio`;
  return `${nf1.format(n / 1024 / 1024)} Mio`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "width") node.style.width = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child == null) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ---------- réseau ---------- */

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({ error: "Réponse illisible du serveur." }));
  if (!response.ok) throw new Error(body.error || `Erreur ${response.status}`);
  return body;
}

function setStatus(text, kind = "", target = "status") {
  for (const id of ["status", "catalogue-status"]) if (id !== target) { $(id).textContent = ""; $(id).className = "status"; }
  const status = $(target);
  status.textContent = text;
  status.className = `status ${kind}`;
}

/* ---------- modèle ---------- */

async function loadModel() {
  try {
    state.model = await api("/api/model");
  } catch (error) {
    $("model-line").textContent = `modèle indisponible : ${error.message}`;
    $("model-line").classList.add("error");
    return;
  }
  const m = state.model;
  $("model-line").replaceChildren(
    el("b", {}, m.label), " · seuil ", el("b", {}, nf2.format(m.params.threshold)),
    ` · lissage ${m.params.smoothing} · ≥ ${m.params.min_windows} fenêtre${m.params.min_windows > 1 ? "s" : ""}`,
  );
  renderMethod();
  updateRunButton();
  await loadCatalogue();
  const wanted = new URLSearchParams(location.search).get("id");
  if (wanted) openRecord(wanted);
}

/* ---------- enregistrements par identifiant ---------- */

const STATUS = {
  test: { label: "test", detail: " · jamais vu", cls: "st-test" },
  apprentissage: { label: "apprentissage", cls: "st-train" },
  "sans annotation": { label: "sans annotation", cls: "st-none" },
  exclu: { label: "exclu", cls: "st-out" },
};

const normalise = (text) => text.toUpperCase().replace(/[^0-9A-Z]/g, "");

async function loadCatalogue() {
  try {
    state.catalogue = (await api("/api/recordings")).recordings;
  } catch (error) {
    setStatus(error.message, "error", "catalogue-status");
    return;
  }
  const body = $("record-list");
  body.replaceChildren(
    ...state.catalogue.map((r) => {
      const status = STATUS[r.status];
      const row = el("tr", { class: r.available ? "" : "unavailable", "data-key": normalise(r.id), title: r.reason },
        el("td", { class: "rec-id" }, r.id),
        el("td", { class: "num" }, r.age ?? "—"),
        el("td", {}, r.sex ?? "—"),
        el("td", { class: "num" }, r.annotated_seizures ?? "—"),
        el("td", { class: "num" }, clock(r.duration_s)),
        el("td", {}, el("span", { class: `tag ${status.cls}` }, status.label, status.detail ? el("span", { class: "wide" }, status.detail) : null),
          r.available ? null : el("span", { class: "quiet wide" }, ` · ${r.reason.toLowerCase()}`)),
      );
      if (r.available) {
        row.tabIndex = 0;
        row.addEventListener("click", () => openRecord(r.id));
        row.addEventListener("keydown", (event) => { if (event.key === "Enter") openRecord(r.id); });
      }
      return row;
    }),
  );
  $("record-count").textContent = `· ${state.catalogue.length}`;
}

function filterCatalogue() {
  const key = normalise($("record-search").value);
  let shown = 0;
  for (const row of $("record-list").children) {
    const match = !key || row.dataset.key.includes(key);
    row.hidden = !match;
    shown += match;
  }
  $("no-match").hidden = shown > 0;
}

function openRecord(id) {
  const key = normalise(id);
  const entry = (state.catalogue || []).find((r) => normalise(r.id) === key);
  const record = entry ? entry.id : id.trim().toUpperCase();
  run(() => api("/api/open", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ record }) }), record, "catalogue-status");
}

function renderMethod() {
  const m = state.model;
  const pct = (x) => `${nf0.format(100 * x)} %`;
  const row = (label, value) => el("tr", {}, el("td", {}, label), el("td", { class: "num" }, value));
  const body = $("method-body");
  body.replaceChildren(
    el("div", {},
      el("h4", {}, "chaîne de traitement"),
      el("p", {}, "Canal F7 à 256 Hz, filtré en passe-haut à 0,5 Hz. Fenêtres de 2 s avec un pas de 1 s. ",
        `${m.features.length} caractéristiques par fenêtre (log-puissances de bandes, puissances relatives, bande des pointes-ondes 2,5–4,5 Hz, longueur de ligne, kurtosis…), centrées-réduites par la médiane et l'IQR de l'enregistrement.`),
      el("p", {}, `Classifieur : ${m.label}. Le score de chaque fenêtre est lissé sur ${m.params.smoothing} fenêtre(s), comparé au seuil ${nf2.format(m.params.threshold)}, et les suites de moins de ${m.params.min_windows} fenêtres sont écartées. Ces trois réglages ont été choisis sur des prédictions hors pli de l'apprentissage, jamais sur le test.`),
    ),
    el("div", {},
      el("h4", {}, `test réservé · ${m.test_records.length} enregistrements, ${nf1.format(m.metrics.test.hours * 60)} min`),
      el("table", {}, el("tbody", {},
        row("crises retrouvées (IoU ≥ 0,3)", `${m.metrics.test.event_TP} / ${m.metrics.test.event_TP + m.metrics.test.event_FN}`),
        row("fausses alarmes par heure", nf1.format(m.metrics.test.false_alarms_per_hour)),
        row("précision par fenêtre", pct(m.metrics.test.precision)),
        row("rappel par fenêtre", pct(m.metrics.test.recall)),
        row("F1 par fenêtre", nf2.format(m.metrics.test.f1)),
      )),
    ),
    el("div", {},
      el("h4", {}, `un enregistrement exclu à tour de rôle · ${m.metrics.loro.records} tracés`),
      el("table", {}, el("tbody", {},
        row("crises retrouvées", pct(m.metrics.loro.event_recall)),
        row("fausses alarmes par heure", nf1.format(m.metrics.loro.false_alarms_per_hour)),
        row("F1 par fenêtre, médiane", nf2.format(m.metrics.loro.f1_median)),
        row("F1 par fenêtre, quartiles", `${nf2.format(m.metrics.loro.f1_q1)} – ${nf2.format(m.metrics.loro.f1_q3)}`),
      )),
      el("p", {}, "Seize enregistrements seulement, sans identifiant patient : ces chiffres décrivent cette cohorte, pas une performance clinique."),
    ),
  );
}

/* ---------- dépôt ---------- */

function updateRunButton() {
  $("run").disabled = !state.file || !state.model || state.busy;
}

function chooseFile(file) {
  if (!file) return;
  state.file = file;
  $("slot").classList.add("ready");
  $("slot-title").textContent = file.name;
  $("slot-hint").textContent = `${bytes(file.size)} · prêt à analyser`;
  setStatus("");
  updateRunButton();
}

async function run(request, label, target = "status") {
  if (state.busy) return;
  state.busy = true;
  updateRunButton();
  setStatus(`lecture de ${label} et calcul des scores`, "busy", target);
  try {
    const result = await request();
    setStatus("", "", target);
    show(result);
  } catch (error) {
    setStatus(error.message, "error", target);
  } finally {
    state.busy = false;
    updateRunButton();
  }
}

function analyzeFile() {
  const file = state.file;
  run(() => file.arrayBuffer().then((buffer) =>
    api("/api/analyze", { method: "POST", headers: { "X-Filename": encodeURIComponent(file.name) }, body: buffer })), file.name);
}


/* ---------- résultat ---------- */

function show(result) {
  state.result = result;
  state.trace = null;
  state.traceKey = "";
  state.gain = 1;
  state.current = -1;
  $("intake").hidden = true;
  $("reading").hidden = false;
  $("record-name").textContent = result.record_id || result.filename;
  $("record-source").textContent = result.record_id && result.record_id !== result.filename
    ? `${result.filename} · ${result.source}` : result.source;

  const detectedPct = result.detected_duration_s / result.analyzed_duration_s;
  const items = [
    ["durée", duration(result.duration_s), `${nf0.format(result.n_windows)} fenêtres`],
    ["fréquence", `${nf0.format(result.fs_hz)} Hz`, "mesurée"],
    ["épisodes", result.events.length, null, "is-detect"],
    ["temps signalé", duration(result.detected_duration_s), `${nf1.format(100 * detectedPct)} %`],
  ];
  if (result.reference) {
    const r = result.reference;
    items.push(["annotées", r.events.length, `${r.tp} retrouvées`]);
    items.push(["fausses alarmes", r.fp, `${nf1.format(r.fp / (result.analyzed_duration_s / 3600))} / h`]);
  }
  $("readout").replaceChildren(
    ...items.map(([label, value, note, cls]) =>
      el("div", cls ? { class: cls } : {}, el("dt", {}, label), el("dd", {}, String(value), note ? el("small", {}, note) : null))),
  );

  const verdict = $("verdict");
  const context = {
    test: "Enregistrement du test réservé : le modèle ne l'a jamais vu.",
    apprentissage: "Attention : cet enregistrement a servi à l'entraînement, la comparaison est optimiste.",
    exclu: `Enregistrement exclu de l'analyse (${(result.status_reason || "").toLowerCase()}) : détections seules, sans référence.`,
    "sans annotation": "Aucune annotation pour cet enregistrement : les épisodes sont à vérifier sur le tracé.",
  }[result.membership];
  verdict.className = `verdict v-${(result.membership || "").replace(" ", "-")}`;
  if (result.reference) {
    const r = result.reference;
    verdict.replaceChildren(
      `${context || "Annotations de référence disponibles."} `,
      el("b", {}, `${r.tp} crise${r.tp > 1 ? "s" : ""} sur ${r.events.length} retrouvée${r.tp > 1 ? "s" : ""}`),
      `, ${r.fp} épisode${r.fp > 1 ? "s" : ""} sans annotation correspondante (IoU ≥ 0,3).`,
    );
    verdict.hidden = false;
  } else {
    verdict.replaceChildren(context || "");
    verdict.hidden = !context;
  }
  const url = result.record_id ? `?id=${encodeURIComponent(result.record_id)}` : location.pathname;
  history.replaceState(null, "", url);
  $("legend-ref").hidden = !result.reference;
  $("th-ref").hidden = !result.reference;

  renderEpisodes();
  const first = result.events[0] || (result.reference && result.reference.events[0]);
  focusOn(first ? (first.start_s + first.end_s) / 2 : state.view.width / 2);
  if (result.events.length) setCurrent(0, false);
  window.scrollTo({ top: 0 });
}

function renderEpisodes() {
  const result = state.result;
  const body = $("episodes");
  $("episode-count").textContent = `· ${result.events.length}`;
  $("no-episodes").hidden = result.events.length > 0;
  body.replaceChildren(
    ...result.events.map((event, index) => {
      const cells = [
        el("td", {}, String(event.id).padStart(2, "0")),
        el("td", {}, clock(event.start_s)),
        el("td", {}, clock(event.end_s)),
        el("td", { class: "num" }, `${nf0.format(event.duration_s)} s`),
        el("td", { class: "num" }, el("span", { class: "bar", width: `${Math.round(event.mean_score * 48)}px` }), nf2.format(event.mean_score)),
        el("td", { class: "num" }, nf2.format(event.peak_score)),
      ];
      if (result.reference) {
        cells.push(event.match
          ? el("td", { class: "tag-match" }, `crise ${event.match.reference} · IoU ${nf2.format(event.match.iou)}`)
          : el("td", { class: "tag-none" }, "non annotée"));
      }
      return el("tr", { onclick: () => setCurrent(index) }, cells);
    }),
  );
  const missed = result.reference ? result.reference.events.filter((e) => !e.matched) : [];
  $("missed").hidden = missed.length === 0;
  $("missed-list").replaceChildren(
    ...missed.map((e) => el("li", {}, el("button", { onclick: () => focusOn((e.start_s + e.end_s) / 2) }, `${clock(e.start_s)} – ${clock(e.end_s)}`))),
  );
}

function setCurrent(index, move = true) {
  const events = state.result.events;
  if (!events.length) return;
  state.current = (index + events.length) % events.length;
  [...$("episodes").children].forEach((row, i) => row.classList.toggle("current", i === state.current));
  const event = events[state.current];
  if (move) {
    focusOn((event.start_s + event.end_s) / 2);
    $("episodes").children[state.current].scrollIntoView({ block: "nearest" });
  }
}

function focusOn(center) {
  const total = state.result.analyzed_duration_s;
  const width = Math.min(state.view.width, total);
  state.view.start = Math.max(0, Math.min(total - width, center - width / 2));
  draw();
}

/* ---------- tracé ---------- */

function sizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = Number(canvas.getAttribute("height"));
  if (canvas.width !== Math.round(width * ratio) || canvas._cssHeight !== height) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.height = `${height}px`;
    canvas._cssHeight = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

let traceTimer = 0;
function draw() {
  if (!state.result) return;
  drawOverview();
  drawPaper();
  const { start, width } = state.view;
  $("range").replaceChildren(el("b", {}, clock(start, 1)), " → ", el("b", {}, clock(start + width, 1)));
  const key = `${start.toFixed(2)}:${width}`;
  if (key !== state.traceKey) {
    clearTimeout(traceTimer);
    traceTimer = setTimeout(() => fetchTrace(start, width, key), 60);
  }
}

async function fetchTrace(start, width, key) {
  const end = Math.min(start + width, state.result.analyzed_duration_s + 2);
  const budget = Math.max(800, Math.round($("paper").clientWidth * 2));
  try {
    const trace = await api(`/api/trace?id=${state.result.analysis_id}&start=${start}&end=${end}&budget=${budget}`);
    state.trace = trace;
    state.traceKey = key;
    drawPaper();
  } catch (error) {
    state.trace = null;
    $("range").append(el("span", { class: "quiet" }, ` · ${error.message}`));
  }
}

function drawOverview() {
  const result = state.result;
  const canvas = $("overview");
  const { ctx, width, height } = sizeCanvas(canvas);
  const total = result.analyzed_duration_s;
  const x = (t) => (t / total) * width;
  const traceH = 62;
  const refY = 70;
  const detY = 82;
  const scoreTop = 96;
  const scoreH = height - scoreTop - 4;
  ctx.clearRect(0, 0, width, height);

  // Enveloppe min/max du signal filtré.
  const env = result.overview;
  const lim = result.amplitude_scale * 12;
  const mid = traceH / 2 + 2;
  ctx.strokeStyle = css("--signal");
  ctx.globalAlpha = 0.75;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < env.time.length; i++) {
    const px = Math.round(x(env.time[i])) + 0.5;
    const lo = Math.max(-1, Math.min(1, env.min[i] / lim));
    const hi = Math.max(-1, Math.min(1, env.max[i] / lim));
    ctx.moveTo(px, mid - hi * (traceH / 2));
    ctx.lineTo(px, mid - lo * (traceH / 2) + 0.5);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Pistes : annotations et détections.
  ctx.font = `10px ${css("--mono")}`;
  ctx.textBaseline = "middle";
  if (result.reference) {
    ctx.fillStyle = css("--ref");
    for (const e of result.reference.events) ctx.fillRect(x(e.start_s), refY, Math.max(2, x(e.end_s) - x(e.start_s)), 8);
  }
  ctx.fillStyle = css("--detect");
  for (const e of result.events) ctx.fillRect(x(e.start_s), detY, Math.max(2, x(e.end_s) - x(e.start_s)), 8);

  // Score lissé et seuil.
  const s = result.scores;
  const [lo, hi] = s.range;
  const sy = (v) => scoreTop + scoreH - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * scoreH;
  ctx.strokeStyle = css("--score");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < s.time.length; i++) {
    const px = x(s.time[i]);
    if (i === 0) ctx.moveTo(px, sy(s.value[i]));
    else ctx.lineTo(px, sy(s.value[i]));
  }
  ctx.stroke();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = css("--detect");
  ctx.beginPath();
  ctx.moveTo(0, Math.round(sy(s.threshold)) + 0.5);
  ctx.lineTo(width, Math.round(sy(s.threshold)) + 0.5);
  ctx.stroke();
  ctx.setLineDash([]);

  // Loupe.
  const vx = x(state.view.start);
  const vw = Math.max(3, x(state.view.start + state.view.width) - vx);
  ctx.fillStyle = css("--ink");
  ctx.globalAlpha = 0.07;
  ctx.fillRect(vx, 0, vw, height);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = css("--ink");
  ctx.lineWidth = 1;
  ctx.strokeRect(Math.round(vx) + 0.5, 0.5, Math.round(vw), height - 1);

  // Graduation en minutes.
  ctx.fillStyle = css("--ink-3");
  ctx.textBaseline = "top";
  const step = total > 1800 ? 300 : 120;
  for (let t = step; t < total; t += step) {
    ctx.fillRect(Math.round(x(t)), height - 4, 1, 4);
  }
}

function niceStep(raw) {
  const exp = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 5, 10]) if (m * exp >= raw) return m * exp;
  return 10 * exp;
}

function drawPaper() {
  const result = state.result;
  const canvas = $("paper");
  const { ctx, width, height } = sizeCanvas(canvas);
  const { start, width: span } = state.view;
  const left = 64;
  const plotW = width - left - 10;
  const traceTop = 14;
  const traceH = height - 120;
  const bandsTop = traceTop + traceH + 8;
  const scoreTop = bandsTop + 30;
  const scoreH = height - scoreTop - 22;
  const x = (t) => left + ((t - start) / span) * plotW;
  ctx.clearRect(0, 0, width, height);

  // Grille de papier : 0,2 s / 1 s, comme un tracé imprimé.
  const minor = span <= 30 ? 0.2 : 1;
  const major = span <= 30 ? 1 : 5;
  ctx.lineWidth = 1;
  for (const [stepT, color] of [[minor, css("--grid-minor")], [major, css("--grid-major")]]) {
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let t = Math.ceil(start / stepT) * stepT; t <= start + span + 1e-9; t += stepT) {
      const px = Math.round(x(t)) + 0.5;
      ctx.moveTo(px, traceTop);
      ctx.lineTo(px, traceTop + traceH);
    }
    const rows = stepT === minor ? 20 : 4;
    for (let i = 0; i <= rows; i++) {
      const py = Math.round(traceTop + (i / rows) * traceH) + 0.5;
      ctx.moveTo(left, py);
      ctx.lineTo(left + plotW, py);
    }
    ctx.stroke();
  }

  // Bandes des épisodes sur toute la hauteur du tracé.
  const visible = (e) => e.end_s >= start && e.start_s <= start + span;
  if (result.reference) {
    ctx.strokeStyle = css("--ref");
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (const e of result.reference.events.filter(visible)) {
      for (const edge of [e.start_s, e.end_s]) {
        ctx.moveTo(Math.round(x(edge)) + 0.5, traceTop);
        ctx.lineTo(Math.round(x(edge)) + 0.5, traceTop + traceH);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  result.events.forEach((e, i) => {
    if (!visible(e)) return;
    ctx.fillStyle = css("--detect-wash");
    ctx.fillRect(x(e.start_s), traceTop, x(e.end_s) - x(e.start_s), traceH);
    ctx.fillStyle = css("--detect");
    ctx.fillRect(x(e.start_s), traceTop, x(e.end_s) - x(e.start_s), 2);
    if (i === state.current) {
      ctx.fillRect(Math.round(x(e.start_s)), traceTop, 1, traceH);
      ctx.fillRect(Math.round(x(e.end_s)), traceTop, 1, traceH);
    }
  });

  // Signal.
  const scale = result.amplitude_scale * 12 / state.gain;
  const mid = traceTop + traceH / 2;
  const toY = (v) => mid - (v / scale) * (traceH / 2);
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, traceTop, plotW, traceH);
  ctx.clip();
  if (state.trace && state.trace.time.length) {
    const t = state.trace.time;
    const v = state.trace.signal;
    ctx.strokeStyle = css("--signal");
    ctx.lineWidth = span <= 20 ? 1.1 : 0.9;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) {
      const px = x(t[i]);
      if (i === 0) ctx.moveTo(px, toY(v[i]));
      else ctx.lineTo(px, toY(v[i]));
    }
    ctx.stroke();
  }
  ctx.restore();

  // Barre d'échelle d'amplitude.
  const barValue = niceStep(scale / 2);
  const barPx = (barValue / scale) * (traceH / 2);
  ctx.strokeStyle = css("--ink");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(left - 14, mid - barPx / 2);
  ctx.lineTo(left - 14, mid + barPx / 2);
  ctx.moveTo(left - 18, mid - barPx / 2);
  ctx.lineTo(left - 10, mid - barPx / 2);
  ctx.moveTo(left - 18, mid + barPx / 2);
  ctx.lineTo(left - 10, mid + barPx / 2);
  ctx.stroke();
  ctx.fillStyle = css("--ink-2");
  ctx.font = `10.5px ${css("--mono")}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(nf0.format(barValue), left - 22, mid);
  ctx.fillText("u.f.", left - 22, mid + 13);

  // Pistes nommées sous le tracé.
  ctx.font = `10.5px ${css("--mono")}`;
  ctx.fillStyle = css("--ink-3");
  ctx.fillText("détect.", left - 8, bandsTop + 5);
  ctx.fillStyle = css("--detect");
  for (const e of result.events.filter(visible)) ctx.fillRect(x(e.start_s), bandsTop + 1, x(e.end_s) - x(e.start_s), 8);
  if (result.reference) {
    ctx.fillStyle = css("--ink-3");
    ctx.fillText("annot.", left - 8, bandsTop + 18);
    ctx.fillStyle = css("--ref");
    for (const e of result.reference.events.filter(visible)) ctx.fillRect(x(e.start_s), bandsTop + 14, x(e.end_s) - x(e.start_s), 8);
  }

  // Score par fenêtre, centré sur chaque fenêtre de 2 s.
  const s = result.scores;
  const [lo, hi] = s.range;
  const sy = (v) => scoreTop + scoreH - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * scoreH;
  ctx.fillStyle = css("--ink-3");
  ctx.fillText("score", left - 8, scoreTop + scoreH / 2);
  ctx.strokeStyle = css("--rule");
  ctx.lineWidth = 1;
  ctx.strokeRect(left + 0.5, scoreTop + 0.5, plotW, scoreH);
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, scoreTop, plotW, scoreH);
  ctx.clip();
  ctx.strokeStyle = css("--score");
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  let begun = false;
  for (let i = 0; i < s.time.length; i++) {
    const t = s.time[i];
    if (t < start - 3 || t > start + span + 3) continue;
    const px = x(t);
    if (!begun) { ctx.moveTo(px, sy(s.value[i])); begun = true; } else ctx.lineTo(px, sy(s.value[i]));
  }
  ctx.stroke();
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = css("--detect");
  ctx.beginPath();
  ctx.moveTo(left, Math.round(sy(s.threshold)) + 0.5);
  ctx.lineTo(left + plotW, Math.round(sy(s.threshold)) + 0.5);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // Axe du temps.
  ctx.fillStyle = css("--ink-2");
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const labelStep = [1, 2, 5, 10, 15, 30].find((step) => (step / span) * plotW >= 52) || 60;
  for (let t = Math.ceil(start / labelStep) * labelStep; t <= start + span + 1e-9; t += labelStep) {
    const label = clock(t);
    const half = ctx.measureText(label).width / 2;
    ctx.fillText(label, Math.min(width - half - 2, Math.max(left, x(t))), height - 16);
  }
}

/* ---------- interactions ---------- */

function pan(seconds) {
  if (!state.result) return;
  focusOn(state.view.start + state.view.width / 2 + seconds);
}

function bindOverview() {
  const canvas = $("overview");
  let dragging = false;
  const move = (event) => {
    const rect = canvas.getBoundingClientRect();
    const t = ((event.clientX - rect.left) / rect.width) * state.result.analyzed_duration_s;
    focusOn(t);
  };
  canvas.addEventListener("pointerdown", (event) => {
    if (!state.result) return;
    dragging = true;
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("dragging");
    move(event);
  });
  canvas.addEventListener("pointermove", (event) => dragging && move(event));
  canvas.addEventListener("pointerup", () => {
    dragging = false;
    canvas.classList.remove("dragging");
  });
}

function bindPaper() {
  const canvas = $("paper");
  let origin = null;
  canvas.addEventListener("pointerdown", (event) => {
    origin = { x: event.clientX, start: state.view.start };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!origin) return;
    const plotW = canvas.clientWidth - 74;
    const dt = ((origin.x - event.clientX) / plotW) * state.view.width;
    const total = state.result.analyzed_duration_s;
    state.view.start = Math.max(0, Math.min(total - state.view.width, origin.start + dt));
    draw();
  });
  canvas.addEventListener("pointerup", () => { origin = null; });
}

function bindControls() {
  const input = $("file-input");
  const slot = $("slot");
  input.addEventListener("change", () => chooseFile(input.files[0]));
  for (const type of ["dragenter", "dragover"]) {
    slot.addEventListener(type, (event) => { event.preventDefault(); slot.classList.add("dragging"); });
  }
  for (const type of ["dragleave", "drop"]) {
    slot.addEventListener(type, (event) => { event.preventDefault(); slot.classList.remove("dragging"); });
  }
  slot.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));
  $("run").addEventListener("click", analyzeFile);
  const search = $("record-search");
  search.addEventListener("input", filterCatalogue);
  search.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    const first = [...$("record-list").children].find((row) => !row.hidden && !row.classList.contains("unavailable"));
    openRecord(first ? first.firstChild.textContent : search.value);
  });

  for (const button of $("widths").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      const center = state.view.start + state.view.width / 2;
      state.view.width = Number(button.dataset.w);
      $("widths").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      focusOn(center);
    });
  }
  $("gain-up").addEventListener("click", () => { state.gain *= 1.5; drawPaper(); });
  $("gain-down").addEventListener("click", () => { state.gain /= 1.5; drawPaper(); });
  $("prev").addEventListener("click", () => setCurrent(state.current - 1));
  $("next").addEventListener("click", () => setCurrent(state.current + 1));
  $("new-file").addEventListener("click", () => {
    $("reading").hidden = true;
    $("intake").hidden = false;
    state.result = null;
    history.replaceState(null, "", location.pathname);
  });
  $("export-csv").addEventListener("click", exportCsv);
  $("export-json").addEventListener("click", exportJson);

  document.addEventListener("keydown", (event) => {
    if (!state.result || event.target.closest("input, select, textarea")) return;
    const keys = {
      ArrowLeft: () => pan(-state.view.width / 2),
      ArrowRight: () => pan(state.view.width / 2),
      "[": () => setCurrent(state.current - 1),
      "]": () => setCurrent(state.current + 1),
      "+": () => { state.gain *= 1.5; drawPaper(); },
      "=": () => { state.gain *= 1.5; drawPaper(); },
      "-": () => { state.gain /= 1.5; drawPaper(); },
    };
    if (keys[event.key]) {
      event.preventDefault();
      keys[event.key]();
    }
  });
  window.addEventListener("resize", () => draw());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => draw());
}

/* ---------- exports ---------- */

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = el("a", { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function baseName() {
  return (state.result.record_id || state.result.filename).replace(/\.[^.]+$/, "");
}

function exportCsv() {
  const header = "episode,debut_s,fin_s,duree_s,score_moyen,score_pic,crise_annotee,iou";
  const lines = state.result.events.map((e) =>
    [e.id, e.start_s, e.end_s, e.duration_s, e.mean_score.toFixed(4), e.peak_score.toFixed(4),
      e.match ? e.match.reference : "", e.match ? e.match.iou.toFixed(3) : ""].join(","));
  download(`${baseName()}_episodes.csv`, [header, ...lines].join("\n"), "text/csv");
}

function exportJson() {
  const { overview, scores, ...rest } = state.result;
  download(`${baseName()}_analyse.json`, JSON.stringify(rest, null, 2), "application/json");
}

bindControls();
bindOverview();
bindPaper();
loadModel();
