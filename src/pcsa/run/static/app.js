const el = (id) => document.getElementById(id);
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const drawn = {};
// A battle number picked in the strip or the list (or ?battle=N in the address), or null to follow the newest battle.
let pinned = /^\d+$/.test(new URLSearchParams(location.search).get("battle") || "") ? Number(new URLSearchParams(location.search).get("battle")) : null;

/* ---------------------------------------------------------------- small helpers */
function node(tag, className, content) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (content !== undefined && content !== null) item.textContent = String(content);
  return item;
}

function svg(tag, attrs = {}) {
  const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => item.setAttribute(k, v));
  return item;
}

function tip(target, message) {
  if (message) target.dataset.tip = message;
  else delete target.dataset.tip;
  return target;
}

/** Set text; a changed value gets a short settle animation. */
function text(id, value, fallback = "--") {
  const target = typeof id === "string" ? el(id) : id;
  const next = value === null || value === undefined || value === "" ? fallback : String(value);
  if (target.textContent === next) return;
  target.textContent = next;
  target.classList.remove("changed");
  void target.offsetWidth; // restart the animation
  target.classList.add("changed");
}

function pct(p) {
  if (!isNum(p)) return "--";
  if (p > 0 && p < 0.005) return "<1%";
  return `${Math.round(p * 100)}%`;
}

const signed = (v, digits = 1) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`;
const fixed = (v, digits = 2) => (isNum(v) ? v.toFixed(digits) : "--");
const seconds = (v) => (isNum(v) ? `${v.toFixed(1)} s` : null);
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const title = (s) => (s ? String(s).charAt(0).toUpperCase() + String(s).slice(1) : "");

function timeLabel(value) {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "--" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function ago(s) {
  if (!isNum(s)) return "No updates";
  if (s < 2) return "updated just now";
  if (s < 60) return `updated ${Math.round(s)}s ago`;
  return s < 3600 ? `updated ${Math.floor(s / 60)}m ago` : `updated ${Math.floor(s / 3600)}h ago`;
}

/** Show a tile's body, or its designed empty state with `caption`. */
function tile(id, filled, caption, captionId) {
  el(id).dataset.empty = filled ? "false" : "true";
  if (!filled && caption && captionId) el(captionId).textContent = caption;
}

/** True when the slice of state a renderer reads changed since it last drew. */
function changed(name, value) {
  const key = JSON.stringify(value ?? null);
  if (drawn[name] === key) return false;
  drawn[name] = key;
  return true;
}

/* ---------------------------------------------------------------- shared marks */
const SIDE_TIP = { you: "Your Pokémon", opp: "The opponent's Pokémon" };

function sideTag(side) {
  return tip(node("span", `side ${side}`, side === "you" ? "YOU" : "OPP"), SIDE_TIP[side]);
}

function mon(name, side) {
  const wrap = node("span", "mon");
  wrap.append(sideTag(side), node("span", "nm", name));
  return wrap;
}

function hpBar(fraction, side) {
  const bar = node("span", `hpbar${side === "opp" ? " is-opp" : ""}`);
  const fill = node("i", isNum(fraction) && fraction <= 0.25 ? "is-crit" : isNum(fraction) && fraction <= 0.5 ? "is-low" : "");
  fill.style.width = `${(isNum(fraction) ? Math.max(0, Math.min(1, fraction)) : 0) * 100}%`;
  bar.append(fill);
  return bar;
}

const DEGRADE_TIP = {
  full: "Full: decided on the tracked battle, with a complete battle log",
  tracked: "Tracked: decided on the tracked battle, though a part of the battle log was shaky",
  screen: "Screen: the battle log could not be used; decided from what the screen shows",
  emergency: "Emergency: out of time; the first legal action was sent",
};

function degradeBadge(level) {
  if (!level) return node("span");
  const badge = node("span", "badge", level);
  badge.dataset.degrade = level;
  return tip(badge, DEGRADE_TIP[level] || `Degrade level: ${level}`);
}

function fieldChip(label, count, message) {
  const chip = node("span", "chipf");
  chip.append(node("span", "", label));
  if (count !== null && count !== undefined) chip.append(node("b", "", count));
  return tip(chip, message);
}

function setMeter(id, value, shown) {
  const meter = el(id);
  meter.hidden = !isNum(value);
  if (!isNum(value)) return;
  meter.querySelector(".val").textContent = shown;
  meter.querySelector(".fill").style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
}

/* The critic's value: expected return, about −1.3 (a lost battle) to +1.3 (a won one); above 0 it expects to win. */
const roughWin = (v) => `≈ ${Math.round(Math.max(0, Math.min(1, (v + 1) / 2)) * 100)}%`;
const valueTone = (v) => (v > 0.02 ? "up" : v < -0.02 ? "down" : "");

function setValueMeter(value) {
  const meter = el("m-v");
  if (!meter) return;
  meter.hidden = !isNum(value);
  if (!isNum(value)) return;
  const v = Math.max(-1, Math.min(1, value));
  meter.classList.toggle("is-neg", v < 0);
  const val = meter.querySelector(".val");
  val.replaceChildren(document.createTextNode(signed(value, 2)), node("small", "", `${roughWin(value)} win`));
  const fill = meter.querySelector(".fill");
  fill.style.left = `${50 + Math.min(0, v) * 50}%`;
  fill.style.width = `${Math.abs(v) * 50}%`;
}

/* ---------------------------------------------------------------- tooltips (hover and keyboard focus) */
const tipCard = el("tip");
let tipTarget = null;

function showTip(target) {
  tipTarget?.removeAttribute("aria-describedby");
  tipTarget = target;
  tipCard.textContent = target.dataset.tip;
  tipCard.hidden = false;
  target.setAttribute("aria-describedby", "tip");
  const r = target.getBoundingClientRect();
  const w = tipCard.offsetWidth;
  const h = tipCard.offsetHeight;
  const left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2));
  const below = r.bottom + 8 + h < window.innerHeight;
  tipCard.style.left = `${left}px`;
  tipCard.style.top = `${below ? r.bottom + 8 : r.top - h - 8}px`;
}

function hideTip() {
  tipTarget?.removeAttribute("aria-describedby");
  tipTarget = null;
  tipCard.hidden = true;
}

function setupTips() {
  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest?.("[data-tip]");
    if (target === tipTarget) return;
    if (target) showTip(target);
    else hideTip();
  });
  document.addEventListener("focusin", (event) => {
    const target = event.target.closest?.("[data-tip]");
    if (target) showTip(target);
    else hideTip();
  });
  document.addEventListener("focusout", hideTip);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") hideTip(); });
  window.addEventListener("scroll", hideTip, true);
}

/** Every element with a tip can take keyboard focus (natively focusable controls already can). */
function armTips() {
  document.querySelectorAll("[data-tip]:not([tabindex])").forEach((item) => {
    if (!["BUTTON", "SUMMARY", "A", "INPUT"].includes(item.tagName)) item.tabIndex = 0;
  });
}

/* ---------------------------------------------------------------- top bar */
const OUTCOME = {
  win: ["is-win", "WIN", "W"], loss: ["is-loss", "LOSS", "L"], draw: ["is-draw", "DRAW", "D"], opponent_forfeit: ["is-win", "FORFEIT", "F"],
};

function outcomeOf(battle) {
  if (!battle) return null;
  return OUTCOME[battle.outcome] || (battle.outcome ? ["is-draw", String(battle.outcome).toUpperCase(), "?"] : ["is-live", "PLAYING", "…"]);
}

function renderPlayer(view) {
  const params = view?.params || [];
  el("params").hidden = params.length === 0;
  if (!changed("params", view)) return;
  el("params-label").textContent = view?.summary || plural(params.length, "setting");
  el("params-list").replaceChildren(...params.map((p) => {
    const li = node("li");
    li.append(node("span", "", p.key), node("b", "", p.value));
    return p.full !== p.value ? tip(li, p.full) : li;
  }));
}

function renderStatus(state) {
  const pill = el("run-status");
  const status = String(state.status || "WAITING");
  pill.dataset.state = status.toLowerCase();
  pill.lastElementChild.textContent = status;
  text("run-name", state.run_name, "No run found");
  text("top-battle", state.battle?.index);
  text("top-turn", state.turn);
  text("top-screen", state.screen);
  text("top-player", state.player);
  renderPlayer(state.player_view);
  const r = state.session_record || {};
  const extra = (r.draws ? ` ${r.draws}D` : "") + (r.forfeits ? ` ${r.forfeits}F` : "");
  text("session-record", `${r.wins || 0}W ${r.losses || 0}L${extra}`);
  const rating = state.rating;
  el("top-rating-fact").hidden = !isNum(rating?.rating);
  if (isNum(rating?.rating)) text("top-rating", `${Math.round(rating.rating)}${isNum(rating.delta) && rating.delta ? ` (${signed(rating.delta, 0)})` : ""}`);
  el("last-updated").textContent = ago(state.age_s);
  if (changed("outcome", [state.battle?.index, state.battle?.outcome])) {
    const outcome = outcomeOf(state.battle);
    el("top-outcome").replaceChildren(...(outcome ? [node("span", `result ${outcome[0]}`, outcome[1])] : []));
  }
}

/* ---------------------------------------------------------------- battle-state box: the two actives, field, sides */
function fighter(mon_, side, switching) {
  const box = node("div", `fighter ${side}`);
  if (!mon_?.name) {
    box.append(sideTag(side), node("span", "nm is-none", switching ? "switching in after a faint" : side === "you" ? "no active yet" : "not seen yet"));
    return box;
  }
  const name = node("span", "nm", mon_.name);
  const hp = node("span", "hptext", mon_.hp || "?");
  box.append(sideTag(side), name, hp, hpBar(mon_.fraction, side));
  const extras = [mon_.status ? mon_.status.toUpperCase() : null, mon_.boosts || null].filter(Boolean);
  if (extras.length) name.append(node("span", "status", extras.join(" ")));
  return tip(box, `${side === "you" ? "Your" : "Their"} ${mon_.name}: HP ${mon_.hp || "unknown"}${mon_.status ? `, ${mon_.status}` : ""}${mon_.boosts ? `, stat stages ${mon_.boosts}` : ""}`);
}

function sideChips(conditions) {
  return conditions.map((c) => {
    if (c.layers) return fieldChip(c.name, c.layers > 1 ? `×${c.layers}` : null, `${c.name}: ${plural(c.layers, "layer")} (stays until removed)`);
    if (c.left) return fieldChip(c.name, c.left, `${c.name}: about ${plural(c.left, "turn")} left (default duration; Light Clay extends screens)`);
    return fieldChip(c.name, null, `${c.name}: stays until removed`);
  });
}

function renderStateBox(state) {
  const field = state.field;
  if (!changed("statebox", [state.active, state.opponent, field, Boolean(state.battle), state.decision?.kind])) return;
  const match = el("sb-match");
  if (state.active?.name || state.opponent?.name) {
    match.replaceChildren(fighter(state.active, "you", state.decision?.kind === "forced"), node("span", "vs", "vs"), fighter(state.opponent, "opp"));
  } else {
    match.replaceChildren(node("span", "sb-none", state.battle ? "Team preview: no actives yet" : "No battle yet"));
  }
  const chips = [];
  if (field?.weather) chips.push(fieldChip(field.weather.name, field.weather.left, `Weather: about ${plural(field.weather.left, "turn")} left (5-turn default; rocks like Heat Rock extend it)`));
  (field?.fields || []).forEach((f) => chips.push(fieldChip(f.name, f.left, `${f.name}: about ${plural(f.left, "turn")} left (5-turn default)`)));
  const fieldCell = el("sb-field");
  fieldCell.replaceChildren(node("span", "sb-k", "Field"), ...(chips.length ? chips : [node("span", "sb-none", field ? "clear" : "--")]));
  tip(fieldCell, chips.length ? null : "Weather, terrain and rooms from the battle log (none active)");
  [["sb-ours", "you", field?.ours], ["sb-theirs", "opp", field?.theirs]].forEach(([id, side, conditions]) => {
    const list = sideChips(conditions || []);
    el(id).replaceChildren(sideTag(side), ...(list.length ? list : [node("span", "sb-none", field ? "no hazards or screens" : "--")]));
  });
}

/* ---------------------------------------------------------------- latest decision */
const KIND = { forced: "forced switch", action: "turn action", team: "team preview" };
const orderLabel = (o) => (o.kind === "switch" ? `⇄ ${o.label}` : o.label);

/** The notes a player wrote, as chips: its rule, its own numbers, and the timing the run measured. */
function noteChips(notes, timing) {
  const chips = [];
  if (notes?.rule) chips.push(fieldChip("rule", notes.rule, `The rule that decided: ${notes.rule}`));
  (notes?.rest || []).forEach((n) => chips.push(fieldChip(n.key, n.value, `${n.key}: ${n.value} (the player's own note)`)));
  if (isNum(notes?.ms)) chips.push(fieldChip("decided in", `${notes.ms.toFixed(1)} ms`, "Time the player took to decide"));
  if (seconds(timing?.log_s)) chips.push(fieldChip("log read", seconds(timing.log_s), "Time spent reading the battle log before this decision"));
  if (seconds(timing?.tap_s)) chips.push(fieldChip("turn", seconds(timing.tap_s), "From the first capture of the turn to the last tap"));
  return chips;
}

function legalChips(decision) {
  const chips = (decision.legal || []).map((o) => {
    const chosen = o.order === decision.order;
    const chip = node("span", `opt${chosen ? " is-chosen" : ""}${o.kind === "switch" ? " is-switch" : ""}`);
    chip.append(node("span", "n", orderLabel(o)));
    if (chosen) chip.append(node("span", "p", "✓"));
    return tip(chip, `${o.kind === "switch" ? `Switch to ${o.label}` : o.label}: ${chosen ? "the action sent" : "legal, not chosen"} (${o.order})`);
  });
  (decision.dropped || []).forEach((o) => {
    const chip = node("span", "opt is-dropped");
    chip.append(node("span", "n", orderLabel(o)));
    chips.push(tip(chip, `${o.label}: dropped from the legal actions before deciding (${o.order})`));
  });
  return chips;
}

function renderDecision(state) {
  const decision = state.decision;
  const waiting = state.preview ? "Team picked; the first battle decision comes next" : state.battle ? "Waiting for this battle's first decision" : "Waiting for the first decision";
  tile("t-decision", Boolean(decision), waiting, "decision-empty");
  if (!decision || !changed("decision", [decision, state.active?.name])) return;
  const notes = decision.notes;
  const label = KIND[decision.kind] || "decision";
  const ours = state.active?.name;
  el("decision-kind").replaceChildren(...(ours ? [mon(ours, "you"), node("span", "", ` · ${label}`)] : [node("span", "", title(label))]));
  text("decision-order", orderLabel({ kind: decision.order_kind, label: decision.label || decision.order }), "Decision recorded");
  tip(el("decision-order"), `The action the agent sent to the game (${decision.order})`);
  el("decision-meta").textContent = `Turn ${decision.turn ?? "--"} · ${timeLabel(decision.ts)}`;
  tip(el("decision-meta"), `Decision turn and when it was recorded${decision.reason ? ` (${decision.reason})` : ""}`);
  el("decision-badge").replaceChildren(degradeBadge(decision.degrade));
  el("fallback").hidden = !notes?.fallback;
  el("fallback-why").textContent = notes?.why || "the network did not decide; the fallback player did";
  el("t-decision").classList.toggle("is-fallback", Boolean(notes?.fallback));
  el("meters").hidden = !(isNum(notes?.p) || isNum(notes?.h) || isNum(notes?.v));
  setMeter("m-p", notes?.p, pct(notes?.p));
  setMeter("m-h", notes?.h, fixed(notes?.h));
  setValueMeter(notes?.v);
  const legal = decision.legal || [];
  el("legal-meta").textContent = `${legal.length} legal${decision.dropped?.length ? ` · ${decision.dropped.length} dropped` : ""}`;
  el("legal").replaceChildren(...legalChips(decision));
  el("decision-notes").replaceChildren(...noteChips(notes, decision.timing), ...(decision.warnings || []).map((w) => tip(node("span", "flag flag-warn", w), `Warning on this decision: ${w}`)));
}

/* ---------------------------------------------------------------- frame: the newest one the run saved */
let frameShown = null;

function renderFrame(state) {
  const frame = el("t-frame");
  const id = state.frame_id;
  if (!id) {
    frame.dataset.empty = "true";
    frameShown = null;
    return;
  }
  if (id === frameShown) return;
  frameShown = id;
  const next = new Image(); // decode off-screen, then swap: no flicker between frames
  next.onload = () => { el("phone-frame").src = next.src; frame.dataset.empty = "false"; };
  next.onerror = () => { frameShown = null; };
  next.src = `/api/frame?v=${encodeURIComponent(id)}`;
}

/* ---------------------------------------------------------------- what are they running? */
function foeCard(f) {
  const card = node("div", `foe${f.active ? " is-active" : ""}${f.fainted ? " is-fainted" : ""}`);
  const top = node("div", "foe-top");
  const name = mon(f.name, "opp");
  name.classList.add("foe-name");
  if (f.status && !f.fainted) name.append(tip(node("span", "status", f.status), "Status condition"));
  top.append(name, tip(hpBar(f.fraction, "opp"), `HP ${f.hp || "unknown"} (the game shows the opponent's HP as a percentage)`), node("span", "hptext", f.fainted ? "fainted" : f.hp || "?"));
  const facts = [f.types, f.item ? `@ ${f.item}` : null, f.ability, f.boosts].filter(Boolean);
  const sub = tip(node("div", "member-sub", facts.join(" · ") || "nothing revealed yet"), "Types, and the item, ability and stat stages revealed so far");
  const chips = node("div", "chips");
  (f.moves || []).forEach((m) => {
    const chip = tip(node("span", "chip is-known"), `${m}: revealed in battle`);
    chip.append(node("span", "n", m));
    chips.append(chip);
  });
  card.append(top, sub, chips);
  if (f.active) tip(card, "On the field now");
  return card;
}

function renderFoes(state) {
  const foes = state.foes || [];
  const unseen = state.unseen || [];
  tile("t-foes", foes.length > 0 || unseen.length > 0, state.battle?.opponent?.length ? `Preview: ${state.battle.opponent.join(", ")}` : "Shown once the battle is tracked and an opponent is seen", "foes-empty");
  if (!changed("foes", [foes, unseen])) return;
  el("foes-source").textContent = foes.length ? `${foes.length} seen · revealed in battle` : "";
  el("foes").replaceChildren(...[...foes].sort((a, b) => (b.active - a.active) || (a.fainted - b.fainted)).map(foeCard));
  el("unseen").hidden = unseen.length === 0;
  el("unseen").replaceChildren(node("span", "", "In their preview, not seen yet"), ...unseen.map((n) => node("b", "", n)));
}

/* ---------------------------------------------------------------- how is it going? record, rating, battles */
let spark = { key: "", width: 0, height: 0 };

function drawSpark(series) {
  const box = el("spark");
  spark = { key: JSON.stringify(series), width: box.clientWidth, height: box.clientHeight };
  const w = Math.max(160, box.clientWidth);
  const h = Math.max(56, box.clientHeight);
  const [top, bottom, left, right] = [16, 6, 34, 8];
  const values = series.map((s) => s.rating);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const x = (i) => left + (i * (w - left - right)) / (series.length - 1);
  const y = (v) => top + (1 - (v - lo) / (hi - lo || 1)) * (h - top - bottom);
  const chart = svg("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": "Rating over the run" });
  [hi, lo].forEach((v) => {
    chart.append(svg("line", { class: "grid", x1: left, x2: w - right, y1: y(v), y2: y(v) }));
    const label = svg("text", { class: "axis", x: left - 6, y: y(v) + 3, "text-anchor": "end" });
    label.textContent = Math.round(v);
    chart.append(label);
  });
  const points = series.map((s, i) => [x(i), y(s.rating)]);
  chart.append(svg("path", { class: "line", d: points.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ") }));
  if (series.length <= 40) points.forEach(([px, py]) => chart.append(svg("circle", { class: "pt", cx: px, cy: py, r: 2.2 })));
  const halo = svg("circle", { class: "halo", r: 7 });
  const dot = svg("circle", { class: "dot", r: 3.6 });
  chart.append(halo, dot);
  const readout = node("span", "spark-read");
  const show = (i) => {
    const [px, py] = points[i];
    [dot, halo].forEach((c) => { c.setAttribute("cx", px); c.setAttribute("cy", py); });
    readout.textContent = `${series[i].battle ? `after battle ${series[i].battle}` : "at the start"} · ${series[i].rating.toFixed(1)}`;
  };
  chart.addEventListener("mousemove", (event) => {
    const rect = chart.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * w;
    show(Math.max(0, Math.min(series.length - 1, Math.round((px - left) / ((w - left - right) / (series.length - 1))))));
  });
  chart.addEventListener("mouseleave", () => show(series.length - 1));
  show(series.length - 1);
  box.replaceChildren(readout, chart);
}

/** The value at each decision of the battle shown, the team pick first; decisions without one are left out. */
function valueSeries(state) {
  const series = [];
  if (!(state.decisions || []).some((d) => isNum(d.notes?.v))) return series; // a run recorded before decisions carried a value
  const pick = state.preview?.notes?.v;
  if (isNum(pick)) series.push({ label: "team pick", v: pick });
  (state.decisions || []).forEach((d) => {
    if (isNum(d.notes?.v)) series.push({ label: `turn ${d.turn ?? "?"}${d.kind === "forced" ? " switch-in" : ""}`, v: d.notes.v });
  });
  return series;
}

let vspark = { key: "", width: 0, height: 0 };

function drawValues(series) {
  const box = el("vspark");
  vspark = { key: JSON.stringify(series), width: box.clientWidth, height: box.clientHeight };
  const w = Math.max(160, box.clientWidth);
  const h = Math.max(50, box.clientHeight);
  const [top, bottom, left, right] = [6, 6, 34, 8];
  const reach = Math.max(1, ...series.map((s) => Math.abs(s.v)));
  const x = (i) => (series.length > 1 ? left + (i * (w - left - right)) / (series.length - 1) : (left + w - right) / 2);
  const y = (v) => top + (1 - (v + reach) / (2 * reach)) * (h - top - bottom);
  const chart = svg("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": "Value at each decision of this battle" });
  [1, -1].forEach((v) => {
    chart.append(svg("line", { class: "grid", x1: left, x2: w - right, y1: y(v), y2: y(v) }));
    const label = svg("text", { class: "axis", x: left - 6, y: y(v) + 3, "text-anchor": "end" });
    label.textContent = v > 0 ? "+1" : "−1";
    chart.append(label);
  });
  chart.append(svg("line", { class: "zero", x1: left, x2: w - right, y1: y(0), y2: y(0) }));
  const zero = svg("text", { class: "axis", x: left - 6, y: y(0) + 3, "text-anchor": "end" });
  zero.textContent = "0";
  chart.append(zero);
  const points = series.map((s, i) => [x(i), y(s.v)]);
  if (points.length > 1) chart.append(svg("path", { class: "line", d: points.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ") }));
  points.forEach(([px, py], i) => chart.append(svg("circle", { class: `pt ${valueTone(series[i].v)}`, cx: px, cy: py, r: 2.4 })));
  const halo = svg("circle", { class: "halo", r: 7 });
  const dot = svg("circle", { class: "dot", r: 3.6 });
  chart.append(halo, dot);
  const readout = node("span", "spark-read");
  const show = (i) => {
    const [px, py] = points[i];
    [dot, halo].forEach((c) => { c.setAttribute("cx", px); c.setAttribute("cy", py); });
    readout.textContent = `${series[i].label} · ${signed(series[i].v, 2)} (${roughWin(series[i].v)})`;
  };
  chart.addEventListener("mousemove", (event) => {
    if (points.length < 2) return;
    const rect = chart.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * w;
    show(Math.max(0, Math.min(series.length - 1, Math.round((px - left) / ((w - left - right) / (series.length - 1))))));
  });
  chart.addEventListener("mouseleave", () => show(series.length - 1));
  show(series.length - 1);
  box.replaceChildren(readout, chart);
}

function renderValues(state) {
  const holder = el("vchart");
  if (!holder) return;
  const series = valueSeries(state);
  holder.hidden = series.length === 0;
  if (!series.length) return;
  el("vchart-title").textContent = `Value · battle ${state.battle?.index ?? "--"}`;
  const box = el("vspark");
  if (JSON.stringify(series) !== vspark.key || box.clientWidth !== vspark.width || box.clientHeight !== vspark.height) drawValues(series);
}

function pin(index) {
  pinned = index;
  poll();
}

function battleFacts(b) {
  return [`Battle ${b.index}`, b.opponent?.length ? `vs ${b.opponent.join(", ")}` : null, b.picks?.length ? `brought ${b.picks.join(", ")}` : null,
    b.turn ? plural(b.turn, "turn") : null, plural(b.decisions || 0, "decision"), b.disconnects ? plural(b.disconnects, "disconnect") : null,
    isNum(b.rating_after) && isNum(b.rating_before) ? `rating ${b.rating_before.toFixed(1)} → ${b.rating_after.toFixed(1)}` : null,
    b.outcome_source ? `result from the ${b.outcome_source}` : null].filter(Boolean).join(" · ");
}

function battleRow(b, viewed) {
  const row = node("button", `battle${b.index === viewed ? " is-viewed" : ""}`);
  row.type = "button";
  const outcome = outcomeOf(b);
  const delta = isNum(b.rating_after) && isNum(b.rating_before) ? b.rating_after - b.rating_before : null;
  const num = node("span", `num${isNum(delta) ? (delta > 0 ? " up" : delta < 0 ? " down" : "") : ""}`, isNum(delta) ? signed(delta) : "");
  row.append(node("b", "", `#${b.index}`), node("span", `result ${outcome[0]}`, outcome[1]), node("span", "vs", (b.opponent || []).join(", ") || "opponent not seen yet"), num);
  tip(row, `${battleFacts(b)}. Click to view this battle.`);
  row.addEventListener("click", () => pin(b.index));
  return row;
}

function stripCell(b, viewed) {
  const outcome = outcomeOf(b);
  const cell = node("button", `cell ${outcome[0]}${b.index === viewed ? " is-viewed" : ""}`, outcome[2]);
  cell.type = "button";
  cell.setAttribute("aria-label", `Battle ${b.index}: ${outcome[1].toLowerCase()}`);
  tip(cell, `${battleFacts(b)}. Click to view this battle.`);
  cell.addEventListener("click", () => pin(b.index));
  return cell;
}

function renderRun(state) {
  const battles = state.battles || [];
  const rating = state.rating;
  tile("t-run", battles.length > 0 || isNum(rating?.rating));
  const r = state.session_record || {};
  const record = `${r.wins || 0}W ${r.losses || 0}L`;
  text("run-number", isNum(rating?.rating) ? Math.round(rating.rating) : record);
  el("run-number-label").textContent = isNum(rating?.rating) ? [rating.tier, rating.rank ? `rank ${rating.rank}` : null].filter(Boolean).join(" · ") || "rating" : "record";
  const delta = el("run-delta");
  delta.hidden = !(isNum(rating?.delta) && rating.delta !== 0);
  delta.textContent = isNum(rating?.delta) ? `${signed(rating.delta)} this run` : "";
  delta.className = `delta ${rating?.delta > 0 ? "up" : "down"}`;
  el("run-meta").textContent = [record, plural(battles.length, "battle"), state.disconnects ? plural(state.disconnects, "disconnect") : null].filter(Boolean).join(" · ");
  el("follow").hidden = pinned === null;
  renderValues(state);
  const series = rating?.series || [];
  const box = el("spark");
  box.hidden = series.length < 2;
  if (series.length >= 2 && (JSON.stringify(series) !== spark.key || box.clientWidth !== spark.width || box.clientHeight !== spark.height)) drawSpark(series);
  if (!changed("battles", [battles, state.battle?.index])) return;
  el("strip").replaceChildren(...battles.map((b) => stripCell(b, state.battle?.index)));
  el("battles").replaceChildren(...[...battles].reverse().map((b) => battleRow(b, state.battle?.index)));
}

/* ---------------------------------------------------------------- your team */
function memberCard(m) {
  const card = node("div", `member${m.active ? " is-active" : ""}${m.fainted ? " is-fainted" : ""}`);
  const top = node("div", "member-top");
  const name = mon(m.name, "you");
  if (m.status) name.append(tip(node("span", "status", m.status.toUpperCase()), "Status condition"));
  top.append(name, tip(hpBar(m.fraction, "you"), `HP ${m.hp} (${isNum(m.fraction) ? pct(m.fraction) : "--"}), from the game's request`), node("span", "hptext", m.fainted ? "fainted" : m.hp));
  const sub = node("div", "member-sub");
  sub.append(node("b", "", m.item || "no item"), node("span", "", ` · ${m.ability || "?"}${m.boosts ? ` · ${m.boosts}` : ""}`));
  tip(sub, `Item ${m.item || "none"}, ability ${m.ability || "unknown"} (your team sheet)${m.types ? `; ${m.types}` : ""}${m.boosts ? `; stat stages ${m.boosts}` : ""}`);
  const chips = node("div", "chips");
  (m.moves || []).forEach((mv) => {
    const chip = node("span", "chip");
    chip.append(node("span", "n", mv.name));
    if (Array.isArray(mv.pp)) chip.append(node("span", "p", `${mv.pp[0]}/${mv.pp[1]}`));
    tip(chip, Array.isArray(mv.pp) ? `${mv.name}: ${mv.pp[0]} of ${mv.pp[1]} PP left` : `${mv.name} (PP is known for the active Pokémon only)`);
    chips.append(chip);
  });
  card.append(top, sub, chips);
  if (m.active) tip(card, "On the field now");
  return card;
}

function renderTeam(state) {
  const team = state.your_team || [];
  tile("t-team", team.length > 0);
  el("move-capture").textContent = state.move_capture && team.length ? `as of ${timeLabel(state.move_capture)}` : "";
  if (changed("team", team)) el("your-team").replaceChildren(...[...team].sort((a, b) => (b.active - a.active) || (a.fainted - b.fainted)).map(memberCard));
}

/* ---------------------------------------------------------------- decisions, battle log, events, warnings */
const openRows = new Set();

function miniMeter(value, shown, kind) {
  const mini = node("span", `mini ${kind}`);
  const track = node("span", "track");
  const fill = node("i", "fill");
  fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  track.append(fill);
  mini.append(track, node("span", "val", shown));
  return mini;
}

function decisionRow(d, key, withValue) {
  const n = d.notes;
  const row = node("details", `drow${n?.fallback ? " is-fallback" : ""}`);
  row.open = openRows.has(key);
  row.addEventListener("toggle", () => { if (row.open) openRows.add(key); else openRows.delete(key); });
  const head = node("summary");
  const turn = node("span", "t", d.kind === "team" ? "Pick" : `T${d.turn ?? "?"}`);
  const what = node("span", "k", d.kind === "team" ? d.label : orderLabel({ kind: d.order_kind, label: d.label || d.order }));
  const tags = node("span", "g");
  if (n?.fallback) tags.append(node("span", "badge is-fallback", "fallback"));
  if (d.kind === "forced") tags.append(node("span", "badge is-kind", "forced"));
  if (d.degrade) tags.append(degradeBadge(d.degrade));
  let meter = node("span", "mini");
  if (isNum(n?.p)) meter = tip(miniMeter(n.p, pct(n.p), "is-p"), `Confidence ${pct(n.p)}${isNum(n.h) ? `, entropy ${fixed(n.h)}` : ""}`);
  else if (isNum(n?.v) && !withValue) meter = tip(miniMeter((n.v + 1) / 2, signed(n.v, 2), "is-v"), `Critic value of the picked team: ${signed(n.v, 2)} (−1 a loss, +1 a win)${isNum(n.h) ? `; entropy ${fixed(n.h)}` : ""}`);
  else if (n?.rule && !withValue) meter = node("span", "mini is-rule", n.rule);
  head.append(turn, what, tags);
  if (withValue) {
    head.append(isNum(n?.v)
      ? tip(node("span", `vnum ${valueTone(n.v)}`, signed(n.v, 2)), `Value ${signed(n.v, 2)}: the critic's expected return here (+1 a win, −1 a loss); a rough win chance of ${roughWin(n.v)}`)
      : tip(node("span", "vnum", "–"), "No value recorded for this decision"));
  }
  head.append(meter);
  const body = node("div", "drow-body");
  if (n?.fallback) body.append(node("p", "callout is-warn", `Fallback: ${n.why || "the network did not decide"}`));
  (d.warnings || []).forEach((w) => body.append(node("p", "callout is-warn", w)));
  if (d.legal?.length || d.dropped?.length) {
    const legal = node("div", "legal");
    legal.append(...legalChips(d));
    body.append(legal);
  }
  const notes = node("div", "notes");
  notes.append(...(isNum(n?.h) ? [fieldChip("entropy", fixed(n.h), "Normalised entropy: 0 = certain, 1 = undecided")] : []),
    ...(isNum(n?.v) ? [fieldChip("value", `${signed(n.v, 2)} (${roughWin(n.v)} win)`, "The critic's expected return (+1 a win, −1 a loss) and a rough win chance")] : []), ...noteChips(n, d.timing));
  body.append(notes);
  row.append(head, body);
  return row;
}

function renderLists(state) {
  const decisions = state.decisions || [];
  const preview = state.preview;
  const battle = state.battle?.index;
  if (changed("decisions", [battle, decisions, preview])) {
    const withValue = decisions.some((d) => isNum(d.notes?.v)); // only battles whose decisions carry a value get the column
    const rows = decisions.map((d, i) => decisionRow(d, `${battle}:${d.sequence ?? i}`, withValue)).reverse();
    if (preview) rows.push(decisionRow({ kind: "team", label: (preview.names || []).join(", "), notes: preview.notes }, `${battle}:team`, withValue));
    el("decision-list").classList.toggle("has-v", withValue);
    el("decision-list").replaceChildren(...(rows.length ? rows : [node("p", "pane-none", battle ? "No decisions in this battle yet." : "Decisions are listed once a battle starts.")]));
    el("tab-decisions").textContent = decisions.length ? `Decisions ${decisions.length}` : "Decisions";
  }
  if (changed("battle-log", state.battle_log)) el("battle-log").textContent = state.battle_log || "No battle log yet: it is read from the game during the battle.";
  const warnings = state.warnings || [];
  if (changed("warnings", [warnings, state.warning_count])) {
    el("tab-warnings").textContent = state.warning_count ? `Warnings ${state.warning_count}` : "Warnings";
    el("tab-warnings").classList.toggle("has-warn", Boolean(state.warning_count));
    el("warnings").replaceChildren(...(warnings.length ? [...warnings].reverse().map((w) => {
      const li = node("li");
      li.dataset.level = w.level || "";
      li.append(node("span", "t", timeLabel(w.ts)), node("span", "k", w.kind), node("span", "d", w.detail));
      return tip(li, `${w.kind}: ${w.detail}`);
    }) : [node("li", "pane-none", "No warnings or disconnects in this run.")]));
  }
  const b = state.battle_state || {};
  if (changed("tracked", b)) {
    const fields = [["status", b.status], ["active", b.active], ["opponent", b.opponent], ["team", b.team], ["message", b.message]];
    el("battle-state-grid").replaceChildren(...fields.filter(([, v]) => v).map(([k, v]) => {
      const row = node("div");
      row.append(node("span", "", k), node("span", "", v));
      return tip(row, `${k}: ${v}`);
    }));
    el("battle-object-summary").textContent = b.battle_object || "No tracked battle yet.";
  }
  if (!changed("timeline", state.timeline)) return;
  el("timeline").replaceChildren(...((state.timeline || []).length ? [...state.timeline].reverse().map((e) => {
    const li = node("li");
    li.dataset.kind = e.kind || "";
    li.append(node("span", "t", timeLabel(e.ts)), node("span", "k", e.label), node("span", "d", e.detail));
    return tip(li, e.detail || e.label);
  }) : [node("li", "pane-none", "The run has not written an event yet.")]));
}

function setupTabs() {
  const tabs = [...document.querySelectorAll(".tab")];
  const select = (name) => {
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.setAttribute("aria-selected", String(on));
      el(`pane-${t.dataset.tab}`).hidden = !on;
    });
    try { localStorage.setItem("pcsa-tab", name); } catch (e) { /* storage blocked: the choice lasts this visit */ }
  };
  tabs.forEach((t) => t.addEventListener("click", () => select(t.dataset.tab)));
  let saved = "decisions";
  try { saved = localStorage.getItem("pcsa-tab") || saved; } catch (e) { /* storage blocked */ }
  select(tabs.some((t) => t.dataset.tab === saved) ? saved : "decisions");
}

/* ---------------------------------------------------------------- theme and loop */
function setupTheme() {
  const button = el("theme");
  const order = ["auto", "light", "dark"];
  const current = () => document.documentElement.dataset.theme || "auto";
  const paint = () => { button.textContent = current()[0].toUpperCase() + current().slice(1); };
  button.addEventListener("click", () => {
    const next = order[(order.indexOf(current()) + 1) % order.length];
    if (next === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = next;
    try { localStorage.setItem("pcsa-theme", next); } catch (e) { /* storage blocked: the choice lasts this visit */ }
    paint();
  });
  paint();
}

function render(state) {
  document.body.classList.toggle("is-pinned", pinned !== null);
  renderStatus(state);
  renderStateBox(state);
  renderDecision(state);
  renderFoes(state);
  renderFrame(state);
  renderRun(state);
  renderTeam(state);
  renderLists(state);
  armTips();
}

let polling = false;

async function poll() {
  if (polling) return;
  polling = true;
  let state = null;
  try {
    const response = await fetch(pinned === null ? "/api/state" : `/api/state?battle=${pinned}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`state request failed (${response.status})`);
    state = await response.json();
  } catch (error) { // the server is gone or restarting: keep the last view and say so
    const pill = el("run-status");
    pill.dataset.state = "offline";
    pill.lastElementChild.textContent = "SERVER OFFLINE";
  } finally {
    polling = false;
  }
  if (!state) return;
  render(state);
  if (location.search.includes("probe")) { // layout check: page scroll height vs. viewport
    document.body.dataset.scroll = `${document.scrollingElement.scrollHeight}/${window.innerHeight}/${document.scrollingElement.scrollWidth}/${window.innerWidth}`;
  }
}

setupTheme();
setupTips();
setupTabs();
el("follow").addEventListener("click", () => pin(null));
window.addEventListener("resize", () => { spark.key = ""; vspark.key = ""; });
poll();
window.setInterval(poll, 1000);
