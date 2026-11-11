/* pcsa watch: shared helpers, the top bar, the phone frame, the theme and the polling loop.
   matrix.js draws the decision matrix; panels.js draws the win rate, the team cards and the battle log. */
const el = (id) => document.getElementById(id);
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const drawn = {};
// A battle number picked in the result strip (or ?battle=N in the address), or null to follow the newest battle.
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

const clamp01 = (v) => Math.max(0, Math.min(1, v));
const signed = (v, digits = 1) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const title = (s) => (s ? String(s).charAt(0).toUpperCase() + String(s).slice(1) : "");
/* The network's value: +1 a won battle, −1 a lost one; read as a win chance it is (value + 1) / 2. */
const winChance = (v) => clamp01((v + 1) / 2);

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
const SIDE_TIP = { you: "Our Pokémon (blue everywhere on this page)", opp: "The opponent's Pokémon (red everywhere on this page)" };

function sideTag(side, label) {
  return tip(node("span", `side ${side}`, label || (side === "you" ? "OURS" : "THEIRS")), SIDE_TIP[side]);
}

function hpBar(fraction) {
  const bar = node("span", "hpbar");
  const fill = node("i", isNum(fraction) && fraction <= 0.25 ? "is-crit" : isNum(fraction) && fraction <= 0.5 ? "is-low" : "");
  fill.style.width = `${(isNum(fraction) ? clamp01(fraction) : 0) * 100}%`;
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

function fieldChip(label, count, message, side) {
  const chip = node("span", `chipf${side ? ` ${side}` : ""}`);
  chip.append(node("span", "", label));
  if (count !== null && count !== undefined) chip.append(node("b", "", count));
  return tip(chip, message);
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

/** Every element with a tip can take keyboard focus (controls already can; matrix cells opt out with tabindex -1). */
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

function renderWarnings(state) {
  const warnings = state.warnings || [];
  el("warn").hidden = !state.warning_count;
  if (!changed("warnings", [warnings, state.warning_count])) return;
  el("warn-count").textContent = plural(state.warning_count || 0, "warning");
  el("warnings").replaceChildren(...[...warnings].reverse().map((w) => {
    const li = node("li");
    li.dataset.level = w.level || "";
    li.append(node("span", "t", timeLabel(w.ts)), node("span", "k", w.kind), node("span", "d", w.detail));
    return tip(li, `${w.kind}${w.battle ? ` (battle ${w.battle})` : ""}: ${w.detail}`);
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
  text("top-player", state.player);
  renderPlayer(state.player_view);
  renderWarnings(state);
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

/** The two drop-down lists of the top bar close on a click anywhere else and on Escape. */
function setupDrops() {
  const drops = [...document.querySelectorAll("details.drop")];
  document.addEventListener("click", (event) => drops.forEach((d) => { if (d.open && !d.contains(event.target)) d.open = false; }));
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") drops.forEach((d) => { d.open = false; }); });
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

function pin(index) {
  pinned = index;
  poll();
}

function render(state) {
  document.body.classList.toggle("is-pinned", pinned !== null);
  renderStatus(state);
  renderFrame(state);
  renderMatrix(state, pinned !== null);
  renderWin(state);
  renderTeams(state);
  renderLog(state);
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

document.addEventListener("DOMContentLoaded", () => { // the deferred panel scripts have run by now
  setupTheme();
  setupTips();
  setupDrops();
  el("follow").addEventListener("click", () => pin(null));
  window.addEventListener("resize", () => { resetCharts(); });
  poll();
  window.setInterval(poll, 1000);
});
