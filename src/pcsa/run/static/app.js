const el = (id) => document.getElementById(id);
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const drawn = {};

/* ---------------------------------------------------------------- small helpers */
function node(tag, className, content) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (content !== undefined && content !== null) item.textContent = String(content);
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

const title = (s) => (s ? String(s).charAt(0).toUpperCase() + String(s).slice(1) : "");

function timeLabel(value) {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "--" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function ago(seconds) {
  if (!isNum(seconds)) return "No updates";
  if (seconds < 2) return "updated just now";
  return seconds < 60 ? `updated ${Math.round(seconds)}s ago` : `updated ${Math.floor(seconds / 60)}m ago`;
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

/* ---------------------------------------------------------------- sides: who owns a Pokémon */
const SIDE_TIP = { you: "Your Pokémon", opp: "The opponent's Pokémon" };

function sideTag(side) {
  return tip(node("span", `side ${side}`, side === "you" ? "YOU" : "OPP"), SIDE_TIP[side]);
}

function mon(name, side) {
  const wrap = node("span", "mon");
  wrap.append(sideTag(side), node("span", "nm", name));
  return wrap;
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

/* ---------------------------------------------------------------- names on the field */
function actives(state) {
  return { ours: title(state.active?.name), theirs: title(state.opponent?.name) };
}

/* ---------------------------------------------------------------- top bar and battle-state box */
function renderStatus(state) {
  const pill = el("run-status");
  const status = String(state.status || "WAITING");
  pill.dataset.state = status.toLowerCase();
  pill.lastElementChild.textContent = status;
  text("run-name", state.run_name, "No run found");
  text("top-battle", state.battle?.index);
  text("top-turn", state.turn);
  text("top-screen", state.screen);
  text("top-policy", state.policy);
  const r = state.session_record || {};
  const extra = (r.draws ? ` ${r.draws}D` : "") + (r.forfeits ? ` ${r.forfeits}F` : "");
  text("session-record", `${r.wins || 0}W ${r.losses || 0}L${extra}`);
  el("last-updated").textContent = ago(state.age_s);
}

function fieldChip(label, count, message) {
  const chip = node("span", "chipf");
  chip.append(node("span", "", label));
  if (count !== null && count !== undefined) chip.append(node("b", "", count));
  return tip(chip, message);
}

function sideChips(conditions) {
  return conditions.map((c) => {
    if (c.layers) return fieldChip(c.name, c.layers > 1 ? `×${c.layers}` : null, `${c.name}: ${c.layers} layer${c.layers > 1 ? "s" : ""} (stays until removed)`);
    if (c.left) return fieldChip(c.name, c.left, `${c.name}: about ${c.left} turn${c.left > 1 ? "s" : ""} left (default duration; Light Clay extends screens)`);
    return fieldChip(c.name, null, `${c.name}: stays until removed`);
  });
}

function renderStateBox(state) {
  const names = actives(state);
  const field = state.field;
  if (!changed("statebox", [names, field])) return;
  const match = el("sb-match");
  if (names.ours || names.theirs) {
    match.replaceChildren(mon(names.ours || "?", "you"), node("span", "vs", "vs"), mon(names.theirs || "?", "opp"));
  } else {
    match.replaceChildren(node("span", "sb-none", "No actives yet"));
  }
  const fieldCell = el("sb-field");
  const chips = [];
  if (field?.weather) chips.push(fieldChip(field.weather.name, field.weather.left, `Weather: about ${field.weather.left} turn${field.weather.left > 1 ? "s" : ""} left (5-turn default; rocks like Heat Rock extend it)`));
  (field?.fields || []).forEach((f) => chips.push(fieldChip(f.name, f.left, `${f.name}: about ${f.left} turn${f.left > 1 ? "s" : ""} left (5-turn default)`)));
  fieldCell.replaceChildren(node("span", "sb-k", "Field"), ...(chips.length ? chips : [node("span", "sb-none", field ? "clear" : "--")]));
  tip(fieldCell, chips.length ? null : "Weather, terrain and rooms from the battle log (none active)");
  [["sb-ours", "you", field?.ours], ["sb-theirs", "opp", field?.theirs]].forEach(([id, side, conditions]) => {
    const list = sideChips(conditions || []);
    el(id).replaceChildren(sideTag(side), ...(list.length ? list : [node("span", "sb-none", field ? "no hazards or screens" : "--")]));
  });
  armTips();
}

/* ---------------------------------------------------------------- decision selection */
function renderDecision(state) {
  const decision = state.decision;
  tile("t-decision", Boolean(decision), "Waiting for the first decision", "decision-empty");
  if (!decision) return;
  const order = (decision.order || decision.action || "").replace(/^\/choose\s+/, "");
  const ours = actives(state).ours;
  const kind = el("decision-kind");
  if (changed("hero-kind", [ours, decision.kind])) {
    kind.replaceChildren(...(ours ? [mon(ours, "you"), node("span", "", decision.kind === "forced" ? " · forced switch" : " · turn action")] : [node("span", "", decision.kind === "forced" ? "Forced switch" : "Turn action")]));
  }
  text("decision-order", order, "Decision recorded");
  el("decision-meta").textContent = `Turn ${decision.turn ?? "--"} · ${decision.source || ""}`;
  tip(el("decision-meta"), "Decision turn, how the battle was tracked, and why");
}

function foeCard(f) {
  const card = node("div", "foe");
  const top = node("div", "foe-top");
  const name = mon(f.name, "opp");
  name.classList.add("foe-name");
  top.append(name);
  const chips = node("div", "chips");
  (f.moves || []).forEach((m) => {
    const chip = tip(node("span", "chip is-known"), `${m}: revealed in battle`);
    chip.append(node("span", "n", m), node("span", "p", "✓"));
    chips.append(chip);
  });
  card.append(top, chips);
  return card;
}

function renderFoes(state) {
  const foes = state.opponent_moves || [];
  tile("t-foes", foes.length > 0);
  if (!changed("foes", foes)) return;
  el("foes-source").textContent = foes.length ? "revealed moves" : "";
  el("foes").replaceChildren(...foes.map((o) => foeCard({ name: title(o.name), moves: o.moves })));
  armTips();
}

/* ---------------------------------------------------------------- frame: polled at the run's display rate */
const frameState = { current: null, rate: 0, timer: null, busy: false };

function showFrame(id) {
  const frame = el("t-frame");
  if (!id) {
    frame.dataset.empty = "true";
    frameState.current = null;
    return;
  }
  if (id === frameState.current) return;
  frameState.current = id;
  const next = new Image(); // decode off-screen, then swap: no flicker between frames
  next.onload = () => { el("phone-frame").src = next.src; frame.dataset.empty = "false"; };
  next.onerror = () => { frameState.current = null; };
  next.src = `/api/frame?v=${encodeURIComponent(id)}`;
}

async function pollFrame() {
  if (frameState.busy) return;
  frameState.busy = true;
  try {
    const response = await fetch("/api/frame-id", { cache: "no-store" });
    if (response.ok) showFrame((await response.json()).id);
  } catch (error) { /* the next tick retries */ } finally {
    frameState.busy = false;
  }
}

function renderFrame(state) {
  const fps = Number(state.display_fps) || 0;
  if (fps > 0 && fps !== frameState.rate) {
    window.clearInterval(frameState.timer);
    frameState.rate = fps;
    frameState.timer = window.setInterval(pollFrame, Math.max(100, 1000 / fps));
  } else if (fps <= 0) {
    if (frameState.timer) window.clearInterval(frameState.timer);
    frameState.rate = 0;
    frameState.timer = null;
    showFrame(state.frame_id);
  }
  tip(el("t-frame"), fps > 0 ? `The newest frame the agent captured, refreshed up to ${fps}× a second (never a capture of its own)` : "The newest frame the agent saved");
}

/* ---------------------------------------------------------------- below the fold */
function memberCard(m) {
  const card = node("div", `member${m.active ? " is-active" : ""}${m.fainted ? " is-fainted" : ""}`);
  const top = node("div", "member-top");
  const name = mon(m.name, "you");
  if (m.status) name.append(tip(node("span", "status", m.status.toUpperCase()), "Status condition"));
  const bar = node("span", "hpbar");
  const fill = node("i", isNum(m.fraction) && m.fraction <= 0.25 ? "is-crit" : isNum(m.fraction) && m.fraction <= 0.5 ? "is-low" : "");
  fill.style.width = `${(isNum(m.fraction) ? m.fraction : 0) * 100}%`;
  bar.append(fill);
  top.append(name, tip(bar, `HP ${m.hp} (${isNum(m.fraction) ? pct(m.fraction) : "--"}), from the game's request`), node("span", "hptext", m.fainted ? "fainted" : m.hp));
  const sub = node("div", "member-sub");
  sub.append(node("b", "", m.item || "no item"), node("span", "", ` · ${m.ability || "?"}${m.boosts ? ` · ${m.boosts}` : ""}`));
  tip(sub, `Item ${m.item || "none"}, ability ${m.ability || "unknown"} (your team sheet)${m.boosts ? `; stat stages ${m.boosts}` : ""}`);
  const chips = node("div", "chips");
  (m.moves || []).forEach((mv) => {
    const chip = node("span", "chip");
    chip.append(node("span", "n", mv.name));
    if (Array.isArray(mv.pp)) chip.append(node("span", "p", `${mv.pp[0]}/${mv.pp[1]}`));
    tip(chip, Array.isArray(mv.pp) ? `${mv.name}: ${mv.pp[0]} of ${mv.pp[1]} PP left (read from the move list)` : `${mv.name} (PP is read for the active Pokémon only)`);
    chips.append(chip);
  });
  card.append(top, sub, chips);
  if (m.active) tip(card, "On the field now");
  return card;
}

function renderState(state) {
  const team = state.your_team || [];
  tile("t-team", team.length > 0);
  el("move-capture").textContent = state.move_capture && team.length ? `PP read ${timeLabel(state.move_capture)}` : "";
  if (changed("team", team)) el("your-team").replaceChildren(...[...team].sort((a, b) => (b.active - a.active) || (a.fainted - b.fainted)).map(memberCard));
  const b = state.battle_state || {};
  const fields = [["status", b.status], ["active", b.active], ["opponent", b.opponent], ["team", b.team], ["message", b.message]];
  el("battle-state-grid").replaceChildren(...fields.filter(([, v]) => v).map(([k, v]) => {
    const row = node("div");
    row.append(node("span", "", k), node("span", "", v));
    return tip(row, `${k}: ${v}`);
  }));
  el("battle-object-summary").textContent = b.battle_object || "No tracked battle yet.";
  el("capture-time").textContent = state.last_event_at ? `last event ${timeLabel(state.last_event_at)}` : "";
  if (!changed("timeline", state.timeline)) return;
  el("timeline").replaceChildren(...(state.timeline || []).slice(-40).reverse().map((e) => {
    const li = node("li");
    li.dataset.kind = e.kind || "";
    li.append(node("span", "t", timeLabel(e.ts)), node("span", "k", e.label), node("span", "d", e.detail));
    return tip(li, e.detail || e.label);
  }));
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
  let saved = "events";
  try { saved = localStorage.getItem("pcsa-tab") || saved; } catch (e) { /* storage blocked */ }
  select(tabs.some((t) => t.dataset.tab === saved) ? saved : "events");
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
  renderStatus(state);
  renderStateBox(state);
  renderDecision(state);
  renderFoes(state);
  renderFrame(state);
  renderState(state);
  armTips();
}

async function poll() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`state request failed (${response.status})`);
    render(await response.json());
    if (location.search.includes("probe")) { // layout check: page scroll height vs. viewport
      document.body.dataset.scroll = `${document.scrollingElement.scrollHeight}/${window.innerHeight}/${document.scrollingElement.scrollWidth}/${window.innerWidth}`;
    }
  } catch (error) {
    const pill = el("run-status");
    pill.dataset.state = "offline";
    pill.lastElementChild.textContent = "SERVER OFFLINE";
  }
}

setupTheme();
setupTips();
setupTabs();
poll();
window.setInterval(poll, 1000);
