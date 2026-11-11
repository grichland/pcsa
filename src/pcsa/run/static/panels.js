/* pcsa watch: the right column. Win rate (record, rating, result strip, win chance of the battle shown), the team
   cards (ours in blue facing theirs in red, with the field between them) and the battle log turn by turn. */

/* ---------------------------------------------------------------- small line charts (rating, win chance) */
const charts = {};

function resetCharts() {
  Object.keys(charts).forEach((key) => { charts[key] = ""; });
}

/** A line over `points` ({label, y}) in the box `id`; `opts`: lo, hi, ticks [{y, text}], mid (a dashed line), tone(y). */
function lineChart(id, points, opts) {
  const box = el(id);
  const key = JSON.stringify([points, box.clientWidth, box.clientHeight]);
  if (charts[id] === key) return;
  charts[id] = key;
  const w = Math.max(120, box.clientWidth);
  const h = Math.max(40, box.clientHeight);
  const [top, bottom, left, right] = [16, 6, 36, 8];
  const x = (i) => (points.length > 1 ? left + (i * (w - left - right)) / (points.length - 1) : (left + w - right) / 2);
  const y = (v) => top + (1 - (v - opts.lo) / (opts.hi - opts.lo || 1)) * (h - top - bottom);
  const chart = svg("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": opts.label });
  opts.ticks.forEach((t) => {
    chart.append(svg("line", { class: t.y === opts.mid ? "zero" : "grid", x1: left, x2: w - right, y1: y(t.y), y2: y(t.y) }));
    const label = svg("text", { class: "axis", x: left - 6, y: y(t.y) + 3, "text-anchor": "end" });
    label.textContent = t.text;
    chart.append(label);
  });
  const at = points.map((p, i) => [x(i), y(p.y)]);
  if (at.length > 1) chart.append(svg("path", { class: "line", d: at.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ") }));
  if (at.length <= 40) at.forEach(([px, py], i) => chart.append(svg("circle", { class: `pt ${opts.tone ? opts.tone(points[i].y) : ""}`, cx: px, cy: py, r: 2.4 })));
  const halo = svg("circle", { class: "halo", r: 7 });
  const dot = svg("circle", { class: "dot", r: 3.6 });
  chart.append(halo, dot);
  const readout = node("span", "spark-read");
  const show = (i) => {
    [dot, halo].forEach((c) => { c.setAttribute("cx", at[i][0]); c.setAttribute("cy", at[i][1]); });
    readout.textContent = points[i].label;
  };
  chart.addEventListener("mousemove", (event) => {
    if (at.length < 2) return;
    const rect = chart.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * w;
    show(Math.max(0, Math.min(points.length - 1, Math.round((px - left) / ((w - left - right) / (points.length - 1))))));
  });
  chart.addEventListener("mouseleave", () => show(points.length - 1));
  show(points.length - 1);
  box.replaceChildren(readout, chart);
}

/* ---------------------------------------------------------------- win rate */
function battleFacts(b) {
  return [`Battle ${b.index}`, b.opponent?.length ? `vs ${b.opponent.join(", ")}` : null, b.picks?.length ? `brought ${b.picks.join(", ")}` : null,
    b.turn ? plural(b.turn, "turn") : null, plural(b.decisions || 0, "decision"), b.disconnects ? plural(b.disconnects, "disconnect") : null,
    isNum(b.rating_after) && isNum(b.rating_before) ? `rating ${b.rating_before.toFixed(1)} → ${b.rating_after.toFixed(1)}` : null,
    b.outcome_source ? `result from the ${b.outcome_source}` : null].filter(Boolean).join(" · ");
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

/** The win chance at each decision of the battle shown, the team pick first; decisions without a value are left out. */
function chanceSeries(state) {
  const series = [];
  if (!(state.decisions || []).some((d) => isNum(d.notes?.v))) return series;
  const pick = state.preview?.notes?.v;
  if (isNum(pick)) series.push({ label: `team pick · ${pct(winChance(pick))}`, y: winChance(pick) });
  (state.decisions || []).forEach((d) => {
    if (isNum(d.notes?.v)) series.push({ label: `turn ${d.turn ?? "?"}${d.kind === "forced" ? " switch-in" : ""} · ${pct(winChance(d.notes.v))}`, y: winChance(d.notes.v) });
  });
  return series;
}

function renderRates(state) {
  const r = state.session_record || {};
  const rates = state.winrate || {};
  const battles = state.battles || [];
  const playing = battles.filter((b) => !b.outcome).length;
  text("win-record", `${r.wins || 0}W ${r.losses || 0}L`);
  el("win-record-sub").textContent = [r.draws ? `${r.draws} drawn` : null, r.forfeits ? `${r.forfeits} by forfeit` : null, playing ? `${playing} playing` : null].filter(Boolean).join(" · ");
  text("win-rate", isNum(rates.rate) ? pct(rates.rate) : "--");
  el("win-rate-sub").textContent = rates.battles ? `${rates.wins} of ${rates.battles}` : "no result yet";
  const recent = rates.recent || {};
  text("win-recent", isNum(recent.rate) ? pct(recent.rate) : "--");
  el("win-recent-sub").textContent = recent.battles ? `${recent.wins} of ${recent.battles}` : "";
  const streak = rates.streak || 0;
  el("win-meta").textContent = [plural(battles.length, "battle"), Math.abs(streak) > 1 ? `${Math.abs(streak)} ${streak > 0 ? "wins" : "losses"} in a row` : null,
    state.disconnects ? plural(state.disconnects, "disconnect") : null].filter(Boolean).join(" · ");
}

function renderRating(rating) {
  el("win-rating-stat").hidden = !isNum(rating?.rating);
  if (isNum(rating?.rating)) {
    text("win-rating", Math.round(rating.rating));
    el("win-rating-label").textContent = [rating.tier, rating.rank ? `rank ${rating.rank}` : null].filter(Boolean).join(" · ") || "Rating";
    const delta = el("win-delta");
    delta.textContent = isNum(rating.delta) && rating.delta !== 0 ? `${signed(rating.delta)} this run` : "";
    delta.className = `delta ${rating.delta > 0 ? "up" : "down"}`;
  }
  const series = rating?.series || [];
  el("spark").hidden = series.length < 2;
  if (series.length < 2) return;
  const values = series.map((s) => s.rating);
  const [lo, hi] = [Math.min(...values), Math.max(...values)];
  lineChart("spark", series.map((s) => ({ label: `${s.battle ? `after battle ${s.battle}` : "at the start"} · ${s.rating.toFixed(1)}`, y: s.rating })),
    { lo, hi, label: "Rating over the run", ticks: [{ y: hi, text: Math.round(hi) }, { y: lo, text: Math.round(lo) }] });
}

function renderWin(state) {
  const battles = state.battles || [];
  tile("t-win", battles.length > 0 || isNum(state.rating?.rating));
  el("follow").hidden = pinned === null;
  renderRates(state);
  renderRating(state.rating);
  if (changed("strip", [battles, state.battle?.index])) el("strip").replaceChildren(...battles.map((b) => stripCell(b, state.battle?.index)));
  const series = chanceSeries(state);
  el("chance").hidden = series.length === 0;
  if (!series.length) return;
  el("chance-title").textContent = `Win chance · battle ${state.battle?.index ?? "--"}`;
  const now = series[series.length - 1].y;
  text("chance-now", pct(now));
  el("chance-now").className = now >= 0.5 ? "up" : "down";
  lineChart("vspark", series, { lo: 0, hi: 1, mid: 0.5, label: "Win chance at each decision of this battle", tone: (v) => (v > 0.51 ? "up" : v < 0.49 ? "down" : ""),
    ticks: [{ y: 1, text: "100%" }, { y: 0.5, text: "50%" }, { y: 0, text: "0%" }] });
}

/* ---------------------------------------------------------------- team cards: ours (blue) facing theirs (red) */
const MOVE_SLOTS = 4;

function moveChip(name, pp, side) {
  const chip = node("span", "chip");
  chip.append(node("span", "n", name));
  if (Array.isArray(pp)) chip.append(node("span", "p", `${pp[0]}/${pp[1]}`));
  if (side === "opp") return tip(chip, `${name}: revealed in battle`);
  return tip(chip, Array.isArray(pp) ? `${name}: ${pp[0]} of ${pp[1]} PP left` : `${name} (PP is known for the active Pokémon only)`);
}

function monCard(m, side) {
  const card = node("div", `card ${side}${m.active ? " is-active" : ""}${m.fainted ? " is-fainted" : ""}`);
  const top = node("div", "card-top");
  top.append(node("span", "nm", m.name));
  if (m.active && !m.fainted) top.append(tip(node("span", "on-field", "active"), "On the field now"));
  if (m.status && !m.fainted) top.append(tip(node("span", "status", String(m.status).toUpperCase()), "Status condition"));
  const hp = node("div", "card-hp");
  hp.append(hpBar(m.fraction), node("span", "hptext", m.fainted ? "fainted" : m.hp || "?"));
  tip(hp, side === "you" ? `HP ${m.hp || "unknown"}${isNum(m.fraction) ? ` (${pct(m.fraction)})` : ""}, from the game's request` : `HP ${m.hp || "unknown"} (the game shows their HP as a percentage)`);
  const facts = node("div", "card-facts");
  const known = [m.item ? `@ ${m.item}` : null, m.ability || null].filter(Boolean);
  facts.append(node("span", "types", m.types || ""), node("span", known.length ? "" : "is-none", known.join(" · ") || (side === "opp" ? "item and ability not revealed" : "")));
  tip(facts, side === "you" ? "Types, item and ability (our team sheet)" : "Types, and the item and ability once the battle has revealed them");
  card.append(top, hp, facts);
  if (m.boosts && !m.fainted) {
    const boosts = node("div", "boosts");
    String(m.boosts).split(/\s+/).filter(Boolean).forEach((b) => boosts.append(node("span", `boost ${b.includes("-") ? "down" : "up"}`, b)));
    card.append(tip(boosts, "Stat stages"));
  }
  const chips = node("div", "chips");
  const moves = side === "you" ? (m.moves || []).map((mv) => moveChip(mv.name, mv.pp, side)) : (m.moves || []).map((name) => moveChip(name, null, side));
  for (let i = moves.length; side === "opp" && i < MOVE_SLOTS; i += 1) moves.push(tip(node("span", "chip is-unknown"), "A move they have not shown yet"));
  chips.append(...moves);
  card.append(chips);
  return card;
}

function emptyCard() {
  const card = node("div", "card opp is-unseen");
  card.append(node("span", "nm", "not seen yet"));
  return tip(card, "One of their three Pokémon that has not come out yet");
}

/** Actives face each other in the middle: bench, active, bench. */
function facing(members) {
  const active = members.findIndex((m) => m.active && !m.fainted);
  if (active < 0 || members.length < 3) return members;
  const rest = members.filter((_, i) => i !== active);
  return [rest[0], members[active], ...rest.slice(1)];
}

function fieldChips(field) {
  const chips = [];
  if (field?.weather) chips.push(fieldChip(field.weather.name, field.weather.left, `Weather: about ${plural(field.weather.left, "turn")} left (5-turn default; rocks like Heat Rock extend it)`));
  (field?.fields || []).forEach((f) => chips.push(fieldChip(f.name, f.left, `${f.name}: about ${plural(f.left, "turn")} left (5-turn default)`)));
  [["you", "Our side", field?.ours], ["opp", "Their side", field?.theirs]].forEach(([side, whose, conditions]) => (conditions || []).forEach((c) => {
    if (c.layers) chips.push(fieldChip(c.name, c.layers > 1 ? `×${c.layers}` : null, `${whose}: ${c.name}, ${plural(c.layers, "layer")} (stays until removed)`, side));
    else if (c.left) chips.push(fieldChip(c.name, c.left, `${whose}: ${c.name}, about ${plural(c.left, "turn")} left (default duration; Light Clay extends screens)`, side));
    else chips.push(fieldChip(c.name, null, `${whose}: ${c.name} (stays until removed)`, side));
  }));
  return chips;
}

function renderTeams(state) {
  const team = state.your_team || [];
  const foes = state.foes || [];
  const unseen = state.unseen || [];
  tile("t-teams", team.length > 0 || foes.length > 0, state.battle?.opponent?.length ? `Their preview: ${state.battle.opponent.join(", ")}` : "Shown once the battle starts", "teams-empty");
  el("teams-meta").textContent = state.move_capture && team.length ? `as of ${timeLabel(state.move_capture)}` : "";
  if (!changed("teams", [team, foes, unseen, state.field])) return;
  el("team-ours").replaceChildren(...facing(team).map((m) => monCard(m, "you")));
  const theirs = facing(foes).map((f) => monCard(f, "opp"));
  while (theirs.length < 3 && (unseen.length || foes.length)) theirs.push(emptyCard());
  el("team-theirs").replaceChildren(...theirs);
  const chips = fieldChips(state.field);
  el("field-strip").replaceChildren(node("span", "fs-k", "Field"), ...(chips.length ? chips : [node("span", "fs-none", "no weather, terrain, hazards or screens")]));
  el("unseen").hidden = unseen.length === 0;
  el("unseen").replaceChildren(node("span", "", "Seen at preview, not brought yet"), ...unseen.map((name) => tip(node("b", "", name), `${name}: in their team preview, not seen in this battle`)));
}

/* ---------------------------------------------------------------- battle log, turn by turn */
const SIDE_CLASS = { ours: "you", theirs: "opp" };

function logLine(line) {
  const row = node("div", `ll ${SIDE_CLASS[line.side] || "none"} is-${line.kind}`);
  if (line.kind === "action") row.append(node("span", "cat", "we chose"), node("span", "tx", line.text.replace(/^chose /, "")));
  else {
    if (line.category) row.append(node("span", "cat", line.category));
    row.append(node("span", "tx", line.text));
  }
  return row;
}

function renderLog(state) {
  const log = state.log || {};
  const turns = log.turns || [];
  tile("t-log", turns.length > 0 || Boolean(log.raw), state.battle ? "No battle log yet: it is read from the game during the battle" : "The battle log is shown once a battle starts", "log-empty");
  el("log-meta").textContent = turns.length ? `${plural(turns.length, "turn")} · ${log.source || ""}` : log.source || "";
  if (!changed("log", [state.battle?.index, log])) return;
  const box = el("battle-log");
  if (!turns.length) {
    box.replaceChildren(node("pre", "", log.raw || ""));
    return;
  }
  const blocks = turns.map((t) => {
    const block = node("section", "lt");
    block.append(node("h3", "", `Turn ${t.turn}`), ...(t.lines.length ? t.lines.map(logLine) : [node("div", "ll none is-note", "no lines for this turn")]));
    return block;
  });
  const notes = (log.notes || []).filter((n) => n && n !== "none");
  if (notes.length) {
    const block = node("section", "lt");
    block.append(node("h3", "", "Coverage notes"), ...notes.map((n) => node("div", "ll none is-note", n)));
    blocks.push(block);
  }
  box.replaceChildren(...blocks);
  const scroller = box.parentElement;
  scroller.scrollTop = scroller.scrollHeight; // the newest turn stays in view
}
