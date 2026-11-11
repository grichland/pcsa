/* pcsa watch: the decision matrix. Their kept actions run across the top (4 moves, then 5 switches) with the
   probability of each; our 11 actions run down the side (4 moves, the same 4 with Mega Evolution, 3 switches).
   A cell says how good our action is if they answer with that column; the bar on the right is our final policy.
   A new decision plays once: columns, cells (most likely column first), row bars, then the row played. */
const MX = { megaAt: 4, switchAt: 8, reach: 1.2 };
const MX_T = { col: 35, cells: 420, rank: 55, row: 7, bars: 960, bar: 22, ring: 1380, overrideGap: 620, done: 3200 };
const KIND = { forced: "forced switch", action: "turn action", team: "team preview" };
const mxSeen = new Set(); // "battle:sequence" of every decision already drawn: none of them animates again
let mxTimer = null;

const orderLabel = (kind, label) => (kind === "switch" ? `⇄ ${label}` : label);
const rowKind = (i) => (i >= MX.switchAt ? "switch" : i >= MX.megaAt ? "mega" : "move");

function mxStat(label, value, message, fraction, tone) {
  const stat = tip(node("span", `mx-stat${tone ? ` ${tone}` : ""}`), message);
  stat.append(node("small", "", label), node("b", "", value));
  if (isNum(fraction)) {
    const track = node("span", "mx-stat-bar");
    const fill = node("i");
    fill.style.width = `${clamp01(fraction) * 100}%`;
    track.append(fill);
    stat.append(track);
  }
  return stat;
}

function mxStats(decision) {
  const n = decision.notes || {};
  const m = n.matrix;
  const stats = [];
  if (isNum(n.v)) {
    const chance = winChance(n.v);
    stats.push(mxStat("Win chance", pct(chance), `The network's value of this position is ${signed(n.v, 2)} (+1 a win, −1 a loss): a win chance of about ${pct(chance)}`, chance, chance >= 0.5 ? "up" : "down"));
  }
  if (isNum(m?.mega_p)) stats.push(mxStat("They Mega", pct(m.mega_p), `The network's chance that they Mega Evolve this turn: ${pct(m.mega_p)}`, m.mega_p, "opp"));
  if (isNum(n.p) && !m) stats.push(mxStat("Confidence", pct(n.p), "Probability the network gave the action it chose"));
  if (isNum(n.ms)) stats.push(mxStat("Decided in", n.ms >= 1000 ? `${(n.ms / 1000).toFixed(1)} s` : `${n.ms.toFixed(0)} ms`, "Time the player took to decide"));
  (n.rest || []).filter((r) => r.key === "retry").forEach(() => stats.push(mxStat("Retry", "same turn", "The same turn was asked again (a re-read screen or a rejected tap): this decision replaced the earlier one")));
  return stats;
}

/** The line under the hero on an override or a fallback: what happened and why. */
function mxBanner(decision) {
  const n = decision.notes || {};
  const banner = el("mx-banner");
  banner.hidden = !(n.override || n.fallback);
  banner.className = `mx-banner${n.fallback ? " is-fallback" : " is-override"}`;
  if (n.fallback) {
    const by = (n.rest || []).filter((r) => r.key === "fallback" || r.key === "fallback_rule").map((r) => r.value).join(" · ");
    banner.replaceChildren(node("b", "", "FALLBACK"), node("span", "", n.why || "the network did not decide; the fallback player did"), ...(by ? [node("em", "", `played by ${by}`)] : []));
    tip(banner, "The network did not decide this one: the fallback player did, for the reason shown");
  } else if (n.override) {
    const wanted = n.network ? orderLabel(n.network.kind, n.network.label) : "another action";
    banner.replaceChildren(node("b", "", "GUARD RULE"), node("strong", "", n.override), node("span", "", n.override_why || ""), node("em", "", `network wanted ${wanted}`));
    tip(banner, `The guard rule "${n.override}" replaced the network's action (${wanted}) with the one played`);
  }
}

function mxColumn(m, j, order, top) {
  const col = node("div", `mx-col is-${m.col_kind[j] === "switch" ? "switch" : "move"}`);
  col.style.setProperty("--d", `${order * MX_T.col}ms`);
  const bar = node("span", "mx-cbar");
  const fill = node("i");
  fill.style.height = `${clamp01(m.p_cols[j] / top) * 100}%`;
  bar.append(fill);
  const name = node("span", "mx-cname");
  if (m.shown[j]) name.append(node("i", "dot-shown"));
  name.append(node("span", "", m.col_labels[j]));
  col.append(node("b", "mx-cnum", pct(m.p_cols[j])), bar, name);
  const what = m.col_kind[j] === "switch" ? `They switch to ${m.col_labels[j]}` : `They use ${m.col_labels[j]}`;
  return tip(col, `${what}: ${pct(m.p_cols[j])} likely${m.shown[j] ? " (a move they have already shown)" : m.col_kind[j] === "switch" ? "" : " (not shown yet: the network expects it)"}`);
}

function mxCell(m, i, j, rank, rowOrder) {
  const v = m.cells[i][j];
  const cell = node("span", "mx-cell");
  cell.tabIndex = -1;
  cell.setAttribute("role", "cell");
  if (!m.legal[i]) return cell;
  const a = Math.min(1, Math.abs(v) / MX.reach);
  if (Math.abs(v) > 0.03) {
    cell.classList.add(v > 0 ? "pos" : "neg");
    cell.style.setProperty("--a", `${Math.round((0.14 + 0.78 * a ** 0.85) * 100)}%`);
  }
  cell.style.setProperty("--d", `${MX_T.cells + rank * MX_T.rank + rowOrder * MX_T.row}ms`);
  const verdict = v > 0.03 ? "good for us" : v < -0.03 ? "bad for us" : "even";
  return tip(cell, `${m.row_labels[i]} against ${m.col_labels[j]}: ${signed(v, 2)} (${verdict}) · they do this ${pct(m.p_cols[j])} of the time`);
}

function mxRow(m, i, cols, ranks, rowOrder, top, groupStart) {
  const kind = rowKind(i);
  const overridden = m.played !== m.chosen;
  const row = node("div", `mx-row is-${kind}${m.legal[i] ? "" : " is-illegal"}${i === m.played ? " is-played" : ""}${overridden && i === m.chosen ? " is-network" : ""}${groupStart ? " is-group" : ""}`);
  row.setAttribute("role", "row");
  const label = node("span", "mx-rl");
  label.append(node("span", "nm", orderLabel(kind === "switch" ? "switch" : "move", m.row_labels[i])));
  if (overridden && i === m.chosen) label.append(node("span", "mx-tag is-net", "network"));
  if (overridden && i === m.played) label.append(node("span", "mx-tag", "played"));
  tip(label, `${kind === "switch" ? `Switch to ${m.row_labels[i]}` : m.row_labels[i]}: ${m.legal[i] ? `our policy gives it ${pct(m.probs[i])}` : "not legal this turn"}${i === m.played ? " · the action played" : ""}${overridden && i === m.chosen ? " · the network's choice, replaced by a guard rule" : ""}`);
  row.append(label);
  cols.forEach((j, at) => {
    if (at > 0 && m.col_kind[j] !== m.col_kind[cols[at - 1]]) row.append(node("span", "mx-div"));
    row.append(mxCell(m, i, j, ranks[j], rowOrder));
  });
  const prob = node("span", "mx-rp");
  prob.style.setProperty("--d", `${MX_T.bars + rowOrder * MX_T.bar}ms`);
  const track = node("span", "mx-rbar");
  const fill = node("i");
  fill.style.width = `${clamp01(m.probs[i] / top) * 100}%`;
  track.append(fill);
  prob.append(track, node("b", "", m.legal[i] ? pct(m.probs[i]) : "–"));
  row.append(prob);
  return row;
}

/** The grid's column tracks: row label, their moves, a divider, their switches, our probability bar. */
function mxTemplate(m, cols) {
  const moves = cols.filter((j) => m.col_kind[j] !== "switch").length;
  const switches = cols.length - moves;
  const parts = ["var(--mx-label)"];
  if (moves) parts.push(`repeat(${moves}, minmax(0, 1fr))`);
  if (moves && switches) parts.push("var(--mx-gap)");
  if (switches) parts.push(`repeat(${switches}, minmax(0, 1fr))`);
  parts.push("var(--mx-prob)");
  return parts.join(" ");
}

function mxGrid(m) {
  const megaLegal = m.legal.slice(MX.megaAt, MX.switchAt).some(Boolean);
  const rows = m.rows.map((_, i) => i).filter((i) => m.rows[i] && (rowKind(i) !== "mega" || megaLegal));
  const cols = m.cols.map((_, j) => j).filter((j) => m.cols[j]);
  const ranks = {};
  [...cols].sort((a, b) => m.p_cols[b] - m.p_cols[a]).forEach((j, rank) => { ranks[j] = rank; });
  const grid = el("mx-grid");
  grid.style.setProperty("--mx-cols", mxTemplate(m, cols));
  grid.dataset.rows = rows.length;
  const head = node("div", "mx-head");
  const corner = node("span", "mx-corner");
  corner.append(sideTag("opp", "THEIRS →"), sideTag("you", "OURS ↓"));
  head.append(corner);
  const topCol = Math.max(0.4, ...cols.map((j) => m.p_cols[j]));
  cols.forEach((j, at) => {
    if (at > 0 && m.col_kind[j] !== m.col_kind[cols[at - 1]]) head.append(node("span", "mx-div"));
    head.append(mxColumn(m, j, at, topCol));
  });
  head.append(tip(node("span", "mx-phead", "Our policy"), "The probability the network gives each of our actions after weighing every column: the action played is outlined"));
  const body = node("div", "mx-rows");
  const topRow = Math.max(0.4, ...rows.map((i) => m.probs[i]));
  rows.forEach((i, at) => body.append(mxRow(m, i, cols, ranks, at, topRow, at > 0 && rowKind(i) !== rowKind(rows[at - 1]))));
  grid.replaceChildren(head, body);
}

/** In place of the grid when the decision has no matrix (a fallback, or a record of an older player). */
function mxAlt(decision) {
  const n = decision.notes || {};
  const alt = el("mx-alt");
  const lead = node("p", "mx-alt-lead", n.fallback
    ? "The fallback player decided this turn, so there is no matrix: the network gave no answer."
    : n.rule ? `No matrix was recorded for this decision (rule: ${n.rule}).` : "No matrix was recorded for this decision.");
  const legal = node("div", "legal");
  (decision.legal || []).forEach((o) => {
    const chosen = o.order === decision.order;
    const chip = node("span", `opt${chosen ? " is-chosen" : ""}${o.kind === "switch" ? " is-switch" : ""}`);
    chip.append(node("span", "n", orderLabel(o.kind, o.label)));
    if (chosen) chip.append(node("span", "p", "played"));
    legal.append(tip(chip, `${o.kind === "switch" ? `Switch to ${o.label}` : o.label}: ${chosen ? "the action sent" : "legal, not chosen"} (${o.order})`));
  });
  alt.replaceChildren(lead, node("div", "sub-head", `Legal actions · ${(decision.legal || []).length}`), legal);
}

/** Start the one-shot animation: the CSS under .is-playing reads each element's --d delay. */
function mxPlay(overridden) {
  const tileEl = el("t-matrix");
  window.clearTimeout(mxTimer);
  tileEl.classList.remove("is-playing");
  const ring = MX_T.ring + (overridden ? MX_T.overrideGap : 0);
  tileEl.style.setProperty("--t-net", `${MX_T.ring}ms`);
  tileEl.style.setProperty("--t-ring", `${ring}ms`);
  tileEl.style.setProperty("--t-hero", `${ring + 60}ms`);
  void tileEl.offsetWidth; // restart
  tileEl.classList.add("is-playing");
  mxTimer = window.setTimeout(() => tileEl.classList.remove("is-playing"), MX_T.done + (overridden ? MX_T.overrideGap : 0));
}

function renderMatrix(state, isPinned) {
  const decision = state.decision;
  const picked = state.preview?.names?.length ? `Team picked: ${state.preview.names.join(", ")}. The matrix appears with the first decision of the battle.` : null;
  const waiting = picked || (state.battle ? "Waiting for this battle's first decision" : "Waiting for the first decision");
  tile("t-matrix", Boolean(decision), waiting, "mx-empty");
  el("mx-legend").hidden = !decision?.notes?.matrix;
  if (!decision) {
    el("mx-meta").textContent = "";
    el("mx-badge").replaceChildren();
    drawn.matrix = null;
    return;
  }
  const key = `${state.battle?.index}:${decision.sequence}`;
  const fresh = !mxSeen.has(key);
  mxSeen.add(key);
  // The timing and the timestamp are left out: they arrive after the decision and must not redraw it mid-animation.
  if (!changed("matrix", [key, decision.notes, decision.order, decision.label, decision.kind, decision.turn, decision.degrade, decision.legal, state.active?.name])) return;
  const n = decision.notes || {};
  const m = n.matrix;
  const label = KIND[decision.kind] || "decision";
  const ours = state.active?.name;
  el("mx-kind").replaceChildren(...(ours ? [sideTag("you"), node("span", "nm", ours), node("span", "", `· ${label}`)] : [node("span", "", title(label))]));
  el("mx-action").textContent = orderLabel(decision.order_kind, decision.label || decision.order || "Decision recorded");
  tip(el("mx-action"), `The action the agent sent to the game (${decision.order})`);
  el("mx-meta").textContent = `Turn ${decision.turn ?? "--"} · ${timeLabel(decision.ts)}`;
  tip(el("mx-meta"), `Decision turn and when it was recorded${decision.reason ? ` (${decision.reason})` : ""}`);
  el("mx-badge").replaceChildren(degradeBadge(decision.degrade));
  el("mx-stats").replaceChildren(...mxStats(decision));
  mxBanner(decision);
  el("t-matrix").classList.toggle("is-fallback", Boolean(n.fallback));
  el("mx-grid").hidden = !m;
  el("mx-alt").hidden = Boolean(m);
  if (m) mxGrid(m);
  else mxAlt(decision);
  if (fresh && !isPinned) mxPlay(Boolean(m) && m.played !== m.chosen);
  else {
    window.clearTimeout(mxTimer);
    el("t-matrix").classList.remove("is-playing");
  }
}
