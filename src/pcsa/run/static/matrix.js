/* pcsa watch: the decision matrix. Their kept actions run across the top (4 moves, then 5 switches) with the
   probability of each; our 11 actions run down the side (4 moves, the same 4 with Mega Evolution, 3 switches).
   A cell says how good our action is if they answer with that column. SCORE is a row weighed by their
   probabilities, the bar on the right is our final policy, and FOR THEM is a column weighed by our policy.
   "Their actions considered" weighs only their N most likely actions and shows what we would then pick.
   "Trust in the prediction" takes their probabilities at another temperature first (higher: flatter, a hedge).
   A new decision plays once: columns, cells (most likely column first), row bars, then the row played. */
const MX = { megaAt: 4, switchAt: 8, reach: 1.2 };
const MX_T = { col: 35, cells: 420, rank: 55, row: 7, bars: 960, bar: 22, ring: 1380, overrideGap: 620, done: 3200 };
const MX_TOPS = [0, 5, 3, 2, 1]; // their actions considered: 0 is all of them
const MX_TOP_KEY = "pcsa-mx-top";
const MX_OPPS = [0.5, 0.75, 1, 1.5, 2, 3, 5]; // trust in the prediction: the temperature on their probabilities
const MX_OPP_KEY = "pcsa-mx-opp";
const KIND = { forced: "forced switch", action: "turn action", team: "team preview" };
const mxSeen = new Set(); // "battle:sequence" of every decision already drawn: none of them animates again
let mxTimer = null;
let mxTop = mxReadTop();
let mxOpp = mxReadOpp(); // null: their probabilities as recorded
let mxShown = null; // the matrix on screen: the controls redraw it without a new decision

const orderLabel = (kind, label) => (kind === "switch" ? `⇄ ${label}` : label);
const rowKind = (i) => (i >= MX.switchAt ? "switch" : i >= MX.megaAt ? "mega" : "move");
const rowName = (m, i) => orderLabel(rowKind(i) === "switch" ? "switch" : "move", m.row_labels[i]);
const topName = (top) => (top ? `Top ${top}` : "All");
const mxT = (t) => String(Number(t.toFixed(2))); // a temperature: 0.75, 1.5, 2
const mxTrec = (m) => (isNum(m.opp_temperature) && m.opp_temperature > 0 ? m.opp_temperature : 1); // old records: 1
/** The name of the what-if in use: All, Top 3, Trust ×2, Top 3 at trust ×2. */
const whatName = (what) => (what.reweighted ? `${what.top ? `${topName(what.top)} at trust` : "Trust"} ×${mxT(what.opp)}` : topName(what.top));
const mxNum = (v, digits = 2) => signed(Number(v.toFixed(digits)), digits); // never "−0.00"
const mxBest = (keys, of) => keys.reduce((best, k) => (best === null || of[k] > of[best] ? k : best), null);

/** The choice of "their actions considered": ?top=N in the address, else the one remembered, else all. */
function mxReadTop() {
  let raw = new URLSearchParams(location.search).get("top");
  try { raw = raw ?? localStorage.getItem(MX_TOP_KEY); } catch (e) { /* storage blocked: all of them */ }
  const top = raw === null || raw === "" ? 0 : Number(raw);
  return MX_TOPS.includes(top) ? top : 0;
}

/** The choice of "trust in the prediction": ?opp=T in the address (any temperature above zero), else the step
    remembered, else null (as recorded). */
function mxReadOpp() {
  const asked = new URLSearchParams(location.search).get("opp");
  if (asked !== null && asked.trim() !== "" && Number.isFinite(Number(asked)) && Number(asked) > 0) return Number(asked);
  let kept = null;
  try { kept = localStorage.getItem(MX_OPP_KEY); } catch (e) { /* storage blocked: as recorded */ }
  return kept !== null && kept !== "" && MX_OPPS.includes(Number(kept)) ? Number(kept) : null;
}

/** What the matrix says with their probabilities taken at the temperature `opp` (null: as recorded) and only their
    `top` most likely columns counted (0: all of them, as the network does).
    base: p_j as recorded or, at another temperature, p_j ^ (T_rec / opp) scaled to add up to 1 over the columns
    (softmax(logits / T) ∝ p ^ (1 / T), so this is exact); weight: base_j, or base_j / Σ base of the `top` columns kept;
    score_i = Σ_j weight_j · cells[i][j] over the legal rows; best: the row with the highest score;
    policy: softmax(score / tau) over the legal rows, when tau is known. */
function mxWhatIf(m, cols, top, opp = null) {
  const trec = mxTrec(m);
  const reweighted = opp !== null && opp !== trec;
  const base = {};
  cols.forEach((j) => { base[j] = reweighted ? Math.max(0, m.p_cols[j]) ** (trec / opp) : m.p_cols[j]; });
  if (reweighted) {
    const total = cols.reduce((sum, j) => sum + base[j], 0);
    cols.forEach((j) => { base[j] = total > 0 ? base[j] / total : 1 / cols.length; });
  }
  const ranked = [...cols].sort((a, b) => base[b] - base[a] || m.p_cols[b] - m.p_cols[a] || a - b);
  const used = top ? ranked.slice(0, top) : ranked;
  const mass = used.reduce((sum, j) => sum + base[j], 0);
  const weight = {};
  used.forEach((j) => { weight[j] = !top ? base[j] : mass > 0 ? base[j] / mass : 1 / used.length; });
  const legal = m.legal.map((ok, i) => (ok && m.rows[i] ? i : -1)).filter((i) => i >= 0);
  const score = {};
  legal.forEach((i) => { score[i] = used.reduce((sum, j) => sum + weight[j] * m.cells[i][j], 0); });
  const best = mxBest(legal, score);
  let policy = null;
  if (isNum(m.tau) && m.tau > 0 && best !== null) {
    const raised = legal.map((i) => Math.exp((score[i] - score[best]) / m.tau));
    const total = raised.reduce((sum, x) => sum + x, 0);
    policy = {};
    legal.forEach((i, at) => { policy[i] = raised[at] / total; });
  }
  return { top, opp, trec, reweighted, differs: Boolean(top) || reweighted, used: new Set(used), weight, legal, score, best, policy };
}

/** In words, which of their actions the what-if weighs and how. */
function mxScope(what) {
  const which = what.top ? (what.top === 1 ? "only their most likely action" : `only their ${what.top} most likely actions`) : "all their actions";
  const scaled = what.top > 1 ? ", scaled to add up to 100%" : "";
  return what.reweighted ? `${which}, their probabilities taken at temperature ${mxT(what.opp)} instead of the recorded ${mxT(what.trec)}${scaled}` : `${which}${scaled}`;
}

/** Each column from their side: value_j = −Σ_i probs[i] · cells[i][j] over our legal rows (what it brings them
    against our policy), mean_j = −(plain mean of the column), their best column and the one we think most likely. */
function mxForThem(m, legal, cols) {
  const value = {};
  const mean = {};
  cols.forEach((j) => {
    value[j] = -legal.reduce((sum, i) => sum + m.probs[i] * m.cells[i][j], 0);
    mean[j] = legal.length ? -legal.reduce((sum, i) => sum + m.cells[i][j], 0) / legal.length : 0;
  });
  return { value, mean, best: legal.length ? mxBest(cols, value) : null, likely: mxBest(cols, m.p_cols) };
}

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

/** The readout next to the control: the row with the highest SCORE, and whether the network chose another. */
function mxPickLine(m, g) {
  const line = el("mx-pick");
  const { what, ref, differs } = g;
  line.className = `mx-pick${differs ? " is-diff" : ""}`;
  if (what.best === null) return line.replaceChildren();
  const label = node("small"); // TOP 3 AT TRUST ×2 WOULD PICK: the settings are dropped when room is short
  if (differs) label.append(node("span", "mx-pick-what", `${whatName(what).toUpperCase()} `));
  label.append("WOULD PICK");
  const tail = differs ? `instead of ${rowName(m, ref)}` : ref === null ? "" : "same as the network";
  line.replaceChildren(label, node("b", "", rowName(m, what.best)), node("span", "", tail));
  const scope = mxScope(what);
  const verdict = differs ? `; the network chose ${rowName(m, ref)} (SCORE ${mxNum(what.score[ref] ?? 0)} here)` : ref === null ? "" : ", the action the network chose";
  const noTau = what.policy ? "" : " This record has no temperature, so the policy under this setting cannot be shown.";
  return tip(line, `Weighing ${scope}, the highest SCORE is ${rowName(m, what.best)} (${mxNum(what.score[what.best])})${verdict}.${noTau}`);
}

function mxColumn(m, j, order, top, what) {
  const used = what.used.has(j);
  const col = node("div", `mx-col is-${m.col_kind[j] === "switch" ? "switch" : "move"}${used ? "" : " is-off"}`);
  col.style.setProperty("--d", `${order * MX_T.col}ms`);
  const nums = node("span", "mx-cnums");
  nums.append(node("b", "mx-cnum", pct(m.p_cols[j])));
  const bars = node("span", "mx-cbars");
  const bar = node("span", "mx-cbar");
  const fill = node("i");
  fill.style.height = `${clamp01(m.p_cols[j] / top) * 100}%`;
  bar.append(fill);
  bars.append(bar);
  if (what.differs) { // the weight in use: its number and a thin bar next to the recorded ones
    nums.append(node("b", "mx-cnum alt", used ? pct(what.weight[j]) : "–"));
    const alt = node("span", "mx-calt");
    const altFill = node("i");
    altFill.style.height = `${clamp01((what.weight[j] ?? 0) / top) * 100}%`;
    alt.append(altFill);
    bars.append(alt);
  }
  const name = node("span", "mx-cname");
  if (m.shown[j]) name.append(node("i", "dot-shown"));
  name.append(node("span", "", m.col_labels[j]));
  col.append(nums, bars, name);
  const does = m.col_kind[j] === "switch" ? `They switch to ${m.col_labels[j]}` : `They use ${m.col_labels[j]}`;
  const counted = !what.differs ? "" : used ? ` · counts for ${pct(what.weight[j])} under ${whatName(what)}` : ` · left out under ${whatName(what)}`;
  return tip(col, `${does}: ${pct(m.p_cols[j])} likely${m.shown[j] ? " (a move they have already shown)" : m.col_kind[j] === "switch" ? "" : " (not shown yet: the network expects it)"}${counted}`);
}

function mxCell(m, i, j, rank, rowOrder, what) {
  const v = m.cells[i][j];
  const used = what.used.has(j);
  const cell = node("span", `mx-cell${used ? "" : " is-off"}`);
  cell.tabIndex = -1;
  cell.setAttribute("role", "cell");
  if (!m.legal[i]) return cell;
  cell.textContent = signed(v, 1);
  const a = Math.min(1, Math.abs(v) / MX.reach);
  if (Math.abs(v) > 0.03) {
    cell.classList.add(v > 0 ? "pos" : "neg");
    cell.style.setProperty("--a", `${Math.round((0.14 + 0.78 * a ** 0.85) * 100)}%`);
  }
  cell.style.setProperty("--d", `${MX_T.cells + rank * MX_T.rank + rowOrder * MX_T.row}ms`);
  const verdict = v > 0.03 ? "good for us" : v < -0.03 ? "bad for us" : "even";
  return tip(cell, `${m.row_labels[i]} against ${m.col_labels[j]}: ${signed(v, 2)} (${verdict}) · they do this ${pct(m.p_cols[j])} of the time${!what.differs ? "" : used ? ` · counts for ${pct(what.weight[j])} under ${whatName(what)}` : ` · left out under ${whatName(what)}`}`);
}

/** The row's SCORE: its cells weighed by the probabilities of their actions in use. */
function mxScore(m, i, rowOrder, what) {
  const cell = node("b", `mx-sc${i === what.best ? " is-best" : ""}`);
  cell.setAttribute("role", "cell");
  cell.style.setProperty("--d", `${MX_T.bars + rowOrder * MX_T.bar}ms`);
  if (!(i in what.score)) return cell;
  cell.textContent = mxNum(what.score[i]);
  const scope = what.differs ? ` (${mxScope(what)}: ${whatName(what)} is on)` : "";
  return tip(cell, `${m.row_labels[i]} ${mxNum(what.score[i])}: what this action is worth on average against what we expect them to do${scope}${i === what.best ? " · the highest score" : ""}`);
}

/** Our policy for the row: the network's own bar and number and, when the temperature is known, the policy the
    current "their actions considered" would give (a thin bar under it, and its number). */
function mxPolicy(m, i, rowOrder, g) {
  const { what, topRow } = g;
  const prob = node("span", "mx-rp");
  prob.style.setProperty("--d", `${MX_T.bars + rowOrder * MX_T.bar}ms`);
  const bars = node("span", "mx-rbars");
  const track = node("span", "mx-rbar");
  const fill = node("i");
  fill.style.width = `${clamp01(m.probs[i] / topRow) * 100}%`;
  track.append(fill);
  bars.append(track);
  prob.append(bars, node("b", "", m.legal[i] ? pct(m.probs[i]) : "–"));
  if (!what.policy) return m.legal[i] ? tip(prob, `${m.row_labels[i]}: our policy gives it ${pct(m.probs[i])}`) : prob;
  const would = what.policy[i];
  const alt = node("span", "mx-ralt");
  const altFill = node("i");
  altFill.style.width = `${clamp01((would ?? 0) / topRow) * 100}%`;
  alt.append(altFill);
  bars.append(alt);
  prob.append(node("b", "alt", isNum(would) ? pct(would) : ""));
  if (!isNum(would)) return prob;
  const how = what.differs
    ? `weighing ${mxScope(what)}, it would be ${pct(would)}`
    : `worked out again from the SCORE column it is ${pct(would)} (the network also caps an action at 99% and its figures are rounded)`;
  return tip(prob, `${m.row_labels[i]}: our policy gives it ${pct(m.probs[i])} · ${how} · softmax of SCORE ÷ ${m.tau.toFixed(2)}, the network's temperature`);
}

function mxRow(m, i, rowOrder, g, groupStart) {
  const kind = rowKind(i);
  const overridden = m.played !== m.chosen;
  const pick = g.differs && i === g.what.best;
  const row = node("div", `mx-row is-${kind}${m.legal[i] ? "" : " is-illegal"}${i === m.played ? " is-played" : ""}${overridden && i === m.chosen ? " is-network" : ""}${pick ? " is-pick" : ""}${groupStart ? " is-group" : ""}`);
  row.setAttribute("role", "row");
  const label = node("span", "mx-rl");
  label.append(node("span", "nm", rowName(m, i)));
  if (pick && i !== m.played) label.append(node("span", "mx-tag is-pick", "would pick")); // the played row keeps its one tag
  if (overridden && i === m.chosen) label.append(node("span", "mx-tag is-net", "network"));
  if (overridden && i === m.played) label.append(node("span", "mx-tag", "played"));
  tip(label, `${kind === "switch" ? `Switch to ${m.row_labels[i]}` : m.row_labels[i]}: ${m.legal[i] ? `our policy gives it ${pct(m.probs[i])}` : "not legal this turn"}${i === m.played ? " · the action played" : ""}${overridden && i === m.chosen ? " · the network's choice, replaced by a guard rule" : ""}${pick ? ` · the highest SCORE under ${whatName(g.what)}: what we would pick` : ""}`);
  row.append(label);
  g.cols.forEach((j, at) => {
    if (at > 0 && m.col_kind[j] !== m.col_kind[g.cols[at - 1]]) row.append(node("span", "mx-div"));
    row.append(mxCell(m, i, j, g.ranks[j], rowOrder, g.what));
  });
  row.append(mxScore(m, i, rowOrder, g.what), mxPolicy(m, i, rowOrder, g));
  return row;
}

/** The row under the grid: each of their actions from their side, weighed by our policy. */
function mxFoot(m, g, rowOrder) {
  const { them, cols } = g;
  const row = node("div", "mx-row mx-foot");
  row.setAttribute("role", "row");
  row.style.setProperty("--d", `${MX_T.bars + rowOrder * MX_T.bar}ms`);
  const label = node("span", "mx-fl");
  label.append(tip(node("span", "side opp", "FOR THEM"), "How each of their actions turns out for them, on average over what we are likely to do (our policy): red is good for them. It does not depend on \"their actions considered\" or on the trust in the prediction."));
  if (them.best !== null) {
    const same = them.best === them.likely;
    const note = node("span", `mx-fnote ${same ? "is-same" : "is-diff"}`);
    note.append(node("b", "", same ? "=" : "≠"), node("span", "", "most likely"));
    label.append(tip(note, same
      ? `Their best action against our policy (${m.col_labels[them.best]}, ${mxNum(them.value[them.best])}) is also the one we think most likely (${pct(m.p_cols[them.likely])})`
      : `Their best action against our policy is ${m.col_labels[them.best]} (${mxNum(them.value[them.best])}), but the one we think most likely is ${m.col_labels[them.likely]} (${pct(m.p_cols[them.likely])}, ${mxNum(them.value[them.likely])} for them)`));
  }
  row.append(label);
  cols.forEach((j, at) => {
    if (at > 0 && m.col_kind[j] !== m.col_kind[cols[at - 1]]) row.append(node("span", "mx-div"));
    const v = them.value[j];
    const cell = node("b", `mx-fc${j === them.best ? " is-best" : ""}`);
    cell.setAttribute("role", "cell");
    if (them.best === null) return row.append(cell);
    cell.textContent = mxNum(v);
    if (v > 0.03) {
      cell.classList.add("pos");
      cell.style.setProperty("--a", `${Math.round((0.14 + 0.78 * Math.min(1, v / MX.reach) ** 0.85) * 100)}%`);
    }
    return row.append(tip(cell, `${m.col_labels[j]} ${mxNum(v)}: if they do this, how it turns out for them on average over what we are likely to do · plain average over our ${g.what.legal.length} legal actions: ${mxNum(them.mean[j])}${j === them.best ? " · their best action" : ""}${j === them.likely ? " · the one we think most likely" : ""}`));
  });
  row.append(node("span", "mx-sc"), node("span", "mx-rp"));
  return row;
}

/** The grid's column tracks: row label, their moves, a divider, their switches, the score, our probability bar. */
function mxTemplate(m, cols) {
  const moves = cols.filter((j) => m.col_kind[j] !== "switch").length;
  const switches = cols.length - moves;
  const parts = ["var(--mx-label)"];
  if (moves) parts.push(`repeat(${moves}, minmax(0, 1fr))`);
  if (moves && switches) parts.push("var(--mx-gap)");
  if (switches) parts.push(`repeat(${switches}, minmax(0, 1fr))`);
  parts.push("var(--mx-score)", "var(--mx-prob)");
  return parts.join(" ");
}

/** The headings of the two columns after the grid: SCORE and our policy (with the key of its two bars). */
function mxTotalsHead(what) {
  const score = tip(node("span", "mx-shead", "SCORE"), `Each cell of the row times the probability of its column, added up over ${mxScope(what)}: what this action is worth on average against what we expect them to do. The highest is marked.`);
  const policy = node("span", "mx-phead");
  policy.append(node("span", "", "Our policy"));
  if (what.policy) {
    const key = node("span", "mx-pkey");
    const net = node("span");
    net.append(node("i", "k-net"), "network");
    const own = node("span");
    own.append(node("i", "k-alt"), `${topName(what.top)}${what.reweighted ? ` · ×${mxT(what.opp)}` : ""}`); // the names on the controls: All, Top 3 · ×2
    key.append(net, own);
    policy.append(key);
  }
  const alt = !what.policy ? "" : what.differs
    ? ` The thin bar and the second number: the policy it would have weighing ${mxScope(what)}.`
    : " The thin bar and the second number: the same policy worked out again from the SCORE column (it should match closely).";
  return [score, tip(policy, `The probability the network gives each of our actions after weighing every column: the action played is outlined.${alt}`)];
}

function mxGrid(m) {
  const megaLegal = m.legal.slice(MX.megaAt, MX.switchAt).some(Boolean);
  const rows = m.rows.map((_, i) => i).filter((i) => m.rows[i] && (rowKind(i) !== "mega" || megaLegal));
  // A column they cannot take is not drawn: an empty slot, or a switch target that has fainted (col_legal from the
  // player; records without it: a switch slot the network gave exactly 0).
  const open = (j) => (Array.isArray(m.col_legal) ? m.col_legal[j] !== false : !(m.col_kind[j] === "switch" && m.p_cols[j] === 0));
  const cols = m.cols.map((_, j) => j).filter((j) => m.cols[j] && open(j));
  const ranks = {};
  [...cols].sort((a, b) => m.p_cols[b] - m.p_cols[a]).forEach((j, rank) => { ranks[j] = rank; });
  const what = mxWhatIf(m, cols, mxTop, mxOpp);
  mxPaintOpp(what.trec);
  const ref = m.chosen ?? m.played ?? null; // the network's own row
  const topRow = Math.max(0.4, ...rows.map((i) => m.probs[i]), ...Object.values(what.policy || {}));
  const g = { cols, ranks, what, ref, topRow, differs: ref !== null && what.best !== null && what.best !== ref, them: mxForThem(m, what.legal, cols) };
  const grid = el("mx-grid");
  grid.style.setProperty("--mx-cols", mxTemplate(m, cols));
  grid.dataset.rows = rows.length;
  grid.classList.toggle("has-alt", Boolean(what.policy));
  grid.classList.toggle("has-w", what.differs);
  const head = node("div", "mx-head");
  const corner = node("span", "mx-corner");
  corner.append(sideTag("opp", "THEIRS →"), sideTag("you", "OURS ↓"));
  head.append(corner);
  const topCol = Math.max(0.4, ...cols.map((j) => m.p_cols[j]), ...Object.values(what.weight));
  cols.forEach((j, at) => {
    if (at > 0 && m.col_kind[j] !== m.col_kind[cols[at - 1]]) head.append(node("span", "mx-div"));
    head.append(mxColumn(m, j, at, topCol, what));
  });
  head.append(...mxTotalsHead(what));
  const body = node("div", "mx-rows");
  rows.forEach((i, at) => body.append(mxRow(m, i, at, g, at > 0 && rowKind(i) !== rowKind(rows[at - 1]))));
  body.append(mxFoot(m, g, rows.length));
  grid.replaceChildren(head, body);
  mxPickLine(m, g);
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

function mxStop() {
  window.clearTimeout(mxTimer);
  el("t-matrix").classList.remove("is-playing");
}

function mxPaintTop() {
  el("mx-top").querySelectorAll("button").forEach((button) => button.setAttribute("aria-pressed", String(Number(button.dataset.top) === mxTop)));
}

/** The control of the trust in the prediction: the temperature in use (marked when it is the recorded one, purple
    when it is not), the steps left on each side, and the way back to "as recorded" once a step is held. */
function mxPaintOpp(trec = mxShown ? mxTrec(mxShown) : 1) {
  const now = mxOpp ?? trec;
  const used = `The recorded decision used ${mxT(trec)}.`;
  tip(el("mx-opp-label"), `Temperature on our prediction of their action. Higher = hedge against their less likely actions; lower = trust the favourite more. ${used}`);
  const shown = el("mx-opp-now");
  shown.className = `mx-opp-now ${now === trec ? "is-rec" : "is-alt"}`;
  shown.replaceChildren(node("b", "", `×${mxT(now)}`), ...(now === trec ? [node("small", "", "recorded")] : []));
  tip(shown, now === trec
    ? `Their probabilities at temperature ${mxT(now)}, the one the recorded decision used: nothing changes${mxOpp === null ? "" : " (this step is held for every decision)"}`
    : `Their probabilities at temperature ${mxT(now)}: ${now > trec ? "flatter, a hedge against their less likely actions" : "sharper, more trust in the favourite"}. ${used}`);
  el("mx-opp").querySelectorAll("button[data-step]").forEach((button) => {
    const way = Number(button.dataset.step);
    if (way) button.disabled = !MX_OPPS.some((t) => (way < 0 ? t < now : t > now));
    else button.hidden = mxOpp === null;
  });
}

/** One step of the temperature down (−1) or up (+1) from the one in use, or back to "as recorded" (0). */
function mxStepOpp(way) {
  if (!way) return mxSetOpp(null);
  const now = mxOpp ?? (mxShown ? mxTrec(mxShown) : 1);
  const next = way < 0 ? MX_OPPS.filter((t) => t < now).pop() : MX_OPPS.find((t) => t > now);
  return next === undefined ? undefined : mxSetOpp(next);
}

/** The matrix on screen drawn again after a control changed. It is not a new decision, so nothing plays again
    (an animation still running is cut short). */
function mxRedraw() {
  if (!mxShown) return;
  mxStop();
  mxGrid(mxShown);
  armTips();
}

/** A new choice of "their actions considered": remembered, and the matrix is drawn again at once. */
function mxSetTop(top) {
  if (!MX_TOPS.includes(top) || top === mxTop) return;
  mxTop = top;
  try { localStorage.setItem(MX_TOP_KEY, String(top)); } catch (e) { /* storage blocked: the choice lasts this visit */ }
  mxPaintTop();
  mxRedraw();
}

/** A new choice of the trust in the prediction (null: as recorded): remembered, and the matrix is drawn again. */
function mxSetOpp(opp) {
  if ((opp !== null && !MX_OPPS.includes(opp)) || opp === mxOpp) return;
  mxOpp = opp;
  try {
    if (opp === null) localStorage.removeItem(MX_OPP_KEY);
    else localStorage.setItem(MX_OPP_KEY, String(opp));
  } catch (e) { /* storage blocked: the choice lasts this visit */ }
  mxPaintOpp();
  mxRedraw();
}

function renderMatrix(state, isPinned) {
  const decision = state.decision;
  const picked = state.preview?.names?.length ? `Team picked: ${state.preview.names.join(", ")}. The matrix appears with the first decision of the battle.` : null;
  const waiting = picked || (state.battle ? "Waiting for this battle's first decision" : "Waiting for the first decision");
  tile("t-matrix", Boolean(decision), waiting, "mx-empty");
  el("mx-legend").hidden = !decision?.notes?.matrix;
  el("mx-what").hidden = !decision?.notes?.matrix;
  if (!decision) {
    el("mx-meta").textContent = "";
    el("mx-badge").replaceChildren();
    drawn.matrix = null;
    mxShown = null;
    return;
  }
  const key = `${state.battle?.index}:${decision.sequence}`;
  const fresh = !mxSeen.has(key);
  mxSeen.add(key);
  // The timing and the timestamp are left out: they arrive after the decision and must not redraw it mid-animation.
  if (!changed("matrix", [key, decision.notes, decision.order, decision.label, decision.kind, decision.turn, decision.degrade, decision.legal, state.active?.name])) return;
  const n = decision.notes || {};
  const m = n.matrix;
  mxShown = m || null;
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
  else mxStop();
}

document.addEventListener("DOMContentLoaded", () => {
  mxPaintTop();
  el("mx-top").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-top]");
    if (button) mxSetTop(Number(button.dataset.top));
  });
  mxPaintOpp();
  el("mx-opp").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-step]");
    if (button && !button.disabled) mxStepOpp(Number(button.dataset.step));
  });
});
