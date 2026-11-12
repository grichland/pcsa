"""The belief feature modes against research (skipped when research/pcsa-rl or research/pcsa-belief is absent, as
in CI): on archived simulator episodes of both seats, every decision rebuilt from the protocol (stage 0's
common.rebuild) once per implementation, pcsa's observation must equal research rl2's in each mode ("base",
"belief", "belief+mega"; one belief per episode and implementation, research's settings on both sides), and
pcsa's "base" must equal pcsa main's former_features (no belief code on that path). The full measurement
(1,579 decisions, bit-identical) is research/pcsa-belief/stage4/equiv_pcsa.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH = Path(os.environ.get("PCSA_RL_RESEARCH", r"C:\Users\geral\Code\research\pcsa-rl"))
BELIEF = RESEARCH.parent / "pcsa-belief"
STEM = RESEARCH / "runs" / "v2-teams-1111" / "archive" / "u00845"
FIELDS = ("rows", "ids", "field", "last", "last_ids", "mask")
BELIEF_COLS = (
    set(range(148, 154)) | {158 + 40 * j + 31 + c for j in range(4) for c in range(9)} | set(range(318, 330))
)

pytestmark = pytest.mark.skipif(
    not (
        (RESEARCH / "rl2" / "belief_features.py").is_file()
        and (BELIEF / "stage0" / "common.py").is_file()
        and Path(str(STEM) + ".npz").is_file()
    ),
    reason=f"no research belief code or archive at {RESEARCH}",
)


@pytest.fixture(scope="module")
def research():
    for p in (RESEARCH, BELIEF, BELIEF / "stage0"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import common  # research/pcsa-belief/stage0
    from rl2 import belief_features, features

    return common, features, belief_features


def _decisions(common, per_seat: int = 3):
    side, z = common.load_update(STEM)
    z = {k: z[k] for k in ("turn", "forced", "action", "start", "length")}
    picks = [g for g, e in enumerate(side) if e["seat"] == "p1"][:per_seat]
    picks += [g for g, e in enumerate(side) if e["seat"] == "p2"][:per_seat]
    for g in picks:
        e = side[g]
        s0, n = int(z["start"][g]), int(z["length"][g])
        lines = common.p1_lines(e)
        cuts = common.decision_cuts(lines, z["turn"][s0 : s0 + n], z["forced"][s0 : s0 + n], z["action"][s0 : s0 + n])
        if cuts:
            yield g, lines, cuts, common.match_sets(common.parse_team(e["team"]), lines, "p1")


def _same(a, b) -> bool:
    return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in FIELDS) and (a.turn, a.forced) == (
        b.turn,
        b.forced,
    )


@pytest.mark.parametrize("mode", ["base", "belief", "belief+mega"])
def test_pcsa_observations_equal_research_in_every_feature_mode(research, mode: str, record_property) -> None:
    from pcsa.players import mirror
    from pcsa.rl_training import belief_features as pbf
    from pcsa.rl_training import former_features as pf

    common, rf, rbf = research
    n = differ = 0
    changed: set = set()
    for g, lines, cuts, ours in _decisions(common):
        pb = rb = None
        if mode != "base":
            pb, rb = pbf.new_battle_belief(crit_reported=True, empty_unknown=False), rbf.new_battle_belief()
        for cut in cuts:
            bp, br, b0 = (common.rebuild(lines[: cut.end], ours, f"e{g}", cut.forced) for _ in range(3))
            if mode == "base":
                mine, theirs = pf.observe(bp, mirror.legal_actions(bp)), rf.observe(br, mirror.legal_actions(br))
            else:
                mine = pf.observe(bp, mirror.legal_actions(bp), belief=pb, feature_mode=mode)
                theirs = rf.observe(br, mirror.legal_actions(br), belief=rb, feature_mode=mode)
            assert _same(mine, theirs), (mode, g, cut.turn, cut.forced)
            base = pf.observe(b0, mirror.legal_actions(b0))
            differ += int(not np.array_equal(base.rows, mine.rows))
            changed |= set(np.nonzero(base.rows != mine.rows)[1].tolist())
            n += 1
    record_property("decisions", n)
    assert n > 30
    if mode == "base":
        assert differ == 0
    else:  # the belief columns, and only those, changed at most decisions
        assert differ > n * 0.8 and changed <= BELIEF_COLS and changed & set(range(318, 330))


def test_base_equals_pcsa_main(research, tmp_path) -> None:
    """pcsa's base observation equals main's former_features.py (read with git; skipped without the main repo)."""
    import importlib.util
    import subprocess

    from pcsa.players import mirror
    from pcsa.rl_training import former_features as pf

    main_repo = Path(os.environ.get("PCSA_MAIN_REPO", r"C:\Users\geral\Code\pcsa"))
    try:
        src = subprocess.run(
            ["git", "-C", str(main_repo), "show", "main:src/pcsa/rl_training/former_features.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no pcsa main repository to compare with")
    path = tmp_path / "former_features_main.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("former_features_main", path)
    assert spec is not None and spec.loader is not None
    main = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = main
    spec.loader.exec_module(main)
    common = research[0]
    n = 0
    for g, lines, cuts, ours in _decisions(common, 2):
        for cut in cuts:
            a, b = (common.rebuild(lines[: cut.end], ours, f"m{g}", cut.forced) for _ in range(2))
            assert _same(pf.observe(a, mirror.legal_actions(a)), main.observe(b, mirror.legal_actions(b)))
            n += 1
    assert n > 15
