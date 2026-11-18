"""CG's live session with the Battle and the network's outputs in each reply: a drop-in for `python -m
champions_gym.live`.

    python -m pcsa.cg.serve --player tower [--seed N] [--log-dir D] [--player-args JSON]

Same flags and protocol: one JSON message per line on stdin, exactly one line per input line on stdout (the reply,
or `null`), flushed per line; nothing else on stdout; diagnostics on stderr; exit code 2 when it cannot start.

The player is built here (`champions_gym.neural.load(**player_args)` for `champions_gym.neural:load`, else a
built-in name or "module:attr" as CG's Session builds it) and a Player object is wrapped in a Proxy that keeps the
Battle it was asked about and forwards every hook. To each picks or order reply this adds, in `notes`:
- `view`: that Battle (pcsa/cg/view.py), for built-in players too
- `net`: the network's outputs (record contract), only when the neural player decided this very request itself
Everything added is plain JSON; the reply's own keys are left as they are.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from champions_gym import live

from pcsa.cg.view import view

NEURAL = "champions_gym.neural:load"
PREVIEW_DECISION = -1


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class Proxy:
    """The player as CG's Session sees it: every attribute and hook forwarded; `battle` is the Battle of the last
    request it was asked to decide. `status()` also carries the checkpoint's run and iteration."""

    def __init__(self, player: Any) -> None:
        self.player = player
        self.name = str(getattr(player, "name", None) or type(player).__name__)
        self.battle: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.player, name)

    def teampreview(self, battle: Any) -> Any:
        self.battle = battle
        return self.player.teampreview(battle)

    def choose_move(self, battle: Any) -> Any:
        self.battle = battle
        choose = getattr(self.player, "choose_move", None) or self.player.choose
        return choose(battle)

    def status(self) -> Any:
        found = getattr(self.player, "status", None)
        if not callable(found):
            return None
        out = found()
        info = getattr(self.player, "info", None)
        if isinstance(out, dict) and isinstance(info, dict):
            out = {**out, "run": _plain(info.get("run")), "iteration": _plain(info.get("iteration"))}
        return out


class ServeSession(live.Session):
    """CG's Session that also keeps the Battle a built-in player is asked about (`captured`)."""

    captured: Any = None

    def _ask(self, seq: int, kind: int) -> str:
        if self._builtin is not None:
            self.captured = self._core.battle()
        return super()._ask(seq, kind)


def make_player(spec: str, player_args: dict | None) -> Any:
    """A built-in name (checked), or a Player object: the neural player, or "module:attr" called with the args."""
    if spec == NEURAL:
        from champions_gym import neural

        return neural.load(**(player_args or {}))
    return live.load_player(spec, player_args)


class Server:
    """One Session; `handle(message)` is the Session's reply with `view` and `net` added."""

    def __init__(self, player: Any, seed: int = 0, log_dir: str | None = None) -> None:
        self.proxy = None if isinstance(player, str) else Proxy(player)
        self.session = ServeSession(player=self.proxy or player, seed=seed, log_dir=log_dir)

    def _last(self) -> Any:
        return getattr(self.proxy.player, "last", None) if self.proxy is not None else None

    def handle(self, message: Any) -> dict | None:
        before = self._last()
        self.session.captured = None
        if self.proxy is not None:
            self.proxy.battle = None
        reply = self.session.handle(message)
        if not isinstance(reply, dict) or "error" in reply or not ("order" in reply or "picks" in reply):
            return reply
        battle = self.proxy.battle if self.proxy is not None else self.session.captured
        if battle is None:
            return reply  # answered without asking: no Battle of a decision
        notes = reply.get("notes")
        if not isinstance(notes, dict):
            notes = reply["notes"] = {}
        notes["view"] = view(battle)
        net = self._net(reply, before, battle)
        if net is not None:
            notes["net"] = net
        return reply

    def _net(self, reply: dict, before: Any, battle: Any) -> dict | None:
        """The neural player's outputs for this request; None when it did not run the network for it (a replayed
        or repeated request leaves `last` as it was; a fallback before the forward pass has no extras)."""
        last = self._last()
        seq = PREVIEW_DECISION if "picks" in reply else reply.get("seq")
        if not isinstance(last, dict) or last is before or last.get("decision") != seq:
            return None
        extras = last.get("extras")
        if not isinstance(extras, dict):
            return None
        net = {"decision": seq, "kind": last.get("kind"), "legal": last.get("legal"), "value": last.get("value"),
               **extras}  # fmt: skip
        if isinstance(net.get("belief"), list):
            net["belief"] = species_of_belief(net["belief"], battle)
        return _plain(net)


def species_of_belief(belief: list, battle: Any) -> list:
    """Belief entries carry the network's row species, which is the current form (a Mega's); each is named by the
    opposing member's species instead, so it matches `view.opp`."""
    by_form = {}
    for p in battle.opponent_team:
        if p.species:
            by_form.setdefault(p.species, p.species)
            if p.form:
                by_form.setdefault(p.form, p.species)
    return [{**e, "species": by_form.get(e.get("species"), e.get("species"))} if isinstance(e, dict) else e
            for e in belief]  # fmt: skip


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m pcsa.cg.serve", description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--player", default="tower", help="a built-in player's name, or module:Class (or module:factory)")
    ap.add_argument("--player-args", default=None, help="JSON object of keyword arguments for module:Class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    stdin = open(sys.stdin.fileno(), encoding="utf-8", errors="replace", newline=None, closefd=False)  # noqa: SIM115
    stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n", closefd=False)  # noqa: SIM115
    try:
        player = make_player(a.player, json.loads(a.player_args) if a.player_args else None)
        server = Server(player, seed=a.seed, log_dir=a.log_dir)
    except Exception as error:
        sys.stderr.write(f"pcsa.cg.serve: {error}\n")
        return 2
    sys.stderr.write(f"pcsa.cg.serve: player {server.session.player_name}, seed {a.seed}, log {a.log_dir}\n")
    sys.stderr.flush()
    with server.session:
        for line in stdin:
            line = line.rstrip("\r\n")
            reply = server.handle(line) if line.strip() else {"error": "empty line"}
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
