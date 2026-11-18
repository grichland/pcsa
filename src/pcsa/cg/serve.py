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
Everything added is plain JSON and display only: a failure while adding it leaves CG's reply as it was, is written
to stderr and counted (`status.serve.failures`); the reply's own keys are never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pcsa.cg.view import view

try:
    from champions_gym import live as _live
except ImportError as error:  # reported by main(); the module stays importable
    _live, CG_MISSING = None, f"{type(error).__name__}: {error}"
else:
    CG_MISSING = None

live: Any = _live
NEURAL = "champions_gym.neural:load"
PREVIEW_DECISION = -1
TORCH_HINT = "the neural player needs torch: uv sync --inexact --extra torch"
CG_HINT = ("Champions Gym (champions_gym) is not installed: it is a pcsa dependency on Windows x64 with Python 3.12 "
           "(uv sync); there is no wheel for other systems")  # fmt: skip


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _say(text: str) -> None:
    try:
        sys.stderr.write(f"pcsa.cg.serve: {text}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


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
            try:
                out = {**out, "run": _plain(info.get("run")), "iteration": _plain(info.get("iteration"))}
            except (TypeError, ValueError) as error:
                _say(f"status: run/iteration left out ({error})")
        return out


def builtin_battle(session: Any) -> Any:
    """The Battle a built-in player is about to be asked about; None when CG's internals are not as expected.
    Relies on two CG 0.4.0 Session internals: `_builtin` (the built-in player's name) and `_core.battle()` (the
    LiveCore's Battle at the open request)."""
    try:
        return session._core.battle() if session._builtin is not None else None
    except Exception as error:
        _say(f"no Battle for the built-in player ({type(error).__name__}: {error})")
        return None


_SessionBase: Any = live.Session if live is not None else object


class ServeSession(_SessionBase):
    """CG's Session that also keeps the Battle a built-in player is asked about (`captured`), only when
    `capture_builtin` is set (never for a Player object, whose Battle the Proxy keeps)."""

    captured: Any = None
    capture_builtin = False

    def _ask(self, seq: int, kind: int) -> str:
        if self.capture_builtin:
            self.captured = builtin_battle(self)
        return super()._ask(seq, kind)  # type: ignore[misc]


def make_player(spec: str, player_args: dict | None) -> Any:
    """A built-in name (checked), or a Player object: the neural player, or "module:attr" called with the args."""
    if spec == NEURAL:
        from champions_gym import neural

        return neural.load(**(player_args or {}))
    return live.load_player(spec, player_args)


class Server:
    """One Session; `handle(message)` is the Session's reply with `view` and `net` added. Never raises."""

    def __init__(self, player: Any, seed: int = 0, log_dir: str | None = None) -> None:
        self.proxy = None if isinstance(player, str) else Proxy(player)
        self.session = ServeSession(player=self.proxy or player, seed=seed, log_dir=log_dir)
        self.session.capture_builtin = self.proxy is None
        self.failures = 0
        self.last_failure: str | None = None

    def _failed(self, what: str, error: BaseException) -> None:
        self.failures += 1
        self.last_failure = f"{what}: {type(error).__name__}: {error}"
        _say(f"{self.last_failure} (reply sent without it)")

    def _last(self) -> Any:
        return getattr(self.proxy.player, "last", None) if self.proxy is not None else None

    def handle(self, message: Any) -> dict | None:
        try:
            before = self._last()
            self.session.captured = None
            if self.proxy is not None:
                self.proxy.battle = None
        except Exception as error:
            self._failed("before the request", error)
            before = None
        reply = self.session.handle(message)  # CG's Session never raises
        try:
            self._add(reply, before)
        except Exception as error:
            self._failed("adding view/net", error)
        return reply

    def _add(self, reply: Any, before: Any) -> None:
        """`view` and `net` into the reply's notes, each only when it was made without an error."""
        if not isinstance(reply, dict) or "error" in reply:
            return
        if "status" in reply and isinstance(reply["status"], dict):
            reply["status"]["serve"] = {"failures": self.failures, "last": self.last_failure}
            return
        if not ("order" in reply or "picks" in reply):
            return
        battle = self.proxy.battle if self.proxy is not None else self.session.captured
        if battle is None:
            return  # answered without asking: no Battle of a decision
        added = {}
        for key, make in (("view", lambda: view(battle)), ("net", lambda: self._net(reply, before, battle))):
            try:
                value = make()
                json.dumps(value, allow_nan=False)
            except Exception as error:
                self._failed(key, error)
                continue
            if value is not None:
                added[key] = value
        if added:
            notes = reply.get("notes")
            if not isinstance(notes, dict):
                notes = reply["notes"] = {}
            notes.update(added)

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


def reply_line(server: Server, line: str) -> str:
    """The output line for one input line, whatever happens: the reply, else the reply without our additions,
    else an error reply."""
    try:
        reply = server.handle(line) if line.strip() else {"error": "empty line"}
    except Exception as error:  # Server.handle guards itself; this is the last line of defence
        server._failed("handle", error)
        return json.dumps({"error": f"pcsa.cg.serve: {type(error).__name__}: {error}"})
    try:
        return json.dumps(reply, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        server._failed("writing the reply", error)
    notes = reply.get("notes") if isinstance(reply, dict) else None
    if isinstance(notes, dict):
        notes.pop("view", None)
        notes.pop("net", None)
    try:
        return json.dumps(reply, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        return json.dumps({"error": f"pcsa.cg.serve: reply not JSON: {error}"})


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m pcsa.cg.serve", description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--player", default="tower", help="a built-in player's name, or module:Class (or module:factory)")
    ap.add_argument("--player-args", default=None, help="JSON object of keyword arguments for module:Class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", default=None)
    return ap


def start(a: argparse.Namespace) -> Server | None:
    """The Server, or None after a stderr line saying why it cannot start."""
    if live is None:
        _say(f"{CG_HINT} ({CG_MISSING})")
        return None
    try:
        player = make_player(a.player, json.loads(a.player_args) if a.player_args else None)
        return Server(player, seed=a.seed, log_dir=a.log_dir)
    except ModuleNotFoundError as error:
        _say(TORCH_HINT if error.name == "torch" else f"{error}")
    except Exception as error:
        _say(f"{error}")
    return None


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    server = start(a)
    if server is None:
        return 2
    _say(f"player {server.session.player_name}, seed {a.seed}, log {a.log_dir}")
    stdin = open(sys.stdin.fileno(), encoding="utf-8", errors="replace", newline=None, closefd=False)  # noqa: SIM115
    stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n", closefd=False)  # noqa: SIM115
    with server.session:
        for line in stdin:
            stdout.write(reply_line(server, line.rstrip("\r\n")) + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
