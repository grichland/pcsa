"""The neural player: a PPO / imitation checkpoint from the research runs (pcsa.rl_training), with a fallback.

Settings in config/players/neural.yaml (NeuralParams). The network sees the Battle the loop gives it (in the
simulator the arena's p1 view, on the phone the Battle rebuilt from the battle log) and decides team preview
and every move or switch. The fallback kind (disciplined by default) decides instead, with the reason in the
notes, whenever the network cannot: no checkpoint, an unreadable one or a size mismatch (found once, at build),
no Battle (a screen-level decision), an exception, or no answer within the decision budget
(min(max_decision_s, the deadline's remaining time - reserve_s)).

Override rules (pcsa.players.neural_rules, config/players/neural_rules.yaml) may replace the network's choice in
clear-cut spots: `rules: auto` turns on the general rules and, when our six match a team profile's roster, that
team's own; `general`, a team name, or `none` (the network alone). An override is in the notes (`override`,
`override_why`, `network`: the network's own choice).

torch is imported only when a checkpoint is loaded; it comes with the `train` extra (`uv sync --extra train`).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

from pcsa.common import log, paths
from pcsa.common.clock import Deadline, precise_now
from pcsa.common.config import Config
from pcsa.common.errors import ConfigError
from pcsa.players import neural_rules
from pcsa.players.base import DecisionView, LegalAction, Player, TeamPreviewView

NETWORKS_ENV = "PCSA_NETWORKS"  # a folder relative checkpoint paths are looked up in first
SAMPLING = ("greedy", "sample")
PREVIEW = ("critic", "greedy")
SIZES = ("S", "M", "L")
TORCH_HINT = "the neural player needs torch: run `uv sync --extra train` (or `uv sync --inexact --extra train`)"


@dataclass
class NeuralParams:
    checkpoint: str = "assets/models/neural.pt"  # absolute, or relative to $PCSA_NETWORKS, then to the repo
    size: str | None = None  # S | M | L; None: the checkpoint's own (a different one is a size mismatch)
    device: str = "cpu"
    sampling: str = "greedy"  # greedy: the most likely action; sample: training's capped (0.99) distribution
    temperature: float = 1.0  # of `sample`
    preview: str = "critic"  # critic: training's softmax(value / 0.1) top-p 0.9 nucleus; greedy: the best value
    fallback: str = "disciplined"  # the player kind that decides when the network cannot
    max_decision_s: float = 5.0  # longest a decision (or team preview) may take before the fallback decides
    reserve_s: float = 0.5  # deadline time kept for the fallback
    threads: int = 1  # torch CPU threads
    seed: int | None = None  # preview and `sample` draws; None: fresh each run
    rules: str = "none"  # override rules: auto | general | none | a team name (config/players/neural_rules.yaml)
    rules_file: str = "players/neural_rules.yaml"  # relative to the config folder

    @classmethod
    def load(cls, path: Path) -> NeuralParams:
        data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown neural player settings in {path}: {sorted(unknown)}")
        params = cls(**data)
        params.check(str(path))
        return params

    def check(self, where: str) -> None:
        if self.sampling not in SAMPLING:
            raise ConfigError(f"{where}: sampling must be one of {SAMPLING}, not {self.sampling!r}")
        if self.preview not in PREVIEW:
            raise ConfigError(f"{where}: preview must be one of {PREVIEW}, not {self.preview!r}")
        if self.size is not None and self.size not in SIZES:
            raise ConfigError(f"{where}: size must be one of {SIZES} or null, not {self.size!r}")
        if self.fallback == "neural":
            raise ConfigError(f"{where}: the fallback cannot be the neural player itself")
        if self.temperature <= 0 or self.max_decision_s <= 0 or self.threads < 1:
            raise ConfigError(f"{where}: temperature, max_decision_s and threads must be positive")


def resolve_checkpoint(raw: str, environ: Mapping[str, str] | None = None) -> Path:
    """An absolute path as is; a relative one under $PCSA_NETWORKS when it exists there, else under the repo."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    folder = (os.environ if environ is None else environ).get(NETWORKS_ENV)
    if folder and (Path(folder).expanduser() / path).is_file():
        return Path(folder).expanduser() / path
    return paths.repo_root() / path


def load_policy(params: NeuralParams, path: Path) -> tuple[Any, dict]:
    """(PolicyPlayer, facts about the checkpoint); ConfigError without torch, ValueError on a bad checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise ConfigError(TORCH_HINT) from error
    from pcsa.battle.builds import Catalog
    from pcsa.rl_training.model import Net
    from pcsa.rl_training.player import PolicyPlayer, checkpoint_config, load_checkpoint

    torch.set_num_threads(params.threads)
    state = load_checkpoint(path)
    cfg, size = checkpoint_config(state, params.size)
    temperature = params.temperature if params.sampling == "sample" else 0.0
    policy = PolicyPlayer(
        Net(cfg),
        cfg,
        catalog=Catalog.load(),
        device=params.device,
        temperature=temperature,
        rng=params.seed,
        tag=path.stem,
        greedy_preview=params.preview == "greedy",
    )
    policy.load_state_dict(state["net"])
    return policy, {"size": size, "update": state.get("update"), "run": (state.get("config") or {}).get("name")}


class NeuralPlayer:
    name = "neural"

    def __init__(
        self,
        params: NeuralParams,
        fallback: Player,
        checkpoint: Path,
        *,
        loader: Callable[[NeuralParams, Path], tuple[Any, dict]] = load_policy,
        rules: neural_rules.RulesConfig | None = None,
    ) -> None:
        self.p, self.fallback, self.checkpoint = params, fallback, checkpoint
        self.rules_config = rules or neural_rules.RulesConfig()
        self.rules_config.select(params.rules, ())  # a bad rules name fails at build, not mid-battle
        self._rules_for: tuple[str | None, tuple[str, ...], neural_rules.Profile | None] = (None, (), None)
        self.policy: Any = None
        self.facts: dict = {}
        self.error: str | None = None
        self._lock = threading.Lock()
        self._notes: dict = {}
        if not checkpoint.is_file():
            self.error = f"no checkpoint at {checkpoint}"
        else:
            try:
                self.policy, self.facts = loader(params, checkpoint)
            except ConfigError:
                raise
            except Exception as error:
                self.error = f"checkpoint {checkpoint.name} not loaded: {type(error).__name__}: {error}"
        if self.error:
            log.warning("neural_unavailable", reason=self.error, fallback=fallback.name)

    def params(self) -> dict:
        out = {
            **asdict(self.p),
            "checkpoint": str(self.checkpoint),
            **self.facts,
            "fallback_kind": self.fallback.name,
            "rules_general": list(self.rules_config.general),
            "rules_teams": {p.name: list(p.rules) for p in self.rules_config.profiles},
        }
        return out | ({"error": self.error} if self.error else {})

    def notes(self) -> dict:
        notes, self._notes = self._notes, {}
        return notes

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        backup = self.fallback.choose_team(view, deadline)  # also resets the fallback's per-game state
        backup_notes = self.fallback.notes()
        why = self.error or ("no battle at preview" if view.battle is None else None)
        if why is None:
            answer, why, ms = self._timed(lambda: self._preview(view, deadline), deadline)
            if why is None:
                team, notes = answer
                self._notes = {**notes, "rule": "neural", "ms": ms}
                return team
        self._notes = self._fallback_notes(why, backup_notes)
        return backup

    def _preview(self, view: TeamPreviewView, deadline: Deadline) -> tuple[list[int], dict]:
        team = self.policy.choose_team(view, deadline)
        notes = self.policy.notes()
        if notes.get("rule") != "rl":
            raise RuntimeError(notes.get("why", "preview failed"))
        return team, notes

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        if not view.legal:
            raise ValueError("no legal action")
        why = self.error or ("no battle (screen-level decision)" if view.battle is None else None)
        if why is None:
            proposal, why, ms = self._timed(lambda: self.policy.propose(view), deadline)
            if why is None:
                proposal, override = self._override(view, proposal)
                chosen = self.policy.commit(proposal)
                self._notes = {**self.policy.notes(), "rule": "neural", "ms": ms, **override}
                return chosen
        chosen = self.fallback.choose_action(view, deadline)
        self._notes = self._fallback_notes(why, self.fallback.notes())
        return chosen

    def _rules(self, battle) -> tuple[tuple[str, ...], neural_rules.Profile | None]:
        """The rules enabled for this battle, chosen once per battle from our six."""
        tag = getattr(battle, "battle_tag", None)
        if self._rules_for[0] != tag or tag is None:
            species = [m.species for m in battle.team.values()]
            self._rules_for = (tag, *self.rules_config.select(self.p.rules, species))
        return self._rules_for[1], self._rules_for[2]

    def _override(self, view: DecisionView, proposal: Any) -> tuple[Any, dict]:
        """The proposal, or a copy with a rule's action, and the notes saying so. A rule that errors is skipped
        (noted): the network's choice stands."""
        if self.p.rules == "none":
            return proposal, {}
        try:
            rules, profile = self._rules(view.battle)
            if not rules:
                return proposal, {}
            spot = neural_rules.Spot(
                view.battle,
                view.legal,
                proposal.chosen,
                self._ranked(view, proposal),
                self.policy.catalog,
                self.policy.tracker,
                profile.params if profile else self.rules_config.params,
            )
            out = neural_rules.apply(rules, spot)
        except Exception as error:
            return proposal, {"override_error": f"{type(error).__name__}: {error}"}
        team = {"rules_team": profile.name} if profile else {}
        if out is None:
            return proposal, team
        action, rule, reason = out
        notes = {**team, "override": rule, "override_why": reason, "network": proposal.chosen.message}
        return replace(proposal, chosen=action), notes

    @staticmethod
    def _ranked(view: DecisionView, proposal: Any) -> list[LegalAction]:
        """The legal actions, the network's most likely first."""
        from pcsa.rl_training import features

        probs = proposal.probs[0].tolist()
        out: list[LegalAction] = []
        for index in sorted(range(len(probs)), key=lambda i: -probs[i]):
            try:
                action = features.legal_action(index, view.legal, view.battle)
            except ValueError:
                continue
            if action not in out:
                out.append(action)
        return out

    def _fallback_notes(self, why: str | None, notes: dict) -> dict:
        """The fallback's own notes, its rule kept as fallback_rule, under rule neural:fallback with the reason."""
        out = {**notes, "rule": "neural:fallback", "why": why, "fallback": self.fallback.name}
        return out | ({"fallback_rule": notes["rule"]} if "rule" in notes else {})

    def _timed(self, work: Callable[[], Any], deadline: Deadline) -> tuple[Any, str | None, float]:
        """(result, None, ms), or (None, why, ms) on an error or no answer within the budget. The work runs in a
        thread so a late answer can be dropped; it is never applied."""
        budget = min(self.p.max_decision_s, deadline.remaining() - self.p.reserve_s)
        if budget <= 0:
            return None, f"deadline: {deadline.remaining():.2f}s left", 0.0
        if not self._lock.acquire(blocking=False):
            return None, "the previous network call is still running", 0.0
        out: dict[str, Any] = {}
        started = precise_now()

        def run() -> None:
            try:
                out["value"] = work()
            except Exception as error:
                out["error"] = f"{type(error).__name__}: {error}"
            finally:
                self._lock.release()

        thread = threading.Thread(target=run, name="neural-decision", daemon=True)
        thread.start()
        thread.join(budget)
        ms = round((precise_now() - started) * 1000, 1)
        if thread.is_alive():
            return None, f"no answer within {budget:.2f}s", ms
        if "error" in out:
            return None, out["error"], ms
        return out["value"], None, ms


def build(config: Config, build_member: Callable[[str], Player], environ: Mapping[str, str] | None = None) -> Player:
    """The neural player from agent.yaml's policy.neural file; policy.neural_checkpoint, when set, replaces the
    file's checkpoint (one `--set policy.neural_checkpoint=<path>` points a run at a network)."""
    policy = config.agent.policy
    params = NeuralParams.load(config.directory / policy.neural)
    if policy.neural_checkpoint:
        params.checkpoint = policy.neural_checkpoint
    if policy.neural_rules:
        params.rules = policy.neural_rules
    rules = neural_rules.RulesConfig.load(config.directory / params.rules_file)
    return NeuralPlayer(
        params, build_member(params.fallback), resolve_checkpoint(params.checkpoint, environ), rules=rules
    )
