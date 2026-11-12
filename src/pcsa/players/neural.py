"""The neural player: a poke-singles-former checkpoint from the research runs (pcsa.rl_training), with a fallback.

Settings in config/players/neural.yaml (NeuralParams). The network sees the Battle the loop gives it (in the
simulator the arena's p1 view, on the phone the Battle rebuilt from the battle log) and decides team preview
and every move or switch. The fallback kind (disciplined by default) decides instead, with the reason in the
notes, whenever the network cannot: no checkpoint, an unreadable one or one of another network (found once, at
build), no Battle (a screen-level decision), an exception, or no answer within the decision budget
(min(max_decision_s, the deadline's remaining time - reserve_s)).

The network's inputs hold no guessed opposing build. The guard rules do use one: this player keeps the build
catalog (seen builds first) and an opponent tracker for them alone, loaded on the first decision with rules on.
A battle resumed in a new process gets its memory back through `restore`.

Guard rules (pcsa.players.neural_rules, the `rules` list) stop plainly wasted actions (a heal at full HP, Stealth
Rock already up, a status move on a statused foe, a lock into moves that do nothing, a missed sure priority KO): the
network's best other action is played and the notes say so (`override`, `override_why`, `network`).

Feature mode (`features`): the observation columns the network reads. A checkpoint carries the mode it was trained
with (its run config's `features`; none = "base", every network before the belief tracker), and by default
(`features: checkpoint`) that is the mode used, so an old network is never fed belief columns. An explicit mode
is refused at build when the checkpoint is "base" and the mode is not; between belief modes it is allowed (logged).
In the belief modes the player keeps one belief tracker per battle (pcsa.belief); `belief_crit_lines` says whether
the protocol prints critical hits (false on the phone: a hit may then have been one), and empty items / abilities
of seen builds count as unknown. See src/pcsa/belief/README.md.

torch is imported only when a checkpoint is loaded; it comes with the `train` extra (`uv sync --extra train`).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
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
FEATURES = ("checkpoint", "base", "belief", "belief+mega")  # checkpoint: the mode the network was trained with
TORCH_HINT = "the neural player needs torch: run `uv sync --extra train` (or `uv sync --inexact --extra train`)"


@dataclass
class NeuralParams:
    checkpoint: str = "assets/models/neural.pt"  # absolute, or relative to $PCSA_NETWORKS, then to the repo
    device: str = "cpu"
    sampling: str = "greedy"  # greedy: the most likely action; sample: training's capped (0.99) distribution
    temperature: float = 1.0  # of `sample`
    opponent_temperature: float = 1.0  # of their predicted actions where they weight the matrix; 1 = as trained
    opponent_top: int = 0  # weigh our actions against only their N likeliest actions; 0 = all, as trained
    preview: str = "critic"  # critic: training's softmax(value / 0.1) top-p 0.9 nucleus; greedy: the best value
    fallback: str = "disciplined"  # the player kind that decides when the network cannot
    max_decision_s: float = 5.0  # longest a decision (or team preview) may take before the fallback decides
    reserve_s: float = 0.5  # deadline time kept for the fallback
    threads: int = 1  # torch CPU threads
    seed: int | None = None  # preview and `sample` draws; None: fresh each run
    rules: list[str] = field(default_factory=list)  # guard rules (pcsa.players.neural_rules); [] = the network alone
    features: str = "checkpoint"  # the observation's feature mode; checkpoint: the one the network was trained with
    belief_crit_lines: bool = False  # belief modes: the protocol shows critical hits (the simulator does; live: no)

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
        neural_rules.check(self.rules, where)
        if self.sampling not in SAMPLING:
            raise ConfigError(f"{where}: sampling must be one of {SAMPLING}, not {self.sampling!r}")
        if self.preview not in PREVIEW:
            raise ConfigError(f"{where}: preview must be one of {PREVIEW}, not {self.preview!r}")
        if self.features not in FEATURES:
            raise ConfigError(f"{where}: features must be one of {FEATURES}, not {self.features!r}")
        if not isinstance(self.opponent_top, int) or isinstance(self.opponent_top, bool) or self.opponent_top < 0:
            raise ConfigError(f"{where}: opponent_top must be a whole number, 0 (all) or more")
        if self.fallback == "neural":
            raise ConfigError(f"{where}: the fallback cannot be the neural player itself")
        if self.temperature <= 0 or self.max_decision_s <= 0 or self.threads < 1 or self.opponent_temperature <= 0:
            raise ConfigError(
                f"{where}: temperature, opponent_temperature, max_decision_s and threads must be positive"
            )


def resolve_checkpoint(raw: str, environ: Mapping[str, str] | None = None) -> Path:
    """An absolute path as is; a relative one under $PCSA_NETWORKS when it exists there, else under the repo."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    folder = (os.environ if environ is None else environ).get(NETWORKS_ENV)
    if folder and (Path(folder).expanduser() / path).is_file():
        return Path(folder).expanduser() / path
    return paths.repo_root() / path


def feature_mode(setting: str, trained: str, checkpoint: str = "the checkpoint") -> str:
    """The feature mode to play with: the checkpoint's (`trained`) unless `setting` names another. ConfigError when
    that would feed belief columns to a network trained without them."""
    if setting in ("checkpoint", trained):
        return trained
    if trained == "base":
        raise ConfigError(
            f"features: {setting} asked for, but {checkpoint} was trained with feature mode base: an old network "
            "cannot read belief columns (set features: checkpoint, or point policy.neural_checkpoint at a belief one)"
        )
    log.warning("neural_features_override", trained=trained, used=setting, checkpoint=checkpoint)
    return setting


def load_policy(params: NeuralParams, path: Path) -> tuple[Any, dict]:
    """(FormerPlayer, facts about the checkpoint); ConfigError without torch or on a feature mode the network cannot
    read, ValueError on a bad checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise ConfigError(TORCH_HINT) from error
    from pcsa.rl_training.former import Net
    from pcsa.rl_training.former_player import FormerPlayer, checkpoint_config, checkpoint_features, load_checkpoint

    torch.set_num_threads(params.threads)
    state = load_checkpoint(path)
    cfg = checkpoint_config(state)
    trained = checkpoint_features(state)
    mode = feature_mode(params.features, trained, path.name)
    # live: no critical-hit lines (a hit may have been one); a seen build's empty item or ability is unknown
    belief_options = {"crit_reported": bool(params.belief_crit_lines), "empty_unknown": True}
    temperature = params.temperature if params.sampling == "sample" else 0.0
    policy = FormerPlayer(
        Net(cfg),
        cfg,
        device=params.device,
        temperature=temperature,
        rng=params.seed,
        tag=path.stem,
        greedy_preview=params.preview == "greedy",
        opponent_temperature=params.opponent_temperature,
        opponent_top=params.opponent_top,
        features=mode,
        belief_options=belief_options,
    )
    policy.load_state_dict(state["net"])
    facts = {"update": state.get("update"), "run": (state.get("config") or {}).get("name")}
    return policy, {**facts, "d": cfg.d, "layers": cfg.layers, "columns": cfg.top_k + 5, "memory": cfg.memory}


def load_threat() -> tuple[Any, Any]:
    """(build catalog with the builds seen on the live ladder first, opponent tracker), for the guard rules."""
    from pcsa.battle.builds import Catalog
    from pcsa.players.opponent import Tracker

    catalog = Catalog.load(seen=True)
    return catalog, Tracker(catalog)


class NeuralPlayer:
    name = "neural"

    def __init__(
        self,
        params: NeuralParams,
        fallback: Player,
        checkpoint: Path,
        *,
        loader: Callable[[NeuralParams, Path], tuple[Any, dict]] = load_policy,
        threat: Callable[[], tuple[Any, Any]] = load_threat,
    ) -> None:
        self.p, self.fallback, self.checkpoint = params, fallback, checkpoint
        self._load_threat, self._threat = threat, None  # (catalog, tracker) of the guard rules, loaded on first use
        neural_rules.check(params.rules, "neural player")  # a bad rule name fails at build, not mid-battle
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
        }
        if self.policy is not None and hasattr(self.policy, "features"):
            out["features_used"] = self.policy.features  # the feature mode played (the checkpoint's, by default)
        return out | ({"error": self.error} if self.error else {})

    def notes(self) -> dict:
        notes, self._notes = self._notes, {}
        return notes

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        backup = self.fallback.choose_team(view, deadline)  # also resets the fallback's per-game state
        backup_notes = self.fallback.notes()
        if self._threat is not None:
            self._threat[1].reset()
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()  # a new battle, also when the fallback picks the team
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

    def _override(self, view: DecisionView, proposal: Any) -> tuple[Any, dict]:
        """The proposal, or a copy with a rule's action, and the notes saying so. A rule that errors is skipped
        (noted): the network's choice stands."""
        if not self.p.rules:
            return proposal, {}
        try:
            if self._threat is None:
                self._threat = self._load_threat()
            catalog, tracker = self._threat
            tracker.observe(view.battle)
            ranked = self._ranked(view, proposal)
            spot = neural_rules.Spot(view.battle, view.legal, proposal.chosen, ranked, catalog, tracker)
            out = neural_rules.apply(self.p.rules, spot)
        except Exception as error:
            return proposal, {"override_error": f"{type(error).__name__}: {error}"}
        if out is None:
            return proposal, {}
        action, rule, reason = out
        notes = {"override": rule, "override_why": reason, "network": proposal.chosen.message}
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

    def restore(self, views: Iterable[DecisionView]) -> int:
        """Rebuild the network's memory from the views of this battle's earlier decisions, oldest first (a battle
        resumed in a new process); returns how many decisions are in the memory (0 when the network is
        unavailable). Never raises."""
        if self.error or self.policy is None:
            return 0
        with self._lock:
            try:
                return int(self.policy.restore(views))
            except Exception as error:
                log.warning("neural_restore_failed", reason=f"{type(error).__name__}: {error}")
                return 0

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
    if policy.neural_rules is not None:  # "none", or rule names separated by commas
        names = [] if policy.neural_rules.strip() == "none" else policy.neural_rules.split(",")
        params.rules = [name.strip() for name in names if name.strip()]
        params.check(f"policy.neural_rules={policy.neural_rules}")
    return NeuralPlayer(params, build_member(params.fallback), resolve_checkpoint(params.checkpoint, environ))
