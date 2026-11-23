"""Settings of the neural player's policy: rule masks, decision-time search, the choice rule and the value
calibration.

The values live in config/agent.yaml (section `cg`); these classes only type and check them. pcsa.network uses
the same classes, so the live player and the tools read one definition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PREVIEW_SLACK_S = 6.0  # over preview_search.budget_s: guesses, environment, reply


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuleSettings(_Strict):
    """Training's rule masks on the network's policy (pcsa.network.rules); they must be the ones the checkpoint was
    trained with."""

    enabled: bool
    max_same_move: int = Field(ge=0)  # the same move by the same Pokémon in a row (0: no such rule)
    max_switches: int = Field(ge=0)  # switches by choice in a row
    max_boosts: int = Field(ge=0)  # setup moves by the same Pokémon in a row
    max_heals: int = Field(ge=0)  # pure recovery by the same Pokémon in a row
    no_null_moves: bool  # damaging moves that do nothing to their active
    no_boost_when_ko: bool  # setup (other than defensive) while their best hit KOs us
    no_failing_status: bool  # status moves that cannot land
    no_redundant_field: bool  # hazards at their maximum, weather or terrain already up
    no_double_protect: bool  # the Protect family by the same Pokémon two turns in a row
    no_wasted_self: bool  # recovery at full HP, setup at +6, Substitute when it would fail
    status_free_mirrors: list[str]  # pcsa's own: no status moves when both actives are one of these species

    def args(self) -> dict | None:
        """The policy's `rules` argument; None when the masks are off."""
        return self.model_dump(exclude={"enabled"}) if self.enabled else None


class SearchSettings(_Strict):
    """Decision-time search at turn requests (pcsa.network.search)."""

    ms: float = Field(gt=0)  # time per turn request
    worlds: int = Field(gt=0)  # sampled completions of the opponent's hidden team, one tree each
    depth: int = Field(gt=0)  # turns below the root
    outcomes: int = Field(gt=0)  # random outcomes per action pair
    c: float = Field(ge=0)  # weight of the network's priors in selection
    n0: float = Field(ge=0)  # weight of the network's matrix cell in each pair's value
    batch: int = Field(gt=0)  # positions per network call
    min_sims: int = Field(ge=0)  # fewer simulations in total: the network's own answer is played
    sims: int = Field(ge=0)  # stop after this many simulations (0: by time only)
    seed: int
    top_ours: int = Field(ge=0)  # our N likeliest actions at each node (0: all)
    top_theirs: int = Field(ge=0)  # their N likeliest actions at each node (0: all)
    from_turn: int = Field(ge=1)  # earlier turns are not searched
    settle_sims: int = Field(ge=0)  # stop early once the pick has led this long (0: never)
    settle_share: float = Field(ge=0, le=1)  # ... with at least this share of the strategy
    settle_gap: float = Field(ge=0)  # or once the best two values are this close, after twice settle_sims


class CGSearch(SearchSettings):
    enabled: bool

    def args(self) -> dict | None:
        """The policy's `search` argument; None when search is off."""
        return self.model_dump(exclude={"enabled"}) if self.enabled else None


class ValueCalibration(_Strict):
    """The value reported as 2 * sigmoid(a * v + b) - 1, so (v + 1) / 2 reads as a win chance; fitted per checkpoint
    on recorded battles (tools/value_calibration.py). The raw value is kept as `value_raw`."""

    enabled: bool
    a: float
    b: float

    def args(self) -> dict | None:
        """The policy's `value_calibration` argument; None when off."""
        return self.model_dump(exclude={"enabled"}) if self.enabled else None


class PreviewSearch(_Strict):
    """Team-preview search (pcsa.network.preview_search): the net plays out every (our pick, their pick) cell
    against guesses of their team and picks our best row; it replaces the preview head's pick."""

    enabled: bool
    budget_s: float = Field(gt=0)  # search time; never past the request's deadline
    guesses: int = Field(ge=1)  # versions of their team, sets drawn from the belief
    race: list[int]  # rows kept at the cuts (1/4, 1/2, 3/4 of the budget for three); the head's pick always kept
    prior_games: float = Field(ge=0)  # stage A counts as this many games in each cell
    choose: Literal["mean", "maximin"]
    keep_head_margin: float = Field(ge=0)  # leave the head's pick only when the best row is this much higher
    device: str  # the playouts' torch device
    threads: int = Field(gt=0)  # Champions Gym threads
    seed: int

    def args(self) -> dict | None:
        """The policy's `preview_search` argument; None when off."""
        return self.model_dump(exclude={"enabled"}) if self.enabled else None

    def wait_s(self) -> float:
        """How long the controller waits for a preview answer while the search is on (0: off)."""
        return self.budget_s + PREVIEW_SLACK_S if self.enabled else 0.0


class ChoiceSettings(_Strict):
    """How the pick is made from the matrix at turn requests (pcsa.network.choosing): `rule` over the opponent's
    `top_theirs` likeliest replies (0: all), among our `top_ours` likeliest actions (0: all legal); `blend` is the
    weight of the mean in `mix`."""

    rule: Literal["mean", "min", "mix", "regret"]
    top_ours: int = Field(ge=0)
    top_theirs: int = Field(ge=0)
    blend: float = Field(ge=0, le=1)

    @property
    def active(self) -> bool:
        """False when the rule reproduces the network's own pick (mean over every reply and action)."""
        return not (self.rule == "mean" and not self.top_ours and not self.top_theirs)

    def args(self) -> dict | None:
        """The policy's `choice` argument; None when it changes nothing."""
        return self.model_dump() if self.active else None
