"""Settings of the neural player's policy: rule masks, decision-time search and the choice rule.

The values live in config/agent.yaml (section `cg`); these classes only type and check them. pcsa.network uses
the same classes, so the live player and the tools read one definition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuleSettings(_Strict):
    """Rule masks applied to the network's policy: the same move at most `max_same_move` turns in a row, at most
    `max_switches` voluntary switches in a row."""

    enabled: bool
    max_same_move: int = Field(ge=0)  # 0: no such rule
    max_switches: int = Field(ge=0)

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
