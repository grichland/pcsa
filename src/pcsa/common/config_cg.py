"""The neural player's settings in the cg section: its decision-time search and its pick from the matrix."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CGSearch(_Strict):
    """Decision-time search of the neural player (pcsa.network.search; Champions Gym 0.5.0). `enabled` puts these
    into the player's policy_args: `ms` per turn request, `worlds` guesses of their team, `depth` turns, `outcomes`
    per action pair, `c` and `n0` weigh the network's priors and matrix, `min_sims` is the least that counts."""

    enabled: bool = False
    ms: float = Field(default=1500.0, gt=0)
    worlds: int = Field(default=6, gt=0)
    depth: int = Field(default=3, gt=0)
    outcomes: int = Field(default=3, gt=0)
    c: float = Field(default=1.5, ge=0)
    n0: float = Field(default=1.0, ge=0)
    batch: int = Field(default=16, gt=0)
    min_sims: int = Field(default=64, ge=0)
    top_ours: int = Field(default=0, ge=0)  # only our likeliest N actions at each node (0: all)
    top_theirs: int = Field(default=0, ge=0)  # only their likeliest N actions at each node (0: all)
    settle_sims: int = Field(default=0, ge=0)  # stop early once the pick has held this long (0: never)
    settle_share: float = Field(default=0.9, gt=0, le=1)
    settle_gap: float = Field(default=0.01, ge=0)
    from_turn: int = Field(default=1, ge=1)  # battle turns before this one: the model's answer is played


class CGChoice(_Strict):
    """How the neural player makes its pick from the matrix at turn requests (pcsa.network.choosing): `rule` mean
    (the network's own), min, mix or regret over their `top_theirs` likeliest replies (0: all), among our
    `top_ours` likeliest actions (0: all legal); `blend` is mix's weight of the mean."""

    rule: Literal["mean", "min", "mix", "regret"] = "mean"
    top_ours: int = Field(default=0, ge=0)
    top_theirs: int = Field(default=0, ge=0)
    blend: float = Field(default=0.5, ge=0, le=1)

    def args(self) -> dict:
        """The policy's `choice` argument; empty when it changes nothing."""
        same = self.rule == "mean" and not self.top_ours and not self.top_theirs
        return {} if same else self.model_dump()
