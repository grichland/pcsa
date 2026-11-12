"""The belief tracker (research/pcsa-belief, stages 1-2): what each opposing Pokémon could be.

A weighted set of complete builds per opposing member (listed builds, listed builds with any spread or item, and
a free spread x item x ability component), narrowed every turn by the evidence in the protocol (reveals, moves,
turn order, damage both ways), with stat intervals, speed probabilities and damage ranges read from the posterior.
Ported from research/pcsa-belief/belief (fast.py and what it needs: space, calc, evidence, hits); keep diffable.
The neural player's belief feature modes read it (pcsa.rl_training.belief_features); see README.md here.

No torch: numpy over pcsa.battle and pcsa.players.opponent.
"""
