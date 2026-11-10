"""Network inference for the neural player (pcsa.players.neural), copied from research/pcsa-rl/rl.

Only the inference path lives here: observations (features*), the network (model), the probability cap
(entropy), team preview (preview) and the policy as a pcsa Player (player). Training stays in research. Each
module names the research file and version it came from; keep them diffable (same names, same layout).
A leaf: only pcsa.players.neural imports it (.importlinter), and only it needs torch (model, entropy, player).
"""
