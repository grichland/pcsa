"""Network inference for the neural player (pcsa.players.neural), copied from research/pcsa-rl/rl.
Only the inference path lives here: observations (features*), the network, the probability cap (entropy), team preview
(preview) and the policy as a pcsa Player. Training stays in research. Each module names its research file; keep them
diffable (same names, same layout). A leaf: only pcsa.players.neural imports it (.importlinter).
"""
