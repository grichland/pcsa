# Neural player checkpoints

Put network checkpoints for the `neural` player here (copies of research runs' `ckpt/*.pt`, e.g.
`research/pcsa-rl/runs/s1/ckpt/latest.pt`). `config/players/neural.yaml` names the one to use with a path relative
to the repo, e.g. `checkpoint: assets/models/team2-fine-tuned.pt`; `--set policy.neural_checkpoint=assets/models/<file>.pt`
picks another for one run. The `.pt` files are not committed (`.gitignore`).

| file | network | from |
|---|---|---|
| team2-fine-tuned.pt | S fine-tuned for 50 PPO updates on the phone team of run 20261108-204434 (Garchomp Scarf, Golisopod, Mimikyu, Lucario, Kingambit, Primarina) | research/pcsa-rl/runs/s1-phone/ckpt/latest.pt (update 50) |
