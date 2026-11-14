# Live fine-tune report (2026-11-14 15:21:37)

Input `C:\Users\geral\Code\pcsa\assets\models\poke-transformer-hard50-20261113-1215.pt` (update 50); output `C:\Users\geral\Code\research\pcsa-rl\runs\live-finetune\opp-20261114-1509\model-e7.pt`. Epochs 7, lr 0.001, minibatch ~256 decisions, seed 0, hold-out 0.0. Runtime 29.7 s (CPU, 2 threads).

## Data

Folders: `C:\Users\geral\Code\research\pcsa-rl\runs\live-reconciled\5672e88\pcsa`. Battle folders 343, recorded decisions 3891, rebuilt 3887. Skipped battles: {}. Outcomes: {"loss": 166, "opponent_forfeit": 46, "no battle.json": 7, "unknown": 20, "win": 104}. Battles whose final faint count was completed (the phone does not record the last turn): 267.

| part | battles | decisions | battles with value target | decisions with value target | battles without | decisions without | decisions with opponent label | won | lost | from | to |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 343 | 3887 | 336 | 3809 | 7 | 78 | 2700 | 170 | 166 | 20261108T170448 | 20261114T145624 |
| held out | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |

Held-out battles by folder {}, by live player {}; train by folder {"pcsa": 343}, by live player {"best_matchup_max_damage": 1, "disciplined": 4, "neural": 338}.

Decision failures (folder: stage: exception):

| failure | n |
|---|---|
| pcsa: legal_actions: none | 4 |

Simulator reference: 300 games, 3606 decisions from `u00800.npz`.

## Before / after

| metric | live held-out before | live held-out after | simulator before | simulator after | live train before | live train after |
|---|---|---|---|---|---|---|
| battles | 0 | 0 | 300 | 300 | 343 | 343 |
| decisions | 0 | 0 | 3606 | 3606 | 3887 | 3887 |
| opp all scored: n | 0 | 0 | 2628 | 2628 | 2697 | 2697 |
| opp all scored: top-1 | - | - | 0.360 | 0.360 | 0.542 | 0.603 |
| opp all scored: top-3 | - | - | 0.672 | 0.674 | 0.804 | 0.863 |
| opp all scored: in kept 9 | - | - | 0.861 | 0.860 | 0.916 | 0.936 |
| opp all scored: mean p(real) | - | - | 0.282 | 0.280 | 0.385 | 0.421 |
| opp move already revealed: n | 0 | 0 | 1030 | 1030 | 756 | 756 |
| opp move already revealed: top-1 | - | - | 0.442 | 0.463 | 0.655 | 0.717 |
| opp move already revealed: top-3 | - | - | 0.781 | 0.775 | 0.903 | 0.938 |
| opp move already revealed: in kept 9 | - | - | 1.000 | 1.000 | 1.000 | 1.000 |
| opp move already revealed: mean p(real) | - | - | 0.346 | 0.349 | 0.464 | 0.497 |
| opp move not yet revealed: n | 0 | 0 | 1399 | 1399 | 1521 | 1521 |
| opp move not yet revealed: top-1 | - | - | 0.327 | 0.309 | 0.464 | 0.517 |
| opp move not yet revealed: top-3 | - | - | 0.634 | 0.639 | 0.772 | 0.827 |
| opp move not yet revealed: in kept 9 | - | - | 0.738 | 0.738 | 0.851 | 0.886 |
| opp move not yet revealed: mean p(real) | - | - | 0.255 | 0.246 | 0.328 | 0.360 |
| opp switch: n | 0 | 0 | 199 | 199 | 420 | 420 |
| opp switch: top-1 | - | - | 0.166 | 0.181 | 0.621 | 0.710 |
| opp switch: top-3 | - | - | 0.377 | 0.397 | 0.738 | 0.860 |
| opp switch: in kept 9 | - | - | 1.000 | 1.000 | 1.000 | 1.000 |
| opp switch: mean p(real) | - | - | 0.143 | 0.158 | 0.448 | 0.504 |
| opp cross-entropy (nats) | - | - | 2.238 | 2.216 | 1.619 | 1.292 |
| actual switch share | - | - | 0.076 | 0.076 | 0.156 | 0.156 |
| predicted switch mass | - | - | 0.099 | 0.105 | 0.150 | 0.149 |
| Mega: scored n | 0 | 0 | 691 | 691 | 521 | 521 |
| Mega: BCE | - | - | 0.506 | 0.527 | 0.408 | 0.376 |
| Mega: mean p when they did | - | - | 0.642 | 0.691 | 0.716 | 0.756 |
| Mega: mean p when they did not | - | - | 0.269 | 0.288 | 0.242 | 0.251 |
| value: decisions with a target | 0 | 0 | 3606 | 3606 | 3809 | 3809 |
| value: battles won / lost | 0 / 0 | 0 / 0 | 190 / 110 | 190 / 110 | 170 / 166 | 170 / 166 |
| value: Brier of (v+1)/2 | - | - | 0.171 | 0.171 | 0.262 | 0.262 |
| value: Brier of constant base rate | - | - | 0.240 | 0.240 | 0.241 | 0.241 |
| value: two-hot cross-entropy | - | - | 3.882 | 3.882 | 4.935 | 4.935 |
| value: mean |v - return| | - | - | 0.733 | 0.733 | 0.928 | 0.928 |
| value: last decision, won battles | - | - | 0.940 | 0.940 | 0.914 | 0.914 |
| value: last decision, lost battles | - | - | 0.145 | 0.145 | 0.232 | 0.232 |

Value calibration (win chance = (v+1)/2):

| win-chance bin | live held-out before: n / predicted / actual | live held-out after: n / predicted / actual | simulator before: n / predicted / actual | simulator after: n / predicted / actual | live train before: n / predicted / actual | live train after: n / predicted / actual |
|---|---|---|---|---|---|---|
| 0.0-0.2 | 0 / None / None | 0 / None / None | 658 / 0.053 / 0.164 | 658 / 0.053 / 0.164 | 435 / 0.072 / 0.06 | 435 / 0.072 / 0.06 |
| 0.2-0.4 | 0 / None / None | 0 / None / None | 511 / 0.304 / 0.401 | 511 / 0.304 / 0.401 | 395 / 0.305 / 0.109 | 395 / 0.305 / 0.109 |
| 0.4-0.6 | 0 / None / None | 0 / None / None | 604 / 0.506 / 0.558 | 604 / 0.506 / 0.558 | 568 / 0.506 / 0.266 | 568 / 0.506 / 0.266 |
| 0.6-0.8 | 0 / None / None | 0 / None / None | 632 / 0.699 / 0.701 | 632 / 0.699 / 0.701 | 885 / 0.711 / 0.393 | 885 / 0.711 / 0.393 |
| 0.8-1.0 | 0 / None / None | 0 / None / None | 1201 / 0.955 / 0.888 | 1201 / 0.955 / 0.888 | 1526 / 0.94 / 0.635 | 1526 / 0.94 / 0.635 |

## Policy change (before vs after; decisions with at least two legal actions)

| set | n | mean KL(before||after) | median KL | 99th pct KL | max KL | mean total variation | greedy changed | share |
|---|---|---|---|---|---|---|---|---|
| live held-out | 0 | - | - | - | - | - | - | - |
| simulator | 3385 | 0.014 | 0.001 | 0.252 | 2.217 | 0.034 | 136 | 0.040 |
| live train | 3556 | 0.022 | 0.001 | 0.386 | 2.517 | 0.039 | 155 | 0.044 |

## After minus before on the same scored decisions (95% interval, resampling battles)

| set | battles | top-1 | top-3 | in kept 9 | mean p(real) |
|---|---|---|---|---|---|
| simulator | 300 | +0.0000 [-0.0135, +0.0126] | +0.0019 [-0.0089, +0.0126] | -0.0004 [-0.0078, +0.0073] | -0.0026 [-0.0073, +0.0020] |
| live train | 338 | +0.0608 [+0.0438, +0.0782] | +0.0593 [+0.0450, +0.0745] | +0.0197 [+0.0126, +0.0268] | +0.0355 [+0.0291, +0.0422] |

## Training

Trained tensors (6, 33153 parameters of 1798048): `opp_move_q.weight`, `opp_move_q.bias`, `opp_switch_q.weight`, `opp_switch_q.bias`, `opp_mega.weight`, `opp_mega.bias`.

Tensors that differ from the input checkpoint: 6 (all inside the trained set); frozen tensors compared bit for bit: 141, identical: 141. Trained tensors left unchanged: [].

| epoch | steps | opp CE | Mega BCE | value CE | opp top-1 | opp top-3 | grad norm |
|---|---|---|---|---|---|---|---|
| 1 | 15 | 1.621 | 0.402 | 4.996 | 0.531 | 0.803 | 0.903 |
| 2 | 30 | 1.517 | 0.398 | 4.993 | 0.558 | 0.813 | 0.822 |
| 3 | 45 | 1.460 | 0.392 | 5.007 | 0.567 | 0.825 | 0.777 |
| 4 | 60 | 1.418 | 0.378 | 5.014 | 0.572 | 0.833 | 0.765 |
| 5 | 75 | 1.381 | 0.387 | 4.993 | 0.584 | 0.838 | 0.756 |
| 6 | 90 | 1.353 | 0.379 | 5.009 | 0.588 | 0.845 | 0.747 |
| 7 | 105 | 1.324 | 0.373 | 4.994 | 0.596 | 0.854 | 0.724 |

Per-epoch numbers are means over that epoch's minibatches while the weights move (training data).

Note: the held-out set is 0 battles / 0 labelled opponent decisions / 0 battles with an outcome: every held-out number is a small sample.

Note: `live train` is in-sample after training; it only shows how far the trained layers moved.

Note: The value target here is the plain discounted return (no bootstrap); training's PPO target is the GAE(0.9) return, which mixes in the critic's own estimates.

Note: Brier and calibration read (v+1)/2 as a win chance, as ladder_check does; v also carries the +-0.1 faint rewards, so it is not exactly a win chance.
