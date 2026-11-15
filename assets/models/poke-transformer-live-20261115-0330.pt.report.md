# Live fine-tune report (2026-11-15 03:36:32)

Input `C:\Users\geral\Code\research\pcsa-rl\runs\live-finetune\job-20261115-0330\policy.pt` (update 50); output `C:\Users\geral\Code\research\pcsa-rl\runs\live-finetune\job-20261115-0330\model.pt`. Epochs 50, lr 0.001, minibatch ~256 decisions, seed 0, hold-out 0.0. Runtime 201.9 s (CPU, 2 threads).

## Data

Folders: `C:\Users\geral\Code\research\pcsa-rl\runs\live-reconciled\9c5d1a0\pcsa`. Battle folders 374, recorded decisions 4278, rebuilt 4272. Skipped battles: {}. Outcomes: {"loss": 179, "opponent_forfeit": 49, "no battle.json": 8, "unknown": 20, "win": 118}. Battles whose final faint count was completed (the phone does not record the last turn): 293.

| part | battles | decisions | battles with value target | decisions with value target | battles without | decisions without | decisions with opponent label | won | lost | from | to |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 374 | 4272 | 366 | 4176 | 8 | 96 | 2960 | 187 | 179 | 20261108T170448 | 20261114T204602 |
| held out | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |

Held-out battles by folder {}, by live player {}; train by folder {"pcsa": 374}, by live player {"best_matchup_max_damage": 1, "disciplined": 4, "neural": 369}.

Decision failures (folder: stage: exception):

| failure | n |
|---|---|
| pcsa: legal_actions: none | 6 |

Simulator reference: 300 games, 3606 decisions from `u00800.npz`.

## Before / after

| metric | live held-out before | live held-out after | simulator before | simulator after | live train before | live train after |
|---|---|---|---|---|---|---|
| battles | 0 | 0 | 300 | 300 | 374 | 374 |
| decisions | 0 | 0 | 3606 | 3606 | 4272 | 4272 |
| opp all scored: n | 0 | 0 | 2628 | 2628 | 2957 | 2957 |
| opp all scored: top-1 | - | - | 0.417 | 0.339 | 0.428 | 0.677 |
| opp all scored: top-3 | - | - | 0.735 | 0.641 | 0.742 | 0.926 |
| opp all scored: in kept 9 | - | - | 0.905 | 0.852 | 0.900 | 0.966 |
| opp all scored: mean p(real) | - | - | 0.328 | 0.260 | 0.335 | 0.461 |
| opp move already revealed: n | 0 | 0 | 1030 | 1030 | 841 | 841 |
| opp move already revealed: top-1 | - | - | 0.538 | 0.415 | 0.546 | 0.767 |
| opp move already revealed: top-3 | - | - | 0.846 | 0.733 | 0.889 | 0.986 |
| opp move already revealed: in kept 9 | - | - | 1.000 | 1.000 | 1.000 | 1.000 |
| opp move already revealed: mean p(real) | - | - | 0.414 | 0.315 | 0.432 | 0.537 |
| opp move not yet revealed: n | 0 | 0 | 1399 | 1399 | 1659 | 1659 |
| opp move not yet revealed: top-1 | - | - | 0.359 | 0.309 | 0.391 | 0.606 |
| opp move not yet revealed: top-3 | - | - | 0.688 | 0.606 | 0.696 | 0.886 |
| opp move not yet revealed: in kept 9 | - | - | 0.822 | 0.721 | 0.822 | 0.939 |
| opp move not yet revealed: mean p(real) | - | - | 0.288 | 0.237 | 0.302 | 0.409 |
| opp switch: n | 0 | 0 | 199 | 199 | 457 | 457 |
| opp switch: top-1 | - | - | 0.196 | 0.151 | 0.346 | 0.770 |
| opp switch: top-3 | - | - | 0.492 | 0.407 | 0.641 | 0.963 |
| opp switch: in kept 9 | - | - | 1.000 | 1.000 | 1.000 | 1.000 |
| opp switch: mean p(real) | - | - | 0.163 | 0.138 | 0.280 | 0.505 |
| opp cross-entropy (nats) | - | - | 1.932 | 2.305 | 1.914 | 1.032 |
| actual switch share | - | - | 0.076 | 0.076 | 0.154 | 0.154 |
| predicted switch mass | - | - | 0.118 | 0.129 | 0.174 | 0.171 |
| Mega: scored n | 0 | 0 | 691 | 691 | 558 | 558 |
| Mega: BCE | - | - | 0.491 | 0.632 | 0.467 | 0.332 |
| Mega: mean p when they did | - | - | 0.630 | 0.704 | 0.687 | 0.771 |
| Mega: mean p when they did not | - | - | 0.254 | 0.312 | 0.213 | 0.213 |
| value: decisions with a target | 0 | 0 | 3606 | 3606 | 4176 | 4176 |
| value: battles won / lost | 0 / 0 | 0 / 0 | 190 / 110 | 190 / 110 | 187 / 179 | 187 / 179 |
| value: Brier of (v+1)/2 | - | - | 0.255 | 0.255 | 0.176 | 0.176 |
| value: Brier of constant base rate | - | - | 0.240 | 0.240 | 0.242 | 0.242 |
| value: two-hot cross-entropy | - | - | 2.312 | 2.312 | 2.143 | 2.143 |
| value: mean |v - return| | - | - | 0.885 | 0.885 | 0.767 | 0.767 |
| value: last decision, won battles | - | - | 0.818 | 0.818 | 0.733 | 0.733 |
| value: last decision, lost battles | - | - | 0.033 | 0.033 | 0.080 | 0.080 |

Value calibration (win chance = (v+1)/2):

| win-chance bin | live held-out before: n / predicted / actual | live held-out after: n / predicted / actual | simulator before: n / predicted / actual | simulator after: n / predicted / actual | live train before: n / predicted / actual | live train after: n / predicted / actual |
|---|---|---|---|---|---|---|
| 0.0-0.2 | 0 / None / None | 0 / None / None | 1560 / 0.022 / 0.356 | 1560 / 0.022 / 0.356 | 1542 / 0.039 / 0.145 | 1542 / 0.039 / 0.145 |
| 0.2-0.4 | 0 / None / None | 0 / None / None | 571 / 0.294 / 0.643 | 571 / 0.294 / 0.643 | 847 / 0.302 / 0.331 | 847 / 0.302 / 0.331 |
| 0.4-0.6 | 0 / None / None | 0 / None / None | 534 / 0.498 / 0.743 | 534 / 0.498 / 0.743 | 823 / 0.501 / 0.531 | 823 / 0.501 / 0.531 |
| 0.6-0.8 | 0 / None / None | 0 / None / None | 487 / 0.697 / 0.844 | 487 / 0.697 / 0.844 | 642 / 0.694 / 0.731 | 642 / 0.694 / 0.731 |
| 0.8-1.0 | 0 / None / None | 0 / None / None | 454 / 0.919 / 0.943 | 454 / 0.919 / 0.943 | 322 / 0.894 / 0.916 | 322 / 0.894 / 0.916 |

## Policy change (before vs after; decisions with at least two legal actions)

| set | n | mean KL(before||after) | median KL | 99th pct KL | max KL | mean total variation | greedy changed | share |
|---|---|---|---|---|---|---|---|---|
| live held-out | 0 | - | - | - | - | - | - | - |
| simulator | 3385 | 0.060 | 0.007 | 0.902 | 5.407 | 0.074 | 278 | 0.082 |
| live train | 3907 | 0.092 | 0.007 | 1.621 | 7.785 | 0.081 | 352 | 0.090 |

## After minus before on the same scored decisions (95% interval, resampling battles)

| set | battles | top-1 | top-3 | in kept 9 | mean p(real) |
|---|---|---|---|---|---|
| simulator | 300 | -0.0780 [-0.0988, -0.0585] | -0.0944 [-0.1125, -0.0760] | -0.0537 [-0.0655, -0.0428] | -0.0679 [-0.0793, -0.0571] |
| live train | 368 | +0.2492 [+0.2293, +0.2708] | +0.1840 [+0.1685, +0.1994] | +0.0656 [+0.0555, +0.0761] | +0.1252 [+0.1150, +0.1354] |

## Training

Trained tensors (6, 33153 parameters of 1798048): `opp_move_q.weight`, `opp_move_q.bias`, `opp_switch_q.weight`, `opp_switch_q.bias`, `opp_mega.weight`, `opp_mega.bias`.

Tensors that differ from the input checkpoint: 6 (all inside the trained set); frozen tensors compared bit for bit: 141, identical: 141. Trained tensors left unchanged: [].

| epoch | steps | opp CE | Mega BCE | value CE | opp top-1 | opp top-3 | grad norm |
|---|---|---|---|---|---|---|---|
| 1 | 17 | 1.843 | 0.460 | 2.272 | 0.418 | 0.747 | 1.036 |
| 2 | 34 | 1.679 | 0.440 | 2.272 | 0.465 | 0.780 | 0.909 |
| 3 | 51 | 1.605 | 0.429 | 2.271 | 0.480 | 0.797 | 0.848 |
| 4 | 68 | 1.551 | 0.414 | 2.270 | 0.497 | 0.809 | 0.808 |
| 5 | 85 | 1.514 | 0.407 | 2.271 | 0.506 | 0.824 | 0.793 |
| 6 | 102 | 1.481 | 0.395 | 2.271 | 0.515 | 0.828 | 0.779 |
| 7 | 119 | 1.456 | 0.390 | 2.270 | 0.527 | 0.836 | 0.761 |
| 8 | 136 | 1.433 | 0.379 | 2.272 | 0.528 | 0.841 | 0.758 |
| 9 | 153 | 1.409 | 0.377 | 2.271 | 0.530 | 0.842 | 0.750 |
| 10 | 170 | 1.391 | 0.372 | 2.270 | 0.541 | 0.850 | 0.740 |
| 11 | 187 | 1.372 | 0.370 | 2.273 | 0.548 | 0.851 | 0.713 |
| 12 | 204 | 1.353 | 0.363 | 2.270 | 0.553 | 0.857 | 0.701 |
| 13 | 221 | 1.339 | 0.364 | 2.271 | 0.561 | 0.860 | 0.717 |
| 14 | 238 | 1.326 | 0.357 | 2.271 | 0.555 | 0.863 | 0.708 |
| 15 | 255 | 1.312 | 0.356 | 2.270 | 0.569 | 0.867 | 0.744 |
| 16 | 272 | 1.300 | 0.357 | 2.271 | 0.573 | 0.868 | 0.694 |
| 17 | 289 | 1.289 | 0.354 | 2.272 | 0.578 | 0.870 | 0.691 |
| 18 | 306 | 1.277 | 0.357 | 2.271 | 0.579 | 0.873 | 0.686 |
| 19 | 323 | 1.266 | 0.350 | 2.270 | 0.578 | 0.874 | 0.689 |
| 20 | 340 | 1.257 | 0.346 | 2.271 | 0.584 | 0.879 | 0.690 |
| 21 | 357 | 1.247 | 0.348 | 2.272 | 0.590 | 0.881 | 0.705 |
| 22 | 374 | 1.238 | 0.351 | 2.270 | 0.596 | 0.885 | 0.684 |
| 23 | 391 | 1.226 | 0.344 | 2.272 | 0.597 | 0.882 | 0.682 |
| 24 | 408 | 1.215 | 0.342 | 2.272 | 0.600 | 0.886 | 0.668 |
| 25 | 425 | 1.207 | 0.338 | 2.271 | 0.603 | 0.892 | 0.648 |
| 26 | 442 | 1.196 | 0.341 | 2.272 | 0.611 | 0.895 | 0.663 |
| 27 | 459 | 1.187 | 0.340 | 2.272 | 0.606 | 0.896 | 0.653 |
| 28 | 476 | 1.181 | 0.340 | 2.270 | 0.615 | 0.897 | 0.670 |
| 29 | 493 | 1.176 | 0.340 | 2.272 | 0.617 | 0.899 | 0.649 |
| 30 | 510 | 1.163 | 0.335 | 2.269 | 0.622 | 0.904 | 0.636 |
| 31 | 527 | 1.157 | 0.330 | 2.270 | 0.621 | 0.901 | 0.664 |
| 32 | 544 | 1.149 | 0.337 | 2.270 | 0.627 | 0.905 | 0.638 |
| 33 | 561 | 1.139 | 0.331 | 2.270 | 0.637 | 0.909 | 0.640 |
| 34 | 578 | 1.137 | 0.335 | 2.272 | 0.632 | 0.906 | 0.639 |
| 35 | 595 | 1.129 | 0.331 | 2.269 | 0.640 | 0.909 | 0.639 |
| 36 | 612 | 1.125 | 0.331 | 2.272 | 0.640 | 0.909 | 0.646 |
| 37 | 629 | 1.115 | 0.328 | 2.270 | 0.644 | 0.913 | 0.630 |
| 38 | 646 | 1.110 | 0.328 | 2.271 | 0.641 | 0.916 | 0.623 |
| 39 | 663 | 1.100 | 0.325 | 2.270 | 0.655 | 0.918 | 0.625 |
| 40 | 680 | 1.096 | 0.324 | 2.269 | 0.643 | 0.914 | 0.611 |
| 41 | 697 | 1.088 | 0.326 | 2.270 | 0.651 | 0.916 | 0.623 |
| 42 | 714 | 1.085 | 0.325 | 2.270 | 0.656 | 0.919 | 0.625 |
| 43 | 731 | 1.078 | 0.323 | 2.270 | 0.654 | 0.920 | 0.625 |
| 44 | 748 | 1.069 | 0.327 | 2.271 | 0.657 | 0.921 | 0.596 |
| 45 | 765 | 1.065 | 0.321 | 2.270 | 0.663 | 0.922 | 0.613 |
| 46 | 782 | 1.057 | 0.322 | 2.271 | 0.669 | 0.923 | 0.617 |
| 47 | 799 | 1.053 | 0.321 | 2.271 | 0.664 | 0.924 | 0.609 |
| 48 | 816 | 1.048 | 0.325 | 2.269 | 0.665 | 0.927 | 0.614 |
| 49 | 833 | 1.040 | 0.322 | 2.271 | 0.669 | 0.927 | 0.598 |
| 50 | 850 | 1.040 | 0.320 | 2.271 | 0.669 | 0.925 | 0.616 |

Per-epoch numbers are means over that epoch's minibatches while the weights move (training data).

Note: the held-out set is 0 battles / 0 labelled opponent decisions / 0 battles with an outcome: every held-out number is a small sample.

Note: `live train` is in-sample after training; it only shows how far the trained layers moved.

Note: The value target here is the plain discounted return (no bootstrap); training's PPO target is the GAE(0.9) return, which mixes in the critic's own estimates.

Note: Brier and calibration read (v+1)/2 as a win chance, as ladder_check does; v also carries the +-0.1 faint rewards, so it is not exactly a win chance.
