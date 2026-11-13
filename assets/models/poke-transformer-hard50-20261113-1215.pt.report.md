# Live fine-tune report (2026-11-13 12:01:42)

Input `C:\Users\geral\Code\research\pcsa-rl\runs\live-finetune\allheads-20261113\hard50-p.pt` (update 50); output `C:\Users\geral\Code\research\pcsa-rl\runs\live-finetune\allheads-20261113\hard50-p-opp.pt`. Epochs 50, lr 0.001, minibatch ~256 decisions, seed 0, hold-out 0.0. Runtime 130.2 s (CPU, 2 threads).

## Data

Folders: `C:/Users/geral/Code/pcsa/output/runs`, `C:/Users/geral/Code/pc-singles-agent/output/runs`. Battle folders 441, recorded decisions 4516, rebuilt 4510. Skipped battles: {}. Outcomes: {"loss": 207, "win": 147, "opponent_forfeit": 44, "no battle.json": 11, "unknown": 32}. Battles whose final faint count was completed (the phone does not record the last turn): 352.

| part | battles | decisions | battles with value target | decisions with value target | battles without | decisions without | decisions with opponent label | won | lost | from | to |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 441 | 4510 | 398 | 4109 | 43 | 401 | 2889 | 191 | 207 | 20261101T193513 | 20261113T055700 |
| held out | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |

Held-out battles by folder {}, by live player {}; train by folder {"pc-singles-agent": 232, "pcsa": 209}, by live player {"heuristic": 1, "search": 25, "policy": 206, "best_matchup_max_damage": 1, "disciplined": 4, "neural": 204}.

Decision failures (folder: stage: exception):

| failure | n |
|---|---|
| pc-singles-agent: legal_actions: none | 3 |
| pcsa: legal_actions: none | 3 |

Simulator reference: 300 games, 3378 decisions from `u00150.npz`.

## Before / after

| metric | live held-out before | live held-out after | simulator before | simulator after | live train before | live train after |
|---|---|---|---|---|---|---|
| battles | 0 | 0 | 300 | 300 | 441 | 441 |
| decisions | 0 | 0 | 3378 | 3378 | 4510 | 4510 |
| opp all scored: n | 0 | 0 | 2572 | 2572 | 2875 | 2875 |
| opp all scored: top-1 | - | - | 0.559 | 0.406 | 0.369 | 0.669 |
| opp all scored: top-3 | - | - | 0.852 | 0.745 | 0.652 | 0.911 |
| opp all scored: in kept 9 | - | - | 0.959 | 0.912 | 0.841 | 0.955 |
| opp all scored: mean p(real) | - | - | 0.424 | 0.323 | 0.276 | 0.462 |
| opp move already revealed: n | 0 | 0 | 826 | 826 | 775 | 775 |
| opp move already revealed: top-1 | - | - | 0.696 | 0.473 | 0.564 | 0.816 |
| opp move already revealed: top-3 | - | - | 0.955 | 0.841 | 0.840 | 0.977 |
| opp move already revealed: in kept 9 | - | - | 1.000 | 1.000 | 1.000 | 1.000 |
| opp move already revealed: mean p(real) | - | - | 0.533 | 0.377 | 0.407 | 0.563 |
| opp move not yet revealed: n | 0 | 0 | 1394 | 1394 | 1742 | 1742 |
| opp move not yet revealed: top-1 | - | - | 0.534 | 0.415 | 0.321 | 0.550 |
| opp move not yet revealed: top-3 | - | - | 0.842 | 0.751 | 0.611 | 0.864 |
| opp move not yet revealed: in kept 9 | - | - | 0.924 | 0.839 | 0.738 | 0.926 |
| opp move not yet revealed: mean p(real) | - | - | 0.399 | 0.328 | 0.242 | 0.379 |
| opp switch: n | 0 | 0 | 352 | 352 | 358 | 358 |
| opp switch: top-1 | - | - | 0.332 | 0.210 | 0.184 | 0.927 |
| opp switch: top-3 | - | - | 0.645 | 0.497 | 0.444 | 0.994 |
| opp switch: in kept 9 | - | - | 1.000 | 1.000 | 1.000 | 1.000 |
| opp switch: mean p(real) | - | - | 0.266 | 0.179 | 0.158 | 0.644 |
| opp cross-entropy (nats) | - | - | 1.325 | 1.917 | 2.200 | 1.058 |
| actual switch share | - | - | 0.137 | 0.137 | 0.124 | 0.124 |
| predicted switch mass | - | - | 0.134 | 0.139 | 0.138 | 0.126 |
| Mega: scored n | 0 | 0 | 481 | 481 | 632 | 632 |
| Mega: BCE | - | - | 0.376 | 0.455 | 0.512 | 0.410 |
| Mega: mean p when they did | - | - | 0.719 | 0.749 | 0.630 | 0.691 |
| Mega: mean p when they did not | - | - | 0.208 | 0.283 | 0.247 | 0.240 |
| value: decisions with a target | 0 | 0 | 3378 | 3378 | 4109 | 4109 |
| value: battles won / lost | 0 / 0 | 0 / 0 | 221 / 79 | 221 / 79 | 191 / 207 | 191 / 207 |
| value: Brier of (v+1)/2 | - | - | 0.107 | 0.107 | 0.244 | 0.244 |
| value: Brier of constant base rate | - | - | 0.214 | 0.214 | 0.237 | 0.237 |
| value: two-hot cross-entropy | - | - | 3.230 | 3.230 | 4.819 | 4.819 |
| value: mean |v - return| | - | - | 0.547 | 0.547 | 0.924 | 0.924 |
| value: last decision, won battles | - | - | 0.980 | 0.980 | 0.840 | 0.840 |
| value: last decision, lost battles | - | - | 0.089 | 0.089 | 0.193 | 0.193 |

Value calibration (win chance = (v+1)/2):

| win-chance bin | live held-out before: n / predicted / actual | live held-out after: n / predicted / actual | simulator before: n / predicted / actual | simulator after: n / predicted / actual | live train before: n / predicted / actual | live train after: n / predicted / actual |
|---|---|---|---|---|---|---|
| 0.0-0.2 | 0 / None / None | 0 / None / None | 540 / 0.042 / 0.074 | 540 / 0.042 / 0.074 | 662 / 0.072 / 0.106 | 662 / 0.072 / 0.106 |
| 0.2-0.4 | 0 / None / None | 0 / None / None | 301 / 0.299 / 0.342 | 301 / 0.299 / 0.342 | 568 / 0.306 / 0.222 | 568 / 0.306 / 0.222 |
| 0.4-0.6 | 0 / None / None | 0 / None / None | 310 / 0.504 / 0.581 | 310 / 0.504 / 0.581 | 789 / 0.5 / 0.324 | 789 / 0.5 / 0.324 |
| 0.6-0.8 | 0 / None / None | 0 / None / None | 508 / 0.713 / 0.756 | 508 / 0.713 / 0.756 | 940 / 0.702 / 0.403 | 940 / 0.702 / 0.403 |
| 0.8-1.0 | 0 / None / None | 0 / None / None | 1719 / 0.964 / 0.942 | 1719 / 0.964 / 0.942 | 1150 / 0.932 / 0.655 | 1150 / 0.932 / 0.655 |

## Policy change (before vs after; decisions with at least two legal actions)

| set | n | mean KL(before||after) | median KL | 99th pct KL | max KL | mean total variation | greedy changed | share |
|---|---|---|---|---|---|---|---|---|
| live held-out | 0 | - | - | - | - | - | - | - |
| simulator | 3208 | 0.077 | 0.007 | 1.224 | 7.054 | 0.080 | 303 | 0.095 |
| live train | 4094 | 0.096 | 0.008 | 1.543 | 7.074 | 0.086 | 389 | 0.095 |

## After minus before on the same scored decisions (95% interval, resampling battles)

| set | battles | top-1 | top-3 | in kept 9 | mean p(real) |
|---|---|---|---|---|---|
| simulator | 299 | -0.1528 [-0.1751, -0.1315] | -0.1061 [-0.1233, -0.0888] | -0.0463 [-0.0571, -0.0363] | -0.1008 [-0.1126, -0.0888] |
| live train | 431 | +0.2995 [+0.2780, +0.3212] | +0.2584 [+0.2387, +0.2794] | +0.1141 [+0.1008, +0.1279] | +0.1857 [+0.1757, +0.1967] |

## Training

Trained tensors (6, 33153 parameters of 1798048): `opp_move_q.weight`, `opp_move_q.bias`, `opp_switch_q.weight`, `opp_switch_q.bias`, `opp_mega.weight`, `opp_mega.bias`.

Tensors that differ from the input checkpoint: 6 (all inside the trained set); frozen tensors compared bit for bit: 141, identical: 141. Trained tensors left unchanged: [].

| epoch | steps | opp CE | Mega BCE | value CE | opp top-1 | opp top-3 | grad norm |
|---|---|---|---|---|---|---|---|
| 1 | 18 | 2.178 | 0.516 | 5.246 | 0.368 | 0.655 | 1.166 |
| 2 | 36 | 1.989 | 0.485 | 5.257 | 0.396 | 0.688 | 1.067 |
| 3 | 54 | 1.884 | 0.478 | 5.252 | 0.417 | 0.719 | 0.953 |
| 4 | 72 | 1.815 | 0.471 | 5.277 | 0.438 | 0.737 | 0.934 |
| 5 | 90 | 1.752 | 0.463 | 5.256 | 0.457 | 0.752 | 0.908 |
| 6 | 108 | 1.706 | 0.461 | 5.251 | 0.466 | 0.768 | 0.882 |
| 7 | 126 | 1.663 | 0.454 | 5.257 | 0.480 | 0.777 | 0.878 |
| 8 | 144 | 1.628 | 0.447 | 5.258 | 0.489 | 0.791 | 0.842 |
| 9 | 162 | 1.594 | 0.451 | 5.250 | 0.495 | 0.799 | 0.836 |
| 10 | 180 | 1.570 | 0.442 | 5.256 | 0.507 | 0.795 | 0.863 |
| 11 | 198 | 1.540 | 0.446 | 5.260 | 0.513 | 0.815 | 0.821 |
| 12 | 216 | 1.514 | 0.440 | 5.262 | 0.520 | 0.817 | 0.819 |
| 13 | 234 | 1.493 | 0.436 | 5.269 | 0.525 | 0.826 | 0.824 |
| 14 | 252 | 1.473 | 0.431 | 5.250 | 0.532 | 0.831 | 0.811 |
| 15 | 270 | 1.450 | 0.432 | 5.248 | 0.549 | 0.832 | 0.800 |
| 16 | 288 | 1.432 | 0.431 | 5.255 | 0.547 | 0.838 | 0.785 |
| 17 | 306 | 1.412 | 0.432 | 5.247 | 0.544 | 0.841 | 0.788 |
| 18 | 324 | 1.395 | 0.433 | 5.260 | 0.555 | 0.844 | 0.792 |
| 19 | 342 | 1.382 | 0.437 | 5.262 | 0.562 | 0.851 | 0.767 |
| 20 | 360 | 1.365 | 0.430 | 5.253 | 0.571 | 0.851 | 0.793 |
| 21 | 378 | 1.353 | 0.426 | 5.254 | 0.573 | 0.855 | 0.767 |
| 22 | 396 | 1.334 | 0.424 | 5.245 | 0.577 | 0.861 | 0.757 |
| 23 | 414 | 1.324 | 0.429 | 5.253 | 0.576 | 0.863 | 0.775 |
| 24 | 432 | 1.308 | 0.418 | 5.247 | 0.582 | 0.864 | 0.756 |
| 25 | 450 | 1.294 | 0.424 | 5.268 | 0.592 | 0.872 | 0.738 |
| 26 | 468 | 1.282 | 0.420 | 5.259 | 0.594 | 0.870 | 0.748 |
| 27 | 486 | 1.271 | 0.420 | 5.266 | 0.597 | 0.873 | 0.724 |
| 28 | 504 | 1.262 | 0.413 | 5.267 | 0.601 | 0.875 | 0.732 |
| 29 | 522 | 1.249 | 0.415 | 5.262 | 0.603 | 0.879 | 0.734 |
| 30 | 540 | 1.237 | 0.417 | 5.257 | 0.618 | 0.878 | 0.736 |
| 31 | 558 | 1.232 | 0.414 | 5.254 | 0.609 | 0.880 | 0.725 |
| 32 | 576 | 1.216 | 0.419 | 5.246 | 0.618 | 0.883 | 0.709 |
| 33 | 594 | 1.209 | 0.415 | 5.261 | 0.619 | 0.885 | 0.759 |
| 34 | 612 | 1.197 | 0.415 | 5.247 | 0.622 | 0.887 | 0.715 |
| 35 | 630 | 1.188 | 0.410 | 5.261 | 0.628 | 0.888 | 0.720 |
| 36 | 648 | 1.175 | 0.414 | 5.251 | 0.633 | 0.892 | 0.688 |
| 37 | 666 | 1.171 | 0.408 | 5.242 | 0.638 | 0.891 | 0.718 |
| 38 | 684 | 1.158 | 0.415 | 5.255 | 0.637 | 0.895 | 0.695 |
| 39 | 702 | 1.151 | 0.406 | 5.258 | 0.641 | 0.896 | 0.707 |
| 40 | 720 | 1.145 | 0.408 | 5.254 | 0.641 | 0.897 | 0.700 |
| 41 | 738 | 1.135 | 0.413 | 5.259 | 0.648 | 0.898 | 0.685 |
| 42 | 756 | 1.127 | 0.411 | 5.253 | 0.646 | 0.899 | 0.672 |
| 43 | 774 | 1.121 | 0.410 | 5.259 | 0.650 | 0.901 | 0.694 |
| 44 | 792 | 1.113 | 0.409 | 5.253 | 0.654 | 0.899 | 0.685 |
| 45 | 810 | 1.104 | 0.409 | 5.260 | 0.649 | 0.904 | 0.678 |
| 46 | 828 | 1.098 | 0.408 | 5.249 | 0.652 | 0.908 | 0.669 |
| 47 | 846 | 1.090 | 0.411 | 5.266 | 0.661 | 0.907 | 0.697 |
| 48 | 864 | 1.083 | 0.409 | 5.261 | 0.656 | 0.909 | 0.659 |
| 49 | 882 | 1.077 | 0.408 | 5.261 | 0.661 | 0.907 | 0.686 |
| 50 | 900 | 1.071 | 0.402 | 5.253 | 0.657 | 0.911 | 0.655 |

Per-epoch numbers are means over that epoch's minibatches while the weights move (training data).

Note: the held-out set is 0 battles / 0 labelled opponent decisions / 0 battles with an outcome: every held-out number is a small sample.

Note: `live train` is in-sample after training; it only shows how far the trained layers moved.

Note: The value target here is the plain discounted return (no bootstrap); training's PPO target is the GAE(0.9) return, which mixes in the critic's own estimates.

Note: Brier and calibration read (v+1)/2 as a win chance, as ladder_check does; v also carries the +-0.1 faint rewards, so it is not exactly a win chance.
