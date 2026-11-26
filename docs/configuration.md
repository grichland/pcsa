# Configuration

Generated from `config/agent.yaml` by `tools/config_reference.py`; edit the YAML file, not this page.

Every value below is the shipped one. `--set section.key=value` on any `pcsa` command, or the environment
variable `PCSA__section__key`, changes one for a single run. Unknown and missing keys are refused at start-up.


## `screen_config`

`screen.yaml`


## `decide`


| Key | Value | Meaning |
|---|---|---|
| `decide.player` | `cg` | cg: Champions Gym (the section below); first: the first legal action (for tests) |

## `cg`

Champions Gym and the player it runs (pcsa.cg)

| Key | Value | Meaning |
|---|---|---|
| `cg.transport` | `subprocess` | subprocess: JSON lines on stdin and stdout; inprocess: a Python entry point |
| `cg.command` | `['{python}', -m, pcsa.cg.serve]` | subprocess: {python} is this environment's interpreter |
| `cg.entry` | `null` | inprocess: "package.module:name", a handle(message) callable or a class with .handle |
| `cg.args` | `{}` | inprocess: keyword arguments for the entry class |
| `cg.player` | `champions_gym.neural:load` | the Session's player; a built-in such as tower needs no player_args |
| `cg.player_args` |  | passed to the player as given; never set allow_mismatch or allow_contract |
| `cg.player_args.policy` | `pcsa.network.policy:Policy` |  |
| `cg.player_args.checkpoint` | `assets/models/finetune-20261124-g75-iter-00010.pt` | relative: the repo root |
| `cg.player_args.device` | `cpu` |  |
| `cg.player_args.policy_args` | `{threads: 4}` | plus rules, search and choice below |
| `cg.player_args.fallback` | `tower` | the shell's player when the network fails or is too slow |
| `cg.player_args.timeout_s` | `2.0` | a slower answer counts against the network |
| `cg.player_args.hard_timeout_s` | `6.0` | an answer this late goes to the fallback |
| `cg.python_path` | `[]` | directories first on sys.path (inprocess) or in the child's PYTHONPATH; relative: repo root |
| `cg.env` | `{}` | subprocess: extra environment variables (the GPU stays visible for preview_search) |
| `cg.run_log_dir` | `cg` | CG's log folder inside the run folder; null: none |
| `cg.timeout_s` | `8.0` | per call, and never past the decision's deadline less the tap reserve |
| `cg.start_timeout_s` | `60.0` | CG is made and its status checked before the first battle |
| `cg.rules` |  | training's rule masks (pcsa.network.rules); set them as the checkpoint was trained |
| `cg.rules.enabled` | `true` |  |
| `cg.rules.max_same_move` | `3` | the same move at most this many turns in a row (0: no limit) |
| `cg.rules.max_switches` | `2` | at most this many voluntary switches in a row (0: no limit) |
| `cg.rules.max_boosts` | `3` | setup moves by the same Pokémon in a row (0: no limit) |
| `cg.rules.max_heals` | `3` | pure recovery moves by the same Pokémon in a row (0: no limit) |
| `cg.rules.no_null_moves` | `true` | close damaging moves that do nothing to their active (immunity, Levitate, terrain) |
| `cg.rules.no_boost_when_ko` | `true` | close setup, other than Def, Sp. Def or evasion, while their best hit KOs us |
| `cg.rules.no_failing_status` | `true` | close status moves that cannot land (already statused, Safeguard, immunity) |
| `cg.rules.no_redundant_field` | `true` | close hazards at their maximum and weather or terrain already up |
| `cg.rules.no_double_protect` | `true` | close the Protect family two turns in a row |
| `cg.rules.no_wasted_self` | `true` | close recovery at full HP, setup at +6, Substitute when it would fail |
| `cg.rules.status_free_mirrors` | `[primarina]` | not training's: close status moves when both actives are one of these |
| `cg.rules.always_mega` | `true` | not training's: close plain moves while their Mega copy is open, so we always Mega Evolve |
| `cg.rules.one_mega_lineup` | `true` | not training's: close team-preview picks that bring two or more Mega stone holders |
| `cg.search` |  | decision-time search at turn requests (pcsa.network.search) |
| `cg.search.enabled` | `false` | needs Champions Gym 0.5.0 or later (0.7.1 is pinned) |
| `cg.search.ms` | `1500` | time per turn request; never past the request's deadline less the tap reserve |
| `cg.search.worlds` | `6` | sampled completions of the opponent's hidden team, one tree each |
| `cg.search.depth` | `3` | turns looked ahead |
| `cg.search.outcomes` | `3` | random outcomes drawn per pair of actions |
| `cg.search.c` | `1.5` | weight of the network's priors in the choice of what to try |
| `cg.search.n0` | `1` | weight of the network's matrix cell in each pair's value |
| `cg.search.batch` | `16` | positions per network call |
| `cg.search.min_sims` | `64` | fewer simulations than this in total: the network's own answer is played |
| `cg.search.sims` | `0` | stop after this many simulations (0: by time only) |
| `cg.search.seed` | `0` | the search's random seed |
| `cg.search.top_ours_moves` | `0` | at every position only our N likeliest moves by the network's policy, Mega copy included (0: all) |
| `cg.search.top_ours_switches` | `0` | ... and our N likeliest switches (0: all) |
| `cg.search.top_theirs_moves` | `3` | at every position only their N likeliest moves by the network's prediction, Mega copy included (0: all) |
| `cg.search.top_theirs_switches` | `2` | ... and their N likeliest switches (0: all) |
| `cg.search.from_turn` | `1` | earlier turns of the battle are not searched |
| `cg.search.settle_sims` | `300` | stop before `ms` once the pick has led this many simulations (0: always use `ms`) |
| `cg.search.settle_share` | `0.9` | ... and holds this share of the search's strategy |
| `cg.search.settle_gap` | `0.01` | or once the best two actions are this close in win chance, after twice settle_sims |
| `cg.search.follow_pivot` | `true` | a pivot move picked: its switch-in is the one the search chose for after it |
| `cg.search.worlds_mode` | `builds` | their sets: whole known builds weighted by what the battle showed (legacy: part by part) |
| `cg.search.preview_rates` | `assets/sets/opponent_preview_rates.json` | their unseen members by live bring rates (null: the head) |
| `cg.search.calibrate` | `false` | values, cells and scores through value_calibration (on the scale of a won +1 or lost -1 game) |
| `cg.choice` |  | how the pick is made from the network's matrix at turn requests (pcsa.network.choosing) |
| `cg.choice.rule` | `regret` | mean: expected score (with top_theirs 0: the network's own pick); min; mix; regret |
| `cg.choice.top_theirs` | `8` | over their N likeliest replies by the network's prediction (0: all) |
| `cg.choice.top_ours` | `8` | among our N likeliest actions by the network's policy (0: all legal) |
| `cg.choice.blend` | `0.5` | mix only: the weight of the mean against the worst case |
| `cg.value_calibration` |  | the value reported as 2 * sigmoid(a * v + b) - 1, fitted per checkpoint |
| `cg.value_calibration.enabled` | `true` | by tools/value_calibration.py; off: the raw value |
| `cg.value_calibration.a` | `1.43017` | finetune-20261124-g75-iter-00010 on 251 recorded battles |
| `cg.value_calibration.b` | `-0.48124` |  |
| `cg.preview_search` |  | replaces the preview head's pick (pcsa.network.preview_search) |
| `cg.preview_search.enabled` | `true` | off: the head's pick, exactly |
| `cg.preview_search.budget_s` | `30` | search time; never past the preview deadline less the tap reserve |
| `cg.preview_search.guesses` | `3` | versions of their team, sets drawn from the belief |
| `cg.preview_search.race` | `[20, 6, 2]` | rows kept at 1/4, 1/2, 3/4 of the budget; the head's pick always kept |
| `cg.preview_search.prior_games` | `2` | stage A counts as this many games in each cell |
| `cg.preview_search.choose` | `mean` | mean (tested best) or maximin |
| `cg.preview_search.keep_head_margin` | `0.03` | leave the head's pick only when the best row's mean is this much higher |
| `cg.preview_search.device` | `cuda` | the playouts run on the GPU; the live player stays on the CPU |
| `cg.preview_search.threads` | `8` | Champions Gym threads |
| `cg.preview_search.seed` | `0` |  |
| `cg.preview_search.worlds_mode` | `builds` | their sets: whole known builds (legacy: part by part) |

## `device`


| Key | Value | Meaning |
|---|---|---|
| `device.serial` | `null` | null: the single authorized device |
| `device.timeout_s` | `15` |  |
| `device.stay_awake` | `true` | svc power stayon usb during runs; previous value restored on exit |
| `device.retry` |  |  |
| `device.retry.attempts` | `4` | per ADB call before the session is restarted |
| `device.retry.base_s` | `0.5` | exponential backoff: base, 2*base, 4*base ... up to max_s |
| `device.retry.max_s` | `8.0` |  |
| `device.retry.reconnect_wait_s` | `20` | adb wait-for-device bound after a reconnect |

## `run`


| Key | Value | Meaning |
|---|---|---|
| `run.name` | `run` | run folder prefix under output/runs |
| `run.seed` | `null` | null: a new random seed; a number repeats choices and taps |
| `run.battles` | `10` | battles with a decided outcome before the run finishes |
| `run.max_battles_total` | `12` | bound on battles started, including opponent forfeits |
| `run.count_forfeits` | `false` |  |
| `run.max_restarts_per_hour` | `20` |  |

## `timing`


| Key | Value | Meaning |
|---|---|---|
| `timing.inter_tap_s` | `0.5` |  |
| `timing.post_action_wait_s` | `10.0` |  |
| `timing.poll_interval_s` | `1.0` | the longest pause after a screen check |
| `timing.poll_cycle_s` | `2.0` | a screen check starts this long after the one before; a slower check is followed at once |
| `timing.fight_settle_s` | `0.9` |  |
| `timing.back_settle_s` | `0.7` |  |
| `timing.party_settle_s` | `0.9` | a switch: after Pokemon and after Switch In, before the party screen is read |
| `timing.move_peek` | `when_needed` | always: look at the move list before every action turn (Fight, read, Back). when_needed: skip the look (about 3.4 s a turn) and take move names from the team sheet, PP from the log count and the Mega button from the log; the move row is then tapped twice. The turn still looks when a move's counted PP is 4 or less, the log read lost or doubted rows, the last taps did not go through, a Mega chosen last turn is not in the log, the battle was picked up midway or could not be tracked. |
| `timing.team_menu_settle_s` | `0.8` |  |
| `timing.log_open_settle_s` | `1.2` |  |
| `timing.log_swipe_settle_s` | `0.6` |  |
| `timing.max_turn_s` | `40.0` | from the first action-screen capture to the last tap |
| `timing.team_budget_s` | `45.0` | lineup deadline when the selection timer cannot be read |
| `timing.no_progress_s` | `420` | no tap and no screen change for this long -> session restart |
| `timing.default_screen_timeout_s` | `60` |  |
| `timing.screen_timeouts` |  | seconds on one screen before the session restarts |
| `timing.screen_timeouts.matchmaking` | `240` |  |
| `timing.screen_timeouts.prep` | `240` |  |
| `timing.screen_timeouts.transition` | `150` |  |
| `timing.screen_timeouts.wait` | `150` |  |
| `timing.screen_timeouts.rewards` | `90` |  |
| `timing.screen_timeouts.result` | `90` |  |
| `timing.screen_timeouts.forfeit_msg` | `90` |  |
| `timing.screen_timeouts.unknown` | `45` |  |
| `timing.screen_timeouts.lobby` | `60` |  |
| `timing.screen_timeouts.options` | `60` |  |
| `timing.screen_timeouts.team_select` | `100` |  |
| `timing.screen_timeouts.action` | `120` |  |
| `timing.screen_timeouts.forced` | `90` |  |
| `timing.screen_timeouts.party` | `30` |  |
| `timing.screen_timeouts.moves` | `30` |  |
| `timing.screen_timeouts.log` | `30` |  |
| `timing.screen_timeouts.switch_popup` | `30` |  |
| `timing.screen_timeouts.rank_up` | `120` |  |
| `timing.screen_timeouts.rating_dialog` | `30` |  |
| `timing.screen_timeouts.dialog` | `30` |  |
| `timing.screen_timeouts.comm_error` | `30` |  |
| `timing.screen_timeouts.title` | `60` |  |
| `timing.screen_timeouts.home` | `60` |  |
| `timing.screen_timeouts.battle_modes` | `60` |  |
| `timing.screen_timeouts.ranked_formats` | `60` |  |

## `log`

in-game battle log collection, per turn

| Key | Value | Meaning |
|---|---|---|
| `log.captures_max` | `10` |  |
| `log.seconds_max` | `12.0` | stop scrolling after this; act on what was read |
| `log.drag` |  | drag toward older entries; start and end drawn at random inside these ranges |
| `log.drag.x` | `[0.513, 0.684]` |  |
| `log.drag.y_start` | `[0.231, 0.278]` |  |
| `log.drag.y_end` | `[0.833, 0.870]` |  |
| `log.drag.steps` | `8` |  |
| `log.drag.hold_s` | `0.35` |  |
| `log.paging` |  | heading jumped past turns with no action screen of ours: page back with the arrows |
| `log.paging.enabled` | `true` |  |
| `log.paging.settle_s` | `0.8` | after an arrow tap, before the heading is read |
| `log.paging.step_s` | `1.6` | planning: one arrow tap and its heading check |
| `log.paging.page_min_s` | `4.0` | planning: one missed page (and the least time left to start one) |
| `log.paging.seconds_max` | `10.0` | climb budget per missed page |

## `taps`


| Key | Value | Meaning |
|---|---|---|
| `taps.focused_move_taps` | `1` | a focused move row submits on the next tap |
| `taps.unfocused_move_taps` | `2` | first tap focuses, second submits |
| `taps.focused_party_taps` | `1` |  |
| `taps.unfocused_party_taps` | `2` |  |

## `navigator`


| Key | Value | Meaning |
|---|---|---|
| `navigator.unknown_wait_s` | `5.0` |  |
| `navigator.unknown_max_rounds` | `6` |  |
| `navigator.edge_settle_s` | `1.5` |  |
| `navigator.max_taps_per_edge` | `3` | Disconnect during a battle: while a battle is in progress and the phone shows an unknown screen, a communication error, a dialog or a long blank/loading frame, tap the screen centre (or an OCR-confirmed OK / Close / Continue / Cancel / Not now) once per round until a battle, post-battle or lobby screen returns. |
| `navigator.disconnect_max_taps` | `40` | rounds (taps, or skipped taps when forbidden text is under the centre) before restart |
| `navigator.disconnect_max_s` | `180.0` | seconds of the whole ride-out before the session restarts |
| `navigator.disconnect_settle_s` | `3.0` | wait after each round before observing again |
| `navigator.disconnect_blank_s` | `45.0` | a transition (black/loading) frame mid-battle counts as a disconnect after this long |

## `degrade`


| Key | Value | Meaning |
|---|---|---|
| `degrade.tap_reserve_s` | `2.5` |  |
| `degrade.decide_reserve_s` | `3.0` |  |
| `degrade.peek_attempts` | `2` |  |

## `ocr`


| Key | Value | Meaning |
|---|---|---|
| `ocr.det_model` | `ch_PP-OCRv5_det_mobile.onnx` |  |
| `ocr.rec_model` | `en_PP-OCRv5_rec_mobile.onnx` |  |
| `ocr.det_limit_side_len` | `736` |  |
| `ocr.det_limit_type` | `min` | max: the long side is shrunk to the limit |
| `ocr.threads` | `8` | a whole frame is read in about 1.8 s |
| `ocr.min_confidence` | `0.80` |  |
