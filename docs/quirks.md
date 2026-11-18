# Live quirks

One entry per hard-won fact about how the game behaves on the phone. Each names the symptom, the run that
showed it, and the config key or code that handles it. Add here; do not scatter run IDs in comments.

| Symptom | Seen in | Handled by |
|---|---|---|
| A focused move row submits on the next tap; an unfocused one needs two (first focuses, second submits) | stage-8 probes, Oct 2026 | `taps.focused_move_taps`, `taps.unfocused_move_taps` (`live/taps.py`) |
| The lobby is dimmed for a moment after Continue Battling and misreads as a transition | run 20261025-214430 | the lobby rule in `phone/screens.py` accepts the dimmed lobby when Begin Matchmaking is read |
| The move timer's first digit is sometimes dropped by OCR ("5" for "15") | live runs, Oct 2026 | `MIN_NEW_TURN_MOVE_TIME` in `live/turn.py` guards a short timer |
| OCR splits a button label into words ("Continue" / "Battling") | run 20261025-214430 | labels are joined inside each button region before matching (`phone/screens.py`) |
| The Rank-Up Reward popup slides in; its Close control is offered only once settled | probe1-20261023-223223 frame 00019 | the rank-up rule requires the Close box inside its region; `live/handlers.py` waits while the popup is in its `reward_popup` stage |
| A forced replacement screen after a pivot move (U-turn, Volt Switch) looks like a faint replacement | live runs, Oct 2026 | `live/turn_sync.leaving_slot` checks on the party screen whether our active member is still alive; `live/request.forced_offer` then marks the request `forced: "pivot"` |
| "8/8" PP is read as "18" or "/8" | run 131901 frame 00032 | `phone/fields.py` re-reads the PP from its own crop when the first read gave none |
| Screencap returns a buffer whose length does not match its header | run 20261029-201042 | `phone/adb.py` raises `Transient`; `phone/retry.py` retries |
| "Communication was interrupted" returns the game to the title screen mid-battle | run 20261029-193300 | `handlers.on_interruption` walks back to the lobby; `live/reconnect.py` rides out a disconnect while a battle is in progress (see below) |
| An opponent forfeit during the move peek: the forfeit message shows only to the peek and the next poll is already the post-battle options screen, so the battle closed as `unknown` | run 20261103-151749 battle 1 | `TurnPipeline.peek` records `opponent_forfeit` when it sees the message |
| The battle log never shows a critical hit: no "critical hit" row in the recorded battles (`docs/battle-log.md`), and the Turn 9 frame of the Play Rough that fainted Gholdengo has none | run 20261102-104355 battle 8 (visit10-0) | not readable; no event carries a crit (`events/live_check.py` rejects the field) |
| A miss is logged only as "The opposing Glaceon avoided the attack!" (never "...'s attack missed!") | runs 20261102-160125 battle 1 turn 4, 20261102-104355 battle 7 turn 15 | sent as a `miss` event naming the Pokémon that avoided it (`events/live.py`); the attacker is not named by the row and is not added |
| "The Pokémon was hit N times!" names no one | run 20261102-220602 battle 20 turn 1 | sent as a `hits` event with the count only; Champions Gym links it to the move before it |
| The climb toward a turn's start ran out of budget and its last capture, never compared with a second one, was discarded with the rows only it showed (our Mega Evolution, Drought, the sun) | runs 20261103-015542 battle 14 visit 2, 20261102-220602 battle 20 visit 2 | `LogCollector.collect` keeps it as `unsettled`; `phone/log_views.usable_views` merges it only when its rows join the settled views with a verified overlap |
| The MOVE TIME warning over one capture also discarded the capture before it, so a climb near the timer lost settled frames | same runs | `LogCollector._panel_ok` rejects only the covered capture |
| "The opposing Snorlax protected itself!" | recorded phone battles, Nov 2026 | sent as a `protect` event |
| The party screen's order is not the tracked one: after a replacement from row 2 the game listed Cinderace, Gholdengo, Golisopod where the record (row swap) had Cinderace, Golisopod, Gholdengo, and a replacement screen can list the outgoing member above the fainted active (frame 00373). A switch row from the tracked order tapped a fainted member, its summary opened without Switch In, the move timer ran out and the locked move was played three turns in a row | run 20261115-034548 battle 7 turns 12-14 (frames 00383, 00389, 00393) | `live/party_switch.py` opens the party, reads it and taps the target by name, tries once more if the party is still up, else backs out; `live/action_check.py` compares each turn's log with the choice (`action_mismatch`) |
| A communication error during team selection rides out back onto the same selection screen; it was taken for a new battle (the first ended `unknown`), the lineup taps came after the 2 s left and the game chose the first three roster rows itself, while the decisions used the trio requested | run 20261115-034548 battles 31-32 (frames 02031-02037) | `handlers.on_team_select` repairs the lineup when the preview is the same match; the battle record is marked `team_auto`; `live/team_check.py` adopts the team brought (`team_mismatch`, `team_played_names`) |

## Known limits of the battle-log reader

The screen reader turns the in-game log panel into rows (`phone/log_panel.py`, `live/log_collector.py`,
`phone/log_views.py`, `phone/log_merge.py`), `battle/log_rows.py` parses each row, and `events/live.py` sends it
to Champions Gym as an event. The logs are not exact: each item names a recorded example; the frames are under
`battle-N/log-frames/`, the OCR'd rows in `battle-N/log.jsonl`. Update this table whenever a limit is found or
lifted.

| # | limit | cause (from the frames) | status |
|---|---|---|---|
| 1 | Critical hits are never known | the game's log panel has no critical-hit line: none in the recorded rows, none in the Turn 9 frame of `20261102-104355-run/battle-8` (Play Rough faints Gholdengo from more HP than a non-critical hit can take) | not readable from the log; a turn with a critical hit shows more damage than its move can deal |
| 2 | A miss names only the Pokémon that avoided the attack | the game logs a miss only as "X avoided the attack!" | sent as `miss`; the attacker is not in the row |
| 3 | Our own Mega Evolution can be missing from a turn | the climb toward the turn's start ran out of budget before reaching its rows, or a MOVE TIME warning covered them | the collector keeps and merges the last capture (table above); rows still not captured are reported to Champions Gym as a `gap` event. Nothing is inferred from a row's sprite |
| 4 | Own-team stats: Rotom's Attack implied by its logged damage (76) differs from the team export (63) | not found | open |

The row parser's cases are tested in `tests/battle/test_log_rows.py`, the events in `tests/events/test_live.py`,
and the collector on frames of `20261103-015542-run/battle-14` visit 2 in `tests/perception/test_log_frames.py`
(`tests/fixtures/log_frames/`; the frames are local-only, their cached reads in `views.json` are tracked).

## OCR reads a final "!" as "l" (run 20261102-104355, battles 3, 4, 9)
"Charizard was damaged by the recoill" matched no template. `log_rows.parse_row` retries a row ending in "l"
both with "!" appended and with the "l" replaced by "!".

## A changed ability is not in the species' pool (run 20261102-104355, battle 9)
"Ability Changed: Alakazam: Trace → Rough Skin" was refused as ambiguous because Rough Skin is not an Alakazam
ability. `events/live_ids.ability_id` falls back to matching against all abilities when the species' own pool
fails; the row becomes an `ability_change` event.

## Phone screen off looks like a transition (run 20261102-104355, after battle 9)
A locked phone screen (black with "Fully charged") classifies as `transition: black frame`; the session restarted
after 150 s and kept seeing black. `device.stay_awake` keeps the screen on while the run is connected. There is
no check that wakes a screen that is already off.

## A run that ends abruptly leaves `pcsa.exe` held (2 Nov 2026)
A `pcsa run` from the repo that is still running holds `pcsa.exe`, which makes `uv run` fail with "Access is
denied". `scripts/run_agent.ps1` stops leftover `pcsa run` processes from this repo before it starts.

## "Charizard: max HP mismatch" fallback (run 20261102-175202)
The summary screen's stat-point boxes were not read for one member, so its export carried no spread and gave
153 HP where the sheet said 185. `live/team_read.from_sheet` solves the points (and nature) from the exact stats
when the points are unread (`battle/stats.solve_points`, `nature_name`).

## The log never names a Choice Scarf (phone session, 2 Nov 2026)
An opponent's scarf is never written in the log. Champions Gym's Battle holds only what the log has shown, so an
unrevealed scarf is not known and is not inferred from the order of moving.

## Our own Choice lock (run 20261102-175202, battle 2)
After Hydreigon (Choice Scarf) used Dark Pulse, the move list still offered all four moves; Draco Meteor was
chosen, the tap could not take, the turn was retried and the game sent Dark Pulse. Champions Gym decides what is
legal from the Battle. On the phone side `battle/hints.py` works out the lock from the log as advice for the
request (`choice_locked`: our member's last move since it was sent out, while it holds a Choice Scarf by the
team sheet and the log has not shown the item removed), alongside Encore, Taunt and trapping. It looks at the
Scarf only, not at a Choice Band or Choice Specs.

## A disconnect mid-battle stalled the run or restarted the session (2 Nov 2026)
When the game lost its connection during a battle, the phone showed "Communication was interrupted", a
dialog, a black or loading frame, or a frame no rule recognised, and the session waited until the screen
watchdog restarted it. Tapping the screen usually brings the battle back. `live/reconnect.py` rides it
out while the run's battle record says a battle is in progress: on an unknown, communication-error or dialog
screen at once, and on a transition frame once it has lasted `navigator.disconnect_blank_s`. Each round taps
one thing, an OCR-confirmed OK / Close / Continue / Cancel / Not now when the screen offers exactly one and no
unsafe word shares it, otherwise the `screen_centre` region; waits `navigator.disconnect_settle_s`; observes
again. A tap with forfeit, quit, edit team, rate/rating, submit, review, purchase, buy, double, shop or delete
within reach is skipped and the round only waits (`taps.BLIND_FORBIDDEN_TEXT`, read at OCR confidence 0.5). The
`screen_centre` tap is allowed only on the unknown, comm-error, dialog and transition screens. The ride-out ends
on any battle screen (the turn pipeline takes over), on a post-battle screen (the usual handlers finalize), or
on the lobby, title or a menu (the battle is finalized as `unknown`, as the comm-error path does). Unknown
frames are saved under `unknown/`. Past `navigator.disconnect_max_taps` rounds or `navigator.disconnect_max_s`
seconds it raises Recoverable and the supervisor restarts the session.
