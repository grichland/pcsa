# Live quirks

One entry per hard-won fact about how the game behaves on the phone. Each names the symptom, the run that
showed it, and the config key or code branch that handles it. Add here; do not scatter run IDs in comments.

| Symptom | Seen in | Handled by |
|---|---|---|
| A focused move row submits on the next tap; an unfocused one needs two (first focuses, second submits) | stage-8 probes, Oct 2026 | `taps.focused_move_taps`, `taps.unfocused_move_taps` |
| The lobby is dimmed for a moment after Continue Battling and misreads as a transition | run 20261025-214430 | lobby rule tolerates low brightness when Begin Matchmaking is read |
| The move timer's first digit is sometimes dropped by OCR ("5" for "15") | live runs, Oct 2026 | `MIN_NEW_TURN_MOVE_TIME` guard before trusting a short timer |
| OCR splits a button label into words ("Continue" / "Battling") | run 20261025-214430 | labels are joined inside each button region before matching |
| The Rank-Up Reward popup slides in; its Close control is offered only once settled | probe1-20261023-223223 frame 00019 | rank-up rule requires the Close box inside its region |
| A forced replacement screen after a pivot move (U-turn, Volt Switch) looks like a faint replacement | live runs, Oct 2026 | the forced handler checks whether our last action this turn was a pivot move (`PIVOT_MOVES`, `live/turn.py`) |
| "8/8" PP is read as "18" or "/8" | run 131901 frame 00032 | PP re-read with the line recognizer when the first read fails |
| Screencap returns a buffer whose length does not match its header | run 20261029-201042 | device retry + reconnect |
| "Communication was interrupted" returns the game to the title screen mid-battle | run 20261029-193300 | comm_error -> title -> lobby edge; pending battle resumed or finalized |
| Before the first switch, poke-env keys the opponent team by species id ("taurospaldeablaze"), not "p2: Tauros" | run 20261101-193426 (team preview tracking failed) | `state.build` counts only "side: name" keys as sent out |
| poke-env takes the first `\|player\|` line it sees as our side when neither name is ours, so a mirrored p2 channel replays with the roles swapped | sim arena, 2 Nov 2026 | `mirror.as_p1` renames the player lines to me/opponent and puts p1 first |
| The Showdown server logs a `better-sqlite3` bindings CRASH at startup (`npm ci --ignore-scripts` skips the native build); battles are unaffected | sim arena, 2 Nov 2026 | ignored; the server's private-message store is never used |
| An opponent forfeit during the move peek: the forfeit message shows only to the peek and the next poll is already the post-battle options screen, so the battle closed as `unknown` | run 20261103-151749 battle 1 | `TurnPipeline.peek` records `opponent_forfeit` when it sees the message |
| The battle log never shows a critical hit: no "critical hit" row in 10,724 OCR'd rows of 222 recorded battles, and the Turn 9 frame of the Play Rough that fainted Gholdengo has none | run 20261102-104355 battle 8 (visit10-0) | not readable: the protocol carries no `-crit` (defect 1a below) |
| A miss is logged only as "The opposing Glaceon avoided the attack!" (never "...'s attack missed!"); it was counted but not written, so the protocol showed a Shadow Ball that hit and a Leech Seed that seeded | runs 20261102-160125 battle 1 turn 4, 20261102-104355 battle 7 turn 15 | `translate_events._miss` writes `\|-miss\|<latest mover>\|<target>` (informational when the turn's move row was not captured) |
| "The Pokémon was hit N times!" names no one | run 20261102-220602 battle 20 turn 1 | `translate_events._hitcount` writes `\|-hitcount\|` on the latest move's target |
| The climb toward a turn's start ran out of budget and its last capture, never compared with a second one, was discarded with the rows only it showed (our Mega Evolution, Drought, the sun) | runs 20261103-015542 battle 14 visit 2, 20261102-220602 battle 20 visit 2 | `LogCollector.collect` keeps it as `unsettled`; `log_merge.usable_views` merges it only when its rows join the settled views with a verified overlap |
| The MOVE TIME warning over one capture also discarded the capture before it, so a climb near the timer lost settled frames | same runs | `LogCollector._panel_ok` rejects only the covered capture; the next uncovered one is compared with the capture before the warning |
| Our Mega Evolution with its rows uncaptured was written only as a form change from the row sprite, after our move | same runs | `translate_megas.mega_from_sprite` writes `-item`, `detailschange`, `-mega` before the first row whose sprite shows the Mega form |
| "The opposing Snorlax protected itself!" is not written as `-activate`: poke-env would keep a PROTECT effect up to a mid-turn forced decision | recorded phone battles, Nov 2026 | counted as informational |

## Known defects of the battle-log reader

The screen reader turns the in-game log panel into protocol rows (`phone/log_panel.py`,
`live/log_collector.py`, `phone/log_merge.py`, then `battle/` translation). The logs are not exact: each item
names a recorded example; the frames are under `battle-N/log-frames/`, the OCR'd rows in `battle-N/log.jsonl`.
Update this table whenever a defect is found or fixed.

| # | defect | root cause (from the frames) | status |
|---|---|---|---|
| 1a | Critical hits never written | the game's log panel has no critical-hit line: none in 10,724 OCR'd rows, none in the Turn 9 frame of `20261102-104355-run/battle-8` (Play Rough faints Gholdengo from more HP than a non-critical hit can take) | not readable from the log; a turn with a critical hit shows more damage than its move can deal |
| 1b | Misses never written | the game logs a miss only as "X avoided the attack!"; the row was read and counted but not written | fixed 4 Nov 2026: `\|-miss\|attacker\|target` |
| 2 | Whole lines missed | both examples are misses (1b), not dropped lines: `20261102-160125-run/battle-1` turn 4 "The opposing Glaceon avoided the attack!", `20261102-104355-run/battle-7` turn 15 "Charizard avoided the attack!" | fixed with 1b |
| 3 | Wrong HP percentage read | not a misread: the frame and the row both say Volcarona 25% -> 75%. The build estimate of the time held a Sitrus Berry, which would have been eaten at 25% (then Morning Sun reaches 100%); the log proves Volcarona has none (it fell to 25% on turn 1 and ate nothing) | not a reader defect: the estimate was wrong, the log right. Live play no longer estimates hidden builds, so nothing assumes an item the log has not shown |
| 4 | Our own Mega Evolution only as a form change after our move | the climb toward the turn's start ran out of budget; the last capture, the only one showing our Mega rows (and Drought, the sun), was never confirmed by a second capture and was discarded, and a MOVE TIME warning over one capture had discarded the one before it | fixed 4 Nov 2026: the last capture is kept and merged when its rows join the others; a Mega seen only in a row sprite is written as `-item`/`detailschange`/`-mega` before our move. Recordings from before the fix still carry the late form change |
| 5 | Protect/Detect activation and multi-hit counts never written | rows read, not written | multi-hit counts written (`-hitcount`); Protect's `-activate` deliberately not (see the table above) |
| 6 | Own-team stats: Rotom's Attack implied by its logged damage (76) differs from the team export (63); Charizard's max HP mismatch | | Rotom open; Charizard: see "Charizard: max HP mismatch" below |

Each fix has a row in the table above and a test: `tests/battle/test_translate_log_gaps.py` on the OCR'd
rows, `tests/perception/test_log_frames.py` on frames of `20261103-015542-run/battle-14` visit 2
(`tests/fixtures/log_frames/`; the frames are local-only, their cached reads in `views.json` are tracked). The
recorded protocols keep the old rows; the fixes apply to new runs, and to recorded battles only when their log
is translated again.

## OCR reads a final "!" as "l" (run 20261102-104355, battles 3, 4, 9)
"Charizard was damaged by the recoill" matched no template, which dropped 19 turns to the `tracked` level.
`log_rows.parse_row` now retries a row ending in "l" both with "!" appended and with the "l" replaced by "!".

## A changed ability is not in the species' pool (run 20261102-104355, battle 9)
"Ability Changed: Alakazam: Trace → Rough Skin" was refused as ambiguous because Rough Skin is not an Alakazam
ability. `translate_events._ability` falls back to the all-abilities match (floor 0.9) when the species pool fails.

## Phone screen off looks like a transition (run 20261102-104355, after battle 9)
A locked phone screen (black with "Fully charged") classifies as `transition: black frame`; the session restarted
after 150 s and kept seeing black. Harmless when the operator ended the session; a wake-on-black check is still open.

## A run that ends abruptly leaves its dashboard behind (2 Nov 2026)
Closing the launcher window kills `pcsa run` but not the `pcsa watch` window (it keeps port 8765), and
anything from the repo still running holds `pcsa.exe`, which makes `uv run` fail with "Access is denied".
`scripts/run_agent.ps1` now stops leftover `pcsa run` / `pcsa watch` processes from this repo, launches the
venv's `pcsa.exe` directly, and moves the dashboard to the next free port if 8765 is still taken by something
else.

## "Charizard: max HP mismatch" fallback (run 20261102-175202)
The summary screen's stat-point boxes were not read for one member, so its export carried no spread and gave
153 HP where the sheet said 185.
`live/team_read.from_sheet` now solves the points (and nature) from the exact stats when the points are unread
(`battle/stats.solve_points`, `nature_name`). Takes effect on the next run's team read.

## The log never names a Choice Scarf (phone session, 2 Nov 2026)
An opponent's scarf is invisible to the tracker, and the heuristic compared base speeds with no items, stages
or paralysis. `stats.effective_speed` now gives the heuristic a known Choice Scarf, the speed stage and
paralysis; an unrevealed scarf is still not inferred from move order.

## Our own Choice lock was not in the request (run 20261102-175202, battle 2)
After Hydreigon (Choice Scarf) used Dark Pulse, the move list the players saw still offered all four moves;
the player chose Draco Meteor, the tap could not take, the turn was retried and the game sent Dark Pulse.
`battle/state.choice_lock` now reads our member's last move since it was sent out from the protocol and marks
every other move disabled while a Choice item is held (the sheet's item when the tracked item is unknown).

## A disconnect mid-battle stalled the run or restarted the session (2 Nov 2026)
When the game lost its connection during a battle, the phone showed "Communication was interrupted", a
dialog, a black or loading frame, or a frame no rule recognised, and the session waited until the screen
watchdog restarted it. Tapping the screen usually brings the battle back. `live/reconnect.py` now rides it
out while the run's battle record says a battle is in progress: on an unknown, communication-error or dialog
screen at once, and on a transition frame once it has lasted `navigator.disconnect_blank_s`. Each round taps
one thing, an OCR-confirmed OK / Close / Continue / Cancel / Not now when the screen offers exactly one and no
unsafe word shares it, otherwise the `screen_centre` region; waits `navigator.disconnect_settle_s`; observes
again. A tap with forfeit, quit, edit team, rate/rating, shop, double or similar text within 60 px is skipped
and the round only waits (`taps.BLIND_FORBIDDEN_TEXT`, read at OCR confidence 0.5). The `screen_centre` tap is
allowed only on the unknown, comm-error, dialog and transition screens. The ride-out ends on any battle screen
(the turn pipeline takes over), on a post-battle screen (the usual handlers finalize), or on the lobby, title
or a menu (the battle is finalized as `unknown`, as the comm-error path does). Unknown frames are saved under
`unknown/` as before; events are `disconnect_start`, `disconnect_tap`, `disconnect_tap_skipped`,
`disconnect_recovered` and `disconnect_exhausted`. Past `navigator.disconnect_max_taps` rounds or
`navigator.disconnect_max_s` seconds it raises Recoverable and the supervisor restarts the session.
