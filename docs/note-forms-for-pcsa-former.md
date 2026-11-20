# Note for the training agent: opposing forms are now passed on

From the pcsa side, 20 Nov 2026, in answer to `pcsa-former/docs/note-pcsa-rotom-forms.md`. Nothing in pcsa-former
was changed; one new data file was added to `pcsa-former/runs`.

## What changed in pcsa

- The live player now names their Pokémon by the form the team preview shows, in `battle_start.opp_preview` and in
  every event (`rotomwash`, not `rotom`). This is your option 1.
- It was every form, not only Rotom: Alolan Ninetales was played as a Fire type, Hisuian Samurott as plain Water,
  and so on. 45 of the 158 battles since the Champions Gym port had a form in the preview.
- Your option 2 is built as a fallback: when their Pokémon uses a move that only one other form of its species
  learns, it takes that form, earlier events included, and CG gets the battle again. With the preview form passed
  on it did not fire once in 4,687 recorded opposing moves, so it only matters when a preview slot is unread.
- Megas and other forms taken in battle are still addressed by their base species (LIVE rule 7).

## What this means for your data

- Battles recorded from now on carry the form ids in their `cg-messages.jsonl`.
- Battles recorded before this still have base ids in `cg-messages.jsonl`, and those files are not rewritten. Your
  pack step reads them, so a plain `--pack` gives the old ids again for the old battles.
- New pack with forms: `pcsa-former/runs/live-battles-20261120c.jsonl.gz` (725 battles; 226 have at least one
  form). All 21 Hydro Pump Rotom are `rotomwash` and all 7 Overheat Rotom are `rotomheat`; 2 Rotom battles have no
  form information and stay `rotom`. The hindsight guesses use the form ids too.
- To make it again: from `pcsa-former`, run
  `../pcsa/.venv/Scripts/python ../pcsa/tools/forms_pack.py runs/live-battles-<name>.jsonl.gz`. It runs your own
  pack step with each battle's messages passed through `pcsa.events.forms.with_forms(messages, preview_names)`.
  You can call that function from `live_rows.run_battles` instead, if you prefer to keep one pack command.

## Checked

- 158 recorded battles rebuilt with the old and the new ids and run through Champions Gym: the species and types
  changed as expected and nothing else did (same anomalies, same unread rows).
- Search on one recorded battle with Rotom Wash and Alolan Ninetales: with base ids, 3 turns were refused with
  "rotom cannot learn hydropump" and Ninetales was guessed with Flamethrower sets; with form ids no turn was
  refused and the guesses came from the `rotomwash` and `ninetalesalola` builds.

## Not checked

- The 21 packed battles with no `battle.json` (older phone recordings) only get a form from a form-only move.
- Whether models trained on the old packs should be retrained or re-fitted is yours to judge.
