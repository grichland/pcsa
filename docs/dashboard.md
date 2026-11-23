# The dashboard

`pcsa watch` serves a read-only page about one run on `http://127.0.0.1:8765/`, while the run plays or after it
has ended:

```
uv run pcsa watch                                             # the newest run under output\runs
uv run pcsa watch --run output\runs\<stamp>-run --port 8770   # a given run on another port; --no-browser to not open it
```

The page only reads the run folder, never touches the phone, and listens on this computer only. It is bound to
the run it was started on.

## Left: the frame and the decision

The newest saved phone frame, and under it the neural player's decision matrix. The opponent's actions run across
the top (their revealed moves, one column for a move not yet seen, their switches, each with the probability the
network gives it), our actions down the side, and each cell is shaded and labelled with the score of our action
against theirs. A bar per row shows our policy. A row's score is its cells weighted by their probabilities, which
is how the network ranks our actions; a "for them" total per column shows how each of their actions turns out for
them.

The network scores a move and the same move with Mega Evolution as separate actions of theirs; the page folds them
into one column and shows the Mega share as "They Mega". Two what-if controls recompute the scores on the page
only: "All / Top 5 / 3 / 2 / 1" keeps only their likeliest actions, and "trust in the prediction" reweights their
probabilities at another temperature.

The action played is written large with the win chance (`(v + 1) / 2` from the value, calibrated when
`cg.value_calibration` is on) and the decision time. A
decision made by a fallback shows its reason instead of the matrix. When a choice rule changed the pick, a banner
names the rule and the network's own pick; a searched decision adds the search's figures, a column with the
search's share per action and a switch that redraws the cells with the search's values.

## Right: the run and the battle

- The record, the win rate over the run and over the last ten battles, the rating with one square per battle
  (click one to look at that battle) and the win-chance line of the current battle.
- The two teams with the field conditions between them. HP is a percentage on both sides, as Champions Gym holds
  it.
- The battle log by turn, saved by the run after every visit to the log.

Warnings open from the top bar.

## The network's belief

On an opposing card, an item or ability the battle has not revealed shows the network's top guess dimmed, with
its probability (the top three in the tooltip), and believed moves fill the empty move slots as dashed chips.
Members seen at team preview but not yet in battle show "brought N%", and "lead N%" during the preview. A revealed
fact always replaces the belief. The belief heads predict no nature or stat spread and do not change the
network's scores; read them as a view into the network, not as scouting.

Runs whose records carry no battle snapshot (`notes.cg.view`) show their record, rating, log and decisions; the
teams, field and matrix panels say why they are empty.
