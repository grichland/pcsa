"""Measuring the sprite network against the retired matcher, in numpy (no torch needed).

Both matchers run through the same matching flow (sprites.preview_matches / log_decision / pick) on
held-out runs. The battle log's acceptance thresholds are calibrated on a validation slice of the training
runs: the most matches whose wrong-species count does not exceed the retired matcher's on the same slice.
A recolour stress test stands in for shiny Pokémon: the same crops with hue rotated, lightness inverted,
part of the sprite recoloured, or all colour removed.
"""

from __future__ import annotations

import json
import math
from typing import cast

import numpy as np

from pcsa.common.clock import SystemClock
from pcsa.phone import sprite_net, sprites
from pcsa.sprite_training import sprite_baseline as sb
from pcsa.sprite_training import sprite_data as sd

CENTRE_MIN_CROPS = 3
RECOLOURS = ("original", "hue rotated", "lightness inverted", "part recoloured", "grey")
STRESS_CROPS = 600
CLOCK = SystemClock()


def in_memory_net(
    weights, sources, data: sd.Dataset | None = None, rows: np.ndarray | None = None
) -> sprite_net.SpriteNet:
    refs = {}
    for domain, source in sources.items():
        vectors = sprite_net.forward(weights, *sprite_net.encode(source.masks, source.labs, source.aspects))
        names, bases = list(source.names), list(source.base_names)
        if data is not None and rows is not None:
            extra_names, extra_bases, extra_vectors = real_centres(
                weights, source, data, rows, sd.DOMAIN_CODE[domain]
            )
            names, bases = names + extra_names, bases + extra_bases
            vectors = np.concatenate([vectors, extra_vectors]) if extra_names else vectors
        refs[domain] = sprite_net.References(names, dict(zip(names, bases, strict=True)), vectors)
    return sprite_net.SpriteNet(weights, refs, 0.0, 0.0)


def real_centres(weights, source: sb.SourceIndex, data: sd.Dataset, rows: np.ndarray, domain: int):
    """One extra view per form with enough form-labelled real crops: the mean of their vectors.

    Reference art and in-game crops of the same form can sit apart; the centre of the real crops pulls
    them back together for every form the runs have shown. Forms never seen keep their reference view.
    """
    name_of: dict[str, tuple[str, str]] = {}
    for name, base in zip(source.names, source.base_names, strict=True):
        name_of.setdefault(sd.form_id(name), (name, base))
    groups: dict[str, list[tuple[int, int]]] = {}
    for i in rows:
        if data.domain[i] == domain and data.species[i] and len(data.allowed[i]) == 1:
            form = data.allowed[i][0]
            if form in name_of:
                groups.setdefault(form, []).extend((int(i), v) for v in (0, 1) if data.variant[i, v] == 0)
    names, bases, vectors = [], [], []
    for form, picks in sorted(groups.items()):
        if len(picks) < CENTRE_MIN_CROPS:
            continue
        idx, var = np.array([p[0] for p in picks]), np.array([p[1] for p in picks])
        v = sprite_net.forward(
            weights, *sprite_net.encode(data.masks[idx, var], data.labs[idx, var], data.aspects[idx, var])
        )
        centre = v.mean(axis=0)
        names.append(name_of[form][0])
        bases.append(name_of[form][1])
        vectors.append(centre / np.linalg.norm(centre))
    return names, bases, (np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, sprite_net.EMBED)))


def _feature(data: sd.Dataset, i: int, v: int) -> sprites.Feature | str:
    code = int(data.variant[i, v])
    if code:
        return "none" if code == 1 else "clipped"
    return sprites.Feature(
        data.masks[i, v].astype(np.float32), data.labs[i, v].astype(np.float32), float(data.aspects[i, v])
    )


def _scores(lib: sprites.Scorer, variants: list) -> list[np.ndarray | None]:
    feats = [v for v in variants if isinstance(v, sprites.Feature)]
    rows = iter(lib.score(feats)) if feats else iter(())
    return [next(rows) if isinstance(v, sprites.Feature) else None for v in variants]


def log_outcomes(lib, data, rows, classes, min_score, min_margin) -> list[dict]:
    rows = [i for i in rows if data.domain[i] == 1 and data.species[i]]
    raw_v = [_feature(data, i, 0) for i in rows]
    clean_v = [_feature(data, i, 1) for i in rows]
    raw_s, clean_s = _scores(lib, raw_v), _scores(lib, clean_v)
    out = []
    for k, i in enumerate(rows):
        box = data.recorded[i].get("box") or []
        result = sprites.log_decision(raw_v[k], raw_s[k], lib, box, min_score, min_margin)
        if sprites.needs_cleaning(result):
            result = sprites.pick(
                result, sprites.log_decision(clean_v[k], clean_s[k], lib, box, min_score, min_margin)
            )
        raw_k = raw_s[k]
        top1 = sprites.top(lib, raw_k, 1)[0][0] if raw_k is not None else None
        out.append(
            {
                "row": int(i),
                "species": data.species[i],
                "status": result["status"],
                "matched": classes.base(sd.form_id(result["name"])) if result["status"] == "matched" else None,
                "top1": classes.base(sd.form_id(top1)) if top1 else None,
                "form": sd.form_id(result["name"]) if result["status"] == "matched" else None,
                "allowed": data.allowed[i],
            }
        )
    return out


def summarize_log(outcomes: list[dict]) -> dict:
    n = len(outcomes)
    matched = [o for o in outcomes if o["status"] == "matched"]
    correct = sum(o["matched"] == o["species"] for o in matched)
    with_top = [o for o in outcomes if o["top1"]]
    single = [o for o in matched if len(o["allowed"]) == 1 and o["matched"] == o["species"]]
    return {
        "rows": n,
        "matched_correct": correct,
        "matched_wrong": len(matched) - correct,
        "unknown": sum(o["status"] == "unknown" for o in outcomes),
        "none": sum(o["status"] == "none" for o in outcomes),
        "top1_species_accuracy": round(sum(o["top1"] == o["species"] for o in with_top) / max(1, len(with_top)), 4),
        "form_agreement": round(sum(o["form"] in o["allowed"] for o in single) / max(1, len(single)), 4),
    }


def preview_summary(lib, data, rows, classes) -> dict:
    by_battle: dict[str, list[int]] = {}
    for i in rows:
        if data.domain[i] == 0:
            by_battle.setdefault(data.battle[i], []).append(int(i))
    sent_total = found = labelled = labelled_right = 0
    for battle, slots in by_battle.items():
        slots.sort(key=lambda i: int(data.slot[i]))
        if len(slots) != len(sprites.OPPONENT_BOXES):
            continue
        feats = [_feature(data, i, 0) for i in slots]
        matches = sprites.preview_matches([f if isinstance(f, sprites.Feature) else None for f in feats], lib)
        tops = [classes.base(sd.form_id(m.top[0][0])) if m.top else "" for m in matches]
        sent = set(data.battles.get(battle, {}))
        sent_total += len(sent)
        found += len(sent & set(tops))
        for i, t in zip(slots, tops, strict=True):
            if data.species[i]:
                labelled += 1
                labelled_right += t == data.species[i]
    return {
        "battles": len(by_battle),
        "sent_out_species": sent_total,
        "sent_out_found_in_top1": found,
        "sent_out_recall": round(found / max(1, sent_total), 4),
        "confirmed_cards": labelled,
        "confirmed_cards_right": labelled_right,
    }


def calibrate(net_out: list[dict], base_out: list[dict], lib, data, classes) -> tuple[float, float, dict]:
    """Thresholds with the most correct matches whose wrong count does not exceed the baseline's."""
    budget = summarize_log(base_out)["matched_wrong"]
    rows = [o["row"] for o in net_out]
    raw = _scores(lib, [_feature(data, i, 0) for i in rows])
    clean = _scores(lib, [_feature(data, i, 1) for i in rows])

    def best2(s):
        if s is None:
            return (-9.0, 0.0, "")
        t = sprites.top(lib, s, 2)
        return (t[0][1], t[0][1] - (t[1][1] if len(t) > 1 else 0), classes.base(sd.form_id(t[0][0])))

    rs, cs = [best2(s) for s in raw], [best2(s) for s in clean]
    truth = [o["species"] for o in net_out]
    raw_none = [_feature(data, i, 0) == "none" for i in rows]
    clean_none = [_feature(data, i, 1) == "none" for i in rows]
    best = (-1, 0, 0.0, 0.0)
    for min_score in np.arange(0.30, 0.96, 0.01):
        for min_margin in np.arange(0.0, 0.31, 0.01):
            right = wrong = 0
            for k in range(len(rows)):
                if raw_none[k]:
                    continue
                s, m, name = rs[k]
                if not (s >= min_score and m >= min_margin):
                    cs_k = cs[k]
                    if clean_none[k] or not (cs_k[0] >= min_score and cs_k[1] >= min_margin):
                        continue
                    name = cs_k[2]
                right += name == truth[k]
                wrong += name != truth[k]
            key = (right if wrong <= budget else -1, -wrong, float(min_margin), float(min_score))
            if key > (best[0], -best[1], best[2], best[3]):
                best = (right, wrong, float(min_margin), float(min_score))
    right, wrong, min_margin, min_score = best
    return round(min_score, 2), round(min_margin, 2), {"val_right": right, "val_wrong": wrong, "val_budget": budget}


def recolour(f: sprites.Feature, kind: str, rng: np.random.Generator) -> sprites.Feature:
    """Shiny stand-ins: the same silhouette with its palette changed in one of RECOLOURS' ways."""
    lab, inside = f.lab.copy(), f.mask > 0.05  # noqa: PLR2004 (any coverage at all)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    if kind in ("hue rotated", "part recoloured"):
        turn = rng.uniform(math.pi / 3, math.pi) * rng.choice([-1, 1])
        chosen = inside
        if kind == "part recoloured":  # one side of the sprite's own lightness or colour split
            key = lab[..., 0] if rng.random() < 0.5 else a  # noqa: PLR2004
            chosen = inside & (key > np.median(key[inside]))
            lab[..., 0] = np.where(chosen, np.clip(lab[..., 0] * rng.uniform(0.6, 1.4), 0, 255), lab[..., 0])
        lab[..., 1] = np.where(chosen, a * math.cos(turn) - b * math.sin(turn) + 128, lab[..., 1])
        lab[..., 2] = np.where(chosen, a * math.sin(turn) + b * math.cos(turn) + 128, lab[..., 2])
    elif kind == "lightness inverted":
        lab[..., 0] = np.where(inside, 255 - lab[..., 0], lab[..., 0])
    elif kind == "grey":
        lab[..., 1:] = np.where(inside[..., None], 128, lab[..., 1:])
    return sprites.Feature(f.mask, np.clip(lab, 0, 255).astype(np.float32), f.aspect)


def recolour_stress(lib, data: sd.Dataset, rows: np.ndarray, classes: sd.Classes) -> dict[str, float]:
    """Best-guess species accuracy on held-out log crops, as recorded and recoloured."""
    rows = np.array([i for i in rows if data.domain[i] == 1 and data.species[i] and data.variant[i, 0] == 0])
    if not len(rows):
        return {}
    rows = np.random.default_rng(0).choice(rows, min(STRESS_CROPS, len(rows)), replace=False)
    base = [cast(sprites.Feature, _feature(data, int(i), 0)) for i in rows]
    out = {}
    for kind in RECOLOURS:
        rng = np.random.default_rng(1)
        scores = lib.score(base if kind == "original" else [recolour(f, kind, rng) for f in base])
        right = sum(
            classes.base(sd.form_id(lib.names[int(np.argmax(s))])) == data.species[i]
            for s, i in zip(scores, rows, strict=True)
        )
        out[kind] = round(right / len(rows), 4)
    return out


def measure(built, sources, data, parts, classes, held, log) -> dict:  # noqa: PLR0913 (the network, its references, the crops and their split, the held-out species, the log)
    """Calibrate the battle-log thresholds on validation, then compare both matchers on val and test."""
    nets = {d: sprites.NetScorer(built, d) for d in sb.SOURCES}
    olds = {d: sb.BaselineScorer(sources[d], d) for d in sb.SOURCES}
    log("calibrating the battle-log thresholds on the validation runs")
    base_val = log_outcomes(olds["log"], data, parts["val"], classes, sb.LOG_MIN_SCORE, sb.LOG_MIN_MARGIN)
    net_val = log_outcomes(nets["log"], data, parts["val"], classes, 0.0, 0.0)
    min_score, min_margin, cal = calibrate(net_val, base_val, nets["log"], data, classes)
    log(f"log thresholds: min_score {min_score}, min_margin {min_margin} ({cal})")

    report: dict = {"calibration": cal, "thresholds": [min_score, min_margin], "held_species": sorted(held)}
    for part in ("val", "test"):
        rows = parts[part]
        net_log = log_outcomes(nets["log"], data, rows, classes, min_score, min_margin)
        old_log = log_outcomes(olds["log"], data, rows, classes, sb.LOG_MIN_SCORE, sb.LOG_MIN_MARGIN)
        report[part] = {
            "log": {"network": summarize_log(net_log), "retired": summarize_log(old_log)},
            "preview": {
                "network": preview_summary(nets["preview"], data, rows, classes),
                "retired": preview_summary(olds["preview"], data, rows, classes),
            },
        }
        if held:
            keep = [o for o in net_log if o["species"] in held]
            keep_old = [o for o in old_log if o["species"] in held]
            report[part]["log_unseen_species"] = {"network": summarize_log(keep), "retired": summarize_log(keep_old)}
        log(f"{part}: {json.dumps(report[part])}")
    stress_rows = parts["test"] if len(parts["test"]) else parts["val"]
    report["recolour_stress"] = {
        "network": recolour_stress(nets["log"], data, stress_rows, classes),
        "retired": recolour_stress(olds["log"], data, stress_rows, classes),
    }
    log(f"recolour stress (best-guess species accuracy): {json.dumps(report['recolour_stress'])}")
    started = CLOCK.now()
    some = [f for f in (_feature(data, int(i), 0) for i in parts["test"][:60]) if isinstance(f, sprites.Feature)]
    for f in some:
        nets["log"].score([f])
    report["ms_per_crop"] = round(1000 * (CLOCK.now() - started) / max(1, len(some)), 2)
    return report
