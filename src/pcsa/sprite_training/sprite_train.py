"""`pcsa train sprites`: train the sprite network, measure it against the retired matcher, export it.

Training mixes two kinds of example in every batch: the reference views (sprite_baseline's local source
indexes), heavily augmented (shifts, rotation, scale, colour, palette swaps and greying for shiny Pokémon, erosion
and dilation, cut-outs, blur), and real labelled crops from recorded runs (sprite_data.py), lightly
augmented. The loss is a cosine classifier with an additive margin over every form id; a crop labelled
only by species accepts any form of it. After training the classifier is dropped: each reference view is
embedded by the exported numpy network (plus, per form the runs have shown, the centre of its real
crops), so a new sprite later needs only its view embedded, not retraining.

Measurement uses held-out runs (the newest `--test-runs`), never trained on, and runs both matchers
through the same matching flow (sprites.preview_matches / log_decision / pick). The battle log's
acceptance thresholds are calibrated on a validation slice of the training runs: the most matches whose
wrong-species count does not exceed the retired matcher's on the same slice.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zlib
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pcsa.common import paths
from pcsa.common.clock import SystemClock
from pcsa.phone import sprite_net
from pcsa.sprite_training import sprite_baseline as sb
from pcsa.sprite_training import sprite_data as sd
from pcsa.sprite_training import sprite_measure as sm

SCALE, MARGIN = 20.0, 0.2
VAL_FRACTION = 0.2
SHINY_P = 0.3  # share of reference views given a swapped palette (half that for real crops)
SHINY_INVERT_P = 0.4  # of those, the share with inverted lightness (a dark sprite turned pale)
SHINY_FADE_P = 0.2  # share of reference views with colour faded toward grey (half that for real crops)
SHINY_GREY_P = 0.5  # of those, the share faded all the way to grey
PARITY_LIMIT = 1e-3  # numpy vs torch, max absolute difference of a unit vector
CLOCK = SystemClock()
INSTALL_FILES = ("sprite_net.json", "sprite_net.npz", "preview_refs.npz", "log_refs.npz")


class Net(nn.Module):
    """sprite_net.py's architecture with batch norm, folded into the convolutions on export."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False) for _, cin, cout in sprite_net.CONVS
        )
        self.norms = nn.ModuleList(nn.BatchNorm2d(cout) for _, _, cout in sprite_net.CONVS)
        self.f1 = nn.Linear(sprite_net.CONVS[-1][2] + 1, sprite_net.HIDDEN)
        self.f2 = nn.Linear(sprite_net.HIDDEN, sprite_net.EMBED)

    def forward(self, x: torch.Tensor, log_aspect: torch.Tensor) -> torch.Tensor:
        for (name, _, _), conv, norm in zip(sprite_net.CONVS, self.convs, self.norms, strict=True):
            x = F.relu(norm(conv(x)))
            if name in sprite_net.POOL_AFTER:
                x = F.max_pool2d(x, 2)
        v = torch.cat([x.mean(dim=(2, 3)), log_aspect[:, None]], dim=1)
        return F.normalize(self.f2(F.relu(self.f1(v))), dim=1)

    def export(self) -> dict[str, np.ndarray]:
        out = {}
        convs = cast(list[nn.Conv2d], list(self.convs))
        norms = cast(list[nn.BatchNorm2d], list(self.norms))
        for (name, _, _), conv, norm in zip(sprite_net.CONVS, convs, norms, strict=True):
            mean, var = cast(torch.Tensor, norm.running_mean), cast(torch.Tensor, norm.running_var)
            scale = norm.weight / torch.sqrt(var + norm.eps)
            out[f"{name}.w"] = conv.weight * scale[:, None, None, None]
            out[f"{name}.b"] = norm.bias - mean * scale
        out["f1.w"], out["f1.b"], out["f2.w"], out["f2.b"] = (
            self.f1.weight,
            self.f1.bias,
            self.f2.weight,
            self.f2.bias,
        )
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}


# Augmentation ------------------------------------------------------------------------------------
def _shiny(light, ab, mask, strong, u, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Shiny Pokémon: the same silhouette in another palette, so colour can help a match but never outweigh
    the shape. Rotate the hue and sometimes invert the lightness; or fade the colour part or all the way to grey."""
    b, dev = len(ab), ab.device
    swap = (torch.rand(b, device=dev, generator=gen) < SHINY_P * torch.where(strong, 1.0, 0.5))[:, None, None]
    turn = u(-math.pi, math.pi)[:, None, None]
    a_ch, b_ch = ab[:, 0], ab[:, 1]
    rotated = torch.stack(
        [a_ch * torch.cos(turn) - b_ch * torch.sin(turn), a_ch * torch.sin(turn) + b_ch * torch.cos(turn)], 1
    )
    ab = torch.where(swap[:, None], rotated, ab)
    invert = swap & (torch.rand(b, device=dev, generator=gen) < SHINY_INVERT_P)[:, None, None]
    light = torch.where(invert, (mask[:, 0] - light).clamp(min=0), light)
    fade = torch.rand(b, device=dev, generator=gen) < SHINY_FADE_P * torch.where(strong, 1.0, 0.5)
    keep_colour = torch.where(fade, torch.where(u(0, 1) < SHINY_GREY_P, 0.0, u(0, 1)), 1.0)
    return light, ab * keep_colour[:, None, None, None]


def augment(x: torch.Tensor, a: torch.Tensor, strong: torch.Tensor, gen: torch.Generator):
    """x: (B, 4, 40, 40) encoded inputs; strong: (B,) bool, heavier for reference views."""
    b, dev = len(x), x.device

    def u(lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(b, device=dev, generator=gen)

    k = torch.where(strong, 1.0, 0.35)
    angle = u(-6, 6) * k * math.pi / 180
    scale = 1 + u(-0.12, 0.08) * k
    stretch = torch.exp(u(-0.08, 0.08) * k)
    sx, sy = scale * stretch, scale / stretch
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(angle) / sx, -torch.sin(angle) / sx
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(angle) / sy, torch.cos(angle) / sy
    theta[:, 0, 2], theta[:, 1, 2] = u(-0.1, 0.1) * k, u(-0.1, 0.1) * k
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    a = a + torch.log(stretch) * 2 + u(-0.05, 0.05) * k

    mask = x[:, :1]
    erode = (torch.rand(b, device=dev, generator=gen) < 0.25 * k)[:, None, None, None]
    dilate = (torch.rand(b, device=dev, generator=gen) < 0.25 * k)[:, None, None, None]
    eroded = -F.max_pool2d(-mask, 3, 1, 1)
    keep = torch.where(erode, eroded / mask.clamp_min(1e-4), torch.ones_like(mask)).clamp(0, 1)
    colour = x[:, 1:] * keep
    mask = torch.where(erode, eroded, torch.where(dilate, F.max_pool2d(mask, 3, 1, 1), mask))

    gain = u(0.8, 1.2) * k + (1 - k)
    shift = u(-12, 12) * k / 255
    light = (colour[:, 0] * gain[:, None, None] + shift[:, None, None] * mask[:, 0]).clamp(min=0)
    light = torch.minimum(light, mask[:, 0])
    ab = colour[:, 1:] + (u(-6, 6) * k / sprite_net.LAB_SPREAD)[:, None, None, None] * mask
    light, ab = _shiny(light, ab, mask, strong, u, gen)
    x = torch.cat([mask, light[:, None], ab], dim=1)

    blur = (torch.rand(b, device=dev, generator=gen) < 0.3 * k)[:, None, None, None]
    x = torch.where(blur, F.avg_pool2d(x, 3, 1, 1, count_include_pad=False), x)
    cut = torch.rand(b, device=dev, generator=gen) < 0.3 * k
    if cut.any():
        size = 6 + (torch.rand(b, device=dev, generator=gen) * 9).long()
        y0 = (torch.rand(b, device=dev, generator=gen) * (sprite_net.SIZE - size)).long()
        x0 = (torch.rand(b, device=dev, generator=gen) * (sprite_net.SIZE - size)).long()
        rows = torch.arange(sprite_net.SIZE, device=dev)
        inside_y = (rows[None] >= y0[:, None]) & (rows[None] < (y0 + size)[:, None])
        inside_x = (rows[None] >= x0[:, None]) & (rows[None] < (x0 + size)[:, None])
        box = inside_y[:, :, None] & inside_x[:, None, :] & cut[:, None, None]
        x = x * (~box)[:, None].float()
    return x, a


# Data ------------------------------------------------------------------------------------------------
def split(data: sd.Dataset, test_runs: int) -> dict[str, np.ndarray]:
    runs = sorted({b.split("/")[0] for b in data.battle})
    test = set(runs[-test_runs:]) if test_runs else set()
    names = np.array(["test" if b.split("/")[0] in test else "train" for b in data.battle], dtype=object)
    for i, b in enumerate(data.battle):
        if names[i] == "train" and zlib.crc32(b.encode()) % 1000 < VAL_FRACTION * 1000:
            names[i] = "val"
    return {s: np.flatnonzero(names == s) for s in ("train", "val", "test")}


def holdout_species(classes: sd.Classes, fraction: float, seed: int) -> set[str]:
    if fraction <= 0:
        return set()
    species = sorted(classes.forms_of)
    rng = np.random.default_rng(seed + 7)
    return set(rng.choice(species, size=max(1, round(fraction * len(species))), replace=False).tolist())


def reference_arrays(sources: dict[str, sb.SourceIndex], index: dict[str, int]):
    masks = np.concatenate([s.masks for s in sources.values()])
    labs = np.concatenate([s.labs for s in sources.values()])
    aspects = np.concatenate([s.aspects for s in sources.values()])
    labels = np.array([index[sd.form_id(n)] for s in sources.values() for n in s.names])
    return masks, labs, aspects, labels


def real_examples(data: sd.Dataset, rows: np.ndarray, index: dict[str, int], skip_species: set[str]):
    """One example per (crop, feature variant) with a label; returns arrays and per-example allowed masks."""
    picks, allowed = [], []
    for i in rows:
        if not data.species[i] or data.species[i] in skip_species:
            continue
        ids = [index[a] for a in data.allowed[i] if a in index]
        if not ids:
            continue
        for v in (0, 1):
            if data.variant[i, v] == 0:
                picks.append((i, v))
                allowed.append(ids)
    if not picks:
        return None
    rows_, variants = np.array([p[0] for p in picks]), np.array([p[1] for p in picks])
    allow = np.zeros((len(picks), len(index)), bool)
    for j, ids in enumerate(allowed):
        allow[j, ids] = True
    species = [data.species[r] for r in rows_]
    return (
        data.masks[rows_, variants].astype(np.float32),
        data.labs[rows_, variants].astype(np.float32),
        data.aspects[rows_, variants],
        allow,
        species,
    )


# Training --------------------------------------------------------------------------------------------
def _tensors(sources, data, rows, classes, skip_species, device) -> dict:  # noqa: PLR0913 (the views, the crops and their split, the classes, the held-out species, the device)
    """Reference views and real crops as device tensors, with their labels, domains and sampling weights."""
    index = {c: i for i, c in enumerate(classes.ids)}
    rm, rl, ra, rlab = reference_arrays(sources, index)
    rx, rasp = (torch.from_numpy(t).to(device) for t in sprite_net.encode(rm, rl, ra))
    rlab_t = torch.from_numpy(rlab).to(device)
    real = real_examples(data, rows, index, skip_species)
    assert real is not None, "no labelled real crops in the training runs"
    qm, ql, qa, allow, species = real
    qx, qasp = (torch.from_numpy(t).to(device) for t in sprite_net.encode(qm, ql, qa))
    counts: dict[str, int] = {}
    for s in species:
        counts[s] = counts.get(s, 0) + 1
    weight = np.array([1 / math.sqrt(counts[s]) for s in species])
    return {
        "rx": rx,
        "rasp": rasp,
        "rlab": rlab_t,
        "qx": qx,
        "qasp": qasp,
        "allow": torch.from_numpy(allow).to(device),
        "weight": weight / weight.sum(),
        "species": len(counts),
    }


def train_network(sources, data, rows, classes, skip_species, steps, seed, device, log) -> Net:  # noqa: PLR0913 (the views, the crops and their split, the classes, the held-out species, the schedule, device and log)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    t = _tensors(sources, data, rows, classes, skip_species, device)
    rx, rasp, rlab_t = t["rx"], t["rasp"], t["rlab"]
    qx, qasp, allow_t, weight = t["qx"], t["qasp"], t["allow"], t["weight"]
    log(f"training on {len(rlab_t)} reference views and {len(weight)} real crops of {t['species']} species")

    net = Net().to(device)
    centres = nn.Parameter(torch.randn(len(classes.ids), sprite_net.EMBED, device=device) * 0.1)
    opt = torch.optim.AdamW([*net.parameters(), centres], lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    half = 128
    started = CLOCK.now()
    for step in range(steps):
        ref = torch.from_numpy(rng.integers(0, len(rlab_t), half)).to(device)
        got = torch.from_numpy(rng.choice(len(weight), half, p=weight)).to(device)
        x = torch.cat([rx[ref], qx[got]])
        a = torch.cat([rasp[ref], qasp[got]])
        target = torch.zeros(2 * half, len(classes.ids), dtype=torch.bool, device=device)
        target[torch.arange(half, device=device), rlab_t[ref]] = True
        target[half:] = allow_t[got]
        strong = torch.cat([torch.ones(half, dtype=torch.bool), torch.zeros(half, dtype=torch.bool)]).to(device)
        x, a = augment(x, a, strong, gen)
        cos = net(x, a) @ F.normalize(centres, dim=1).T
        logits = SCALE * (cos - MARGIN * target.float())
        loss = (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~target, -1e9), 1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            log(f"step {step:5d}  loss {loss.item():.4f}  {CLOCK.now() - started:.0f}s")
    return net.eval()


def parity(net: Net, weights, device) -> float:
    x = torch.rand(16, 4, sprite_net.SIZE, sprite_net.SIZE, device=device)
    a = torch.randn(16, device=device) * 0.3
    with torch.no_grad():
        theirs = net(x, a).cpu().numpy()
    ours = sprite_net.forward(weights, x.cpu().numpy(), a.cpu().numpy())
    return float(np.abs(theirs - ours).max())


# Export ------------------------------------------------------------------------------------------------
def _write_npz(path: Path, **arrays) -> dict:
    np.savez_compressed(path, **arrays)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def export(out: Path, weights, net: sprite_net.SpriteNet, thresholds, summary: dict) -> None:
    refs = {}
    for domain, file in (("preview", "preview_refs.npz"), ("log", "log_refs.npz")):
        r = net.references[domain]
        refs[domain] = _write_npz(
            out / file,
            names=np.array(r.names),
            base_names=np.array([r.base_names[n] for n in r.names]),
            vectors=r.vectors.astype(np.float32),
        )
    metadata = {
        "schema": sprite_net.SCHEMA,
        "weights": _write_npz(out / "sprite_net.npz", **weights),
        "references": refs,
        "log_thresholds": {"min_score": thresholds[0], "min_margin": thresholds[1]},
        "trained": summary,
    }
    (out / "sprite_net.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")


def install(out: Path) -> Path:
    target = paths.assets_dir() / "sprites"
    for name in INSTALL_FILES:
        shutil.copy2(out / name, target / name)
    for old in ("roster_index.npz", "roster_index.json", "log_index.npz", "log_index.json"):
        (target / old).unlink(missing_ok=True)
    return target


# Command -----------------------------------------------------------------------------------------------
def run(args, log=print) -> int:
    source = Path(args.source) if args.source else paths.output_dir() / "sprite_source"
    runs = Path(args.runs) if args.runs else paths.output_dir() / "runs"
    out = Path(args.out) if args.out else paths.output_dir() / "sprite_net" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = {d: sb.load_source(source, d) for d in sb.SOURCES}
    classes = sd.Classes.from_sources(sources)
    cache = paths.output_dir() / "sprite_net" / "data.npz"
    if args.refresh_data or not cache.is_file():
        log(f"extracting labelled crops from {runs}")
        crops, battles = sd.extract(runs, classes, sd.Lexicon(sources), log)
        sd.save(crops, battles, cache)
    data = sd.load(cache)
    parts = split(data, args.test_runs)
    held = holdout_species(classes, args.holdout_species, args.seed)
    log(f"crops: {', '.join(f'{k} {len(v)}' for k, v in parts.items())}; species held out of training: {len(held)}")

    net = train_network(sources, data, parts["train"], classes, held, args.steps, args.seed, device, log)
    weights = net.export()
    drift = parity(net, weights, device)
    log(f"numpy vs torch max difference: {drift:.2e}")
    if drift > PARITY_LIMIT:
        raise RuntimeError(f"the numpy forward pass disagrees with torch by {drift}")
    built = sm.in_memory_net(weights, sources, data, parts["train"])
    report = sm.measure(built, sources, data, parts, classes, held, log)
    min_score, min_margin = report["thresholds"]
    summary = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "steps": args.steps,
        "seed": args.seed,
        "classes": len(classes.ids),
        "test_runs": sorted({data.battle[i].split("/")[0] for i in parts["test"]}),
        "test": report.get("test"),
    }
    export(out, weights, built, (min_score, min_margin), summary)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    torch.save({"net": net.state_dict()}, out / "train_state.pt")
    log(f"wrote {out}")
    if args.install:
        log(f"installed into {install(out)}")
    return 0
