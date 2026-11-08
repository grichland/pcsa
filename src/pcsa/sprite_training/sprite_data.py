"""Training material for the sprite network: reference views and labelled real crops from recorded runs.

Reference views come from the local source indexes (sprite_baseline.py). Real crops come from runs:

- Battle-log rows (`log.jsonl`): the row's own text names the Pokémon ("Move: The opposing Tauros used
  ..."), independent of any sprite matcher. A row whose text names exactly one species is labelled with
  that species; with the form the recorded matcher chose when it agreed on the species, otherwise with
  every form of the species (the loss accepts any of them). Rows the run recorded as showing no sprite
  are left unlabelled: what segments there is a speck, not the Pokémon the text names.
- Team-preview cards (`battle.json` + the preview frame): a card is labelled when the species the
  recorded matcher put first was sent out by the opponent (`showdown.log`), with the forms sent out.
  Every battle also keeps the opponent's sent-out species, for the recall measure.

Labels are Showdown form ids (`to_id_str` of the Showdown species), shared by both domains.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from poke_env.data.normalize import to_id_str

from pcsa.battle.names import showdown_species
from pcsa.phone import sprites
from pcsa.sprite_training.sprite_baseline import SourceIndex

DOMAIN_CODE = {"preview": 0, "log": 1}
NO_FEATURE = {"none": 1, "clipped": 2}  # variant codes; 0 is a feature


def form_id(name: str) -> str:
    return to_id_str(showdown_species(name).species)


@dataclass
class Classes:
    """Every form id the reference views name, and the base species id of each."""

    ids: list[str]
    base_of: dict[str, str]
    forms_of: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_sources(cls, sources: dict[str, SourceIndex]) -> Classes:
        base_of: dict[str, str] = {}
        for source in sources.values():
            for name, base in zip(source.names, source.base_names, strict=True):
                base_of.setdefault(form_id(name), form_id(base))
        ids = sorted(base_of)
        forms: dict[str, list[str]] = {}
        for form in ids:
            forms.setdefault(base_of[form], []).append(form)
        return cls(ids, base_of, forms)

    def base(self, form: str) -> str:
        return self.base_of.get(form, form)


class Lexicon:
    """Base species names as the game writes them in log text."""

    def __init__(self, sources: dict[str, SourceIndex]) -> None:
        names = {_plain(b) for source in sources.values() for b in source.base_names}
        self.ids = {n: to_id_str(showdown_species(n).species) for n in names}
        alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        self.pattern = re.compile(rf"(?<![A-Za-z])({alternatives})(?![A-Za-z])")

    def species(self, text: str) -> set[str]:
        return {self.ids[m] for m in self.pattern.findall(_plain(text))}


def _plain(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


@dataclass
class Crops:
    """Real crops as flat arrays (one row per crop) plus their labels."""

    domain: list[int] = field(default_factory=list)
    battle: list[str] = field(default_factory=list)  # "<run>/<battle-N>"
    masks: list[np.ndarray] = field(default_factory=list)  # (2, SIZE, SIZE) float16: raw, panel-cleaned (log)
    labs: list[np.ndarray] = field(default_factory=list)  # (2, SIZE, SIZE, 3) uint8
    aspects: list[np.ndarray] = field(default_factory=list)  # (2,) float32
    variant: list[np.ndarray] = field(default_factory=list)  # (2,) int8: 0 feature, 1 none, 2 clipped
    species: list[str] = field(default_factory=list)
    allowed: list[list[str]] = field(default_factory=list)
    recorded: list[dict] = field(default_factory=list)  # the run's own sprite result (log) or slot (preview)
    slot: list[int] = field(default_factory=list)  # preview card index, -1 for log rows


def _pack(variants: list[sprites.LogVariant | sprites.Feature | None]):
    masks = np.zeros((2, sprites.SIZE, sprites.SIZE), np.float16)
    labs = np.zeros((2, sprites.SIZE, sprites.SIZE, 3), np.uint8)
    aspects = np.ones(2, np.float32)
    codes = np.ones(2, np.int8)
    for i, v in enumerate(variants):
        if isinstance(v, sprites.Feature):
            masks[i], labs[i], aspects[i], codes[i] = v.mask, np.round(v.lab), v.aspect, 0
        elif isinstance(v, str):
            codes[i] = NO_FEATURE[v]
    return masks, labs, aspects, codes


def _add(crops: Crops, domain: str, battle: str, variants, species: str, allowed, recorded, slot: int) -> None:  # noqa: PLR0913 (one crop: where, its feature variants, its labels, its record)
    masks, labs, aspects, codes = _pack(variants)
    crops.domain.append(DOMAIN_CODE[domain])
    crops.battle.append(battle)
    crops.masks.append(masks)
    crops.labs.append(labs)
    crops.aspects.append(aspects)
    crops.variant.append(codes)
    crops.species.append(species)
    crops.allowed.append(sorted(allowed))
    crops.recorded.append(recorded)
    crops.slot.append(slot)


@dataclass
class Battle:
    key: str
    opponent_sent: dict[str, set[str]]  # base species id -> form ids sent out


def opponent_sent(showdown_log: Path, classes: Classes) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not showdown_log.is_file():
        return out
    for line in showdown_log.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) > 3 and parts[1] in ("switch", "drag") and parts[2].startswith("p2"):  # noqa: PLR2004 (|kind|who|details)
            form = to_id_str(parts[3].split(",")[0])
            out.setdefault(classes.base(form), set()).add(form)
    return out


def _frames_by_id(run: Path) -> dict[str, Path]:
    folder = run / "frames"
    return {p.name.rsplit("-", 1)[0]: p for p in folder.glob("*.jpg")} if folder.is_dir() else {}


def extract(runs: Path, classes: Classes, lexicon: Lexicon, log=print) -> tuple[Crops, list[Battle]]:
    crops, battles = Crops(), []
    for run in sorted(p for p in runs.iterdir() if p.is_dir()):
        frames = _frames_by_id(run)
        for folder in sorted(run.glob("battle-*"), key=lambda p: int(p.name.split("-")[1])):
            key = f"{run.name}/{folder.name}"
            sent = opponent_sent(folder / "showdown.log", classes)
            battles.append(Battle(key, sent))
            _preview_crops(crops, key, folder, frames, sent, classes)
            _log_crops(crops, key, run, folder, classes, lexicon)
        log(f"{run.name}: {len(crops.domain)} crops so far")
    return crops, battles


def _preview_crops(crops, key, folder, frames, sent, classes) -> None:
    path = folder / "battle.json"
    if not path.is_file():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    frame_path = frames.get(record.get("preview_frame") or "")
    slots = record.get("opponent_preview") or []
    if frame_path is None or len(slots) != len(sprites.OPPONENT_BOXES):
        return
    image = cv2.imread(str(frame_path))
    if image is None or image.shape[:2] != (1080, 2340):
        return
    for i, f in enumerate(sprites.preview_features(image)):
        slot = slots[i]
        top = slot.get("top") or []
        recorded_base = classes.base(form_id(top[0][0])) if top else ""
        species = recorded_base if recorded_base in sent else ""
        allowed = sent.get(species, set())
        _add(crops, "preview", key, [f, None], species, allowed, slot, i)


def _log_crops(crops, key, run, folder, classes, lexicon) -> None:
    path = folder / "log.jsonl"
    if not path.is_file():
        return
    images: dict[str, np.ndarray | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sprite = row.get("sprite") or {}
        source, box = sprite.get("source"), sprite.get("box")
        if not source or not box:
            continue
        if source not in images:
            images[source] = cv2.imread(str(run / source))
        image = images[source]
        if image is None:
            continue
        crop = image[box[1] : box[3], box[0] : box[2]]
        raw = sprites.log_variant(crop)
        cleaned = sprites.log_variant(sprites.without_panel(crop))
        if not isinstance(raw, sprites.Feature) and not isinstance(cleaned, sprites.Feature):
            continue
        named = lexicon.species(row.get("text", ""))
        # rows the run itself saw no sprite on ("Go! Garchomp!") only segment specks: never labelled
        species = next(iter(named)) if len(named) == 1 and sprite.get("status") != "none" else ""
        allowed: set[str] = set()
        if species:
            recorded = sprite.get("name")
            if sprite.get("status") == "matched" and recorded and classes.base(form_id(recorded)) == species:
                allowed = {form_id(recorded)}
            else:
                allowed = set(classes.forms_of.get(species, []))
        _add(crops, "log", key, [raw, cleaned], species, allowed, {**sprite, "text": row.get("text", "")}, -1)


def save(crops: Crops, battles: list[Battle], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        domain=np.array(crops.domain, np.int8),
        masks=np.stack(crops.masks),
        labs=np.stack(crops.labs),
        aspects=np.stack(crops.aspects),
        variant=np.stack(crops.variant),
        slot=np.array(crops.slot, np.int8),
    )
    meta = {
        "battle": crops.battle,
        "species": crops.species,
        "allowed": crops.allowed,
        "recorded": crops.recorded,
        "battles": [
            {"key": b.key, "opponent_sent": {k: sorted(v) for k, v in b.opponent_sent.items()}} for b in battles
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


@dataclass
class Dataset:
    domain: np.ndarray
    masks: np.ndarray
    labs: np.ndarray
    aspects: np.ndarray
    variant: np.ndarray
    slot: np.ndarray
    battle: list[str]
    species: list[str]
    allowed: list[list[str]]
    recorded: list[dict]
    battles: dict[str, dict[str, set[str]]]


def load(path: Path) -> Dataset:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    battles = {b["key"]: {k: set(v) for k, v in b["opponent_sent"].items()} for b in meta["battles"]}
    return Dataset(
        arrays["domain"], arrays["masks"], arrays["labs"], arrays["aspects"], arrays["variant"], arrays["slot"],
        meta["battle"], meta["species"], meta["allowed"], meta["recorded"], battles,
    )  # fmt: skip
