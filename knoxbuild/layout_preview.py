"""A picture of what the building step decided, to look at before compiling.

Compiling a big map takes the best part of an hour, and the settings that shape
it (spawn density, apartment share, street scale) are only judged by looking at
the result. This draws the ground, every building coloured by what it is, and
optionally how many zombies each part of the map will start with, from files
the build already wrote - so it takes a second and needs no WorldEd.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bitmaps import read_rgb

MAX_SIDE = 2400

# Kinds the build makes, with fixed colours so a colour means the same thing on
# every map. Anything else gets the grey at the end.
KIND_COLOURS = {
    "house": (240, 200, 80), "apartment": (230, 120, 60), "shop": (70, 140, 230),
    "restaurant": (230, 80, 160), "shed": (170, 170, 170), "school": (90, 200, 120),
    "civic": (150, 110, 220), "church": (250, 250, 250), "medical": (230, 60, 60),
    "industrial": (110, 90, 70), "fire": (255, 120, 0), "police": (40, 70, 200),
    "library": (120, 220, 220), "stadium": (60, 160, 90), "mall": (200, 60, 200),
}
OTHER = (200, 200, 200)

LAYERS = ("kinds", "zombies")


def _info(map_dir: str) -> tuple[str, dict]:
    name = os.path.basename(os.path.normpath(map_dir))
    with open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8") as f:
        return name, json.load(f)


def _step(width: int, height: int) -> int:
    return max(1, -(-max(width, height) // MAX_SIDE))


def draw(map_dir: str, layer: str = "kinds") -> Image.Image:
    """The preview as a PIL image. `layer` is "kinds" (buildings by use) or
    "zombies" (buildings in outline over a heat map of the spawn map)."""
    name, info = _info(map_dir)
    width, height = info["width_tiles"], info["height_tiles"]
    step = _step(width, height)
    base = Image.fromarray(read_rgb(os.path.join(map_dir, f"{name}.bmp"), step=step))
    # Washed out: the colours drawn on top are what is being looked at.
    base = Image.blend(base, Image.new("RGB", base.size, (235, 235, 235)), 0.55)

    rows = []
    csv_path = os.path.join(map_dir, f"{name}_placements.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    if layer == "zombies":
        spawn_path = os.path.join(map_dir, f"{name}_ZombieSpawnMap.bmp")
        if os.path.exists(spawn_path):
            # Written as a grey picture the size of the map in tenths, but
            # stored as 24-bit, so the red channel is the level.
            spawn = read_rgb(spawn_path)[:, :, 0]
            top = max(1, int(spawn.max()))
            heat = Image.fromarray((spawn.astype(np.float32) / top * 255).astype(np.uint8))
            heat = heat.resize(base.size, Image.BILINEAR)
            h = np.asarray(heat).astype(np.float32) / 255.0
            arr = np.asarray(base).astype(np.float32)
            red = np.array([220, 30, 30], dtype=np.float32)
            arr = arr * (1 - 0.8 * h[..., None]) + red * (0.8 * h[..., None])
            base = Image.fromarray(arr.astype(np.uint8))

    pen = ImageDraw.Draw(base)
    used: dict[tuple[str, str], int] = {}
    labels = []
    for r in rows:
        kind = r.get("kind") or "house"
        # What it sells, where the build said: a pizza place is a pizza place
        # here, not one more "shop" (knoxbuild/uses.trade_of).
        trade = [t for t in (r.get("trade") or "").split("; ") if t]
        business = [b for b in (r.get("business") or "").split("; ") if b]
        key = (trade[0] if trade else kind, kind)
        used[key] = used.get(key, 0) + 1
        x0, y0 = int(r["tile_x"]) // step, int(r["tile_y"]) // step
        x1 = max(x0 + 1, (int(r["tile_x"]) + int(r["width"])) // step)
        y1 = max(y0 + 1, (int(r["tile_y"]) + int(r["height"])) // step)
        if layer == "zombies":
            pen.rectangle([x0, y0, x1, y1], outline=(30, 30, 30))
        else:
            pen.rectangle([x0, y0, x1, y1], fill=KIND_COLOURS.get(kind, OTHER),
                          outline=(40, 40, 40))
        if trade and layer == "kinds":
            short = ", ".join(trade)
            long = short + ("\n" + ", ".join(business) if business else "")
            labels.append(((x1 - x0) * (y1 - y0), (x0 + x1) / 2, (y0 + y1) / 2, long, short))
    _labels(pen, labels, _font(base.size, 110))
    _legend(pen, base.size, used, layer)
    return base


def _labels(pen: ImageDraw.ImageDraw, labels: list, font) -> None:
    """What each shop is, written on it, on top of every building. A street of
    shops side by side wrote them over one another until none could be read,
    so each takes room nothing else has taken: the biggest buildings first,
    with the business's name if it fits, else what it is, else nothing - the
    legend still counts it."""
    taken: list[tuple[float, float, float, float]] = []

    def free(box):
        return all(box[2] < t[0] or box[0] > t[2] or box[3] < t[1] or box[1] > t[3]
                   for t in taken)

    for _area, x, y, long, short in sorted(labels, key=lambda l: -l[0]):
        for text in dict.fromkeys((long, short)):
            box = pen.multiline_textbbox((x, y), text, font=font, anchor="mm",
                                         align="center", stroke_width=2)
            if free(box):
                taken.append(box)
                pen.multiline_text((x, y), text, fill=(20, 20, 20), font=font,
                                   anchor="mm", align="center", stroke_width=2,
                                   stroke_fill=(255, 255, 255))
                break


def _legend(pen: ImageDraw.ImageDraw, size: tuple[int, int], used: dict,
            layer: str) -> None:
    font = _font(size, 90)
    if layer == "zombies":
        items = [("more zombies = redder", (220, 30, 30), None)]
    else:
        # Grouped by kind, the commonest first, so the shops' trades sit
        # together under one colour.
        per_kind: dict[str, int] = {}
        for (_label, kind), n in used.items():
            per_kind[kind] = per_kind.get(kind, 0) + n
        order = sorted(used.items(), key=lambda kv: (-per_kind[kv[0][1]], kv[0][1], -kv[1]))
        items = [(label, KIND_COLOURS.get(kind, OTHER), n) for (label, kind), n in order]
    line = max(14, size[0] // 80)
    pad = line // 2
    wide = max([pen.textlength(label if n is None else f"{label} {n}", font=font)
                for label, _c, n in items] + [line * 10])
    pen.rectangle([pad, pad, pad * 3 + line + wide, pad * 2 + line * len(items) + pad],
                  fill=(255, 255, 255), outline=(60, 60, 60))
    for i, (label, colour, n) in enumerate(items):
        y = pad * 2 + i * line
        pen.rectangle([pad * 2, y, pad * 2 + line - 4, y + line - 4], fill=colour,
                      outline=(40, 40, 40))
        pen.text((pad * 2 + line, y - 1), label if n is None else f"{label} {n}",
                 fill=(20, 20, 20), font=font)


def _font(size: tuple[int, int], per: int):
    """The default font, one `per`th of the picture's width."""
    try:
        return ImageFont.load_default(size=max(10, size[0] // per))
    except TypeError:                      # an older Pillow has one size only
        return ImageFont.load_default()


def render_to(map_dir: str, layer: str = "kinds") -> str:
    """Write the preview next to the map and return its path. It is drawn again
    only when something it is made from is newer."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}")
    name, _info_ = _info(map_dir)
    out = os.path.join(map_dir, f"{name}_layout_{layer}.png")
    sources = [os.path.join(map_dir, f"{name}{s}") for s in
               (".bmp", "_placements.csv", "_ZombieSpawnMap.bmp", "_info.json")]
    newest = max((os.path.getmtime(p) for p in sources if os.path.exists(p)),
                 default=0)
    if not os.path.exists(out) or os.path.getmtime(out) < newest:
        draw(map_dir, layer).save(out, format="PNG")
    return out
