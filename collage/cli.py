"""Collages that keep every image whole and in reading order.

    python -m collage INPUT... -o OUT [options]

INPUT is any mix of image files, directories (their images, sorted by name) and
.toml manifests, concatenated in the order given; that order is the reading
order. A manifest lists images and may give each a crop tolerance and a caption:

    [[image]]
    path = "shots/01.jpg"          # relative to the manifest
    crop = 0.05                    # optional; overrides --crop-tol
    caption = "The harbour at dawn"

Layouts: `lines` breaks the sequence into lines as TeX breaks a paragraph;
`strip` and `column` put every image in one line; `tree` searches all slicing
layouts (images beside or above one another, recursively) that read correctly.

Reading order: `rows` reads left to right, then top down, like a book;
`columns` reads top down, then left to right; `either` keeps whichever composes
better. --rtl mirrors the placement (not the images) for right-to-left reading.

Images are never cropped unless allowed. With --aspect (or a print size) the
canvas is brought to that shape by flexing, all together and in proportion to
their tolerances, the gutters (--gutter-tol), the margins (--margin-tol) and,
if allowed, the crop (--crop-tol). Margins never go below the gutter width, left
and right stay equal, and the bottom may grow further than the top. If the
bounds cannot reach the shape, the nearest reachable one is used and reported.
The only empty space is a "thumb spot": a short last line may stop early, if
the gap is no wider than one of its images.

Image sizes are kept within --max-scale of one another. By default this is a
preference: the best layout within it wins, and if there is none the program
still produces the closest one and says so. Given explicitly, it is strict and
the program refuses instead.

Lengths (gutter, margins, radius, caption size) are fractions of the layout's
breadth: its width when reading in rows, its height when reading in columns
or for a strip.

Every run prints the layout's badness: 100 x the sum of squares of the spread of
log image sizes; the log ratio of the canvas aspect to the one requested (or,
without --aspect, to 3:2, for lines and tree only, which choose their shape);
the relative change of gutters and of margins; the largest crop; and the
fraction left empty by a thumb spot. Lower is better. Layouts are chosen first
by closeness to a requested aspect, then by staying within --max-scale, then by
badness. The tree search runs several independent restarts and keeps the best;
--seed changes them all.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageColor

from .captions import Captions
from .families import FAMILIES, ORDERS, PREFERRED_ASPECT, Margins, badness, build
from .geometry import Glue, Infeasible, Leaf
from .render import Source, canvas_size, peak_bytes, probe, render, save

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
UNITS = {"mm": 25.4, "cm": 2.54, "in": 1.0}
MB = 1 << 20


@dataclass(frozen=True)
class Size:
    kind: str           # "w" | "h" | "print" | "long"
    a: float
    b: float = 0.0
    label: str = ""

    @property
    def aspect(self):
        return self.a / self.b if self.kind == "print" else None

    def px_width(self, aspect: float, dpi: int) -> int:
        if self.kind == "w":
            return int(self.a)
        if self.kind == "h":
            return round(self.a * aspect)
        if self.kind == "print":
            return round(self.a * dpi)
        return int(self.a) if aspect >= 1 else round(self.a * aspect)


def parse_size(spec: str) -> Size:
    if m := re.fullmatch(r"(\d+)(?:px)?", spec):
        return Size("w", int(m[1]), label=f"{m[1]}w")
    if m := re.fullmatch(r"x(\d+)(?:px)?", spec):
        return Size("h", int(m[1]), label=f"{m[1]}h")
    if m := re.fullmatch(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(mm|cm|in)", spec):
        k = UNITS[m[3]]
        return Size("print", float(m[1]) / k, float(m[2]) / k, label=f"{m[1]}x{m[2]}{m[3]}")
    raise argparse.ArgumentTypeError(
        f"bad size {spec!r}: use 2400 (px width), x1600 (px height) or 297x210mm / 30x20cm / 8x10in")


def parse_aspect(spec: str) -> float:
    try:
        if ":" in spec:
            w, h = spec.split(":")
            value = float(w) / float(h)
        else:
            value = float(spec)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad aspect {spec!r}: use 16:9 or 1.778")
    if value <= 0:
        raise argparse.ArgumentTypeError("aspect must be positive")
    return value


def parse_margin(spec: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(v) for v in spec.split(","))
    except ValueError:
        values = ()
    if len(values) not in (1, 4) or not all(0 <= v < 0.5 for v in values):
        raise argparse.ArgumentTypeError(f"bad margin {spec!r}: one value or left,right,top,bottom, each in [0, 0.5)")
    return values * 4 if len(values) == 1 else values


def fraction(lo: float, hi: float, name: str):
    def check(text):
        v = float(text)
        if not lo <= v < hi:
            raise argparse.ArgumentTypeError(f"{name} must be in [{lo}, {hi})")
        return v
    return check


def collect(inputs: list[str], limit: int) -> list[Source]:
    entries: list[tuple[Path, float | None, str | None]] = []
    for arg in inputs:
        p = Path(arg)
        if p.is_dir():
            entries += [(f, None, None) for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
        elif p.suffix.lower() == ".toml":
            for e in tomllib.loads(p.read_text()).get("image", []):
                entries.append((p.parent / e["path"], e.get("crop"), e.get("caption")))
        elif p.is_file():
            entries.append((p, None, None))
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
        if len(entries) > limit:
            raise ValueError(f"more than --max-images {limit} images; layout time grows with the count")
    sources = []
    for path, crop, caption in entries:
        w, h, fmt = probe(path)
        sources.append(Source(path, w, h, crop, caption, fmt))
    return sources


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="collage", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", help="image files, directories, or .toml manifests")
    p.add_argument("-o", "--output", type=Path, help="output file; the extension sets the format")

    g = p.add_argument_group("layout")
    g.add_argument("--layout", choices=FAMILIES, default="lines", help="default: lines")
    g.add_argument("--order", choices=ORDERS,
                   help="reading order (default: rows; either with --any-order)")
    g.add_argument("--rtl", action="store_true", help="read right to left (mirrors the placement)")
    g.add_argument("--any-order", action="store_true",
                   help="tree: the sequence does not matter, so images may be reordered")
    g.add_argument("--aspect", type=parse_aspect,
                   help="canvas aspect, e.g. 16:9; met exactly if the slack reaches it, else as nearly as it can")
    g.add_argument("--max-scale", type=float,
                   help="largest ratio between the linear sizes (square root of area) of any two images, "
                        "for lines and tree. Default: 1.5 as a preference; given explicitly: strict, refusing "
                        "if unmet ('inf' for none). Strip and column only report it")
    g.add_argument("--row-height", type=float,
                   help="lines: target image height, as a fraction of breadth (default: chosen automatically)")
    g.add_argument("--last-row", choices=("ragged", "fill"), default="ragged",
                   help="lines: set a short last line at the size of the others, or stretch it (default ragged)")
    g.add_argument("--seed", type=int, default=0, help="tree: search seed")

    g = p.add_argument_group("spacing and cropping")
    g.add_argument("--gutter", type=fraction(0, 0.5, "--gutter"), default=0.01,
                   help="nominal gutter, as a fraction of breadth (default 0.01)")
    g.add_argument("--gutter-tol", type=fraction(0, 1, "--gutter-tol"), default=0.25,
                   help="largest relative change of any gutter when meeting --aspect (default 0.25)")
    g.add_argument("--crop-tol", type=fraction(0, 1, "--crop-tol"), default=0.0,
                   help="largest fraction cut from one side of an image (default 0: never crop)")
    g.add_argument("--margin", type=parse_margin,
                   help="nominal outer margin, one value or left,right,top,bottom (default: twice the gutter)")
    g.add_argument("--margin-tol", type=fraction(0, 1, "--margin-tol"), default=0.5,
                   help="largest relative change of the margins, never below the gutter; the bottom may "
                        "stretch twice as far (default 0.5)")
    g.add_argument("--radius", type=float, default=0.0, help="image corner radius, as a fraction of breadth")
    g.add_argument("--canvas-radius", type=float, default=0.0, help="outer corner radius, as a fraction of breadth")
    g.add_argument("--background", default="white", help="colour, or 'none' for transparent (PNG, TIFF, WebP)")

    g = p.add_argument_group("captions")
    g.add_argument("--caption-size", type=float, default=0.018,
                   help="font size, as a fraction of breadth (default 0.018)")
    g.add_argument("--caption-lines", type=int, default=3, help="most lines a caption may take (default 3)")
    g.add_argument("--caption-color", default="#202020")
    g.add_argument("--font", help="TrueType/OpenType font file (default: Pillow's built-in)")

    g = p.add_argument_group("output")
    g.add_argument("--size", type=parse_size, action="append",
                   help="repeatable: 2400 (px width), x1600 (px height), 297x210mm (print; also sets the "
                        "aspect). Default: 2400 px on the long side")
    g.add_argument("--dpi", type=int, default=300, help="resolution for print sizes (default 300)")

    g = p.add_argument_group("limits")
    g.add_argument("--max-images", type=int, default=200, help="most input images (default 200)")
    g.add_argument("--max-memory", type=int, default=1024,
                   help="refuse any output whose estimated peak memory exceeds this many MB (default 1024)")

    p.add_argument("--selftest", action="store_true",
                   help="check geometry and rendering on synthetic inputs, then exit")
    return p


def report(layout, pref, max_scale, strict) -> str:
    lines = [f"canvas aspect {layout.aspect:.4f}, {len(layout.rects)} images, "
             f"sizes within {layout.size_ratio:.2f}x", str(badness(layout, pref))]
    if layout.target and not math.isclose(layout.aspect, layout.target, rel_tol=1e-6):
        lines.append(f"note: aspect {layout.target:.4f} was requested; the gutters, margins and crop "
                     f"at their limits reach {layout.aspect:.4f} (widen --margin-tol, --gutter-tol "
                     "or --crop-tol, or relax --max-scale, to get closer)")
    if layout.size_ratio > max_scale * (1 + 1e-9) and not strict:
        lines.append(f"note: sizes exceed the preferred --max-scale {max_scale:g}; "
                     "no layout of this kind was found within it")
    if layout.ratio:
        lines.append(f"glue set ratio {layout.ratio:+.3f}")
    return "\n".join(lines)


def fail(message: str, code: int = 1) -> int:
    print(f"collage: {message}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    if args.selftest:
        from .selftest import run
        return run()
    if not args.inputs or args.output is None:
        ap.error("inputs and -o/--output are required")
    strict = args.max_scale is not None
    max_scale = args.max_scale if strict else 1.5
    if max_scale < 1:
        ap.error("--max-scale must be at least 1")
    order = args.order or ("either" if args.any_order else "rows")

    sizes = args.size or [Size("long", 2400, label="2400")]
    print_aspects = {round(s.aspect, 9) for s in sizes if s.aspect}
    if len(print_aspects) > 1:
        ap.error("print sizes with different aspects need separate runs")
    aspect = args.aspect
    if print_aspects:
        (pa,) = print_aspects
        if aspect is not None and not math.isclose(aspect, pa, rel_tol=1e-6):
            ap.error(f"--aspect {aspect:.4g} conflicts with the print size aspect {pa:.4g}")
        aspect = pa

    try:
        background = None if args.background.lower() in ("none", "transparent") else ImageColor.getrgb(args.background)
        sources = collect(args.inputs, args.max_images)
    except (OSError, ValueError, tomllib.TOMLDecodeError, KeyError, Image.DecompressionBombError) as e:
        return fail(str(e), 2)
    if not sources:
        return fail("no images found", 2)
    try:
        captions = Captions({i: s.caption for i, s in enumerate(sources) if s.caption},
                            args.caption_size, args.font, ImageColor.getrgb(args.caption_color),
                            args.caption_lines)
    except (OSError, ValueError) as e:
        return fail(f"caption font or colour: {e}", 2)

    glue = Glue(args.gutter, args.gutter_tol)
    margins = Margins(args.margin or (2 * args.gutter,) * 4, args.margin_tol)

    def lay_out(band):
        leaves = [Leaf(i, s.width / s.height, args.crop_tol if s.crop is None else s.crop, by=band)
                  for i, s in enumerate(sources)]
        return build(args.layout, leaves, glue, margins, aspect, max_scale=max_scale, strict=strict,
                     order=order, rtl=args.rtl, row_height=args.row_height,
                     ragged=args.last_row == "ragged", any_order=args.any_order, seed=args.seed)

    try:
        layout, caption_lines = captions.fit(lay_out)
    except (Infeasible, ValueError) as e:
        return fail(str(e))
    chooses_shape = aspect is None and args.layout in ("lines", "tree")
    print(report(layout, PREFERRED_ASPECT if chooses_shape else None, max_scale,
                 strict or args.layout in ("strip", "column"))
          + (f"\ncaption band: {caption_lines} line(s)" if captions else ""))

    jobs = []
    for size in sizes:
        px = size.px_width(layout.aspect, args.dpi)
        canvas, worst = peak_bytes(layout, sources, px)
        if canvas + worst > args.max_memory * MB:
            W, H, _ = canvas_size(layout, px)
            return fail(f"{W}x{H} px would need about {(canvas + worst) / MB:.0f} MB "
                        f"({canvas / MB:.0f} for the canvas, {worst / MB:.0f} for the largest image), "
                        f"above --max-memory {args.max_memory}; choose a smaller --size or raise the limit")
        jobs.append((size, px))

    for size, px in jobs:
        out = args.output if len(sizes) == 1 else args.output.with_name(
            f"{args.output.stem}-{size.label}{args.output.suffix}")
        dpi = args.dpi if size.kind == "print" else None
        img, upscales = render(layout, sources, px, background, args.radius, args.canvas_radius, captions)
        save(img, out, dpi, background or (255, 255, 255))
        print(f"wrote {out}  {img.width}x{img.height} px" + (f" at {dpi} dpi" if dpi else ""))
        for i, f in sorted(upscales.items()):
            extra = f" (effective {args.dpi / f:.0f} dpi)" if dpi else ""
            print(f"  warning: {sources[i].path.name} enlarged {f:.2f}x beyond its pixels{extra}")
    return 0
