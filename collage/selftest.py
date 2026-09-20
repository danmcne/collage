"""Synthetic checks. They show that the code runs and that its geometric
invariants hold; they say nothing about any real set of images."""
from __future__ import annotations

import itertools
import math
import random
import tempfile
from pathlib import Path

from PIL import Image

from .captions import Captions
from .families import FAMILIES, ORDERS, Margins, build
from .geometry import H, V, Frame, Glue, Infeasible, Leaf, Node, height
from .render import Source, peak_bytes, probe, render, save


def check_layout(layout, leaves, aspect, max_scale=math.inf, gutter=0.01):
    eps = 1e-9
    assert sorted(r.index for r in layout.rects) == sorted(l.index for l in leaves), "every image exactly once"
    by_index = {l.index: l for l in leaves}
    for r in layout.rects:
        leaf = by_index[r.index]
        _, _, w, h = r.image
        assert w > 0 and h > 0
        assert math.isclose(r.bx, leaf.bx) and math.isclose(r.by, leaf.by), "bands kept"
        assert -eps <= r.x and r.x + r.w <= layout.width + eps, "inside canvas (x)"
        assert -eps <= r.y and r.y + r.h <= layout.height + eps, "inside canvas (y)"
        ratio = (w / h) / leaf.aspect
        band = 1 - leaf.crop
        assert band - 1e-9 <= ratio <= 1 / band + 1e-9, "aspect within crop band"
        if leaf.crop == 0:
            assert math.isclose(ratio, 1, rel_tol=1e-9), "uncropped aspect exact"
    for a, b in itertools.combinations(layout.rects, 2):
        ox = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
        oy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
        assert ox <= eps or oy <= eps, "no overlap"
    if aspect is not None:
        assert (math.isclose(layout.aspect, aspect, rel_tol=1e-7)
                or math.isclose(abs(layout.ratio), 1.0)), "target met, or slack exhausted"
    left = min(r.x for r in layout.rects)
    right = layout.width - max(r.x + r.w for r in layout.rects)
    top = min(r.y for r in layout.rects)
    bottom = layout.height - max(r.y + r.h for r in layout.rects)
    assert math.isclose(left, right, abs_tol=1e-9), "left and right margins equal"
    assert top <= bottom + 1e-9, "top margin not larger than bottom"
    assert min(left, top) >= gutter * (1 - 1e-9), "margins at least the gutter"
    assert layout.size_ratio <= max_scale * (1 + 1e-9), "size limit respected"
    assert -1 - eps <= layout.ratio <= 1 + eps
    assert 0 <= layout.empty < 1


def reads_in(layout, order, rtl):
    """Geometric consequence of the reading-order rule: no image is entirely above
    (row order) or entirely before in the cross direction of an image read earlier,
    and images sharing a line come in reading direction."""
    rects = []
    for r in sorted(layout.rects, key=lambda r: r.index):
        x = layout.width - r.x - r.w if rtl else r.x
        rects.append((x, r.y, r.w, r.h) if order == "rows" else (r.y, x, r.h, r.w))
    eps = 1e-9
    for (ax, ay, aw, ah), (bx, by, bw, bh) in itertools.combinations(rects, 2):
        if by + bh <= ay + eps:
            return False
        if by < ay + ah - eps and ay < by + bh - eps and bx + bw <= ax + eps:
            return False
    return True


def run() -> int:
    rng = random.Random(1)
    solved = refused = 0
    cases = list(itertools.product(
        (1, 2, 5, 13), FAMILIES, ORDERS, (False, True), (0.0, 0.08), (None, 1.5, 0.6), (0.0, 0.03),
        (math.inf, 1.5)))
    for n, family, order, rtl, crop, target, band, limit in rng.sample(cases, 160):
        leaves = [Leaf(i, rng.choice((4 / 3, 3 / 4, 1.0, 16 / 9, 2 / 3)) * rng.uniform(0.95, 1.05), crop, by=band)
                  for i in range(n)]
        try:
            layout = build(family, leaves, Glue(0.01, 0.25), Margins((0.02,) * 4, 0.5), target, max_scale=limit,
                           strict=limit < math.inf, order=order, rtl=rtl, seed=n)
        except Infeasible:
            refused += 1
            continue
        check_layout(layout, leaves, target, limit if family in ("lines", "tree") else math.inf)
        wanted = ("rows", "columns") if order == "either" or family in ("strip", "column") else (order,)
        assert any(reads_in(layout, o, rtl) for o in wanted), "reads in the requested order"
        solved += 1

    leaves = [Leaf(i, a, 0.1, by=0.02) for i, a in enumerate((1.5, 0.7, 1.0, 1.33, 0.8))]
    tree = Node(V, (Node(H, tuple(leaves[:3])), Node(H, tuple(leaves[3:]))))
    frame = Frame.make((0.03,) * 4, 0.5, 0.02, False)
    hs = [height(tree, Glue(0.02, 0.3), frame, t / 10) for t in range(-10, 11)]
    assert all(a < b for a, b in zip(hs, hs[1:])), "height increases with the glue set ratio"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sources = []
        for i, (w, h) in enumerate(((400, 300), (300, 400), (500, 500), (640, 360))):
            p = tmp / f"s{i}.jpg"
            img = Image.new("RGB", (w, h), (60 * i, 120, 200 - 40 * i))
            exif = Image.Exif()
            if i == 1:
                exif[0x0112] = 6   # stored landscape, displayed portrait
                img = img.rotate(90, expand=True)
            img.save(p, exif=exif)
            w_, h_, fmt = probe(p)
            sources.append(Source(p, w_, h_, None, f"caption number {i + 1} for testing", fmt))
        assert (sources[1].width, sources[1].height) == (300, 400), "EXIF orientation respected"
        captions = Captions({i: s.caption for i, s in enumerate(sources)}, 0.03)
        for family in FAMILIES:
            layout, lines = captions.fit(lambda band: build(
                family, [Leaf(i, s.width / s.height, by=band) for i, s in enumerate(sources)],
                Glue(0.02, 0.25), Margins((0.04,) * 4, 0.5), max_scale=math.inf))
            assert all(len(captions.wrap(captions.texts[r.index], r.image[2])) <= lines for r in layout.rects)
            img, _ = render(layout, sources, 800, (255, 255, 255), 0.03, 0.03, captions)
            assert img.width == 800
            assert sum(peak_bytes(layout, sources, 800)) > 0
        save(img, tmp / "out.png", None, (255, 255, 255))
        save(img, tmp / "out.jpg", 300, (255, 255, 255))
        assert Image.open(tmp / "out.jpg").size == img.size

    print(f"self-test passed: {solved} layouts checked, {refused} refused under a strict size limit, "
          "captioned rendering and saving ran. Synthetic inputs only: this shows the code works, "
          "not how any real images will lay out.")
    return 0
