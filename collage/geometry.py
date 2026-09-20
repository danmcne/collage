"""Exact geometry of slicing layouts.

A layout is a tree. A leaf is an image with a fixed aspect ratio (width / height).
An internal node either sets its children side by side at a shared height (cut H)
or stacks them at a shared width (cut V). Reading the leaves left to right gives
the reading order: left before right, top before bottom.

Allotted a width w, every subtree has height  alpha * w + beta,  where alpha comes
from the images and beta from the gutters. The coefficients compose exactly:

    leaf:    alpha = 1 / aspect,           beta = by - bx / aspect
    V node:  alpha = sum alpha_i,          beta = sum beta_i + (k-1) g
    H node:  alpha = 1 / sum(1/alpha_i),   beta = alpha * (sum beta_i/alpha_i - (k-1) g)

so a layout is solved in one bottom-up pass and placed in one top-down pass.

Slack. Everything flexible is glue with a nominal value and two bounds, and all
of it is set by one ratio tau in [-1, 1], as TeX sets a line: at tau = 0 all is
nominal; as tau rises every piece moves toward the bound that makes the layout
taller, reaching it at tau = 1, and as tau falls toward the other bound.
Taller means: gutters between side-by-side images narrower and between stacked
ones wider; side margins narrower (wider content) and top and bottom margins
wider; images with a crop tolerance narrower. Height is therefore monotone in
tau, and a prescribed canvas aspect is met by bisection, or as nearly as the
bounds allow, at tau = +1 or -1.

Margins. In the finished picture the left and right margins are always equal,
and the bottom margin may stretch twice as far as the top, so the top is never
the larger. Each margin moves between the larger of the gutter and
nominal x (1 - tol), and nominal x (1 + tol) (1 + 2 tol for the bottom).

A node with `fill` set is an H node at that fixed height whose children keep
their natural size, leaving the rest of the width empty: TeX's \\parfillskip,
used for a short last line. The empty area is recorded on the layout.

A leaf may carry fixed bands beside (bx) or below (by) its image, such as a
caption; the image keeps its aspect and the band keeps its size.

Units: the canvas width is 1 in the frame where layouts are solved (row order);
column order is solved transposed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

H, V = "h", "v"


@dataclass(frozen=True)
class Leaf:
    index: int          # position in the input sequence
    aspect: float       # natural width / height
    crop: float = 0.0   # largest fraction that may be cut from one side, in [0, 1)
    bx: float = 0.0     # fixed band beside the image
    by: float = 0.0     # fixed band below the image


@dataclass(frozen=True)
class Node:
    cut: str
    children: tuple
    fill: float | None = None


@dataclass(frozen=True)
class Flex:
    """A length with a nominal value and bounds lo <= nominal <= hi."""
    nominal: float
    lo: float
    hi: float

    def at(self, s: float) -> float:
        """s in [-1, 1]: toward hi as s rises, toward lo as it falls."""
        return self.nominal + s * ((self.hi if s > 0 else self.nominal) - (self.nominal if s > 0 else self.lo))

    def change(self, s: float) -> float:
        return abs(self.at(s) - self.nominal) / self.nominal if self.nominal else 0.0


@dataclass(frozen=True)
class Frame:
    """Margins in the solving frame. left/right shrink as tau rises (the content widens);
    top/bottom grow."""
    left: Flex
    right: Flex
    top: Flex
    bottom: Flex

    def at(self, tau: float) -> tuple[float, float, float, float]:
        return self.left.at(-tau), self.right.at(-tau), self.top.at(tau), self.bottom.at(tau)

    def change(self, tau: float) -> float:
        return max(self.left.change(-tau), self.right.change(-tau), self.top.change(tau), self.bottom.change(tau))

    @staticmethod
    def make(nominal: tuple[float, float, float, float], tol: float, floor: float, transposed: bool) -> "Frame":
        """nominal (left, right, top, bottom) in the finished picture, whose bottom may stretch
        twice as far; transposed: build it for the transposed solving frame of column order."""
        def flex(n, stretch):
            return Flex(n, min(n, max(floor, n * (1 - tol))), n * (1 + stretch * tol))
        l, r, t, b = nominal
        if transposed:      # solving left/right/top/bottom = final top/bottom/left/right
            return Frame(flex(t, 1), flex(b, 2), flex(l, 1), flex(r, 1))
        return Frame(flex(l, 1), flex(r, 1), flex(t, 1), flex(b, 2))


@dataclass(frozen=True)
class Glue:
    gutter: float       # nominal gutter
    tol: float = 0.0    # largest relative change of any gutter, in [0, 1)


@dataclass(frozen=True)
class Rect:
    index: int
    x: float
    y: float
    w: float
    h: float
    bx: float = 0.0
    by: float = 0.0
    natural: float = 1.0    # the image's own aspect, before any crop

    @property
    def crop(self) -> float:
        """Fraction removed from one side of the image."""
        _, _, w, h = self.image
        q = (w / h) / self.natural
        return 1.0 - min(q, 1.0 / q)

    @property
    def image(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.w - self.bx, self.h - self.by

    @property
    def size(self) -> float:
        return math.sqrt((self.w - self.bx) * (self.h - self.by))


@dataclass(frozen=True)
class Layout:
    width: float
    height: float
    rects: tuple
    ratio: float            # the glue set ratio tau actually used
    gutter_change: float    # largest relative change of a gutter
    margin_change: float    # largest relative change of a margin
    empty: float = 0.0      # fraction of the canvas left empty by a ragged last line
    target: float | None = None   # the aspect that was asked for

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def size_ratio(self) -> float:
        sizes = [r.size for r in self.rects]
        return max(sizes) / min(sizes)

    def _with(self, width, height, rects, target):
        return Layout(width, height, tuple(rects), self.ratio, self.gutter_change, self.margin_change,
                      self.empty, target)

    def transposed(self) -> "Layout":
        return self._with(self.height, self.width,
                          (Rect(r.index, r.y, r.x, r.h, r.w, r.by, r.bx, 1.0 / r.natural) for r in self.rects),
                          None if self.target is None else 1.0 / self.target)

    def mirrored(self) -> "Layout":
        """Mirror the placement left to right; the images themselves are not flipped."""
        return self._with(self.width, self.height,
                          (Rect(r.index, self.width - r.x - r.w, r.y, r.w, r.h, r.bx, r.by, r.natural)
                           for r in self.rects), self.target)


class Infeasible(ValueError):
    pass


def leaf_aspect(leaf: Leaf, tau: float) -> float:
    return leaf.aspect * (1.0 - leaf.crop) ** tau


def gap(node: Node, glue: Glue, tau: float) -> float:
    if node.fill is not None:
        return glue.gutter
    sign = 1.0 if node.cut == V else -1.0
    return glue.gutter * (1.0 + sign * glue.tol * tau)


def affine(t, glue: Glue, tau: float) -> tuple[float, float]:
    """(alpha, beta) with subtree height = alpha * width + beta."""
    if isinstance(t, Leaf):
        r = leaf_aspect(t, tau)
        return 1.0 / r, t.by - t.bx / r
    if t.fill is not None:
        return 0.0, t.fill
    parts = [affine(c, glue, tau) for c in t.children]
    gaps = gap(t, glue, tau) * (len(parts) - 1)
    if t.cut == V:
        return sum(a for a, _ in parts), sum(b for _, b in parts) + gaps
    alpha = 1.0 / sum(1.0 / a for a, _ in parts)
    return alpha, alpha * (sum(b / a for a, b in parts) - gaps)


def place(t, glue: Glue, tau: float, x: float, y: float, w: float, out: list, holes: list | None = None) -> None:
    if isinstance(t, Leaf):
        out.append(Rect(t.index, x, y, w, (w - t.bx) / leaf_aspect(t, tau) + t.by, t.bx, t.by, t.aspect))
        return
    g = gap(t, glue, tau)
    if t.cut == V:
        for c in t.children:
            a, b = affine(c, glue, tau)
            place(c, glue, tau, x, y, w, out, holes)
            y += a * w + b + g
        return
    if t.fill is not None:
        h = t.fill
    else:
        a, b = affine(t, glue, tau)
        h = a * w + b
    x0 = x
    for c in t.children:
        a, b = affine(c, glue, tau)
        cw = (h - b) / a
        place(c, glue, tau, x, y, cw, out, holes)
        x += cw + g
    if t.fill is not None and holes is not None:
        holes.append((x0 + w - (x - g)) * h)


def inner_width(frame: Frame, tau: float) -> float:
    left, right, _, _ = frame.at(tau)
    return 1.0 - left - right


def height(tree, glue: Glue, frame: Frame, tau: float) -> float:
    left, right, top, bottom = frame.at(tau)
    a, b = affine(tree, glue, tau)
    return a * (1.0 - left - right) + b + top + bottom


def reach(tree, glue: Glue, frame: Frame) -> tuple[float, float]:
    """The range of canvas aspects the slack can reach: (narrowest, widest)."""
    return 1.0 / height(tree, glue, frame, 1.0), 1.0 / height(tree, glue, frame, -1.0)


def solve(tree, glue: Glue, frame: Frame, aspect: float | None = None) -> Layout:
    """Place `tree` with canvas width 1; with `aspect`, as near that shape as the slack reaches."""
    tau = 0.0
    if aspect is not None:
        want = 1.0 / aspect
        if height(tree, glue, frame, 1.0) <= want:
            tau = 1.0
        elif height(tree, glue, frame, -1.0) >= want:
            tau = -1.0
        else:
            a, b = -1.0, 1.0
            for _ in range(60):
                mid = 0.5 * (a + b)
                if height(tree, glue, frame, mid) < want:
                    a = mid
                else:
                    b = mid
            tau = 0.5 * (a + b)
    left, right, top, bottom = frame.at(tau)
    rects: list[Rect] = []
    holes: list[float] = []
    place(tree, glue, tau, left, top, 1.0 - left - right, rects, holes)
    if any(r.w <= r.bx or r.h <= r.by for r in rects):
        raise Infeasible("gutters and margins leave no room for the images; reduce --gutter or --margin")
    h = height(tree, glue, frame, tau)
    return Layout(1.0, h, tuple(rects), tau, glue.tol * abs(tau), frame.change(tau), sum(holes) / h, aspect)
