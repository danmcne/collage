"""Layout families, reading order, direction, and badness.

Families restrict the shape of the slicing tree:

    lines    lines of images, breaks chosen as TeX breaks a paragraph
    strip    one horizontal line, equal heights
    column   one vertical line, equal widths
    tree     any slicing tree consistent with the reading order, found by search

Reading order. The sequence is read as the tree is traversed (left child before
right, top before bottom). In row order (left to right, then top down) that
traversal is unambiguous when a stack of images appears only as the last item of
a row: the reader finishes the row and then reads the stack downward. Column
order (top down, then left to right) is the same rule transposed. `either` tries
both and keeps the less bad. Everything is computed in row order; column order
is row order on the transposed problem. Strip and column read correctly in
either order.

Direction. Right-to-left layouts are the left-to-right ones with the placement
mirrored (the images themselves are not flipped).

Breadth. The side that stays fixed while the layout grows (the width in row
order, the height in column order and for a strip) is the unit for gutter,
margin, caption size and corner radius.

Size limit. No image's linear size (square root of displayed area) should
exceed another's by more than max_scale. Among the layouts that make choices
(lines, tree) those within the limit always win; if none is, the smallest
excess wins, and only a strict limit refuses. In a strip or column the sizes
follow from the images' shapes alone, so there it is only reported.

Badness is 100 x the sum of squares of: the spread of log image sizes (standard
deviation); the log ratio of the canvas aspect to the one requested, or, without
a request, to the preferred 3:2 (only where the layout chooses its shape); the
relative change of the gutters and of the margins; the largest crop fraction;
and the fraction left empty by a ragged last line. Lower is better. Layouts are
ranked by distance from a requested aspect first, then by excess over the size
limit, then by badness (a strict size limit comes first instead).
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from .geometry import H, V, Frame, Glue, Infeasible, Layout, Leaf, Node, height, inner_width, place, reach, solve

FAMILIES = ("lines", "strip", "column", "tree")
ORDERS = ("rows", "columns", "either")
PREFERRED_ASPECT = 1.5   # soft preference used only when no target aspect is given


@dataclass(frozen=True)
class Badness:
    total: float
    spread: float
    aspect: float
    gutter: float
    margin: float
    crop: float
    empty: float

    def __str__(self):
        parts = [f"size spread {self.spread:.3f}"]
        if self.aspect:
            parts.append(f"aspect off {self.aspect:+.3f}")
        parts += [f"gutter change {self.gutter:.1%}", f"margin change {self.margin:.1%}",
                  f"crop {self.crop:.1%}", f"empty {self.empty:.1%}"]
        return f"badness {self.total:.2f}  ({', '.join(parts)}; 100 x sum of squares)"


def badness(layout: Layout, pref: float | None) -> Badness:
    """The aspect term measures the miss against a requested aspect, or else against
    the soft preference `pref` (None where the layout does not choose its shape)."""
    spread = statistics.pstdev(math.log(r.size) for r in layout.rects)
    goal = layout.target or pref
    aspect = math.log(layout.aspect / goal) if goal else 0.0
    crop = max(r.crop for r in layout.rects)
    terms = (spread, aspect, layout.gutter_change, layout.margin_change, crop, layout.empty)
    return Badness(100 * sum(t * t for t in terms), *terms)


def miss(layout: Layout) -> float:
    """How far (in log aspect) the canvas is from a requested aspect; 0 if none was requested."""
    return abs(math.log(layout.aspect / layout.target)) if layout.target else 0.0


def excess(layout: Layout, max_scale: float) -> float:
    """How far (in log size) the largest size ratio exceeds the limit; 0 if within it."""
    return max(0.0, math.log(layout.size_ratio) - math.log(max_scale))


@dataclass(frozen=True)
class Margins:
    nominal: tuple[float, float, float, float]   # left, right, top, bottom of the finished picture
    tol: float

    def frame(self, floor: float, transposed: bool) -> Frame:
        return Frame.make(self.nominal, self.tol, floor, transposed)


def build(family: str, leaves: list[Leaf], glue: Glue, margins: Margins,
          aspect: float | None = None, *, max_scale: float = 1.5, strict: bool = False,
          order: str = "rows", rtl: bool = False, row_height: float | None = None,
          ragged: bool = True, any_order: bool = False, seed: int = 0) -> Layout:
    """Candidates are ranked by: miss against a requested aspect, then excess over the
    size limit, then badness; with a strict limit the excess comes first and must be 0."""
    if family not in FAMILIES:
        raise ValueError(f"unknown layout {family!r}")
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}")
    if any_order and family != "tree":
        raise ValueError("--any-order applies only to the tree layout; the others keep the given order")
    pref = None if aspect is not None else PREFERRED_ASPECT
    rank = ((lambda l: (excess(l, max_scale), miss(l), badness(l, pref).total)) if strict else
            (lambda l: (miss(l), excess(l, max_scale), badness(l, pref).total)))
    if family in ("strip", "column"):
        layout = _oriented("column", "rows" if family == "column" else "columns", leaves, glue, margins,
                           aspect, pref, max_scale, rank, row_height, ragged, any_order, seed)
    else:
        found = [_oriented(family, o, leaves, glue, margins, aspect, pref, max_scale, rank,
                           row_height, ragged, any_order, seed)
                 for o in (("rows", "columns") if order == "either" else (order,))]
        layout = min(found, key=rank)
        if strict and excess(layout, max_scale) > 0:
            raise Infeasible(f"no layout found keeps image sizes within the strict --max-scale {max_scale:g} "
                             f"(the best differs by {layout.size_ratio:.2f}x); raise it, allow cropping, "
                             "or choose another layout")
    return layout.mirrored() if rtl else layout


def _oriented(family, order, leaves, glue, margins, aspect, pref, max_scale, rank,
              row_height, ragged, any_order, seed):
    """Solve in row order, transposing the problem and the result for column order."""
    frame = margins.frame(glue.gutter, order == "columns")
    if order == "rows":
        return _canonical(family, leaves, glue, frame, aspect, pref, max_scale, rank,
                          row_height, ragged, any_order, seed)
    flipped = [Leaf(l.index, 1.0 / l.aspect, l.crop, l.by, l.bx) for l in leaves]
    inv = lambda x: None if x is None else 1.0 / x
    return _canonical(family, flipped, glue, frame, inv(aspect), inv(pref), max_scale, rank,
                      row_height, ragged, any_order, seed).transposed()


def _canonical(family, leaves, glue, frame, aspect, pref, max_scale, rank, row_height, ragged, any_order, seed):
    if family == "column":
        tree = Node(V, tuple(leaves))
    elif family == "lines":
        tree = _lines(leaves, glue, frame, aspect, pref, max_scale, rank, row_height, ragged)
    else:
        # Lines are trees in row order too: the best full-width lines seed the search, and the
        # best lines with a thumb spot (which the search cannot express) remain a candidate.
        start = _lines(leaves, glue, frame, aspect, pref, max_scale, rank, None, False)
        searched = _search(leaves, glue, frame, aspect, pref, max_scale, any_order, seed, start)
        lines = _lines(leaves, glue, frame, aspect, pref, max_scale, rank, row_height, ragged)
        return min((solve(t, glue, frame, aspect) for t in (searched, lines)), key=rank)
    return solve(tree, glue, frame, aspect)


def _order_violations(t) -> int:
    """Stacks that are not the last item of their row; 0 means the tree reads unambiguously in row order."""
    if isinstance(t, Leaf):
        return 0
    bad = 0
    if t.cut == H:
        bad = sum(isinstance(c, Node) and c.cut == V for c in t.children[:-1])
    return bad + sum(_order_violations(c) for c in t.children)


def _lines(leaves, glue, frame, aspect, pref, max_scale, rank, row_height, ragged) -> Node:
    """Break the sequence into rows as TeX breaks a paragraph into lines.

    For a target linear size m, a row's cost is the sum over its images of
    (log(size / m))^2. Candidates are generated twice: once admitting only rows
    whose sizes lie within about a factor sqrt(max_scale) of m, and once without
    that restriction. Each solved candidate is ranked by its excess over the size
    limit, then by badness.
    A short last row may be set ragged, at the height that best matches m, but only
    if the space it leaves is a thumb spot: no wider than the widest image in that
    row. Otherwise it is stretched to the full width like the others.
    m is scanned, or derived from row_height.

    Bands are uniform across leaves, so a row's image height is one number and
    each row costs O(1) from prefix sums.
    """
    inner, g, n = inner_width(frame, 0.0), glue.gutter, len(leaves)
    bx, by = leaves[0].bx, leaves[0].by
    c = [0.5 * math.log(l.aspect) for l in leaves]          # log sqrt(aspect): size = sqrt(a) * h
    A, C, C2 = [0.0], [0.0], [0.0]
    for leaf, ci in zip(leaves, c):
        A.append(A[-1] + leaf.aspect)
        C.append(C[-1] + ci)
        C2.append(C2[-1] + ci * ci)
    # The band only generates candidates; the exact limit is applied to each solved layout.
    # Generating with a little slack keeps layouts that sit exactly on the limit (a 3:2
    # landscape beside a 2:3 portrait differ by exactly 1.5) from falling between grid points.
    # Candidates are also generated without the band, so there is always a best layout.
    halves = (0.5 * math.log(max_scale) + 0.05, math.inf) if max_scale < math.inf else (math.inf,)

    def full(i, j):
        """Image height of row i..j filled to the full width, or None if it cannot fit."""
        free = inner - (j - i - 1) * g - (j - i) * bx
        return free / (A[j] - A[i]) if free > 0 else None

    def breaks(lm, half):
        best = [0.0] + [math.inf] * n
        back = [None] * (n + 1)
        for j in range(1, n + 1):
            lo_c, hi_c, widest = math.inf, -math.inf, 0.0
            for i in range(j - 1, -1, -1):
                lo_c, hi_c, widest = min(lo_c, c[i]), max(hi_c, c[i]), max(widest, leaves[i].aspect)
                h = full(i, j)
                if h is None:
                    break
                k, s1, s2 = j - i, C[j] - C[i], C2[j] - C2[i]
                fill = None
                if ragged and j == n and i > 0:
                    hr = math.exp(lm - s1 / k)
                    used = (A[j] - A[i]) * hr + k * bx + (k - 1) * g
                    if hr < h and inner - used <= widest * hr + bx + g:      # a thumb spot at most
                        h, fill = hr, hr
                lh = math.log(h)
                if lo_c + lh < lm - half or hi_c + lh > lm + half:
                    continue
                d = lh - lm
                cost = s2 + 2 * d * s1 + k * d * d
                if best[i] + cost < best[j]:
                    best[j], back[j] = best[i] + cost, (i, fill)
        if back[n] is None:
            return None
        out, j = [], n
        while j > 0:
            i, fill = back[j]
            out.append((i, j, fill))
            j = i
        return tuple(reversed(out))

    def tree(bks):
        return Node(V, tuple(Node(H, tuple(leaves[i:j]), None if fill is None else fill + by)
                             for i, j, fill in bks))

    if row_height is not None:
        grid = [math.log(row_height) + statistics.fmean(c)]
    else:
        top = max(math.log(inner - bx) - ci for ci in c)                 # one image alone in a row
        one = full(0, n)
        bottom = min(c) + math.log(one) if one else top - math.log(4 * n)
        steps = 100
        grid = [bottom + (top - bottom) * k / (steps - 1) for k in range(steps)]

    best_key, best_tree, seen = None, None, set()
    for half in halves:
        for lm in grid:
            bks = breaks(lm, half)
            if bks is None or bks in seen:
                continue
            seen.add(bks)
            t = tree(bks)
            key = rank(solve(t, glue, frame, aspect))
            if best_key is None or key < best_key:
                best_key, best_tree = key, t
    return best_tree


def _search(leaves, glue, frame, aspect, pref, max_scale, any_order, seed, start=None) -> Node:
    """Annealed local search over binary slicing trees that read in row order.

    Scored by badness at nominal gutters, with steep penalties for a requested
    aspect beyond the slack's reach, sizes beyond max_scale, and stacks placed
    mid-row; only trees free of the latter are
    kept. Leaves keep their order unless any_order. Independent restarts (each
    effectively another seed) are run and the best kept; the first starts from
    `start` if given."""
    n = len(leaves)
    if n == 1:
        return Node(H, (leaves[0],))
    rng = random.Random(seed)

    def grow(seq):
        if len(seq) == 1:
            return seq[0]
        k = rng.randrange(1, len(seq))
        return (rng.choice((H, V)), grow(seq[:k]), grow(seq[k:]))

    def order(b):
        return [b] if isinstance(b, int) else order(b[1]) + order(b[2])

    def paths(b, p=()):
        return [] if isinstance(b, int) else [p] + paths(b[1], p + (1,)) + paths(b[2], p + (2,))

    def get(b, p):
        for step in p:
            b = b[step]
        return b

    def put(b, p, new):
        if not p:
            return new
        parts = list(b)
        parts[p[0]] = put(b[p[0]], p[1:], new)
        return tuple(parts)

    def relabel(b, m):
        return m.get(b, b) if isinstance(b, int) else (b[0], relabel(b[1], m), relabel(b[2], m))

    def mutate(b):
        r = rng.random()
        if any_order and r < 1 / 3:
            i, j = rng.sample(range(n), 2)
            return relabel(b, {i: j, j: i})
        p = rng.choice(paths(b))
        sub = get(b, p)
        if r < 2 / 3:
            return put(b, p, (H if sub[0] == V else V, sub[1], sub[2]))
        return put(b, p, grow(order(sub)))

    def node(b):
        if isinstance(b, int):
            return leaves[b]
        kids = []
        for ch in (b[1], b[2]):
            m = node(ch)
            kids.extend(m.children if isinstance(m, Node) and m.cut == b[0] else (m,))
        return Node(b[0], tuple(kids))

    def binary(t, counter):
        if isinstance(t, Leaf):
            counter[0] += 1
            return counter[0] - 1
        kids = [binary(ch, counter) for ch in t.children]
        b = kids[0]
        for k in kids[1:]:
            b = (t.cut, b, k)
        return b

    def cost(b):
        t = node(b)
        rects = []
        left, _, top, _ = frame.at(0.0)
        place(t, glue, 0.0, left, top, inner_width(frame, 0.0), rects)
        if any(r.w <= r.bx or r.h <= r.by for r in rects):
            return math.inf, False
        logs = [math.log(r.size) for r in rects]
        over = max(0.0, max(logs) - min(logs) - math.log(max_scale))
        if aspect is None:
            shape, off = math.log(height(t, glue, frame, 0.0) * pref), 0.0   # distance from the preference
        else:
            narrow, wide = reach(t, glue, frame)
            shape, off = 0.0, max(0.0, math.log(narrow / aspect), math.log(aspect / wide))
        wrong = _order_violations(t)
        return (100 * (statistics.pstdev(logs) ** 2 + shape ** 2)
                + 3000 * off + 1000 * over + 50 * wrong), wrong == 0

    best = None
    steps = max(600, 20 * n)
    for run in range(max(4, min(12, 480 // n))):
        seq = list(range(n))
        if any_order:
            rng.shuffle(seq)
        b = binary(start, [0]) if run == 0 and start is not None else grow(seq)
        cb, ok = cost(b)
        run_best = (b, cb) if ok else None
        for step in range(steps):          # annealing: accept worse moves early, only better ones late
            temp = 30 * (1 - step / steps)
            b2 = mutate(b)
            c2, ok2 = cost(b2)
            if c2 <= cb or (temp > 0 and rng.random() < math.exp((cb - c2) / temp)):
                b, cb = b2, c2
                if ok2 and (run_best is None or cb < run_best[1]):
                    run_best = (b, cb)
        if run_best and (best is None or run_best[1] < best[1]):
            best = run_best
    if best is None:
        raise Infeasible("the tree search found no layout that reads unambiguously in the requested order")
    return node(best[0])
