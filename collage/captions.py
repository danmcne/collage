"""Captions: a band of wrapped text below each image.

The band has the same height under every image, so pictures in a row stay
aligned and captions line up. That height is a whole number of lines, found by
iteration: lay out with the current line count, wrap every caption to the width
its image received, and if some caption needs more lines, grow the band and lay
out again. The count never decreases, so this ends: when every caption fits,
or with a refusal once the allowed maximum is passed.
Wrapping is computed in layout units, so the renderer draws exactly the lines
that were checked.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import ImageFont

from .geometry import Infeasible, Layout

REF_PX = 200        # measuring size; lengths scale linearly with font size
LEADING = 1.25      # line pitch, in font sizes
PAD = 0.4           # space between image and first line, in font sizes


def load_font(path: str | None, px: float):
    px = max(1, round(px))
    return ImageFont.truetype(path, px) if path else ImageFont.load_default(px)


@dataclass
class Captions:
    texts: dict[int, str]    # input index -> caption
    size: float              # font size, fraction of breadth
    font: str | None = None
    color: tuple = (32, 32, 32)
    max_lines: int = 3

    def __post_init__(self):
        self._ref = load_font(self.font, REF_PX)

    def __bool__(self):
        return bool(self.texts)

    def band(self, lines: int) -> float:
        return self.size * (PAD + LEADING * lines) if lines else 0.0

    def wrap(self, text: str, width: float) -> list[str] | None:
        """Greedy word wrap to `width` layout units; None if a single word is too long."""
        scale = self.size / REF_PX
        fits = lambda s: self._ref.getlength(s) * scale <= width
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}" if line else word
            if fits(trial):
                line = trial
            elif not fits(word):
                return None
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    def lines_needed(self, layout: Layout) -> int:
        need = 0
        for r in layout.rects:
            text = self.texts.get(r.index)
            if not text:
                continue
            lines = self.wrap(text, r.w - r.bx)
            if lines is None:
                raise Infeasible(f"a word in caption {r.index + 1} ({text[:30]!r}) is wider than its image; "
                                 "shorten it or reduce --caption-size")
            need = max(need, len(lines))
        return need

    def fit(self, lay_out) -> tuple[Layout, int]:
        """Run lay_out(band) until the band holds every caption."""
        lines = 1 if self else 0
        while True:
            layout = lay_out(self.band(lines))
            need = self.lines_needed(layout) if self else 0
            if need <= lines:
                return layout, lines
            if need > self.max_lines:
                raise Infeasible(f"captions need {need} lines, above --caption-lines {self.max_lines}; "
                                 "raise it, reduce --caption-size, or shorten the captions")
            lines = need
