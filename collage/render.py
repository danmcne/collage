"""Draw a layout at a chosen pixel width, and save it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .captions import LEADING, PAD, Captions, load_font
from .geometry import Layout

ORIENTATION = 0x0112


@dataclass(frozen=True)
class Source:
    path: Path
    width: int          # after EXIF orientation
    height: int
    crop: float | None  # per-image crop tolerance, if given
    caption: str | None = None
    format: str | None = None


def canvas_size(layout: Layout, px_width: int) -> tuple[int, int, float]:
    s = px_width / layout.width
    return round(layout.width * s), round(layout.height * s), s


def boxes(layout: Layout, s: float):
    """Pixel boxes of the images: (rect, x0, y0, width, height)."""
    for r in layout.rects:
        ix, iy, iw, ih = r.image
        x0, y0 = round(ix * s), round(iy * s)
        yield r, x0, y0, round((ix + iw) * s) - x0, round((iy + ih) * s) - y0


def peak_bytes(layout: Layout, sources: list[Source], px_width: int) -> tuple[int, int]:
    """Conservative estimate of peak memory: (for the canvas, for the worst single image).

    Images are decoded one at a time. A JPEG is decoded at the largest 1/2, 1/4 or
    1/8 reduction that still covers its box; other formats are decoded in full.
    Allowances: 8 bytes per canvas pixel (RGBA canvas, then an RGB copy and alpha
    mask when saving) and 10 bytes per decoded pixel (decode, orientation copy,
    RGBA conversion), plus 6 per box pixel (tile and corner mask)."""
    W, H, s = canvas_size(layout, px_width)
    worst = 0
    for r, _, _, bw, bh in boxes(layout, s):
        src = sources[r.index]
        m, reduce = max(bw, bh), 1
        if src.format == "JPEG":
            while reduce < 8 and min(src.width, src.height) / (2 * reduce) >= m:
                reduce *= 2
        decoded = (src.width // reduce) * (src.height // reduce)
        worst = max(worst, 10 * decoded + 6 * bw * bh)
    return 8 * W * H, worst


def probe(path: Path) -> tuple[int, int, str]:
    """Displayed size (after EXIF orientation) and format, read from the header only."""
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(ORIENTATION) in (5, 6, 7, 8):
            w, h = h, w
        return w, h, im.format


def rounded_mask(w: int, h: int, r: float, supersample: int = 4) -> Image.Image:
    """Opaque mask with anti-aliased rounded corners; only the corners are supersampled."""
    mask = Image.new("L", (w, h), 255)
    r = int(round(min(r, w / 2, h / 2)))
    if r <= 0:
        return mask
    big = Image.new("L", (r * supersample, r * supersample), 0)
    ImageDraw.Draw(big).pieslice((0, 0, 2 * r * supersample - 1, 2 * r * supersample - 1), 180, 270, fill=255)
    corner = big.resize((r, r), Image.Resampling.LANCZOS)
    T = Image.Transpose
    mask.paste(corner, (0, 0))
    mask.paste(corner.transpose(T.FLIP_LEFT_RIGHT), (w - r, 0))
    mask.paste(corner.transpose(T.FLIP_TOP_BOTTOM), (0, h - r))
    mask.paste(corner.transpose(T.ROTATE_180), (w - r, h - r))
    return mask


def render(layout: Layout, sources: list[Source], px_width: int,
           background: tuple | None, radius: float = 0.0, canvas_radius: float = 0.0,
           captions: Captions | None = None):
    """Return (image, upscales): upscales maps input index to how far its source
    pixels were enlarged, for every image rendered above its native resolution."""
    W, H, s = canvas_size(layout, px_width)
    fill = (0, 0, 0, 0) if background is None else tuple(background[:3]) + (255,)
    canvas = Image.new("RGBA", (W, H), fill)
    upscales = {}
    font = load_font(captions.font, captions.size * s) if captions else None
    draw = ImageDraw.Draw(canvas)
    for r, x0, y0, bw, bh in boxes(layout, s):
        ix, iy, iw, ih = r.image
        text = captions.texts.get(r.index) if captions else None
        if text:
            pitch = captions.size * LEADING * s
            top = (iy + ih + captions.size * PAD) * s
            for k, line in enumerate(captions.wrap(text, iw)):
                draw.text(((ix + iw / 2) * s, top + k * pitch), line, font=font,
                          fill=tuple(captions.color[:3]) + (255,), anchor="ma")
        if bw <= 0 or bh <= 0:
            continue
        src = sources[r.index]
        with Image.open(src.path) as im:
            m = max(bw, bh)
            im.draft("RGB", (m, m))
            im = ImageOps.exif_transpose(im).convert("RGBA")
        used = min(im.width, im.height * bw / bh) / im.width * src.width
        if bw / used > 1.01:
            upscales[r.index] = bw / used
        tile = ImageOps.fit(im, (bw, bh), Image.Resampling.LANCZOS)
        mask = rounded_mask(bw, bh, radius * s)
        tile.putalpha(Image.composite(tile.getchannel("A"), Image.new("L", (bw, bh), 0), mask))
        canvas.alpha_composite(tile, (x0, y0))
    if canvas_radius > 0:
        mask = rounded_mask(W, H, canvas_radius * s)
        canvas.putalpha(Image.composite(canvas.getchannel("A"), Image.new("L", (W, H), 0), mask))
    return canvas, upscales


def save(img: Image.Image, path: Path, dpi: int | None, flatten_onto: tuple) -> None:
    fmt = Image.registered_extensions().get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"unknown image format for {path}")
    kwargs = {"dpi": (dpi, dpi)} if dpi else {}
    if fmt in ("JPEG", "BMP") or img.getchannel("A").getextrema() == (255, 255):
        flat = Image.new("RGB", img.size, tuple(flatten_onto[:3]))
        flat.paste(img, mask=img.getchannel("A"))
        img = flat
    if fmt == "JPEG":
        kwargs.update(quality=95, subsampling=0)
    elif fmt == "TIFF":
        kwargs["compression"] = "tiff_lzw"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, fmt, **kwargs)
