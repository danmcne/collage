# collage

Collages and storyboards that keep every image whole and in reading order.

Most collage tools crop images to fit a grid and scatter them in no particular
order. This one does neither. Images keep their aspect ratios unless you allow
cropping, the sequence you give is the sequence a reader follows, and the
program tells you, in one number, how good the compromise it found is.

```
python3 collage.py photos/ -o board.png
python3 collage.py photos/ -o board.png --layout tree --aspect 3:2 --crop-tol 0.05
python3 collage.py story.toml -o strip.jpg --layout strip --size x600
python3 collage.py photos/ -o print.tif --size 297x210mm --dpi 300 --radius 0.01
```

Requires Python 3.11 or later and [Pillow](https://python-pillow.org)
(`pip install -r requirements.txt`). There is nothing to install: clone the
repository and run `collage.py`. `python3 -m collage` works the same way.

## How it works

A layout is a **slicing tree**. Each leaf is an image; each internal node puts
its children side by side at a shared height, or stacks them at a shared width.
Given the width a subtree is allotted, its height is an exact affine function of
that width, and these compose, so a layout is computed in one pass up the tree
and placed in one pass down. No image is cropped and no space is wasted.

Reading the leaves left to right gives the **reading order**. Row order (left to
right, then top down, like a book or a comic) is unambiguous as long as a stack
of images appears only as the last item of a row: you finish the row, then read
the stack downward. Column order is the same rule transposed, and `--rtl`
mirrors the placement, not the images, for right-to-left reading.

Everything flexible is **glue**, in the sense TeX uses when setting a line: the
gutters, the margins, and any cropping you allow each have a nominal value and
bounds, and all of them move together with a single setting. A requested canvas
aspect is met by adjusting that setting; if the bounds cannot reach it, you get
the nearest shape reachable and a note saying so. The only empty space allowed
is a **thumb spot**: a short last line may stop early if the gap is no wider
than one of its own images.

Every run reports **badness**, on the model of TeX's: 100 times the sum of
squares of the spread of image sizes, the miss against the aspect sought, the
relative change of gutters and margins, the largest crop, and the empty
fraction. Lower is better, and the breakdown shows which compromise was made.

## Layouts

| `--layout` | Result |
|---|---|
| `lines` (default) | Images broken into lines, as TeX breaks a paragraph into lines |
| `strip` | One horizontal line, all images the same height |
| `column` | One vertical line, all images the same width |
| `tree` | Any slicing layout that reads correctly, found by search |

`--order rows` (the default) reads left to right then top down; `--order
columns` reads top down then left to right; `--order either` builds both and
keeps the better. `--any-order` lets the `tree` search reorder images, for when
composition matters more than sequence.

## Keeping sizes comparable

Images side by side share a height, so their displayed sizes differ by the
square root of their aspect ratios, whatever the layout does. `--max-scale`
bounds that: no image may exceed another by more than this factor in linear
size (the square root of displayed area). The default, 1.5, is a preference:
the best layout within it wins, and if none exists you still get the closest
one, with a note. Set it explicitly and it becomes strict, refusing instead.

Some shape mixes cannot meet a tight limit in a single line. A 4:3 and a 3:4
differ by 1.33, a 3:2 and a 2:3 by exactly 1.5, but a 16:9 and a 9:16 differ by
1.78, so a sequence that alternates between those needs either a larger
`--max-scale` or `--layout tree`, which can stack images instead.

## Captions and manifests

A manifest sets the order, and gives each image an optional caption and crop
tolerance:

```toml
[[image]]
path = "shots/01.jpg"          # relative to the manifest
caption = "The harbour at dawn"
crop = 0.05                    # may lose up to 5% of one side
```

Captions sit in a band below each image. The band is the same height everywhere,
so pictures in a row stay aligned, and its height in lines is found by laying
out, wrapping each caption to the width its image received, and growing the band
if any caption needs more room. Inputs may mix files, directories (their images,
sorted by name) and manifests, concatenated in the order given.

## Output

`--size` is repeatable: `2400` (pixel width), `x1600` (pixel height), or a
physical size such as `297x210mm`, `30x20cm`, `8x10in`, which also sets the
aspect and uses `--dpi`. The layout is computed once, resolution-independent,
and rendered at each size, so a screen preview and a print are the same
composition. Images rendered beyond their own pixels are reported, with the
effective dpi for print sizes.

Lengths — gutter, margins, corner radius, caption size — are fractions of the
layout's breadth: its width when reading in rows, its height in column order or
for a strip.

## Limits

`--max-memory` (1024 MB by default) is checked before anything is allocated,
and covers both the output canvas and the largest single decoded input, since
inputs are processed one at a time. `--max-images` (200) bounds layout time,
which grows with the image count rather than memory.

## Self-test

```
python3 collage.py --selftest
```

This checks the geometry and the rendering path on synthetic inputs: that
images never overlap, stay within the canvas, keep their aspect ratios within
the crop allowed, read in the requested order, and respect the margin rules. It
says nothing about how any particular set of photographs will lay out.

## Known limitations

- The `tree` search is heuristic. Against an exhaustive enumeration of small
  cases it finds the optimum often but not always, so a different `--seed` can
  give a better result.
- The `tree` search cannot express a thumb spot; the best `lines` layout is
  kept as a candidate to cover that.
- The soft 3:2 preference used when no aspect is requested ignores reading
  order, so `--order either` leans toward row order.
- Badness weights every term equally. If empty space, cropping or uneven sizes
  matter differently to you, that weighting is the thing to change.

## Licence

MIT. See [LICENSE](LICENSE).
