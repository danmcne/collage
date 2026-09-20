#!/usr/bin/env python3
"""Entry point: run `python3 collage.py ...` (same as `python3 -m collage ...`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collage.cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:     # output piped into head, less, and the like
        sys.stderr.close()
        raise SystemExit(0)
