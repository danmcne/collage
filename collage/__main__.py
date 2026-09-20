import sys

from .cli import main

try:
    raise SystemExit(main())
except BrokenPipeError:         # output piped into head, less, and the like
    sys.stderr.close()
    raise SystemExit(0)
