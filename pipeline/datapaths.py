"""Where the hand-traced images live.

The scripts in this repository originally read the traced TIFFs from an absolute
path inside the environment they were developed in. That path is replaced here by a
single root so the pipeline runs anywhere.

The images are not shipped with the repository. Resolution order:
  1. the TRACED_ROOT environment variable, if set — point it at the traced set
  2. a `data/` directory at the repository root, as a fallback

Only the LOCATION of the input files changed. No measurement, threshold or constant
was touched, so every number the pipeline produces is unchanged.
"""
import os
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
ROOT = os.environ.get("TRACED_ROOT") or str(_repo / "data")
