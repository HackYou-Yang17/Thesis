"""Compatibility alias.

`analysis/calibration/v13_crossings.py` imports `_bias_check`, while the
crossing-angle scripts import the same module as `bias_check`. Both names
refer to `lib/bias_check.py`; this shim keeps the original scripts unedited.
"""
from bias_check import *          # noqa: F401,F403
from bias_check import real_crossings   # noqa: F401
