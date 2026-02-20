"""
Compatibility wrapper for the baseline (hydra-free) configuration.

The real implementation lives in `baseline/core/config.py`. We keep a top-level
`config.py` so existing imports (`from config import ...`) remain stable.
"""

from baseline.core.config import *  # noqa: F401,F403

