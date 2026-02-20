from __future__ import annotations

"""
`utils` package (shared).

SSR3DLLM utilities live directly in this folder.
Baseline Grounded 3D-LLM utilities live under `baseline/core/utils/` and are
made importable via a search-path shim.
"""

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
_BASE = Path(__file__).resolve().parent.parent / "baseline" / "core" / "utils"
if _BASE.is_dir():
    __path__.append(str(_BASE))  # type: ignore[attr-defined]
