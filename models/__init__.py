from __future__ import annotations

"""
`models` package (SSR3DLLM-first, baseline-compatible).

Release layout note:
  - SSR3DLLM modules live directly in this folder (e.g. `models/relation_field.py`).
  - Baseline Grounded 3D-LLM implementations live under `baseline/core/models/`.
    We keep `import models.*` stable by adding that directory to this package's
    search path.

Important:
  Keep this module lightweight. Do NOT import heavy backbones (e.g. MinkowskiEngine)
  at import time, otherwise `import models.metrics...` would fail in environments
  that only need evaluation utilities.
"""

from pathlib import Path
import importlib
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
_BASE = Path(__file__).resolve().parent.parent / "baseline" / "core" / "models"
if _BASE.is_dir():
    __path__.append(str(_BASE))  # type: ignore[attr-defined]

_LAZY_EXPORTS: dict[str, str] = {
    # Baseline configs reference these symbols as `models.<Name>`.
    # Keep them lazy to avoid importing heavy deps (e.g. MinkowskiEngine) unless needed.
    "Mask3DLang": "mask3d_lang",
    "Mask3D": "mask3d",
    "Res16UNet34C": "res16unet",
}


def __getattr__(name: str):
    mod_name = _LAZY_EXPORTS.get(name)
    if not mod_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    obj = getattr(mod, name)
    globals()[name] = obj
    return obj


__all__: list[str] = sorted(list(_LAZY_EXPORTS.keys()))
