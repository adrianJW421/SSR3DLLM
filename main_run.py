"""
Unified root entrypoint for baseline runtime modes.

Default mode delegates to `baseline/core/main_run.py` (standard train/test path).
Use `--entry interface` to delegate to `baseline/core/main_run_interface.py`.
"""

from __future__ import annotations

import sys

__all__ = ["run_train", "run_test", "get_parameters", "main"]


def _load_standard_module():
    from baseline.core import main_run as standard_module

    return standard_module


def __getattr__(name: str):
    """
    Backward-compatibility shim so existing imports still work:
    `from main_run import run_train, run_test`.
    """
    module = _load_standard_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'main_run' has no attribute {name!r}") from exc


def __dir__():
    return sorted(set(globals().keys()) | set(dir(_load_standard_module())))


def _pop_entry(argv: list[str]) -> tuple[str, list[str]]:
    """Extract `--entry` from argv while preserving other arguments."""
    entry = "standard"
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--entry":
            if i + 1 >= len(argv):
                raise SystemExit("[main_run] missing value for --entry (use standard|interface)")
            entry = argv[i + 1].strip()
            i += 2
            continue
        if token.startswith("--entry="):
            entry = token.split("=", 1)[1].strip()
            i += 1
            continue
        cleaned.append(token)
        i += 1
    if entry not in {"standard", "interface"}:
        raise SystemExit(f"[main_run] invalid --entry={entry!r} (use standard|interface)")
    return entry, cleaned


def _dispatch_main(argv: list[str] | None = None) -> None:
    """Dispatch to the selected baseline entry implementation."""
    if argv is None:
        argv = sys.argv[1:]
    entry, forwarded = _pop_entry(list(argv))

    if entry == "interface":
        from baseline.core.main_run_interface import main as interface_main

        interface_main(forwarded)
        return

    standard_main = _load_standard_module().main

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *forwarded]
        standard_main()
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> None:
    _dispatch_main(argv)


if __name__ == "__main__":  # pragma: no cover
    main()
