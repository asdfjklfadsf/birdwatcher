"""Primary Bird Watcher entrypoint.

Running ``python main.py`` uses the tracked-event runtime with broader BioCLIP
reranking and per-species cooldowns. The original implementation is preserved
in ``legacy_main.py`` for compatibility with existing imports and tests.
"""
from __future__ import annotations

import sys

import legacy_main as _legacy

# Preserve the existing public helper/model API for code that imports ``main``.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


def main() -> None:
    """Run the current tracked-event Bird Watcher runtime."""
    from birdwatcher_improved import main as improved_main

    improved_main()


# birdwatcher_improved historically imports ``main`` for shared helpers. Point
# that import at the preserved implementation so there is no circular import,
# while making its ``main()`` entrypoint delegate back to the current runtime.
_legacy.main = main
sys.modules["main"] = _legacy


if __name__ == "__main__":
    main()
