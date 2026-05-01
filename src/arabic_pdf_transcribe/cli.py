"""CLI entry point.

Phase 1 ships only the entry-point function so the ``arabic-pdf-transcribe``
console script registered in ``pyproject.toml`` resolves cleanly during
``pip install`` and during ``--help`` introspection. The real argument parser
and pipeline wiring land in phase 8 (per the implementation plan).

Calling ``main`` before phase 8 raises :class:`NotImplementedError` with a
message that points at the spec / plan; this is intentional. We never want a
half-wired CLI to silently produce empty or wrong output.
"""

from __future__ import annotations

from collections.abc import Sequence

_PHASE_8_MESSAGE = (
    "arabic-pdf-transcribe CLI is not implemented yet — it lands in plan phase 8 "
    "(see codev/plans/1-arabic-pdf-transcriber-extract.md). "
    "Phase 1 only ships the package skeleton and the license-audit harness."
)


def main(argv: Sequence[str] | None = None) -> int:
    raise NotImplementedError(_PHASE_8_MESSAGE)


__all__ = ["main"]
