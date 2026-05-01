"""Unified document model — full schema lands in phase 2.

Phase 1 ships only the namespace so other modules can import it without a
forward reference. The real ``Region`` dataclass, ``RegionRole`` enum, and
companion types are defined in phase 2 (see ``codev/plans/1-…`` for details).
"""

from __future__ import annotations

REGION_SCHEMA_VERSION = "1"


__all__ = ["REGION_SCHEMA_VERSION"]
