"""Role classification + heading-level inference.

Phase 6 deterministic, pure-logic refinement of :class:`RegionRole`
assignments. Modifies only ``role``, ``heading_level``,
``list_marker``, and ``group_id`` — never ``text``.

See :mod:`arabic_pdf_transcribe.roles.classify` for the public entry
point.
"""

from __future__ import annotations

from arabic_pdf_transcribe.roles.classify import classify_page

__all__ = ["classify_page"]
