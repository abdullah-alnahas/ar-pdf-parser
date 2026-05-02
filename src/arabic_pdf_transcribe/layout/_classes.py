"""Mapping from the detector's class labels to the project's ``RegionRole``.

Phase 4 ships ``cmarkea/dit-base-layout-detection`` (Apache-2.0,
BeitForSemanticSegmentation; 12 classes). The mapping below is the only
place in the codebase that knows that model's label set. Phase 5's OCR
adapter and phase 6's role classifier consume :class:`RegionRole` values
exclusively, so swapping the layout model is a one-file change here.

The mapping is intentionally conservative: classes the spec carves out
explicitly (heading / list / table / figure / caption / paragraph /
header-footer) get their own ``RegionRole``. Anything ambiguous falls
through to :data:`RegionRole.UNKNOWN`, which phase 6 handles
deterministically.
"""

from __future__ import annotations

from arabic_pdf_transcribe.regions import RegionRole

# ``DiTLayoutLabels`` carries the upstream label strings verbatim — the same
# strings the model's ``id2label`` config reports. Keeping them as a flat tuple
# (instead of an enum mirrored from the model) avoids drift when phase 9
# re-tunes the corpus and possibly switches to a different revision: the
# adapter looks up the label string and falls through if unrecognised.
DIT_LAYOUT_LABELS: tuple[str, ...] = (
    "Background",
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)

# ``Background`` is intentionally absent: it does not produce a Region.
# ``Title`` maps to HEADING with ``heading_level=1``; ``Section-header`` maps
# to HEADING with ``heading_level=2`` (capped at H3 per spec; no class in this
# model produces an H3 directly — phase 6 may demote based on font size, but
# v1 does not on the ML path).
LABEL_TO_ROLE: dict[str, RegionRole] = {
    "Caption": RegionRole.CAPTION,
    "Footnote": RegionRole.HEADER_FOOTER,
    # ``Formula`` is mapped to PARAGRAPH for Arabic-first usage. The
    # default DiT model (cmarkea/dit-base-layout-detection) was trained
    # on English research papers and frequently mislabels justified
    # Arabic body text as ``Formula``. Mapping to UNKNOWN previously
    # caused phase 7's emitter to drop body text silently — the worst
    # failure mode. Mathematical formulas are rare in general Arabic
    # books, so the false-positive cost (lost text) outweighs the
    # false-negative cost (a real formula rendered as plain text).
    # See issue #16 root cause #2.
    "Formula": RegionRole.PARAGRAPH,
    "List-item": RegionRole.LIST_ITEM,
    "Page-footer": RegionRole.HEADER_FOOTER,
    "Page-header": RegionRole.HEADER_FOOTER,
    "Picture": RegionRole.FIGURE,
    "Section-header": RegionRole.HEADING,
    "Table": RegionRole.TABLE,
    "Text": RegionRole.PARAGRAPH,
    "Title": RegionRole.HEADING,
}

LABEL_HEADING_LEVEL: dict[str, int] = {
    "Title": 1,
    "Section-header": 2,
}

DROPPED_LABELS: frozenset[str] = frozenset({"Background"})


def role_for_label(label: str) -> RegionRole:
    """Return the project ``RegionRole`` for an upstream label.

    Unknown labels fall through to :data:`RegionRole.UNKNOWN`. The
    adapter logs a warning when this happens so the gap surfaces in
    phase 9's corpus tuning.
    """
    return LABEL_TO_ROLE.get(label, RegionRole.UNKNOWN)


def heading_level_for_label(label: str) -> int | None:
    """Return ``1``/``2`` for HEADING labels; ``None`` otherwise."""
    return LABEL_HEADING_LEVEL.get(label)
