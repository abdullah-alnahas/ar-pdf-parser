"""Heading levels, list markers, header/footer pruning, caption-figure linkage.

Pure deterministic role refinement. The phase 4 layout detector
emits roles that the model recognises (HEADING, PARAGRAPH, LIST_ITEM,
TABLE, FIGURE, CAPTION, HEADER_FOOTER); the phase 2 native extractor
emits looser roles (mostly PARAGRAPH and TABLE). This module produces
the unified, phase-7-emitter-ready output:

* **Heading levels.** Bin the available size signal into 3 quantiles
  → H1 (largest), H2 (middle), H3 (smallest). When no size signal
  exists for a region (no font sizes, no region heights) the
  region's ``heading_level`` is set to ``2`` so the emitter renders
  it as ``##``. This is the spec's "single, consistent rule" in the
  Resolved Decisions section.

* **List markers.** Detect bullet / numbered prefixes in the
  region's leading whitespace-stripped text. Supports Arabic
  bullets (`-`, `•`, `–`, `*`) and ordered markers in both Arabic
  digits (٠-٩) and Western digits (0-9). Captures the original
  raw marker on ``Region.list_marker.raw_marker`` so phase 7's
  emitter can preserve the document's style.

* **Header / footer pruning.** Regions whose vertical centre lies
  in the top or bottom ``HEADER_FOOTER_BAND_FRACTION`` of the page
  height (default 5 %) are reclassified to
  :data:`RegionRole.HEADER_FOOTER`. Phase 7's emitter suppresses
  these by default. The pruner is conservative: it only down-
  classifies ``PARAGRAPH`` and ``UNKNOWN`` regions; existing
  HEADING / TABLE / FIGURE / CAPTION roles are never demoted.

* **Caption-figure linkage.** When a CAPTION region appears within
  ``CAPTION_LINKAGE_FRACTION * page_height`` of a FIGURE region's
  bottom edge AND overlaps the figure horizontally, both regions
  are assigned the same generated ``group_id``. Ungrouped captions
  keep ``group_id=None`` and emit as standalone italic paragraphs
  in phase 7.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import StatisticsError, quantiles

from arabic_pdf_transcribe.regions import (
    ListMarker,
    Region,
    RegionRole,
    RegionSource,
)

HEADER_FOOTER_BAND_FRACTION = 0.05
CAPTION_LINKAGE_FRACTION = 0.05


@dataclass(frozen=True, slots=True)
class ClassifyConfig:
    """Tuneable knobs for the role classifier.

    Defaults match the spec; phase 9 may revisit
    ``header_footer_band_fraction`` after corpus tuning.
    """

    rtl: bool = True
    header_footer_band_fraction: float = HEADER_FOOTER_BAND_FRACTION
    caption_linkage_fraction: float = CAPTION_LINKAGE_FRACTION
    prune_header_footer: bool = True


def classify_page(
    regions: Sequence[Region],
    page_width: float,
    page_height: float,
    *,
    config: ClassifyConfig | None = None,
) -> list[Region]:
    """Return regions with refined roles, heading levels, list markers, group ids.

    Pure: same input → same output. Region order is preserved (the
    caller should run :func:`reorder` first if it wants reading
    order); this function never reorders.
    """
    cfg = config or ClassifyConfig()
    _ = page_width  # reserved for column-aware classification (phase 9)
    if not regions:
        return []
    size_signals = _collect_size_signals(regions)
    quantile_thresholds = _compute_heading_quantiles(size_signals)
    out: list[Region] = []
    for region in regions:
        new_region = region
        # Heading level inference.
        if region.role is RegionRole.HEADING:
            new_region = new_region.with_heading_level(
                _heading_level_for(region, quantile_thresholds)
            )
        # List-item detection.
        new_region = _classify_list(new_region)
        # Header / footer prune.
        if cfg.prune_header_footer:
            new_region = _maybe_prune_header_footer(
                new_region,
                page_height=page_height,
                band_fraction=cfg.header_footer_band_fraction,
            )
        out.append(new_region)
    out = _link_caption_figure(out, page_height=page_height, fraction=cfg.caption_linkage_fraction)
    return out


# ---------------------------------------------------------------------------
# Heading-level inference
# ---------------------------------------------------------------------------


def _collect_size_signals(regions: Iterable[Region]) -> list[float]:
    """Return the available size signal per HEADING region.

    Native pages put a ``font_size`` entry in ``Region.meta``; ML
    pages have only the bbox height. We use the metadata first, fall
    back to height. Regions without either contribute nothing — the
    quantile boundaries are computed over what we have.
    """
    signals: list[float] = []
    for region in regions:
        if region.role is not RegionRole.HEADING:
            continue
        signal = _size_signal_for(region)
        if signal is not None:
            signals.append(signal)
    return signals


def _size_signal_for(region: Region) -> float | None:
    """Return the size signal for heading binning, source-sensitive.

    Spec rule (phase 6 plan + Resolved Decisions):
    * Native path uses native font-size signals ONLY. Native headings
      without ``font_size`` meta fall back to H2 — height is NOT a
      substitute on the native path because PDF text-extractor
      bboxes track glyph extents, not visual heading prominence,
      and would mis-rank a long heading line as taller than a
      short one.
    * ML / OCR path has no font-size signal at all, so region
      height (from the layout detector's bbox) is the only
      available proxy.
    """
    meta_value = region.meta.get("font_size")
    if isinstance(meta_value, int | float) and meta_value > 0:
        return float(meta_value)
    if region.source is RegionSource.NATIVE:
        return None
    height = region.bbox.height
    if height > 0:
        return float(height)
    return None


def _compute_heading_quantiles(signals: list[float]) -> tuple[float, float] | None:
    """Return ``(q1_3, q2_3)`` thresholds or ``None`` when input is too sparse."""
    if len(signals) < 3:
        return None
    try:
        cuts = quantiles(signals, n=3, method="exclusive")
    except StatisticsError:  # pragma: no cover — only fires for n=2 sample
        return None
    return (cuts[0], cuts[1])


def _heading_level_for(region: Region, thresholds: tuple[float, float] | None) -> int:
    """Return 1 / 2 / 3 for a HEADING region.

    Spec rule: when no size signal exists at all, every heading is
    H2. When a size signal exists, bin into three quantile buckets
    and map (largest → H1, middle → H2, smallest → H3).
    """
    signal = _size_signal_for(region)
    if signal is None or thresholds is None:
        return 2
    q1_3, q2_3 = thresholds
    if signal <= q1_3:
        return 3
    if signal <= q2_3:
        return 2
    return 1


# ---------------------------------------------------------------------------
# List-item detection
# ---------------------------------------------------------------------------


_BULLET_GLYPHS = (
    "-",
    "•",  # • bullet
    "–",  # en dash
    "*",
    "—",  # em dash
)
# Ordered markers: digits (Western 0-9 or Arabic-Indic U+0660-U+0669)
# followed by '.' or ')'. Arabic-Indic range expressed as Unicode
# escapes so ruff's ambiguous-character lint (RUF001) does not fire —
# the visual similarity between U+0660..U+0669 and Western punctuation
# is the exact reason we want to match these characters here.
# Western digits 0-9 plus Arabic-Indic digits U+0660-U+0669.
_ORDERED_RE = re.compile(r"^(?P<num>[0-9٠-٩]+)\s*(?P<sep>[.)])\s+")


def _classify_list(region: Region) -> Region:
    """Detect a list marker on the region's leading text and stamp it."""
    if region.role not in (RegionRole.LIST_ITEM, RegionRole.PARAGRAPH):
        return region
    text = region.text.lstrip()
    if not text:
        return region
    # Bullet markers.
    for glyph in _BULLET_GLYPHS:
        if text.startswith(glyph + " ") or text == glyph:
            marker = ListMarker(kind="bullet", raw_marker=glyph)
            return region.with_role(RegionRole.LIST_ITEM).with_list_marker(marker)
    # Ordered markers.
    match = _ORDERED_RE.match(text)
    if match is not None:
        ordinal = _digits_to_int(match.group("num"))
        raw_marker = match.group(0).strip()
        marker = ListMarker(kind="ordered", ordinal=ordinal, raw_marker=raw_marker)
        return region.with_role(RegionRole.LIST_ITEM).with_list_marker(marker)
    return region


def _digits_to_int(digits: str) -> int:
    """Convert a string of mixed Arabic-Indic / Western digits to int."""
    western_chars: list[str] = []
    for ch in digits:
        try:
            western_chars.append(str(unicodedata.digit(ch)))
        except (ValueError, TypeError):
            western_chars.append(ch)
    return int("".join(western_chars))


# ---------------------------------------------------------------------------
# Header / footer prune
# ---------------------------------------------------------------------------


def _maybe_prune_header_footer(
    region: Region, *, page_height: float, band_fraction: float
) -> Region:
    """Reclassify regions in the top/bottom band as HEADER_FOOTER.

    Conservative: only acts on PARAGRAPH and UNKNOWN. Existing
    HEADING / TABLE / FIGURE / CAPTION / LIST_ITEM stay put — their
    role was assigned by an upstream signal stronger than position.
    """
    if region.role not in (RegionRole.PARAGRAPH, RegionRole.UNKNOWN):
        return region
    if page_height <= 0:
        return region
    cy = 0.5 * (region.bbox.y0 + region.bbox.y1)
    band = band_fraction * page_height
    if cy < band or cy > page_height - band:
        return region.with_role(RegionRole.HEADER_FOOTER)
    return region


# ---------------------------------------------------------------------------
# Caption-figure linkage
# ---------------------------------------------------------------------------


def _link_caption_figure(
    regions: list[Region], *, page_height: float, fraction: float
) -> list[Region]:
    """Pair captions to nearby figures via ``group_id``.

    A caption pairs with the figure whose bottom edge is closest to
    the caption's top edge AND whose horizontal range overlaps the
    caption's. Pairing is one-to-one: each figure pairs with at
    most one caption (the closest), and each caption pairs with at
    most one figure.
    """
    if page_height <= 0:
        return regions
    threshold = fraction * page_height
    out: list[Region] = list(regions)
    figure_indices = [i for i, r in enumerate(out) if r.role is RegionRole.FIGURE]
    caption_indices = [i for i, r in enumerate(out) if r.role is RegionRole.CAPTION]
    used_captions: set[int] = set()
    for fig_idx in figure_indices:
        figure = out[fig_idx]
        best_cap_idx: int | None = None
        best_dy = float("inf")
        for cap_idx in caption_indices:
            if cap_idx in used_captions:
                continue
            caption = out[cap_idx]
            if caption.group_id is not None:
                continue
            if not _bboxes_overlap_x(figure, caption):
                continue
            dy = caption.bbox.y0 - figure.bbox.y1
            if dy < 0 or dy > threshold:
                continue
            if dy < best_dy:
                best_dy = dy
                best_cap_idx = cap_idx
        if best_cap_idx is None:
            continue
        gid = uuid.uuid5(
            uuid.NAMESPACE_OID,
            f"figcap:{figure.page_index}:{figure.bbox.x0}:{figure.bbox.y0}",
        ).hex
        out[fig_idx] = figure.with_group_id(gid)
        out[best_cap_idx] = out[best_cap_idx].with_group_id(gid)
        used_captions.add(best_cap_idx)
    return out


def _bboxes_overlap_x(a: Region, b: Region) -> bool:
    return not (a.bbox.x1 < b.bbox.x0 or b.bbox.x1 < a.bbox.x0)


__all__ = ["ClassifyConfig", "classify_page"]
