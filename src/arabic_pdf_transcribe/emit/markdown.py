"""Markdown emitter.

Converts an ordered, role-classified :class:`Region` stream into a
GFM-compatible Markdown document. Pure function: same input → same
output, no I/O, no environment-dependent data.

Mapping rules (see plan section "Phase 7"):

* ``HEADING`` → ``#`` × ``heading_level`` (defaults to 2 if missing).
* ``PARAGRAPH`` → plain paragraph (escape-safe).
* ``LIST_ITEM`` → ``- `` (bullet) or ``N. `` (ordered, ``N`` from
  ``list_marker.ordinal``); consecutive list items render in the
  same list block.
* ``TABLE`` → Markdown pipe-table from ``region.table_grid``;
  cell text is escape-safe; v1 cells with merged-cell flattening
  precede the table with ``<!-- v1: merged cells flattened -->``.
* ``FIGURE`` → ``![alt](#)``; when grouped via ``group_id`` with a
  ``CAPTION`` region, the caption text is the alt-text and the
  caption region is suppressed (it was emitted as part of the
  figure line).
* ``CAPTION`` (ungrouped) → italic paragraph (``*caption text*``).
* ``HEADER_FOOTER`` → suppressed.
* ``FAILURE_PLACEHOLDER`` → HTML comment with the page index and
  reason.
* ``UNKNOWN`` → emitted as a plain paragraph (best-effort).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from arabic_pdf_transcribe.emit._bidi import add_rlm_if_needed
from arabic_pdf_transcribe.emit._md_escape import (
    escape_paragraph,
    escape_table_cell,
)
from arabic_pdf_transcribe.emit._normalise import (
    DEFAULT_FORM,
    NormalisationForm,
    normalise_text,
)
from arabic_pdf_transcribe.regions import (
    Region,
    RegionRole,
    TableGrid,
)

_BLOCK_SEPARATOR = "\n\n"


def emit_markdown(
    regions: Iterable[Region],
    *,
    normalisation: NormalisationForm = DEFAULT_FORM,
    apply_bidi: bool = True,
) -> str:
    """Render ``regions`` as Markdown.

    Parameters
    ----------
    regions:
        Iterable of role-classified regions in reading order.
    normalisation:
        Unicode normalisation form. ``"NFC"`` (default) preserves
        Arabic presentation forms; ``"NFKC"`` collapses them — see
        :mod:`arabic_pdf_transcribe.emit._normalise`.
    apply_bidi:
        When ``True`` (default), Arabic-dominant paragraphs that
        contain LTR runs get a leading U+200F RLM mark for stable
        rendering on permissive bidi engines.
    """
    region_list = list(regions)
    caption_group_ids = _collect_paired_caption_group_ids(region_list)
    blocks: list[str] = []
    pending_list: list[str] = []
    for region in region_list:
        rendered_list_item = _render_list_item(region, normalisation, apply_bidi)
        if rendered_list_item is not None:
            pending_list.append(rendered_list_item)
            continue
        if pending_list:
            blocks.append("\n".join(pending_list))
            pending_list = []
        block = _render_block(
            region,
            caption_group_ids=caption_group_ids,
            region_list=region_list,
            normalisation=normalisation,
            apply_bidi=apply_bidi,
        )
        if block:
            blocks.append(block)
    if pending_list:
        blocks.append("\n".join(pending_list))
    return _BLOCK_SEPARATOR.join(blocks) + ("\n" if blocks else "")


# ---------------------------------------------------------------------------
# Block dispatch
# ---------------------------------------------------------------------------


def _collect_paired_caption_group_ids(regions: Sequence[Region]) -> set[str]:
    """Return ``group_id``s where BOTH a caption and figure exist.

    Captions are suppressed during the main pass only when there is a
    paired figure that will consume their text. An orphan grouped
    caption (no matching figure) still renders as a caption so the
    text is not dropped.
    """
    figure_ids = {
        region.group_id
        for region in regions
        if region.role is RegionRole.FIGURE and region.group_id is not None
    }
    caption_ids = {
        region.group_id
        for region in regions
        if region.role is RegionRole.CAPTION and region.group_id is not None
    }
    return figure_ids & caption_ids


def _render_block(
    region: Region,
    *,
    caption_group_ids: set[str],
    region_list: Sequence[Region],
    normalisation: NormalisationForm,
    apply_bidi: bool,
) -> str | None:
    role = region.role
    if role is RegionRole.HEADER_FOOTER:
        return None
    if role is RegionRole.CAPTION and region.group_id in caption_group_ids:
        # Already rendered as the figure's alt-text.
        return None
    if role is RegionRole.HEADING:
        return _render_heading(region, normalisation, apply_bidi)
    if role is RegionRole.TABLE:
        return _render_table(region, normalisation)
    if role is RegionRole.FIGURE:
        return _render_figure(region, region_list=region_list, normalisation=normalisation)
    if role is RegionRole.CAPTION:
        return _render_caption(region, normalisation, apply_bidi)
    if role is RegionRole.FAILURE_PLACEHOLDER:
        return _render_failure(region)
    # PARAGRAPH / UNKNOWN.
    return _render_paragraph(region, normalisation, apply_bidi)


# ---------------------------------------------------------------------------
# Per-role renderers
# ---------------------------------------------------------------------------


def _prepare_text(text: str, *, normalisation: NormalisationForm, apply_bidi: bool) -> str:
    out = normalise_text(text, form=normalisation)
    if apply_bidi:
        out = add_rlm_if_needed(out)
    return out


def _render_heading(region: Region, normalisation: NormalisationForm, apply_bidi: bool) -> str:
    level = region.heading_level if region.heading_level is not None else 2
    level = max(1, min(level, 6))
    text = _prepare_text(region.text, normalisation=normalisation, apply_bidi=apply_bidi)
    return f"{'#' * level} {escape_paragraph(text).strip()}"


def _render_paragraph(region: Region, normalisation: NormalisationForm, apply_bidi: bool) -> str:
    text = _prepare_text(region.text, normalisation=normalisation, apply_bidi=apply_bidi)
    return escape_paragraph(text)


def _render_caption(region: Region, normalisation: NormalisationForm, apply_bidi: bool) -> str:
    text = _prepare_text(region.text, normalisation=normalisation, apply_bidi=apply_bidi)
    inner = escape_paragraph(text).strip()
    if not inner:
        return ""
    return f"*{inner}*"


def _render_failure(region: Region) -> str:
    reason = region.failure_reason or "unknown"
    page_n = region.page_index + 1
    safe_reason = reason.replace("--", "- -")
    return f"<!-- transcription-failed: page {page_n} reason: {safe_reason} -->"


def _render_list_item(
    region: Region, normalisation: NormalisationForm, apply_bidi: bool
) -> str | None:
    if region.role is not RegionRole.LIST_ITEM:
        return None
    marker = region.list_marker
    text = _prepare_text(region.text, normalisation=normalisation, apply_bidi=apply_bidi)
    body = escape_paragraph(_strip_leading_marker(text, marker)).strip()
    if marker is not None and marker.kind == "ordered" and marker.ordinal is not None:
        return f"{marker.ordinal}. {body}"
    return f"- {body}"


def _strip_leading_marker(text: str, marker: object) -> str:
    """Drop the raw marker prefix from ``text`` if present.

    Phase 6 stamps the marker but does not edit the text. The
    Markdown emitter re-emits the marker in canonical form, so the
    raw prefix is stripped here to avoid double-marker output (e.g.
    ``- - text``).
    """
    if marker is None:
        return text
    raw = getattr(marker, "raw_marker", None)
    if not raw:
        return text
    stripped = text.lstrip()
    leading_ws = text[: len(text) - len(stripped)]
    if stripped.startswith(raw):
        rest = stripped[len(raw) :]
        # Eat one separator space if present.
        if rest.startswith(" "):
            rest = rest[1:]
        return leading_ws + rest
    return text


def _render_figure(
    region: Region,
    *,
    region_list: Sequence[Region],
    normalisation: NormalisationForm,
) -> str:
    page_n = region.page_index + 1
    alt = f"figure on page {page_n}"
    if region.group_id is not None:
        for other in region_list:
            if other.role is RegionRole.CAPTION and other.group_id == region.group_id:
                cap_text = normalise_text(other.text, form=normalisation).strip()
                if cap_text:
                    alt = cap_text
                break
    safe_alt = alt.replace("[", "\\[").replace("]", "\\]")
    return f"![{safe_alt}](#)"


def _render_table(region: Region, normalisation: NormalisationForm) -> str:
    grid: TableGrid | None = region.table_grid
    if grid is None or grid.n_rows == 0 or grid.n_cols == 0:
        return ""
    n_cols = grid.n_cols
    rows: list[str] = []
    for row in grid.rows:
        cells = [escape_table_cell(normalise_text(cell.text, form=normalisation)) for cell in row]
        # Pad short rows so every output row has the same column count.
        while len(cells) < n_cols:
            cells.append("")
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    header_row = rows[0]
    separator = "| " + " | ".join(["---"] * n_cols) + " |"
    body_rows = rows[1:]
    table_lines = [header_row, separator, *body_rows]
    table = "\n".join(table_lines)
    if region.meta.get("v1_table_simplification") is True:
        return "<!-- v1: merged cells flattened -->\n" + table
    return table


__all__ = ["emit_markdown"]
