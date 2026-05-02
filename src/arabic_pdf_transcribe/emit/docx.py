"""Word (.docx) emitter.

Converts an ordered, role-classified :class:`Region` stream into a
``python-docx`` document. The mapping mirrors :mod:`emit.markdown`
but uses Word's built-in styles instead of Markdown syntax.

Mapping (see plan section "Phase 7"):

* ``HEADING`` → built-in style ``Heading {level}``.
* ``PARAGRAPH`` → ``Normal``.
* ``LIST_ITEM`` → ``List Bullet`` or ``List Number``.
* ``TABLE`` → real Word table from ``region.table_grid``; one
  paragraph per cell.
* ``FIGURE`` / ``CAPTION`` → placeholder paragraph
  ``"Figure on page N"``; grouped caption text is appended.
* ``HEADER_FOOTER`` → suppressed.
* ``FAILURE_PLACEHOLDER`` → ``Quote`` style paragraph.
* ``UNKNOWN`` → ``Normal``.

The emitter is **deterministic**: no timestamps, no environment
data, no document author / company metadata. Output is suitable for
snapshot-style structural tests (open with ``python-docx``, walk
paragraphs, assert styles).

``python-docx`` is imported lazily — importing this module costs
nothing until :func:`emit_docx` is first called.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from arabic_pdf_transcribe.emit._bidi import add_rlm_if_needed
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

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from docx.document import Document as _DocxDocument


def emit_docx(
    regions: Iterable[Region],
    output_path: Path | str,
    *,
    normalisation: NormalisationForm = DEFAULT_FORM,
    apply_bidi: bool = True,
) -> None:
    """Render ``regions`` to a Word file at ``output_path``.

    The output file is overwritten if it exists. The function does
    not perform any network I/O.
    """
    from docx import Document

    document = Document()
    region_list = list(regions)
    caption_group_ids = _collect_paired_caption_group_ids(region_list)
    pending_caption_text: dict[str, str] = _collect_caption_texts(
        region_list, normalisation=normalisation, paired=caption_group_ids
    )

    for region in region_list:
        role = region.role
        if role is RegionRole.HEADER_FOOTER:
            continue
        if role is RegionRole.CAPTION and region.group_id in caption_group_ids:
            # Already rendered alongside the figure.
            continue
        if role is RegionRole.HEADING:
            _add_heading(
                document,
                region,
                normalisation=normalisation,
                apply_bidi=apply_bidi,
            )
        elif role is RegionRole.LIST_ITEM:
            _add_list_item(
                document,
                region,
                normalisation=normalisation,
                apply_bidi=apply_bidi,
            )
        elif role is RegionRole.TABLE:
            _add_table(document, region, normalisation=normalisation)
        elif role is RegionRole.FIGURE:
            _add_figure(
                document,
                region,
                pending_caption_text=pending_caption_text,
            )
        elif role is RegionRole.CAPTION:
            _add_caption(
                document,
                region,
                normalisation=normalisation,
                apply_bidi=apply_bidi,
            )
        elif role is RegionRole.FAILURE_PLACEHOLDER:
            _add_failure(document, region)
        else:  # PARAGRAPH / UNKNOWN
            _add_paragraph(
                document,
                region,
                normalisation=normalisation,
                apply_bidi=apply_bidi,
            )

    document.save(str(output_path))


# ---------------------------------------------------------------------------
# Pre-pass collectors
# ---------------------------------------------------------------------------


def _collect_paired_caption_group_ids(regions: Sequence[Region]) -> set[str]:
    """Group ids where BOTH a caption and figure exist.

    Captions are suppressed only when a paired figure consumes their
    text; orphan grouped captions still render so the text is not
    dropped.
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


def _collect_caption_texts(
    regions: Sequence[Region],
    *,
    normalisation: NormalisationForm,
    paired: set[str],
) -> dict[str, str]:
    """Return ``group_id → caption-text`` for paired (figure+caption) groups."""
    out: dict[str, str] = {}
    for region in regions:
        if (
            region.role is RegionRole.CAPTION
            and region.group_id is not None
            and region.group_id in paired
        ):
            out[region.group_id] = normalise_text(region.text, form=normalisation).strip()
    return out


# ---------------------------------------------------------------------------
# Per-role writers
# ---------------------------------------------------------------------------


def _prepare_text(text: str, *, normalisation: NormalisationForm, apply_bidi: bool) -> str:
    out = normalise_text(text, form=normalisation)
    if apply_bidi:
        out = add_rlm_if_needed(out)
    return out


def _add_heading(
    document: _DocxDocument,
    region: Region,
    *,
    normalisation: NormalisationForm,
    apply_bidi: bool,
) -> None:
    level = region.heading_level if region.heading_level is not None else 2
    level = max(1, min(level, 9))
    text = _prepare_text(region.text, normalisation=normalisation, apply_bidi=apply_bidi)
    document.add_paragraph(text, style=f"Heading {level}")


def _add_paragraph(
    document: _DocxDocument,
    region: Region,
    *,
    normalisation: NormalisationForm,
    apply_bidi: bool,
) -> None:
    text = _prepare_text(region.text, normalisation=normalisation, apply_bidi=apply_bidi)
    document.add_paragraph(text, style="Normal")


def _add_caption(
    document: _DocxDocument,
    region: Region,
    *,
    normalisation: NormalisationForm,
    apply_bidi: bool,
) -> None:
    text = _prepare_text(region.text, normalisation=normalisation, apply_bidi=apply_bidi)
    paragraph = document.add_paragraph(style="Normal")
    run = paragraph.add_run(text)
    run.italic = True


def _add_list_item(
    document: _DocxDocument,
    region: Region,
    *,
    normalisation: NormalisationForm,
    apply_bidi: bool,
) -> None:
    marker = region.list_marker
    raw_text = region.text
    if marker is not None and marker.raw_marker:
        stripped = raw_text.lstrip()
        if stripped.startswith(marker.raw_marker):
            leading_ws = raw_text[: len(raw_text) - len(stripped)]
            rest = stripped[len(marker.raw_marker) :]
            if rest.startswith(" "):
                rest = rest[1:]
            raw_text = leading_ws + rest
    text = _prepare_text(raw_text, normalisation=normalisation, apply_bidi=apply_bidi)
    style = "List Number" if marker is not None and marker.kind == "ordered" else "List Bullet"
    document.add_paragraph(text, style=style)


def _add_failure(document: _DocxDocument, region: Region) -> None:
    page_n = region.page_index + 1
    reason = region.failure_reason or "unknown"
    document.add_paragraph(f"Transcription failed (page {page_n}): {reason}", style="Quote")


def _add_figure(
    document: _DocxDocument,
    region: Region,
    *,
    pending_caption_text: dict[str, str],
) -> None:
    page_n = region.page_index + 1
    text = f"Figure on page {page_n}"
    if region.group_id is not None and region.group_id in pending_caption_text:
        caption = pending_caption_text[region.group_id]
        if caption:
            text = f"{text}: {caption}"
    document.add_paragraph(text, style="Normal")


def _add_table(
    document: _DocxDocument,
    region: Region,
    *,
    normalisation: NormalisationForm,
) -> None:
    grid: TableGrid | None = region.table_grid
    if grid is None or grid.n_rows == 0 or grid.n_cols == 0:
        return
    if region.meta.get("v1_table_simplification") is True:
        document.add_paragraph("(v1: merged cells flattened)", style="Normal")
    table = document.add_table(rows=grid.n_rows, cols=grid.n_cols)
    for r_idx, row in enumerate(grid.rows):
        for c_idx in range(grid.n_cols):
            cell_text = (
                normalise_text(row[c_idx].text, form=normalisation) if c_idx < len(row) else ""
            )
            table.cell(r_idx, c_idx).text = cell_text


__all__ = ["emit_docx"]
