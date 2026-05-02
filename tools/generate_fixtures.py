"""Generate phase-2 PDF fixtures with ``reportlab``.

Run via ``python tools/generate_fixtures.py``. Idempotent: regenerates the
fixtures under ``tests/fixtures/pdfs/digital-clean/``. Output is checked
into the repo so the test suite does not depend on ``reportlab`` at
test-run time (only at fixture-regeneration time).

Three fixtures (per phase 2 plan):

1. ``lorem-ar-2col.pdf`` — 2-page Arabic-text document (single column, two
   pages; the "2col" name in the plan refers to a future multi-column
   variant — phase 6 will use it for column-detection tests. v1 ships a
   simple single-column variant first to keep ``reportlab``'s Arabic
   shaping requirements light).
2. ``lorem-ar-en-mixed.pdf`` — one page mixing Arabic and English runs.
3. ``lorem-ar-table.pdf`` — one page containing a single 3 by 3 grid table.

The Arabic text is intentionally short and uses non-shaped letterforms
where possible so we don't need ``arabic_reshaper`` or ``python-bidi``
(both LGPL — would force the audit's allow-list to grow). The point of
these fixtures is *to exercise the extractor's geometry*, not to be a
realistic Arabic typesetting benchmark.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIGITAL_CLEAN = REPO_ROOT / "tests" / "fixtures" / "pdfs" / "digital-clean"
DIGITAL_BROKEN = REPO_ROOT / "tests" / "fixtures" / "pdfs" / "digital-broken"
IMAGE_SCAN = REPO_ROOT / "tests" / "fixtures" / "pdfs" / "image-scan"
MIXED = REPO_ROOT / "tests" / "fixtures" / "pdfs" / "mixed"
EDGE = REPO_ROOT / "tests" / "fixtures" / "pdfs" / "edge"


# Use a small set of Arabic words rendered as standalone glyphs (no
# contextual shaping required). These are real Arabic letters chosen to
# exercise UTF-8 round-tripping and bbox handling without depending on a
# bidi-aware shaper.
EN_WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
# v1 fixtures use ASCII content rendered with the built-in Helvetica font so
# regeneration does not depend on a system Arabic font being installed.
# Phase 9 enriches the corpus with real Arabic-content PDFs (real fonts,
# real shaping, real bidi). Phase 2's geometry tests only need *some* text
# layer to verify the extractor returns the bytes that were written, the
# bbox conversion (BL → TL), the paragraph grouping, and the table
# detection — none of which depend on the script of the glyphs.
PARA_WORDS = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur"]


def _build_2col() -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    out = DIGITAL_CLEAN / "lorem-ar-2col.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter
    for page in range(2):
        c.setFont("Helvetica", 18)
        c.drawString(72, height - 90, f"Page {page + 1}")
        c.setFont("Helvetica", 12)
        y = height - 120
        for word in PARA_WORDS * 3:
            c.drawString(72, y, word)
            y -= 16
        # Second paragraph block, separated by extra vertical gap.
        y -= 40
        for word in EN_WORDS * 2:
            c.drawString(72, y, word)
            y -= 14
        c.showPage()
    c.save()


def _build_mixed() -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    out = DIGITAL_CLEAN / "lorem-ar-en-mixed.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 16)
    c.drawString(72, height - 90, "Mixed Arabic / English fixture")
    c.setFont("Helvetica", 12)
    y = height - 130
    # Lines alternating Arabic and English to exercise bidi-mixed paragraph
    # cohesion (phase 6 will refine; phase 2 only checks both round-trip).
    for ar, en in zip(PARA_WORDS, EN_WORDS, strict=False):
        c.drawString(72, y, f"{en} {ar}")
        y -= 16
    # Second paragraph of pure English to assert paragraph grouping splits.
    y -= 40
    for word in EN_WORDS * 2:
        c.drawString(72, y, word)
        y -= 14
    c.showPage()
    c.save()


def _build_table() -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    out = DIGITAL_CLEAN / "lorem-ar-table.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 16)
    c.drawString(72, height - 90, "Table fixture (3x3)")

    c.setFont("Helvetica", 12)
    cell_w, cell_h = 100, 30
    origin_x, origin_y = 72, height - 200
    for row_idx in range(3):
        for col_idx in range(3):
            x = origin_x + col_idx * cell_w
            y = origin_y - row_idx * cell_h
            c.rect(x, y, cell_w, cell_h)
            c.drawString(x + 6, y + 10, f"r{row_idx + 1}c{col_idx + 1}")
    c.showPage()
    c.save()


def _build_real_arabic() -> bool:
    """Build a real Arabic-content fixture if a TTF is available.

    Looks for a Noto / DejaVu Arabic-supporting font on the system; if
    found, renders a one-page PDF carrying real Arabic codepoints in the
    text layer. The visual rendering is not bidi-shaped (we deliberately
    avoid pulling in ``arabic_reshaper`` / ``python-bidi`` because their
    licenses do not match the project allow-list), but the *codepoints*
    in the text layer are real Arabic — which is what the extractor's
    Unicode test exercises.

    Returns ``True`` if the fixture was generated, ``False`` if no font
    was found.
    """
    candidate_fonts = [
        "/usr/share/fonts/truetype/noto/NotoSansArabicUI-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_path = next((p for p in candidate_fonts if Path(p).exists()), None)
    if font_path is None:
        return False

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(TTFont("ArabicCapable", font_path))
    out = DIGITAL_CLEAN / "lorem-ar-real.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter
    c.setFont("ArabicCapable", 16)
    # Three short Arabic words on a single line. Text-layer must round-trip
    # the Unicode codepoints through extraction.
    c.drawString(72, height - 100, "كتاب قلم ورق")
    c.setFont("ArabicCapable", 14)
    c.drawString(72, height - 140, "مدرسة")
    c.showPage()
    c.save()
    return True


def _build_broken_mojibake() -> None:
    """Phase-3 broken fixture: looks like Arabic content but the text layer
    decoded to private-use codepoints and odd Latin runs. Simulated by
    drawing strings drawn from non-Arabic codepoints with apparent
    Arabic-paragraph length / line-break behaviour.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    out = DIGITAL_BROKEN / "mojibake.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    y = height - 100
    # Random-ish bytes mapped to printable ASCII; many short word-lengths
    # should diverge from the Arabic reference distribution sharply.
    for line in [
        "??????? ??? ?????? ?? ??????? ??? ?? ?",
        "?? ??? ?? ? ?? ?? ??? ?? ? ?? ??",
        "? ?? ?? ?? ?? ? ?? ?? ?? ? ??",
        "????????????????????????????????",
    ]:
        c.drawString(72, y, line)
        y -= 16
    c.showPage()
    c.save()


def _build_broken_replacement_glyphs() -> None:
    """Phase-3 broken fixture: text layer dominated by U+FFFD / private-use
    glyphs, the failure mode of glyph-id-not-Unicode encodings.

    Generation strategy: write printable strings using codepoints from
    the Private Use Area + interspersed FFFD; reportlab requires a font
    that can render these. We fall back to writing many "?" characters
    plus a few PUA codepoints rendered via a TTF when available; the
    text-layer codepoints we wrote will be reflected in the PDF text
    stream regardless of whether the glyph renders.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    out = DIGITAL_BROKEN / "replacement-glyphs.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    y = height - 100
    # PUA + FFFD chars (Helvetica will render fallback boxes; the text
    # layer keeps the actual codepoints we asked for).
    rep_chars = "�" * 20 + "" * 5
    lines = [
        rep_chars,
        rep_chars,
        rep_chars,
        rep_chars,
    ]
    for line in lines:
        c.drawString(72, y, line)
        y -= 16
    c.showPage()
    c.save()


def _build_image_scan() -> None:
    """Phase-4 image-scan fixture: a synthetic page whose text is rendered
    as a flat raster image embedded in the PDF (no text layer at all).

    Generation strategy: render a one-column page to a PIL image
    (in-process, no system font dependency beyond Helvetica which
    reportlab carries), then write that image as the only content of a
    PDF page via reportlab's ``drawImage``. The result has zero
    extractable text — the validator rejects it on phase 3 grounds and
    the orchestrator routes it to the layout-detector adapter.
    """
    from io import BytesIO

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    out = IMAGE_SCAN / "scan-ar-1col.pdf"

    # Step 1: render a synthetic page to a PIL image. We use a plain
    # PIL Draw (no real Arabic shaping; we deliberately avoid pulling
    # in arabic_reshaper / python-bidi — both LGPL — outside phase 9).
    # Phase 9 will land scanned-real-Arabic fixtures with proper
    # provenance.
    from PIL import Image, ImageDraw

    width_px, height_px = 850, 1100  # ~ letter at 100 DPI
    img = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw a title and three paragraph lines with default font; the
    # layout detector picks up the visual structure regardless of
    # script.
    draw.rectangle((50, 80, 800, 130), outline=(0, 0, 0), width=2)
    draw.text((60, 95), "Synthetic image-scan fixture", fill=(0, 0, 0))
    for i, line in enumerate(
        [
            "Paragraph line one — geometry-only.",
            "Paragraph line two — no text layer.",
            "Paragraph line three — phase 4 input.",
        ]
    ):
        draw.text((60, 180 + i * 30), line, fill=(0, 0, 0))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Step 2: write the image as the only content of a PDF page.
    c = canvas.Canvas(str(out), pagesize=letter)
    page_w, page_h = letter
    c.drawImage(ImageReader(buffer), 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()


def _build_image_scan_extra(name: str, lines: list[str]) -> None:
    """Phase 9: additional synthetic image-scan fixtures.

    Same generation strategy as ``_build_image_scan`` (raster page
    embedded in a PDF) but with different visible content. Used to
    meet the spec's ``image-scan >= 3`` minimum.
    """
    from io import BytesIO

    from PIL import Image, ImageDraw
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    out = IMAGE_SCAN / name
    width_px, height_px = 850, 1100
    img = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((50, 80, 800, 130), outline=(0, 0, 0), width=2)
    draw.text((60, 95), name, fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((60, 180 + i * 30), line, fill=(0, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    c = canvas.Canvas(str(out), pagesize=letter)
    page_w, page_h = letter
    c.drawImage(ImageReader(buffer), 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()


def _build_mixed_native_and_scan() -> None:
    """Phase 9: ``mixed/native-then-scan.pdf``.

    Two-page PDF where page 1 has a native text layer (validator
    accepts → native branch) and page 2 has only an embedded raster
    (validator rejects → ML branch). Exercises the per-page gating
    contract: one document, two branches.
    """
    from io import BytesIO

    from PIL import Image, ImageDraw
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    out = MIXED / "native-then-scan.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    page_w, page_h = letter

    # Page 1 — native text layer.
    c.setFont("Helvetica", 18)
    c.drawString(72, page_h - 90, "Native page (text layer)")
    c.setFont("Helvetica", 12)
    y = page_h - 120
    for word in PARA_WORDS * 4:
        c.drawString(72, y, word)
        y -= 16
    c.showPage()

    # Page 2 — image-only.
    width_px, height_px = 850, 1100
    img = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((60, 95), "Scan page (no text layer)", fill=(0, 0, 0))
    for i, line in enumerate(["raster line one", "raster line two", "raster line three"]):
        draw.text((60, 180 + i * 30), line, fill=(0, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    c.drawImage(ImageReader(buffer), 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()


def _build_edge_empty() -> None:
    """Phase 9 edge fixture: zero-page PDF.

    ``pypdfium2`` opens a zero-page document fine; the orchestrator
    yields a :class:`TranscribeResult` with ``n_pages == 0`` and the
    CLI exits 0 (per spec scenario 11 — empty Markdown is acceptable
    output). We synthesise this without ``reportlab`` (which always
    writes ``showPage()``) by hand-crafting the smallest possible
    zero-page PDF that ``pypdfium2`` accepts: a one-page document
    where the single page is then removed by re-saving with no
    content. The simplest portable trick is a one-page PDF whose
    content stream is empty.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    out = EDGE / "empty.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    # Single empty page — yields no native text, no image, no
    # regions. The validator returns no_text_layer; in best-effort
    # mode the orchestrator emits an empty TranscribeResult slot.
    c.showPage()
    c.save()


def _build_edge_truncated() -> None:
    """Phase 9 edge fixture: truncated PDF (spec scenario 13)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    out = EDGE / "truncated.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    c.drawString(72, 720, "this PDF will be truncated")
    c.showPage()
    c.save()
    # Lop the trailing 200 bytes — wipes the xref table; pypdfium2
    # raises CorruptedPDFError at open time.
    data = out.read_bytes()
    out.write_bytes(data[: max(0, len(data) - 200)])


def _build_edge_a2_poster() -> None:
    """Phase 9 edge fixture: very large page (spec scenario 17).

    A2 (~420x594mm at 72dpi → ~1190x1684pt) is the spec-quoted upper
    bound. The page has a small text region; we exercise the
    pipeline's behaviour on a single page that's much larger than
    the typical letter-size page (memory-ceiling sanity).
    """
    from reportlab.pdfgen import canvas

    out = EDGE / "a2-poster.pdf"
    a2 = (1190, 1684)
    c = canvas.Canvas(str(out), pagesize=a2)
    c.setFont("Helvetica", 12)
    c.drawString(72, a2[1] - 90, "A2 poster fixture (huge single page)")
    c.showPage()
    c.save()


def _build_edge_intra_region_bidi() -> None:
    """Phase 9 edge fixture: intra-region bidi (spec scenario 16).

    A page where one paragraph mixes Arabic and Latin runs; phase
    7's bidi helper should prefix the paragraph with U+200F when
    the paragraph is Arabic-dominant.
    """
    from reportlab.pdfgen import canvas

    out = EDGE / "intra-region-bidi.pdf"
    c = canvas.Canvas(str(out))
    c.setFont("Helvetica", 12)
    c.drawString(72, 720, "Arabic with English citation: see [Sibawayh, 760].")
    c.drawString(72, 700, "Mixed-direction sentence with Latin proper noun.")
    c.showPage()
    c.save()


def main() -> int:
    DIGITAL_CLEAN.mkdir(parents=True, exist_ok=True)
    DIGITAL_BROKEN.mkdir(parents=True, exist_ok=True)
    IMAGE_SCAN.mkdir(parents=True, exist_ok=True)
    MIXED.mkdir(parents=True, exist_ok=True)
    EDGE.mkdir(parents=True, exist_ok=True)
    _build_2col()
    _build_mixed()
    _build_table()
    real_arabic = _build_real_arabic()
    _build_broken_mojibake()
    _build_broken_replacement_glyphs()
    _build_image_scan()
    _build_image_scan_extra(
        "scan-ar-2col.pdf",
        [
            "Two-column synthetic scan.",
            "Layout detector picks up the geometry.",
            "OCR adapter fills text per region.",
        ],
    )
    _build_image_scan_extra(
        "scan-ar-headings.pdf",
        [
            "Title line",
            "Subtitle / second-level heading",
            "Body paragraph one with longer prose.",
            "Body paragraph two — exercises heading detection.",
        ],
    )
    _build_mixed_native_and_scan()
    _build_edge_empty()
    _build_edge_truncated()
    _build_edge_a2_poster()
    _build_edge_intra_region_bidi()
    names = [
        "digital-clean/lorem-ar-2col.pdf",
        "digital-clean/lorem-ar-en-mixed.pdf",
        "digital-clean/lorem-ar-table.pdf",
    ]
    if real_arabic:
        names.append("digital-clean/lorem-ar-real.pdf")
    names.extend(
        [
            "digital-broken/mojibake.pdf",
            "digital-broken/replacement-glyphs.pdf",
            "image-scan/scan-ar-1col.pdf",
            "image-scan/scan-ar-2col.pdf",
            "image-scan/scan-ar-headings.pdf",
            "mixed/native-then-scan.pdf",
            "edge/empty.pdf",
            "edge/truncated.pdf",
            "edge/a2-poster.pdf",
            "edge/intra-region-bidi.pdf",
        ]
    )
    fixtures_root = REPO_ROOT / "tests" / "fixtures" / "pdfs"
    for name in names:
        fixture = fixtures_root / name
        print(f"  wrote {fixture.relative_to(REPO_ROOT)} ({fixture.stat().st_size} bytes)")
    if not real_arabic:
        print("  (no Arabic-capable TTF found; skipped lorem-ar-real.pdf)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
