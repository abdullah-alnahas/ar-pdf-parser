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


def main() -> int:
    DIGITAL_CLEAN.mkdir(parents=True, exist_ok=True)
    DIGITAL_BROKEN.mkdir(parents=True, exist_ok=True)
    _build_2col()
    _build_mixed()
    _build_table()
    real_arabic = _build_real_arabic()
    _build_broken_mojibake()
    _build_broken_replacement_glyphs()
    names = [
        "digital-clean/lorem-ar-2col.pdf",
        "digital-clean/lorem-ar-en-mixed.pdf",
        "digital-clean/lorem-ar-table.pdf",
    ]
    if real_arabic:
        names.append("digital-clean/lorem-ar-real.pdf")
    names.extend(["digital-broken/mojibake.pdf", "digital-broken/replacement-glyphs.pdf"])
    fixtures_root = REPO_ROOT / "tests" / "fixtures" / "pdfs"
    for name in names:
        fixture = fixtures_root / name
        print(f"  wrote {fixture.relative_to(REPO_ROOT)} ({fixture.stat().st_size} bytes)")
    if not real_arabic:
        print("  (no Arabic-capable TTF found; skipped lorem-ar-real.pdf)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
