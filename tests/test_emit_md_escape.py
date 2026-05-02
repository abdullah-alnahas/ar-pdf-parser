"""Phase-7 markdown-escape tests."""

from __future__ import annotations

from arabic_pdf_transcribe.emit._md_escape import (
    escape_inline,
    escape_paragraph,
    escape_table_cell,
)


def test_escape_inline_passes_arabic_through() -> None:
    text = "السلام عليكم"
    assert escape_inline(text) == text


def test_escape_inline_escapes_asterisk() -> None:
    assert escape_inline("**bold**") == r"\*\*bold\*\*"


def test_escape_inline_escapes_backtick_and_underscore() -> None:
    assert escape_inline("`x` _y_") == r"\`x\` \_y\_"


def test_escape_inline_escapes_brackets() -> None:
    assert escape_inline("[link](url)") == r"\[link\](url)"


def test_escape_paragraph_neutralises_leading_heading() -> None:
    assert escape_paragraph("# not a heading") == r"\# not a heading"


def test_escape_paragraph_neutralises_leading_bullet() -> None:
    assert escape_paragraph("- not a bullet") == r"\- not a bullet"


def test_escape_paragraph_neutralises_leading_blockquote() -> None:
    assert escape_paragraph("> not a quote") == r"\> not a quote"


def test_escape_paragraph_neutralises_leading_ordered_list() -> None:
    assert escape_paragraph("1. not a list") == r"\1. not a list"
    assert escape_paragraph("12) still not") == r"\12) still not"


def test_escape_paragraph_multi_line_protects_each_line() -> None:
    out = escape_paragraph("first line\n# second line\nthird")
    assert out == "first line\n\\# second line\nthird"


def test_escape_paragraph_passes_arabic() -> None:
    text = "هذه فقرة عربية"
    assert escape_paragraph(text) == text


def test_escape_table_cell_escapes_pipes() -> None:
    assert escape_table_cell("a | b") == r"a \| b"


def test_escape_table_cell_escapes_newlines_to_br() -> None:
    assert escape_table_cell("a\nb") == "a<br>b"


def test_escape_table_cell_escapes_inline_syntax() -> None:
    assert escape_table_cell("**bold**") == r"\*\*bold\*\*"


def test_escape_paragraph_idempotent_on_safe_text() -> None:
    safe = "ordinary prose"
    assert escape_paragraph(safe) == safe
