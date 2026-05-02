"""Markdown escaping helpers.

Region text may contain characters that Markdown interprets
structurally:

* Leading ``# `` → unintended heading.
* Leading ``- ``, ``* ``, ``+ `` → unintended bullet.
* Leading digits + ``.`` or ``)`` → unintended ordered list.
* Leading ``> `` → unintended blockquote.
* Pipe ``|`` inside a table cell → unintended column break.
* Backslash, backtick, asterisk, underscore, square brackets — inline
  emphasis / link syntax.

The emitter calls :func:`escape_paragraph` for plain prose and
:func:`escape_table_cell` for table cell text. Both are conservative
— they escape what would actually be parsed structurally, not every
ambiguous character. Over-escaping makes the output noisy and breaks
round-trip with downstream Markdown renderers.

Escapes are **deterministic** so snapshot tests are stable.
"""

from __future__ import annotations

import re

# Inline characters that always need escaping when they appear in a
# paragraph or cell. The list mirrors CommonMark's ASCII punctuation
# rule but is restricted to characters that are *actively* used as
# Markdown syntax — escaping every ASCII punctuation mark would be
# noisy and harm readability.
_INLINE_ESCAPE_RE = re.compile(r"([\\`*_\[\]<>])")

# A leading-line pattern for block constructs we need to neutralise.
# Order matters: longer matches first.
_LEADING_BLOCK_RE = re.compile(
    r"^(?P<prefix>"
    r"#{1,6}\s"  # ATX heading
    r"|>\s?"  # blockquote
    r"|[-*+]\s"  # bullet list
    r"|\d+[.)]\s"  # ordered list
    r")"
)


def escape_inline(text: str) -> str:
    """Escape inline Markdown syntax characters.

    Applied to all paragraph / heading text. Keeps Arabic letters
    untouched (they are not in the escape set).
    """
    return _INLINE_ESCAPE_RE.sub(r"\\\1", text)


def escape_paragraph(text: str) -> str:
    """Escape a full paragraph, including any leading block construct.

    Two-pass:

    1. Escape inline syntax characters in the body.
    2. If the (possibly escaped) line begins with a Markdown block
       construct (``# ``, ``- ``, ``> ``, ``1. ``…), prefix the
       construct's first character with a backslash so it is rendered
       as literal text rather than a heading / list / quote.

    Multi-line input: each line is processed independently for the
    leading-block check so a paragraph containing an embedded line
    that *looks* like a heading does not silently render as one.
    """
    out_lines: list[str] = []
    for line in text.split("\n"):
        escaped = escape_inline(line)
        match = _LEADING_BLOCK_RE.match(escaped)
        if match is not None:
            escaped = "\\" + escaped
        out_lines.append(escaped)
    return "\n".join(out_lines)


def escape_table_cell(text: str) -> str:
    """Escape text for use inside a Markdown pipe-table cell.

    Pipes (``|``) and newlines must be escaped because they would
    otherwise terminate the cell or row. Inline syntax characters are
    *also* escaped so that adversarial cell text (e.g. ``**bold**``)
    renders as literal characters.
    """
    escaped = escape_inline(text)
    escaped = escaped.replace("|", "\\|")
    # Newlines inside cells: replace with HTML <br> so Markdown
    # renderers preserve them. Plain ``\n`` would close the row.
    escaped = escaped.replace("\n", "<br>")
    return escaped


__all__ = ["escape_inline", "escape_paragraph", "escape_table_cell"]
