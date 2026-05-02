"""Phase-7 lazy-import regression: emit package must not load docx eagerly.

``python-docx`` is only needed by the docx emitter. Importing
``arabic_pdf_transcribe.emit`` (or any of its non-docx submodules)
must not pull ``docx`` into ``sys.modules``.
"""

from __future__ import annotations

import json
import subprocess
import sys

HEAVY_MODULES = ("docx",)


def _import_and_dump_modules(import_target: str) -> set[str]:
    code = (
        "import sys, json\n"
        f"import {import_target}\n"
        "print(json.dumps([m for m in sys.modules]))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(proc.stdout))


def _leaks(loaded: set[str]) -> list[str]:
    return [m for m in HEAVY_MODULES if any(k == m or k.startswith(m + ".") for k in loaded)]


def test_importing_emit_package_does_not_load_docx() -> None:
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.emit")
    assert _leaks(loaded) == [], f"emit package leaked: {_leaks(loaded)}"


def test_importing_markdown_module_does_not_load_docx() -> None:
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.emit.markdown")
    assert _leaks(loaded) == [], f"emit.markdown leaked: {_leaks(loaded)}"


def test_importing_normalise_module_does_not_load_docx() -> None:
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.emit._normalise")
    assert _leaks(loaded) == [], f"emit._normalise leaked: {_leaks(loaded)}"


def test_importing_bidi_module_does_not_load_docx() -> None:
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.emit._bidi")
    assert _leaks(loaded) == [], f"emit._bidi leaked: {_leaks(loaded)}"


def test_importing_md_escape_module_does_not_load_docx() -> None:
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.emit._md_escape")
    assert _leaks(loaded) == [], f"emit._md_escape leaked: {_leaks(loaded)}"
