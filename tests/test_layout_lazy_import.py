"""Phase-4 lazy-import regression test.

The phase-1 lazy-import test in ``test_skeleton.py`` already asserts
that importing the top-level package does not pull in
``transformers``/``torch``/``Pillow``/``huggingface_hub``. Phase 4
adds the ``layout`` subpackage; this test additionally asserts that
importing ``arabic_pdf_transcribe.layout`` (the public Protocol module)
does not load those heavy deps either. Only the concrete adapter at
``arabic_pdf_transcribe.layout.hf_detector`` is allowed to pull them
in, and only when its constructor's ``warm_up`` / ``detect`` runs.
"""

from __future__ import annotations

import json
import subprocess
import sys

HEAVY_MODULES = ("transformers", "torch", "huggingface_hub", "PIL")


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


def test_importing_layout_package_does_not_load_heavy_deps() -> None:
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.layout")
    assert _leaks(loaded) == [], f"importing layout package leaked heavy modules: {_leaks(loaded)}"


def test_importing_layout_classes_does_not_load_heavy_deps() -> None:
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.layout._classes")
    assert _leaks(loaded) == [], f"importing layout._classes leaked heavy modules: {_leaks(loaded)}"


def test_importing_layout_table_cells_does_not_load_heavy_deps() -> None:
    """``_table_cells`` uses Pillow but only via local imports inside
    :func:`detect_table_cells`. Importing the module must not pull
    Pillow into ``sys.modules``.
    """
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.layout._table_cells")
    assert _leaks(loaded) == [], f"importing _table_cells leaked heavy modules: {_leaks(loaded)}"


def test_importing_hf_detector_module_does_not_load_heavy_deps() -> None:
    """``hf_detector`` itself must not import torch/transformers at
    module load — they're imported inside ``_ensure_loaded`` /
    ``detect``. Pillow is only imported by the ``Image`` type alias
    under ``TYPE_CHECKING`` and so does not load at runtime.
    """
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.layout.hf_detector")
    assert _leaks(loaded) == [], f"importing hf_detector leaked heavy modules: {_leaks(loaded)}"
