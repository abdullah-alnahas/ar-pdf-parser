"""Phase-5 lazy-import regression test.

Importing :mod:`arabic_pdf_transcribe.ocr` (the public Protocol
module) and :mod:`arabic_pdf_transcribe.ocr.hf_ocr` (the concrete
adapter) must NOT pull ``transformers`` / ``torch`` / ``Pillow`` /
``huggingface_hub`` into ``sys.modules``. Heavy deps load only when
``HFQwen2VLOCRTranscriber._ensure_loaded()`` runs (on first
``transcribe`` call or explicit ``warm_up()``).
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


def test_importing_ocr_package_does_not_load_heavy_deps() -> None:
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.ocr")
    assert _leaks(loaded) == [], f"importing ocr package leaked heavy modules: {_leaks(loaded)}"


def test_importing_ocr_crop_does_not_load_heavy_deps() -> None:
    """``ocr._crop`` uses Pillow only inside ``crop_region``. Module
    import alone must not load PIL."""
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.ocr._crop")
    assert _leaks(loaded) == [], f"importing ocr._crop leaked heavy modules: {_leaks(loaded)}"


def test_importing_hf_ocr_does_not_load_heavy_deps() -> None:
    """The concrete adapter must keep transformers / torch out of the
    import graph until ``_ensure_loaded`` runs."""
    loaded = _import_and_dump_modules("arabic_pdf_transcribe.ocr.hf_ocr")
    assert _leaks(loaded) == [], f"importing hf_ocr leaked heavy modules: {_leaks(loaded)}"
