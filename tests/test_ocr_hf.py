"""Phase-5 HF GOT-OCR adapter tests with stubbed model.

The adapter wraps ``stepfun-ai/GOT-OCR-2.0-hf``. Tests stub the
``transformers`` ``processor`` + ``model`` so the adapter exercises
its end-to-end flow against deterministic fake outputs.

A separate ``@pytest.mark.slow`` test loads the real model and runs
it on a region from the phase-4 image-scan fixture; it is skipped by
default.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from arabic_pdf_transcribe.errors import ModelDownloadError, OCRTranscriptionError  # noqa: E402
from arabic_pdf_transcribe.ocr import OCRTranscriber  # noqa: E402
from arabic_pdf_transcribe.ocr.hf_ocr import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    HFGotOCRTranscriber,
    OCRConfig,
)
from arabic_pdf_transcribe.regions import (  # noqa: E402
    BBox,
    Region,
    RegionRole,
    RegionSource,
    TableCell,
    TableGrid,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProcessor:
    """Minimal stub mirroring the parts of HF AutoProcessor we touch."""

    def __init__(self, fake_text: str = "أ ب ج") -> None:
        self.fake_text = fake_text
        self.calls: list[object] = []

    def __call__(self, *, images: object, return_tensors: str) -> dict[str, object]:
        import torch

        self.calls.append(images)
        return {
            "pixel_values": torch.zeros((1, 3, 224, 224)),
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
        }

    def batch_decode(self, sequences: object, *, skip_special_tokens: bool = True) -> list[str]:
        return [self.fake_text]


class _FakeOutputs:
    """Stand-in for transformers' ``GenerateOutput``."""

    def __init__(self, sequences: object, scores: tuple[object, ...]) -> None:
        self.sequences = sequences
        self.scores = scores


class _FakeModel:
    def __init__(self, output_text_tokens: list[int], vocab_size: int = 32000) -> None:
        self._tokens = output_text_tokens
        self._vocab_size = vocab_size

    def generate(self, *, max_new_tokens: int, **_: object) -> _FakeOutputs:
        import torch

        # Sequence: prompt ([0]) + new tokens.
        prompt = torch.zeros((1, 1), dtype=torch.long)
        new = torch.tensor([self._tokens], dtype=torch.long)
        sequences = torch.cat([prompt, new], dim=1)
        # Scores: one tensor per generated step.
        scores: list[object] = []
        for tok in self._tokens:
            step = torch.full((1, self._vocab_size), -10.0)
            step[0, tok] = 5.0  # high logit for chosen token → high prob
            scores.append(step)
        return _FakeOutputs(sequences=sequences, scores=tuple(scores))


_DEFAULT_BBOX = BBox(10.0, 10.0, 100.0, 50.0)


def _make_paragraph(bbox: BBox | None = None) -> Region:
    return Region(
        page_index=0,
        bbox=bbox or _DEFAULT_BBOX,
        text="",
        role=RegionRole.PARAGRAPH,
        source=RegionSource.OCR,
    )


def _attach_stubs(
    transcriber: HFGotOCRTranscriber,
    *,
    text: str = "أ ب ج",
    tokens: list[int] | None = None,
) -> _FakeProcessor:
    pytest.importorskip("torch")
    processor = _FakeProcessor(fake_text=text)
    transcriber._processor = processor
    transcriber._model = _FakeModel(output_text_tokens=tokens or [42, 7, 13])
    return processor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_config_pins_apache_licensed_got_ocr() -> None:
    cfg = OCRConfig()
    assert cfg.model == DEFAULT_MODEL
    assert cfg.revision == DEFAULT_REVISION
    assert DEFAULT_MODEL == "stepfun-ai/GOT-OCR-2.0-hf"
    assert len(DEFAULT_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in DEFAULT_REVISION)
    # Deterministic decoding defaults — required for the spec's
    # reproducibility contract.
    assert cfg.do_sample is False
    assert cfg.num_beams == 1
    # Issue #18 lowered the default from 4096 to 1024 to bound CPU
    # runaway generation; still > 3× a typical Arabic paragraph.
    assert cfg.max_new_tokens == 1024
    assert cfg.no_repeat_ngram_size == 3
    assert cfg.repetition_penalty == 1.05
    assert cfg.device == "auto"


def test_transcriber_satisfies_protocol() -> None:
    assert isinstance(HFGotOCRTranscriber(), OCRTranscriber)


def test_transcribe_paragraph_fills_text_and_confidence() -> None:
    pytest.importorskip("torch")
    transcriber = HFGotOCRTranscriber()
    _attach_stubs(transcriber, text="أ ب ج")
    region = _make_paragraph()
    image = Image.new("RGB", (200, 100), color=(255, 255, 255))

    result = transcriber.transcribe(region, image)

    assert result.text == "أ ب ج"
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
    # High-logit chosen tokens → near-1 confidence.
    assert result.confidence > 0.9
    assert result.role is RegionRole.PARAGRAPH
    assert result.bbox == region.bbox


def test_figure_region_passes_through_unchanged() -> None:
    """FIGURE regions are not OCR'd in v1 (spec contract)."""
    pytest.importorskip("torch")
    transcriber = HFGotOCRTranscriber()
    # Stubs intentionally NOT attached — calling them would error.
    figure = Region(
        page_index=0,
        bbox=BBox(0.0, 0.0, 100.0, 100.0),
        text="",
        role=RegionRole.FIGURE,
        source=RegionSource.OCR,
    )
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))
    result = transcriber.transcribe(figure, image)
    assert result is figure
    assert result.text == ""


def test_table_region_walks_cells_filling_each() -> None:
    pytest.importorskip("torch")
    transcriber = HFGotOCRTranscriber()
    _attach_stubs(transcriber, text="cell")
    grid = TableGrid(
        rows=(
            (
                TableCell(text="", confidence=None, bbox=BBox(10.0, 10.0, 50.0, 30.0)),
                TableCell(text="", confidence=None, bbox=BBox(50.0, 10.0, 90.0, 30.0)),
            ),
            (
                TableCell(text="", confidence=None, bbox=BBox(10.0, 30.0, 50.0, 50.0)),
                TableCell(text="", confidence=None, bbox=BBox(50.0, 30.0, 90.0, 50.0)),
            ),
        )
    )
    table = Region(
        page_index=0,
        bbox=BBox(10.0, 10.0, 90.0, 50.0),
        text="",
        role=RegionRole.TABLE,
        source=RegionSource.OCR,
        table_grid=grid,
    )
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))
    result = transcriber.transcribe(table, image)
    assert result.role is RegionRole.TABLE
    assert result.text == ""  # outer Region's text stays empty
    assert result.table_grid is not None
    assert result.table_grid.n_rows == 2
    assert result.table_grid.n_cols == 2
    for row in result.table_grid.rows:
        for cell in row:
            assert cell.text == "cell"
            assert cell.confidence is not None
            assert 0.0 <= cell.confidence <= 1.0


def test_table_region_without_grid_raises_ocr_error() -> None:
    """Plan contract: phase 4 must populate ``table_grid``. If we see
    a TABLE region with ``None``, that is a bug upstream — surface it
    as :class:`OCRTranscriptionError`."""
    pytest.importorskip("torch")
    transcriber = HFGotOCRTranscriber()
    table = Region(
        page_index=2,
        bbox=BBox(10.0, 10.0, 90.0, 50.0),
        text="",
        role=RegionRole.TABLE,
        source=RegionSource.OCR,
        table_grid=None,
    )
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))
    with pytest.raises(OCRTranscriptionError, match="no table_grid"):
        transcriber.transcribe(table, image)


def test_degenerate_bbox_raises_ocr_error() -> None:
    pytest.importorskip("torch")
    transcriber = HFGotOCRTranscriber()
    _attach_stubs(transcriber)
    region = Region(
        page_index=0,
        bbox=BBox(500.0, 500.0, 600.0, 600.0),
        text="",
        role=RegionRole.PARAGRAPH,
        source=RegionSource.OCR,
    )
    # Bbox entirely outside image; crop helper raises.
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    with pytest.raises(OCRTranscriptionError, match="degenerate crop"):
        transcriber.transcribe(region, image)


def test_generate_failure_raises_ocr_transcription_error() -> None:
    pytest.importorskip("torch")
    transcriber = HFGotOCRTranscriber()

    class _ExplodingModel:
        def generate(self, **kwargs: object) -> object:
            raise RuntimeError("synthetic decoder failure")

    transcriber._processor = _FakeProcessor()
    transcriber._model = _ExplodingModel()
    region = _make_paragraph()
    image = Image.new("RGB", (200, 100), color=(255, 255, 255))
    with pytest.raises(OCRTranscriptionError, match="OCR generate failed"):
        transcriber.transcribe(region, image)


def test_ensure_loaded_raises_model_download_error_on_transformers_missing() -> None:
    """transformers ImportError → ModelDownloadError (CLI exit 5)."""
    transcriber = HFGotOCRTranscriber()
    sentinel = "transformers"
    saved = sys.modules.pop(sentinel, None)
    fake_module = types.ModuleType("transformers")
    sys.modules[sentinel] = fake_module
    try:
        with pytest.raises(ModelDownloadError):
            transcriber._ensure_loaded()
    finally:
        if saved is not None:
            sys.modules[sentinel] = saved
        else:
            sys.modules.pop(sentinel, None)


def test_degenerate_table_cell_returns_empty_silently() -> None:
    """A single bad cell should not abort the whole table — empty
    cells render fine in MD/Word."""
    pytest.importorskip("torch")
    transcriber = HFGotOCRTranscriber()
    _attach_stubs(transcriber, text="ok")
    grid = TableGrid(
        rows=(
            (
                TableCell(text="", confidence=None, bbox=BBox(10.0, 10.0, 50.0, 30.0)),
                # Out-of-image bbox → degenerate crop.
                TableCell(text="", confidence=None, bbox=BBox(500.0, 500.0, 600.0, 600.0)),
            ),
        )
    )
    table = Region(
        page_index=0,
        bbox=BBox(10.0, 10.0, 600.0, 600.0),
        text="",
        role=RegionRole.TABLE,
        source=RegionSource.OCR,
        table_grid=grid,
    )
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))
    result = transcriber.transcribe(table, image)
    assert result.table_grid is not None
    cells = result.table_grid.rows[0]
    assert cells[0].text == "ok"
    assert cells[1].text == ""
    assert cells[1].confidence is None


@pytest.mark.slow
def test_real_model_loads_and_transcribes_image_scan_region() -> None:
    """Optional real-model run; off by default in CI.

    Run locally via ``pytest -m slow``. PR author attaches the output
    to the PR description.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    fixture = (
        Path(__file__).resolve().parent / "fixtures" / "pdfs" / "image-scan" / "scan-ar-1col.pdf"
    )
    if not fixture.exists():
        pytest.skip("phase-4 image-scan fixture missing")

    import pypdfium2 as pdfium

    from arabic_pdf_transcribe.layout._rasterise import rasterise_page

    pdf = pdfium.PdfDocument(str(fixture))
    page = pdf[0]
    image = rasterise_page(page, dpi=150)

    region = Region(
        page_index=0,
        bbox=BBox(0.0, 0.0, float(image.width), float(image.height)),
        text="",
        role=RegionRole.PARAGRAPH,
        source=RegionSource.OCR,
    )
    transcriber = HFGotOCRTranscriber()
    result = transcriber.transcribe(region, image)
    assert result.text != ""
    # Confidence is optional but the default model exposes scores.
    assert result.confidence is None or (0.0 <= result.confidence <= 1.0)
