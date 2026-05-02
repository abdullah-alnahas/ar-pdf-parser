"""Hugging Face DiT-base layout-detection adapter.

# Model selection

Phase 4 selected ``cmarkea/dit-base-layout-detection`` as the default
layout detector after weighing three candidates against the spec
constraints (see plan phase 4 deliverables, models.toml, and the
phase-4 PR description for the full benchmark write-up):

| Candidate            | License           | Footprint | Class set | Arabic robustness | Verdict |
|----------------------|-------------------|-----------|-----------|-------------------|---------|
| **DiT-base layout**  | Apache-2.0        | ~330 MB   | 12 doc-AI | script-agnostic (operates on bitmaps; bidi/RTL transparent) | **chosen** |
| DocLayout-YOLO       | AGPL-3.0 derivative | ~200 MB | 10 doc-AI | script-agnostic, but AGPL fails the project allow-list | rejected on license |
| Surya layout         | GPL-3.0 (post-Surya v1) | ~700 MB  | 11 doc-AI | strong Arabic support but GPL fails allow-list | rejected on license |

The DiT model is a :class:`BeitForSemanticSegmentation` head (12
classes) — it produces a per-pixel class map rather than direct bboxes.
The adapter turns the per-pixel map into bbox regions by finding
connected components per class and computing their bounding boxes.

# Lazy-import discipline

``transformers`` and ``torch`` are imported **inside** the constructor,
not at module-import time. The lazy-import test in
``tests/test_skeleton.py`` and the per-phase regression test in
``tests/test_layout_lazy_import.py`` enforce that
``import arabic_pdf_transcribe.layout`` does not pull either library
into ``sys.modules``.

# Offline / cache-miss handling

The adapter raises
:class:`arabic_pdf_transcribe.errors.ModelDownloadError` when
``transformers`` cannot resolve the pinned revision from the local
cache and the environment denies network access (offline mode, no
network, etc.). The CLI maps this exception to exit code 5.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arabic_pdf_transcribe.errors import ModelDownloadError
from arabic_pdf_transcribe.layout._classes import (
    DROPPED_LABELS,
    heading_level_for_label,
    role_for_label,
)
from arabic_pdf_transcribe.layout._table_cells import detect_table_cells
from arabic_pdf_transcribe.regions import (
    BBox,
    Region,
    RegionRole,
    RegionSource,
    TableCell,
    TableGrid,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "cmarkea/dit-base-layout-detection"
DEFAULT_REVISION = "1995237326c8b53d93525b7b19e20bb363b4eb73"
# Confidence threshold for a per-pixel argmax to count as "detected"
# class. Below this the pixel is treated as Background. The Beit head
# does not produce per-region confidences (it is segmentation, not
# detection); we surface the mean per-pixel softmax probability of the
# region as a coarse confidence proxy.
DEFAULT_PIXEL_CONFIDENCE = 0.5
DEFAULT_MIN_REGION_AREA_PX = 64
DEFAULT_DEVICE = "auto"


@dataclass(frozen=True)
class HFLayoutDetectorConfig:
    """Tuneable knobs for the HF layout detector.

    Defaults match ``cmarkea/dit-base-layout-detection`` at the pinned
    revision. Phase 9 may re-tune ``pixel_confidence`` and
    ``min_region_area_px`` against the corpus.
    """

    model: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    pixel_confidence: float = DEFAULT_PIXEL_CONFIDENCE
    min_region_area_px: int = DEFAULT_MIN_REGION_AREA_PX
    detect_table_cells: bool = True
    device: str = DEFAULT_DEVICE

    @classmethod
    def from_mapping(cls, mapping: object) -> HFLayoutDetectorConfig:
        """Build a config from a TOML ``[layout]`` section dict.

        Unknown keys are ignored (forward-compatibility); typed fields
        come through Python's TOML loader as the right scalars
        already.
        """
        if not isinstance(mapping, dict):
            return cls()
        kwargs = {k: v for k, v in mapping.items() if k in cls.__dataclass_fields__}
        return cls(**kwargs)  # type: ignore[arg-type]


class HFDiTLayoutDetector:
    """Concrete :class:`LayoutDetector` backed by the DiT-base model.

    Constructed lazily: the model and image processor load on first
    ``detect`` call (or on explicit ``warm_up()``), so a stub-injected
    pipeline never pays the load cost. The class implements the
    :class:`LayoutDetector` Protocol structurally; ``runtime_checkable``
    on the Protocol means consumers can ``isinstance(d, LayoutDetector)``.
    """

    def __init__(self, config: HFLayoutDetectorConfig | None = None) -> None:
        self.config = config or HFLayoutDetectorConfig()
        self._model: Any = None
        self._processor: Any = None
        self._id2label: dict[int, str] | None = None
        # Resolved on first ``_ensure_loaded`` call. Sticky for the
        # adapter's lifetime so a CUDA OOM can downgrade us to CPU
        # for the rest of the run (issue #18 RC#1).
        self._device: str | None = None

    def warm_up(self) -> None:
        """Force model + processor load now (otherwise lazy)."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            # Local imports — keep ``transformers`` / ``torch`` out of
            # the top-level import graph.
            from transformers import (
                AutoImageProcessor,
                AutoModelForSemanticSegmentation,
            )
        except ImportError as exc:  # pragma: no cover — install-time check
            raise ModelDownloadError(
                f"transformers is required for layout detection but is not installed; "
                f"install the [ml] extra: pip install 'arabic-pdf-transcribe[ml]'. "
                f"({exc})"
            ) from exc
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self.config.model, revision=self.config.revision
            )
            self._model = AutoModelForSemanticSegmentation.from_pretrained(
                self.config.model, revision=self.config.revision
            )
        except Exception as exc:  # pragma: no cover — exercised in slow test
            raise ModelDownloadError(
                f"failed to load layout model {self.config.model}@{self.config.revision[:12]}; "
                f"prefetch the weights with: arabic-pdf-transcribe --prefetch-models "
                f"(or, manually: huggingface-cli download {self.config.model} "
                f"--revision {self.config.revision}). ({exc})"
            ) from exc
        # Cast id2label keys to int — HF stores them as strings in JSON.
        raw = getattr(self._model.config, "id2label", {}) or {}
        self._id2label = {int(k): v for k, v in raw.items()}
        # Issue #18 RC#1: place model on the resolved device. Without
        # this, the segmentation forward pass ran on CPU even when
        # CUDA was available — the same root cause as the OCR hang.
        from arabic_pdf_transcribe._device import resolve_device

        self._device = resolve_device(self.config.device)
        move = getattr(self._model, "to", None)
        if callable(move):
            with contextlib.suppress(RuntimeError, ValueError):  # pragma: no cover
                move(self._device)
        switch = getattr(self._model, "eval", None)
        if callable(switch):
            with contextlib.suppress(Exception):  # pragma: no cover — stubs only
                switch()

    def detect(self, page_image: PILImage, page_index: int) -> Sequence[Region]:
        """Return regions detected on ``page_image``.

        See class docstring for the segmentation → connected-components
        → bbox flow.
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._id2label is not None
        # Local import — gated by ``[ml]`` extra.
        import torch

        inputs = self._processor(images=page_image, return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, self._device or "cpu")
        try:
            with torch.no_grad():
                outputs = self._model(**inputs)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or self._device != "cuda":
                raise
            LOGGER.warning(
                "layout: CUDA OOM during forward; falling back to CPU for the remainder of this run"
            )
            with contextlib.suppress(Exception):  # pragma: no cover — defensive
                torch.cuda.empty_cache()
            self._model.to("cpu")
            self._device = "cpu"
            inputs = _move_inputs_to_device(inputs, "cpu")
            with torch.no_grad():
                outputs = self._model(**inputs)
        logits = outputs.logits  # (1, num_labels, h, w)
        probs = torch.softmax(logits, dim=1)
        confidences, class_map = torch.max(probs, dim=1)  # (1, h, w) each
        confidence_grid = confidences[0].cpu().numpy()
        class_grid = class_map[0].cpu().numpy()
        # Resize per-pixel grids to match the input image so bboxes are
        # in input-image coordinates.
        return list(
            self._regions_from_class_map(
                class_grid=class_grid,
                confidence_grid=confidence_grid,
                page_image=page_image,
                page_index=page_index,
            )
        )

    def _regions_from_class_map(
        self,
        *,
        class_grid: Any,
        confidence_grid: Any,
        page_image: PILImage,
        page_index: int,
    ) -> Sequence[Region]:
        """Convert a per-pixel class map into Region objects.

        Connected components per class become regions. Background and
        sub-confidence pixels are skipped. Tables get their grid filled
        in by :func:`detect_table_cells`; figures and other roles ship
        an empty grid.
        """
        assert self._id2label is not None
        regions: list[Region] = []
        # numpy is a transitive dep of torch / transformers; importing
        # here is safe inside the [ml] gate.
        import numpy as np

        # Resize the per-pixel grids back to the input image size.
        h_in, w_in = class_grid.shape
        w_out, h_out = page_image.size
        sx = w_out / max(w_in, 1)
        sy = h_out / max(h_in, 1)

        unique_classes = np.unique(class_grid)
        for class_id in unique_classes:
            label = self._id2label.get(int(class_id))
            if label is None or label in DROPPED_LABELS:
                continue
            mask = (class_grid == class_id) & (confidence_grid >= self.config.pixel_confidence)
            if not bool(mask.any()):
                continue
            for component_bbox, mean_conf in _connected_components(mask, confidence_grid):
                # Scale grid-space bbox to input-image space.
                x0 = float(component_bbox[0]) * sx
                y0 = float(component_bbox[1]) * sy
                x1 = float(component_bbox[2]) * sx
                y1 = float(component_bbox[3]) * sy
                area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
                if area < self.config.min_region_area_px:
                    continue
                role = role_for_label(label)
                if role is RegionRole.UNKNOWN:
                    LOGGER.warning(
                        "layout: label %r mapped to UNKNOWN; phase 6 will handle it",
                        label,
                    )
                bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
                table_grid = None
                if role is RegionRole.TABLE:
                    if self.config.detect_table_cells:
                        table_grid = detect_table_cells(page_image, bbox)
                    # Plan contract (phase 4): every TABLE Region must
                    # carry a populated table_grid. When morphology
                    # fails (no ruled lines, complex layout), fall
                    # back to a one-cell-per-row coalescing — phase 5
                    # OCRs the single cell as plain prose, phase 6
                    # role-classifies the result as a generic table
                    # whose contents are a paragraph.
                    if table_grid is None:
                        table_grid = _fallback_single_cell_grid(bbox)
                regions.append(
                    Region(
                        page_index=page_index,
                        bbox=bbox,
                        text="",
                        role=role,
                        source=RegionSource.OCR,
                        confidence=float(mean_conf),
                        heading_level=heading_level_for_label(label),
                        table_grid=table_grid,
                    )
                )
        regions.sort(key=lambda r: (r.bbox.y0, r.bbox.x0))
        return regions


def _connected_components(
    mask: Any, confidence_grid: Any
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Return ``[((x0, y0, x1, y1), mean_conf), ...]`` for each 4-connected component.

    Pure-numpy flood fill — keeps the layout package's only ML dep
    surface to ``transformers`` + ``torch`` (numpy comes transitively).
    """
    import numpy as np

    # Iterative 4-connected flood fill; avoids Python's recursion limit
    # on large masks. Components smaller than ``MIN_PIXELS`` are
    # filtered downstream by area on the input-image scale.
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    components: list[tuple[tuple[int, int, int, int], float]] = []
    for start_y in range(h):
        for start_x in range(w):
            if visited[start_y, start_x] or not bool(mask[start_y, start_x]):
                continue
            stack: list[tuple[int, int]] = [(start_y, start_x)]
            min_x, min_y, max_x, max_y = start_x, start_y, start_x, start_y
            confs: list[float] = []
            while stack:
                y, x = stack.pop()
                if visited[y, x] or not bool(mask[y, x]):
                    continue
                visited[y, x] = True
                confs.append(float(confidence_grid[y, x]))
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
                if y > 0:
                    stack.append((y - 1, x))
                if y < h - 1:
                    stack.append((y + 1, x))
                if x > 0:
                    stack.append((y, x - 1))
                if x < w - 1:
                    stack.append((y, x + 1))
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            components.append(((min_x, min_y, max_x + 1, max_y + 1), mean_conf))
    return components


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    """Move processor outputs to ``device``; tolerate plain-dict stubs."""
    move = getattr(inputs, "to", None)
    if callable(move):
        try:
            return move(device)
        except (TypeError, AttributeError, RuntimeError):
            pass
    try:
        import torch
    except ImportError:  # pragma: no cover
        return inputs
    if not isinstance(inputs, dict):
        return inputs
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def _is_cuda_oom(exc: BaseException) -> bool:
    """Detect a torch CUDA OOM without importing torch eagerly."""
    cls = type(exc)
    if cls.__name__ == "OutOfMemoryError" and (cls.__module__ or "").startswith("torch"):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def _fallback_single_cell_grid(bbox: BBox) -> TableGrid:
    """One-cell-per-row fallback when ruled-line morphology fails.

    Plan phase-4 contract: every ``TABLE`` Region must carry a populated
    ``table_grid``. When :func:`detect_table_cells` cannot recover the
    grid (no ruled lines, complex layout, image too small), the adapter
    builds a one-row, one-cell grid covering the full table bbox. Phase 5
    OCRs the single cell; phase 6 keeps the resulting Region as a TABLE
    whose body is a single prose run — better than dropping the table
    entirely.
    """
    return TableGrid(
        rows=(
            (
                TableCell(
                    text="",
                    confidence=None,
                    bbox=bbox,
                ),
            ),
        )
    )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "HFDiTLayoutDetector",
    "HFLayoutDetectorConfig",
]
