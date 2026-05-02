"""Rasterise a single PDF page to a PIL image.

Lives next to the layout adapter so the boundary is clean: the
orchestrator hands a :class:`pypdfium2.PdfPage` and a DPI to this module
and gets back a :class:`PIL.Image.Image`. The pypdfium2 dependency is
already a runtime dep (phase 2 pinned it for native extraction); Pillow
is gated behind the ``[ml]`` extra and imported lazily here.

Default DPI is 200, the spec-quoted target for the ML branch (8 GB peak
RSS budget on the CPU path with the default models).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pypdfium2 as pdfium
    from PIL.Image import Image as PILImage

DEFAULT_DPI = 200
DEFAULT_SCALE = DEFAULT_DPI / 72.0


def rasterise_page(page: pdfium.PdfPage, *, dpi: int = DEFAULT_DPI) -> PILImage:
    """Render ``page`` at ``dpi`` and return a PIL image.

    The image is RGB; pypdfium2 produces RGBA by default so we explicitly
    convert. Caller is responsible for closing the underlying page (this
    helper does not own the page handle).
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    scale = dpi / 72.0
    # pypdfium2 accepts float scale at runtime; its public type stub
    # narrows to int — float is documented in the upstream docstring.
    bitmap = page.render(scale=scale)  # pyright: ignore[reportArgumentType]
    image = bitmap.to_pil()
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
