"""Ray-actor executor for the layout + OCR stages.

Activated via ``--executor ray``. Wraps the existing layout / OCR
adapters in long-lived Ray actors so:

* The model loads once per actor (in the worker process), not once per
  CLI invocation. Same warm-up amortisation as thread mode but each
  actor lives in its own process — useful for crash isolation and as
  the foundation for multi-GPU scale-out (PR 3).
* GPU resources are declared explicitly (``num_gpus`` per actor) so
  Ray's scheduler can place actors on the right device when more than
  one is available.

Design constraints (single-GPU 6 GB hardware in mind):

* Both actors share GPU 0 by default with ``num_gpus=0.5`` each. Ray's
  GPU share is virtual (a token, not VRAM enforcement) — the user must
  ensure both models fit. Override via ``--num-gpus-per-actor``.
* The proxy classes implement the project's
  :class:`LayoutDetector` / :class:`OCRTranscriber` Protocols so the
  pipeline orchestrator is unchanged: same per-page calls, same
  ``transcribe_page`` fast path. No code in ``pipeline.py`` knows Ray
  exists.
* Ray is initialised on first proxy construction and shut down via
  :func:`atexit`. ``ray.init(ignore_reinit_error=True)`` is safe under
  re-entry (tests, repeated CLI invocations in one process).
* PIL images cross the actor boundary via Ray's plasma object store
  (zero-copy after the first ``put``). Pillow has been picklable since
  v8 so this works without serialisation hooks.
"""

from __future__ import annotations

import atexit
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from arabic_pdf_transcribe.errors import ModelDownloadError

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from PIL.Image import Image as PILImage

    from arabic_pdf_transcribe.regions import Region


# ---------------------------------------------------------------------------
# Lazy Ray bootstrap
# ---------------------------------------------------------------------------


_RAY_INITIALISED = False


def _ensure_ray_initialised() -> Any:
    """Import + boot Ray exactly once per process.

    Returns the ``ray`` module so callers can use it without a top-level
    import (Ray is an optional dep behind the ``[ray]`` extra).
    """
    global _RAY_INITIALISED
    try:
        import ray
    except ImportError as exc:
        raise ModelDownloadError(
            "the --executor ray option requires the 'ray' package; install with: "
            "pip install 'arabic-pdf-transcribe[ray]' or 'pip install ray[default]'"
        ) from exc
    if not _RAY_INITIALISED:
        # ``ignore_reinit_error`` makes the call idempotent across
        # repeated invocations in the same process (tests).
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            include_dashboard=False,
            configure_logging=False,
        )
        atexit.register(_safe_shutdown)
        _RAY_INITIALISED = True
    return ray


def _safe_shutdown() -> None:
    """``ray.shutdown`` wrapped to never raise out of ``atexit``."""
    try:
        import ray
    except ImportError:  # pragma: no cover — atexit handler, defensive
        return
    try:
        ray.shutdown()
    except Exception:  # pragma: no cover — shutdown is best-effort
        pass


# ---------------------------------------------------------------------------
# Actor classes (built dynamically so the ``@ray.remote`` decorator
# only fires once Ray is importable / installed).
# ---------------------------------------------------------------------------


def _build_layout_actor_class(num_cpus: float, num_gpus: float) -> Any:
    ray = _ensure_ray_initialised()

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    class _LayoutActor:
        """Ray actor wrapping a :class:`LayoutDetector` adapter.

        The adapter is instantiated inside the worker process so its
        weights load there (not in the driver). One actor → one model
        in VRAM.
        """

        def __init__(self, backend: str, device: str) -> None:
            # Imported here so the driver-side import graph stays clean.
            from arabic_pdf_transcribe.cli import _build_layout

            self._adapter = _build_layout(backend, device=device)
            if self._adapter is None:
                raise ModelDownloadError(
                    f"layout backend {backend!r} could not be constructed inside the Ray actor"
                )

        def detect(self, image: PILImage, page_index: int) -> list[Region]:
            assert self._adapter is not None
            return list(self._adapter.detect(image, page_index))  # type: ignore[attr-defined]

    return _LayoutActor


def _build_ocr_actor_class(num_cpus: float, num_gpus: float) -> Any:
    ray = _ensure_ray_initialised()

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    class _OCRActor:
        """Ray actor wrapping an :class:`OCRTranscriber` adapter.

        Exposes both ``transcribe`` (per-region) and ``transcribe_page``
        (per-page batched) so the pipeline's existing fast-path detection
        keeps working.
        """

        def __init__(
            self,
            backend: str,
            device: str,
            disable_formula: bool,
            batch_size: int | None,
        ) -> None:
            from arabic_pdf_transcribe.cli import _build_ocr

            self._adapter = _build_ocr(
                backend,
                device=device,
                disable_formula=disable_formula,
                batch_size=batch_size,
            )
            if self._adapter is None:
                raise ModelDownloadError(
                    f"OCR backend {backend!r} could not be constructed inside the Ray actor"
                )

        def transcribe(self, region: Region, image: PILImage) -> Region:
            assert self._adapter is not None
            return self._adapter.transcribe(region, image)  # type: ignore[attr-defined]

        def transcribe_page(
            self, regions: Sequence[Region], image: PILImage
        ) -> list[Region]:
            assert self._adapter is not None
            page_batch = getattr(self._adapter, "transcribe_page", None)
            if callable(page_batch):
                return list(page_batch(regions, image))
            # Fallback for adapters without transcribe_page (none ship with
            # the project right now, but the Protocol allows it).
            from arabic_pdf_transcribe.regions import RegionRole

            out: list[Region] = []
            for region in regions:
                if region.role is RegionRole.FIGURE:
                    out.append(region)
                    continue
                out.append(self._adapter.transcribe(region, image))  # type: ignore[attr-defined]
            return out

    return _OCRActor


# ---------------------------------------------------------------------------
# Driver-side proxies — implement the project Protocols transparently.
# ---------------------------------------------------------------------------


class RayLayoutProxy:
    """Driver-side stand-in for a :class:`LayoutDetector`.

    Forwards every ``detect`` call to a :class:`_LayoutActor` running
    in a Ray worker.
    """

    def __init__(
        self,
        *,
        backend: str,
        device: str,
        num_cpus: float = 1.0,
        num_gpus: float = 0.5,
    ) -> None:
        actor_cls = _build_layout_actor_class(num_cpus=num_cpus, num_gpus=num_gpus)
        self._ray = _ensure_ray_initialised()
        self._actor = actor_cls.remote(backend, device)

    def detect(self, page_image: PILImage, page_index: int) -> Sequence[Region]:
        return self._ray.get(self._actor.detect.remote(page_image, page_index))


class RayOCRProxy:
    """Driver-side stand-in for an :class:`OCRTranscriber`."""

    def __init__(
        self,
        *,
        backend: str,
        device: str,
        disable_formula: bool,
        batch_size: int | None,
        num_cpus: float = 1.0,
        num_gpus: float = 0.5,
    ) -> None:
        actor_cls = _build_ocr_actor_class(num_cpus=num_cpus, num_gpus=num_gpus)
        self._ray = _ensure_ray_initialised()
        self._actor = actor_cls.remote(backend, device, disable_formula, batch_size)

    def transcribe(self, region: Region, page_image: PILImage) -> Region:
        return self._ray.get(self._actor.transcribe.remote(region, page_image))

    def transcribe_page(
        self, regions: Sequence[Region], page_image: PILImage
    ) -> list[Region]:
        return self._ray.get(
            self._actor.transcribe_page.remote(list(regions), page_image)
        )


__all__ = ["RayLayoutProxy", "RayOCRProxy"]
