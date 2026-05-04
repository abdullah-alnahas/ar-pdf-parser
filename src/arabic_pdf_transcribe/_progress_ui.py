"""Two-bar live progress UI powered by ``rich``.

Driven by the per-event progress callback emitted by
:func:`arabic_pdf_transcribe.pipeline.transcribe`. Active only when
stderr is a TTY and the user did not pass ``--quiet`` /
``--json-logs``; otherwise the CLI falls back to the line-format
:class:`arabic_pdf_transcribe._logging.ProgressLogger`.

Layout
------

Two stacked bars on stderr::

   overall         ━━━━━━━━━━━━━━━━╸━━━━  37/200 • 00:42 • 03:50
   ocr             ━━╸━━━━━━━━━━━━━━━━━    7/180 • 00:11 • 04:30

* **overall** advances once per page that finishes (success or
  failure). Total = total pages of the input document.
* **phase** (validate → rasterise → layout → ocr) re-targets at every
  ``phase:start:<name>:<count>`` event from the pipeline. Per-event
  semantics within each phase:

  - ``validate``: advance on each ``start`` event.
  - ``rasterise``: advance on each ``rasterise`` event.
  - ``layout``: advance on each ``layout`` event.
  - ``ocr``: advance on each ``complete:ml`` event.

  Failures during a phase advance the phase bar too (the page is
  done, even if it failed) so totals stay accurate.

Failures decoupled from per-phase totals
----------------------------------------

If a page fails before reaching its expected phase, every later phase
bar is one short. We accept that — the *overall* bar is the
authoritative completion counter; the phase bar is a "what's the
machine doing right now" indicator and approximate by design.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, TextIO

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from rich.progress import Progress, TaskID


_PHASE_LABELS: dict[str, str] = {
    "validate": "validate",
    "rasterise": "rasterise",
    "layout": "layout",
    "ocr": "ocr",
}


@dataclass
class _State:
    """Mutable counters for the active phase + completed pages."""

    active_phase: str = "validate"
    completed_pages: set[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.completed_pages is None:
            self.completed_pages = set()


class RichTwoBarUI:
    """Render pipeline progress as two live bars on a TTY.

    Construct with the document's total page count and a TTY stream
    (typically ``sys.stderr``). Call :meth:`handle_event` from the
    pipeline's progress callback. Call :meth:`close` when the pipeline
    finishes (or use the :func:`build_ui` context manager which
    closes for you).
    """

    def __init__(self, total_pages: int, stream: TextIO) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        # ``force_terminal=True`` lets the caller inject any TextIO that
        # quacks like a TTY (e.g. a wrapping context). The CLI only
        # constructs us when ``stream.isatty()`` is true.
        self._console = Console(file=stream, force_terminal=True, soft_wrap=True)
        self._progress: Progress = Progress(
            TextColumn("[bold cyan]{task.description:>10}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
            refresh_per_second=8,
        )
        self._overall_id: TaskID = self._progress.add_task("overall", total=total_pages)
        self._phase_id: TaskID = self._progress.add_task("validate", total=total_pages)
        self._state = _State()
        self._progress.start()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_event(self, page_index: int, total: int, event: str) -> None:
        """Route one pipeline event to the right bar."""
        if event.startswith("phase:start:"):
            self._handle_phase_start(event)
            return
        if event == "start":
            # validate phase only emits ``start`` per page; layout/OCR
            # phases also see this on their ML pages but we count
            # validate-phase advances only.
            if self._state.active_phase == "validate":
                self._progress.advance(self._phase_id)
            return
        if event == "rasterise":
            if self._state.active_phase == "rasterise":
                self._progress.advance(self._phase_id)
            return
        if event == "layout":
            if self._state.active_phase == "layout":
                self._progress.advance(self._phase_id)
            return
        if event.startswith("region:"):
            # Per-region events are too noisy for the bar; ignore.
            return
        if event.startswith("complete:"):
            branch = event.split(":", 1)[1]
            self._mark_page_done(page_index)
            # Only the OCR phase increments on complete:ml; native
            # completes happen during validate phase.
            if branch == "ml" and self._state.active_phase == "ocr":
                self._progress.advance(self._phase_id)
            return
        if event.startswith("failure:"):
            self._mark_page_done(page_index)
            # A failure inside the active phase counts as that phase
            # finishing one more page (even if unsuccessfully).
            self._progress.advance(self._phase_id)
            return

    def close(self) -> None:
        """Stop the live display. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._progress.stop()

    def write_summary(self, line: str) -> None:
        """Print a final line below the bars (after :meth:`close`)."""
        self._console.print(line)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_phase_start(self, event: str) -> None:
        # ``phase:start:<name>:<count>``
        parts = event.split(":")
        if len(parts) != 4:  # pragma: no cover — defensive
            return
        name = parts[2]
        try:
            count = int(parts[3])
        except ValueError:  # pragma: no cover — defensive
            return
        label = _PHASE_LABELS.get(name, name)
        self._state.active_phase = name
        # Reset the phase task with the new total so completed-bar +
        # ETA reflect the current phase only.
        self._progress.reset(self._phase_id, description=label, total=count or 1)
        if count == 0:
            # Nothing to do in this phase — mark complete so the bar
            # doesn't sit at 0 forever.
            self._progress.update(self._phase_id, completed=1, total=1)

    def _mark_page_done(self, page_index: int) -> None:
        if page_index in self._state.completed_pages:
            return
        self._state.completed_pages.add(page_index)
        self._progress.advance(self._overall_id)


@contextmanager
def build_ui(total_pages: int, stream: TextIO) -> Iterator[RichTwoBarUI]:
    """Construct + auto-close a :class:`RichTwoBarUI`."""
    ui = RichTwoBarUI(total_pages=total_pages, stream=stream)
    try:
        yield ui
    finally:
        ui.close()


__all__ = ["RichTwoBarUI", "build_ui"]
