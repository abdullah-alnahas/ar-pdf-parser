"""Pipeline progress logging.

The orchestrator emits one event per page (start, complete, failure)
plus a final summary. The same event stream feeds three outputs:

* **Default text mode** (``ProgressMode.TEXT``) — human-readable
  ``"page X of N — extracted (native)"`` lines on stderr.
* **Quiet mode** (``ProgressMode.QUIET``) — no output. The pipeline
  still emits events; the logger swallows them.
* **JSON mode** (``ProgressMode.JSON``) — one ``json.dumps`` line per
  event on stderr. Schema:
  ``{"page": int, "of": int, "event": str, "branch": str|None,
  "reason": str|None, "ts": null}``. Timestamps are intentionally
  absent so the JSON stream is byte-stable across runs (same input
  → same stderr).

The logger writes to ``stderr`` so primary output (Markdown to stdout
or the docx file path) is never polluted by progress noise. It does
not buffer: each event is flushed so a long-running run shows
progress immediately.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TextIO


class ProgressMode(Enum):
    TEXT = "text"
    QUIET = "quiet"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One observable event in the pipeline's progress stream."""

    page: int  # 1-based
    of: int  # total page count
    event: str  # "start" | "complete" | "failure" | "summary" | "layout" | "region"
    branch: str | None = None  # "native" | "ml" | None
    reason: str | None = None  # populated on failure / summary
    region: int | None = None  # 1-based region index within page (issue #18)
    of_regions: int | None = None  # region count within page
    role: str | None = None  # region role label, e.g. "PARAGRAPH"


class ProgressLogger:
    """Render :class:`ProgressEvent`s to a text stream.

    Construct with the chosen :class:`ProgressMode` and a stream
    (defaults to ``sys.stderr``). Tests inject a ``StringIO`` to
    capture output deterministically.
    """

    def __init__(
        self,
        mode: ProgressMode = ProgressMode.TEXT,
        *,
        stream: TextIO | None = None,
    ) -> None:
        self.mode = mode
        self._stream: TextIO = stream if stream is not None else sys.stderr

    def emit(self, event: ProgressEvent) -> None:
        if self.mode is ProgressMode.QUIET:
            return
        if self.mode is ProgressMode.JSON:
            self._stream.write(json.dumps(_event_to_dict(event), ensure_ascii=False) + "\n")
        else:
            self._stream.write(_format_text(event) + "\n")
        self._stream.flush()

    # Convenience helpers — keep the orchestrator one-liner-friendly.

    def start(self, *, page: int, of: int, branch: str | None = None) -> None:
        self.emit(ProgressEvent(page=page, of=of, event="start", branch=branch))

    def complete(self, *, page: int, of: int, branch: str) -> None:
        self.emit(ProgressEvent(page=page, of=of, event="complete", branch=branch))

    def failure(self, *, page: int, of: int, reason: str, branch: str | None = None) -> None:
        self.emit(
            ProgressEvent(
                page=page,
                of=of,
                event="failure",
                branch=branch,
                reason=reason,
            )
        )

    def summary(self, *, of: int, ok_pages: int, failed_pages: int) -> None:
        self.emit(
            ProgressEvent(
                page=of,
                of=of,
                event="summary",
                reason=f"ok={ok_pages} failed={failed_pages}",
            )
        )

    def layout(self, *, page: int, of: int) -> None:
        """Page entered the ML layout-detection step (issue #18)."""
        self.emit(ProgressEvent(page=page, of=of, event="layout"))

    def region(self, *, page: int, of: int, region: int, of_regions: int, role: str) -> None:
        """One OCR region is about to run (issue #18 progress visibility)."""
        self.emit(
            ProgressEvent(
                page=page,
                of=of,
                event="region",
                region=region,
                of_regions=of_regions,
                role=role,
            )
        )


def _event_to_dict(event: ProgressEvent) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "page": event.page,
        "of": event.of,
        "event": event.event,
        "branch": event.branch,
        "reason": event.reason,
    }
    if event.event == "region":
        payload["region"] = event.region
        payload["of_regions"] = event.of_regions
        payload["role"] = event.role
    return payload


def _format_text(event: ProgressEvent) -> str:
    if event.event == "summary":
        return f"summary: {event.of} pages, {event.reason}"
    page_n = f"page {event.page} of {event.of}"
    if event.event == "start":
        branch = f" [{event.branch}]" if event.branch else ""
        return f"{page_n} — start{branch}"
    if event.event == "complete":
        return f"{page_n} — complete ({event.branch})"
    if event.event == "failure":
        return f"{page_n} — FAILED: {event.reason}"
    if event.event == "layout":
        return f"{page_n} — layout (ml)"
    if event.event == "region":
        return f"{page_n} — region {event.region}/{event.of_regions} ({event.role})"
    return f"{page_n} — {event.event}"  # pragma: no cover — defensive


__all__ = ["ProgressEvent", "ProgressLogger", "ProgressMode"]
