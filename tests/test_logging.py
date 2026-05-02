"""Phase-8 progress logger tests.

The logger is a small adapter from :class:`ProgressEvent`s to a text
stream in one of three modes (text / quiet / JSON). All tests use a
captured stream so we never write to real stderr.
"""

from __future__ import annotations

import io
import json

from arabic_pdf_transcribe._logging import (
    ProgressEvent,
    ProgressLogger,
    ProgressMode,
)


def _logger(mode: ProgressMode) -> tuple[ProgressLogger, io.StringIO]:
    buf = io.StringIO()
    return ProgressLogger(mode, stream=buf), buf


def test_text_mode_emits_human_readable_lines() -> None:
    log, buf = _logger(ProgressMode.TEXT)
    log.start(page=1, of=3, branch="native")
    log.complete(page=1, of=3, branch="native")
    out = buf.getvalue()
    assert "page 1 of 3" in out
    assert "start" in out
    assert "complete (native)" in out


def test_quiet_mode_emits_nothing() -> None:
    log, buf = _logger(ProgressMode.QUIET)
    log.start(page=1, of=3)
    log.complete(page=1, of=3, branch="ml")
    log.failure(page=1, of=3, reason="oom")
    log.summary(of=3, ok_pages=2, failed_pages=1)
    assert buf.getvalue() == ""


def test_json_mode_emits_one_object_per_line() -> None:
    log, buf = _logger(ProgressMode.JSON)
    log.start(page=2, of=5, branch="native")
    log.failure(page=2, of=5, reason="ocr_failed")
    lines = [line for line in buf.getvalue().splitlines() if line]
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {"page": 2, "of": 5, "event": "start", "branch": "native", "reason": None}
    second = json.loads(lines[1])
    assert second["event"] == "failure"
    assert second["reason"] == "ocr_failed"


def test_json_mode_byte_stable_across_runs() -> None:
    """Acceptance: JSON stream has no timestamps; same input → same bytes."""
    a_log, a_buf = _logger(ProgressMode.JSON)
    b_log, b_buf = _logger(ProgressMode.JSON)
    for log in (a_log, b_log):
        log.start(page=1, of=2, branch="native")
        log.complete(page=1, of=2, branch="native")
        log.summary(of=2, ok_pages=2, failed_pages=0)
    assert a_buf.getvalue() == b_buf.getvalue()


def test_summary_event_text_format() -> None:
    log, buf = _logger(ProgressMode.TEXT)
    log.summary(of=10, ok_pages=8, failed_pages=2)
    out = buf.getvalue().strip()
    assert "summary" in out
    assert "10 pages" in out
    assert "ok=8" in out
    assert "failed=2" in out


def test_failure_text_includes_reason() -> None:
    log, buf = _logger(ProgressMode.TEXT)
    log.failure(page=4, of=10, reason="OCRTranscriptionError")
    out = buf.getvalue().strip()
    assert "page 4 of 10" in out
    assert "FAILED" in out
    assert "OCRTranscriptionError" in out


def test_event_dataclass_is_immutable() -> None:
    event = ProgressEvent(page=1, of=2, event="start")
    import dataclasses

    assert dataclasses.is_dataclass(event)
    # frozen=True means setattr raises
    try:
        event.page = 99  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("ProgressEvent should be frozen")
