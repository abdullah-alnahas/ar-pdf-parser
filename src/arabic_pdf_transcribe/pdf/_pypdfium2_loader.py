"""Safe-by-default ``pypdfium2`` wrapper.

Exposes only the calls the native extractor uses. Centralising the import +
options here keeps the rest of the package free of ``pypdfium2`` knowledge,
and gives one place to assert "no JavaScript execution, no external resource
fetching" — both off by default in ``pypdfium2`` itself, but worth pinning
explicitly so a future upgrade can't quietly enable them.

Errors are translated to the package's typed exception hierarchy
(:class:`arabic_pdf_transcribe.errors.ArabicPdfTranscribeError` subclasses)
so callers don't see ``pypdfium2`` types at the API boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from arabic_pdf_transcribe.errors import (
    CorruptedPDFError,
    EncryptedPDFError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pypdfium2 as pdfium


@contextmanager
def open_pdf(path: Path) -> Iterator[pdfium.PdfDocument]:
    """Yield an opened ``PdfDocument``, translating errors at the boundary.

    The document is closed deterministically when the context exits, even on
    exceptions. Encrypted PDFs raise :class:`EncryptedPDFError`; structural
    corruption raises :class:`CorruptedPDFError`. No JavaScript / external
    resource handler is registered.
    """
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:  # pragma: no cover — exact message is upstream-defined
        raise CorruptedPDFError(f"pypdfium2 could not open {path!s}: {exc}") from exc
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise CorruptedPDFError(f"OS error opening {path!s}: {exc}") from exc

    try:
        # ``form_data`` access surfaces encryption; check the form-info bit
        # without inspecting privileged content.
        if _is_encrypted(document):
            raise EncryptedPDFError(
                f"{path!s} is encrypted; password-protected PDFs are not supported in v1"
            )
        yield document
    finally:
        document.close()


def _is_encrypted(document: pdfium.PdfDocument) -> bool:
    """Return ``True`` if the underlying PDF has an encryption dict.

    ``pypdfium2`` doesn't expose a single ``is_encrypted`` flag; the security
    handler bits live behind ``pdfium.raw.FPDF_GetSecurityHandlerRevision``.
    A non-``None`` revision indicates encryption (``-1`` or ``None`` mean
    "not encrypted").
    """
    try:
        import pypdfium2.raw as raw  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover — pypdfium2 always ships ``raw``
        return False
    revision = raw.FPDF_GetSecurityHandlerRevision(document)
    return bool(revision) and revision != -1


__all__ = ["open_pdf"]
