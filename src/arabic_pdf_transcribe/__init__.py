"""Arabic-first PDF transcriber.

Public surface is intentionally tiny in v0.1: importing this package must not
trigger heavy dependencies (``transformers`` / ``torch`` / ``Pillow``). The
ML adapters live behind lazy-imported submodules (``arabic_pdf_transcribe.layout``
and ``arabic_pdf_transcribe.ocr``) and are only loaded when those submodules
are explicitly imported.
"""

from arabic_pdf_transcribe._version import __version__

__all__ = ["__version__"]
