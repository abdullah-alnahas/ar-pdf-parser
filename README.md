# arabic-pdf-transcribe

Arabic-first PDF transcriber. Native text extraction first; layout detection +
OCR fallback when the text layer is missing or broken. RTL-aware reading order.
Emits Markdown and Word.

**Status:** alpha — phase 1 of 9 (skeleton + license-audit harness).
See [`codev/specs/1-arabic-pdf-transcriber-extract.md`](codev/specs/1-arabic-pdf-transcriber-extract.md)
for the full specification and
[`codev/plans/1-arabic-pdf-transcriber-extract.md`](codev/plans/1-arabic-pdf-transcriber-extract.md)
for the per-phase implementation plan.

## Install (from source, dev mode)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The optional ``ml`` extra pulls in ``transformers``, ``torch``, ``Pillow``, and
``huggingface-hub``; these are not required to install or run the deterministic
native-extraction path.

## Local checks

```bash
make lint    # ruff check + format check
make test    # pytest with coverage
make audit   # license audit (runtime deps + models.toml)
make all     # run all of the above
```

## License

MIT — see [LICENSE](LICENSE).
