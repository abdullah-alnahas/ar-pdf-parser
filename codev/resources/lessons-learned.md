# Lessons learned

Project-level patterns harvested from the SPIR-1 build of
`arabic-pdf-transcribe` v0.1.0. Each entry is generalisable beyond
this project — anti-patterns, debugging insights, process notes.

## Patterns that worked

### Lazy-import discipline + subprocess regression tests

For packages with optional heavy dependencies (`transformers`, `torch`,
`Pillow`, `python-docx`), gating each import behind the function that
actually needs it kept `import arabic_pdf_transcribe` fast and
side-effect-free. The discipline was enforced by **subprocess-isolated**
tests that imported the package in a fresh interpreter and asserted no
heavy module appeared in `sys.modules`. In-process tests would have
been polluted by other tests' imports — subprocess isolation is the
only reliable check.

### Protocol boundaries between adapters and orchestrator

`LayoutDetector`, `OCRTranscriber`, `Validator`, and emitter callables
are all `typing.Protocol` shapes. Tests inject stubs that conform
structurally; production wiring uses the HF adapters. This let the
phase-8 orchestrator land + be exhaustively tested before the
phase-9 corpus existed.

### Per-phase PR + 3-way consult per phase

Building one PR per phase (instead of one mega-PR) with a 3-way
consult (codex / claude / gemini) at each one caught real bugs that a
single end-of-project review would have missed:

- Phase 7: orphan grouped captions silently dropped (codex).
- Phase 8: `--max-workers` parsed but discarded; `--format docx`
  without `-o` raised in the writer instead of `validate_args`; CUDA
  OOM not caught (both reviewers).
- Phase 9: missing perf smoke, missing scenario-12/15 tests,
  models.toml/docs footprint inconsistency (both reviewers).

The architect-approved 2/3 acceptance pattern (codex + claude when
gemini hits its quota) kept the loop unblocked without compromising
review breadth.

### Architect carry-over notes between phases

Reviewing PR #N often surfaced concerns that didn't block PR #N but
mattered for phase M > N. Persisting these in
`~/.claude/projects/.../memory/architect_future_notes.md` made them
visible to whichever builder picked up the next phase, even across
session compactions. The carry-overs from PRs #4 (NFC/NFKC), #5
(footnote / morphology), and #9 (5 polish items) all landed cleanly
in their target phases because the notes survived.

## Anti-patterns avoided

### Materialising iterators just to count them

Phase 8's first cut of the orchestrator did
`list(extract_native(pdf_path))` to know the page count for progress
reporting — but that defeated the page filter (every page got
text-extracted before being thrown away). Codex flagged it on
review; phase 9 fixed it by opening the PDF once for `len(document)`
and streaming the iterator with the page filter pushed into
`extract_native_from_document`. Lesson: always check whether the
"materialise just for length" was needed at all.

### Bare-except as catch-all without justification

Phase 8's per-page boundary used `except Exception` as the catch-all
for the best-effort contract. Both PR reviewers flagged the breadth.
Phase 9 added an explicit comment (and the architect filed it as a
carry-over to ensure it landed) explaining the trade-off and
pointing at the typed-exception arms above it. Lesson: bare-except
is sometimes correct, but always needs a code comment that says
*why*; otherwise reviewers (rightly) treat it as a smell.

### Premature optimisation in CI

Phase 9 added a `nightly.yml` workflow with HF cache for slow tests
instead of putting them in the main CI matrix. Result: PR CI stays
fast (no model downloads), nightly catches real-model regressions.
Lesson: not every test belongs on every PR. Two workflows are
better than one expensive one.

### Reusing the same pyright cache across worktree files

Pyright's caching surfaced "could not be resolved" warnings for valid
imports throughout the build, even when `python -c "import ..."`
worked. The diagnostics were noise; ignored them. Lesson: pyright
cache invalidation is opaque — trust the actual interpreter output
over the diagnostic stream when they disagree.

## Process improvements for SPIR

### Architect carry-over file is a near-mandatory pattern

The `architect_future_notes.md` memory file accumulated five sets of
carry-overs across the build (PR #4, #5, #9 all flagged items for
later phases). Without persistent storage these would have been
forgotten across session compactions. Recommendation: every SPIR
project gets one carry-over file by default, indexed in `MEMORY.md`.

### Squash merges silently dropping changes

PR #8's squash merge dropped `pyproject.toml` from the merge
commit even though it was staged on the PR branch. Recovery was
trivial (re-add in next phase) but the surprise cost a debugging
loop. Recommendation: after `gh pr merge --squash` always run
`git diff <prev-main>..<new-main> -- <file>` for any non-source
file the PR touched, to catch the dropped-file pattern early.

### Three-way consult parallel with `&`-subshell + `wait` is fragile

Two attempts at running `consult -m X &; consult -m Y &; consult -m
Z &; wait` returned `DONE` immediately because the parent's `wait`
saw no jobs (the subshells already detached). Serial consults
worked. Recommendation: drop the parallel-consult pattern; serial is
~3x slower wall-clock but actually finishes.

## Debugging insights

### Test stubs vs real adapters at integration boundaries

The phase-8 pipeline tests use stub `LayoutDetector` / `OCRTranscriber`
that don't actually rasterise — so when the orchestrator started
sharing a single document handle in phase 9, the stubs needed no
update. Decoupling at Protocol boundaries paid off twice: once for
test speed, once for refactor safety.

### Synthetic CUDA OOM testing without torch

Detecting `torch.cuda.OutOfMemoryError` without importing `torch` —
inspecting `__module__` and `__name__` on the exception class —
kept the orchestrator light AND made testing trivial. The test
constructs a synthetic exception class with the matching attributes:

```python
class _SyntheticCudaOOM(RuntimeError):
    pass
_SyntheticCudaOOM.__module__ = "torch.cuda"
_SyntheticCudaOOM.__name__ = "OutOfMemoryError"
raise _SyntheticCudaOOM("CUDA out of memory")
```

Lesson: when you can't take a hard dependency, dispatching on class
metadata is a viable alternative to `isinstance`.

### Deterministic Word output via document.xml comparison

`python-docx` writes timestamps into `core.xml` so the whole zip
isn't byte-identical across runs. But `word/document.xml` (the body)
is — and that's the surface that matters for snapshot tests.
Reading just the relevant member out of the zip and comparing bytes
is the cleanest determinism contract for python-docx output.
