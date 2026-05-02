"""License audit for runtime distributions and pinned ML models.

Walks every distribution installed in the active environment, normalises its
declared license, and refuses any entry whose normalised license is not on the
project's allow-list. Likewise reads ``models.toml`` and validates that every
pinned ML model entry has the required keys (``name``, ``revision``,
``license``, ``stage``) and that the license is on the same allow-list.

Stdlib only — no third-party dependencies — so the audit can run before the
package's own runtime deps are installed.

Exit codes:
    0   audit passed
    1   audit failed (one or more violations)
    2   internal error (bad ``models.toml`` schema, missing files, etc.)

Usage::

    python tools/license_audit.py
    python tools/license_audit.py --models tools/../models.toml
    python tools/license_audit.py --allow MIT --allow Apache-2.0   # restrict
    python tools/license_audit.py --overrides tools/license_audit_overrides.toml
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

DEFAULT_ALLOW = (
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
    "PSF-2.0",
    "Unlicense",
    "0BSD",
    # HPND ("MIT-CMU style") is the historical Pillow license: an
    # OSI-approved permissive license functionally equivalent to MIT
    # plus a no-endorsement clause. Adding it as a first-class entry
    # rather than an override because the [ml] extra makes Pillow a
    # shipped runtime dep of the project.
    "HPND",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_TOML = REPO_ROOT / "models.toml"
DEFAULT_OVERRIDES_TOML = REPO_ROOT / "tools" / "license_audit_overrides.toml"

# Maps loose, free-form license labels to their SPDX-style canonical name. Only
# obvious aliases are listed; anything not matched falls through unchanged and
# the allow-list comparison decides.
ALIASES: dict[str, str] = {
    "mit license": "MIT",
    "mit": "MIT",
    "mit-cmu": "HPND",  # Pillow's classifier label is "MIT-CMU"; SPDX is HPND.
    "hpnd": "HPND",
    "historical permission notice and disclaimer (hpnd)": "HPND",
    "apache license, version 2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "the apache software license, version 2.0": "Apache-2.0",
    "bsd license": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "bsd 2-clause": "BSD-2-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "isc license": "ISC",
    "isc": "ISC",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "mpl-2.0": "MPL-2.0",
    "python software foundation license": "PSF-2.0",
    "psf": "PSF-2.0",
    "psf-2.0": "PSF-2.0",
    "the unlicense (unlicense)": "Unlicense",
    "unlicense": "Unlicense",
    "0bsd": "0BSD",
}

CLASSIFIER_PREFIX = "License :: OSI Approved :: "

CLASSIFIER_MAP: dict[str, str] = {
    "MIT License": "MIT",
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
}


@dataclass(frozen=True)
class Violation:
    kind: str  # "dist" or "model" or "schema"
    name: str
    detail: str

    def format(self) -> str:
        return f"  [{self.kind}] {self.name}: {self.detail}"


def _normalise(raw: str | None) -> str | None:
    """Normalise a free-form license string to its canonical form, or None."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    # Some packages put the entire license text in the License field; if it's
    # very long, fall back to ``None`` and rely on classifiers instead.
    if "\n" in cleaned or len(cleaned) > 200:
        return None
    key = cleaned.casefold()
    return ALIASES.get(key, cleaned)


def _classifier_license(classifiers: Iterable[str]) -> str | None:
    """Pull the most specific license out of a Trove classifier list."""
    candidates: list[str] = []
    for raw in classifiers:
        if not raw.startswith(CLASSIFIER_PREFIX):
            continue
        tail = raw[len(CLASSIFIER_PREFIX) :]
        if tail in CLASSIFIER_MAP:
            candidates.append(CLASSIFIER_MAP[tail])
        else:
            normalised = _normalise(tail)
            if normalised is not None:
                candidates.append(normalised)
    # If the package is dual-licensed via classifiers, prefer the first
    # SPDX-canonical entry.
    return candidates[0] if candidates else None


def _expression_license(expr: str | None) -> list[str]:
    """Split a PEP 639 license expression into individual SPDX-style clauses.

    The expression grammar is the SPDX one: clauses joined by ``OR`` / ``AND``
    with optional grouping parens. We split on the operators and strip parens;
    we don't enforce the AND/OR semantics — the caller decides whether *any*
    clause being allow-listed is enough (the project policy: yes, it is).
    """
    if not expr:
        return []
    cleaned = expr.strip()
    parts = re.split(r"\s+(?:OR|AND)\s+", cleaned)
    return [p.strip().strip("()").strip() for p in parts if p.strip()]


def _dist_licenses(dist: metadata.Distribution) -> list[str]:
    """Return every plausible canonical license for a distribution.

    The metadata may declare more than one (PEP 639 expression, classifiers,
    free-form ``License``); the audit accepts the dist if *any* declared
    clause is on the allow-list, mirroring the SPDX ``OR`` semantics that
    upstream projects use to grant the consumer a choice.
    """
    out: list[str] = []
    meta = dist.metadata
    expression = meta.get("License-Expression") or meta.get("License-File-Expression")
    if expression:
        for clause in _expression_license(expression):
            normalised = _normalise(clause)
            if normalised is not None:
                out.append(normalised)
    classifiers = meta.get_all("Classifier") or []
    cls = _classifier_license(classifiers)
    if cls is not None:
        out.append(cls)
    raw = meta.get("License")
    if raw:
        # The free-form License field sometimes carries an SPDX-style expression
        # too (older sdists). Try splitting it as well; fall back to a direct
        # normalise on the whole string.
        clauses = _expression_license(raw)
        if len(clauses) > 1:
            for clause in clauses:
                normalised = _normalise(clause)
                if normalised is not None:
                    out.append(normalised)
        else:
            normalised = _normalise(raw)
            if normalised is not None:
                out.append(normalised)
    # De-dup while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for entry in out:
        if entry not in seen:
            seen.add(entry)
            unique.append(entry)
    return unique


def _dist_license(dist: metadata.Distribution) -> str | None:
    """Backward-compatible single-license helper for tests / callers."""
    licenses = _dist_licenses(dist)
    return licenses[0] if licenses else None


PROJECT_DIST_NAME = "arabic-pdf-transcribe"

# Optional-dependency groups that are *shipped* to end users. Phase 1 does not
# audit any extras — the runtime closure (``pypdfium2`` + ``python-docx``) is
# the only shipped surface so far. When phases 4 and 5 introduce the [ml]
# extra (transformers / torch / Pillow), they will add ``"ml"`` to this set
# and at the same time land the documented overrides for the NVIDIA CUDA
# bundle (which carries the redistributable NVIDIA EULA, not an SPDX license).
# The CLI ``--include-extras NAME`` flag lets a developer audit any extra ad
# hoc without changing this list.
DEFAULT_SHIPPED_EXTRAS: frozenset[str] = frozenset()

# Names that appear as marker-only requirements ("extra == 'foo'") and resolve
# to nothing — skip rather than treat as missing distributions.
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")
_EXTRA_MARKER_RE = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")


def _requirement_name(spec: str) -> str | None:
    match = _REQ_NAME_RE.match(spec)
    if match is None:
        return None
    return match.group(1).strip().lower()


def _extra_marker(req_line: str) -> str | None:
    """Return the extra name a requirement is gated on, or ``None``."""
    match = _EXTRA_MARKER_RE.search(req_line)
    return match.group(1) if match else None


def _resolve_dep_closure(
    root: str,
    *,
    shipped_extras: frozenset[str] = DEFAULT_SHIPPED_EXTRAS,
) -> set[str]:
    """Return the lowercased names of every shipped distribution reachable from ``root``.

    The walker follows two kinds of requirement lines:

    1. **Unconditional** requirements (no ``extra == ...`` marker) — always
       included; these are the project's runtime deps.
    2. **Extras-gated** requirements where the extra name is in
       ``shipped_extras`` — included; these are optional-but-shipped deps
       (``[ml]`` for the Hugging Face stack).

    Dev-only extras (``[dev]``) are deliberately excluded: ``pytest`` /
    ``ruff`` / ``coverage`` are tooling, never shipped to end users, and so
    not part of the project's license-compliance surface.

    Once an unconditional requirement is followed into a transitive
    distribution, that distribution's full ``requires`` list is walked
    unconditionally (the dependency's own extras gate themselves; if it has
    a hard dep on a copyleft library, we want to see it).
    """
    seen: set[str] = set()
    queue: list[str] = [root.lower()]
    is_root = True
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            is_root = False
            continue
        for raw in dist.requires or []:
            extra = _extra_marker(raw)
            # Only follow extras-gated requirements when the gate names a
            # shipped extra AND we're at the project root. Transitive deps'
            # own optional-extras are the user's concern.
            if extra is not None and (not is_root or extra not in shipped_extras):
                continue
            dep = _requirement_name(raw)
            if dep and dep not in seen:
                queue.append(dep)
        is_root = False
    return seen


def iter_distributions(
    *, shipped_extras: frozenset[str] = DEFAULT_SHIPPED_EXTRAS
) -> Iterator[metadata.Distribution]:
    """Yield distributions in the project's dependency closure.

    Skips the project itself. Falls back to "everything installed" only when
    the project is not installed (e.g. running from a CI shell before
    ``pip install -e .``).
    """
    closure = _resolve_dep_closure(PROJECT_DIST_NAME, shipped_extras=shipped_extras)
    if not closure or closure == {PROJECT_DIST_NAME}:
        # Fallback: project not installed yet. Audit everything to fail loudly
        # rather than silently passing.
        for dist in metadata.distributions():
            name = (dist.metadata.get("Name") or "").strip().lower()
            if name == PROJECT_DIST_NAME:
                continue
            yield dist
        return

    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip().lower()
        if name == PROJECT_DIST_NAME:
            continue
        if name not in closure:
            continue
        yield dist


def audit_distributions(
    allow: frozenset[str],
    overrides: Mapping[str, str],
    *,
    shipped_extras: frozenset[str] = DEFAULT_SHIPPED_EXTRAS,
) -> list[Violation]:
    violations: list[Violation] = []
    for dist in iter_distributions(shipped_extras=shipped_extras):
        name = dist.metadata.get("Name") or "<unknown>"
        version = dist.metadata.get("Version") or "0"
        key = f"{name.lower()}=={version}"

        licenses = [overrides[key]] if key in overrides else _dist_licenses(dist)

        if not licenses:
            violations.append(
                Violation(
                    kind="dist",
                    name=key,
                    detail=(
                        "no license metadata found and no override entry; "
                        "add an entry to tools/license_audit_overrides.toml "
                        "with documented justification, or upgrade the package"
                    ),
                )
            )
            continue
        if not any(lic in allow for lic in licenses):
            joined = " / ".join(licenses)
            violations.append(
                Violation(
                    kind="dist",
                    name=key,
                    detail=f"no allow-listed license among {{{joined}}}",
                )
            )
    return violations


_REQUIRED_MODEL_KEYS = ("name", "revision", "license", "stage")


def audit_models(
    models_toml: Path,
    allow: frozenset[str],
) -> list[Violation]:
    if not models_toml.exists():
        return [Violation("schema", str(models_toml), "file does not exist")]
    raw = tomllib.loads(models_toml.read_text(encoding="utf-8"))
    entries = raw.get("models", [])
    if not isinstance(entries, list):
        return [Violation("schema", str(models_toml), "'models' must be a list")]
    violations: list[Violation] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            violations.append(Violation("schema", f"models[{index}]", "entry must be a table"))
            continue
        for required in _REQUIRED_MODEL_KEYS:
            if required not in entry:
                violations.append(
                    Violation(
                        kind="model",
                        name=entry.get("name", f"models[{index}]"),
                        detail=f"missing required key {required!r}",
                    )
                )
        license_id = _normalise(entry.get("license")) if entry.get("license") else None
        if license_id is None:
            if "license" in entry:
                violations.append(
                    Violation(
                        kind="model",
                        name=entry.get("name", f"models[{index}]"),
                        detail=f"could not normalise license {entry['license']!r}",
                    )
                )
            continue
        if license_id not in allow:
            violations.append(
                Violation(
                    kind="model",
                    name=entry["name"],
                    detail=f"license {license_id!r} not in allow-list",
                )
            )
    return violations


def load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    overrides = raw.get("overrides", [])
    if not isinstance(overrides, list):
        return {}
    for entry in overrides:
        if not isinstance(entry, dict):
            continue
        key = entry.get("dist")
        license_id = entry.get("license")
        if isinstance(key, str) and isinstance(license_id, str):
            out[key.lower()] = _normalise(license_id) or license_id
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="license_audit",
        description="Audit runtime distributions and pinned ML models for license compliance.",
    )
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS_TOML)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES_TOML)
    parser.add_argument(
        "--allow",
        action="append",
        help="Restrict allow-list to the licenses passed via repeated --allow flags.",
    )
    parser.add_argument(
        "--include-extras",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Audit the named optional-dependency extra in addition to the runtime "
            "closure. Pass repeatedly to include multiple extras. Phase 1 default "
            "is empty; phases 4 and 5 will pass --include-extras=ml in CI once the "
            "[ml] extra carries pinned model dependencies."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only emit output when there are violations.",
    )
    args = parser.parse_args(argv)

    allow = frozenset(args.allow) if args.allow else frozenset(DEFAULT_ALLOW)
    overrides = load_overrides(args.overrides)
    shipped_extras = (
        frozenset(args.include_extras) if args.include_extras else DEFAULT_SHIPPED_EXTRAS
    )

    try:
        dist_violations = audit_distributions(allow, overrides, shipped_extras=shipped_extras)
        model_violations = audit_models(args.models, allow)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"license_audit: internal error: {exc}", file=sys.stderr)
        return 2

    violations = dist_violations + model_violations
    if violations:
        print("license_audit: violations found", file=sys.stderr)
        for v in violations:
            print(v.format(), file=sys.stderr)
        return 1

    if not args.quiet:
        print("license_audit: OK", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
