"""Multi-signal native-text quality validator.

The validator decides per-page whether the native extraction's text is
trustworthy. The decision is the conjunction of independent signals
(any one signal may flag the page):

1. **Arabic codepoint ratio** -- the share of letter-class codepoints that
   fall in the Arabic Unicode blocks. A page that *should* be Arabic but
   yields almost no Arabic codepoints (mojibake encoding swap) fails this
   signal.
2. **Replacement-glyph ratio** -- the share of codepoints that are U+FFFD,
   private-use, ASCII control characters, or known no-glyph placeholders.
   Healthy text has near-zero here; broken text-layers (glyph-id-not-Unicode
   mappings) produce significant amounts. Foulabook-style PDFs serialize
   font-specific glyph IDs as ASCII control codepoints (\\x01..\\x1F); those
   are counted here so the signal flags such pages.
3. **Word-boundary plausibility** -- KL divergence between the page's
   token-length distribution and a small bundled Arabic reference
   distribution. Cleanly extracted Arabic produces a distribution close
   to the reference; missing-spaces or every-char-its-own-word patterns
   diverge sharply.
4. **Presentation-form ratio** -- among Arabic letters, the share that fall
   in the Arabic Presentation Forms blocks (FB50-FDFF, FE70-FEFF). A few
   ligatures are normal; an overwhelming majority indicates a visual-order
   text layer (broken, e.g. body text serialized as shaped glyphs rather
   than logical-order base codepoints).

A page with no text layer (``has_text_layer == False``) bypasses the
signals and is rejected with a single deterministic reason.

Threshold tuning is a phase-3 deliverable -- the rule **shape** is locked,
but the numeric thresholds are revisited in phase 9 against the broader
corpus. The values here are derived from the in-tree fixtures and
documented per signal below.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arabic_pdf_transcribe.extract.native import NativePage

REFERENCE_DIST_PATH = Path(__file__).resolve().parent / "_reference_dist.json"

# Arabic Unicode block ranges we treat as "Arabic letters" for the codepoint
# ratio signal. Includes the basic block plus the Supplement and Extended-A
# blocks; presentation forms (FB50-FDFF / FE70-FEFF) are NOT counted because
# their presence usually indicates a bidi-shaper-rendered text layer, which
# is fine but does not by itself prove the underlying text is healthy.
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
)

# Arabic Presentation Forms: legitimate ligatures + visual-order shaped
# glyphs. A few of these in body text is normal; an overwhelming majority
# indicates a broken (visual-order) text layer.
_ARABIC_PRESENTATION_RANGES: tuple[tuple[int, int], ...] = (
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

# Whitespace control codepoints we always preserve (skip from
# replacement-ratio counting). Other Cc codepoints (\\x01..\\x08, \\x0B,
# \\x0C, \\x0E..\\x1F, \\x7F, \\x80..\\x9F) indicate a glyph-id-not-Unicode
# encoding leak.
_WHITESPACE_CONTROL = frozenset({"\t", "\n", "\v", "\f", "\r"})

# Replacement-glyph code points: U+FFFD plus the Private Use Area. Some
# broken text layers produce private-use codepoints when glyph ID is not
# mapped to Unicode.
_REPLACEMENT_RANGES: tuple[tuple[int, int], ...] = (
    (0xFFFD, 0xFFFD),
    (0xE000, 0xF8FF),  # Private Use Area
    (0xF0000, 0xFFFFD),  # Supplementary Private Use Area-A
    (0x100000, 0x10FFFD),  # Supplementary Private Use Area-B
    # Geometric-shapes block: a common ``.notdef`` glyph fallback when a
    # font cannot render the requested codepoint and the host extracts
    # via an explicit "no-glyph" indicator. We treat U+25A0 specifically
    # plus a few neighbouring sentinels (U+25A1 outline square, U+2588
    # full block, U+25CF black circle).
    (0x25A0, 0x25A1),
    (0x25CB, 0x25CF),
    (0x2588, 0x2588),
)

# Pre-compiled token splitter (whitespace, after NFC normalisation).
_TOKEN_SPLIT = re.compile(r"\s+", flags=re.UNICODE)


# ---- Public types ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidatorConfig:
    """Per-signal accept thresholds.

    Defaults derived from the in-tree fixtures and documented per signal
    in the module docstring. Override values via the constructor or via
    ``ValidatorConfig.from_toml`` for ``--config`` (phase 8).
    """

    # Minimum acceptable Arabic-letter share among letter-class codepoints
    # on a page that *appears* to contain Arabic text. A page with no
    # Arabic at all (pure Latin) is exempt — see ``validate_page``.
    min_arabic_ratio: float = 0.5

    # Maximum acceptable replacement-glyph ratio of total codepoints.
    max_replacement_ratio: float = 0.05

    # Maximum acceptable KL divergence (nats) of the page's token-length
    # distribution against the bundled Arabic reference. Above this the
    # signal flags the page. Tuned against the in-tree fixtures: clean
    # Latin pages (lorem-ar-2col) sit around 0.74; the mojibake fixture
    # at 1.0+. Phase 9 will retune against the broader corpus.
    max_word_boundary_kl: float = 1.0

    # Maximum acceptable share of Arabic letters that fall in the
    # Presentation Forms blocks. A handful of ligatures is normal; pages
    # whose body text is overwhelmingly shaped/visual-order glyphs
    # (broken layers) trip this gate. Pages with no Arabic at all
    # abstain. Tuned against the in-tree fixtures + the Foulabook-class
    # broken-layer regression fixture.
    max_presentation_form_ratio: float = 0.5

    # Minimum number of letter codepoints required before the Arabic-ratio
    # signal participates; below this the signal abstains (returns the
    # midpoint of its accept band) so very short pages don't trip the
    # gate purely on small sample size.
    min_letter_count: int = 30

    @classmethod
    def from_toml(cls, path: Path) -> ValidatorConfig:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        section = raw.get("validator", raw)
        return cls(**{k: v for k, v in section.items() if k in cls.__dataclass_fields__})

    def to_toml(self) -> str:
        # Inline emitter — the standard library only ships a TOML *reader*.
        # The output is round-trip stable through ``from_toml``.
        lines = ["[validator]"]
        for fname in self.__dataclass_fields__:
            value = getattr(self, fname)
            lines.append(f"{fname} = {value!r}")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    accept: bool
    signals: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


# ---- Signal 1: Arabic codepoint ratio ------------------------------------


def arabic_codepoint_ratio(text: str) -> float:
    """Return the share of letter codepoints that are Arabic.

    "Letter" is determined via Unicode general-category prefix ``L``. Pages
    with no letters at all return ``0.0`` (handled as "abstain" by the
    page-level validator below).
    """
    arabic = 0
    letters = 0
    for ch in text:
        if not _is_letter(ch):
            continue
        letters += 1
        if _in_ranges(ord(ch), _ARABIC_RANGES):
            arabic += 1
    if letters == 0:
        return 0.0
    return arabic / letters


# ---- Signal 2: replacement-glyph ratio -----------------------------------


def replacement_glyph_ratio(text: str) -> float:
    """Return the share of codepoints that look like glyph-fallback artefacts.

    Counts U+FFFD, Private Use Areas, ASCII control codepoints (other
    than tabs/newlines), and known no-glyph placeholders. Whitespace
    and ASCII punctuation are excluded from the denominator so a sparse
    page with one bad glyph does not trigger purely because the page is
    short. ASCII control codepoints (e.g. ``\\x01``..``\\x1F`` excluding
    common whitespace) are counted as replacement: legitimate text never
    contains them, but Foulabook-style broken text layers serialize
    font-specific glyph IDs as raw ASCII control bytes.
    """
    total = 0
    bad = 0
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        is_control = unicodedata.category(ch) == "Cc" and ch not in _WHITESPACE_CONTROL
        if cp < 0x80 and not ch.isalnum() and not is_control:
            # ASCII punctuation/symbols: ignore. Control chars below
            # fall through to the replacement count.
            continue
        total += 1
        if is_control or _in_ranges(cp, _REPLACEMENT_RANGES):
            bad += 1
    if total == 0:
        return 0.0
    return bad / total


def presentation_form_ratio(text: str) -> float:
    """Return the share of Arabic letters that fall in the Presentation Forms blocks.

    A page with no Arabic letters at all returns ``0.0`` and the
    page-level validator abstains (the gate is gated on a minimum
    Arabic-letter count). Body text >50% in presentation forms
    indicates a visual-order text layer where shaped glyphs were
    serialized as codepoints rather than logical-order base
    characters -- the validator routes such pages to the ML branch.
    """
    arabic_total = 0
    presentation = 0
    for ch in text:
        if not _is_letter(ch):
            continue
        cp = ord(ch)
        in_base = _in_ranges(cp, _ARABIC_RANGES)
        in_pres = _in_ranges(cp, _ARABIC_PRESENTATION_RANGES)
        if not (in_base or in_pres):
            continue
        arabic_total += 1
        if in_pres:
            presentation += 1
    if arabic_total == 0:
        return 0.0
    return presentation / arabic_total


# ---- Signal 3: word-boundary plausibility --------------------------------


def word_boundary_plausibility(text: str, *, reference: Mapping[int, float] | None = None) -> float:
    """Return KL divergence (nats) of the page's token-length distribution.

    Lower is better -- closer to the reference Arabic distribution. The
    page distribution is built from whitespace-separated tokens after
    Unicode NFC normalisation, with token lengths capped at
    ``max_length`` from the reference (longer tokens are bucketed into
    the tail). A page with fewer than 8 tokens returns ``0.0`` so we
    don't flag legitimate short pages purely on small sample size.
    """
    ref = reference if reference is not None else _load_reference_distribution()
    max_length = max(ref)
    tokens = _tokenise(text)
    if len(tokens) < 8:
        return 0.0
    page_dist = _build_distribution(tokens, max_length=max_length)
    return _kl_divergence(page_dist, ref)


# ---- Page-level validator -------------------------------------------------


def validate_page(
    page: NativePage,
    *,
    config: ValidatorConfig | None = None,
) -> ValidationResult:
    """Apply the three signals to a :class:`NativePage` and return the verdict."""
    cfg = config or ValidatorConfig()

    if not page.has_text_layer:
        return ValidationResult(
            accept=False,
            signals={},
            reasons=("no_text_layer",),
        )

    text = "\n".join(region.text for region in page.regions)
    text = unicodedata.normalize("NFC", text)
    letter_count = sum(1 for ch in text if _is_letter(ch))
    has_arabic = any(
        _in_ranges(ord(ch), _ARABIC_RANGES) or _in_ranges(ord(ch), _ARABIC_PRESENTATION_RANGES)
        for ch in text
        if _is_letter(ch)
    )
    arabic_letter_count = sum(
        1
        for ch in text
        if _is_letter(ch)
        and (
            _in_ranges(ord(ch), _ARABIC_RANGES) or _in_ranges(ord(ch), _ARABIC_PRESENTATION_RANGES)
        )
    )

    arabic_ratio = arabic_codepoint_ratio(text)
    replacement_ratio = replacement_glyph_ratio(text)
    boundary_kl = word_boundary_plausibility(text)
    pres_ratio = presentation_form_ratio(text)

    signals: dict[str, float] = {
        "arabic_codepoint_ratio": arabic_ratio,
        "replacement_glyph_ratio": replacement_ratio,
        "word_boundary_kl": boundary_kl,
        "presentation_form_ratio": pres_ratio,
    }
    reasons: list[str] = []

    # Replacement-glyph signal: independent of letter count. Surfacing this
    # before the empty-text gate means a page consisting entirely of
    # replacement glyphs gets the more specific, actionable reason rather
    # than the generic "no letters" one.
    if replacement_ratio > cfg.max_replacement_ratio:
        reasons.append(
            f"replacement_glyph_ratio {replacement_ratio:.3f} > max {cfg.max_replacement_ratio:.3f}"
        )

    # Empty-content gate: text layer claims to exist but produced no
    # letters AND no replacement glyphs were observed. This is typically
    # a font-encoded glyph stream that did not decode to Unicode at all.
    if letter_count == 0 and replacement_ratio <= cfg.max_replacement_ratio:
        reasons.append("empty_text_layer")

    # Arabic ratio: only enforced when the page genuinely seems Arabic AND
    # we have enough letters to judge AND the actual ratio is below the
    # accept band.
    if has_arabic and letter_count >= cfg.min_letter_count and arabic_ratio < cfg.min_arabic_ratio:
        reasons.append(
            f"arabic_codepoint_ratio {arabic_ratio:.3f} < min {cfg.min_arabic_ratio:.3f}"
        )

    if boundary_kl > cfg.max_word_boundary_kl:
        reasons.append(f"word_boundary_kl {boundary_kl:.3f} > max {cfg.max_word_boundary_kl:.3f}")

    # Presentation-form gate: only enforced when the page carries enough
    # Arabic letters for the ratio to mean something.
    if arabic_letter_count >= cfg.min_letter_count and pres_ratio > cfg.max_presentation_form_ratio:
        reasons.append(
            f"presentation_form_ratio {pres_ratio:.3f} > max {cfg.max_presentation_form_ratio:.3f}"
        )

    return ValidationResult(
        accept=not reasons,
        signals=signals,
        reasons=tuple(reasons),
    )


# ---- Internals -----------------------------------------------------------


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _in_ranges(cp: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def _tokenise(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in _TOKEN_SPLIT.split(text.strip()) if t]


def _build_distribution(tokens: Iterable[str], *, max_length: int) -> dict[int, float]:
    counter: Counter[int] = Counter()
    total = 0
    for token in tokens:
        length = min(len(token), max_length)
        if length == 0:
            continue
        counter[length] += 1
        total += 1
    if total == 0:
        return {k: 0.0 for k in range(1, max_length + 1)}
    return {k: counter.get(k, 0) / total for k in range(1, max_length + 1)}


def _kl_divergence(p: Mapping[int, float], q: Mapping[int, float]) -> float:
    """KL(p || q) in nats with Laplace smoothing on q (and clamped p)."""
    eps = 1e-9
    total = 0.0
    keys = set(p) | set(q)
    for k in keys:
        pi = p.get(k, 0.0)
        qi = q.get(k, 0.0)
        if pi <= 0:
            continue
        total += pi * math.log((pi + eps) / (qi + eps))
    return max(total, 0.0)


_reference_cache: dict[int, float] | None = None


def _load_reference_distribution() -> dict[int, float]:
    global _reference_cache
    if _reference_cache is not None:
        return _reference_cache
    raw = json.loads(REFERENCE_DIST_PATH.read_text(encoding="utf-8"))
    _reference_cache = {int(k): float(v) for k, v in raw["frequencies"].items()}
    return _reference_cache


def reset_reference_cache() -> None:
    """Test helper: drop the in-memory reference distribution cache."""
    global _reference_cache
    _reference_cache = None


__all__ = [
    "REFERENCE_DIST_PATH",
    "ValidationResult",
    "ValidatorConfig",
    "arabic_codepoint_ratio",
    "presentation_form_ratio",
    "replacement_glyph_ratio",
    "reset_reference_cache",
    "validate_page",
    "word_boundary_plausibility",
]


# Avoid an "unused import" complaint when ``replace`` is absent locally;
# kept for callers that want to use it on `ValidatorConfig`.
_ = replace
