"""Phase 3: native-text quality validator tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from textwrap import dedent

import pytest

from arabic_pdf_transcribe.extract import extract_native
from arabic_pdf_transcribe.extract.native import NativePage
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource
from arabic_pdf_transcribe.validate import (
    ValidationResult,
    ValidatorConfig,
    arabic_codepoint_ratio,
    presentation_form_ratio,
    replacement_glyph_ratio,
    reset_reference_cache,
    validate_page,
    word_boundary_plausibility,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DIGITAL_CLEAN = REPO_ROOT / "tests" / "fixtures" / "pdfs" / "digital-clean"
DIGITAL_BROKEN = REPO_ROOT / "tests" / "fixtures" / "pdfs" / "digital-broken"


# ---- Per-signal unit tests -----------------------------------------------


def test_arabic_codepoint_ratio_pure_arabic() -> None:
    assert arabic_codepoint_ratio("كتاب قلم") == 1.0


def test_arabic_codepoint_ratio_pure_latin() -> None:
    assert arabic_codepoint_ratio("hello world") == 0.0


def test_arabic_codepoint_ratio_mixed_half() -> None:
    text = "abcd" + "كتاب"  # 4 latin + 4 arabic letters
    assert arabic_codepoint_ratio(text) == pytest.approx(0.5)


def test_arabic_codepoint_ratio_no_letters() -> None:
    assert arabic_codepoint_ratio("123 !!! ...") == 0.0


def test_replacement_glyph_ratio_clean_text() -> None:
    assert replacement_glyph_ratio("hello world") == 0.0
    assert replacement_glyph_ratio("كتاب قلم") == 0.0


def test_replacement_glyph_ratio_fffd() -> None:
    # 5 FFFD chars + 5 letters = 50% replacement
    assert replacement_glyph_ratio("�����hello") == pytest.approx(0.5)


def test_replacement_glyph_ratio_pua() -> None:
    # PUA char counts as replacement.
    assert replacement_glyph_ratio("hello") == pytest.approx(2 / 7)


def test_replacement_glyph_ratio_geometric_shapes_box() -> None:
    """U+25A0 (BLACK SQUARE) is a common ``.notdef`` fallback."""
    assert replacement_glyph_ratio("■■■■hello") == pytest.approx(4 / 9)


def test_replacement_glyph_ratio_ascii_control_bytes() -> None:
    """Foulabook-style broken layers serialize glyph IDs as raw control bytes.

    ``\\x01..\\x1F`` codepoints (excluding the standard whitespace controls
    \\t\\n\\v\\f\\r) must count as replacement glyphs so the signal flags
    pages whose text layer is a font-id-leaking glyph stream.
    """
    text = "\x01\x02\x03\x04\x05\x06\x07\x08hello"
    assert replacement_glyph_ratio(text) == pytest.approx(8 / 13)


def test_replacement_glyph_ratio_ascii_whitespace_excluded() -> None:
    """Tabs and newlines are not counted as replacement glyphs."""
    assert replacement_glyph_ratio("\t\n\r\v\fhello") == 0.0


def test_presentation_form_ratio_pure_base_arabic() -> None:
    """Pure base-form Arabic returns zero presentation share."""
    assert presentation_form_ratio("كتاب قلم") == 0.0


def test_presentation_form_ratio_pure_presentation_arabic() -> None:
    """All-presentation-form text scores 1.0 — broken visual-order layer."""
    # FB50-FDFF + FE70-FEFF presentation-form letters.
    text = "ﭖﭘﭚﺑﺓﺗ"
    assert presentation_form_ratio(text) == 1.0


def test_presentation_form_ratio_no_arabic_abstains() -> None:
    """Pages with no Arabic letters return 0.0 (signal abstains)."""
    assert presentation_form_ratio("hello world") == 0.0
    assert presentation_form_ratio("") == 0.0


def test_presentation_form_ratio_mixed() -> None:
    """Mixed base + presentation forms produces the expected ratio."""
    # 4 base-form + 4 presentation-form letters.
    text = "كتابﭖﭘﭚﺑ"
    assert presentation_form_ratio(text) == pytest.approx(0.5)


def test_word_boundary_plausibility_short_input_returns_zero() -> None:
    """Pages with fewer than 8 tokens skip the signal."""
    assert word_boundary_plausibility("a b c d") == 0.0


def test_word_boundary_plausibility_arabic_like_distribution() -> None:
    """A token-length distribution close to the reference must score low."""
    # 30 tokens, lengths roughly matching the reference distribution.
    tokens = (
        ["xx"] * 4
        + ["xxx"] * 6
        + ["xxxx"] * 6
        + ["xxxxx"] * 5
        + ["xxxxxx"] * 4
        + ["xxxxxxx"] * 3
        + ["xxxxxxxx"] * 2
    )
    text = " ".join(tokens)
    assert word_boundary_plausibility(text) < 0.3


def test_word_boundary_plausibility_pathological_distribution() -> None:
    """A page with only 1-character tokens diverges sharply from the reference."""
    text = " ".join(["a"] * 30)
    assert word_boundary_plausibility(text) > 1.0


# ---- ValidatorConfig TOML round-trip --------------------------------------


def test_validator_config_toml_round_trip(tmp_path: Path) -> None:
    cfg = ValidatorConfig(
        min_arabic_ratio=0.6,
        max_replacement_ratio=0.04,
        max_word_boundary_kl=0.9,
        min_letter_count=40,
    )
    path = tmp_path / "validator.toml"
    path.write_text(cfg.to_toml(), encoding="utf-8")
    restored = ValidatorConfig.from_toml(path)
    assert restored == cfg


def test_validator_config_toml_round_trip_with_section(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        dedent(
            """
            [validator]
            min_arabic_ratio = 0.7
            max_replacement_ratio = 0.02
            max_word_boundary_kl = 1.2
            min_letter_count = 25
            """
        ),
        encoding="utf-8",
    )
    cfg = ValidatorConfig.from_toml(path)
    assert cfg.min_arabic_ratio == 0.7
    assert cfg.max_replacement_ratio == 0.02
    assert cfg.max_word_boundary_kl == 1.2
    assert cfg.min_letter_count == 25


# ---- Page-level validator -------------------------------------------------


def _page_with_text(text: str, *, has_text_layer: bool = True) -> NativePage:
    region = Region(
        page_index=0,
        bbox=BBox(0, 0, 100, 50),
        text=text,
        role=RegionRole.UNKNOWN,
        source=RegionSource.NATIVE,
    )
    return NativePage(
        page_index=0,
        page_width=612.0,
        page_height=792.0,
        regions=[region],
        font_size_hist=Counter({12.0: max(1, len(text))}),
        has_text_layer=has_text_layer,
    )


def test_validate_page_no_text_layer_rejected_immediately() -> None:
    page = _page_with_text("", has_text_layer=False)
    result = validate_page(page)
    assert not result.accept
    assert result.reasons == ("no_text_layer",)


def test_validate_page_empty_text_layer_rejected() -> None:
    """A page that claims a text layer but produced no letters fails.

    Whitespace-only / punctuation-only output is the failure mode of a
    font-encoded glyph stream that never decoded to Unicode. The
    validator must reject so the ML branch runs.
    """
    page = _page_with_text("   \n\t  \n  ")
    result = validate_page(page)
    assert not result.accept
    assert result.reasons == ("empty_text_layer",)


def test_validate_page_punctuation_only_rejected() -> None:
    page = _page_with_text("!!! ??? ... ,,, ;;; ")
    result = validate_page(page)
    assert not result.accept
    assert result.reasons == ("empty_text_layer",)


def test_validate_page_clean_arabic_accepted() -> None:
    text = " ".join(["كتاب", "قلم", "ورق", "مدرسة", "علم", "كلمة"] * 5)
    page = _page_with_text(text)
    result = validate_page(page)
    assert result.accept, f"unexpected rejection: {result.reasons}"


def test_validate_page_clean_latin_accepted() -> None:
    text = " ".join(["alpha", "beta", "gamma", "delta", "epsilon"] * 6)
    page = _page_with_text(text)
    result = validate_page(page)
    assert result.accept, f"unexpected rejection: {result.reasons}"


def test_validate_page_replacement_glyph_storm_rejected() -> None:
    page = _page_with_text("�" * 100)
    result = validate_page(page)
    assert not result.accept
    assert any("replacement_glyph_ratio" in r for r in result.reasons)


def test_validate_page_mojibake_rejected() -> None:
    """Page whose token-length distribution diverges sharply from Arabic."""
    page = _page_with_text("a " * 100)
    result = validate_page(page)
    assert not result.accept
    assert any("word_boundary_kl" in r for r in result.reasons)


def test_validate_page_mostly_arabic_with_minor_latin_accepted() -> None:
    """A page with predominantly Arabic and a few English citations passes.

    Uses varied-length Arabic tokens so the word-boundary KL signal stays
    in band — the reference distribution is for general printed Arabic,
    not single-length corpora.
    """
    arabic_tokens = (
        ["لا"] * 4
        + ["كتب"] * 6
        + ["كتاب"] * 6
        + ["مدرسة"] * 5
        + ["المدارس"] * 4
        + ["الأكاديمي"] * 3
        + ["كلمة"] * 2
    )
    text = " ".join([*arabic_tokens, "English", "citation"])
    page = _page_with_text(text)
    result = validate_page(page)
    assert result.accept, f"unexpected rejection: {result.reasons}"


def test_validate_page_arabic_swapped_to_latin_rejected() -> None:
    """A page that should be Arabic but extracted to Latin is rejected.

    Simulated: at least one Arabic codepoint present (so the Arabic-ratio
    gate engages) but the page is overwhelmingly Latin.
    """
    text = "ك " + "abcdef " * 30  # 1 arabic letter + many latin words
    page = _page_with_text(text)
    result = validate_page(page)
    assert not result.accept
    assert any("arabic_codepoint_ratio" in r for r in result.reasons)


# ---- Real-fixture validation ---------------------------------------------


def test_validate_page_real_arabic_fixture_accepted() -> None:
    fixture = DIGITAL_CLEAN / "lorem-ar-real.pdf"
    if not fixture.exists():
        pytest.skip("real-arabic fixture only generated when an Arabic-capable TTF is installed")
    pages = list(extract_native(fixture))
    assert pages
    for pg in pages:
        result = validate_page(pg)
        assert result.accept, f"page {pg.page_index} rejected: {result.reasons}"


def test_validate_page_clean_2col_fixture_accepted() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-2col.pdf"))
    assert pages
    for pg in pages:
        result = validate_page(pg)
        assert result.accept, f"page {pg.page_index} rejected: {result.reasons}"


def test_validate_page_mojibake_fixture_rejected() -> None:
    pages = list(extract_native(DIGITAL_BROKEN / "mojibake.pdf"))
    assert pages
    rejected = [validate_page(pg) for pg in pages]
    assert any(not r.accept for r in rejected), "expected at least one page rejected"


def test_validate_page_replacement_glyphs_fixture_rejected() -> None:
    pages = list(extract_native(DIGITAL_BROKEN / "replacement-glyphs.pdf"))
    assert pages
    rejected = [validate_page(pg) for pg in pages]
    assert all(not r.accept for r in rejected)
    assert all(any("replacement_glyph_ratio" in reason for reason in r.reasons) for r in rejected)


def test_validate_page_broken_glyph_id_layer_fixture_rejected() -> None:
    """Foulabook-style PDF whose text layer is raw glyph IDs (control bytes).

    Regression for issue #14: the validator's three original signals all
    abstained on this failure mode (no Arabic letters at all → arabic gate
    skipped; control bytes ignored as ASCII punctuation → replacement gate
    abstained; KL stayed under threshold). The replacement-glyph signal
    must now flag the page so the orchestrator routes it to ML.
    """
    pages = list(extract_native(DIGITAL_BROKEN / "broken-glyph-id-layer.pdf"))
    assert pages
    results = [validate_page(pg) for pg in pages]
    assert all(not r.accept for r in results), "every page in the fixture must be rejected"
    assert all(any("replacement_glyph_ratio" in reason for reason in r.reasons) for r in results)


def test_validate_page_presentation_form_storm_rejected() -> None:
    """Page whose Arabic body is overwhelmingly presentation-form letters fails."""
    # 60+ presentation-form letters (well above min_letter_count).
    text = "ﭖﭘﭚﺑﺓﺗﺙﺞ " * 8 + "ﺠﺡﺢﺣﺤﺥﺦﺧ " * 8
    page = _page_with_text(text)
    result = validate_page(page)
    assert not result.accept
    assert any("presentation_form_ratio" in r for r in result.reasons)


def test_validate_page_few_presentation_forms_accepted() -> None:
    """Mostly base-form Arabic with a few ligatures stays accepted."""
    # 6 base-form Arabic words (varied lengths) plus a single ligature.
    arabic_tokens = (
        ["لا"] * 4
        + ["كتب"] * 6
        + ["كتاب"] * 6
        + ["مدرسة"] * 5
        + ["المدارس"] * 4
        + ["الأكاديمي"] * 3
    )
    text = " ".join([*arabic_tokens, "ﻟﺍ"])  # one presentation-form ligature
    page = _page_with_text(text)
    result = validate_page(page)
    assert result.accept, f"unexpected rejection: {result.reasons}"


# ---- Sanity: every threshold has bite -----------------------------------


_FIXTURES_PASS_DEFAULTS = (
    "lorem-ar-2col.pdf",
    "lorem-ar-en-mixed.pdf",
)


def _all_fixture_pages() -> Iterable[tuple[str, NativePage]]:
    for name in _FIXTURES_PASS_DEFAULTS:
        for pg in extract_native(DIGITAL_CLEAN / name):
            yield name, pg


def test_threshold_arabic_too_strict_flips_some_clean_fixture() -> None:
    """Set ``min_arabic_ratio`` to 1.5 — even pure Arabic gets rejected."""
    text = " ".join(["كتاب", "قلم", "ورق", "مدرسة", "علم", "كلمة"] * 5)
    page = _page_with_text(text)
    cfg = ValidatorConfig(min_arabic_ratio=1.5)
    result = validate_page(page, config=cfg)
    assert not result.accept


def test_threshold_replacement_too_loose_flips_synthetic_page() -> None:
    """Loosening ``max_replacement_ratio`` flips a replacement-heavy synthetic page.

    Uses a synthetic ``NativePage`` carrying both letters (so the
    empty-text-layer short-circuit doesn't fire) and many replacement
    glyphs. The default config rejects; loosening the threshold accepts.
    """
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " + ("■" * 100)
    page = _page_with_text(text)
    default_result = validate_page(page)
    assert not default_result.accept, "synthetic replacement-heavy page should reject by default"
    loose_cfg = ValidatorConfig(max_replacement_ratio=1.5, max_word_boundary_kl=10.0)
    loose_result = validate_page(page, config=loose_cfg)
    assert loose_result.accept, f"loosening should accept; reasons: {loose_result.reasons}"


def test_threshold_kl_too_loose_flips_synthetic_page() -> None:
    """Loosening ``max_word_boundary_kl`` flips a KL-heavy synthetic page."""
    page = _page_with_text(("a " * 60) + "alpha beta gamma delta epsilon")
    default_result = validate_page(page)
    assert not default_result.accept, "single-char-token page should reject by default"
    loose_cfg = ValidatorConfig(max_word_boundary_kl=10.0)
    loose_result = validate_page(page, config=loose_cfg)
    assert loose_result.accept, f"loosening should accept; reasons: {loose_result.reasons}"


def test_reset_reference_cache_is_idempotent() -> None:
    reset_reference_cache()
    reset_reference_cache()
    # Subsequent validation still works.
    page = _page_with_text("alpha beta gamma delta epsilon zeta eta theta iota")
    result = validate_page(page)
    assert isinstance(result, ValidationResult)
