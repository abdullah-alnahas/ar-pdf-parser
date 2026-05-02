"""Phase-7 bidi helper tests."""

from __future__ import annotations

from arabic_pdf_transcribe.emit._bidi import (
    RLM,
    add_rlm_if_needed,
    has_ltr_run,
    is_arabic_dominant,
)


def test_pure_arabic_is_arabic_dominant() -> None:
    assert is_arabic_dominant("السلام عليكم")


def test_pure_latin_is_not_arabic_dominant() -> None:
    assert not is_arabic_dominant("Hello world")


def test_empty_text_is_not_arabic_dominant() -> None:
    assert not is_arabic_dominant("")


def test_majority_arabic_with_latin_run_is_arabic_dominant() -> None:
    text = "اللغة العربية هي Arabic"
    assert is_arabic_dominant(text)


def test_minority_arabic_is_not_dominant() -> None:
    text = "the word for Arabic is عربي"
    assert not is_arabic_dominant(text)


def test_has_ltr_run_detects_latin() -> None:
    assert has_ltr_run("Arabic")


def test_has_ltr_run_no_latin() -> None:
    assert not has_ltr_run("السلام")


def test_add_rlm_inserts_when_arabic_with_ltr() -> None:
    text = "اللغة العربية هي Arabic"
    out = add_rlm_if_needed(text)
    assert out.startswith(RLM)
    assert out[1:] == text


def test_add_rlm_no_change_for_pure_arabic() -> None:
    text = "السلام عليكم"
    assert add_rlm_if_needed(text) == text


def test_add_rlm_no_change_for_pure_latin() -> None:
    text = "Hello world"
    assert add_rlm_if_needed(text) == text


def test_add_rlm_idempotent() -> None:
    text = "اللغة hello"
    once = add_rlm_if_needed(text)
    twice = add_rlm_if_needed(once)
    assert once == twice


def test_add_rlm_handles_empty() -> None:
    assert add_rlm_if_needed("") == ""


def test_rlm_constant_is_u200f() -> None:
    assert RLM == "‏"


def test_bidi_class_check_is_pure() -> None:
    text = "السلام Arabic"
    assert add_rlm_if_needed(text) == add_rlm_if_needed(text)
