"""Phase-8 CLI argument parsing + format-selection tests.

Covers spec test scenario 18 (format-extension mismatch → exit 4)
and the resolved-decisions rules: extension wins when present,
``--format`` wins otherwise, stdout defaults to Markdown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arabic_pdf_transcribe._logging import ProgressMode
from arabic_pdf_transcribe.cli import (
    EXIT_CORRUPTED_OR_FORMAT,
    EXIT_OK,
    EXIT_RUNTIME,
    build_parser,
    main,
    validate_args,
)
from arabic_pdf_transcribe.errors import FormatExtensionMismatch


def _parse(*argv: str):
    return build_parser().parse_args(list(argv))


# ---------------------------------------------------------------------------
# Format selection
# ---------------------------------------------------------------------------


def test_format_default_is_markdown_to_stdout(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf)))
    assert args.format == "md"
    assert args.output is None


def test_md_extension_drives_format(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "-o", str(tmp_path / "out.md")))
    assert args.format == "md"


def test_markdown_extension_drives_format(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "-o", str(tmp_path / "out.markdown")))
    assert args.format == "md"


def test_docx_extension_drives_format(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "-o", str(tmp_path / "out.docx")))
    assert args.format == "docx"


def test_explicit_format_wins_when_extension_unknown(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "-o", str(tmp_path / "out.txt"), "--format", "md"))
    assert args.format == "md"


def test_format_extension_mismatch_raises(tmp_path: Path) -> None:
    """Spec scenario 18: --format docx -o out.md → exit 4."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    with pytest.raises(FormatExtensionMismatch):
        validate_args(_parse(str(pdf), "-o", str(tmp_path / "out.md"), "--format", "docx"))


def test_format_extension_mismatch_main_returns_exit_4(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    rc = main([str(pdf), "-o", str(tmp_path / "out.md"), "--format", "docx"])
    assert rc == EXIT_CORRUPTED_OR_FORMAT


# ---------------------------------------------------------------------------
# --pages parsing
# ---------------------------------------------------------------------------


def test_pages_single(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "--pages", "3"))
    assert args.pages == (2,)  # 0-based


def test_pages_range_and_singletons(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "--pages", "1-3,7,10-12"))
    assert args.pages == (0, 1, 2, 6, 9, 10, 11)


def test_pages_dedupes_and_sorts(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "--pages", "5,1,3,1-2"))
    assert args.pages == (0, 1, 2, 4)


def test_pages_invalid_range_returns_exit_2(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    rc = main([str(pdf), "--pages", "5-3"])
    assert rc == EXIT_RUNTIME


def test_pages_zero_returns_exit_2(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    rc = main([str(pdf), "--pages", "0"])
    assert rc == EXIT_RUNTIME


# ---------------------------------------------------------------------------
# Progress mode
# ---------------------------------------------------------------------------


def test_default_progress_mode_is_text(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf)))
    assert args.progress_mode is ProgressMode.TEXT


def test_quiet_overrides_text(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "--quiet"))
    assert args.progress_mode is ProgressMode.QUIET


def test_json_logs_selects_json_mode(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "--json-logs"))
    assert args.progress_mode is ProgressMode.JSON


def test_quiet_takes_precedence_over_json_logs(tmp_path: Path) -> None:
    """--quiet wins; useful for CI piping where stderr is discarded anyway."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "--quiet", "--json-logs"))
    assert args.progress_mode is ProgressMode.QUIET


# ---------------------------------------------------------------------------
# --max-workers
# ---------------------------------------------------------------------------


def test_max_workers_default_is_one(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf)))
    assert args.max_workers == 1


def test_max_workers_int(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "--max-workers", "4"))
    assert args.max_workers == 4


def test_max_workers_auto_clamps_to_4(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    args = validate_args(_parse(str(pdf), "--max-workers", "auto"))
    assert 1 <= args.max_workers <= 4


def test_max_workers_invalid_returns_exit_2(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    rc = main([str(pdf), "--max-workers", "0"])
    assert rc == EXIT_RUNTIME


# ---------------------------------------------------------------------------
# Smoke: --help short-circuits
# ---------------------------------------------------------------------------


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == EXIT_OK
    captured = capsys.readouterr()
    assert "arabic-pdf-transcribe" in captured.out


def test_help_surface_matches_documented_flags(capsys: pytest.CaptureFixture[str]) -> None:
    """Phase-9 acceptance criterion: ``--help`` matches the documented
    CLI surface (README + docs). All flags listed in the README's
    "CLI usage" / "Exit codes" section must appear in --help output.
    """
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    out = captured.out
    for flag in (
        "--output",
        "--format",
        "--pages",
        "--strict",
        "--quiet",
        "--json-logs",
        "--config",
        "--debug-json",
        "--max-workers",
        "--dpi",
    ):
        assert flag in out, f"--help missing documented flag {flag}"


# ---------------------------------------------------------------------------
# Missing input file → exit 4
# ---------------------------------------------------------------------------


def test_missing_input_returns_exit_4(tmp_path: Path) -> None:
    rc = main([str(tmp_path / "does-not-exist.pdf")])
    assert rc == EXIT_CORRUPTED_OR_FORMAT


# ---------------------------------------------------------------------------
# TOML config sections (architect carry-over 3 from PR #9)
# ---------------------------------------------------------------------------


def test_config_validator_section_loaded(tmp_path: Path) -> None:
    """``[validator]`` section overrides ValidatorConfig defaults."""
    from arabic_pdf_transcribe.cli import _load_config_doc, _validator_cfg_from_doc

    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text(
        "[validator]\nmin_arabic_ratio = 0.7\nmax_replacement_ratio = 0.02\n",
        encoding="utf-8",
    )
    doc = _load_config_doc(cfg_path)
    cfg = _validator_cfg_from_doc(doc)
    assert cfg is not None
    assert cfg.min_arabic_ratio == 0.7
    assert cfg.max_replacement_ratio == 0.02


def test_config_render_section_overrides_dpi(tmp_path: Path) -> None:
    """``[render].dpi`` is honoured when --dpi is not given."""
    from arabic_pdf_transcribe.cli import _load_config_doc, _resolve_dpi

    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text("[render]\ndpi = 300\n", encoding="utf-8")
    doc = _load_config_doc(cfg_path)
    assert _resolve_dpi(None, doc) == 300


def test_cli_dpi_flag_wins_over_render_section(tmp_path: Path) -> None:
    """Explicit --dpi beats config-file [render].dpi."""
    from arabic_pdf_transcribe.cli import _load_config_doc, _resolve_dpi

    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text("[render]\ndpi = 300\n", encoding="utf-8")
    doc = _load_config_doc(cfg_path)
    assert _resolve_dpi(150, doc) == 150


def test_config_layout_section_parsed(tmp_path: Path) -> None:
    """``[layout]`` section feeds HFLayoutDetectorConfig.from_mapping."""
    from arabic_pdf_transcribe.layout.hf_detector import HFLayoutDetectorConfig

    cfg = HFLayoutDetectorConfig.from_mapping(
        {"pixel_confidence": 0.7, "min_region_area_px": 128, "unknown_key": "ignored"}
    )
    assert cfg.pixel_confidence == 0.7
    assert cfg.min_region_area_px == 128


def test_config_ocr_section_parsed() -> None:
    """``[ocr]`` section feeds OCRConfig.from_mapping."""
    from arabic_pdf_transcribe.ocr.hf_ocr import OCRConfig

    cfg = OCRConfig.from_mapping({"max_new_tokens": 2048, "num_beams": 3, "junk_key": True})
    assert cfg.max_new_tokens == 2048
    assert cfg.num_beams == 3


def test_config_missing_sections_yield_defaults(tmp_path: Path) -> None:
    """Missing config file → empty doc → all defaults preserved."""
    from arabic_pdf_transcribe.cli import (
        _load_config_doc,
        _resolve_dpi,
        _validator_cfg_from_doc,
    )

    doc = _load_config_doc(None)
    assert doc == {}
    assert _validator_cfg_from_doc(doc) is None
    assert _resolve_dpi(None, doc) == 200
