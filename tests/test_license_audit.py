"""License-audit harness tests."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS))
import license_audit as audit_mod  # noqa: E402


def test_normalise_canonicalises_aliases() -> None:
    assert audit_mod._normalise("MIT License") == "MIT"
    assert audit_mod._normalise("apache 2.0") == "Apache-2.0"
    assert audit_mod._normalise("Apache Software License") == "Apache-2.0"
    assert audit_mod._normalise("BSD 3-Clause") == "BSD-3-Clause"
    assert audit_mod._normalise("ISC License") == "ISC"
    assert audit_mod._normalise("Unknown weird license string") == "Unknown weird license string"


def test_normalise_rejects_long_blob_without_alias() -> None:
    blob = "Permission is hereby granted... " * 20
    assert audit_mod._normalise(blob) is None


def test_classifier_license_picks_first_match() -> None:
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ]
    assert audit_mod._classifier_license(classifiers) == "MIT"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("MIT", ["MIT"]),
        ("MIT OR Apache-2.0", ["MIT", "Apache-2.0"]),
        ("Apache-2.0 AND MIT", ["Apache-2.0", "MIT"]),
        (
            "(Apache-2.0 OR BSD-3-Clause) AND LicenseRef-PdfiumThirdParty",
            ["Apache-2.0", "BSD-3-Clause", "LicenseRef-PdfiumThirdParty"],
        ),
        ("", []),
        (None, []),
    ],
)
def test_expression_license_splits_clauses(expression: str | None, expected: list[str]) -> None:
    assert audit_mod._expression_license(expression) == expected


def test_extra_marker_extraction() -> None:
    assert audit_mod._extra_marker("transformers==4.46.3 ; extra == 'ml'") == "ml"
    assert audit_mod._extra_marker('pytest==8.3.4 ; extra == "dev"') == "dev"
    assert audit_mod._extra_marker("pypdfium2==4.30.0") is None


def test_audit_models_accepts_well_formed_apache(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(
        dedent(
            """
            [[models]]
            name = "owner/repo"
            revision = "abc1234567"
            license = "Apache-2.0"
            stage = "ocr"
            """
        ),
        encoding="utf-8",
    )
    assert audit_mod.audit_models(path, frozenset(audit_mod.DEFAULT_ALLOW)) == []


def test_audit_models_rejects_forbidden_license(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(
        dedent(
            """
            [[models]]
            name = "owner/copyleft-model"
            revision = "deadbeefcafe"
            license = "GPL-3.0"
            stage = "ocr"
            """
        ),
        encoding="utf-8",
    )
    violations = audit_mod.audit_models(path, frozenset(audit_mod.DEFAULT_ALLOW))
    assert violations
    assert any("GPL-3.0" in v.detail for v in violations)
    assert all(v.kind == "model" for v in violations)


def test_audit_models_rejects_missing_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(
        dedent(
            """
            [[models]]
            name = "owner/incomplete"
            license = "MIT"
            """
        ),
        encoding="utf-8",
    )
    violations = audit_mod.audit_models(path, frozenset(audit_mod.DEFAULT_ALLOW))
    missing = {v.detail for v in violations if v.kind == "model"}
    assert any("revision" in d for d in missing)
    assert any("stage" in d for d in missing)


def test_audit_models_accepts_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text("models = []\n", encoding="utf-8")
    assert audit_mod.audit_models(path, frozenset(audit_mod.DEFAULT_ALLOW)) == []


def test_audit_models_rejects_bad_schema(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text("models = 'not a list'\n", encoding="utf-8")
    violations = audit_mod.audit_models(path, frozenset(audit_mod.DEFAULT_ALLOW))
    assert violations
    assert violations[0].kind == "schema"


def test_audit_models_missing_file_returns_schema_violation(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.toml"
    violations = audit_mod.audit_models(missing, frozenset(audit_mod.DEFAULT_ALLOW))
    assert violations
    assert violations[0].kind == "schema"


def test_audit_distributions_with_narrow_allow_fails() -> None:
    """Restricting the allow-list to a single rare license must produce violations.

    This exercises the negative path required by the plan: the harness must
    fail loudly when the allow-list is artificially narrowed. We intentionally
    pick a license that is essentially never used so the check is robust to
    upstream metadata drift.
    """
    violations = audit_mod.audit_distributions(
        allow=frozenset({"WTFPL"}),  # nothing on PyPI uses this
        overrides={},
    )
    assert violations, "audit must fail when the allow-list excludes every dep"
    # All violations should be at the dist level here; no model file involved.
    assert all(v.kind == "dist" for v in violations)


def test_audit_distributions_default_allow_passes_for_known_devdeps() -> None:
    """The default allow-list must be wide enough to cover the project's installed deps.

    Exact coverage depends on what's installed in the test env, but every
    declared runtime + dev dependency uses one of the allow-listed licenses
    (verified manually before pinning in pyproject.toml). Any failure here is
    a red flag that an upstream package's metadata changed.
    """
    violations = audit_mod.audit_distributions(
        allow=frozenset(audit_mod.DEFAULT_ALLOW),
        overrides={},
    )
    if violations:
        # Surface helpful detail rather than a silent skip.
        msg = "\n".join(v.format() for v in violations)
        pytest.fail(f"unexpected dist violations under default allow-list:\n{msg}")


def test_overrides_apply(tmp_path: Path) -> None:
    overrides_file = tmp_path / "overrides.toml"
    overrides_file.write_text(
        dedent(
            """
            [[overrides]]
            dist = "fake-package==1.0.0"
            license = "MIT"
            note = "fabricated for unit test"
            """
        ),
        encoding="utf-8",
    )
    overrides = audit_mod.load_overrides(overrides_file)
    assert overrides == {"fake-package==1.0.0": "MIT"}


def test_main_returns_zero_when_clean(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = tmp_path / "models.toml"
    models.write_text("models = []\n", encoding="utf-8")
    overrides = tmp_path / "overrides.toml"
    overrides.write_text("overrides = []\n", encoding="utf-8")

    rc = audit_mod.main(
        [
            "--models",
            str(models),
            "--overrides",
            str(overrides),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_main_returns_one_when_models_violate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = tmp_path / "models.toml"
    models.write_text(
        dedent(
            """
            [[models]]
            name = "x/copyleft"
            revision = "abc"
            license = "GPL-3.0"
            stage = "ocr"
            """
        ),
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.toml"
    overrides.write_text("overrides = []\n", encoding="utf-8")

    rc = audit_mod.main(
        [
            "--models",
            str(models),
            "--overrides",
            str(overrides),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "GPL-3.0" in err
