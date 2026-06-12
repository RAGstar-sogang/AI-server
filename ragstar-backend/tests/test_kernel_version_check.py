import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agent.tools.kernel_version_check import kernel_version_check


REQUIRED_BUG_KEYS = {
    "bug_id",
    "description",
    "affected_range",
    "fix_version",
    "reference",
}


@pytest.mark.parametrize(
    ("kernel_version", "expected_bug_id"),
    [
        pytest.param("4.14.39", "CVE-2018-1000200", id="4_14_last_affected"),
        pytest.param("4.15.17", "CVE-2018-1000200", id="4_15_last_affected"),
        pytest.param("5.3.18", "K8S-ISSUE-61937", id="5_3_last_affected"),
        pytest.param("5.4.52", "BZ-1090150", id="5_4_last_affected"),
        pytest.param("6.19.9", "CVE-2026-23453", id="6_19_last_affected"),
    ],
)
def test_kernel_version_check_flags_known_affected_versions(kernel_version, expected_bug_id):
    result = kernel_version_check(kernel_version)

    assert result["kernel_version"] == kernel_version
    assert result["has_known_issues"] is True
    assert len(result["known_bugs"]) >= 1

    first_bug = result["known_bugs"][0]
    assert set(first_bug.keys()) == REQUIRED_BUG_KEYS
    assert first_bug["bug_id"] == expected_bug_id
    assert first_bug["description"].strip()
    assert first_bug["affected_range"].strip()
    assert first_bug["fix_version"].strip()
    assert first_bug["reference"].startswith("http")


@pytest.mark.parametrize(
    "kernel_version",
    [
        pytest.param("4.14.40", id="4_14_fixed"),
        pytest.param("4.15.18", id="4_15_fixed"),
        pytest.param("5.4.53", id="5_4_fixed"),
        pytest.param("6.19.10", id="6_19_fixed"),
        pytest.param("5.4.53-150-generic", id="distro_suffix_fixed"),
    ],
)
def test_kernel_version_check_excludes_fixed_versions(kernel_version):
    result = kernel_version_check(kernel_version)

    assert result["kernel_version"] == kernel_version
    assert result["has_known_issues"] is False
    assert result["known_bugs"] == []


def test_kernel_version_check_understands_distro_suffix_for_affected_version():
    result = kernel_version_check("5.4.52-150-generic")

    assert result["kernel_version"] == "5.4.52-150-generic"
    assert result["has_known_issues"] is True
    assert result["known_bugs"][0]["bug_id"] == "BZ-1090150"


@pytest.mark.parametrize(
    "kernel_version",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty_string"),
    ],
)
def test_kernel_version_check_returns_error_for_missing_kernel_version(kernel_version):
    result = kernel_version_check(kernel_version)

    assert result["kernel_version"] is None
    assert result["known_bugs"] == []
    assert result["has_known_issues"] is False
    assert result["error"] == "No kernel version information provided."


@pytest.mark.parametrize(
    "kernel_version",
    [
        pytest.param("not-a-version", id="invalid_version_string"),
        pytest.param("5", id="missing_minor_version"),
    ],
)
def test_kernel_version_check_returns_no_known_issues_for_malformed_versions(kernel_version):
    result = kernel_version_check(kernel_version)

    assert result["kernel_version"] == kernel_version
    assert result["has_known_issues"] is False
    assert result["known_bugs"] == []
    assert "error" not in result


def test_kernel_version_check_returns_no_known_issues_for_valid_version_not_in_db():
    result = kernel_version_check("7.1.0")

    assert result["kernel_version"] == "7.1.0"
    assert result["has_known_issues"] is False
    assert result["known_bugs"] == []
    assert "error" not in result


def test_kernel_version_check_handles_custom_suffix_for_known_affected_version():
    result = kernel_version_check("5.4.52-custom")

    assert result["kernel_version"] == "5.4.52-custom"
    assert result["has_known_issues"] is True
    assert result["known_bugs"][0]["bug_id"] == "BZ-1090150"