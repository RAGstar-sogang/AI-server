import re
from packaging.version import Version

# Manually researched kernel OOM/memory-related vulnerability and bug database
KNOWN_OOM_BUGS = {
    "4.14": [
        {
            "bug_id": "CVE-2018-1000200",
            "description": "Null pointer dereference when a large mlocked process is killed by OOM (causes system crash)",
            "affected_range": "4.14.0 ~ 4.14.39",
            "fix_version": "4.14.40",
            "reference": "https://nvd.nist.gov/vuln/detail/cve-2018-1000200"
        }
    ],
    "4.15": [
        {
            "bug_id": "CVE-2018-1000200",
            "description": "Null pointer dereference when a large mlocked process is killed by OOM (causes system crash)",
            "affected_range": "4.15.0 ~ 4.15.17",
            "fix_version": "4.15.18",
            "reference": "https://nvd.nist.gov/vuln/detail/cve-2018-1000200"
        }
    ],
    "5.3": [
        {
            "bug_id": "K8S-ISSUE-61937",
            "description": "Cgroup v1 kmem (kernel memory) leak causes unreturned memory from terminated containers, leading to node-wide OOM",
            "affected_range": "4.0.0 ~ 5.3.18",
            "fix_version": "5.4.0",
            "reference": "https://github.com/kubernetes/kubernetes/issues/61937"
        }
    ],
    "5.4": [
        {
            "bug_id": "BZ-1090150",
            "description": "Memory compaction enters infinite loop under fragmentation, consuming CPU and triggering premature OOM",
            "affected_range": "5.4.0 ~ 5.4.52",
            "fix_version": "5.4.53",
            "reference": "https://bugzilla.suse.com/show_bug.cgi?id=1090150"
        }
    ],
    "6.19": [
        {
            "bug_id": "CVE-2026-23453",
            "description": "Memory leak and OOM caused by unreturned pages on packet drop in network driver non-zero-copy mode",
            "affected_range": "6.19.0 ~ 6.19.9",
            "fix_version": "6.19.10",
            "reference": "https://access.redhat.com/security/cve/CVE-2026-23453"
        }
    ]
}

def _extract_base_version(kernel_version: str) -> str:
    """Extract and normalize pure version (e.g. '5.4.0') from strings like '5.4.0-150-generic'."""
    match = re.match(r"(\d+\.\d+\.\d+)", kernel_version)
    return match.group(1) if match else kernel_version

def _is_affected(kernel_version: str, fix_version: str) -> bool:
    """
    Returns True if kernel_version is lower than fix_version (not yet patched).
    Note: Distro kernels may have backported patches not reflected in the version string.
    """
    try:
        current_ver = Version(_extract_base_version(kernel_version))
        fixed_ver = Version(fix_version)
        return current_ver < fixed_ver
    except Exception:
        return False

def kernel_version_check(kernel_version: str) -> dict:
    """
    Look up the parsed kernel version against a hardcoded known OOM bug database.
    """
    if not kernel_version:
        return {
            "kernel_version": None,
            "known_bugs": [],
            "has_known_issues": False,
            "error": "No kernel version information provided."
        }

    # "5.15.0-76-generic" → "5.15" (major.minor)
    parts = _extract_base_version(kernel_version).split(".")
    major_minor = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else None

    if major_minor and major_minor in KNOWN_OOM_BUGS:
        # Filter to only bugs not yet patched for this specific version
        applicable_bugs = [
            bug for bug in KNOWN_OOM_BUGS[major_minor]
            if _is_affected(kernel_version, bug["fix_version"])
        ]

        return {
            "kernel_version": kernel_version,
            "known_bugs": applicable_bugs,
            "has_known_issues": len(applicable_bugs) > 0
        }
    else:
        # Version not in DB, or no known bugs for this version
        return {
            "kernel_version": kernel_version,
            "known_bugs": [],
            "has_known_issues": False
        }