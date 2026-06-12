from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INCLUDE_PATTERNS = (
    "test_*.py",
    "*_test.py",
    "live_test_*.py",
)

DEFAULT_EXCLUDE_DIR_PATTERNS = (
    "__pycache__",
    ".pytest_cache",
    "backup_*",
)


@dataclass
class TestRunResult:
    path: Path
    returncode: int

    @property
    def status(self) -> str:
        if self.returncode == 0:
            return "PASS"
        if self.returncode == 130:
            return "INTERRUPTED"
        return "FAIL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all current and future Python test files under tests/.",
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory to search for Python test files.",
    )
    parser.add_argument(
        "--include-pattern",
        dest="include_patterns",
        action="append",
        default=list(DEFAULT_INCLUDE_PATTERNS),
        help="Filename glob to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-dir-pattern",
        dest="exclude_dir_patterns",
        action="append",
        default=list(DEFAULT_EXCLUDE_DIR_PATTERNS),
        help="Directory-name glob to exclude recursively. Can be passed multiple times.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered test files without executing pytest.",
    )
    parser.add_argument(
        "--live-llm",
        type=str,
        default=None,
        help="LLM alias/model forwarded to live tests as `--llm`.",
    )
    parser.add_argument(
        "--live-runs",
        type=int,
        default=None,
        help="Run count forwarded to live tests as `--runs`.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to pytest. Prefix with --pytest-args.",
    )
    return parser.parse_args()


def should_exclude(path: Path, tests_dir: Path, exclude_dir_patterns: list[str]) -> bool:
    relative_parts = path.relative_to(tests_dir).parts[:-1]
    for part in relative_parts:
        for pattern in exclude_dir_patterns:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def is_included_test_file(path: Path, include_patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in include_patterns)


def discover_test_files(
    tests_dir: Path,
    include_patterns: list[str],
    exclude_dir_patterns: list[str],
) -> list[Path]:
    discovered: list[Path] = []

    for path in sorted(tests_dir.rglob("*.py")):
        if should_exclude(path, tests_dir, exclude_dir_patterns):
            continue
        if not is_included_test_file(path, include_patterns):
            continue
        discovered.append(path)

    return discovered


def normalize_pytest_args(pytest_args: list[str]) -> list[str]:
    forwarded_args = list(pytest_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    return forwarded_args


def build_pytest_command(project_root: Path, test_file: Path, pytest_args: list[str]) -> list[str]:
    relative_file = str(test_file.relative_to(project_root))
    forwarded_args = normalize_pytest_args(pytest_args)
    return [sys.executable, "-m", "pytest", relative_file, *forwarded_args]


def build_live_pytest_args(args: argparse.Namespace, test_file: Path) -> list[str]:
    forwarded_args = normalize_pytest_args(args.pytest_args)

    if not is_live_test(test_file):
        return forwarded_args

    live_args = list(forwarded_args)

    if "-s" not in live_args and "--capture=no" not in live_args:
        live_args.insert(0, "-s")

    if args.live_llm:
        live_args.extend([f"--llm={args.live_llm}"])

    if args.live_runs is not None:
        live_args.extend([f"--runs={args.live_runs}"])

    return live_args


def is_live_test(path: Path) -> bool:
    return fnmatch.fnmatch(path.name, "live_test_*.py")


def print_test_header(index: int, total: int, project_root: Path, test_file: Path) -> None:
    kind = "LIVE TEST" if is_live_test(test_file) else "TEST"
    relative_path = test_file.relative_to(project_root)
    print("\n" + "=" * 88)
    print(f"[{kind} {index}/{total}] {relative_path}")
    print("=" * 88)


def print_final_summary(project_root: Path, results: list[TestRunResult]) -> None:
    passed = sum(1 for result in results if result.returncode == 0)
    failed = sum(1 for result in results if result.returncode not in (0, 130))
    interrupted = sum(1 for result in results if result.returncode == 130)

    print("\n" + "#" * 88)
    print("[SUMMARY] Python test run results")
    print("#" * 88)
    for result in results:
        print(f" - {result.status:<11} {result.path.relative_to(project_root)}")
    print("-" * 88)
    print(
        f"[SUMMARY] total={len(results)}, passed={passed}, failed={failed}, interrupted={interrupted}"
    )


def main() -> int:
    args = parse_args()
    tests_dir = args.tests_dir.resolve()
    project_root = tests_dir.parent

    if not tests_dir.exists():
        print(f"[ERROR] tests directory not found: {tests_dir}")
        return 1

    test_files = discover_test_files(
        tests_dir=tests_dir,
        include_patterns=args.include_patterns,
        exclude_dir_patterns=args.exclude_dir_patterns,
    )

    print(f"[INFO] Discovered {len(test_files)} Python test file(s) under {tests_dir}")
    for path in test_files:
        print(f" - {path.relative_to(project_root)}")

    if not test_files:
        print("[ERROR] No matching Python test files were discovered.")
        return 1

    if args.list_only:
        return 0

    results: list[TestRunResult] = []

    for index, test_file in enumerate(test_files, start=1):
        print_test_header(index, len(test_files), project_root, test_file)
        command = build_pytest_command(
            project_root,
            test_file,
            build_live_pytest_args(args, test_file),
        )
        print("[INFO] Running pytest command:")
        print(" ".join(command))

        completed = subprocess.run(command, cwd=project_root)
        results.append(TestRunResult(path=test_file, returncode=completed.returncode))

    print_final_summary(project_root, results)

    if any(result.returncode not in (0,) for result in results):
        if any(result.returncode == 130 for result in results):
            return 130
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())