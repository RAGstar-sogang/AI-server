import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agent.tools.memory_calculator import memory_calculator


def test_memory_calculator_returns_sorted_top5_and_swap_summary():
    parsed_fields = {
        "total_ram_pages": 524288,
        "swap_total_kb": 1048576,
        "swap_free_kb": 524288,
        "process_table": [
            {"name": "proc-a", "pid": 101, "rss_kb": 102400, "oom_score_adj": 0},
            {"name": "proc-b", "pid": 102, "rss_kb": 204800, "oom_score_adj": -950},
            {"name": "proc-c", "pid": 103, "rss_kb": 51200, "oom_score_adj": 100},
            {"name": "proc-d", "pid": 104, "rss_kb": 307200, "oom_score_adj": 0},
            {"name": "proc-e", "pid": 105, "rss_kb": 153600, "oom_score_adj": 0},
            {"name": "proc-f", "pid": 106, "rss_kb": 25600, "oom_score_adj": 0},
        ],
    }

    result = memory_calculator(parsed_fields)

    assert result["ram_total_mb"] == 2048.0
    assert result["swap_total_mb"] == 1024.0
    assert result["swap_free_mb"] == 512.0
    assert result["swap_status"] == "enabled"

    top_processes = result["top_processes"]
    assert len(top_processes) == 5
    assert [proc["name"] for proc in top_processes] == [
        "proc-d",
        "proc-b",
        "proc-e",
        "proc-a",
        "proc-c",
    ]
    assert top_processes[0]["rss_mb"] == 300.0
    assert top_processes[-1]["rss_mb"] == 50.0
    assert top_processes[1]["protected"] is True
    assert result["total_top5_pct"] == 39.0


def test_memory_calculator_treats_oom_score_adj_minus_900_as_protected():
    result = memory_calculator(
        {
            "total_ram_pages": 1024,
            "swap_total_kb": 0,
            "swap_free_kb": 0,
            "process_table": [
                {"name": "boundary-proc", "pid": 1, "rss_kb": 1024, "oom_score_adj": -900},
            ],
        }
    )

    assert result["top_processes"][0]["protected"] is True


def test_memory_calculator_handles_zero_total_ram_without_division_error():
    result = memory_calculator(
        {
            "total_ram_pages": 0,
            "swap_total_kb": 0,
            "swap_free_kb": 0,
            "process_table": [
                {"name": "java", "pid": 3201, "rss_kb": 1048576, "oom_score_adj": 0},
            ],
        }
    )

    assert result["ram_total_mb"] == 0.0
    assert result["swap_status"] == "disabled"
    assert result["top_processes"][0]["rss_mb"] == 1024.0
    assert result["top_processes"][0]["ram_pct"] == 0.0
    assert result["total_top5_pct"] == 0.0


def test_memory_calculator_defaults_missing_process_fields():
    result = memory_calculator(
        {
            "total_ram_pages": 1024,
            "swap_total_kb": 0,
            "swap_free_kb": 0,
            "process_table": [{}],
        }
    )

    process = result["top_processes"][0]
    assert process == {
        "name": "unknown",
        "pid": 0,
        "rss_mb": 0.0,
        "ram_pct": 0.0,
        "protected": False,
    }


@pytest.mark.parametrize(
    ("process", "expected"),
    [
        pytest.param(
            {"pid": 7, "rss_kb": 2048},
            {
                "name": "unknown",
                "pid": 7,
                "rss_mb": 2.0,
                "ram_pct": 50.0,
                "protected": False,
            },
            id="missing_name_and_oom_score_adj",
        ),
        pytest.param(
            {"name": "rss-missing", "pid": 8, "oom_score_adj": -1000},
            {
                "name": "rss-missing",
                "pid": 8,
                "rss_mb": 0.0,
                "ram_pct": 0.0,
                "protected": True,
            },
            id="missing_rss_kb",
        ),
        pytest.param(
            {"name": "pid-missing", "rss_kb": 1024, "oom_score_adj": 0},
            {
                "name": "pid-missing",
                "pid": 0,
                "rss_mb": 1.0,
                "ram_pct": 25.0,
                "protected": False,
            },
            id="missing_pid",
        ),
    ],
)
def test_memory_calculator_handles_partially_missing_process_fields(process, expected):
    result = memory_calculator(
        {
            "total_ram_pages": 1024,
            "swap_total_kb": 0,
            "swap_free_kb": 0,
            "process_table": [process],
        }
    )

    assert result["top_processes"][0] == expected


@pytest.mark.parametrize(
    "parsed_fields",
    [
        pytest.param(
            {
                "total_ram_pages": 1024,
                "swap_total_kb": 0,
                "swap_free_kb": 0,
                "process_table": [
                    {"name": "broken", "pid": 1, "rss_kb": "not-a-number", "oom_score_adj": 0},
                ],
            },
            id="rss_kb_not_numeric",
        ),
        pytest.param(
            {
                "total_ram_pages": "not-a-number",
                "swap_total_kb": 0,
                "swap_free_kb": 0,
                "process_table": [],
            },
            id="total_ram_pages_not_numeric",
        ),
        pytest.param(
            {
                "total_ram_pages": 1024,
                "swap_total_kb": 0,
                "swap_free_kb": 0,
                "process_table": {"not": "a-list"},
            },
            id="process_table_not_list",
        ),
        pytest.param(
            {
                "total_ram_pages": 1024,
                "swap_total_kb": 0,
                "swap_free_kb": 0,
                "process_table": ["not-a-dict"],
            },
            id="process_entry_not_dict",
        ),
    ],
)
def test_memory_calculator_returns_fallback_structure_on_malformed_input(parsed_fields):
    result = memory_calculator(parsed_fields)

    assert result["top_processes"] == []
    assert result["total_top5_pct"] == 0.0
    assert result["ram_total_mb"] == 0.0
    assert result["swap_total_mb"] == 0.0
    assert result["swap_free_mb"] == 0.0
    assert result["swap_status"] == "unknown"
    assert "error" in result
    assert "Memory calculation error:" in result["error"]