import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agent.tools.kernel_param_recommender import kernel_param_recommender


REQUIRED_RECOMMENDATION_KEYS = {
    "name",
    "description",
    "current_implication",
    "recommendation",
    "command",
}


@pytest.mark.parametrize(
    ("oom_type", "expected_count", "expected_names"),
    [
        pytest.param(
            "global_oom",
            3,
            ["vm.overcommit_memory", "vm.min_free_kbytes", "vm.admin_reserve_kbytes"],
            id="global_oom",
        ),
        pytest.param(
            "swap_exhaustion",
            1,
            ["vm.swappiness"],
            id="swap_exhaustion",
        ),
        pytest.param(
            "cgroup_oom",
            3,
            ["memory.max", "memory.high", "memory.oom.group"],
            id="cgroup_oom",
        ),
        pytest.param(
            "page_alloc_failure",
            2,
            ["vm.min_free_kbytes", "vm.compact_memory"],
            id="page_alloc_failure",
        ),
    ],
)
def test_kernel_param_recommender_returns_expected_recommendations_for_known_oom_types(
    oom_type,
    expected_count,
    expected_names,
):
    result = kernel_param_recommender(oom_type, parsed_fields={"ignored": True})

    assert result["oom_type"] == oom_type
    assert result["count"] == expected_count

    recommendations = result["recommendations"]
    assert len(recommendations) == expected_count
    assert [item["name"] for item in recommendations] == expected_names
    assert all(set(item.keys()) == REQUIRED_RECOMMENDATION_KEYS for item in recommendations)
    assert all(item["description"].strip() for item in recommendations)
    assert all(item["current_implication"].strip() for item in recommendations)
    assert all(item["recommendation"].strip() for item in recommendations)
    assert all(item["command"].strip() for item in recommendations)
    assert all(len(item["command"].strip()) > 5 for item in recommendations)
    assert all(
        item["command"].startswith("sysctl ") or item["command"].startswith("echo ")
        for item in recommendations
    )


@pytest.mark.parametrize(
    "oom_type",
    [
        pytest.param("unknown_oom_type", id="unknown_label"),
        pytest.param("", id="empty_string"),
    ],
)
def test_kernel_param_recommender_returns_empty_list_for_unknown_oom_type(oom_type):
    result = kernel_param_recommender(oom_type, parsed_fields={})

    assert result == {
        "oom_type": oom_type,
        "recommendations": [],
        "count": 0,
    }


def test_kernel_param_recommender_accepts_noisy_parsed_fields_without_changing_result_shape():
    result = kernel_param_recommender(
        "global_oom",
        parsed_fields={
            "constraint": "CONSTRAINT_NONE",
            "swap_total_kb": 0,
            "order": 3,
            "nested": {"unexpected": [1, 2, 3]},
        },
    )

    assert result["oom_type"] == "global_oom"
    assert result["count"] == 3
    assert len(result["recommendations"]) == 3
    assert [item["name"] for item in result["recommendations"]] == [
        "vm.overcommit_memory",
        "vm.min_free_kbytes",
        "vm.admin_reserve_kbytes",
    ]
    assert all(isinstance(item, dict) for item in result["recommendations"])
    assert all(set(item.keys()) == REQUIRED_RECOMMENDATION_KEYS for item in result["recommendations"])