import os
import sys
from copy import deepcopy

import pytest


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.agent.nodes.node_3_executor as executor_module
from app.agent.nodes.node_3_executor import node_3_executor


def create_mock_state(
    *,
    tools_needed: list[str] | None = None,
    needs_kb: bool = False,
    oom_type: str = "global_oom",
    parsed_overrides: dict | None = None,
) -> dict:
    state = {
        "raw_log": "Mock log data...",
        "user_kernel_version": None,
        "parsed_fields": {
            "kernel_version": "5.4.52-150-generic",
            "total_ram_pages": 524288,
            "swap_total_kb": 0,
            "swap_free_kb": 0,
            "constraint": "CONSTRAINT_NONE",
            "process_table": [
                {"pid": 1001, "name": "java", "rss_kb": 1048576, "oom_score_adj": 0},
                {"pid": 1002, "name": "sshd", "rss_kb": 51200, "oom_score_adj": -1000},
            ],
        },
        "classification": {
            "oom_type": oom_type,
            "tools_needed": tools_needed or [],
            "needs_kb": needs_kb,
            "confidence": "high",
        },
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }

    if parsed_overrides:
        state["parsed_fields"].update(parsed_overrides)

    return state


def assert_state_passthrough(original_state: dict, updated_state: dict) -> None:
    assert updated_state["raw_log"] == original_state["raw_log"]
    assert updated_state["user_kernel_version"] == original_state["user_kernel_version"]
    assert updated_state["parsed_fields"] == original_state["parsed_fields"]
    assert updated_state["classification"] == original_state["classification"]
    assert updated_state["diagnosis"] == original_state["diagnosis"]
    assert updated_state["error"] == original_state["error"]


def test_node_3_merges_memory_kernel_params_and_kb_results(monkeypatch):
    captured = {
        "memory_called": 0,
        "kernel_params_called": 0,
        "search_kb_called": 0,
    }

    fake_memory = {
        "top_processes": [{"name": "java", "rss_mb": 1024.0}],
        "total_top5_pct": 50.0,
        "ram_total_mb": 2048.0,
        "swap_total_mb": 0.0,
        "swap_free_mb": 0.0,
        "swap_status": "disabled",
    }
    fake_kernel_params = {
        "oom_type": "global_oom",
        "recommendations": [
            {"name": "vm.overcommit_memory", "command": "sysctl -w vm.overcommit_memory=2"}
        ],
        "count": 1,
    }
    fake_kb = {
        "query_used": "global_oom CONSTRAINT_NONE no swap space",
        "chunks": [
            {
                "chunk_id": "chunk_general_1",
                "content": "general OOM tuning 문서",
                "score": 0.25,
                "metadata": {"error_category": "general"},
            }
        ],
        "total_found": 1,
    }

    def fake_memory_calculator(parsed_fields):
        captured["memory_called"] += 1
        captured["memory_args"] = parsed_fields
        return fake_memory

    def fake_kernel_param_recommender(oom_type, parsed_fields):
        captured["kernel_params_called"] += 1
        captured["kernel_params_args"] = (oom_type, parsed_fields)
        return fake_kernel_params

    def fake_search_kb(oom_type, parsed_fields, collection=None):
        captured["search_kb_called"] += 1
        captured["search_kb_args"] = (oom_type, parsed_fields, collection)
        return fake_kb

    monkeypatch.setattr(executor_module, "memory_calculator", fake_memory_calculator)
    monkeypatch.setattr(
        executor_module,
        "kernel_param_recommender",
        fake_kernel_param_recommender,
    )
    monkeypatch.setattr(executor_module, "search_kb", fake_search_kb)

    state = create_mock_state(
        tools_needed=["memory_calculator", "kernel_param_recommender"],
        needs_kb=True,
        oom_type="global_oom",
    )
    original_state = deepcopy(state)

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert_state_passthrough(original_state, updated_state)
    assert set(results.keys()) == {"memory", "kernel_params", "kb_chunks"}
    assert results["memory"] == fake_memory
    assert results["kernel_params"] == fake_kernel_params
    assert results["kb_chunks"] == fake_kb

    assert captured["memory_called"] == 1
    assert captured["memory_args"] == state["parsed_fields"]
    assert captured["kernel_params_called"] == 1
    assert captured["kernel_params_args"] == ("global_oom", state["parsed_fields"])
    assert captured["search_kb_called"] == 1
    assert captured["search_kb_args"] == ("global_oom", state["parsed_fields"], None)


def test_node_3_keeps_other_results_when_one_tool_returns_error_dict(monkeypatch):
    fake_memory = {
        "top_processes": [{"name": "java", "rss_mb": 1024.0}],
        "total_top5_pct": 50.0,
        "ram_total_mb": 2048.0,
        "swap_total_mb": 0.0,
        "swap_free_mb": 0.0,
        "swap_status": "disabled",
    }
    fake_kernel_params_error = {
        "error": "kernel_param_recommender temporary failure",
        "oom_type": "global_oom",
        "recommendations": [],
        "count": 0,
    }
    fake_kb = {
        "query_used": "global_oom CONSTRAINT_NONE no swap space",
        "chunks": [],
        "total_found": 0,
    }

    monkeypatch.setattr(executor_module, "memory_calculator", lambda parsed_fields: fake_memory)
    monkeypatch.setattr(
        executor_module,
        "kernel_param_recommender",
        lambda oom_type, parsed_fields: fake_kernel_params_error,
    )
    monkeypatch.setattr(
        executor_module,
        "search_kb",
        lambda oom_type, parsed_fields, collection=None: fake_kb,
    )

    state = create_mock_state(
        tools_needed=["memory_calculator", "kernel_param_recommender"],
        needs_kb=True,
        oom_type="global_oom",
    )

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert set(results.keys()) == {"memory", "kernel_params", "kb_chunks"}
    assert results["memory"] == fake_memory
    assert results["kernel_params"] == fake_kernel_params_error
    assert results["kb_chunks"] == fake_kb
    assert results["kernel_params"]["error"] == "kernel_param_recommender temporary failure"


def test_node_3_propagates_exception_when_one_tool_raises(monkeypatch):
    captured = {"search_kb_called": 0}

    fake_memory = {
        "top_processes": [{"name": "java", "rss_mb": 1024.0}],
        "total_top5_pct": 50.0,
        "ram_total_mb": 2048.0,
        "swap_total_mb": 0.0,
        "swap_free_mb": 0.0,
        "swap_status": "disabled",
    }
    fake_kernel_params = {
        "oom_type": "global_oom",
        "recommendations": [],
        "count": 0,
    }

    def fake_search_kb(oom_type, parsed_fields, collection=None):
        captured["search_kb_called"] += 1
        raise RuntimeError("search backend exploded")

    monkeypatch.setattr(executor_module, "memory_calculator", lambda parsed_fields: fake_memory)
    monkeypatch.setattr(
        executor_module,
        "kernel_param_recommender",
        lambda oom_type, parsed_fields: fake_kernel_params,
    )
    monkeypatch.setattr(executor_module, "search_kb", fake_search_kb)

    state = create_mock_state(
        tools_needed=["memory_calculator", "kernel_param_recommender"],
        needs_kb=True,
        oom_type="global_oom",
    )

    with pytest.raises(RuntimeError, match="search backend exploded"):
        node_3_executor(state)

    assert captured["search_kb_called"] == 1