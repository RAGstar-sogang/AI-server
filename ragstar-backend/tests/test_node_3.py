import os
import sys
from copy import deepcopy

import pytest

# 프로젝트 루트 경로 추가
# tests 디렉터리에서 실행해도 app 패키지를 import할 수 있게 한다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.agent.nodes.node_3_executor as executor_module
import app.agent.tools.search_kb as search_kb_module
from app.agent.nodes.node_3_executor import node_3_executor


def create_mock_state(
    *,
    tools_needed: list[str] | None = None,
    needs_kb: bool = False,
    oom_type: str = "global_oom",
    kernel_version: str | None = None,
    parsed_overrides: dict | None = None,
) -> dict:
    """
    현재 OOMState 스키마에 맞는 기본 mock state 생성기.

    Node 3가 실제로 참조하는 필드만 중심적으로 구성한다.
    필요한 경우 parsed_overrides로 일부 필드를 덮어쓴다.
    """
    state = {
        "raw_log": "Mock log data...",
        "user_kernel_version": None,
        "parsed_fields": {
            "kernel_version": kernel_version,
            "total_ram_pages": 524288,  # 2GB RAM
            "swap_total_kb": 0,
            "swap_free_kb": 0,
            "constraint": "CONSTRAINT_NONE",
            "process_table": [
                {"pid": 1001, "name": "java", "rss_kb": 1048576, "oom_score_adj": 0},
                {"pid": 1002, "name": "sshd", "rss_kb": 51200, "oom_score_adj": -1000},
                {"pid": 1003, "name": "nginx", "rss_kb": 204800, "oom_score_adj": 0},
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
    """
    Node 3 wrapper 계약 검증:

    - tool_results만 새로 채워지고
    - 나머지 state는 그대로 유지되어야 한다.
    """
    assert updated_state["raw_log"] == original_state["raw_log"]
    assert updated_state["user_kernel_version"] == original_state["user_kernel_version"]
    assert updated_state["parsed_fields"] == original_state["parsed_fields"]
    assert updated_state["classification"] == original_state["classification"]
    assert updated_state["diagnosis"] == original_state["diagnosis"]
    assert updated_state["error"] == original_state["error"]


def test_node_3_memory_calculator_only(monkeypatch):
    """
    Node 3는 memory_calculator를 호출하고,
    그 반환값을 tool_results['memory']에 그대로 담아야 한다.

    또한 parsed_fields를 정확히 인자로 넘겨야 한다.
    """
    fake_memory = {
        "summary": "mocked memory analysis",
        "top_process": "java",
        "risk": "high",
    }
    captured = {"called": 0, "args": None, "kwargs": None}

    def fake_memory_calculator(*args, **kwargs):
        captured["called"] += 1
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_memory

    monkeypatch.setattr(executor_module, "memory_calculator", fake_memory_calculator)

    state = create_mock_state(tools_needed=["memory_calculator"])
    original_state = deepcopy(state)

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert_state_passthrough(original_state, updated_state)

    assert set(results.keys()) == {"memory"}
    assert results["memory"] == fake_memory

    assert captured["called"] == 1
    assert captured["args"] == (state["parsed_fields"],)
    assert captured["kwargs"] == {}


def test_node_3_kernel_version_check_only(monkeypatch):
    """
    Node 3는 kernel_version_check를 호출하고,
    그 반환값을 tool_results['kernel_bugs']에 그대로 담아야 한다.

    또한 kernel_version 문자열을 정확히 인자로 넘겨야 한다.
    """
    fake_kernel_bugs = {
        "kernel_version": "5.4.0-30-generic",
        "has_known_issues": True,
        "known_issues": ["mock issue"],
    }
    captured = {"called": 0, "args": None, "kwargs": None}

    def fake_kernel_version_check(*args, **kwargs):
        captured["called"] += 1
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_kernel_bugs

    monkeypatch.setattr(executor_module, "kernel_version_check", fake_kernel_version_check)

    state = create_mock_state(
        tools_needed=["kernel_version_check"],
        kernel_version="5.4.0-30-generic",
    )
    original_state = deepcopy(state)

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert_state_passthrough(original_state, updated_state)

    assert set(results.keys()) == {"kernel_bugs"}
    assert results["kernel_bugs"] == fake_kernel_bugs

    assert captured["called"] == 1
    assert captured["args"] == ("5.4.0-30-generic",)
    assert captured["kwargs"] == {}


def test_node_3_memory_and_kernel_version_check_together(monkeypatch):
    """
    여러 도구가 동시에 요청되면 Node 3가 둘 다 실행하고,
    같은 tool_results 아래에 각각 담아야 한다.

    또한 각 도구가 올바른 인자를 받아 호출되어야 한다.
    """
    fake_memory = {
        "summary": "mocked memory analysis",
        "top_process": "java",
    }
    fake_kernel_bugs = {
        "kernel_version": "4.15.0-10-generic",
        "has_known_issues": True,
        "known_issues": ["mock CVE"],
    }

    captured = {
        "memory_called": 0,
        "memory_args": None,
        "memory_kwargs": None,
        "kernel_called": 0,
        "kernel_args": None,
        "kernel_kwargs": None,
    }

    def fake_memory_calculator(*args, **kwargs):
        captured["memory_called"] += 1
        captured["memory_args"] = args
        captured["memory_kwargs"] = kwargs
        return fake_memory

    def fake_kernel_version_check(*args, **kwargs):
        captured["kernel_called"] += 1
        captured["kernel_args"] = args
        captured["kernel_kwargs"] = kwargs
        return fake_kernel_bugs

    monkeypatch.setattr(executor_module, "memory_calculator", fake_memory_calculator)
    monkeypatch.setattr(executor_module, "kernel_version_check", fake_kernel_version_check)

    state = create_mock_state(
        tools_needed=["memory_calculator", "kernel_version_check"],
        kernel_version="4.15.0-10-generic",
    )
    original_state = deepcopy(state)

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert_state_passthrough(original_state, updated_state)

    assert set(results.keys()) == {"memory", "kernel_bugs"}
    assert results["memory"] == fake_memory
    assert results["kernel_bugs"] == fake_kernel_bugs

    assert captured["memory_called"] == 1
    assert captured["memory_args"] == (state["parsed_fields"],)
    assert captured["memory_kwargs"] == {}

    assert captured["kernel_called"] == 1
    assert captured["kernel_args"] == ("4.15.0-10-generic",)
    assert captured["kernel_kwargs"] == {}


def test_node_3_kernel_param_recommender_only(monkeypatch):
    """
    Node 3는 kernel_param_recommender를 호출하고,
    그 반환값을 tool_results['kernel_params']에 그대로 담아야 한다.

    또한 (oom_type, parsed_fields)를 정확히 인자로 넘겨야 한다.
    """
    fake_kernel_params = {
        "oom_type": "global_oom",
        "recommendations": [
            {
                "name": "vm.overcommit_memory",
                "recommendation": "set to 2",
                "command": "sysctl -w vm.overcommit_memory=2",
            }
        ],
    }
    captured = {"called": 0, "args": None, "kwargs": None}

    def fake_kernel_param_recommender(*args, **kwargs):
        captured["called"] += 1
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_kernel_params

    monkeypatch.setattr(
        executor_module,
        "kernel_param_recommender",
        fake_kernel_param_recommender,
    )

    state = create_mock_state(tools_needed=["kernel_param_recommender"])
    original_state = deepcopy(state)

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert_state_passthrough(original_state, updated_state)

    assert set(results.keys()) == {"kernel_params"}
    assert results["kernel_params"] == fake_kernel_params

    assert captured["called"] == 1
    assert captured["args"] == ("global_oom", state["parsed_fields"])
    assert captured["kwargs"] == {}


def test_node_3_search_kb_passes_oom_and_parsed_fields(monkeypatch):
    """
    needs_kb=True이면 Node 3가 search_kb에
    (oom_type, parsed_fields, collection=None)을 전달하고,
    반환 dict를 kb_chunks에 그대로 저장해야 한다.

    이 테스트는 성공 경로를 본다.
    """
    captured = {"called": 0}

    fake_result = {
        "query_used": "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space",
        "chunks": [
            {
                "chunk_id": "chunk_1",
                "content": "cgroup memory limit exceeded guide",
                "score": 0.11,
                "metadata": {"error_category": "cgroup_oom"},
            },
            {
                "chunk_id": "chunk_2",
                "content": "general OOM troubleshooting",
                "score": 0.24,
                "metadata": {"error_category": "general"},
            },
        ],
        "total_found": 2,
    }

    def fake_search_kb(oom_type, parsed_fields, collection=None):
        captured["called"] += 1
        captured["oom_type"] = oom_type
        captured["parsed_fields"] = parsed_fields
        captured["collection"] = collection
        return fake_result

    monkeypatch.setattr(executor_module, "search_kb", fake_search_kb)

    state = create_mock_state(
        tools_needed=[],
        needs_kb=True,
        oom_type="cgroup_oom",
        parsed_overrides={
            "constraint": "CONSTRAINT_MEMCG",
            "cgroup_path": "/docker/my_container",
            "swap_total_kb": 0,
            "swap_free_kb": 0,
        },
    )
    original_state = deepcopy(state)

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert_state_passthrough(original_state, updated_state)

    assert set(results.keys()) == {"kb_chunks"}

    assert captured["called"] == 1
    assert captured["oom_type"] == "cgroup_oom"
    assert captured["parsed_fields"] == state["parsed_fields"]
    assert captured["collection"] is None

    assert results["kb_chunks"] == fake_result


def test_node_3_search_kb_error_response_passthrough(monkeypatch):
    """
    search_kb가 error 포함 dict를 반환하면
    Node 3는 해당 dict를 그대로 kb_chunks에 저장해야 한다.
    """
    captured = {"called": 0}

    fake_result = {
        "query_used": "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space",
        "chunks": [],
        "total_found": 0,
        "error": "KB 검색 오류: simulated backend failure",
    }

    def fake_search_kb(oom_type, parsed_fields, collection=None):
        captured["called"] += 1
        captured["oom_type"] = oom_type
        captured["parsed_fields"] = parsed_fields
        captured["collection"] = collection
        return fake_result

    monkeypatch.setattr(executor_module, "search_kb", fake_search_kb)

    state = create_mock_state(
        tools_needed=[],
        needs_kb=True,
        oom_type="cgroup_oom",
        parsed_overrides={
            "constraint": "CONSTRAINT_MEMCG",
            "cgroup_path": "/docker/my_container",
        },
    )
    original_state = deepcopy(state)

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert_state_passthrough(original_state, updated_state)

    assert set(results.keys()) == {"kb_chunks"}

    assert captured["called"] == 1
    assert captured["oom_type"] == "cgroup_oom"
    assert captured["parsed_fields"] == state["parsed_fields"]
    assert captured["collection"] is None

    assert results["kb_chunks"] == fake_result


def test_node_3_search_kb_passes_order_field(monkeypatch):
    """
    parsed_fields의 order 값이 존재하면
    Node 3가 해당 필드를 포함한 parsed_fields를 search_kb에 전달해야 한다.
    """
    captured = {"called": 0}

    fake_result = {
        "query_used": "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space order 3 page allocation",
        "chunks": [],
        "total_found": 0,
    }

    def fake_search_kb(oom_type, parsed_fields, collection=None):
        captured["called"] += 1
        captured["oom_type"] = oom_type
        captured["parsed_fields"] = parsed_fields
        captured["collection"] = collection
        return fake_result

    monkeypatch.setattr(executor_module, "search_kb", fake_search_kb)

    state = create_mock_state(
        tools_needed=[],
        needs_kb=True,
        oom_type="cgroup_oom",
        parsed_overrides={
            "constraint": "CONSTRAINT_MEMCG",
            "cgroup_path": "/docker/my_container",
            "swap_total_kb": 0,
            "swap_free_kb": 0,
            "order": 3,
        },
    )
    original_state = deepcopy(state)

    updated_state = node_3_executor(state)
    results = updated_state["tool_results"]

    assert_state_passthrough(original_state, updated_state)

    assert set(results.keys()) == {"kb_chunks"}

    assert captured["called"] == 1
    assert captured["oom_type"] == "cgroup_oom"
    assert captured["parsed_fields"]["order"] == 3
    assert captured["parsed_fields"] == state["parsed_fields"]
    assert captured["collection"] is None

    assert results["kb_chunks"] == fake_result


def test_node_3_kb_search_propagates_type_error_on_non_scalar_order_input(monkeypatch):
    """
    현재 구현에서는 parsed_fields['order']에 비정상 타입이 들어오면
    search_kb 내부의 TypeError가 Node 3까지 전파된다.

    이 테스트는 해당 현재 동작을 회귀용으로 고정한다.
    """
    class FakeCollection:
        def __init__(self):
            self.called = 0
            self.last_kwargs = None

        def query(self, **kwargs):
            self.called += 1
            self.last_kwargs = kwargs
            return {
                "documents": [["general OOM tuning 문서"]],
                "ids": [["chunk_general_1"]],
                "distances": [[0.25]],
                "metadatas": [[{"error_category": "general"}]],
            }

    collection = FakeCollection()
    monkeypatch.setattr(search_kb_module, "get_collection", lambda: collection)

    state = create_mock_state(
        needs_kb=True,
        parsed_overrides={
            "order": [0],
            "constraint": "CONSTRAINT_NONE",
            "swap_total_kb": 0,
        },
    )

    with pytest.raises(TypeError):
        node_3_executor(state)

    assert collection.called == 0
    assert collection.last_kwargs is None
