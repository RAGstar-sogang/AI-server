import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import app.agent.tools.search_kb as search_kb_module
from app.agent.tools.search_kb import search_kb


def sanitized_query_tokens() -> tuple[str, ...]:
    return ("global_oom", "CONSTRAINT_NONE", "no swap space")


class DummyCollection:
    def __init__(self, result=None):
        self.called = 0
        self.last_kwargs = None
        self.result = result or {
            "documents": [[]],
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

    def query(self, **kwargs):
        self.called += 1
        self.last_kwargs = kwargs
        return self.result


class FailingCollection:
    def __init__(self):
        self.called = 0
        self.last_kwargs = None

    def query(self, **kwargs):
        self.called += 1
        self.last_kwargs = kwargs
        raise RuntimeError("simulated query failure")


@pytest.mark.parametrize(
    "malformed_order",
    [
        pytest.param("", id="empty_string_order"),
        pytest.param("abc", id="nonnumeric_string_order"),
    ],
)
def test_search_kb_ignores_malformed_order_and_returns_structured_response(malformed_order):
    """
    문자열 기반의 비정상 order 값은 현재 구현에서 ValueError로 무시되고,
    search_kb는 계속 query를 수행한다.
    """
    collection = DummyCollection()

    result = search_kb(
        oom_type="global_oom",
        parsed_fields={
            "constraint": "CONSTRAINT_NONE",
            "swap_total_kb": 0,
            "order": malformed_order,
        },
        collection=collection,
    )

    assert result["total_found"] == 0
    assert result["chunks"] == []
    assert "error" not in result

    query_used = result["query_used"]
    for token in sanitized_query_tokens():
        assert token in query_used
    assert "order" not in query_used
    assert "page allocation" not in query_used

    assert collection.called == 1
    assert collection.last_kwargs is not None
    assert collection.last_kwargs["n_results"] == 5
    assert collection.last_kwargs["where"] == {
        "error_category": {"$in": ["global_oom", "general"]}
    }
    assert len(collection.last_kwargs["query_texts"]) == 1
    assert collection.last_kwargs["query_texts"][0] == query_used


@pytest.mark.parametrize(
    "malformed_order",
    [
        pytest.param([0], id="list_order"),
        pytest.param({}, id="dict_order"),
        pytest.param((1,), id="tuple_order"),
        pytest.param(object(), id="object_order"),
    ],
)
def test_search_kb_raises_type_error_on_non_scalar_order_input(malformed_order):
    """
    현재 구현은 비스칼라 order 입력을 방어하지 못하고 TypeError를 전파한다.
    이 테스트는 그 동작을 회귀용으로 고정한다.
    """
    with pytest.raises(TypeError):
        search_kb(
            oom_type="global_oom",
            parsed_fields={
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": 0,
                "order": malformed_order,
            },
            collection=DummyCollection(),
        )


def test_search_kb_returns_structured_error_on_query_failure():
    """
    ChromaDB query가 실패해도 search_kb는 예외를 전파하지 않고
    구조화된 에러 응답을 반환해야 한다.
    """
    collection = FailingCollection()

    result = search_kb(
        oom_type="global_oom",
        parsed_fields={
            "constraint": "CONSTRAINT_NONE",
            "swap_total_kb": 0,
            "order": 0,
        },
        collection=collection,
    )

    assert result["total_found"] == 0
    assert result["chunks"] == []
    assert "error" in result
    assert "KB 검색 오류:" in result["error"]
    assert "simulated query failure" in result["error"]

    query_used = result["query_used"]
    for token in sanitized_query_tokens():
        assert token in query_used
    assert "order" not in query_used
    assert "page allocation" not in query_used

    assert collection.called == 1
    assert collection.last_kwargs is not None
    assert collection.last_kwargs["n_results"] == 5
    assert collection.last_kwargs["where"] == {
        "error_category": {"$in": ["global_oom", "general"]}
    }
    assert collection.last_kwargs["query_texts"] == [query_used]


def test_search_kb_builds_cgroup_query_and_maps_query_results():
    """
    search_kb 직접 계약 테스트.

    cgroup 관련 parsed_fields가 들어오면 해당 신호를 쿼리에 반영하고,
    collection.query 결과를 chunks 구조로 변환해야 한다.
    """
    collection = DummyCollection(
        result={
            "documents": [[
                "cgroup memory limit exceeded 문서",
                "general OOM tuning 문서",
            ]],
            "ids": [["chunk_1", "chunk_2"]],
            "distances": [[0.12, 0.34]],
            "metadatas": [[
                {"error_category": "cgroup_oom"},
                {"error_category": "general"},
            ]],
        }
    )

    result = search_kb(
        oom_type="cgroup_oom",
        parsed_fields={
            "constraint": "CONSTRAINT_MEMCG",
            "cgroup_path": "/docker/my_container",
            "swap_total_kb": 0,
            "order": 3,
        },
        collection=collection,
    )

    assert collection.called == 1
    assert collection.last_kwargs == {
        "query_texts": [
            "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space order 3 page allocation"
        ],
        "n_results": 5,
        "where": {"error_category": {"$in": ["cgroup_oom", "general"]}},
    }

    assert result["query_used"] == (
        "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space order 3 page allocation"
    )
    assert result["total_found"] == 2
    assert [chunk["chunk_id"] for chunk in result["chunks"]] == ["chunk_1", "chunk_2"]
    assert result["chunks"][0]["metadata"]["error_category"] == "cgroup_oom"
    assert result["chunks"][1]["metadata"]["error_category"] == "general"
    assert result["chunks"][0]["score"] == pytest.approx(0.12)
    assert result["chunks"][1]["score"] == pytest.approx(0.34)


def test_search_kb_returns_structured_error_when_get_collection_fails(monkeypatch):
    """
    collection을 주입하지 않은 상태에서 get_collection이 실패하면
    현재 구현은 예외를 전파한다.
    """
    def raising_get_collection():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(search_kb_module, "get_collection", raising_get_collection)

    with pytest.raises(RuntimeError, match="connection refused"):
        search_kb(
            oom_type="global_oom",
            parsed_fields={
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": 0,
                "order": 0,
            },
            collection=None,
        )


def test_search_kb_uses_get_collection_on_success(monkeypatch):
    """
    collection을 주입하지 않아도 get_collection이 정상 반환되면
    search_kb는 그 컬렉션으로 검색을 수행하고 구조화된 성공 응답을 반환해야 한다.
    """
    collection = DummyCollection(
        result={
            "documents": [["general OOM tuning 문서"]],
            "ids": [["chunk_general_1"]],
            "distances": [[0.25]],
            "metadatas": [[{"error_category": "general"}]],
        }
    )

    monkeypatch.setattr(search_kb_module, "get_collection", lambda: collection)

    result = search_kb(
        oom_type="global_oom",
        parsed_fields={
            "constraint": "CONSTRAINT_NONE",
            "swap_total_kb": 0,
            "order": 0,
        },
        collection=None,
    )

    assert result["total_found"] == 1
    assert "error" not in result
    assert result["chunks"][0]["chunk_id"] == "chunk_general_1"
    assert result["chunks"][0]["metadata"]["error_category"] == "general"

    query_used = result["query_used"]
    for token in sanitized_query_tokens():
        assert token in query_used
    assert "order" not in query_used
    assert "page allocation" not in query_used

    assert collection.called == 1
    assert collection.last_kwargs is not None
    assert collection.last_kwargs["query_texts"] == [query_used]


def test_search_kb_returns_structured_error_on_malformed_result_shape():
    """
    현재 구현은 손상된 Chroma 결과 스키마를 방어하지 못하고 IndexError를 전파한다.
    """
    collection = DummyCollection(
        result={
            "documents": [["doc-1", "doc-2"]],
            "ids": [["chunk_1"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"error_category": "general"}, {"error_category": "general"}]],
        }
    )

    with pytest.raises(IndexError):
        search_kb(
            oom_type="global_oom",
            parsed_fields={
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": 0,
                "order": 0,
            },
            collection=collection,
        )