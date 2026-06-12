import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agent.nodes.node_1_parser import node_1_parser
from app.agent.nodes.node_3_executor import node_3_executor
from app.agent.tools.search_kb import search_kb


class SimpleKeywordEmbeddingFunction:
    """
    테스트 전용 deterministic embedding function.
    외부 Ollama 없이도 query_texts 검색 흐름을 검증할 수 있게 한다.
    """

    KEYWORDS = [
        "global_oom",
        "cgroup_oom",
        "swap_exhaustion",
        "page_alloc_failure",
        "constraint_none",
        "constraint_memcg",
        "swap",
        "cgroup",
        "memory",
        "limit",
        "order",
        "allocation",
        "kernel",
        "oom",
        "general",
        "java",
        "httpd",
    ]

    def name(self) -> str:
        return "simple-keyword-test-embedding"

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input: str) -> List[List[float]]:
        return [self([input])[0]]

    def __call__(self, input: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []

        for text in input:
            normalized = str(text).lower()
            tokens = normalized.replace("-", "_").split()

            vector: List[float] = []
            for keyword in self.KEYWORDS:
                score = float(normalized.count(keyword))
                score += float(tokens.count(keyword))
                vector.append(score)

            vector.append(float(len(normalized)))
            vector.append(float(len(tokens)))
            vectors.append(vector)

        return vectors


@pytest.fixture
def chroma_test_collection(tmp_path):
    """
    실제 ChromaDB 임시 컬렉션.
    mock이 아니라 진짜 query 흐름을 검증한다.
    """
    db_path = tmp_path / "chroma_test_db"
    client = chromadb.PersistentClient(path=str(db_path))

    collection = client.get_or_create_collection(
        name="oom_kb_test",
        embedding_function=SimpleKeywordEmbeddingFunction(),
    )

    documents = [
        "global_oom CONSTRAINT_NONE no swap space memory exhaustion java httpd troubleshooting guide",
        "general linux oom troubleshooting memory pressure general guide",
        "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit exceeded container guide",
        "swap_exhaustion swap fully consumed oom analysis guide",
        "page_alloc_failure order 2 page allocation failure fragmentation guide",
    ]

    metadatas = [
        {"title": "Global OOM Guide", "error_category": "global_oom"},
        {"title": "General OOM Guide", "error_category": "general"},
        {"title": "Cgroup OOM Guide", "error_category": "cgroup_oom"},
        {"title": "Swap Exhaustion Guide", "error_category": "swap_exhaustion"},
        {"title": "Page Allocation Failure Guide", "error_category": "page_alloc_failure"},
    ]

    ids = [
        "chunk_global_001",
        "chunk_general_001",
        "chunk_cgroup_001",
        "chunk_swap_001",
        "chunk_pagealloc_001",
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


@pytest.fixture
def case1_global_raw_log() -> str:
    """
    실제 테스트 로그 파일을 읽는다.
    요구사항: tests/node_1_logs/case1_global.txt 사용
    """
    log_path = Path(__file__).resolve().parent / "node_1_logs" / "case1_global.txt"
    assert log_path.exists(), f"테스트 로그 파일이 없습니다: {log_path}"
    return log_path.read_text(encoding="utf-8")


@pytest.fixture
def case1_global_parsed_fields(case1_global_raw_log: str) -> Dict[str, Any]:
    """
    실제 case1_global.txt를 Node 1 파서에 넣어 parsed_fields를 만든다.
    """
    initial_state = {
        "raw_log": case1_global_raw_log,
        "user_kernel_version": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }

    parsed_state = node_1_parser(initial_state)
    return parsed_state["parsed_fields"]


def test_case1_global_file_is_parsed_as_expected(case1_global_parsed_fields):
    """
    이 테스트는 정말로 case1_global.txt가 사용되고 있는지 확인하는 고정점이다.
    """
    parsed = case1_global_parsed_fields

    assert parsed["trigger_process"] == "httpd"
    assert parsed["killed_process"] == "java"
    assert parsed["killed_pid"] == 3201
    assert parsed["constraint"] == "CONSTRAINT_NONE"
    assert parsed["swap_total_kb"] == 0
    assert parsed["swap_free_kb"] == 0
    assert parsed["order"] == 0
    assert parsed["kernel_version"] == "4.18.0-305.el8.x86_64"


def test_search_kb_spec_signature_returns_chunk_list_from_real_chromadb(
    chroma_test_collection,
    case1_global_parsed_fields,
):
    """
    [현재 구현 기준 계약 테스트]

    기대 계약:
        search_kb(
            oom_type: str,
            parsed_fields: dict,
            collection: Any | None = None,
        ) -> dict
    """

    parsed = case1_global_parsed_fields

    result = search_kb(
        oom_type="global_oom",
        parsed_fields=parsed,
        collection=chroma_test_collection,
    )

    assert set(result.keys()) == {"query_used", "chunks", "total_found"}
    assert isinstance(result["query_used"], str)
    assert isinstance(result["chunks"], list)
    assert isinstance(result["total_found"], int)
    assert result["total_found"] >= 1, "최소 1개 이상의 KB chunk가 검색되어야 합니다."

    first = result["chunks"][0]
    assert set(first.keys()) == {"chunk_id", "content", "score", "metadata"}

    assert isinstance(first["chunk_id"], str)
    assert isinstance(first["content"], str)
    assert isinstance(first["score"], (int, float))
    assert isinstance(first["metadata"], dict)

    returned_categories = {
        chunk["metadata"].get("error_category")
        for chunk in result["chunks"]
    }
    assert returned_categories.issubset({"global_oom", "general"})


def test_search_kb_spec_signature_respects_top_k(
    chroma_test_collection,
    case1_global_parsed_fields,
):
    """
    [현재 구현 기준 계약 테스트]
    전역 OOM 질의에서 global/general 범주만 반환되어야 한다.
    """
    parsed = case1_global_parsed_fields

    result = search_kb(
        oom_type="global_oom",
        parsed_fields=parsed,
        collection=chroma_test_collection,
    )

    assert isinstance(result, dict)
    assert result["total_found"] == len(result["chunks"])
    assert result["total_found"] <= 5
    assert all(
        chunk["metadata"].get("error_category") in {"global_oom", "general"}
        for chunk in result["chunks"]
    )


def test_node_3_executor_populates_kb_chunks_using_case1_global_log(
    monkeypatch,
    chroma_test_collection,
    case1_global_raw_log,
):
    """
    [명세 기준 통합 테스트]

    실제 case1_global.txt를 입력으로 써서
    Node 1 -> Node 3 흐름 중 KB 검색 계약을 검증한다.

    기대:
        state["tool_results"]["kb_chunks"] = {
            "query_used": str,
            "chunks": list[dict],
            "total_found": int
        }
    """
    import app.agent.tools.search_kb as search_kb_module

    monkeypatch.setattr(
        search_kb_module,
        "get_collection",
        lambda: chroma_test_collection,
    )

    initial_state = {
        "raw_log": case1_global_raw_log,
        "user_kernel_version": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }
    parsed_state = node_1_parser(initial_state)

    state_for_node3 = {
        **parsed_state,
        "classification": {
            "oom_type": "global_oom",
            "tools_needed": [],
            "needs_kb": True,
            "confidence": "high",
        },
    }

    result_state = node_3_executor(state_for_node3)

    assert "tool_results" in result_state
    assert "kb_chunks" in result_state["tool_results"]

    kb_result = result_state["tool_results"]["kb_chunks"]

    assert set(kb_result.keys()) >= {"query_used", "chunks", "total_found"}
    assert isinstance(kb_result["query_used"], str)
    assert isinstance(kb_result["chunks"], list)
    assert isinstance(kb_result["total_found"], int)
    assert "error" not in kb_result

    assert "global_oom" in kb_result["query_used"]
    assert "CONSTRAINT_NONE" in kb_result["query_used"]
    assert "no swap space" in kb_result["query_used"]
    assert "order 0 page allocation failure" not in kb_result["query_used"]
    assert "cgroup memory limit exceeded" not in kb_result["query_used"]

    assert kb_result["total_found"] >= 1

    first_chunk = kb_result["chunks"][0]
    assert set(first_chunk.keys()) == {"chunk_id", "content", "score", "metadata"}
    assert first_chunk["metadata"]["error_category"] in {"global_oom", "general"}


def test_node_3_executor_query_for_case1_global_log_contains_expected_signals(
    monkeypatch,
    chroma_test_collection,
    case1_global_raw_log,
):
    """
    case1_global.txt 기준으로 Node 3가 조립한 쿼리에
    핵심 단서가 포함되는지만 검증한다.
    """
    import app.agent.tools.search_kb as search_kb_module

    monkeypatch.setattr(
        search_kb_module,
        "get_collection",
        lambda: chroma_test_collection,
    )

    initial_state = {
        "raw_log": case1_global_raw_log,
        "user_kernel_version": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }
    parsed_state = node_1_parser(initial_state)

    state_for_node3 = {
        **parsed_state,
        "classification": {
            "oom_type": "global_oom",
            "tools_needed": [],
            "needs_kb": True,
            "confidence": "high",
        },
    }

    result_state = node_3_executor(state_for_node3)
    query_used = result_state["tool_results"]["kb_chunks"]["query_used"]

    assert "global_oom" in query_used
    assert "CONSTRAINT_NONE" in query_used
    assert "no swap space" in query_used