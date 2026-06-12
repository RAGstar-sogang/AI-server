# tests/live_test_ollama.py

import os
import sys
from pathlib import Path

import requests
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agent.nodes.node_1_parser import node_1_parser
from app.agent.nodes.node_2_classifier import node_2_classifier
from app.agent.tools.search_kb import search_kb
from app.database.chromadb_client import get_collection


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"


def _read_case1_global_log() -> str:
    """
    실테스트 입력 로그는 반드시 tests/node_1_logs/case1_global.txt 를 사용한다.
    """
    log_path = Path(__file__).resolve().parent / "node_1_logs" / "case1_global.txt"
    assert log_path.exists(), f"테스트 로그 파일이 없습니다: {log_path}"
    return log_path.read_text(encoding="utf-8")


def _read_log_from_node_1_logs(filename: str) -> str:
    """
    tests/node_1_logs/<filename> 파일을 읽는다.
    """
    log_path = Path(__file__).resolve().parent / "node_1_logs" / filename
    assert log_path.exists(), f"테스트 로그 파일이 없습니다: {log_path}"
    return log_path.read_text(encoding="utf-8")


def _build_minimal_state_from_case1() -> dict:
    """
    case1_global.txt를 실제 Node 1 파서에 태워
    Node 2 / ChromaDB live test에서 재사용할 기본 state를 만든다.
    """
    raw_log = _read_case1_global_log()

    state = {
        "raw_log": raw_log,
        "user_kernel_version": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }

    parsed_state = node_1_parser(state)
    return parsed_state


def _build_minimal_state_from_log_file(filename: str) -> dict:
    """
    지정한 실제 로그 파일을 Node 1 파서에 태워 기본 state를 만든다.
    """
    raw_log = _read_log_from_node_1_logs(filename)

    state = {
        "raw_log": raw_log,
        "user_kernel_version": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }

    return node_1_parser(state)


def test_live_ollama_server_is_reachable_and_has_models():
    """
    실제 Ollama 서버가 떠 있고,
    최소한 1개 이상의 모델이 로드 가능한 상태인지 확인한다.

    이 테스트는:
    - 서버 접근 가능 여부
    - 모델 목록 조회 가능 여부
    를 검증한다.
    """
    response = requests.get(OLLAMA_TAGS_URL, timeout=10)
    response.raise_for_status()

    payload = response.json()
    assert "models" in payload, "Ollama /api/tags 응답에 models 필드가 없습니다."
    assert isinstance(payload["models"], list), "models 필드는 list여야 합니다."

    model_names = []
    for model in payload["models"]:
        name = model.get("name")
        if name:
            model_names.append(name)

    assert model_names, "Ollama에 로드 가능한 모델이 하나도 없습니다."


def test_live_selected_models_are_available_when_overridden(pytestconfig):
    """
    live test CLI 인자로 모델 override를 준 경우,
    해당 모델이 실제 Ollama tags에 존재하는지 검증한다.
    """
    requested_models = []
    for option_name in ("--live-model", "--llm", "--live-node2-model", "--live-node4-model"):
        value = pytestconfig.getoption(option_name)
        if value:
            requested_models.append(str(value))

    if not requested_models:
        pytest.skip("No live test model override was provided.")

    response = requests.get(OLLAMA_TAGS_URL, timeout=10)
    response.raise_for_status()

    payload = response.json()
    model_names = {model.get("name") for model in payload.get("models", []) if model.get("name")}

    missing = []
    for model in requested_models:
        if model in model_names:
            continue
        if ":" not in model and f"{model}:latest" in model_names:
            continue
        missing.append(model)

    assert not missing, f"선택한 live test 모델이 Ollama에 없습니다: missing={missing}, available={sorted(model_names)}"


def test_live_chromadb_query_uses_real_ollama_embeddings_with_case1_global():
    """
    실제 ChromaDB + 실제 OllamaEmbeddingFunction 경로를 검증한다.

    검증 내용:
    - get_collection()이 실제 컬렉션을 로드하는지
    - query_texts 기반 질의가 실제로 수행되는지
    - case1_global.txt에서 유도한 global OOM 성격의 질의가
      최소 1개 이상의 KB 결과를 반환하는지
    """
    parsed_state = _build_minimal_state_from_case1()
    parsed = parsed_state["parsed_fields"]

    collection = get_collection()
    assert collection is not None, "ChromaDB 컬렉션 로드에 실패했습니다."

    query_parts = ["global_oom"]
    if parsed.get("constraint"):
        query_parts.append(str(parsed["constraint"]))
    if parsed.get("swap_total_kb") == 0:
        query_parts.append("no swap space")
    if parsed.get("order", 0) and parsed["order"] > 0:
        query_parts.append(f"order {parsed['order']} page allocation failure")

    query_text = " ".join(query_parts)

    results = collection.query(
        query_texts=[query_text],
        n_results=3,
        where={"error_category": {"$in": ["global_oom", "general"]}},
    )

    assert "ids" in results, "ChromaDB query 결과에 ids가 없습니다."
    assert "documents" in results, "ChromaDB query 결과에 documents가 없습니다."
    assert "metadatas" in results, "ChromaDB query 결과에 metadatas가 없습니다."

    assert results["ids"], "검색 결과 ids가 비어 있습니다."
    assert results["ids"][0], "첫 번째 검색 결과가 비어 있습니다."

    returned_ids = results["ids"][0]
    returned_docs = results["documents"][0]
    returned_metas = results["metadatas"][0]

    assert len(returned_ids) >= 1, "최소 1개 이상의 KB 결과가 검색되어야 합니다."
    assert len(returned_ids) == len(returned_docs) == len(returned_metas), (
        "ids / documents / metadatas 길이가 서로 다릅니다."
    )

    for meta in returned_metas:
        assert meta.get("error_category") in {"global_oom", "general"}, (
            f"허용되지 않은 error_category가 반환되었습니다: {meta}"
        )


def test_live_node_2_classifier_returns_real_json_for_case1_global(live_node2_llm):
    """
    실제 Ollama 기반 Node 2 호출을 통해 JSON 분류 결과를 반환하는지 검증한다.

    검증 내용:
    - Node 1 파싱 결과를 입력으로 실제 Node 2 호출
    - JSON 파싱이 성공하는지
    - classification 스키마가 맞는지
    """
    parsed_state = _build_minimal_state_from_case1()

    state_for_node2 = {
        **parsed_state,
        "classification": {
            "oom_type": None,
            "tools_needed": [],
            "needs_kb": False,
            "confidence": "low",
        },
        "llm": live_node2_llm,
    }

    result_state = node_2_classifier(state_for_node2)

    assert "classification" in result_state, "Node 2 결과에 classification이 없습니다."
    result = result_state["classification"]

    assert isinstance(result, dict), "classification은 dict여야 합니다."
    required_keys = {
        "oom_type",
        "tools_needed",
        "needs_kb",
        "confidence",
    }
    assert required_keys.issubset(result.keys()), (
        f"classification 필수 키가 누락되었습니다: {result}"
    )

    if "raw_llm_oom_type" in result:
        assert result["raw_llm_oom_type"] is None or isinstance(result["raw_llm_oom_type"], str), (
            f"raw_llm_oom_type은 str 또는 None이어야 합니다: {result}"
        )

    if "deterministic_oom_type" in result:
        assert result["deterministic_oom_type"] is None or isinstance(
            result["deterministic_oom_type"], str
        ), f"deterministic_oom_type은 str 또는 None이어야 합니다: {result}"

    if "normalized_llm_oom_type" in result:
        assert result["normalized_llm_oom_type"] is None or isinstance(
            result["normalized_llm_oom_type"], str
        ), f"normalized_llm_oom_type은 str 또는 None이어야 합니다: {result}"

    if "embedding_debug" in result:
        assert isinstance(result["embedding_debug"], dict), (
            f"embedding_debug는 dict여야 합니다: {result}"
        )

    assert result["oom_type"] in {
        "global_oom",
        "cgroup_oom",
        "swap_exhaustion",
        "page_alloc_failure",
        "unknown",
    }, f"예상 범위를 벗어난 oom_type: {result['oom_type']}"

    assert isinstance(result["tools_needed"], list), "tools_needed는 list여야 합니다."
    assert isinstance(result["needs_kb"], bool), "needs_kb는 bool이어야 합니다."
    assert result["confidence"] in {"high", "medium", "low"}, (
        f"예상 범위를 벗어난 confidence: {result['confidence']}"
    )

    assert result["oom_type"] == "global_oom", (
        f"case1_global.txt는 global_oom으로 분류되어야 합니다: {result}"
    )


@pytest.mark.parametrize(
    "filename, oom_type, expected_query_tokens, allowed_categories",
    [
        (
            "case1_global.txt",
            "global_oom",
            ["global_oom", "CONSTRAINT_NONE", "no swap space"],
            {"global_oom", "general"},
        ),
        (
            "case2_cgroup.txt",
            "cgroup_oom",
            ["cgroup_oom", "cgroup memory limit"],
            {"cgroup_oom", "general"},
        ),
    ],
)
def test_live_search_kb_returns_case_specific_results_for_representative_logs(
    filename,
    oom_type,
    expected_query_tokens,
    allowed_categories,
):
    """
    실제 search_kb 경로가 대표 로그별로 쿼리를 다르게 구성하고,
    최소 1개 이상의 KB 결과를 반환하는지 확인한다.
    """
    parsed_state = _build_minimal_state_from_log_file(filename)
    parsed = parsed_state["parsed_fields"]

    collection = get_collection()
    result = search_kb(oom_type=oom_type, parsed_fields=parsed, collection=collection)

    assert result["total_found"] >= 1, f"{filename}에 대해 KB 결과가 비어 있습니다: {result}"
    assert result["chunks"], f"{filename}에 대해 chunks가 비어 있습니다: {result}"

    query_used = result["query_used"]
    for token in expected_query_tokens:
        assert token in query_used, f"query_used에 기대 토큰이 없습니다: token={token}, query={query_used}"

    returned_ids = [chunk["chunk_id"] for chunk in result["chunks"]]
    assert len(returned_ids) == len(set(returned_ids)), (
        f"동일 chunk_id가 중복 반환되었습니다: {returned_ids}"
    )

    for chunk in result["chunks"]:
        metadata = chunk.get("metadata", {})
        assert metadata.get("error_category") in allowed_categories, (
            f"{filename}에 대해 허용되지 않은 error_category가 반환되었습니다: {metadata}"
        )


def test_live_raw_log_queries_do_not_collapse_to_identical_top5_for_distinct_cases():
    """
    exp1 raw 모드와 동일하게 원문 로그를 직접 query_texts로 던졌을 때,
    성격이 다른 두 로그가 완전히 동일한 top-5 결과로 붕괴되지 않는지 확인한다.

    이 테스트는 raw retrieval 품질이 대표 케이스 간 구분 신호를 잃었는지 탐지하기 위한 회귀 장치다.
    """
    collection = get_collection()

    case1_raw_log = _read_log_from_node_1_logs("case1_global.txt")
    case2_raw_log = _read_log_from_node_1_logs("case2_cgroup.txt")

    case1_results = collection.query(query_texts=[case1_raw_log], n_results=5)
    case2_results = collection.query(query_texts=[case2_raw_log], n_results=5)

    assert case1_results.get("ids") and case1_results["ids"][0], (
        f"case1_global.txt raw query 결과가 비어 있습니다: {case1_results}"
    )
    assert case2_results.get("ids") and case2_results["ids"][0], (
        f"case2_cgroup.txt raw query 결과가 비어 있습니다: {case2_results}"
    )

    case1_ids = case1_results["ids"][0]
    case2_ids = case2_results["ids"][0]

    assert case1_ids != case2_ids, (
        "서로 다른 raw 로그가 완전히 동일한 top-5 KB 결과를 반환했습니다. "
        f"case1_ids={case1_ids}, case2_ids={case2_ids}"
    )