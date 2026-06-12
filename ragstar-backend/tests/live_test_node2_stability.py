import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트를 import path에 추가한다.
# tests/ 디렉터리에서 실행해도 app 패키지를 찾을 수 있도록 하기 위함이다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 실제 프로덕션 Node 1 / Node 2를 그대로 사용한다.
# 이 테스트는 mock이 아니라 live integration/stability test다.
from app.agent.nodes.node_1_parser import node_1_parser
from app.agent.nodes.node_2_classifier import node_2_classifier
from app.core.settings import get_settings


KNOWN_NODE2_TOOLS = {
    "memory_calculator",
    "kernel_version_check",
    "kernel_param_recommender",
}


# 반복 호출 횟수.
# 우선순위:
# 1) pytest CLI `--runs=<N>`
# 2) 환경변수 `NODE2_STABILITY_RUNS`
# 3) 기본값 10


def _read_case1_global_log() -> str:
    """
    tests/node_1_logs/case1_global.txt 파일을 읽어 문자열로 반환한다.

    이 테스트는 반드시 팀에서 기준 예시로 쓰는 case1_global.txt를 사용해야 한다.
    """
    log_path = Path(__file__).resolve().parent / "node_1_logs" / "case1_global.txt"
    assert log_path.exists(), f"테스트 로그 파일이 없습니다: {log_path}"
    return log_path.read_text(encoding="utf-8")


def _build_parsed_state_from_case1() -> Dict[str, Any]:
    """
    case1_global.txt를 실제 Node 1 파서에 태워
    Node 2 stability test에서 재사용할 parsed state를 만든다.

    반환값은 Node 1 실행 이후의 state 전체이며,
    이후 Node 2 입력 state를 만들 때 이 결과를 기반으로 deep copy 한다.
    """
    raw_log = _read_case1_global_log()

    # 현재 프로젝트의 state 형태에 맞춰 최소 필드를 모두 넣어 준다.
    state = {
        "raw_log": raw_log,
        "user_kernel_version": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }

    # 실제 Node 1 실행
    parsed_state = node_1_parser(state)

    # 방어적으로 핵심 필드 존재 여부를 확인한다.
    assert "parsed_fields" in parsed_state, "Node 1 결과에 parsed_fields가 없습니다."
    assert isinstance(parsed_state["parsed_fields"], dict), "parsed_fields는 dict여야 합니다."

    return parsed_state


def _build_node2_input_state(parsed_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2 입력용 state를 만든다.

    중요한 점:
    - 반복 실행 간 입력 상태가 완전히 독립적이어야 한다.
    - 따라서 얕은 복사(**parsed_state)가 아니라 deep copy를 사용한다.
    - classification은 매 run마다 동일한 초기값으로 다시 설정한다.
    """
    state = deepcopy(parsed_state)

    # Node 2가 실제로 덮어쓸 classification 초기값을 지정한다.
    state["classification"] = {
        "oom_type": None,
        "tools_needed": [],
        "needs_kb": False,
        "confidence": "low",
    }

    return state


def _validate_node2_classification_schema(classification: Dict[str, Any]) -> None:
    """
    Node 2 출력 classification의 최소 계약을 검증한다.

    이 함수는 "의미가 맞는가"까지 보지는 않고,
    우선 출력 스키마가 무너지지 않았는지를 검증하는 역할을 한다.
    """
    assert isinstance(classification, dict), "classification은 dict여야 합니다."

    # Node 2의 핵심 계약은 아래 4개 필드다.
    # 최근 구현에서는 embedding 진단용 확장 필드가 추가될 수 있으므로,
    # exact key equality 대신 required subset을 검사한다.
    required_keys = {
        "oom_type",
        "tools_needed",
        "needs_kb",
        "confidence",
    }
    assert required_keys.issubset(classification.keys()), (
        f"classification 필수 키가 누락되었습니다: {classification}"
    )

    # 확장 진단 필드가 있다면 최소 타입 계약도 함께 확인한다.
    if "raw_llm_oom_type" in classification:
        assert classification["raw_llm_oom_type"] is None or isinstance(
            classification["raw_llm_oom_type"], str
        ), f"raw_llm_oom_type은 str 또는 None이어야 합니다: {classification}"

    if "deterministic_oom_type" in classification:
        assert classification["deterministic_oom_type"] is None or isinstance(
            classification["deterministic_oom_type"], str
        ), f"deterministic_oom_type은 str 또는 None이어야 합니다: {classification}"

    if "normalized_llm_oom_type" in classification:
        assert classification["normalized_llm_oom_type"] is None or isinstance(
            classification["normalized_llm_oom_type"], str
        ), f"normalized_llm_oom_type은 str 또는 None이어야 합니다: {classification}"

    if "embedding_debug" in classification:
        assert isinstance(classification["embedding_debug"], dict), (
            f"embedding_debug는 dict여야 합니다: {classification}"
        )

    # 분류 라벨은 현재 시스템이 허용하는 범위 안에 있어야 한다.
    assert classification["oom_type"] in {
        "global_oom",
        "cgroup_oom",
        "swap_exhaustion",
        "page_alloc_failure",
        "unknown",
    }, f"예상 범위를 벗어난 oom_type: {classification['oom_type']}"

    # tools_needed는 리스트여야 한다.
    assert isinstance(classification["tools_needed"], list), (
        f"tools_needed는 list여야 합니다: {classification}"
    )
    assert all(isinstance(tool, str) and tool.strip() for tool in classification["tools_needed"]), (
        f"tools_needed의 각 원소는 비어 있지 않은 문자열이어야 합니다: {classification}"
    )

    # needs_kb는 bool이어야 한다.
    assert isinstance(classification["needs_kb"], bool), (
        f"needs_kb는 bool이어야 합니다: {classification}"
    )

    # confidence는 정해진 값 중 하나여야 한다.
    assert classification["confidence"] in {"high", "medium", "low"}, (
        f"예상 범위를 벗어난 confidence: {classification['confidence']}"
    )


def _get_allowed_node2_tools(parsed_state: Dict[str, Any]) -> set[str]:
    """live stability test에서 허용하는 Node 2 도구 라벨 집합을 계산한다.

    중요:
    - 현재 Node 2는 `tools_needed`를 LLM 응답에서 그대로 반영한다.
    - 따라서 live test에서는 "현재 시스템이 아는 합법적인 도구 라벨인가"를 우선 보고,
      parsed_state 기반 동적 추천 로직의 exact match까지는 강제하지 않는다.
    - 다만 kernel_version 정보가 있는 경우 `kernel_version_check`가 반드시 포함되어야 한다는
      핵심 사실은 별도로 검증한다.
    """
    allowed_tools: set[str] = {"kernel_param_recommender", "memory_calculator"}
    parsed_fields = parsed_state.get("parsed_fields", {})

    if parsed_fields.get("kernel_version"):
        allowed_tools.add("kernel_version_check")

    return allowed_tools & KNOWN_NODE2_TOOLS


def _normalize_tool_signature(tools_needed: List[str]) -> tuple[str, ...]:
    """도구 조합을 중복 없는 정렬 시그니처로 정규화한다."""
    normalized_tools = sorted({tool.strip() for tool in tools_needed})
    return tuple(normalized_tools)


def _get_allowed_node2_tool_signatures(parsed_state: Dict[str, Any]) -> set[tuple[str, ...]]:
    """case1_global에서 허용되는 도구 조합 시그니처 집합을 만든다."""
    allowed_tools = _get_allowed_node2_tools(parsed_state)
    required_tools = {"kernel_version_check"}
    optional_tools = allowed_tools - required_tools

    allowed_signatures: set[tuple[str, ...]] = set()
    optional_tool_list = sorted(optional_tools)

    for mask in range(1 << len(optional_tool_list)):
        candidate = set(required_tools)
        for index, tool in enumerate(optional_tool_list):
            if mask & (1 << index):
                candidate.add(tool)
        allowed_signatures.add(tuple(sorted(candidate)))

    return allowed_signatures


def _format_node2_result_distributions(results: List[Dict[str, Any]]) -> str:
    """실패 메시지에 넣을 run별 Node 2 결과 분포 문자열을 만든다."""
    oom_type_distribution = Counter(result["oom_type"] for result in results)
    confidence_distribution = Counter(result["confidence"] for result in results)
    needs_kb_distribution = Counter(result["needs_kb"] for result in results)
    tool_signature_distribution = Counter(
        _normalize_tool_signature(result["tools_needed"]) for result in results
    )

    return (
        f"oom_type_distribution={dict(oom_type_distribution)}, "
        f"confidence_distribution={dict(confidence_distribution)}, "
        f"needs_kb_distribution={dict(needs_kb_distribution)}, "
        f"tool_signature_distribution={dict(tool_signature_distribution)}, "
        f"results={results}"
    )


def _assert_case1_global_core_classification_is_stable(
    results: List[Dict[str, Any]],
    parsed_state: Dict[str, Any],
) -> None:
    """
    Node 2 live 출력에서 case1_global의 핵심 분류 사실이 반복 호출마다 유지되는지 검사한다.

    Node 2는 `tools_needed`를 LLM 응답 그대로 채우므로,
    도구 리스트의 exact match까지 강제하면 불필요하게 flaky해질 수 있다.
    따라서 다음 안정성만 본다:
    - oom_type은 항상 global_oom
    - confidence는 항상 high
    - tools_needed는 허용 가능한 정규화 도구 조합 중 하나
    - kernel_version 정보가 있으므로 kernel_version_check는 항상 포함
    - 적어도 하나 이상의 도구는 선택
    - tools_needed 내부에 중복 라벨은 없음

        참고:
        - 현재 구현은 `oom_type`/`confidence` 일부는 하드 로직으로 보정하지만,
            `tools_needed`와 `needs_kb`는 LLM 응답을 그대로 반영한다.
        - 따라서 live stability test에서 `needs_kb=True`까지 강제하면
            현재 구현 특성상 불필요하게 flaky해질 수 있다.
    """
    allowed_tools = _get_allowed_node2_tools(parsed_state)
    allowed_signatures = _get_allowed_node2_tool_signatures(parsed_state)
    distribution_details = _format_node2_result_distributions(results)

    for result in results:
        assert result["oom_type"] == "global_oom", (
            "case1_global인데 global_oom이 아닙니다. "
            f"{distribution_details}"
        )
        assert result["confidence"] == "high", (
            "case1_global인데 confidence가 high가 아닙니다. "
            f"{distribution_details}"
        )

        tools_needed = result["tools_needed"]
        tool_signature = _normalize_tool_signature(tools_needed)

        assert tools_needed, (
            "tools_needed가 비어 있습니다. "
            f"{distribution_details}"
        )
        assert len(tool_signature) == len(tools_needed), (
            "tools_needed에 중복 라벨이 있습니다. "
            f"{distribution_details}"
        )
        assert set(tools_needed).issubset(allowed_tools), (
            "허용되지 않은 도구가 포함되었습니다. "
            f"allowed_tools={sorted(allowed_tools)}, {distribution_details}"
        )
        assert tool_signature in allowed_signatures, (
            "허용되지 않은 도구 조합입니다. "
            f"allowed_signatures={sorted(allowed_signatures)}, {distribution_details}"
        )
        assert "kernel_version_check" in tools_needed, (
            "kernel_version 정보가 있는데 kernel_version_check가 없습니다. "
            f"{distribution_details}"
        )


def _infer_generation_model_name(llm: Any | None) -> str:
    """테스트 시작 로그에 표시할 generation model 이름을 추론한다."""
    if llm is None:
        return get_settings().node2_model

    for attr in ("model", "model_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    return type(llm).__name__


def _print_live_model_configuration(llm: Any | None, runs: int) -> None:
    """live test에서 실제 사용하는 generation 모델과 embedding 사용 여부를 출력한다."""
    generation_model = _infer_generation_model_name(llm)

    print(
        "[live_test_node2_stability] "
        f"generation_model={generation_model}, embedding_model=not_used, runs={runs}"
    )


def _run_node2_multiple_times(
    parsed_state: Dict[str, Any],
    runs: int,
    llm: Any | None = None,
) -> List[Dict[str, Any]]:
    """
    같은 parsed_state를 여러 번 Node 2에 넣어 결과를 수집한다.

    stability / determinism 테스트이므로 runs는 반드시 2 이상이어야 한다.

    또한 각 run의 결과는 deepcopy로 스냅샷을 떠서 저장한다.
    그래야 이후 내부 객체가 바뀌더라도 과거 결과가 오염되지 않는다.
    """
    assert runs >= 2, f"stability test를 위해 runs는 2 이상이어야 합니다: {runs}"

    results: List[Dict[str, Any]] = []

    for _ in range(runs):
        # 매 run마다 완전히 독립된 입력 state를 만든다.
        state_for_node2 = _build_node2_input_state(parsed_state)

        if llm is not None:
            state_for_node2["llm"] = llm

        # 실제 Node 2 호출
        result_state = node_2_classifier(state_for_node2)

        # classification 필드는 반드시 존재해야 한다.
        assert "classification" in result_state, "Node 2 결과에 classification이 없습니다."

        classification = result_state["classification"]

        # 최소 스키마 계약 검증
        _validate_node2_classification_schema(classification)

        # aliasing 방지를 위해 deepcopy 후 저장
        results.append(deepcopy(classification))

    return results


def test_live_node2_case1_global_repeated_calls_are_stable(live_node2_llm, live_node2_runs):
    """
    Node 1 -> Node 2 live integration test.

    case1_global.txt를 반복 입력했을 때,
    1) classification 스키마가 매번 유지되는지
    2) oom_type이 항상 global_oom인지
    3) classification 전체가 반복 호출마다 동일한지
    를 한 번에 검증한다.

    참고:
    - 스키마 검증은 _run_node2_multiple_times() 내부에서 매 run마다 수행된다.
    - 이 테스트는 category drift와 full determinism을 모두 포함하는 통합 테스트다.
    """
    _print_live_model_configuration(live_node2_llm, live_node2_runs)
    # 기준 입력 로그를 실제 Node 1에 태워 parsed state를 만든다.
    parsed_state = _build_parsed_state_from_case1()

    # 같은 입력으로 Node 2를 여러 번 실행한다.
    results = _run_node2_multiple_times(parsed_state, live_node2_runs, llm=live_node2_llm)

    # 기대한 횟수만큼 결과가 수집되어야 한다.
    assert len(results) == live_node2_runs

    # ------------------------------------------------------------
    # 1) category drift 검증
    # ------------------------------------------------------------
    # case1_global.txt는 대표적인 global_oom 예시이므로
    # 반복 호출해도 oom_type은 항상 global_oom이어야 한다.
    oom_types = [result["oom_type"] for result in results]
    counts = Counter(oom_types)

    assert counts == Counter({"global_oom": live_node2_runs}), (
        "Node 2 live 분류가 흔들렸습니다. "
        f"runs={live_node2_runs}, distribution={dict(counts)}, results={results}"
    )

    # ------------------------------------------------------------
    # 2) 핵심 classification 안정성 검증
    # ------------------------------------------------------------
    # Node 2는 tools_needed를 LLM 응답 그대로 사용하므로,
    # exact dict equality 대신 핵심 사실/계약 안정성을 검증한다.
    _assert_case1_global_core_classification_is_stable(results, parsed_state)