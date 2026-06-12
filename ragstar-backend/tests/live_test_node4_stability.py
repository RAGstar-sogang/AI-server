# tests/live_test_node4_stability.py

from functools import lru_cache
from math import sqrt
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트를 import path에 추가한다.
# tests/ 디렉터리에서 실행해도 app 패키지를 찾을 수 있게 하기 위함이다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 실제 프로덕션 Node 1 / Node 4를 그대로 사용한다.
# 이 테스트는 mock 기반이 아니라 live stability test다.
from app.agent.nodes.node_1_parser import node_1_parser
import app.agent.nodes.node_4_synthesizer as node4_module
from app.core.llm_factory import build_node4_live_stability_embeddings
from app.core.settings import get_settings


# Node 4는 출력이 길고 LLM 호출 비용이 있으므로
# 기본 반복 횟수는 Node 2보다 보수적으로 5회로 둔다.
# 우선순위:
# 1) pytest CLI `--runs=<N>`
# 2) 환경변수 `NODE4_STABILITY_RUNS`
# 3) 기본값 5


def _read_case1_global_log() -> str:
    """
    tests/node_1_logs/case1_global.txt 파일을 읽어 문자열로 반환한다.

    이 테스트는 반드시 팀 기준 예시인 case1_global.txt를 사용한다.
    """
    log_path = Path(__file__).resolve().parent / "node_1_logs" / "case1_global.txt"
    assert log_path.exists(), f"테스트 로그 파일이 없습니다: {log_path}"
    return log_path.read_text(encoding="utf-8")


def _build_parsed_state_from_case1() -> Dict[str, Any]:
    """
    case1_global.txt를 실제 Node 1 파서에 태워
    Node 4 stability test에서 재사용할 parsed state를 만든다.

    반환값은 Node 1 실행 이후의 state 전체다.
    이후 Node 4 입력 state를 만들 때 이 결과를 deep copy해서 사용한다.
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

    # 방어적으로 핵심 필드 존재 여부를 확인한다.
    assert "parsed_fields" in parsed_state, "Node 1 결과에 parsed_fields가 없습니다."
    assert isinstance(parsed_state["parsed_fields"], dict), "parsed_fields는 dict여야 합니다."

    return parsed_state


def _build_node4_input_state(parsed_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 4 입력용 state를 만든다.

    중요한 점:
    - Node 4 stability test의 대상은 Node 4 자체의 live 출력 안정성이다.
    - 따라서 Node 2 / Node 3의 변동성이 끼지 않도록
      classification / tool_results는 고정된 입력으로 주입한다.
    - 단, parsed_fields는 실제 case1_global.txt를 Node 1로 파싱한 값을 사용한다.
    """
    state = deepcopy(parsed_state)
    parsed = state["parsed_fields"]

    # case1_global은 대표적인 global_oom 예시이므로
    # Node 4 입력용 classification도 그 기준에 맞게 고정한다.
    state["classification"] = {
        "oom_type": "global_oom",
        "tools_needed": ["memory_calculator"],
        "needs_kb": True,
        "confidence": "high",
    }

    # Node 4는 Node 3 결과를 받아 종합 진단을 만드는 노드이므로
    # tool_results도 현실적인 최소 구조로 고정 입력한다.
    #
    # 여기서는 Node 4를 안정적으로 검증하기 위해
    # memory 분석 결과와 KB 검색 결과만 넣는다.
    total_ram_mb = round((parsed.get("total_ram_pages", 0) * 4) / 1024, 1)
    anon_rss_kb = parsed.get("anon_rss_kb", 0)
    anon_rss_mb = round(anon_rss_kb / 1024, 1)
    total_ram_kb = parsed.get("total_ram_pages", 0) * 4
    ram_ratio = round((anon_rss_kb / total_ram_kb), 3) if total_ram_kb else 0.0

    state["tool_results"] = {
        "memory": {
            "top5_processes": [
                {
                    "name": parsed.get("killed_process", "unknown"),
                    "pid": parsed.get("killed_pid", -1),
                    "rss_kb": anon_rss_kb,
                    "rss_mb": anon_rss_mb,
                    "ram_ratio": ram_ratio,
                    "oom_score_adj": parsed.get("oom_score_adj", 0),
                    "oom_protected": parsed.get("oom_score_adj", 0) == -1000,
                }
            ],
            "top5_total_ratio": ram_ratio,
            "total_ram_mb": total_ram_mb,
            "swap_configured": parsed.get("swap_total_kb", 0) > 0,
            "free_below_min": (
                parsed.get("node_free_kb") is not None
                and parsed.get("node_min_kb") is not None
                and parsed.get("node_free_kb") < parsed.get("node_min_kb")
            ),
            "protected_procs": [],
        },
        "kb_chunks": {
            "query_used": "global_oom CONSTRAINT_NONE no swap space",
            "chunks": [
                {
                    "chunk_id": "chunk_case1_global_001",
                    "content": (
                        "Swap 미설정 환경에서는 순간적인 메모리 압박을 흡수하지 못해 "
                        "CONSTRAINT_NONE 기반 global OOM이 쉽게 발생할 수 있다. "
                        "특히 특정 프로세스가 RAM을 크게 점유하면 시스템 전체 OOM으로 이어질 수 있다."
                    ),
                    "metadata": {
                        "title": "Global OOM / No Swap Guide",
                        "error_category": "global_oom",
                        "source": "test-fixture",
                    },
                    "score": 0.01,
                }
            ],
            "total_found": 1,
        },
    }

    # Node 4 wrapper state 형식을 맞추기 위해 diagnosis / error도 초기화한다.
    state["diagnosis"] = {}
    state["error"] = None

    return state


def _flatten_strings(value: Any) -> List[str]:
    """
    dict / list / scalar 내부의 문자열을 재귀적으로 평탄화한다.

    Node 4 출력의 핵심 사실 보존 여부를 넓게 검사할 때 사용한다.
    """
    flattened: List[str] = []

    if isinstance(value, dict):
        for item in value.values():
            flattened.extend(_flatten_strings(item))
    elif isinstance(value, list):
        for item in value:
            flattened.extend(_flatten_strings(item))
    elif value is None:
        pass
    else:
        flattened.append(str(value))

    return flattened


def _contains_any(text: str, candidates: List[str]) -> bool:
    """
    주어진 텍스트에 후보 문자열 중 하나라도 포함되는지 검사한다.
    비교는 소문자 기준으로 수행한다.
    """
    normalized = text.lower()
    return any(candidate.lower() in normalized for candidate in candidates)


def _normalize_semantic_text(text: Any) -> str:
    """
    자유 서술 문자열을 의미 비교용으로 완만하게 정규화한다.

    - 대소문자를 무시한다.
    - `_` / `-` 차이를 무시한다.
    - 연속 공백을 하나로 정리한다.
    """
    return " ".join(str(text).lower().replace("_", " ").replace("-", " ").split())


@lru_cache(maxsize=1)
def _get_constraint_embeddings():
    """constraint_type 의미 판정에 사용할 Ollama embedding 클라이언트를 캐시한다."""
    return build_node4_live_stability_embeddings()


@lru_cache(maxsize=None)
def _embed_constraint_text(text: str) -> tuple[float, ...]:
    """같은 텍스트를 반복 임베딩하지 않도록 결과를 캐시한다."""
    vector = _get_constraint_embeddings().embed_query(text)
    return tuple(float(value) for value in vector)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """외부 의존성 없이 두 벡터의 cosine similarity를 계산한다."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(lhs * rhs for lhs, rhs in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _is_system_wide_constraint(text: Any) -> bool:
    scores = _get_system_wide_constraint_scores(text)
    return scores["is_match"]


def _get_system_wide_constraint_scores(text: Any) -> Dict[str, Any]:
    """
    constraint_type이 case1_global에 맞는 system-wide / global 계열 의미인지
    Ollama embedding 유사도 기반으로 판정한다.

    문자열 literal이 조금 달라도,
    system-wide/global anchor에 더 가깝고 cgroup/local anchor보다 충분히 유사하면 통과시킨다.
    """
    normalized = _normalize_semantic_text(text)

    if not normalized:
        return {
            "normalized": normalized,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "margin": 0.0,
            "is_match": False,
        }

    positive_anchors = (
        "CONSTRAINT_NONE",
        "global OOM",
        "global memory pressure",
        "system wide memory constraint",
        "NONE system wide",
    )
    negative_anchors = (
        "cgroup memory limit",
        "memcg constraint",
        "container scoped memory limit",
        "swap exhaustion",
        "page allocation failure",
    )

    target_vector = _embed_constraint_text(normalized)
    positive_score = max(
        _cosine_similarity(target_vector, _embed_constraint_text(_normalize_semantic_text(anchor)))
        for anchor in positive_anchors
    )
    negative_score = max(
        _cosine_similarity(target_vector, _embed_constraint_text(_normalize_semantic_text(anchor)))
        for anchor in negative_anchors
    )

    margin = positive_score - negative_score

    return {
        "normalized": normalized,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "margin": margin,
        "is_match": positive_score >= 0.60 and margin >= 0.05,
    }


def _validate_node4_bundle_schema(bundle: Dict[str, Any]) -> None:
    """
    Node 4의 live 출력 bundle이 현재 프로덕션 스키마를 만족하는지 검증한다.

    bundle 구조:
    {
      "reasoning_trace": {...},
      "final_answer": {...}
    }
    """
    assert isinstance(bundle, dict), "Node 4 bundle은 dict여야 합니다."

    assert node4_module._validate_llm_bundle_schema(bundle), (
        "Node 4 bundle 스키마가 유효하지 않습니다. "
        f"bundle={bundle}"
    )

    reasoning_trace = bundle["reasoning_trace"]
    final_answer = bundle["final_answer"]

    # reasoning_trace 구조가 현재 계약을 따르는지 추가로 확인한다.
    assert node4_module._validate_reasoning_trace_schema(reasoning_trace), (
        f"reasoning_trace 스키마가 유효하지 않습니다: {reasoning_trace}"
    )

    # final_answer 구조가 현재 계약을 따르는지 추가로 확인한다.
    assert node4_module._validate_final_answer_schema(final_answer), (
        f"final_answer 스키마가 유효하지 않습니다: {final_answer}"
    )


def _assert_case1_global_core_facts_preserved(final_answer: Dict[str, Any]) -> None:
    """
    case1_global.txt의 핵심 사실이 Node 4 출력에서 왜곡되지 않았는지 검사한다.

    Node 4는 자유도가 높은 생성 노드이므로,
    Node 2처럼 완전 동일성을 요구하지 않고 다음을 본다:
    - 필수 key_metrics가 case1_global의 핵심 사실과 맞는가
    - root_cause / evidence / summary 등에서 핵심 신호가 유지되는가
    - 명백히 비어 있는 출력이 아닌가
    """
    log_analysis = final_answer["log_analysis"]
    diagnosis = final_answer["diagnosis"]
    action_guide = final_answer["action_guide"]
    key_metrics = log_analysis["key_metrics"]

    # ------------------------------------------------------------
    # 1) key_metrics 수준의 핵심 사실 확인
    # ------------------------------------------------------------

    # case1_global에서 killed_process는 java여야 한다.
    killed_process_text = str(key_metrics.get("killed_process", ""))
    assert "java" in killed_process_text.lower(), (
        f"killed_process가 case1_global 사실과 다릅니다: {key_metrics}"
    )

    # constraint_type은 literal 하나에 고정되지 않고,
    # system-wide / global OOM 계열 의미를 드러내면 된다.
    constraint_text = str(key_metrics.get("constraint_type", ""))
    constraint_scores = _get_system_wide_constraint_scores(constraint_text)
    assert constraint_scores["is_match"], (
        "constraint_type이 case1_global 사실과 다릅니다. "
        f"raw={constraint_text!r}, normalized={constraint_scores['normalized']!r}, "
        f"positive_score={constraint_scores['positive_score']:.4f}, "
        f"negative_score={constraint_scores['negative_score']:.4f}, "
        f"margin={constraint_scores['margin']:.4f}, key_metrics={key_metrics}"
    )

    # swap_status는 swap 미설정 / 0 상태를 반영해야 한다.
    swap_status_text = str(key_metrics.get("swap_status", ""))
    assert _contains_any(
        swap_status_text,
        ["0", "미설정", "no swap", "not configured", "없음", "none"],
    ), f"swap_status가 case1_global 사실과 다릅니다: {key_metrics}"

    # ------------------------------------------------------------
    # 2) 주요 서술 필드가 비어 있지 않은지 확인
    # ------------------------------------------------------------
    assert str(log_analysis.get("summary", "")).strip(), (
        f"log_analysis.summary가 비어 있습니다: {final_answer}"
    )
    assert str(diagnosis.get("root_cause", "")).strip(), (
        f"diagnosis.root_cause가 비어 있습니다: {final_answer}"
    )
    assert isinstance(diagnosis.get("evidence"), list) and diagnosis["evidence"], (
        f"diagnosis.evidence가 비어 있습니다: {final_answer}"
    )
    assert isinstance(action_guide.get("immediate"), list) and action_guide["immediate"], (
        f"action_guide.immediate가 비어 있습니다: {final_answer}"
    )
    assert isinstance(action_guide.get("recommended"), list) and action_guide["recommended"], (
        f"action_guide.recommended가 비어 있습니다: {final_answer}"
    )

    # ------------------------------------------------------------
    # 3) 전체 텍스트 수준에서 핵심 사실 신호가 유지되는지 확인
    # ------------------------------------------------------------
    flattened_text = " ".join(_flatten_strings(final_answer))

    # 프로세스 신호
    assert _contains_any(flattened_text, ["java"]), (
        "최종 진단 전체 텍스트에서 killed process(java) 신호가 사라졌습니다. "
        f"final_answer={final_answer}"
    )

    # global OOM / 시스템 전체 메모리 부족 계열 신호
    assert _contains_any(
        flattened_text,
        ["global", "global_oom", "시스템 전체", "전체 메모리", "CONSTRAINT_NONE"],
    ), (
        "최종 진단 전체 텍스트에서 global OOM 신호가 부족합니다. "
        f"final_answer={final_answer}"
    )

    # swap 미설정 / no swap 계열 신호
    assert _contains_any(
        flattened_text,
        ["swap", "미설정", "no swap", "not configured", "SwapTotal=0", "0KB", "0 kB"],
    ), (
        "최종 진단 전체 텍스트에서 swap 미설정 신호가 부족합니다. "
        f"final_answer={final_answer}"
    )


def _assert_live_llm_path_was_used(bundle: Dict[str, Any]) -> None:
    """
    이 테스트는 live LLM 안정성을 보는 것이므로,
    fallback bundle이 조용히 통과하면 안 된다.

    다만 Node 4의 공개 계약은 reasoning_trace/final_answer 스키마이지,
    decision_basis 내부 문자열 포맷 자체는 아니다.
    따라서 fallback 여부는 diagnosis.llm_failed 플래그 기준으로 본다.
    """
    final_answer = bundle["final_answer"]

    assert final_answer.get("diagnosis", {}).get("llm_failed") is not True, (
        "final_answer.diagnosis.llm_failed=True 이므로 실제 live LLM 성공이 아닙니다. "
        f"final_answer={final_answer}"
    )


def _infer_generation_model_name(llm: Any | None) -> str:
    """테스트 시작 로그에 표시할 generation model 이름을 추론한다."""
    if llm is None:
        return get_settings().node4_model

    for attr in ("model", "model_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    return type(llm).__name__


def _print_live_model_configuration(llm: Any | None, runs: int) -> None:
    """live test에서 실제 사용하는 generation / embedding 모델 조합을 출력한다."""
    settings = get_settings()
    generation_model = _infer_generation_model_name(llm)
    embedding_model = settings.ollama_embedding_model

    print(
        "[live_test_node4_stability] "
        f"generation_model={generation_model}, embedding_model={embedding_model}, runs={runs}"
    )


def _run_node4_multiple_times(
    parsed_state: Dict[str, Any],
    runs: int,
    llm: Any | None = None,
) -> List[Dict[str, Any]]:
    """
    같은 parsed_state를 기반으로 Node 4를 여러 번 실행해 bundle 결과를 수집한다.

    Node 4 stability test의 목표는 다음과 같다:
    - 매 run마다 JSON envelope 스키마가 유지되는가
    - 매 run마다 실제 live LLM 성공 경로가 사용되는가
    - 매 run마다 case1_global의 핵심 사실이 보존되는가

    참고:
    - Node 4는 자유도가 높은 생성 노드이므로
      Node 2처럼 완전 동일한 문자열 출력을 기대하지 않는다.
    - 대신 schema drift / fallback drift / fact drift를 막는 데 초점을 둔다.
    """
    assert runs >= 2, f"stability test를 위해 runs는 2 이상이어야 합니다: {runs}"

    bundles: List[Dict[str, Any]] = []

    for _ in range(runs):
        state_for_node4 = _build_node4_input_state(parsed_state)

        # Node 4 core live 호출:
        # generate_diagnosis_bundle_with_retry()는 실제 Ollama를 호출해
        # reasoning_trace + final_answer bundle을 반환한다.
        bundle = node4_module.generate_diagnosis_bundle_with_retry(
            parsed_fields=state_for_node4["parsed_fields"],
            classification=state_for_node4["classification"],
            tool_results=state_for_node4["tool_results"],
            llm=llm,
            max_retries=0,     # 재시도 없이 1회만 시도한다. 실패 시 현재 구현은 fallback bundle을 반환할 수 있다.
        )

        # 매 run마다 bundle 스키마를 검증한다.
        _validate_node4_bundle_schema(bundle)

        # live stability test이므로 fallback이 아니라
        # 실제 LLM 성공 경로가 사용되었는지 확인한다.
        _assert_live_llm_path_was_used(bundle)

        # 매 run마다 case1_global 핵심 사실이 유지되는지 검증한다.
        _assert_case1_global_core_facts_preserved(bundle["final_answer"])

        # aliasing 방지를 위해 deepcopy 후 저장한다.
        bundles.append(deepcopy(bundle))

    return bundles


def test_live_node4_case1_global_repeated_calls_preserve_schema_and_core_facts(live_node4_llm, live_node4_runs):
    """
    Node 1(real parse) + Node 4(core live generation) stability test.

    case1_global.txt를 실제 Node 1로 파싱한 뒤,
    Node 2 / Node 3 변동성은 고정 입력으로 차단한 상태에서
    Node 4가 반복 호출마다
    - 유효한 JSON envelope(reasoning_trace + final_answer)를 반환하는지
    - fallback이 아니라 실제 LLM 성공 경로를 유지하는지
    - case1_global의 핵심 사실(java / CONSTRAINT_NONE / no swap)을 왜곡하지 않는지
    를 검증한다.
    """
    _print_live_model_configuration(live_node4_llm, live_node4_runs)
    parsed_state = _build_parsed_state_from_case1()
    bundles = _run_node4_multiple_times(parsed_state, live_node4_runs, llm=live_node4_llm)

    # 기대한 횟수만큼 결과가 수집되어야 한다.
    assert len(bundles) == live_node4_runs