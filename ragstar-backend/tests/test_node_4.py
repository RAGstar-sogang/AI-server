import os
import sys
import copy
import json
import pytest

# -----------------------------------------------------------------------------
# 프로젝트 루트 import 경로 설정
# -----------------------------------------------------------------------------
# tests/ 디렉터리에서 실행해도 app/... 모듈을 import할 수 있도록
# 프로젝트 루트를 sys.path에 추가한다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# -----------------------------------------------------------------------------
# 테스트 대상 모듈
# -----------------------------------------------------------------------------
import app.agent.nodes.node_4_synthesizer as node4_module


# -----------------------------------------------------------------------------
# 디버그 출력 헬퍼
# -----------------------------------------------------------------------------
# pytest는 기본적으로 stdout을 캡처하므로,
# 실제 input/output 또는 예외 메시지를 눈으로 확인하려면
# -s 옵션과 함께 실행한다.
#
# 예:
#   pytest -s tests/test_node_4.py
# -----------------------------------------------------------------------------
def _pretty_json(data) -> str:
    """
    JSON 직렬화 가능한 객체를 사람이 읽기 좋은 문자열로 변환한다.
    """
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except TypeError:
        return str(data)



def dump_case(title: str, state: dict, output=None):
    """
    Node 4 테스트에서 공통으로 쓰는 입출력 출력 함수.
    """
    print(f"\n{'=' * 80}")
    print(f"[CASE] {title}")
    print(f"{'=' * 80}")

    print("\n[INPUT] parsed_fields")
    print(_pretty_json(state.get("parsed_fields", {})))

    print("\n[INPUT] classification")
    print(_pretty_json(state.get("classification", {})))

    print("\n[INPUT] tool_results")
    print(_pretty_json(state.get("tool_results", {})))

    print("\n[OUTPUT]")
    print(_pretty_json(output))



def dump_simple_io(title: str, input_data, output_data):
    """
    extract_json 같은 단일 함수 테스트에서 쓰는 간단 출력 함수.
    """
    print(f"\n{'=' * 80}")
    print(f"[CASE] {title}")
    print(f"{'=' * 80}")

    print("\n[INPUT]")
    if isinstance(input_data, str):
        print(input_data)
    else:
        print(_pretty_json(input_data))

    print("\n[OUTPUT]")
    print(_pretty_json(output_data))



def dump_exception_case(title: str, state: dict, exc: Exception):
    """
    예외 발생이 기대되는 테스트에서 쓰는 출력 함수.
    """
    print(f"\n{'=' * 80}")
    print(f"[CASE] {title}")
    print(f"{'=' * 80}")

    print("\n[INPUT] parsed_fields")
    print(_pretty_json(state.get("parsed_fields", {})))

    print("\n[INPUT] classification")
    print(_pretty_json(state.get("classification", {})))

    print("\n[INPUT] tool_results")
    print(_pretty_json(state.get("tool_results", {})))

    print("\n[EXCEPTION TYPE]")
    print(type(exc).__name__)

    print("\n[EXCEPTION MESSAGE]")
    print(str(exc))



def dump_wrapper_case(title: str, before_state: dict, after_state: dict):
    """
    wrapper 테스트에서 state 전후를 비교해 보기 위한 출력 함수.
    """
    print(f"\n{'=' * 80}")
    print(f"[CASE] {title}")
    print(f"{'=' * 80}")

    print("\n[INPUT STATE]")
    print(_pretty_json(before_state))

    print("\n[OUTPUT STATE]")
    print(_pretty_json(after_state))


# -----------------------------------------------------------------------------
# 테스트 데이터 생성 헬퍼
# -----------------------------------------------------------------------------
def create_base_state() -> dict:
    """
    현재 Node 1~4 구현과 팀 state contract를 함께 반영한 기본 입력 state를 생성한다.

    포인트:
    - parsed_fields는 현재 node_1_parser 출력 키 이름을 따른다.
    - classification은 현재 node_2_classifier 출력 키 이름을 따른다.
    - tool_results는 현재 node_3_executor 출력 키 이름을 따른다.
    - state에는 diagnosis만 있고 diagnosis_trace는 두지 않는다.
    """
    return {
        "raw_log": "Mock OOM log data...",
        "user_kernel_version": None,
        "parsed_fields": {
            "trigger_process": "httpd",
            "killed_process": "java",
            "killed_pid": 3201,
            "total_vm_kb": 273728,
            "anon_rss_kb": 875680,
            "oom_score_adj": 0,
            "total_ram_pages": 524288,
            "node_free_kb": 7296,
            "node_min_kb": 7360,
            "swap_total_kb": 0,
            "swap_free_kb": 0,
            "constraint": "CONSTRAINT_NONE",
            "gfp_mask": "0x280da",
            "order": 0,
            "cgroup_path": None,
            "cgroup_usage_kb": None,
            "cgroup_limit_kb": None,
            "cgroup_failcnt": None,
            "process_table": [
                {"pid": 3201, "name": "java", "rss_kb": 875680, "oom_score_adj": 0},
                {"pid": 3244, "name": "httpd", "rss_kb": 209360, "oom_score_adj": 0},
                {"pid": 1002, "name": "sshd", "rss_kb": 51200, "oom_score_adj": -1000},
            ],
            "kernel_version": "4.18.0-305.el8.x86_64",
        },
        "classification": {
            "oom_type": "global_oom",
            "tools_needed": ["memory_calculator", "kernel_version_check", "kernel_param_recommender"],
            "needs_kb": True,
            "confidence": "high",
        },
        "tool_results": {
            "memory": {
                "top5_processes": [
                    {
                        "name": "java",
                        "pid": 3201,
                        "rss_kb": 875680,
                        "rss_mb": 855.2,
                        "ram_ratio": 0.418,
                        "oom_score_adj": 0,
                        "oom_protected": False,
                    },
                    {
                        "name": "httpd",
                        "pid": 3244,
                        "rss_kb": 209360,
                        "rss_mb": 204.5,
                        "ram_ratio": 0.100,
                        "oom_score_adj": 0,
                        "oom_protected": False,
                    },
                    {
                        "name": "sshd",
                        "pid": 1002,
                        "rss_kb": 51200,
                        "rss_mb": 50.0,
                        "ram_ratio": 0.024,
                        "oom_score_adj": -1000,
                        "oom_protected": True,
                    },
                ],
                "top5_total_ratio": 0.542,
                "total_ram_mb": 2048.0,
                "swap_configured": False,
                "free_below_min": True,
                "protected_procs": ["sshd"],
                "sentinel_memory_note": "MEMORY-SIGNAL-JAVA-42PCT",
            },
            "kernel_bugs": {
                "kernel_version": "4.18.0-305.el8.x86_64",
                "known_issues": [
                    "BUG-TEST-418: early OOM may occur due to delayed reclaim"
                ],
                "has_known_issues": True,
            },
            "kernel_params": {
                "oom_type": "global_oom",
                "recommendations": [
                    {
                        "name": "vm.overcommit_memory",
                        "recommendation": "2로 설정 검토",
                        "command": "sysctl -w vm.overcommit_memory=2",
                    }
                ],
            },
            "kb_chunks": {
                "query_used": "global_oom smoke query",
                "chunks": [
                    {
                        "chunk_id": "chunk_001",
                        "content": (
                            "KB-SWAP-TEST: Swap 미설정 환경에서는 순간적인 메모리 압박을 "
                            "흡수하지 못해 global OOM이 쉽게 발생한다."
                        ),
                        "source": "kernel_doc",
                    }
                ],
                "total_found": 1,
            },
        },
        "diagnosis": {},
        "error": None,
    }



def make_valid_bundle_payload() -> dict:
    """
    현재 Node 4의 정상 경로가 기대하는
    envelope JSON(reasoning_trace + final_answer) 샘플을 만든다.
    """
    return {
        "reasoning_trace": {
            "facts": [
                "constraint=CONSTRAINT_NONE",
                "SwapTotal=0KB",
                "java RSS 875680kB (42.0%)",
            ],
            "causal_inference": [
                "Swap 미설정이라 순간 메모리 압박을 완충하지 못했다.",
                "java의 메모리 사용량이 높아 OOM 희생 대상이 되었다.",
            ],
            "kb_application": [
                "KB-SWAP-TEST 문서상 swap 없음은 global OOM 위험을 높인다.",
                "BUG-TEST-418은 reclaim 지연 관련 known issue이다.",
            ],
            "decision_basis": [
                "정상 경로에서는 final_answer를 그대로 사용한다.",
                "현재 결과는 envelope 스키마를 만족한다.",
            ],
        },
        "final_answer": {
            "log_analysis": {
                "summary": (
                    "java 프로세스가 OOM 상황에서 종료되었다. "
                    "Swap이 설정되지 않았고, free memory가 min watermark보다 낮았다."
                ),
                "key_metrics": {
                    "total_ram": "2097152KB",
                    "swap_status": "미설정",
                    "killed_process": "java",
                    "kill_reason": "global memory pressure",
                    "constraint_type": "CONSTRAINT_NONE",
                },
            },
            "diagnosis": {
                "root_cause": (
                    "Swap이 없는 상태에서 시스템 전체 메모리 압박이 누적되며 "
                    "global OOM이 발생했다."
                ),
                "contributing_factors": [
                    "java의 메모리 사용량이 높았다.",
                    "free memory가 min watermark보다 낮았다.",
                ],
                "evidence": [
                    "constraint=CONSTRAINT_NONE",
                    "SwapTotal=0KB",
                    "SwapFree=0KB",
                    "java RSS 875680kB (42.0%)",
                ],
                "severity": "high",
            },
            "action_guide": {
                "immediate": [
                    "현재 메모리 사용량과 재발 여부를 즉시 점검한다.",
                ],
                "recommended": [
                    "Swap 추가 또는 java 메모리 상한 점검을 검토한다.",
                ],
                "further_investigation": [
                    "java 메모리 사용 증가가 정상 패턴인지 릭인지 확인한다.",
                ],
            },
        },
    }


def assert_non_empty_string(value):
    assert isinstance(value, str)
    assert value.strip()


def assert_string_list(value):
    assert isinstance(value, list)
    for item in value:
        assert_non_empty_string(item)



def assert_required_bundle_schema(bundle: dict):
    """
    현재 fail-fast Node 4가 요구하는 envelope 최소 계약을 검사한다.
    """
    assert {"reasoning_trace", "final_answer"}.issubset(bundle.keys())

    reasoning_trace = bundle["reasoning_trace"]
    final_answer = bundle["final_answer"]

    assert isinstance(reasoning_trace, dict)
    assert {
        "facts",
        "causal_inference",
        "kb_application",
        "decision_basis",
    }.issubset(reasoning_trace.keys())
    assert_string_list(reasoning_trace["facts"])
    assert_string_list(reasoning_trace["causal_inference"])
    assert_string_list(reasoning_trace["kb_application"])
    assert_string_list(reasoning_trace["decision_basis"])

    assert_required_final_answer_schema(final_answer)



def assert_required_final_answer_schema(final_answer: dict):
    """
    generate_diagnosis_with_retry()가 반환하는 final_answer 최소 계약을 검사한다.
    """
    assert {"log_analysis", "diagnosis", "action_guide"}.issubset(final_answer.keys())

    log_analysis = final_answer["log_analysis"]
    diagnosis = final_answer["diagnosis"]
    action_guide = final_answer["action_guide"]

    assert isinstance(log_analysis, dict)
    assert isinstance(diagnosis, dict)
    assert isinstance(action_guide, dict)

    assert {"summary", "key_metrics"}.issubset(log_analysis.keys())
    assert_non_empty_string(log_analysis["summary"])
    assert isinstance(log_analysis["key_metrics"], dict)
    for key in ["total_ram", "swap_status", "killed_process", "kill_reason", "constraint_type"]:
        assert key in log_analysis["key_metrics"]
        assert_non_empty_string(log_analysis["key_metrics"][key])

    assert {"root_cause", "contributing_factors", "evidence", "severity"}.issubset(diagnosis.keys())
    assert_non_empty_string(diagnosis["root_cause"])
    assert_string_list(diagnosis["contributing_factors"])
    assert_string_list(diagnosis["evidence"])
    assert diagnosis["severity"] in {"high", "medium", "low"}

    assert {"immediate", "recommended", "further_investigation"}.issubset(action_guide.keys())
    assert_string_list(action_guide["immediate"])
    assert_string_list(action_guide["recommended"])
    assert_string_list(action_guide["further_investigation"])


# -----------------------------------------------------------------------------
# 테스트용 더미 LLM
# -----------------------------------------------------------------------------
class WrappedEnvelopeLLM:
    """
    envelope JSON을 markdown code block 안에 감싸서 반환하는 더미 LLM.
    """

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return (
            "아래는 분석 결과입니다.\n"
            "```json\n"
            f"{json.dumps(self.payload, ensure_ascii=False)}\n"
            "```"
        )


class PlainTextNonJsonLLM:
    """
    JSON이 전혀 없는 일반 텍스트만 반환하는 더미 LLM.
    """

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return "이 응답은 JSON이 아닙니다. 단순한 텍스트입니다."


class JsonPayloadLLM:
    """
    전달받은 payload를 raw JSON 문자열로 반환하는 중립적인 더미 LLM.
    """

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


class ExplodingLLM:
    """
    invoke() 호출 시 예외를 발생시키는 더미 LLM.
    """

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise RuntimeError("simulated llm failure")


# -----------------------------------------------------------------------------
# 테스트
# -----------------------------------------------------------------------------
def test_build_prompt_renders_without_template_brace_error():
    """
    기본 템플릿 안의 JSON 예시 중괄호가 format 과정에서 깨지지 않는지 검증한다.
    """
    assert hasattr(node4_module, "_build_prompt"), (
        "_build_prompt 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()

    prompt = node4_module._build_prompt(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
    )

    dump_simple_io(
        "build_prompt / no template brace error",
        {
            "parsed_fields": state["parsed_fields"],
            "classification": state["classification"],
            "tool_results": state["tool_results"],
        },
        prompt,
    )

    assert isinstance(prompt, str)
    assert "reasoning_trace" in prompt
    assert "final_answer" in prompt
    assert "java" in prompt
    assert "CONSTRAINT_NONE" in prompt



def test_extract_json_parses_valid_json_codeblock():
    """
    extract_json이 정상적인 json code block 안의 JSON을 올바르게 파싱하는지 검증한다.
    """
    if not hasattr(node4_module, "extract_json"):
        pytest.skip("extract_json 유틸리티가 아직 구현되지 않았습니다.")

    wrapped = '''설명문
```json
{"a": 1, "b": 2}
```
추가 설명
'''
    parsed = node4_module.extract_json(wrapped)

    dump_simple_io(
        "extract_json / valid json codeblock",
        wrapped,
        parsed,
    )

    assert parsed == {"a": 1, "b": 2}



def test_extract_json_returns_none_for_invalid_text():
    """
    extract_json이 JSON이 전혀 없는 텍스트에 대해 None을 반환하는지 검증한다.
    """
    if not hasattr(node4_module, "extract_json"):
        pytest.skip("extract_json 유틸리티가 아직 구현되지 않았습니다.")

    input_text = "not json at all"
    parsed = node4_module.extract_json(input_text)

    dump_simple_io(
        "extract_json / invalid text",
        input_text,
        parsed,
    )

    assert parsed is None



def test_extract_json_prefers_json_codeblock_over_other_brace_noise():
    """
    응답 안에 다른 brace 텍스트가 섞여 있어도,
    정상적인 json code block이 있으면 그것을 우선 파싱해야 한다.
    """
    if not hasattr(node4_module, "extract_json"):
        pytest.skip("extract_json 유틸리티가 아직 구현되지 않았습니다.")

    messy_response = '''
머리말 {this is not valid json}
중간 텍스트

```json
{"picked": "codeblock", "value": 123}
```

꼬리말 {still not valid json}
'''
    parsed = node4_module.extract_json(messy_response)

    dump_simple_io(
        "extract_json / prefer codeblock over brace noise",
        messy_response,
        parsed,
    )

    assert parsed == {"picked": "codeblock", "value": 123}


def test_extract_json_prefers_envelope_dict_over_earlier_json_fragment():
    """
    응답 앞부분에 작은 JSON 조각이 있어도, 실제 envelope dict를 우선 선택해야 한다.
    """
    payload = make_valid_bundle_payload()
    response = (
        '{"note": "small fragment"}\n'
        f'{json.dumps(payload, ensure_ascii=False)}\n'
        '{"tail": true}'
    )

    parsed = node4_module.extract_json(response)

    dump_simple_io(
        "extract_json / prefer envelope over earlier json fragment",
        response,
        parsed,
    )

    assert parsed == payload


def test_extract_json_accepts_uppercase_json_fence_and_selects_envelope():
    """
    ```JSON 대문자 fence label이어도 envelope payload를 우선 선택해야 한다.
    """
    payload = make_valid_bundle_payload()
    response = (
        "앞부분 단편\n"
        "```JSON\n"
        "{\"note\": \"small fragment\"}\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "```\n"
        "뒷부분"
    )

    parsed = node4_module.extract_json(response)

    dump_simple_io(
        "extract_json / uppercase JSON fence selects envelope",
        response,
        parsed,
    )

    assert parsed == payload


def test_extract_json_prefers_envelope_from_later_fenced_block_over_earlier_debug_fragment():
    """
    fenced block가 여러 개면 전체 후보를 모은 뒤 later envelope를 우선 선택해야 한다.
    """
    payload = make_valid_bundle_payload()
    response = (
        "```json\n"
        "{\"debug\": \"fragment\"}\n"
        "```\n"
        "중간 설명\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "```"
    )

    parsed = node4_module.extract_json(response)

    dump_simple_io(
        "extract_json / prefer later fenced envelope over earlier debug fragment",
        response,
        parsed,
    )

    assert parsed == payload


def test_generate_diagnosis_bundle_with_retry_returns_valid_bundle_on_success():
    """
    현재 Node 4의 핵심 경로는 envelope bundle 생성이다.
    """
    assert hasattr(node4_module, "generate_diagnosis_bundle_with_retry"), (
        "generate_diagnosis_bundle_with_retry 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()
    payload = make_valid_bundle_payload()
    llm = WrappedEnvelopeLLM(payload)

    bundle = node4_module.generate_diagnosis_bundle_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=llm,
        max_retries=2,
    )

    dump_case(
        "generate_diagnosis_bundle_with_retry / success",
        state,
        bundle,
    )

    assert_required_bundle_schema(bundle)
    assert bundle["final_answer"]["diagnosis"]["severity"] == "high"
    assert bundle["final_answer"]["log_analysis"]["key_metrics"]["killed_process"] == "java"
    assert "constraint=CONSTRAINT_NONE" in bundle["reasoning_trace"]["facts"]
    assert llm.calls == 1


def test_generate_diagnosis_with_retry_returns_only_final_answer_on_success():
    """
    하위 호환용 generate_diagnosis_with_retry()는 bundle 전체가 아니라 final_answer만 반환해야 한다.
    """
    assert hasattr(node4_module, "generate_diagnosis_with_retry"), (
        "generate_diagnosis_with_retry 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()
    payload = make_valid_bundle_payload()
    llm = WrappedEnvelopeLLM(payload)

    final_answer = node4_module.generate_diagnosis_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=llm,
        max_retries=2,
    )

    dump_case(
        "generate_diagnosis_with_retry / final_answer only",
        state,
        final_answer,
    )

    assert_required_final_answer_schema(final_answer)
    assert final_answer["diagnosis"]["severity"] == "high"
    assert final_answer["log_analysis"]["key_metrics"]["killed_process"] == "java"
    assert llm.calls == 1


def test_generate_diagnosis_bundle_with_retry_returns_fallback_bundle_on_non_json_text():
    """
    JSON이 전혀 없어도 contract를 만족하는 fallback bundle을 반환해야 한다.
    """
    assert hasattr(node4_module, "generate_diagnosis_bundle_with_retry"), (
        "generate_diagnosis_bundle_with_retry 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()
    llm = PlainTextNonJsonLLM()

    bundle = node4_module.generate_diagnosis_bundle_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=llm,
        max_retries=2,
    )

    dump_case(
        "generate_diagnosis_bundle_with_retry / non-json text returns fallback bundle",
        state,
        bundle,
    )

    assert llm.calls == 1
    assert_required_bundle_schema(bundle)
    assert bundle["final_answer"]["diagnosis"]["llm_failed"] is True
    assert "valid JSON object" in bundle["final_answer"]["diagnosis"]["llm_failure_reason"]


def test_generate_diagnosis_with_retry_returns_fallback_final_answer_on_non_json_text():
    """
    하위 호환 API도 non-JSON 응답 시 fallback final_answer만 반환해야 한다.
    """
    assert hasattr(node4_module, "generate_diagnosis_with_retry"), (
        "generate_diagnosis_with_retry 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()
    llm = PlainTextNonJsonLLM()

    final_answer = node4_module.generate_diagnosis_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=llm,
        max_retries=2,
    )

    dump_case(
        "generate_diagnosis_with_retry / non-json text returns fallback final_answer",
        state,
        final_answer,
    )

    assert llm.calls == 1
    assert_required_final_answer_schema(final_answer)
    assert "reasoning_trace" not in final_answer
    assert final_answer["diagnosis"]["llm_failed"] is True
    assert "valid JSON object" in final_answer["diagnosis"]["llm_failure_reason"]


def test_generate_diagnosis_bundle_with_retry_repairs_wrong_envelope_schema():
    """
    JSON은 유효하지만 envelope 구조가 틀리면 repair/fallback bundle을 반환해야 한다.
    """
    assert hasattr(node4_module, "generate_diagnosis_bundle_with_retry"), (
        "generate_diagnosis_bundle_with_retry 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()

    wrong_schema_payload = {
        "message": "this is valid JSON but wrong schema",
        "final_answer": {
            "log_analysis": {
                "summary": "some summary",
                "key_metrics": {
                    "total_ram": "2097152KB",
                    "swap_status": "미설정",
                    "killed_process": "java",
                    "kill_reason": "global memory pressure",
                    "constraint_type": "CONSTRAINT_NONE",
                },
            },
            "diagnosis": {
                "root_cause": "wrong envelope because reasoning_trace is missing",
                "contributing_factors": ["factor1"],
                "evidence": ["e1"],
                "severity": "high",
            },
            "action_guide": {
                "immediate": ["i1"],
                "recommended": ["r1"],
                "further_investigation": ["f1"],
            },
        },
    }

    llm = JsonPayloadLLM(wrong_schema_payload)

    bundle = node4_module.generate_diagnosis_bundle_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=llm,
        max_retries=2,
    )

    dump_case(
        "generate_diagnosis_bundle_with_retry / wrong envelope schema repaired",
        state,
        bundle,
    )

    assert llm.calls == 1
    assert_required_bundle_schema(bundle)
    assert bundle["final_answer"]["diagnosis"]["llm_failed"] is True
    assert "missing the required top-level envelope keys" in bundle["final_answer"]["diagnosis"]["llm_failure_reason"]
    assert bundle["final_answer"]["diagnosis"]["root_cause"] == (
        "wrong envelope because reasoning_trace is missing"
    )
    assert bundle["reasoning_trace"]["decision_basis"]


def test_generate_diagnosis_bundle_with_retry_marks_reasoning_trace_schema_failure():
    """
    reasoning_trace만 불량한 envelope가 와도 recovery path가 llm_failed와 failure_reason를 남겨야 한다.
    """
    state = create_base_state()
    payload = make_valid_bundle_payload()
    payload["reasoning_trace"] = {
        "facts": ["ok fact", 123],
        "causal_inference": ["valid causal inference"],
        "kb_application": ["valid kb application"],
        "decision_basis": ["valid basis"],
    }

    bundle = node4_module.generate_diagnosis_bundle_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=JsonPayloadLLM(payload),
        max_retries=2,
    )

    dump_case(
        "generate_diagnosis_bundle_with_retry / malformed reasoning_trace triggers recovery",
        state,
        bundle,
    )

    assert_required_bundle_schema(bundle)
    assert bundle["final_answer"]["diagnosis"]["llm_failed"] is True
    assert "reasoning_trace failed internal schema/type validation" in bundle["final_answer"]["diagnosis"]["llm_failure_reason"]
    assert bundle["reasoning_trace"]["facts"] == ["ok fact"]


def test_generate_diagnosis_bundle_with_retry_repairs_partial_fields_and_fills_required_fallbacks():
    """
    부분적으로만 맞는 JSON이 오면, 유효한 LLM 필드는 최대한 유지하면서
    누락/오염된 필수 필드는 fallback 값으로 복구해야 한다.
    """
    state = create_base_state()

    partially_broken_payload = {
        "reasoning_trace": {
            "facts": "java killed by oom",
            "decision_basis": ["raw llm basis"],
        },
        "final_answer": {
            "log_analysis": {
                "summary": "LLM summary should be preserved.",
                "key_metrics": {
                    "killed_process": "java",
                    "pressure_score": "0.91",
                },
                "confidence": "medium",
            },
            "diagnosis": {
                "root_cause": "LLM root cause should survive repair.",
                "contributing_factors": "single contributing factor",
                "evidence": ["anon-rss=875680kB"],
                "severity": "critical",
                "confidence": 0.73,
            },
            "action_guide": {
                "recommended": "Inspect swap configuration",
                "notes": ["preserve me"],
            },
        },
    }

    bundle = node4_module.generate_diagnosis_bundle_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=JsonPayloadLLM(partially_broken_payload),
        max_retries=2,
    )

    dump_case(
        "generate_diagnosis_bundle_with_retry / partial schema repair preserves llm fields",
        state,
        bundle,
    )

    assert_required_bundle_schema(bundle)
    assert bundle["reasoning_trace"]["facts"] == ["java killed by oom"]
    assert bundle["reasoning_trace"]["decision_basis"][0] == "raw llm basis"
    assert bundle["reasoning_trace"]["causal_inference"]
    assert bundle["reasoning_trace"]["kb_application"]

    final_answer = bundle["final_answer"]
    assert final_answer["log_analysis"]["summary"] == "LLM summary should be preserved."
    assert final_answer["log_analysis"]["confidence"] == "medium"
    assert final_answer["log_analysis"]["key_metrics"]["killed_process"] == "java"
    assert final_answer["log_analysis"]["key_metrics"]["total_ram"] == "2097152KB"
    assert final_answer["log_analysis"]["key_metrics"]["pressure_score"] == "0.91"
    assert final_answer["log_analysis"]["key_metrics"]["swap_status"] == "not configured"
    assert final_answer["diagnosis"]["root_cause"] == "LLM root cause should survive repair."
    assert final_answer["diagnosis"]["contributing_factors"] == ["single contributing factor"]
    assert final_answer["diagnosis"]["evidence"] == ["anon-rss=875680kB"]
    assert final_answer["diagnosis"]["severity"] == "high"
    assert final_answer["diagnosis"]["llm_failed"] is True
    assert "internal schema/type validation" in final_answer["diagnosis"]["llm_failure_reason"]
    assert final_answer["diagnosis"]["confidence"] == 0.73
    assert final_answer["action_guide"]["recommended"] == ["Inspect swap configuration"]
    assert final_answer["action_guide"]["immediate"]
    assert final_answer["action_guide"]["further_investigation"]
    assert final_answer["action_guide"]["notes"] == ["preserve me"]


def test_generate_diagnosis_bundle_with_retry_preserves_top_level_extra_fields():
    """
    envelope 바깥 top-level extra field도 normalize 이후 유지되어야 한다.
    """
    state = create_base_state()
    payload = make_valid_bundle_payload()
    payload["response_meta"] = {"model": "test-model", "latency_ms": 123}
    payload["debug_tag"] = "node4-extra-preserve"

    bundle = node4_module.generate_diagnosis_bundle_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=JsonPayloadLLM(payload),
        max_retries=2,
    )

    dump_case(
        "generate_diagnosis_bundle_with_retry / preserve top-level extra fields",
        state,
        bundle,
    )

    assert_required_bundle_schema(bundle)
    assert bundle["response_meta"] == {"model": "test-model", "latency_ms": 123}
    assert bundle["debug_tag"] == "node4-extra-preserve"


def test_build_fallback_key_metrics_accepts_string_numbers_and_invalid_swap_values_safely():
    """
    fallback 경로는 문자열 숫자와 이상값이 섞여도 예외 없이 key_metrics를 만들어야 한다.
    """
    parsed_fields = {
        "total_ram_pages": "524288",
        "swap_total_kb": "invalid",
        "swap_free_kb": "128",
        "killed_process": "java",
        "constraint": "CONSTRAINT_NONE",
    }
    classification = {"oom_type": "global_oom"}

    key_metrics = node4_module._build_fallback_key_metrics(parsed_fields, classification)

    dump_simple_io(
        "_build_fallback_key_metrics / safe numeric parsing",
        {"parsed_fields": parsed_fields, "classification": classification},
        key_metrics,
    )

    assert key_metrics["total_ram"] == "2097152KB"
    assert key_metrics["swap_status"] == "unknown"
    assert key_metrics["killed_process"] == "java"
    assert key_metrics["constraint_type"] == "CONSTRAINT_NONE"


def test_validate_final_answer_schema_rejects_non_string_and_non_string_list_members():
    """
    success 판정은 키 존재만이 아니라 문자열/문자열 배열 계약도 확인해야 한다.
    """
    invalid_final_answer = make_valid_bundle_payload()["final_answer"]
    invalid_final_answer["log_analysis"]["summary"] = {"bad": "type"}
    invalid_final_answer["diagnosis"]["contributing_factors"] = ["ok", 123]

    dump_simple_io(
        "_validate_final_answer_schema / reject weakly typed success payload",
        invalid_final_answer,
        node4_module._validate_final_answer_schema(invalid_final_answer),
    )

    assert node4_module._validate_final_answer_schema(invalid_final_answer) is False


def test_validate_final_answer_schema_rejects_empty_summary():
    invalid_final_answer = make_valid_bundle_payload()["final_answer"]
    invalid_final_answer["log_analysis"]["summary"] = ""

    dump_simple_io(
        "_validate_final_answer_schema / reject empty summary",
        invalid_final_answer,
        node4_module._validate_final_answer_schema(invalid_final_answer),
    )

    assert node4_module._validate_final_answer_schema(invalid_final_answer) is False


def test_validate_llm_bundle_schema_allows_empty_lists_when_types_are_valid():
    bundle = make_valid_bundle_payload()
    bundle["reasoning_trace"]["facts"] = []
    bundle["reasoning_trace"]["causal_inference"] = []
    bundle["reasoning_trace"]["kb_application"] = []
    bundle["reasoning_trace"]["decision_basis"] = []
    bundle["final_answer"]["diagnosis"]["contributing_factors"] = []
    bundle["final_answer"]["diagnosis"]["evidence"] = []
    bundle["final_answer"]["action_guide"]["immediate"] = []
    bundle["final_answer"]["action_guide"]["recommended"] = []
    bundle["final_answer"]["action_guide"]["further_investigation"] = []

    dump_simple_io(
        "_validate_llm_bundle_schema / allow empty typed lists",
        bundle,
        node4_module._validate_llm_bundle_schema(bundle),
    )

    assert node4_module._validate_llm_bundle_schema(bundle) is True


def test_validate_reasoning_trace_schema_rejects_non_string_members():
    """
    reasoning_trace도 문자열 배열 계약을 만족하지 않으면 성공 payload로 취급하면 안 된다.
    """
    invalid_reasoning_trace = make_valid_bundle_payload()["reasoning_trace"]
    invalid_reasoning_trace["facts"] = ["ok", 42]
    invalid_reasoning_trace["decision_basis"] = {"bad": "type"}

    dump_simple_io(
        "_validate_reasoning_trace_schema / reject weakly typed reasoning trace",
        invalid_reasoning_trace,
        node4_module._validate_reasoning_trace_schema(invalid_reasoning_trace),
    )

    assert node4_module._validate_reasoning_trace_schema(invalid_reasoning_trace) is False


def test_generate_diagnosis_bundle_with_retry_builds_minimal_fallback_when_signals_are_sparse():
    """
    parsed/classification/tool 결과가 거의 비어 있어도 fallback bundle은 계약을 유지해야 한다.
    """
    parsed_fields = {
        "killed_process": None,
        "constraint": None,
        "total_ram_pages": None,
        "swap_total_kb": None,
        "swap_free_kb": None,
    }
    classification = {}
    tool_results = {}
    llm = PlainTextNonJsonLLM()

    bundle = node4_module.generate_diagnosis_bundle_with_retry(
        parsed_fields,
        classification,
        tool_results,
        llm=llm,
        max_retries=2,
    )

    dump_simple_io(
        "generate_diagnosis_bundle_with_retry / minimal sparse fallback",
        {
            "parsed_fields": parsed_fields,
            "classification": classification,
            "tool_results": tool_results,
        },
        bundle,
    )

    assert llm.calls == 1
    assert_required_bundle_schema(bundle)
    assert bundle["reasoning_trace"]["facts"] == ["Node 4 fallback path was used."]
    assert bundle["reasoning_trace"]["kb_application"] == ["Available tool result groups: none."]
    assert bundle["final_answer"]["log_analysis"]["key_metrics"] == {
        "total_ram": "unknown",
        "swap_status": "unknown",
        "killed_process": "unknown",
        "kill_reason": "unknown",
        "constraint_type": "unknown",
    }
    assert bundle["final_answer"]["diagnosis"]["evidence"] == ["Node 4 fallback path was used."]
    assert bundle["final_answer"]["diagnosis"]["llm_failed"] is True


def test_node_4_synthesizer_preserves_minimal_contract_for_sparse_state_fallback():
    """
    sparse state에서도 wrapper 경로가 fallback diagnosis 최소 계약을 유지해야 한다.
    """
    state = {
        "raw_log": "",
        "user_kernel_version": None,
        "parsed_fields": {
            "killed_process": None,
            "constraint": None,
            "total_ram_pages": None,
            "swap_total_kb": None,
            "swap_free_kb": None,
        },
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
        "llm": PlainTextNonJsonLLM(),
    }

    updated_state = node4_module.node_4_synthesizer(state)

    dump_wrapper_case(
        "node_4_synthesizer / sparse state fallback keeps minimal contract",
        state,
        updated_state,
    )

    assert_required_final_answer_schema(updated_state["diagnosis"])
    assert updated_state["diagnosis"]["diagnosis"]["llm_failed"] is True
    assert updated_state["diagnosis"]["log_analysis"]["key_metrics"] == {
        "total_ram": "unknown",
        "swap_status": "unknown",
        "killed_process": "unknown",
        "kill_reason": "unknown",
        "constraint_type": "unknown",
    }


@pytest.mark.parametrize(
    ("payload", "expected_reason_substring"),
    [
        pytest.param(
            {"final_answer": make_valid_bundle_payload()["final_answer"]},
            "missing key: reasoning_trace",
            id="missing_reasoning_trace_only",
        ),
        pytest.param(
            {"reasoning_trace": make_valid_bundle_payload()["reasoning_trace"]},
            "missing key: final_answer",
            id="missing_final_answer_only",
        ),
        pytest.param(
            {"message": "no envelope keys"},
            "reasoning_trace and final_answer",
            id="missing_both_keys",
        ),
    ],
)
def test_generate_diagnosis_bundle_with_retry_distinguishes_missing_envelope_keys(
    payload,
    expected_reason_substring,
):
    """
    envelope top-level key 누락 유형별로 failure reason이 구분되어야 한다.
    """
    state = create_base_state()

    bundle = node4_module.generate_diagnosis_bundle_with_retry(
        state["parsed_fields"],
        state["classification"],
        state["tool_results"],
        llm=JsonPayloadLLM(payload),
        max_retries=2,
    )

    dump_case(
        f"generate_diagnosis_bundle_with_retry / differentiated missing envelope keys / {expected_reason_substring}",
        state,
        bundle,
    )

    assert_required_bundle_schema(bundle)
    assert bundle["final_answer"]["diagnosis"]["llm_failed"] is True
    assert expected_reason_substring in bundle["final_answer"]["diagnosis"]["llm_failure_reason"]


def test_generate_diagnosis_bundle_with_retry_propagates_llm_invoke_exception():
    """
    현재 방향성은 fail-fast이므로, llm.invoke()에서 예외가 발생하면 그대로 전파되어야 한다.
    """
    assert hasattr(node4_module, "generate_diagnosis_bundle_with_retry"), (
        "generate_diagnosis_bundle_with_retry 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()
    llm = ExplodingLLM()

    with pytest.raises(RuntimeError) as exc_info:
        node4_module.generate_diagnosis_bundle_with_retry(
            state["parsed_fields"],
            state["classification"],
            state["tool_results"],
            llm=llm,
            max_retries=2,
        )

    dump_exception_case(
        "generate_diagnosis_bundle_with_retry / llm.invoke exception propagates",
        state,
        exc_info.value,
    )

    assert llm.calls == 1
    assert "simulated llm failure" in str(exc_info.value)


def test_node_4_synthesizer_returns_fallback_diagnosis_when_live_llm_output_is_non_json():
    """
    wrapper 단에서도 실제 fallback 경로가 diagnosis에 반영되는지 검증한다.
    """
    state = create_base_state()
    state["llm"] = PlainTextNonJsonLLM()

    updated_state = node4_module.node_4_synthesizer(state)

    dump_wrapper_case(
        "node_4_synthesizer / non-json llm output returns fallback diagnosis",
        state,
        updated_state,
    )

    assert updated_state["parsed_fields"] == state["parsed_fields"]
    assert updated_state["classification"] == state["classification"]
    assert updated_state["tool_results"] == state["tool_results"]
    assert "diagnosis_trace" not in updated_state
    assert_required_final_answer_schema(updated_state["diagnosis"])
    assert updated_state["diagnosis"]["diagnosis"]["llm_failed"] is True
    assert "valid JSON object" in updated_state["diagnosis"]["diagnosis"]["llm_failure_reason"]


def test_node_4_synthesizer_updates_state_with_diagnosis_only(monkeypatch):
    """
    현재 wrapper는 bundle-first 구조를 유지하되,
    팀 state contract에 맞춰 diagnosis만 state에 저장해야 한다.

    즉:
    - bundle['final_answer'] -> state['diagnosis']
    - bundle['reasoning_trace'] -> state에 저장하지 않음
    """
    assert hasattr(node4_module, "node_4_synthesizer"), (
        "node_4_synthesizer 함수가 구현되어 있어야 합니다."
    )
    assert hasattr(node4_module, "generate_diagnosis_bundle_with_retry"), (
        "wrapper가 위임할 generate_diagnosis_bundle_with_retry 함수가 필요합니다."
    )

    state = create_base_state()
    fake_bundle = make_valid_bundle_payload()

    def fake_generate_bundle(parsed_fields, classification, tool_results, llm=None, max_retries=2):
        return copy.deepcopy(fake_bundle)

    monkeypatch.setattr(node4_module, "generate_diagnosis_bundle_with_retry", fake_generate_bundle)

    updated_state = node4_module.node_4_synthesizer(state)

    dump_wrapper_case(
        "node_4_synthesizer / updates state with diagnosis only",
        state,
        updated_state,
    )

    assert updated_state["diagnosis"] == fake_bundle["final_answer"]
    assert "diagnosis_trace" not in updated_state
    assert updated_state["parsed_fields"]["killed_process"] == "java"
    assert updated_state["classification"]["oom_type"] == "global_oom"
    assert "memory" in updated_state["tool_results"]
    assert "kernel_bugs" in updated_state["tool_results"]
    assert "kb_chunks" in updated_state["tool_results"]


def test_node_4_synthesizer_accepts_partial_tool_results_without_rewriting_them(monkeypatch):
    """
    wrapper는 tool_results를 정규화하거나 보정하지 않고,
    현재 state에 들어온 부분 성공/부분 실패 결과를 그대로 하위 함수에 전달해야 한다.
    """
    assert hasattr(node4_module, "node_4_synthesizer"), (
        "node_4_synthesizer 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()
    state["tool_results"] = {
        "memory": {
            "sentinel_memory_note": "PARTIAL-MEMORY-RESULT",
            "top5_processes": [],
        },
        "kb_chunks": {
            "error": "KB 검색 오류: synthetic timeout"
        },
    }

    captured = {}
    fake_bundle = make_valid_bundle_payload()

    def fake_generate_bundle(parsed_fields, classification, tool_results, llm=None, max_retries=2):
        captured["parsed_fields"] = copy.deepcopy(parsed_fields)
        captured["classification"] = copy.deepcopy(classification)
        captured["tool_results"] = copy.deepcopy(tool_results)
        return copy.deepcopy(fake_bundle)

    monkeypatch.setattr(node4_module, "generate_diagnosis_bundle_with_retry", fake_generate_bundle)

    updated_state = node4_module.node_4_synthesizer(state)

    dump_wrapper_case(
        "node_4_synthesizer / accepts partial tool_results without rewriting them",
        state,
        updated_state,
    )

    assert captured["parsed_fields"] == state["parsed_fields"]
    assert captured["classification"] == state["classification"]
    assert captured["tool_results"] == state["tool_results"]
    assert updated_state["tool_results"] == state["tool_results"]
    assert updated_state["diagnosis"] == fake_bundle["final_answer"]
    assert "diagnosis_trace" not in updated_state



def test_node_4_synthesizer_propagates_exception_without_swallowing(monkeypatch):
    """
    현재 wrapper는 fail-fast이므로,
    내부 bundle 생성 함수가 예외를 던지면 그대로 전파해야 한다.
    """
    assert hasattr(node4_module, "node_4_synthesizer"), (
        "node_4_synthesizer 함수가 구현되어 있어야 합니다."
    )

    state = create_base_state()

    def fake_generate_bundle(parsed_fields, classification, tool_results, llm=None, max_retries=2):
        raise node4_module.Node4SchemaError("forced schema failure")

    monkeypatch.setattr(node4_module, "generate_diagnosis_bundle_with_retry", fake_generate_bundle)

    with pytest.raises(node4_module.Node4SchemaError) as exc_info:
        node4_module.node_4_synthesizer(state)

    dump_exception_case(
        "node_4_synthesizer / propagates exception without swallowing",
        state,
        exc_info.value,
    )

    assert "forced schema failure" in str(exc_info.value)
