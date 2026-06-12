import os
import sys
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
# 실제 파이프라인 노드 import
# -----------------------------------------------------------------------------
# 이 smoke test는 graph.py 없이도
# node_1 -> node_2 -> node_3 -> node_4를 직접 연결해서
# 인터페이스가 실제로 맞물리는지 확인하는 목적이다.
import app.agent.nodes.node_1_parser as node1_module
import app.agent.nodes.node_2_classifier as node2_module
import app.agent.nodes.node_3_executor as node3_module
import app.agent.nodes.node_4_synthesizer as node4_module


# -----------------------------------------------------------------------------
# 디버그 출력 헬퍼
# -----------------------------------------------------------------------------
# pytest는 기본적으로 stdout을 캡처하므로,
# 사람이 단계별 상태를 직접 보고 싶다면 아래처럼 -s 옵션과 함께 실행하면 된다.
#
#   pytest -s tests/test_all_smoke.py
#
# state 안에는 더미 LLM 같은 JSON 직렬화 불가 객체가 들어갈 수 있으므로,
# 출력 전에 사람이 읽을 수 있는 형태로 sanitize 하는 헬퍼를 둔다.
# -----------------------------------------------------------------------------
def sanitize_for_print(data):
    """
    JSON 직렬화가 되지 않는 객체가 섞여 있어도
    사람이 읽기 쉬운 값으로 재귀 변환한다.

    규칙:
    - dict -> 값 재귀 변환
    - list/tuple -> 원소 재귀 변환
    - str/int/float/bool/None -> 그대로
    - 그 외 객체 -> repr(obj)
    """
    if isinstance(data, dict):
        return {k: sanitize_for_print(v) for k, v in data.items()}

    if isinstance(data, list):
        return [sanitize_for_print(v) for v in data]

    if isinstance(data, tuple):
        return [sanitize_for_print(v) for v in data]

    if isinstance(data, (str, int, float, bool)) or data is None:
        return data

    return repr(data)


def pretty_json(data) -> str:
    """
    객체를 사람이 읽기 쉬운 pretty JSON 문자열로 변환한다.
    """
    safe = sanitize_for_print(data)
    return json.dumps(safe, indent=2, ensure_ascii=False)


def dump_pipeline_state(title: str, state: dict):
    """
    단계별 state를 보기 좋게 출력한다.
    """
    print(f"\n{'=' * 100}")
    print(f"[PIPELINE STATE] {title}")
    print(f"{'=' * 100}")
    print(pretty_json(state))


def read_log_from_subdir(filename: str) -> str:
    """
    tests/node_1_logs/<filename> 파일을 읽어 온다.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "node_1_logs", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_initial_state(raw_log: str) -> dict:
    """
    파이프라인 시작 상태를 만든다.

    실제 OOMState에 맞춰 최소 필수 필드를 모두 채워 둔다.
    현재 팀 contract 기준으로 state에는 diagnosis만 두고,
    diagnosis_trace는 두지 않는다.
    """
    return {
        "raw_log": raw_log,
        "user_kernel_version": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }


REAL_LOG_PIPELINE_CASES = [
    {
        "case_id": 1,
        "filename": "case1_global.txt",
        "expected_trigger_process": "httpd",
        "expected_killed_process": "java",
        "expected_constraint": "CONSTRAINT_NONE",
        "expected_oom_type": "global_oom",
    },
    {
        "case_id": 2,
        "filename": "case2_cgroup.txt",
        "expected_trigger_process": "s1-agent",
        "expected_killed_process": "s1-agent",
        "expected_constraint": None,
        "expected_oom_type": "cgroup_oom",
    },
    {
        "case_id": 3,
        "filename": "case3_log017.txt",
        "expected_trigger_process": "httpd",
        "expected_killed_process": "httpd",
        "expected_constraint": "CONSTRAINT_NONE",
        "expected_oom_type": "global_oom",
    },
    {
        "case_id": 4,
        "filename": "case4_keystone.txt",
        "expected_trigger_process": "keystone-all",
        "expected_killed_process": "keystone-all",
        "expected_constraint": None,
        "expected_oom_type": "global_oom",
    },
    {
        "case_id": 5,
        "filename": "case5_flasherav.txt",
        "expected_trigger_process": "flasherav",
        "expected_killed_process": "flasherav",
        "expected_constraint": None,
        "expected_oom_type": "global_oom",
    },
    {
        "case_id": 6,
        "filename": "case6_memleak.txt",
        "expected_trigger_process": None,
        "expected_killed_process": "invoke_memleak",
        "expected_constraint": "CONSTRAINT_NONE",
        "expected_oom_type": "global_oom",
    },
    {
        "case_id": 7,
        "filename": "case7_storm.txt",
        "expected_trigger_process": "modprobe",
        "expected_killed_process": "systemd-stdout-",
        "expected_constraint": None,
        "expected_oom_type": "global_oom",
    },
    {
        "case_id": 8,
        "filename": "case8_java_high_ram.txt",
        "expected_trigger_process": "telegraf",
        "expected_killed_process": "java",
        "expected_constraint": None,
        "expected_oom_type": "global_oom",
    },
    {
        "case_id": 9,
        "filename": "case9_malloc_global_oom.txt",
        "expected_trigger_process": "node",
        "expected_killed_process": "malloc",
        "expected_constraint": "CONSTRAINT_NONE",
        "expected_oom_type": "global_oom",
    },
    {
        "case_id": 10,
        "filename": "case10_malloc_cgroup_oom.txt",
        "expected_trigger_process": "runc:[2:INIT]",
        "expected_killed_process": "node",
        "expected_constraint": "CONSTRAINT_MEMCG",
        "expected_oom_type": "cgroup_oom",
    },
]


def build_fake_node2_payload(parsed_fields: dict) -> dict:
    """
    실제 Node 2의 하드 제약을 단순 모사해,
    실로그 기반 smoke test에서도 도구 선택이 입력 파싱 결과에 반응하도록 만든다.
    """
    tools_needed = ["kernel_param_recommender"]

    if len(parsed_fields.get("process_table", [])) >= 2:
        tools_needed.insert(0, "memory_calculator")

    if parsed_fields.get("kernel_version"):
        insert_at = 1 if "memory_calculator" in tools_needed else 0
        tools_needed.insert(insert_at, "kernel_version_check")

    oom_type = "cgroup_oom" if (
        parsed_fields.get("constraint") == "CONSTRAINT_MEMCG"
        or parsed_fields.get("cgroup_path")
        or parsed_fields.get("cgroup_usage_kb") is not None
        or parsed_fields.get("cgroup_limit_kb") is not None
    ) else "global_oom"

    return {
        "oom_type": oom_type,
        "tools_needed": tools_needed,
        "needs_kb": oom_type in {"global_oom", "cgroup_oom"},
        "confidence": "high" if parsed_fields.get("killed_process") else "low",
    }


# -----------------------------------------------------------------------------
# Node 2 mocking helpers
# -----------------------------------------------------------------------------
# 현재 node_2_classifier는 내부에서 다음 흐름을 사용한다.
#   - ChatOllama(...)
#   - load_prompt(...)
#   - ChatPromptTemplate.from_template(...)
#   - prompt_template | llm
#   - chain.invoke(...)
#
# 실제 Ollama 서버 없이 smoke test를 돌리기 위해,
# chain.invoke(...)가 response.content에 JSON 문자열을 담아 반환하도록 흉내 낸다.
# -----------------------------------------------------------------------------
class FakeNode2Response:
    """
    Node 2의 chain.invoke(...) 반환값 대용 객체.

    실제 구현은 response.content를 json.loads() 하므로,
    content 속성에 JSON 문자열을 넣는다.
    """

    def __init__(self, payload: dict):
        self.content = json.dumps(payload, ensure_ascii=False)


class FakeNode2Chain:
    """
    Node 2의 prompt_template | llm 결과를 흉내 내는 체인 객체.
    invoke(...) 호출 시 고정 분류 결과를 반환한다.
    """

    def __init__(self, payload: dict | None = None):
        self.payload = payload

    def invoke(self, _inputs: dict):
        parsed_fields = json.loads(_inputs["parsed_fields"])
        payload = self.payload or build_fake_node2_payload(parsed_fields)
        return FakeNode2Response(payload)


class FakeNode2PromptTemplate:
    """
    ChatPromptTemplate.from_template(...)가 반환하는 객체 대용.
    __or__ 연산자를 구현해 prompt_template | llm 을 흉내 낸다.
    """

    def __init__(self, payload: dict | None = None):
        self.payload = payload

    def __or__(self, _llm):
        return FakeNode2Chain(self.payload)


class FakeNode2ChatPromptTemplate:
    """
    node_2_classifier.py 안의 ChatPromptTemplate를 monkeypatch하기 위한 래퍼.
    from_template(...) 호출 시 FakeNode2PromptTemplate를 반환한다.
    """

    payload = None

    @classmethod
    def from_template(cls, _template_text: str):
        return FakeNode2PromptTemplate(cls.payload)


class DummyNode2LLM:
    """
    Node 2용 더미 LLM.

    실제 동작은 prompt_template | llm 이후 chain.invoke(...) 쪽에서 처리하므로,
    객체 자체는 비어 있어도 충분하다.
    """

    pass


# -----------------------------------------------------------------------------
# Node 4 mocking helper
# -----------------------------------------------------------------------------
# 현재 node_4_synthesizer는 state["llm"]이 있으면 그것을 사용한다.
# 따라서 smoke test에서는 state에 더미 LLM을 주입해서
# 실제 Ollama 호출 없이 bundle-first 경로를 검증할 수 있다.
#
# 중요:
# - Node 4는 내부적으로 reasoning_trace + final_answer envelope를 생성한다.
# - 하지만 wrapper는 팀 state contract에 맞춰 final_answer만 diagnosis에 저장한다.
# - 따라서 더미 LLM은 envelope 전체를 반환하되,
#   최종 assertion은 diagnosis(final_answer) 기준으로만 본다.
# -----------------------------------------------------------------------------
class DummyNode4LLM:
    """
    Node 4에서 사용할 프롬프트 인식형 더미 LLM.

    동작:
    - invoke(prompt)를 호출하면 prompt 안 문자열을 검사한다.
    - Node 3의 tool 결과 sentinel / KB / kernel bug 신호가 prompt 안에 들어오면
      final_answer와 reasoning_trace에 반영한다.
    - 반환 형식은 ```json code block```으로 감싸서
      extract_json() 경로도 같이 검증한다.
    """

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1

        has_bug_signal = "BUG-SMOKE-001" in prompt
        has_kb_signal = "KB-SMOKE-001" in prompt
        has_memory_signal = "MEMORY-SIGNAL-SMOKE" in prompt

        facts = [
            "constraint=CONSTRAINT_NONE",
            "SwapTotal=0KB",
            "Killed process is java",
        ]
        causal_inference = [
            "Swap 미설정 상태에서 메모리 압박을 흡수할 버퍼가 없다.",
            "상위 프로세스 메모리 점유가 높아 global OOM으로 이어졌다.",
        ]
        kb_application = [
            "도구 결과와 KB 결과를 함께 고려해 global OOM 대응 방향을 정리했다.",
        ]
        decision_basis = [
            "로그의 constraint/free/min/swap 수치와 프로세스 테이블을 함께 사용했다.",
        ]

        contributing_factors = [
            "Swap 미설정",
            "java RSS 과다",
            "상위 프로세스 메모리 비중 높음",
        ]
        evidence = [
            "constraint=CONSTRAINT_NONE",
            "SwapTotal=0KB",
            "java anon-rss=875680kB",
        ]
        further_investigation = [
            "java 메모리 사용이 정상 동작인지 메모리 릭인지 확인 필요",
        ]

        if has_memory_signal:
            facts.append("memory_calculator 결과가 prompt에 포함되었다.")
            decision_basis.append(
                "MEMORY-SIGNAL-SMOKE가 확인되어 Node 3 → Node 4 전달이 검증되었다."
            )
            contributing_factors.append("MEMORY-SIGNAL-SMOKE")
            evidence.append("MEMORY-SIGNAL-SMOKE")

        if has_bug_signal:
            kb_application.append("커널 버그 참고 정보가 최종 판단에 반영되었다.")
            evidence.append("BUG-SMOKE-001: test-only known issue for integration smoke test")
            further_investigation.append(
                "BUG-SMOKE-001: test-only known issue for integration smoke test"
            )

        if has_kb_signal:
            kb_application.append("RAG 검색 결과가 최종 판단에 반영되었다.")
            evidence.append(
                "KB-SMOKE-001: Swap 미설정 환경에서는 global OOM이 쉽게 발생할 수 있다."
            )
            further_investigation.append(
                "KB-SMOKE-001: Swap 미설정 환경에서는 global OOM이 쉽게 발생할 수 있다."
            )

        payload = {
            "reasoning_trace": {
                "facts": facts,
                "causal_inference": causal_inference,
                "kb_application": kb_application,
                "decision_basis": decision_basis,
            },
            "final_answer": {
                "log_analysis": {
                    "summary": (
                        "java 프로세스가 global OOM 상황에서 종료되었고, "
                        "Swap 미설정 및 메모리 압박이 함께 관찰되었다."
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
                        "Swap 미설정 상태에서 메모리 압박이 누적되었고 "
                        "java 프로세스의 큰 RSS와 상위 프로세스 비중이 겹쳐 global OOM이 발생했다."
                    ),
                    "contributing_factors": contributing_factors,
                    "evidence": evidence,
                    "severity": "high",
                },
                "action_guide": {
                    "immediate": [
                        "시스템 메모리 상태를 즉시 점검한다.",
                    ],
                    "recommended": [
                        "java 프로세스 힙 메모리 제한 확인 (-Xmx)",
                        "httpd 워커 프로세스 수 조정 (MaxRequestWorkers)",
                    ],
                    "further_investigation": further_investigation,
                },
            },
        }

        return (
            "아래는 분석 결과입니다.\n"
            "```json\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "```"
        )


# -----------------------------------------------------------------------------
# 공통 assertion helpers
# -----------------------------------------------------------------------------
def assert_node_1_shape(state: dict):
    """
    Node 1 결과가 현재 구현 명세에 맞는 최소 스키마를 만족하는지 검사한다.
    """
    assert "parsed_fields" in state
    parsed = state["parsed_fields"]

    required_keys = {
        "trigger_process",
        "killed_process",
        "killed_pid",
        "total_vm_kb",
        "anon_rss_kb",
        "oom_score_adj",
        "total_ram_pages",
        "node_free_kb",
        "node_min_kb",
        "swap_total_kb",
        "swap_free_kb",
        "constraint",
        "gfp_mask",
        "order",
        "cgroup_path",
        "cgroup_usage_kb",
        "cgroup_limit_kb",
        "cgroup_failcnt",
        "process_table",
        "kernel_version",
    }

    assert required_keys.issubset(parsed.keys())
    assert isinstance(parsed["process_table"], list)


def assert_node_2_shape(state: dict):
    """
    Node 2 결과가 현재 구현 명세에 맞는 최소 스키마를 만족하는지 검사한다.
    """
    assert "classification" in state
    classification = state["classification"]

    required_keys = {"oom_type", "tools_needed", "needs_kb", "confidence"}
    assert required_keys.issubset(classification.keys())
    assert isinstance(classification["tools_needed"], list)
    assert isinstance(classification["needs_kb"], bool)


def assert_node_3_shape(state: dict):
    """
    Node 3 결과가 현재 구현 명세에 맞는 최소 스키마를 만족하는지 검사한다.
    """
    assert "tool_results" in state
    tool_results = state["tool_results"]
    assert isinstance(tool_results, dict)


def assert_final_diagnosis_shape(final_state: dict):
    """
    Node 4 최종 결과가 현재 구현 명세와 팀 state contract에 맞는
    최소 3단 구조를 만족하는지 검사한다.
    """
    assert "diagnosis" in final_state
    diagnosis = final_state["diagnosis"]

    assert isinstance(diagnosis, dict)
    assert {"log_analysis", "diagnosis", "action_guide"}.issubset(diagnosis.keys())

    assert "summary" in diagnosis["log_analysis"]
    assert "key_metrics" in diagnosis["log_analysis"]

    assert {"root_cause", "contributing_factors", "evidence", "severity"}.issubset(
        diagnosis["diagnosis"].keys()
    )

    assert {"immediate", "recommended", "further_investigation"}.issubset(
        diagnosis["action_guide"].keys()
    )

    assert isinstance(diagnosis["diagnosis"]["contributing_factors"], list)
    assert isinstance(diagnosis["diagnosis"]["evidence"], list)
    assert isinstance(diagnosis["action_guide"]["immediate"], list)
    assert isinstance(diagnosis["action_guide"]["recommended"], list)
    assert isinstance(diagnosis["action_guide"]["further_investigation"], list)

    # 현재 팀 contract 기준으로 diagnosis_trace는 state에 두지 않는다.
    assert "diagnosis_trace" not in final_state


# -----------------------------------------------------------------------------
# Smoke integration test
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("case", REAL_LOG_PIPELINE_CASES, ids=lambda case: f"case{case['case_id']}")
def test_pipeline_smoke_real_log_regression(monkeypatch, case):
    """
    Node 1 -> Node 2 -> Node 3 -> Node 4를 직접 이어 붙이는 얇은 smoke test.

    목적:
    - 각 노드의 세부 정답성보다, 노드 사이 인터페이스가 실제 구현과 맞물리는지 확인한다.
    - graph.py가 비어 있어도 파이프라인 연결 문제를 조기에 잡는다.

    이 테스트가 확인하는 것:
    1. Node 1이 실제 raw_log에서 parsed_fields를 만든다.
    2. Node 2의 classification이 Node 3에서 소비 가능한 구조다.
    3. Node 3의 tool_results가 Node 4 프롬프트까지 전달된다.
    4. Node 4가 현재 팀 contract에 맞게 diagnosis만 state에 저장한다.
    """
    # -------------------------------------------------------------------------
    # 0) 초기 상태 준비
    # -------------------------------------------------------------------------
    state = create_initial_state(read_log_from_subdir(case["filename"]))
    dump_pipeline_state("initial state", state)

    # -------------------------------------------------------------------------
    # 1) Node 2 mock 설정
    # -------------------------------------------------------------------------
    FakeNode2ChatPromptTemplate.payload = None

    monkeypatch.setattr(node2_module, "build_node2_classifier_llm", lambda: DummyNode2LLM())
    monkeypatch.setattr(node2_module, "ChatPromptTemplate", FakeNode2ChatPromptTemplate)
    monkeypatch.setattr(node2_module, "load_prompt", lambda filename: "dummy prompt")

    # -------------------------------------------------------------------------
    # 2) Node 3 mock 설정
    # -------------------------------------------------------------------------
    # node_3_executor.py는 import된 함수 이름 자체를 호출하므로,
    # node3_module의 심볼을 직접 monkeypatch 해야 한다.
    def fake_memory_calculator(parsed_fields: dict) -> dict:
        process_table = parsed_fields.get("process_table", [])
        total_ram_kb = max((parsed_fields.get("total_ram_pages") or 1) * 4, 1)
        swap_total_kb = parsed_fields.get("swap_total_kb")

        return {
            "top5_processes": [
                {
                    "name": p["name"],
                    "pid": p["pid"],
                    "rss_kb": p["rss_kb"],
                    "rss_mb": round(p["rss_kb"] / 1024, 1),
                    "ram_ratio": round(p["rss_kb"] / total_ram_kb, 3),
                    "oom_score_adj": p["oom_score_adj"],
                    "oom_protected": p["oom_score_adj"] == -1000,
                }
                for p in process_table[:5]
            ],
            "top5_total_ratio": round(
                sum(p["rss_kb"] for p in process_table[:5]) / total_ram_kb,
                3,
            ),
            "total_ram_mb": round(total_ram_kb / 1024, 1),
            "swap_configured": isinstance(swap_total_kb, (int, float)) and swap_total_kb > 0,
            "free_below_min": (
                parsed_fields.get("node_free_kb") is not None
                and parsed_fields.get("node_min_kb") is not None
                and parsed_fields.get("node_free_kb") < parsed_fields.get("node_min_kb")
            ),
            "protected_procs": [
                p["name"] for p in process_table if p.get("oom_score_adj") == -1000
            ],
            "sentinel_memory_note": "MEMORY-SIGNAL-SMOKE",
        }

    def fake_kernel_version_check(kernel_version: str) -> dict:
        return {
            "kernel_version": kernel_version,
            "known_issues": [
                "BUG-SMOKE-001: test-only known issue for integration smoke test"
            ],
            "has_known_issues": True,
        }

    def fake_kernel_param_recommender(oom_type: str, parsed_fields: dict) -> dict:
        return {
            "oom_type": oom_type,
            "recommendations": [
                {
                    "name": "vm.overcommit_memory",
                    "recommendation": "2로 설정 검토",
                    "command": "sysctl -w vm.overcommit_memory=2",
                }
            ],
            "observed_constraint": parsed_fields.get("constraint"),
        }

    # 중요:
    # 현재 Node 3 계약에서는 search_kb가
    #   search_kb(oom_type=..., parsed_fields=..., collection=None)
    # 형태로 호출되고,
    # 반환값은 query_used/total_found/chunks를 포함한 dict다.
    kb_call = {
        "called": 0,
        "oom_type": None,
        "parsed_fields": None,
        "collection": None,
    }

    def fake_search_kb(oom_type, parsed_fields, collection=None):
        kb_call["called"] += 1
        kb_call["oom_type"] = oom_type
        kb_call["parsed_fields"] = parsed_fields
        kb_call["collection"] = collection

        return {
            "query_used": "global_oom CONSTRAINT_NONE no swap space",
            "chunks": [
                {
                    "chunk_id": "chunk_smoke_001",
                    "content": "KB-SMOKE-001: Swap 미설정 환경에서는 global OOM이 쉽게 발생할 수 있다.",
                    "score": 0.07,
                    "metadata": {
                        "error_category": "global_oom",
                        "source": "smoke_kb_doc",
                    },
                }
            ],
            "total_found": 1,
        }

    monkeypatch.setattr(node3_module, "memory_calculator", fake_memory_calculator)
    monkeypatch.setattr(node3_module, "kernel_version_check", fake_kernel_version_check)
    monkeypatch.setattr(node3_module, "kernel_param_recommender", fake_kernel_param_recommender)
    monkeypatch.setattr(node3_module, "search_kb", fake_search_kb)

    # -------------------------------------------------------------------------
    # 3) Node 1 실행
    # -------------------------------------------------------------------------
    state = node1_module.node_1_parser(state)
    dump_pipeline_state("after node_1_parser", state)

    assert_node_1_shape(state)

    parsed = state["parsed_fields"]
    assert parsed["trigger_process"] == case["expected_trigger_process"]
    assert parsed["killed_process"] == case["expected_killed_process"]
    assert parsed["constraint"] == case["expected_constraint"]

    # -------------------------------------------------------------------------
    # 4) Node 2 실행
    # -------------------------------------------------------------------------
    state = node2_module.node_2_classifier(state)
    dump_pipeline_state("after node_2_classifier", state)

    assert_node_2_shape(state)

    classification = state["classification"]
    expected_classification = build_fake_node2_payload(parsed)
    assert classification["oom_type"] == case["expected_oom_type"]
    assert classification["tools_needed"] == expected_classification["tools_needed"]
    assert classification["needs_kb"] == expected_classification["needs_kb"]
    assert classification["confidence"] == "high"

    # -------------------------------------------------------------------------
    # 5) Node 3 실행
    # -------------------------------------------------------------------------
    state = node3_module.node_3_executor(state)
    dump_pipeline_state("after node_3_executor", state)

    assert_node_3_shape(state)

    tool_results = state["tool_results"]
    if "memory_calculator" in classification["tools_needed"]:
        assert "memory" in tool_results
        assert tool_results["memory"]["sentinel_memory_note"] == "MEMORY-SIGNAL-SMOKE"
    else:
        assert "memory" not in tool_results

    if "kernel_version_check" in classification["tools_needed"]:
        assert "kernel_bugs" in tool_results
        assert tool_results["kernel_bugs"]["has_known_issues"] is True
    else:
        assert "kernel_bugs" not in tool_results

    assert "kernel_params" in tool_results
    if classification["needs_kb"]:
        assert "kb_chunks" in tool_results
    else:
        assert "kb_chunks" not in tool_results

    # Node 3의 각 tool 결과가 잘 실렸는지 확인
    assert tool_results["kernel_params"]["oom_type"] == classification["oom_type"]

    # Node 3가 현재 search_kb 계약으로 호출했는지 확인
    if classification["needs_kb"]:
        assert kb_call["called"] == 1
        assert kb_call["oom_type"] == classification["oom_type"]
        assert kb_call["parsed_fields"]["killed_process"] == case["expected_killed_process"]
        assert kb_call["collection"] is None

        # Node 3가 search_kb 반환 dict를 그대로 저장하는지 확인
        assert tool_results["kb_chunks"]["query_used"] == "global_oom CONSTRAINT_NONE no swap space"
        assert tool_results["kb_chunks"]["total_found"] == 1
        assert len(tool_results["kb_chunks"]["chunks"]) == 1
        assert tool_results["kb_chunks"]["chunks"][0]["chunk_id"] == "chunk_smoke_001"
        assert tool_results["kb_chunks"]["chunks"][0]["metadata"]["source"] == "smoke_kb_doc"

    # -------------------------------------------------------------------------
    # 6) Node 4 실행
    # -------------------------------------------------------------------------
    # node_4_synthesizer는 state["llm"]이 있으면 그것을 사용하므로,
    # 여기서 더미 LLM을 주입한다.
    state["llm"] = DummyNode4LLM()

    state = node4_module.node_4_synthesizer(state)
    dump_pipeline_state("after node_4_synthesizer", state)

    # -------------------------------------------------------------------------
    # 7) 최종 smoke assertions
    # -------------------------------------------------------------------------
    assert_final_diagnosis_shape(state)

    final_diag = state["diagnosis"]

    combined_text = " ".join(
        [final_diag["log_analysis"]["summary"]]
        + final_diag["diagnosis"]["contributing_factors"]
        + final_diag["diagnosis"]["evidence"]
        + final_diag["action_guide"]["recommended"]
        + final_diag["action_guide"]["further_investigation"]
    )

    assert final_diag["diagnosis"]["severity"] in {"high", "medium", "low"}

    # Node 3의 fake tool 결과가 Node 4 prompt를 거쳐 최종 결과까지 이어졌는지 확인
    if "kernel_version_check" in classification["tools_needed"]:
        assert "BUG-SMOKE-001" in combined_text

    if classification["needs_kb"]:
        assert "KB-SMOKE-001" in combined_text

    if "memory_calculator" in classification["tools_needed"]:
        assert "MEMORY-SIGNAL-SMOKE" in combined_text

    # Node 4 더미 LLM이 실제로 한 번 호출되었는지도 확인
    assert state["llm"].calls == 1
