import json
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from app.agent.state import OOMState
from app.agent.nodes.node_1_parser import node_1_parser
from app.agent.nodes.node_2_classifier import node_2_classifier
from app.agent.nodes.node_3_executor import node_3_executor
from app.agent.nodes.node_4_synthesizer import node_4_synthesizer


# -----------------------------------------------------------------------------
# Graph routing helpers
# -----------------------------------------------------------------------------
# 현재 구현 기준에서의 핵심 포인트는 다음과 같다.
#
# 1. Node 4는 fail-fast 설계다.
#    - fallback / retry 없이 한 번만 시도한다.
#    - JSON이 깨졌거나 스키마가 틀리면 예외를 그대로 올린다.
#
# 2. Node 2는 '방어적 fallback classification'을 제공한다.
#    - LLM 응답 JSON 파싱이 실패해도
#      classification = {
#          "oom_type": "unknown",
#          "tools_needed": [],
#          "needs_kb": True,
#          "confidence": "low"
#        }
#      형태를 채우고 error 문자열도 함께 기록할 수 있다.
#    - 따라서 Node 2 이후에는 error가 있더라도 classification이 존재하면
#      Node 3 / Node 4로 계속 흘려 보내는 편이 현재 설계 의도에 더 맞다.
#
# 3. Node 1과 Node 3은 현재 기준으로 사실상 직진 노드다.
#    - Node 1은 parsed_fields를 채워 반환하는 파서 역할이다.
#    - Node 3은 tool_results를 준비해 Node 4에 넘기는 실행 노드다.
#    - 따라서 두 구간은 conditional edge 대신 일반 edge로 단순화한다.
#
# 이 graph.py는 위 정책을 반영해,
# 기본 흐름은 1 -> 2 -> 3 -> 4를 유지하되
# 실제 의미 있는 분기는 Node 2 이후에만 남긴다.
# -----------------------------------------------------------------------------
"""
def _route_after_node_2(state: OOMState) -> str:
    
    Node 2 이후 분기 함수.

    현재 정책:
    - Node 2는 분류 실패 시에도 fallback classification을 채우도록 설계되어 있다.
    - 따라서 error 문자열이 있더라도 classification이 존재하면 Node 3으로 진행한다.
    - 즉, Node 2 이후에는 'error 존재 여부'보다 'classification이 준비되었는지'가 더 중요하다.

    종료가 필요한 경우:
    - classification 자체가 비어 있거나 dict가 아닌 비정상 상태인 경우만 END
    추가) 이미 node2에서 방어로직이 짜여있으므로 함수 제외
    
    classification = state.get("classification")
    if not isinstance(classification, dict) or not classification:
        return "end"
    return "node_3_executor"
"""


# -----------------------------------------------------------------------------
# Graph factory
# -----------------------------------------------------------------------------
# 이 프로젝트에서는 graph 객체를 import 해서 바로 쓸 수도 있고,
# 테스트/서버 코드에서 새로 생성할 수도 있다.
# 둘 다 지원하기 위해 build_oom_graph()와 compiled graph singleton을 함께 둔다.
# -----------------------------------------------------------------------------
def build_oom_graph():
    """
    OOM 진단 LangGraph 워크플로를 생성하고 compile 한다.

    흐름:
        START
          -> Node 1: parser
          -> Node 2: classifier
          -> Node 3: executor
          -> Node 4: synthesizer
          -> END

    분기 정책:
    - 기본 흐름은 1 -> 2 -> 3 -> 4
    - Node 1 -> Node 2는 일반 edge
    - Node 3 -> Node 4도 일반 edge
    - 실제 의미 있는 조건부 분기는 Node 2 이후에만 남긴다
    """
    workflow = StateGraph(OOMState)

    # -------------------------------------------------------------------------
    # 노드 등록
    # -------------------------------------------------------------------------
    workflow.add_node("node_1_parser", node_1_parser)
    workflow.add_node("node_2_classifier", node_2_classifier)
    workflow.add_node("node_3_executor", node_3_executor)
    workflow.add_node("node_4_synthesizer", node_4_synthesizer)

    # -------------------------------------------------------------------------
    # 시작점 설정
    # -------------------------------------------------------------------------
    workflow.set_entry_point("node_1_parser")

    # -------------------------------------------------------------------------
    # 엣지 등록
    # -------------------------------------------------------------------------
    # Node 1 -> Node 2는 현재 구현 기준으로 사실상 직진이므로 일반 edge를 사용한다.
    workflow.add_edge("node_1_parser", "node_2_classifier")

    # Node 2 -> Node 3
    workflow.add_edge("node_2_classifier", "node_3_executor")

    # Node 3 -> Node 4도 현재 구현 기준으로 직진이므로 일반 edge를 사용한다.
    workflow.add_edge("node_3_executor", "node_4_synthesizer")

    # Node 4 -> END
    workflow.add_edge("node_4_synthesizer", END)

    # compile 된 runnable graph 반환
    return workflow.compile()


# -----------------------------------------------------------------------------
# Compiled graph singleton
# -----------------------------------------------------------------------------
# 다른 모듈에서 간단히 import 해서 바로 쓸 수 있도록
# 모듈 로드 시점에 compile 한 graph 객체를 제공한다.
# -----------------------------------------------------------------------------
oom_graph = build_oom_graph()


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------
# FastAPI, CLI, 테스트 코드에서 공통으로 쓰기 쉽게
# 초기 state 생성 함수와 invoke 래퍼를 함께 둔다.
# -----------------------------------------------------------------------------
_TEXT_METADATA_KEYS = ("user_note", "text", "description", "free_text", "note", "context")


def _derive_metadata_text(metadata: Any) -> str:
    """Convert user-provided metadata into a plain-text snippet for prompts.

    Accepts:
    - None / empty -> ""
    - str         -> stripped string
    - dict        -> first non-empty value among _TEXT_METADATA_KEYS, else
                     a compact "key=value; key=value" rendering of the dict
    - other       -> str(value)
    """
    if metadata is None:
        return ""
    if isinstance(metadata, str):
        return metadata.strip()
    if isinstance(metadata, dict):
        for k in _TEXT_METADATA_KEYS:
            v = metadata.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        parts = []
        for k, v in metadata.items():
            if v is None or v == "":
                continue
            parts.append(f"{k}={v}")
        return "; ".join(parts)
    return str(metadata).strip()


def create_initial_state(
    raw_log: str,
    metadata: Dict[str, Any] | str | None = None,
) -> OOMState:
    """
    워크플로 시작용 기본 상태를 생성한다.

    필드 설명:
    - raw_log: 사용자 입력 원문 로그
    - metadata: dict (server_info/service/recent_changes 등 구조화 입력)
                또는 str (유저가 직접 적어 보낸 자유 텍스트)
                또는 None (없음)

    구조화 metadata는 그대로 state["metadata"]에 저장되어 Node 1의 fallback
    필드 (kernel_version 등) 로직에서 사용된다. 거기서 추출 가능한
    텍스트 + 직접 전달된 string은 state["metadata_text"]에 저장되어
    Node 2/4 프롬프트에 주입된다 (비어 있으면 주입 생략).
    """
    if isinstance(metadata, str):
        metadata_dict: Dict[str, Any] = {}
    else:
        metadata_dict = dict(metadata or {})
    return {
        "raw_log": raw_log,
        "metadata": metadata_dict,
        "metadata_text": _derive_metadata_text(metadata),
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }



def invoke_oom_workflow(
    raw_log: str,
    metadata: Dict[str, Any] | str | None = None,
    extra_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    raw_log 하나로 전체 OOM 워크플로를 실행하는 편의 함수.

    사용 예:
        result = invoke_oom_workflow(raw_log)

    extra_state 용도:
    - 테스트에서 state["llm"] 주입
    - 향후 collection / request metadata / trace id 같은 추가 상태 주입

    반환값:
    - 최종 graph state 전체를 dict 형태로 반환한다.
    """
    state = create_initial_state(raw_log=raw_log, metadata=metadata)
    if extra_state:
        state.update(extra_state)

    return oom_graph.invoke(state)



def stream_oom_workflow(
    raw_log: str,
    metadata: Dict[str, Any] | str | None = None,
    extra_state: Dict[str, Any] | None = None,
):
    """
    LangGraph의 stream 인터페이스를 이용해 단계별 이벤트를 받고 싶을 때 사용하는 래퍼.

    사용 예:
        for event in stream_oom_workflow(raw_log):
            print(event)
    """
    state = create_initial_state(raw_log=raw_log, metadata=metadata)

    if extra_state:
        state.update(extra_state)

    return oom_graph.stream(state)


# -----------------------------------------------------------------------------
# Optional manual smoke entrypoint
# -----------------------------------------------------------------------------
# 직접 python app/agent/graph.py 형태로 실행했을 때
# 최소 예제로 동작 확인을 해볼 수 있도록 간단한 main 블록을 둔다.
#
# 주의:
# - 이 블록은 실제 Ollama / Node 4 LLM / 도구 환경이 준비되어 있어야 동작한다.
# - 테스트 환경에서는 tests/test_all_smoke.py 같은 별도 테스트를 사용하는 편이 낫다.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample_log = """
    [11686.040460] httpd invoked oom-killer: gfp_mask=0x280da, order=0, oom_score_adj=0
    [11686.040466] CPU: 2 PID: 3244 Comm: httpd Not tainted 4.18.0-305.el8.x86_64 #1
    [11686.040500] Node 0 Normal free:7296kB min:7360kB low:9200kB high:11040kB
    [11686.040515] Swap:  SwapTotal:       0 kB   SwapFree:        0 kB
    [11686.040523] 524288 pages RAM
    [11686.040530] Tasks state (memory values in pages):
    [11686.040532] [  pid  ]   uid  tgid total_vm      rss pgtables_bytes swapents oom_score_adj name
    [11686.040549] [   3201]     0  3201    68432   218920  1789952        0             0 java
    [11686.040552] [   3244]    48  3244    65120    52340   430080        0             0 httpd
    [11686.040567] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,
                   global_oom,task_memcg=/,task=java,pid=3201,uid=0,pgtables=1789952,score=836
    [11686.040575] Out of memory: Killed process 3201 (java) total-vm:273728kB, anon-rss:875680kB, file-rss:0kB, shmem-rss:0kB, UID:0
    """.strip()

    final_state = invoke_oom_workflow(sample_log)
    print(json.dumps(final_state, indent=2, ensure_ascii=False))