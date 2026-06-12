import os
from typing import Dict, Any
from app.agent.state import OOMState

from app.agent.tools.memory_calculator import memory_calculator
from app.agent.tools.kernel_version_check import kernel_version_check
from app.agent.tools.kernel_param_recommender import kernel_param_recommender
from app.agent.tools.search_kb import search_kb

def node_3_executor(state: OOMState) -> OOMState:
    """
    Node 3: Node 2(분류기)의 결과에 따라 필요한 도구를 순수 코드로 실행합니다.
    
    도구 목록:
      A. memory_calculator      — 프로세스별 RSS/RAM 비율 계산
      B. kernel_version_check   — 커널 버전별 알려진 OOM 버그 조회
      C. kernel_param_recommender — OOM 유형별 sysctl 파라미터 추천
      D. search_kb              — ChromaDB에서 관련 KB 청크 검색 (RAG)
    """
    # print("🛠️ Node 3 (Executor) 도구 실행 시작...")
    
    results: Dict[str, Any] = {}
    classification = state.get("classification", {})
    parsed = state.get("parsed_fields", {})

    tools_needed = classification.get("tools_needed", [])
    needs_kb = classification.get("needs_kb", False)
    oom_type = classification.get("oom_type", "unknown")

    # [도구 A] memory_calculator 실행
    if "memory_calculator" in tools_needed:
        #   print("   -> 🧮 memory_calculator 실행 중")
        results["memory"] = memory_calculator(parsed)

    # [도구 B] kernel_version_check 실행
    if "kernel_version_check" in tools_needed:
        # print("   -> 🐧 kernel_version_check 실행 중")
        # 파싱된 커널 버전을 꺼내서 전달
        kernel_version = parsed.get("kernel_version")
        results["kernel_bugs"] = kernel_version_check(kernel_version)

    # [도구 C] kernel_param_recommender 실행
    if "kernel_param_recommender" in tools_needed:
        # print("   -> ⚙️ kernel_param_recommender 실행 중")
        results["kernel_params"] = kernel_param_recommender(oom_type, parsed)

    # [도구 D] KB 검색 (ChromaDB) 실행
    if needs_kb:
        # print("   -> 📚 search_kb 실행 중 (RAG)")
        
        # # 1. Node 3가 맥락을 파악하여 쿼리(Query) 직접 조립
        # query_parts = [oom_type]
        # if parsed.get("constraint"):
        #     query_parts.append(str(parsed["constraint"]))
        # if parsed.get("cgroup_path"):
        #     query_parts.append("cgroup memory limit exceeded")
            
        # swap_total = parsed.get("swap_total_kb")
        # if swap_total == 0 or swap_total == "0":
        #     query_parts.append("no swap space")
            
        # order_val = parsed.get("order")
        # if order_val is not None:
        #     try:
        #         order_int = int(order_val)
        #         if order_int > 0:
        #             query_parts.append(f"order {order_int} page allocation failure")
        #     except ValueError:
        #         pass

        # query = " ".join(query_parts)
        # filter_dict = {"error_category": {"$in": [oom_type, "general"]}}

        # # 2. 순수해진 search_kb 함수 호출 (팀 요구사항 시그니처와 완벽 일치)
        # # TODO: 향후 FastAPI 연동 시 collection 주입
        # chunks = search_kb(query=query, top_k=5, filter_dict=filter_dict, collection=None)

        # # 3. 결과를 Node 4가 기대하는 딕셔너리 포맷으로 포장하여 저장
        # is_error = len(chunks) > 0 and "error" in chunks[0]
        # results["kb_chunks"] = {
        #     "query_used": query,
        #     "chunks": chunks,
        #     "total_found": 0 if is_error else len(chunks)
        # }

        """
        #   search_kb가 내부에서 쿼리 조립 + ChromaDB 검색을 전부 처리한다.
        #   - oom_type → 쿼리 첫 단어 + error_category 필터
        #   - parsed  → constraint, cgroup_path, swap, order 등으로 쿼리 보강
        #   - collection은 chromadb_client 싱글턴에서 자동 획득 (Ollama 임베딩)
        """
        results["kb_chunks"] = search_kb(oom_type=oom_type, parsed_fields=parsed)

    # print("✅ Node 3 도구 실행 완료!")

    # 결과를 v2.1 스키마에 맞게 tool_results 딕셔너리에 덮어씌움
    return {
        **state,
        "tool_results": results
    }
