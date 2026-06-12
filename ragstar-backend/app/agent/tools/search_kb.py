# from typing import Dict, Any, Optional, List

# def search_kb(query: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None, collection: Optional[Any] = None) -> List[Dict[str, Any]]:
#     """
#     ChromaDB에서 OOM 유형 및 로그 특징과 관련된 지식베이스(KB) 청크를 검색합니다.
#     """
#     #  아직 DB가 연결되지 않은 로컬 테스트 환경을 위한 처리
#     if collection is None:
#         return [{"error": "ChromaDB collection 객체가 주입되지 않았습니다."}]

#     try:
#         # 필터 조건이 있으면 where 파라미터에 추가
#         query_kwargs = {
#             "query_texts": [query],
#             "n_results": top_k
#         }
#         if filter_dict:
#             query_kwargs["where"] = filter_dict

#         results = collection.query(**query_kwargs)

#         chunks = []
#         if results and "documents" in results and results["documents"]:
#             for i, doc in enumerate(results["documents"][0]):
#                 chunks.append({
#                     "chunk_id": results["ids"][0][i] if "ids" in results else f"chunk_{i}",
#                     "content": doc,
#                     "score": results["distances"][0][i] if "distances" in results else 0.0,
#                     "metadata": results["metadatas"][0][i] if "metadatas" in results else {}
#                 })

#         return chunks

#     except Exception as e:
#         return [{"error": f"지식베이스 검색 중 오류 발생: {str(e)}"}]

"""
도구: search_kb — ChromaDB에서 OOM 관련 KB 청크 검색

호출 시점: Node 3 (needs_kb == true일 때)
데이터 출처: ChromaDB oom_kb 컬렉션 (nomic-embed-text 임베딩)

사용:
    from app.agent.tools.search_kb import search_kb
    result = search_kb(oom_type, parsed_fields)
"""

from typing import Dict, Any, List, Optional
from app.database.chromadb_client import get_collection


def search_kb(
    oom_type: str,
    parsed_fields: Dict[str, Any],
    collection: Optional[Any] = None,
) -> Dict[str, Any]:
    """OOM 유형과 파싱 결과를 기반으로 KB 청크를 검색한다.

    Args:
        oom_type: 분류된 OOM 유형 (global_oom, cgroup_oom, swap_exhaustion, page_alloc_failure)
        parsed_fields: Node 1에서 파싱한 필드 dict
        collection: ChromaDB 컬렉션 (None이면 싱글턴 사용, 테스트 시 주입 가능)

    Returns:
        {
            "query_used": str,
            "chunks": [{"chunk_id", "content", "score", "metadata"}, ...],
            "total_found": int
        }
    """
    # ── 컬렉션 획득 ──
    if collection is None:
        collection = get_collection()

    # ── 1. 쿼리 구성 (parsed_fields 기반) ──
    query_parts = [oom_type]

    if parsed_fields.get("constraint"):
        query_parts.append(parsed_fields["constraint"])

    if parsed_fields.get("cgroup_path"):
        query_parts.append("cgroup memory limit")

    if str(parsed_fields.get("swap_total_kb")) == "0":
        query_parts.append("no swap space")

    order_val = parsed_fields.get("order")
    if order_val is not None:
        try:
            if int(order_val) > 0:
                query_parts.append(f"order {int(order_val)} page allocation")
        except ValueError:
            pass

    query = " ".join(query_parts)

    # ── 2. ChromaDB 검색 (벡터 유사도 + 메타데이터 필터) ──
    try:
        results = collection.query(
            query_texts=[query],
            n_results=5,
            where={"error_category": {"$in": [oom_type, "general"]}},
        )
    except Exception as e:
        return {
            "query_used": query,
            "chunks": [],
            "total_found": 0,
            "error": f"KB 검색 오류: {str(e)}",
        }

    # ── 3. 결과 정리 ──
    chunks: List[Dict[str, Any]] = []
    if results and results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "content": doc,
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })

    return {
        "query_used": query,
        "chunks": chunks,
        "total_found": len(chunks),
    }