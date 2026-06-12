# app/agent/nodes/node_2_classifier.py

import os
import json
from langchain_core.prompts import ChatPromptTemplate
from app.agent.state import OOMState
from app.core.llm_factory import (
    build_node2_classifier_embeddings,
    build_node2_classifier_llm,
)


OOM_TYPE_EMBEDDING_PROTOTYPES = {
    "global_oom": [
        "global oom host-wide memory exhaustion system out of memory",
        "system-wide oom caused by total memory pressure outside a cgroup limit",
    ],
    "cgroup_oom": [
        "cgroup oom container memory limit exceeded memcg out of memory",
        "memory cgroup limit hit and kernel killed a process inside the cgroup",
    ],
    "swap_exhaustion": [
        "swap exhaustion swap space fully used memory plus swap limit reached",
        "oom caused by exhausted swap capacity after ram and swap were consumed",
    ],
    "page_alloc_failure": [
        "page allocation failure high order allocation cannot allocate contiguous pages",
        "kernel page allocator failed to satisfy requested order allocation",
    ],
}

OOM_TYPE_ALIASES = {
    "global_oom": "global_oom",
    "global oom": "global_oom",
    "host oom": "global_oom",
    "host out of memory": "global_oom",
    "system oom": "global_oom",
    "system out of memory": "global_oom",
    "system-wide oom": "global_oom",
    "cgroup_oom": "cgroup_oom",
    "cgroup oom": "cgroup_oom",
    "container oom": "cgroup_oom",
    "memcg oom": "cgroup_oom",
    "memory cgroup oom": "cgroup_oom",
    "swap_exhaustion": "swap_exhaustion",
    "swap exhaustion": "swap_exhaustion",
    "swap full": "swap_exhaustion",
    "swap depleted": "swap_exhaustion",
    "page_alloc_failure": "page_alloc_failure",
    "page alloc failure": "page_alloc_failure",
    "page allocation failure": "page_alloc_failure",
    "high order allocation failure": "page_alloc_failure",
}

DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD = 0.62
DEFAULT_EMBEDDING_SIMILARITY_MARGIN = 0.08


def _canonicalize_oom_type_label(candidate: object) -> str | None:
    if candidate is None:
        return None

    normalized = str(candidate).strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    return OOM_TYPE_ALIASES.get(normalized)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return -1.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot / (left_norm * right_norm)


def inspect_embedding_oom_type(
    candidate: object,
    embeddings: object | None,
    similarity_threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    similarity_margin: float = DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "candidate_text": "" if candidate is None else str(candidate).strip(),
        "canonical_label": None,
        "accepted_label": None,
        "best_label": None,
        "best_score": -1.0,
        "second_label": None,
        "second_score": -1.0,
        "score_margin": -1.0,
        "similarity_threshold": similarity_threshold,
        "similarity_margin": similarity_margin,
        "passes_threshold": False,
        "passes_margin": False,
        "normalization_source": "unresolved",
    }

    canonical = _canonicalize_oom_type_label(candidate)
    if canonical:
        diagnostics["canonical_label"] = canonical
        diagnostics["accepted_label"] = canonical
        diagnostics["best_label"] = canonical
        diagnostics["best_score"] = 1.0
        diagnostics["second_score"] = 0.0
        diagnostics["score_margin"] = 1.0
        diagnostics["passes_threshold"] = True
        diagnostics["passes_margin"] = True
        diagnostics["normalization_source"] = "canonical_alias"
        return diagnostics

    if candidate is None or embeddings is None:
        diagnostics["normalization_source"] = "missing_embeddings" if embeddings is None else "missing_candidate"
        return diagnostics

    candidate_text = str(candidate).strip()
    if not candidate_text:
        diagnostics["normalization_source"] = "empty_candidate"
        return diagnostics

    label_texts: list[str] = []
    label_lookup: list[str] = []
    for label, prototypes in OOM_TYPE_EMBEDDING_PROTOTYPES.items():
        for prototype in prototypes:
            label_texts.append(prototype)
            label_lookup.append(label)

    try:
        if hasattr(embeddings, "embed_query"):
            candidate_vector = embeddings.embed_query(candidate_text)
        elif hasattr(embeddings, "embed_documents"):
            candidate_vector = embeddings.embed_documents([candidate_text])[0]
        else:
            diagnostics["normalization_source"] = "unsupported_embeddings"
            return diagnostics

        if hasattr(embeddings, "embed_documents"):
            label_vectors = embeddings.embed_documents(label_texts)
        else:
            label_vectors = [embeddings.embed_query(text) for text in label_texts]
    except Exception:
        diagnostics["normalization_source"] = "embedding_error"
        return diagnostics

    label_scores: dict[str, float] = {}
    for label, vector in zip(label_lookup, label_vectors):
        score = _cosine_similarity(candidate_vector, vector)
        label_scores[label] = max(label_scores.get(label, -1.0), score)

    ranked_scores = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked_scores:
        diagnostics["normalization_source"] = "no_ranked_scores"
        return diagnostics

    best_label, best_score = ranked_scores[0]
    second_label = ranked_scores[1][0] if len(ranked_scores) > 1 else None
    second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else -1.0
    score_margin = best_score - second_score

    diagnostics["best_label"] = best_label
    diagnostics["best_score"] = best_score
    diagnostics["second_label"] = second_label
    diagnostics["second_score"] = second_score
    diagnostics["score_margin"] = score_margin

    passes_threshold = best_score >= similarity_threshold
    passes_margin = score_margin >= similarity_margin
    diagnostics["passes_threshold"] = passes_threshold
    diagnostics["passes_margin"] = passes_margin

    if not passes_threshold:
        diagnostics["normalization_source"] = "below_threshold"
        return diagnostics

    if not passes_margin:
        diagnostics["normalization_source"] = "below_margin"
        return diagnostics

    diagnostics["accepted_label"] = best_label
    diagnostics["normalization_source"] = "embedding_match"
    return diagnostics


def embedding_normalize_oom_type(
    candidate: object,
    embeddings: object | None,
    similarity_threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    similarity_margin: float = DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
) -> str | None:
    diagnostics = inspect_embedding_oom_type(
        candidate,
        embeddings,
        similarity_threshold=similarity_threshold,
        similarity_margin=similarity_margin,
    )
    accepted_label = diagnostics.get("accepted_label")
    return str(accepted_label) if accepted_label else None

def load_prompt(filename: str) -> str:
    """prompts 폴더에서 텍스트 파일을 읽어옵니다."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    # nodes -> agent -> prompts 순서로 경로 구성
    prompt_path = os.path.join(base_path, "..", "prompts", filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
    
def get_available_tools_desc(parsed_fields: dict) -> str:
    """
    하드 제약 로직: 데이터 존재 여부에 따라 LLM에게 제안할 도구 목록을 동적으로 생성합니다.
    """
    available = []

    # memory_calculator: 프로세스 테이블이 2개 이상이어야 의미 있음
    if len(parsed_fields.get("process_table", [])) >= 2:
        available.append("memory_calculator")

    # kernel_version_check: 커널 버전 정보가 있어야 조회 가능
    if parsed_fields.get("kernel_version"):
        available.append("kernel_version_check")

    # kernel_param_recommender: 항상 후보에 포함
    available.append("kernel_param_recommender")

    return "\n".join(available)

def deterministic_oom_classification(parsed_fields: dict) -> str:
    """
    하드코딩으로 OOM 유형을 분류하고 모르는 경우 LLM에게 넘깁니다.
    현재 파싱된 필드만을 봐야하는 상황에서, 몇 가지 명백한 패턴이 있다면 이를 우선적으로 처리하는 방어적 fallback 로직입니다.
    """
    raw_constraint = parsed_fields.get("constraint")
    constraint = str(raw_constraint).upper() if raw_constraint is not None else ""
    order = parsed_fields.get("order", 0)
    
    def _safe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    swap_total = _safe_int(parsed_fields.get("swap_total_kb"))
    swap_free = _safe_int(parsed_fields.get("swap_free_kb"))
    cgroup_swap_usage = _safe_int(parsed_fields.get("cgroup_swap_usage_kb"))
    cgroup_swap_limit = _safe_int(parsed_fields.get("cgroup_swap_limit_kb"))
    
    # 1. Page Allocation Failure (order가 0보다 큼)
    try:
        if int(order) > 0:
            return "page_alloc_failure"
    except (ValueError, TypeError):
        pass
    
    # 2. Global OOM (제한이 없는데 죽은 경우)
    # global_oom은 swap이 바닥났더라도 우선 시스템 전체 OOM으로 본다.
    if "NONE" in constraint:
        return "global_oom"

    # 3. Swap Exhaustion
    # cgroup 안에서 RAM+swap이 모두 한계에 도달한 경우는 cgroup_oom보다 별도 유형으로 우선 분류한다.
    if (
        "MEMCG" in constraint
        and cgroup_swap_limit is not None
        and cgroup_swap_limit > 0
        and cgroup_swap_usage == cgroup_swap_limit
    ):
        return "swap_exhaustion"

    # 4. Cgroup OOM
    if "MEMCG" in constraint or parsed_fields.get("cgroup_path"):
        return "cgroup_oom"
        
    return "unknown"

def node_2_classifier(state: OOMState) -> OOMState:
    """
    LLM을 통해 OOM 유형을 분류하고 필요한 도구를 선정합니다.
    """
    # 1. Ollama 모델 설정 (설정 파일 기반)
    llm = state.get("llm") or build_node2_classifier_llm()
    embeddings = state.get("label_embeddings")
    if embeddings is None:
        try:
            embeddings = build_node2_classifier_embeddings()
        except Exception:
            embeddings = None

    # 2. .txt 파일에서 프롬프트 로드
    system_prompt_text = load_prompt("node_2_template.txt")
    prompt_template = ChatPromptTemplate.from_template(system_prompt_text)

    # 3. 하드웨어 제약 기반 도구 목록 생성
    parsed = state["parsed_fields"]
    determined_type = deterministic_oom_classification(parsed)
    tools_desc = get_available_tools_desc(parsed)
    similarity_threshold = float(
        state.get("label_similarity_threshold", DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD)
    )
    similarity_margin = float(
        state.get("label_similarity_margin", DEFAULT_EMBEDDING_SIMILARITY_MARGIN)
    )

    # 4. 체인 생성 및 실행
    chain = prompt_template | llm

    # 유저 자유 텍스트 metadata (없으면 placeholder)
    user_metadata = (state.get("metadata_text") or "").strip() or "None provided."

    # Node 1에서 파싱된 결과를 문자열로 변환하여 주입
    try:
        response = chain.invoke({
            "available_tools_desc": tools_desc,
            "user_metadata": user_metadata,
            "parsed_fields": json.dumps(parsed, indent=2, ensure_ascii=False)
        })
        
        result = json.loads(response.content)

        llm_oom_type = result.get("oom_type", "unknown")
        embedding_debug = inspect_embedding_oom_type(
            llm_oom_type,
            embeddings,
            similarity_threshold=similarity_threshold,
            similarity_margin=similarity_margin,
        )
        normalized_llm_type = embedding_debug.get("accepted_label")
        
        # LLM이 엉뚱한 대답을 해도, 하드 로직이 성공했다면 하드 로직을 우선시함
        final_oom_type = determined_type if determined_type != "unknown" else (normalized_llm_type or "unknown")
        
        return {
        **state,
        "classification": {
            "oom_type": final_oom_type,
            "raw_llm_oom_type": llm_oom_type,
            "deterministic_oom_type": determined_type,
            "normalized_llm_oom_type": normalized_llm_type,
            "embedding_debug": embedding_debug,
            "tools_needed": result.get("tools_needed", []),
            "needs_kb": result.get("needs_kb", False),
            "confidence": "high" if determined_type != "unknown" else result.get("confidence", "low")
            }
        }
    except Exception as e:
        # 파싱 에러 시에도 하드 로직으로 분류한 type은 살려감
        return {
        **state,
        "classification": {
            "oom_type": determined_type,
            "tools_needed": ["kernel_param_recommender"], # 기본 도구 할당
            "needs_kb": True,
            "confidence": "low"
        },
        "error": f"Node 2 Error: {str(e)}"
    }