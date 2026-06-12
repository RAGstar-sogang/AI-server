from collections.abc import Callable
from typing import Any

from app.agent.graph import create_initial_state
from app.agent.nodes.node_1_parser import node_1_parser
from app.agent.nodes.node_2_classifier import (
    DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    node_2_classifier,
)
from app.agent.nodes.node_3_executor import node_3_executor
from app.agent.nodes.node_4_synthesizer import node_4_synthesizer
from app.core.llm_factory import build_chat_llm, build_node2_classifier_embeddings
from app.core.settings import get_settings
from app.core.vllm_manager import ensure_vllm_model


def build_rag_chat_llm(model_name: str):
    return build_chat_llm(model_name, json_mode=True)


def run_rag_agent(
    raw_log: str,
    *,
    label_similarity_threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    label_similarity_margin: float = DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
    model_name: str | None = None,
    llm_name: str | None = None,
    llm: Any | None = None,
    label_embeddings: Any | None = None,
    metadata: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
    preserve_partial_on_error: bool = True,
    create_state_fn: Callable[..., dict[str, Any]] = create_initial_state,
    embeddings_factory: Callable[[], Any] = build_node2_classifier_embeddings,
    chat_llm_factory: Callable[[str], Any] = build_rag_chat_llm,
    node_1: Callable[[dict[str, Any]], dict[str, Any]] = node_1_parser,
    node_2: Callable[[dict[str, Any]], dict[str, Any]] = node_2_classifier,
    node_3: Callable[[dict[str, Any]], dict[str, Any]] = node_3_executor,
    node_4: Callable[[dict[str, Any]], dict[str, Any]] = node_4_synthesizer,
) -> dict[str, Any]:
    """
    Run the full RAGstar OOM diagnosis pipeline from a raw kernel log.

    This is the reusable application-level runner. It keeps the same sequential
    node flow as the LangGraph workflow, while allowing callers such as
    experiments, tests, or future FastAPI endpoints to inject model/embedding
    objects and request-specific state.
    """
    try:
        state = create_state_fn(raw_log=raw_log, metadata=metadata)
    except TypeError:
        state = create_state_fn(raw_log)

    if extra_state:
        state.update(extra_state)

    state["label_embeddings"] = label_embeddings if label_embeddings is not None else embeddings_factory()
    state["label_similarity_threshold"] = label_similarity_threshold
    state["label_similarity_margin"] = label_similarity_margin

    requested_model = model_name or llm_name
    if llm is None and get_settings().use_vllm:
        requested_model = ensure_vllm_model(requested_model or get_settings().default_chat_model)

    if llm is not None:
        state["llm"] = llm
    elif requested_model:
        state["llm"] = chat_llm_factory(requested_model)

    try:
        state = node_1(state)
        state = node_2(state)
        state = node_3(state)
        state = node_4(state)
        if isinstance(state, dict):
            return state
        return {"_error": "rag_agent_state_not_dict"}
    except Exception as exc:
        if not preserve_partial_on_error:
            raise
        return {
            **state,
            "_error": f"rag_agent_error: {exc}",
        }
