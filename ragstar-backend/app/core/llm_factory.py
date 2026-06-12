from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings

from app.core.settings import get_settings


def _build_chat_vllm(model: str, *, json_mode: bool = False) -> ChatOpenAI:
    settings = get_settings()
    model_kwargs = {}
    if json_mode and settings.vllm_json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        model=model,
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        temperature=settings.vllm_temperature,
        timeout=settings.vllm_timeout,
        max_retries=settings.vllm_max_retries,
        model_kwargs=model_kwargs,
    )


def build_chat_llm(model: str, *, json_mode: bool = False):
    return _build_chat_vllm(model, json_mode=json_mode)


def build_chat_ollama(model: str, *, json_mode: bool = False):
    """
    Backward-compatible factory name.

    Existing experiment code imports build_chat_ollama directly. Keep that API
    stable while routing chat generation through vLLM only.
    """
    return build_chat_llm(model, json_mode=json_mode)


def build_node2_classifier_llm():
    settings = get_settings()
    return build_chat_llm(settings.node2_model, json_mode=True)


def build_node2_classifier_embeddings() -> OllamaEmbeddings:
    return build_node4_live_stability_embeddings()


def build_node4_synthesizer_llm():
    settings = get_settings()
    return build_chat_llm(settings.node4_model, json_mode=True)


def build_exp2_base_llm(model_name: str | None = None):
    settings = get_settings()
    model = model_name or settings.exp2_base_model
    return _build_chat_vllm(model, json_mode=False)


def build_node4_live_stability_embeddings() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def build_exp2_embeddings() -> OllamaEmbeddings:
    return build_node4_live_stability_embeddings()
