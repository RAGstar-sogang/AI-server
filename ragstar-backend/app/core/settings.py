from pathlib import Path
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_api_key: str = "EMPTY"
    vllm_default_chat_model: str = "qwen3.5-9b"
    vllm_node2_model: str | None = None
    vllm_node4_model: str | None = None
    vllm_exp2_base_model: str | None = None
    vllm_temperature: float = 0.0
    vllm_timeout: float = 120.0
    vllm_max_retries: int = 2
    vllm_json_mode: bool = True
    vllm_auto_switch_model: bool = True
    vllm_manage_script: str = "/workspace/vllm/manage.sh"
    vllm_serve_env: str = "/workspace/vllm/serve.env"
    vllm_model_profiles: str = "/workspace/vllm/model_profiles.json"
    vllm_models_root: str = "/workspace/vllm/models"
    vllm_switch_timeout_seconds: float = 240.0

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    chroma_collection_name: str = "oom_kb"

    @property
    def use_vllm(self) -> bool:
        return True

    @property
    def chat_base_url(self) -> str:
        return self.vllm_base_url

    @property
    def default_chat_model(self) -> str:
        return self.vllm_default_chat_model

    @property
    def node2_model(self) -> str:
        return self.vllm_node2_model or self.vllm_default_chat_model

    @property
    def node4_model(self) -> str:
        return self.vllm_node4_model or self.vllm_default_chat_model

    @property
    def exp2_base_model(self) -> str:
        return self.vllm_exp2_base_model or self.vllm_default_chat_model

    @property
    def ollama_embedding_api_url(self) -> str:
        parts = urlsplit(self.ollama_base_url)
        base_path = parts.path.rstrip("/")
        embedding_path = f"{base_path}/api/embeddings" if base_path else "/api/embeddings"
        return urlunsplit((parts.scheme, parts.netloc, embedding_path, "", ""))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
