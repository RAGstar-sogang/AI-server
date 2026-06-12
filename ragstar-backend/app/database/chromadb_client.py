"""
ChromaDB 클라이언트 싱글턴

인덱싱(kb_chunks_to_chromaDB.py)과 동일한 Ollama 임베딩 함수를 사용하여
쿼리 벡터와 저장된 벡터의 공간을 일치시킨다.

사용:
    from app.database.chromadb_client import get_collection
    collection = get_collection()
"""

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pathlib import Path

from app.core.settings import get_settings

# ── 설정 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# ── 싱글턴 ────────────────────────────────────────────────────
_collection = None


def get_collection():
    """oom_kb 컬렉션 반환. 최초 호출 시 연결, 이후 재사용."""
    global _collection
    settings = get_settings()

    if _collection is None:
        ef = OllamaEmbeddingFunction(
            url=settings.ollama_embedding_api_url,
            model_name=settings.ollama_embedding_model,
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        _collection = client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
        )

    return _collection