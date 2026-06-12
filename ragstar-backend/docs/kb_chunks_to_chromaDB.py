#!/usr/bin/env python3
"""
kb_chunks.jsonl → ChromaDB 인덱싱

위치: docs/kb_chunks_to_chromaDB.py
DB:   chroma_db/  (프로젝트 루트 기준)

사용법:
    cd ragstar-backend
    python docs/kb_chunks_to_chromaDB.py --reset              # 전체 재구축
    python docs/kb_chunks_to_chromaDB.py --append             # 새 청크만 추가 (중복 스킵)
    python docs/kb_chunks_to_chromaDB.py --embed default      # MiniLM 임베딩 (테스트용)
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

import chromadb


# ══════════════════════════════════════════════════════════════
# 경로 설정 (프로젝트 루트 기준)
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # docs/ → ragstar-backend/
DEFAULT_INPUT = PROJECT_ROOT / "data" / "kb_chunks.jsonl"
DEFAULT_DB    = PROJECT_ROOT / "chroma_db"


# ══════════════════════════════════════════════════════════════
# 임베딩 함수
# ══════════════════════════════════════════════════════════════

def get_embedding_function(embed_type: str, model_name: str | None = None):
    if embed_type == "default":
        # ChromaDB 기본: all-MiniLM-L6-v2 (384차원)
        return None

    elif embed_type == "ollama":
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        model = model_name or "nomic-embed-text"
        return OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=model,
        )

    else:
        print(f"✗ 미지원 임베딩: {embed_type}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# 메타데이터 변환 (list → string)
# ══════════════════════════════════════════════════════════════

def chunk_to_chroma_metadata(chunk: dict) -> dict:
    """ChromaDB 호환 flat dict. (list 불가 → keywords를 csv string으로)"""
    meta = chunk["metadata"]
    return {
        "error_category": meta["error_category"],
        "keywords": ",".join(meta.get("keywords", [])),
        "doc_id": chunk["doc_id"],
        "chunk_index": chunk["chunk_index"],
        "title": chunk["title"],
    }


# ══════════════════════════════════════════════════════════════
# 배치 삽입
# ══════════════════════════════════════════════════════════════

BATCH_SIZE = 100

def batch_add(collection, ids, documents, metadatas):
    total = len(ids)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  → {end}/{total} 삽입 완료")


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main(input_path: Path, db_path: Path, reset: bool, append: bool,
         embed_type: str, embed_model: str | None):

    # ── 1. 로드 ──
    with open(input_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    print(f"[1/4] 청크 {len(chunks)}개 로드 ({input_path})")

    ids = [c["chunk_id"] for c in chunks]
    dupes = {k: v for k, v in Counter(ids).items() if v > 1}
    if dupes:
        print(f"  ✗ 입력 파일 내 중복 chunk_id: {dupes}")
        sys.exit(1)

    # ── 2. 컬렉션 준비 ──
    client = chromadb.PersistentClient(path=str(db_path))

    if reset:
        try:
            client.delete_collection("oom_kb")
            print("[2/4] 기존 oom_kb 삭제 (--reset)")
        except Exception:
            pass

    ef = get_embedding_function(embed_type, embed_model)
    create_kwargs = {
        "name": "oom_kb",
        "metadata": {"hnsw:space": "cosine"},
    }
    if ef is not None:
        create_kwargs["embedding_function"] = ef

    collection = client.get_or_create_collection(**create_kwargs)

    existing = collection.count()

    if existing > 0 and not reset and not append:
        print(f"  ⚠ 이미 {existing}개 존재.")
        print(f"    --reset  : 전부 삭제 후 재구축")
        print(f"    --append : 기존 유지, 새 청크만 추가")
        sys.exit(1)

    embed_label = f"{embed_type}/{embed_model}" if embed_model else embed_type
    print(f"[2/4] oom_kb 준비 완료 (cosine, {embed_label}, 기존 {existing}개)")

    # ── 3. 인덱싱 ──
    # append 모드: 이미 있는 ID 제외
    if append and existing > 0:
        existing_ids = set(collection.get(limit=existing)["ids"])
        before = len(chunks)
        filtered = [(c, i) for c, i in zip(chunks, ids) if i not in existing_ids]
        chunks = [x[0] for x in filtered]
        ids = [x[1] for x in filtered]
        skipped = before - len(chunks)
        print(f"[3/4] append 모드: {skipped}개 중복 스킵, {len(chunks)}개 신규 추가")
    else:
        print(f"[3/4] 인덱싱...")

    if not chunks:
        print("  추가할 새 청크 없음. 완료.")
        return

    documents = [c["content"] for c in chunks]
    metadatas = [chunk_to_chroma_metadata(c) for c in chunks]

    batch_add(collection, ids, documents, metadatas)

    # ── 4. 검증 ──
    count = collection.count()
    expected = existing + len(ids) if append else len(ids)
    assert count == expected, f"불일치: 예상 {expected}, 실제 {count}"
    print(f"[4/4] 검증 통과 ✓ (총 {count}개, 이번에 +{len(ids)}개)")

    # 카테고리별
    print(f"\n{'='*45}")
    print(f"  DB 경로:  {db_path.resolve()}")
    print(f"  컬렉션:   oom_kb")
    print(f"  임베딩:   {embed_label}")
    print(f"  총 청크:  {count}")
    print(f"{'='*45}")

    for cat in ["global_oom", "cgroup_oom", "swap_exhaustion",
                "page_alloc_failure", "general"]:
        result = collection.get(where={"error_category": cat}, limit=1000)
        print(f"  {cat:25s}: {len(result['ids']):4d}")

    # 검색 테스트
    print(f"\n=== 검색 테스트 (search_kb 시뮬레이션) ===")
    test_queries = [
        ("global_oom",         "global_oom CONSTRAINT_NONE no swap space"),
        ("cgroup_oom",         "cgroup_oom cgroup memory limit"),
        ("swap_exhaustion",    "swap_exhaustion no swap space"),
        ("page_alloc_failure", "page_alloc_failure order 2 page allocation"),
    ]
    for cat, query in test_queries:
        result = collection.query(
            query_texts=[query],
            n_results=3,
            where={"error_category": {"$in": [cat, "general"]}},
        )
        print(f"\n  [{cat}] \"{query}\"")
        for i, (cid, dist) in enumerate(
            zip(result["ids"][0], result["distances"][0])
        ):
            meta = result["metadatas"][0][i]
            print(f"    {i+1}. {cid:40s} dist={dist:.4f}  ({meta['error_category']})")

    print(f"\n✓ 인덱싱 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="kb_chunks.jsonl → ChromaDB (chroma_db/oom_kb)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--reset", action="store_true",
                        help="컬렉션 삭제 후 재구축")
    parser.add_argument("--append", action="store_true",
                        help="기존 유지, 새 청크만 추가 (중복 ID 스킵)")
    parser.add_argument("--embed", default="ollama",
                        choices=["default", "ollama"])
    parser.add_argument("--embed-model", default=None,
                        help="Ollama 모델명 (기본: nomic-embed-text)")
    args = parser.parse_args()

    if args.reset and args.append:
        print("✗ --reset과 --append는 동시에 쓸 수 없음")
        sys.exit(1)

    main(args.input, args.db, args.reset, args.append, args.embed, args.embed_model)