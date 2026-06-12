#!/usr/bin/env python3
"""
kb_docs.jsonl → kb_chunks.jsonl 청크 생성 스크립트

위치: docs/kb_docs_to_chunks.py

사용법:
    cd ragstar-backend
    python docs/kb_docs_to_chunks.py --reset              # 전체 재생성
    python docs/kb_docs_to_chunks.py --append             # 새 문서만 청킹해서 추가
"""

import json
import re
import argparse
import sys
from pathlib import Path
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ══════════════════════════════════════════════════════════════
# 경로 설정
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "kb_docs.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "kb_chunks.jsonl"


# ══════════════════════════════════════════════════════════════
# 토큰 근사 + 청킹
# ══════════════════════════════════════════════════════════════

_TOKEN_SPLIT_RE = re.compile(r'[\s/\-_:;.,=\(\)\[\]{}|&<>]+')

def approx_token_count(text: str) -> int:
    parts = _TOKEN_SPLIT_RE.split(text)
    return sum(1 for p in parts if p)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    length_function=approx_token_count,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)


# ══════════════════════════════════════════════════════════════
# 키워드 추출
# ══════════════════════════════════════════════════════════════

KEYWORD_PATTERNS = [
    ("oom_score_adj",              r"oom_score_adj"),
    ("oom_score",                  r"oom_score(?!_adj)"),
    ("oom_adj",                    r"\boom_adj\b"),
    ("oom_kill",                   r"oom[_\- ]?kill"),
    ("panic_on_oom",               r"panic_on_oom"),
    ("oom_dump_tasks",             r"oom_dump_tasks"),
    ("oom_kill_allocating_task",   r"oom_kill_allocating_task"),
    ("overcommit_memory",          r"overcommit_memory"),
    ("overcommit_ratio",           r"overcommit_ratio"),
    ("overcommit",                 r"overcommit(?!_memory|_ratio)"),
    ("swappiness",                 r"swappiness"),
    ("zswap",                      r"zswap"),
    ("zram",                       r"zram"),
    ("swap",                       r"\bswap\b"),
    ("min_free_kbytes",            r"min_free_kbytes"),
    ("watermark",                  r"watermark"),
    ("watermark_scale_factor",     r"watermark_scale_factor"),
    ("kswapd",                     r"kswapd"),
    ("reclaim",                    r"\breclaim"),
    ("compaction",                 r"\bcompaction\b"),
    ("fragmentation",              r"fragmentation"),
    ("zone_reclaim_mode",          r"zone_reclaim_mode"),
    ("cgroup",                     r"\bcgroup"),
    ("memcg",                      r"memcg"),
    ("memory.max",                 r"memory\.max\b"),
    ("memory.high",                r"memory\.high\b"),
    ("memory.limit_in_bytes",      r"memory\.limit_in_bytes"),
    ("memory.oom_control",         r"memory\.oom_control"),
    ("memory.oom.group",           r"memory\.oom\.group"),
    ("gfp_mask",                   r"gfp_mask"),
    ("gfp_flags",                  r"gfp.flag"),
    ("page_alloc",                 r"page.alloc"),
    ("meminfo",                    r"meminfo"),
    ("slabinfo",                   r"slabinfo"),
    ("dmesg",                      r"\bdmesg\b"),
    ("vmstat",                     r"\bvmstat\b"),
    ("psi",                        r"\bpsi\b"),
    ("mlock",                      r"\bmlock"),
    ("hugepages",                  r"huge.?pages?\b"),
    ("transparent_hugepage",       r"transparent.?huge"),
    ("numa",                       r"\bnuma\b"),
    ("drop_caches",                r"drop_caches"),
    ("mglru",                      r"multi.gen.lru|mglru"),
    ("damon",                      r"\bdamon\b"),
    ("lru",                        r"\blru\b"),
    ("slab",                       r"\bslab\b"),
    ("docker",                     r"\bdocker\b"),
    ("kubernetes",                 r"kubernetes|k8s"),
    ("oomkilled",                  r"oomkilled"),
    ("exit_code_137",              r"exit.code.137|code 137"),
    ("memory_limit",               r"memory.limit"),
    ("memory_request",             r"memory.request"),
    ("systemd-oomd",               r"systemd.oomd"),
    ("earlyoom",                   r"earlyoom"),
    ("SIGKILL",                    r"sigkill"),
]

_KW_RE = [(kw, re.compile(pat, re.IGNORECASE)) for kw, pat in KEYWORD_PATTERNS]

def extract_keywords(text: str) -> list[str]:
    return [kw for kw, regex in _KW_RE if regex.search(text)]

def doc_id_to_prefix(doc_id: str) -> str:
    return doc_id.replace("-", "_")


# ══════════════════════════════════════════════════════════════
# 단일 문서 → 청크 리스트
# ══════════════════════════════════════════════════════════════

def chunk_one_doc(doc: dict) -> list[dict]:
    """문서 1개를 청크 리스트로 변환."""
    raw_text = doc.get("raw_text", "").strip()
    if not raw_text:
        return []

    splits = splitter.split_text(raw_text)
    prefix = doc_id_to_prefix(doc["doc_id"])

    return [
        {
            "chunk_id": f"{prefix}_chunk_{idx}",
            "doc_id": doc["doc_id"],
            "chunk_index": idx,
            "title": doc["title"],
            "content": chunk_text,
            "metadata": {
                "error_category": doc.get("error_category", ""),
                "keywords": extract_keywords(chunk_text),
            },
        }
        for idx, chunk_text in enumerate(splits)
    ]


# ══════════════════════════════════════════════════════════════
# 기존 chunks 파일에서 처리 완료된 doc_id 수집
# ══════════════════════════════════════════════════════════════

def load_existing_chunks(path: Path) -> tuple[list[dict], set[str]]:
    """기존 chunks 파일 로드. (청크 리스트, doc_id 집합) 반환."""
    if not path.exists():
        return [], set()

    chunks = []
    doc_ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunks.append(chunk)
                doc_ids.add(chunk["doc_id"])
    return chunks, doc_ids


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main(input_path: Path, output_path: Path, reset: bool, append: bool):
    # ── 1. 문서 로드 ──
    with open(input_path, encoding="utf-8") as f:
        all_docs = [json.loads(line) for line in f if line.strip()]
    print(f"[1/3] 문서 {len(all_docs)}개 로드 ({input_path})")

    # ── 2. 모드별 분기 ──
    if reset or not output_path.exists():
        # 전체 재생성
        existing_chunks = []
        docs_to_process = all_docs
        if reset and output_path.exists():
            print(f"  기존 {output_path.name} 삭제 (--reset)")
    elif append:
        # 기존 유지 + 새 문서만 처리
        existing_chunks, existing_doc_ids = load_existing_chunks(output_path)
        docs_to_process = [d for d in all_docs if d["doc_id"] not in existing_doc_ids]
        skipped = len(all_docs) - len(docs_to_process)
        print(f"  기존 청크 {len(existing_chunks)}개 유지 (doc {len(existing_doc_ids)}개)")
        print(f"  신규 문서 {len(docs_to_process)}개 처리 ({skipped}개 스킵)")
    else:
        # 안전장치: 이미 파일 있는데 옵션 없으면 경고
        existing_chunks, _ = load_existing_chunks(output_path)
        if existing_chunks:
            print(f"  ⚠ {output_path.name}에 이미 {len(existing_chunks)}개 존재.")
            print(f"    --reset  : 전부 삭제 후 재생성")
            print(f"    --append : 새 문서만 추가")
            sys.exit(1)
        existing_chunks = []
        docs_to_process = all_docs

    if not docs_to_process:
        print("\n  추가할 새 문서 없음. 완료.")
        return

    # ── 3. 청킹 ──
    new_chunks = []
    empty_count = 0
    for doc in docs_to_process:
        chunks = chunk_one_doc(doc)
        if not chunks:
            empty_count += 1
            print(f"  ⚠ {doc['doc_id']}: raw_text 비어있음 → 스킵")
            continue
        new_chunks.extend(chunks)

    # chunk_id 중복 체크 (기존 + 신규)
    existing_ids = {c["chunk_id"] for c in existing_chunks}
    new_ids = [c["chunk_id"] for c in new_chunks]

    # 신규 내부 중복
    new_dupes = {k: v for k, v in Counter(new_ids).items() if v > 1}
    if new_dupes:
        print(f"  ✗ 신규 청크 내 중복: {new_dupes}")
        sys.exit(1)

    # 기존과 충돌
    conflicts = existing_ids & set(new_ids)
    if conflicts:
        print(f"  ✗ 기존 청크와 ID 충돌 {len(conflicts)}건: {list(conflicts)[:5]}...")
        sys.exit(1)

    print(f"[2/3] 신규 청크 {len(new_chunks)}개 생성 "
          f"(문서 {len(docs_to_process) - empty_count}개)")

    # ── 4. 저장 ──
    final_chunks = existing_chunks + new_chunks

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in final_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[3/3] {output_path.name} 저장 (총 {len(final_chunks)}개 = "
          f"기존 {len(existing_chunks)} + 신규 {len(new_chunks)})")

    # ── 통계 ──
    print(f"\n{'='*50}")

    cat_counts = Counter(c["metadata"]["error_category"] for c in final_chunks)
    print("error_category 분포:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {cnt:4d}")

    lengths = [len(c["content"]) for c in final_chunks]
    tok_est = [approx_token_count(c["content"]) for c in final_chunks]
    print(f"\n청크 길이 (chars): min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
    print(f"청크 길이 (≈tok):  min={min(tok_est)}, max={max(tok_est)}, avg={sum(tok_est)/len(tok_est):.0f}")

    print(f"\n청크 ≈토큰 분포:")
    for lo, hi in [(0,200),(200,350),(350,500),(500,600),(600,999)]:
        n = sum(1 for t in tok_est if lo <= t < hi)
        bar = '█' * (n // 2)
        print(f"  {lo:3d}-{hi:3d}: {n:3d} {bar}")

    kw_counts = Counter()
    for c in final_chunks:
        kw_counts.update(c["metadata"]["keywords"])
    print(f"\n상위 키워드 15개:")
    for kw, cnt in kw_counts.most_common(15):
        print(f"  {kw:30s}: {cnt}")

    no_kw = sum(1 for c in final_chunks if not c["metadata"]["keywords"])
    print(f"\n키워드 없는 청크: {no_kw}/{len(final_chunks)} ({no_kw/len(final_chunks)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="kb_docs.jsonl → kb_chunks.jsonl 청크 생성")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reset", action="store_true",
                        help="전체 삭제 후 재생성")
    parser.add_argument("--append", action="store_true",
                        help="기존 유지, 새 문서만 청킹해서 추가")
    args = parser.parse_args()

    if args.reset and args.append:
        print("✗ --reset과 --append는 동시에 쓸 수 없음")
        sys.exit(1)

    main(args.input, args.output, args.reset, args.append)