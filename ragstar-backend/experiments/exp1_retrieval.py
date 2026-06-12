import sys
import json
import argparse
from collections import defaultdict
import pandas as pd
from pathlib import Path
from tqdm import tqdm

"""
python experiments/exp1_retrieval.py --query-mode raw --top-k 3
python experiments/exp1_retrieval.py --query-mode raw --top-k 5
python experiments/exp1_retrieval.py --query-mode parsed --top-k 3
python experiments/exp1_retrieval.py --query-mode parsed --top-k 5

"""

# ==========================================
# 1. 시스템 경로 및 디렉토리 세팅 
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 실제 검색 모듈 및 팀원이 구축한 DB 연결 싱글턴 임포트
from app.agent.tools.search_kb import search_kb
from app.database.chromadb_client import get_collection
from app.agent.nodes.node_1_parser import node_1_parser

# ==========================================
# 2. 평가 지표 계산 함수
# ==========================================
def calculate_recall(retrieved_ids: list, ground_truth_ids: list) -> float:
    """
    검색된 청크 ID와 정답 청크 ID를 비교하여 Recall 값을 계산합니다.
    """
    retrieved_set = set(retrieved_ids)
    truth_set = set(ground_truth_ids)
    
    if not truth_set:
        return 0.0 # 정답이 없는 경우 예외 처리
        
    # 교집합(찾아낸 정답)의 개수를 전체 정답의 개수로 나눔
    intersection = retrieved_set.intersection(truth_set)
    return len(intersection) / len(truth_set)


def build_parser_input_state(raw_log: str) -> dict:
    """
    Step 1 파서를 호출하기 위한 최소 상태를 만든다.

    실험 명세 기준 입력은 raw_log 원문이므로,
    parsed 모드도 데이터셋에 precomputed parsed_fields가 있다고 가정하지 않는다.
    """
    return {
        "raw_log": raw_log,
        "metadata": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }


def load_oom_logs_by_id(path: Path) -> dict[str, dict]:
    """
    oom_logs.jsonl을 log_id 기준 dict로 로드한다.

    Experiment 1의 평가 라벨은 qa_ground_truth.jsonl에 있고,
    실제 원문 로그는 oom_logs.jsonl에 있으므로 둘을 log_id로 조인한다.
    """
    with open(path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["log_id"]: row for row in rows}


def get_raw_log_for_log_id(log_id: str, oom_logs_by_id: dict[str, dict]) -> str:
    """
    log_id에 대응하는 raw_log를 반환한다.
    """
    row = oom_logs_by_id.get(log_id)
    if not row:
        return ""
    return row.get("raw_log", "")


def build_parsed_retrieval_inputs(data: dict, raw_log: str) -> tuple[str, dict]:
    """
    Experiment 1의 parsed 모드 입력을 명세에 맞게 구성한다.

    - parsed_fields: raw_log에서 Step 1 파서를 직접 실행해 생성
    - oom_type: retrieval 자체를 평가하기 위해 정답 라벨(expected_oom_type)을 사용

    참고:
    Step 2 분류기까지 실제 호출하면 retrieval 품질과 분류 품질이 섞이므로,
    이 실험은 검색 단계만 따로 본다.
    """
    parsed_state = node_1_parser(build_parser_input_state(raw_log))
    parsed_fields = parsed_state.get("parsed_fields", {})
    oom_type = data.get("expected_oom_type", "unknown")
    return oom_type, parsed_fields


def build_search_kb_query(oom_type: str, parsed_fields: dict) -> str:
    """
    search_kb와 동일한 규칙으로 Experiment 1용 KB query를 구성한다.

    실험에서 top_k를 조정하기 위해 query 실행만 로컬에서 수행하되,
    query formulation 자체는 운영 경로와 맞춘다.
    """
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

    return " ".join(query_parts)


def run_parsed_retrieval_query(
    oom_type: str,
    parsed_fields: dict,
    collection,
    top_k: int,
) -> dict:
    """
    Experiment 1의 parsed retrieval을 top_k 가변값으로 실행한다.
    """
    query = build_search_kb_query(oom_type, parsed_fields)

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"error_category": {"$in": [oom_type, "general"]}},
        )
    except Exception as e:
        return {
            "query_used": query,
            "chunks": [],
            "total_found": 0,
            "error": f"KB 검색 오류: {str(e)}",
        }

    chunks = []
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

# ==========================================
# 3. 메인 실험 루프
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="[Experiment 1] 검색 성능(Recall) 평가")
    parser.add_argument(
        "--query-mode", 
        type=str, 
        choices=["raw", "parsed"], 
        required=True,
        help="질의 방식: raw(원본 로그) vs parsed(구조화된 로그)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="검색 결과 개수 (예: 3이면 recall@3, 5이면 recall@5)",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        print(" 에러: --top-k는 1 이상의 정수여야 합니다.")
        sys.exit(1)

    # 경로 동적 할당
    DATA_DIR = PROJECT_ROOT / "data"
    GROUND_TRUTH_PATH = DATA_DIR / "qa_ground_truth.jsonl"
    OOM_LOGS_PATH = DATA_DIR / "oom_logs.jsonl"
    RESULT_DIR = DATA_DIR / "exp_results"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 성능 최적화: 루프 시작 전 ChromaDB 컬렉션 로드 (단 1회)
    # ==========================================
    print(" ChromaDB에 연결 중...")
    # 팀원이 구현한 싱글턴 클라이언트를 사용하여 컬렉션 획득 (경로 하드코딩 제거)
    collection = get_collection()

    print(f" [실험 1 시작] Query Mode: {args.query_mode}, Top-K: {args.top_k}")

    # 데이터 로드
    try:
        with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(" 에러: qa_ground_truth.jsonl 파일을 찾을 수 없습니다.")
        sys.exit(1)

    try:
        oom_logs_by_id = load_oom_logs_by_id(OOM_LOGS_PATH)
    except FileNotFoundError:
        print(" 에러: oom_logs.jsonl 파일을 찾을 수 없습니다.")
        sys.exit(1)

    results = []
    total_recall = 0.0
    recall_by_oom_type = defaultdict(list)

    for i, data in enumerate(tqdm(dataset, desc="평가 진행 중")):
        log_id = data.get("log_id", f"log_{i}")
        expected_oom_type = data.get("expected_oom_type", "unknown")
        ground_truth_ids = data.get("relevant_chunk_ids", [])
        retrieved_ids = []
        raw_log = get_raw_log_for_log_id(log_id, oom_logs_by_id)
        
        # ==========================================
        # 핵심 로직: Query Mode에 따른 완벽한 분기 처리
        # ==========================================
        if args.query_mode == "raw":
            # 1. Raw 모드: 날것의 긴 로그를 파싱/필터 없이 그대로 벡터 DB에 던짐
            raw_query = raw_log
            try:
                raw_results = collection.query(
                    query_texts=[raw_query],
                    n_results=args.top_k
                )
                if raw_results and raw_results.get("ids") and raw_results["ids"][0]:
                    retrieved_ids = raw_results["ids"][0]
            except Exception as e:
                pass # 에러 발생 시 retrieved_ids는 빈 리스트로 유지됨

        else:
            # 2. Parsed 모드: raw_log에서 Step 1 파서를 실행해 structured query를 만든다.
            # retrieval 평가이므로 oom_type은 정답 라벨(expected_oom_type)을 사용해
            # 검색 단계 품질만 측정한다.
            oom_type, parsed_fields = build_parsed_retrieval_inputs(data, raw_log)
            
            # 리팩토링된 함수 시그니처에 맞게 호출
            search_results = run_parsed_retrieval_query(
                oom_type=oom_type,
                parsed_fields=parsed_fields,
                collection=collection,
                top_k=args.top_k,
            )
            
            # 반환된 Dict 구조 {"chunks": [...]} 에서 chunk_id만 추출
            for chunk in search_results.get("chunks", []):
                retrieved_ids.append(chunk["chunk_id"])

        # 지표 계산
        recall_score = calculate_recall(retrieved_ids, ground_truth_ids)
        total_recall += recall_score
        recall_by_oom_type[expected_oom_type].append(recall_score)
        
        results.append({
            "log_id": log_id,
            "query_mode": args.query_mode,
            "top_k": args.top_k,
            "expected_oom_type": expected_oom_type,
            "expected_chunks": str(ground_truth_ids),
            "retrieved_chunks": str(retrieved_ids),
            "recall": recall_score
        })

    # ==========================================
    # 4. 결과 집계 및 저장
    # ==========================================
    """
    avg_recall = total_recall / len(dataset) if dataset else 0
    print(f"\n 실험 완료! 평균 Recall: {avg_recall:.4f}")
    print(" OOM type별 평균 Recall:")
    for oom_type in sorted(recall_by_oom_type):
        scores = recall_by_oom_type[oom_type]
        avg_type_recall = sum(scores) / len(scores) if scores else 0.0
        print(
            f"  - {oom_type}: {avg_type_recall:.4f} "
            f"(cases={len(scores)})"
        )
    """
    df = pd.DataFrame(results)

    result_filename = f"exp1_retrieval_{args.query_mode}_top{args.top_k}_results.csv"
    df.to_csv(
        RESULT_DIR / result_filename,
        index=False,
        encoding='utf-8-sig',
    )

    summary = df.groupby('expected_oom_type').agg(
        n=('log_id', 'count'),
        recall=('recall', 'mean')
    ).reset_index()

    overall_row = pd.DataFrame([{
        'expected_oom_type': 'Overall',
        'n': len(df),
        'recall': df['recall'].mean()
    }])
    summary = pd.concat([summary, overall_row], ignore_index=True)

    summary_filename = f"exp1_retrieval_{args.query_mode}_top{args.top_k}_summary.csv"
    summary.to_csv(
        RESULT_DIR / summary_filename,
        index=False,
        encoding='utf-8-sig',
    )

    """
    df.to_csv(
        RESULT_DIR / f"exp1_retrieval_{args.query_mode}_top{args.top_k}_results.csv",
        index=False,
        encoding='utf-8-sig',
    )
    """
    print(f"\n{'='*60}")
    print(f"🚀 실험 완료! (Mode: {args.query_mode}, Top-K: {args.top_k})")
    print(f"{'='*60}")
    print(summary.to_string(index=False)) # 예쁘게 표 형태로 출력
    print(f"{'='*60}")
    print(f"💾 상세 결과 저장됨: {result_filename}")
    print(f"💾 요약 결과 저장됨: {summary_filename}")

if __name__ == "__main__":
    main()