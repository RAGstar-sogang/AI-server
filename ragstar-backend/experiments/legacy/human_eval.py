"""
[Human Evaluation] RAG 시스템 출력 인간 검증 스크립트

목적:
  임베딩 유사도 점수와 별개로, 실제로 사람이 봤을 때 RAG가 뽑아주는
  진단·조치 가이드가 도움이 되는 답인지 직접 검증한다.

동작:
  1. qa_ground_truth.jsonl + oom_logs.jsonl 로드
  2. 4개 OOM 유형별 stratified 랜덤 샘플링 (기본: 유형당 3개, 총 12개)
  3. 각 샘플을 RAGstar 파이프라인에 투입
  4. raw_log / 정답 / RAG 출력 / 자동 점수를 모은 Markdown 파일 생성
  5. 사람이 직접 채울 평가 항목(구체성·정확성·실행가능성·적합성)을 빈 칸으로 포함

사용법:
  python experiments/human_eval.py
  python experiments/human_eval.py --llm llama3:latest --samples-per-type 3 --seed 42
  python experiments/human_eval.py --all  # 56개 전체

출력:
  data/exp_results/human_eval_<llm_slug>_<timestamp>.md
"""

import sys
import json
import argparse
import random
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ==========================================
# 1. 시스템 경로 세팅
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.agent.graph import create_initial_state
from app.agent.nodes.node_1_parser import node_1_parser
from app.agent.nodes.node_2_classifier import (
    DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    node_2_classifier,
)
from app.agent.nodes.node_3_executor import node_3_executor
from app.agent.nodes.node_4_synthesizer import node_4_synthesizer
from app.core.llm_factory import build_chat_ollama, build_exp2_embeddings
from app.core.settings import get_settings

# ==========================================
# 2. 헬퍼
# ==========================================
def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    return slug.strip("_") or "default"


def truncate_lines(text: str, max_lines: int = 25) -> str:
    """raw_log를 보기 좋게 앞부분만 자르기"""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines}줄 생략)"


def safe_get(d: dict, *keys, default=None):
    """중첩 dict 안전 접근: safe_get(d, 'a', 'b', 'c')"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur if cur is not None else default


def format_list(items, indent="  "):
    """리스트를 markdown bullet으로 포맷"""
    if not items:
        return f"{indent}_(없음)_"
    if not isinstance(items, list):
        items = [items]
    return "\n".join(f"{indent}- {x}" for x in items if x)


# ==========================================
# 3. RAG 파이프라인 실행
# ==========================================
def run_rag(raw_log: str, llm_name: str, embeddings) -> dict:
    state = create_initial_state(raw_log)
    state["label_embeddings"] = embeddings
    state["label_similarity_threshold"] = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD
    state["label_similarity_margin"] = DEFAULT_EMBEDDING_SIMILARITY_MARGIN
    state["llm"] = build_chat_ollama(llm_name, json_mode=True)

    try:
        state = node_1_parser(state)
        state = node_2_classifier(state)
        state = node_3_executor(state)
        state = node_4_synthesizer(state)
        return state if isinstance(state, dict) else {"_error": "state_not_dict"}
    except Exception as e:
        return {**state, "_error": f"rag_error: {e}"}


# ==========================================
# 4. Markdown 렌더러
# ==========================================
def render_sample(idx: int, total: int, gt: dict, raw_log: str, rag_output: dict) -> str:
    log_id = gt.get("log_id", "?")
    expected = gt.get("expected_oom_type", "?")
    gt_inner = gt.get("ground_truth", {}) or {}
    must_evidence = gt_inner.get("must_include_evidence", []) or []
    gt_action = gt_inner.get("action_guide", []) or []
    gt_root = gt_inner.get("accepted_root_causes", []) or []

    # RAG 출력 파싱
    classification = rag_output.get("classification", {}) or {}
    rag_oom_type = classification.get("oom_type", "")
    raw_llm_oom = classification.get("raw_llm_oom_type", "")
    deterministic = classification.get("deterministic_oom_type", "")
    needs_kb = classification.get("needs_kb", False)
    tools_needed = classification.get("tools_needed", []) or []

    diagnosis = rag_output.get("diagnosis", {}) or {}
    log_analysis = safe_get(diagnosis, "log_analysis", "summary", default="")
    diag_inner = diagnosis.get("diagnosis", {}) or {}
    root_cause = diag_inner.get("root_cause", "")
    contributing = diag_inner.get("contributing_factors", []) or []
    rag_evidence = diag_inner.get("evidence", []) or []
    severity = diag_inner.get("severity", "")
    action = diagnosis.get("action_guide", {}) or {}
    immediate = action.get("immediate", []) or []
    recommended = action.get("recommended", []) or []
    further = action.get("further_investigation", []) or []

    error = rag_output.get("_error", "") or rag_output.get("error", "")

    # 자동 점수 계산
    cat_match = "✓ 일치" if expected.strip().lower() == str(rag_oom_type).strip().lower() else "✗ 불일치"
    classification_corrected = ""
    if raw_llm_oom and deterministic and raw_llm_oom != deterministic and deterministic != "unknown":
        classification_corrected = f" (LLM은 `{raw_llm_oom}` → 결정 규칙이 `{deterministic}`로 보정)"

    # ===== Markdown 빌드 =====
    md = []
    md.append(f"## 샘플 {idx} / {total} — `{log_id}` ({expected})")
    md.append("")

    # --- Raw Log ---
    md.append("### 입력 — Raw OOM Log")
    md.append("")
    md.append("```")
    md.append(truncate_lines(raw_log, 30))
    md.append("```")
    md.append("")

    # --- Ground Truth ---
    md.append("### 정답 (Ground Truth)")
    md.append("")
    md.append(f"- **OOM 유형**: `{expected}`")
    md.append(f"- **인정 가능한 근본 원인**:")
    md.append(format_list(gt_root, indent="  "))
    md.append(f"- **필수 근거 항목**:")
    md.append(format_list(must_evidence, indent="  "))
    md.append(f"- **조치 가이드**:")
    md.append(format_list(gt_action, indent="  "))
    md.append("")

    # --- RAG 출력 ---
    md.append("### RAGstar 출력")
    md.append("")

    if error:
        md.append(f"> ⚠ **추론 에러**: {error}")
        md.append("")

    # 분류
    md.append("**1) 분류 결과**")
    md.append("")
    md.append(f"- **최종 OOM 유형**: `{rag_oom_type}`{classification_corrected}")
    md.append(f"- LLM 원시 분류: `{raw_llm_oom or '(없음)'}`")
    md.append(f"- 결정 규칙 분류: `{deterministic or '(없음)'}`")
    md.append(f"- 도구 호출: {tools_needed if tools_needed else '없음'}")
    md.append(f"- KB 검색 사용: {'예' if needs_kb else '아니오'}")
    md.append("")

    # 로그 요약
    md.append("**2) 로그 요약**")
    md.append("")
    md.append(f"> {log_analysis or '_(없음)_'}")
    md.append("")

    # 진단
    md.append("**3) 진단**")
    md.append("")
    md.append(f"- **근본 원인**: {root_cause or '_(없음)_'}")
    md.append(f"- **심각도**: {severity or '_(없음)_'}")
    md.append(f"- **기여 요인**:")
    md.append(format_list(contributing, indent="  "))
    md.append(f"- **근거**:")
    md.append(format_list(rag_evidence, indent="  "))
    md.append("")

    # 조치
    md.append("**4) 조치 가이드**")
    md.append("")
    md.append(f"- **즉시 조치 (immediate)**:")
    md.append(format_list(immediate, indent="  "))
    md.append(f"- **권장 조치 (recommended)**:")
    md.append(format_list(recommended, indent="  "))
    md.append(f"- **추가 조사 (further_investigation)**:")
    md.append(format_list(further, indent="  "))
    md.append("")

    # --- 자동 점수 요약 ---
    md.append("### 자동 점수 (참고)")
    md.append("")
    md.append(f"- **분류 일치**: {cat_match}")
    md.append(f"- 정답 근거 항목 수: {len(must_evidence)} / RAG 근거 항목 수: {len(rag_evidence)}")
    md.append(f"- 정답 조치 항목 수: {len(gt_action)} / RAG 조치 항목 수: {len(immediate) + len(recommended) + len(further)}")
    md.append("")

    # --- 인간 평가 ---
    md.append("### 🧑 인간 평가 (직접 채워주세요)")
    md.append("")
    md.append("| 항목 | 점수 (0~3) | 코멘트 |")
    md.append("|---|---|---|")
    md.append("| **구체성** — 일반론 vs 구체적 명령어/수치 |  |  |")
    md.append("| **정확성** — Linux/OOM 분야 지식과 부합 |  |  |")
    md.append("| **실행 가능성** — 운영자가 그대로 실행 가능 |  |  |")
    md.append("| **적합성** — 이 로그의 실제 원인에 맞는 조언 |  |  |")
    md.append("")
    md.append("**총평** _(이 답이 실제 운영 도움이 될까? 어떤 점이 좋고 어떤 점이 약한가?)_:")
    md.append("")
    md.append("> ")
    md.append("")
    md.append("---")
    md.append("")

    return "\n".join(md)


# ==========================================
# 5. 메인
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="[Human Eval] RAG 출력 인간 검증")
    parser.add_argument("--llm", type=str, default=None, help="추론 LLM (예: llama3:latest)")
    parser.add_argument("--samples-per-type", type=int, default=3, help="유형당 샘플 수 (기본 3)")
    parser.add_argument("--all", action="store_true", help="56개 전체 평가")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    DATA_DIR = PROJECT_ROOT / "data"
    GT_PATH = DATA_DIR / "qa_ground_truth.jsonl"
    OOM_LOGS_PATH = DATA_DIR / "oom_logs.jsonl"
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else DATA_DIR / "exp_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 결정
    llm_name = args.llm or get_settings().exp2_base_model
    print(f"🚀 Human Eval | LLM: {llm_name}")

    # 데이터 로드
    try:
        gt_list = load_jsonl(GT_PATH)
        oom_logs = {row["log_id"]: row.get("raw_log", "") for row in load_jsonl(OOM_LOGS_PATH)}
    except FileNotFoundError as e:
        print(f"❌ 파일 없음: {e}")
        sys.exit(1)

    # 샘플링
    random.seed(args.seed)
    if args.all:
        samples = gt_list
        print(f"📋 전체 {len(samples)}개 평가")
    else:
        # OOM 유형별 stratified 샘플링
        by_type = defaultdict(list)
        for row in gt_list:
            by_type[row.get("expected_oom_type", "unknown")].append(row)

        samples = []
        for ot, items in sorted(by_type.items()):
            n = min(args.samples_per_type, len(items))
            picked = random.sample(items, n)
            samples.extend(picked)
            print(f"  {ot}: {n}개 추출 (전체 {len(items)}개 중)")
        print(f"📋 총 {len(samples)}개 평가")

    # 임베딩 모델 (한 번만 로드)
    print("🔧 임베딩 모델 로딩...")
    embeddings = build_exp2_embeddings()

    # 헤더
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"human_eval_{slugify(llm_name)}_{timestamp}.md"

    md_lines = [
        f"# RAGstar 인간 검증 시트",
        f"",
        f"- **모델**: `{llm_name}`",
        f"- **샘플 수**: {len(samples)}개",
        f"- **샘플링 방식**: {'전체' if args.all else f'OOM 유형별 {args.samples_per_type}개 stratified'}",
        f"- **시드**: {args.seed}",
        f"- **생성 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## 평가 가이드",
        f"",
        f"각 샘플에 대해 RAG 출력을 읽고, 인간 평가 표의 4개 항목을 0~3점으로 채워주세요.",
        f"",
        f"| 점수 | 의미 |",
        f"|---|---|",
        f"| **3** | 매우 좋음 — 운영에 그대로 쓸 수 있음 |",
        f"| **2** | 양호 — 약간 보강하면 쓸 수 있음 |",
        f"| **1** | 부족 — 의미는 맞지만 실용성이 떨어짐 |",
        f"| **0** | 나쁨 — 잘못된 정보거나 도움이 안 됨 |",
        f"",
        f"---",
        f"",
    ]

    # 각 샘플 처리
    for i, gt in enumerate(samples, 1):
        log_id = gt.get("log_id", f"log_{i}")
        raw_log = oom_logs.get(log_id, "")
        if not raw_log:
            print(f"⚠ {log_id}: raw_log 없음, skip")
            continue

        print(f"[{i}/{len(samples)}] {log_id} ({gt.get('expected_oom_type', '?')}) 추론 중...")
        rag_output = run_rag(raw_log, llm_name, embeddings)

        md_section = render_sample(i, len(samples), gt, raw_log, rag_output)
        md_lines.append(md_section)

    # 저장
    output_file.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n💾 저장 완료: {output_file}")
    print(f"📝 파일을 열어서 각 샘플의 인간 평가 표를 채워주세요.")


if __name__ == "__main__":
    main()