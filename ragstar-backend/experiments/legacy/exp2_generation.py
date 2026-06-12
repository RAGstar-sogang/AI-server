"""
[Experiment 2] End-to-End Generation Quality Evaluation
Naive Local LLM vs RAGstar (full pipeline)

Evaluation metrics (per log, then averaged):
  - category_match: expected_oom_type == generated oom_type (0 or 1)
  - evidence_recall: |must_include_evidence ∩ generated_evidence| / |must_include_evidence|
  - action_guide_similarity: mean over GT items of max cosine(GT_i, any generated item)
"""

"""
# Node 4 model을 gemma4로 수정한 뒤
python experiments/exp2_generation.py --model-mode base_llm
python experiments/exp2_generation.py --model-mode rag_agent
"""

import sys
import json
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate

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
    embedding_normalize_oom_type,
    node_2_classifier,
)
from app.agent.nodes.node_3_executor import node_3_executor
from app.agent.nodes.node_4_synthesizer import node_4_synthesizer
from app.agent.rag_runner import run_rag_agent as run_ragstar_agent
from app.core.llm_factory import build_chat_ollama, build_exp2_base_llm, build_exp2_embeddings
from app.core.settings import get_settings
from app.core.vllm_manager import ensure_vllm_model

# ==========================================
# 2. 모델 및 임베딩 세팅 (임베딩은 고정, 생성 모델은 CLI override 가능)
# ==========================================
@lru_cache(maxsize=8)
def get_exp2_llm(model_name: str | None = None):
    if get_settings().use_vllm:
        model_name = ensure_vllm_model(model_name or get_settings().exp2_base_model)
    return build_exp2_base_llm(model_name=model_name)


@lru_cache(maxsize=8)
def get_exp2_chat_llm(model_name: str | None = None):
    resolved_model = model_name or get_settings().exp2_base_model
    if get_settings().use_vllm:
        resolved_model = ensure_vllm_model(resolved_model)
    return build_chat_ollama(resolved_model, json_mode=True)


@lru_cache(maxsize=1)
def get_exp2_embeddings():
    return build_exp2_embeddings()


def resolve_exp2_llm_name(cli_value: str | None) -> str:
    return cli_value or get_settings().exp2_base_model


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(model_name).strip().lower())
    return slug.strip("_") or "default"

NAIVE_LLM_PROMPT = ChatPromptTemplate.from_template("""You are a Linux OOM diagnosis expert.
Analyze the following OOM kernel log and output a diagnosis.

Output ONLY a JSON object. No explanations, no markdown fences.

The JSON MUST have this exact structure:
{{
  "classification": {{
    "oom_type": "global_oom | cgroup_oom | swap_exhaustion | page_alloc_failure"
  }},
  "final_answer": {{
    "log_analysis": {{
      "summary": "brief factual summary (3-5 sentences)"
    }},
    "diagnosis": {{
      "root_cause": "root cause (1-2 sentences)",
      "contributing_factors": ["factor 1", "factor 2"],
      "evidence": ["specific number or fact 1", "specific number or fact 2"],
      "severity": "high | medium | low"
    }},
    "action_guide": {{
      "immediate": ["immediate action 1"],
      "recommended": ["recommended action 1"],
      "further_investigation": ["item 1"]
    }}
  }}
}}

OOM log:
{raw_log}
""")

# ==========================================
# 3. 데이터 로딩 헬퍼
# ==========================================
def load_jsonl(path: Path) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def build_oom_logs_index(path: Path) -> dict:
    rows = load_jsonl(path)
    return {row["log_id"]: row.get("raw_log", "") for row in rows}


def normalize_generated_oom_type(raw_oom_type: object) -> str:
    normalized = embedding_normalize_oom_type(raw_oom_type, get_exp2_embeddings())
    if normalized:
        return normalized
    if raw_oom_type is None:
        return ""
    return str(raw_oom_type).strip().lower()

# ==========================================
# 4. 필드 추출 (두 모드의 출력 구조가 다르므로 분기)
# ==========================================
def extract_fields(generated_json: dict, mode: str) -> dict:
    """
    두 모드의 다른 출력 구조를 통일:
      Naive LLM: {"classification": {...}, "final_answer": {"diagnosis": {...}, "action_guide": {...}}}
      RAG:       LangGraph state - {"classification": {...}, "diagnosis": {"diagnosis": {...}, "action_guide": {...}}}
                 (state의 "diagnosis" 키 안에 Node 4 final_answer가 통째로 들어있음)
    """
    if not isinstance(generated_json, dict):
        return {"oom_type": "", "evidence": [], "action_guide": []}

    # ── oom_type: 두 모드 모두 classification.oom_type ──
    classification = generated_json.get("classification", {}) or {}
    oom_type = ""
    if isinstance(classification, dict):
        oom_type = normalize_generated_oom_type(classification.get("oom_type", "") or "")

    # ── mode별 final_answer 경로 ──
    if mode == "rag_agent":
        # RAG: state["diagnosis"]가 곧 final_answer (log_analysis/diagnosis/action_guide 포함)
        final_answer = generated_json.get("diagnosis", {}) or {}
    else:
        # Naive LLM: final_answer 탑레벨
        final_answer = generated_json.get("final_answer", {}) or {}

    if not isinstance(final_answer, dict):
        final_answer = {}

    # ── evidence (final_answer.diagnosis.evidence) ──
    inner_diag = final_answer.get("diagnosis", {}) or {}
    if not isinstance(inner_diag, dict):
        inner_diag = {}
    evidence = inner_diag.get("evidence", []) or []

    if not isinstance(evidence, list):
        evidence = []

    # ── action_guide (immediate + recommended + further_investigation flatten) ──
    ag_dict = final_answer.get("action_guide", {}) or {}
    if not isinstance(ag_dict, dict):
        ag_dict = {}
    action_items = []
    for key in ["immediate", "recommended", "further_investigation"]:
        items = ag_dict.get(key, []) or []
        if isinstance(items, list):
            action_items.extend([str(x) for x in items if x])

    return {
        "oom_type": str(oom_type).strip().lower(),
        "evidence": [str(e) for e in evidence if e],
        "action_guide": action_items,
    }

# ==========================================
# 5. 평가 지표
# ==========================================

def eval_category_match(expected: str, generated: str) -> int:
    if not generated or not expected:
        return 0
    return 1 if expected.strip().lower() == generated.strip().lower() else 0

def eval_evidence_recall(must_include: list, generated_evidence: list) -> float:
    """
    단순 문자열 매칭의 한계를 극복하기 위해 임베딩 유사도 기반 Recall을 사용합니다.
    """
    if not must_include:
        return 1.0
    if not generated_evidence:
        return 0.0
    
    try:
        embeddings = get_exp2_embeddings()
        gt_vecs = [embeddings.embed_query(str(x)) for x in must_include]
        gen_vecs = [embeddings.embed_query(str(x)) for x in generated_evidence]
    except Exception as e:
        print(f"⚠ Evidence 임베딩 에러: {e}")
        return 0.0
        
    per_gt_max = []
    for gvec in gt_vecs:
        sims = [cosine_sim(gvec, pvec) for pvec in gen_vecs]
        # GT Evidence 1개에 대해 생성된 Evidence 중 가장 유사한 값의 점수 채택
        per_gt_max.append(max(sims) if sims else 0.0)
        
    # 평균 유사도를 반환
    return sum(per_gt_max) / len(per_gt_max)

def cosine_sim(v1, v2) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def eval_action_guide_similarity(gt_items: list, gen_items: list) -> float:
    if not gt_items:
        return 1.0
    if not gen_items:
        return 0.0
    try:
        embeddings = get_exp2_embeddings()
        gt_vecs = [embeddings.embed_query(str(x)) for x in gt_items]
        gen_vecs = [embeddings.embed_query(str(x)) for x in gen_items]

    except Exception as e:
        print(f"⚠ 임베딩 에러: {e}")
        return 0.0

    per_gt_max = []
    for gvec in gt_vecs:
        sims = [cosine_sim(gvec, pvec) for pvec in gen_vecs]
        per_gt_max.append(max(sims) if sims else 0.0)

    return sum(per_gt_max) / len(per_gt_max)


# ==========================================
# 6. 모델 추론
# ==========================================
def run_naive_llm(raw_log: str, *, llm_name: str | None = None) -> dict:
    try:
        prompt = NAIVE_LLM_PROMPT.format(raw_log=raw_log)
        llm = get_exp2_llm(llm_name)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start == -1 or end <= start:
            return {"_error": "naive_llm_response_missing_json"}
        parsed = json.loads(response_text[start:end])
        if isinstance(parsed, dict):
            return parsed
        return {"_error": "naive_llm_response_not_dict"}
    except Exception as e:
        return {"_error": f"naive_llm_error: {e}"}

def run_rag_agent(
    raw_log: str,
    *,
    label_similarity_threshold: float,
    label_similarity_margin: float,
    llm_name: str | None = None,
) -> dict:
    """RAGstar 전체 파이프라인을 실험 전용 설정으로 실행한다.

    목적:
    - Node 4 fail-fast 예외가 나더라도 Node 1~3에서 확보한 classification/tool 결과를
      exp2 평가에 남긴다.
    - app.agent.rag_runner.run_rag_agent를 사용하되, 기존 테스트가 monkeypatch하는
      exp2 모듈의 factory/node 함수들을 그대로 주입한다.
    """
    return run_ragstar_agent(
        raw_log,
        label_similarity_threshold=label_similarity_threshold,
        label_similarity_margin=label_similarity_margin,
        model_name=llm_name,
        llm_name=llm_name,
        create_state_fn=create_initial_state,
        embeddings_factory=get_exp2_embeddings,
        chat_llm_factory=get_exp2_chat_llm,
        node_1=node_1_parser,
        node_2=node_2_classifier,
        node_3=node_3_executor,
        node_4=node_4_synthesizer,
    )


def extract_inference_error(generated_json: dict) -> str:
    if not isinstance(generated_json, dict):
        return "invalid_generated_payload"

    for key in ["_error", "error"]:
        value = generated_json.get(key)
        if value:
            return str(value)
    return ""


def extract_embedding_diagnostics(generated_json: dict, expected_oom_type: str) -> dict | None:
    if not isinstance(generated_json, dict):
        return None

    classification = generated_json.get("classification", {}) or {}
    if not isinstance(classification, dict):
        return None

    embedding_debug = classification.get("embedding_debug", {}) or {}
    if not isinstance(embedding_debug, dict):
        embedding_debug = {}

    return {
        "log_id": generated_json.get("_log_id", ""),
        "expected_oom_type": expected_oom_type,
        "deterministic_oom_type": classification.get("deterministic_oom_type", ""),
        "final_oom_type": classification.get("oom_type", ""),
        "raw_llm_oom_type": classification.get("raw_llm_oom_type", ""),
        "normalized_llm_oom_type": classification.get("normalized_llm_oom_type", ""),
        "candidate_text": embedding_debug.get("candidate_text", ""),
        "canonical_label": embedding_debug.get("canonical_label", ""),
        "accepted_label": embedding_debug.get("accepted_label", ""),
        "best_label": embedding_debug.get("best_label", ""),
        "best_score": embedding_debug.get("best_score", -1.0),
        "second_label": embedding_debug.get("second_label", ""),
        "second_score": embedding_debug.get("second_score", -1.0),
        "score_margin": embedding_debug.get("score_margin", -1.0),
        "passes_threshold": int(bool(embedding_debug.get("passes_threshold", False))),
        "passes_margin": int(bool(embedding_debug.get("passes_margin", False))),
        "normalization_source": embedding_debug.get("normalization_source", ""),
        "similarity_threshold": embedding_debug.get("similarity_threshold", DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD),
        "similarity_margin": embedding_debug.get("similarity_margin", DEFAULT_EMBEDDING_SIMILARITY_MARGIN),
    }


def _predict_with_threshold(row: pd.Series, threshold: float, margin: float) -> str:
    deterministic = str(row.get("deterministic_oom_type", "") or "")
    if deterministic and deterministic != "unknown":
        return deterministic

    canonical = str(row.get("canonical_label", "") or "")
    if canonical:
        return canonical

    best_label = str(row.get("best_label", "") or "")
    try:
        best_score = float(row.get("best_score", -1.0))
        score_margin = float(row.get("score_margin", -1.0))
    except (TypeError, ValueError):
        return "unknown"

    if not best_label:
        return "unknown"
    if best_score < threshold:
        return "unknown"
    if score_margin < margin:
        return "unknown"
    return best_label


def build_embedding_calibration_summary(embedding_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float] | None]:
    empty_columns = ["threshold", "margin", "n", "category_match", "accepted_rate"]

    if embedding_df.empty:
        return pd.DataFrame(columns=empty_columns), None

    candidate_rows = embedding_df[
        embedding_df["deterministic_oom_type"].fillna("") == "unknown"
    ].copy()

    if candidate_rows.empty:
        return pd.DataFrame(columns=empty_columns), None

    thresholds = [round(x, 2) for x in np.arange(0.45, 0.91, 0.03)]
    margins = [round(x, 2) for x in np.arange(0.00, 0.25, 0.02)]

    calibration_rows = []
    for threshold in thresholds:
        for margin in margins:
            predicted = [
                _predict_with_threshold(row, threshold, margin)
                for _, row in candidate_rows.iterrows()
            ]
            expected = [str(x).strip().lower() for x in candidate_rows["expected_oom_type"].tolist()]
            matches = [int(exp == pred) for exp, pred in zip(expected, predicted)]
            accepted = [int(pred != "unknown") for pred in predicted]
            calibration_rows.append(
                {
                    "threshold": threshold,
                    "margin": margin,
                    "n": len(candidate_rows),
                    "category_match": round(float(np.mean(matches)) if matches else 0.0, 4),
                    "accepted_rate": round(float(np.mean(accepted)) if accepted else 0.0, 4),
                }
            )

    calibration_df = pd.DataFrame(calibration_rows)
    if calibration_df.empty:
        return calibration_df, None

    calibration_df = calibration_df.sort_values(
        by=["category_match", "accepted_rate", "threshold", "margin"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best_row = calibration_df.iloc[0].to_dict()
    return calibration_df, {
        "threshold": float(best_row["threshold"]),
        "margin": float(best_row["margin"]),
        "category_match": float(best_row["category_match"]),
        "accepted_rate": float(best_row["accepted_rate"]),
        "n": float(best_row["n"]),
    }

# ==========================================
# 7. 메인 루프
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="[Experiment 2] E2E Generation Quality")
    parser.add_argument(
        "--model-mode", type=str, choices=["base_llm", "rag_agent"], required=True,
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="Override inference model for the selected run. Embeddings remain fixed.",
    )
    parser.add_argument(
        "--label-similarity-threshold",
        type=float,
        default=DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--label-similarity-margin",
        type=float,
        default=DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
    )
    args = parser.parse_args()

    DATA_DIR = PROJECT_ROOT / "data"
    GT_PATH = DATA_DIR / "qa_ground_truth.jsonl"
    OOM_LOGS_PATH = DATA_DIR / "oom_logs.jsonl"
    RESULT_DIR = DATA_DIR / "exp_results"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    llm_name = resolve_exp2_llm_name(args.llm)
    llm_slug = slugify_model_name(llm_name)
    result_file = RESULT_DIR / f"exp2_generation_{args.model_mode}_{llm_slug}_results.csv"
    summary_file = RESULT_DIR / f"exp2_generation_{args.model_mode}_{llm_slug}_summary.csv"
    embedding_diag_file = RESULT_DIR / f"exp2_generation_{args.model_mode}_{llm_slug}_embedding_diagnostics.csv"
    embedding_calibration_file = RESULT_DIR / f"exp2_generation_{args.model_mode}_{llm_slug}_embedding_calibration.csv"

    print(f"🚀 [Experiment 2] Model Mode: {args.model_mode} | LLM: {llm_name}")

    try:
        dataset = load_jsonl(GT_PATH)
    except FileNotFoundError:
        print(f"❌ {GT_PATH} 없음"); sys.exit(1)

    try:
        log_index = build_oom_logs_index(OOM_LOGS_PATH)
    except FileNotFoundError:
        print(f"❌ {OOM_LOGS_PATH} 없음"); sys.exit(1)

    new_results = []
    embedding_rows = []

    for i, data in enumerate(tqdm(dataset, desc="평가 진행 중")):
        log_id = data.get("log_id", f"log_{i}")

        expected_oom_type = data.get("expected_oom_type", "")
        gt = data.get("ground_truth", {}) or {}
        gt_evidence = gt.get("must_include_evidence", []) or []
        gt_action = gt.get("action_guide", []) or []

        raw_log = log_index.get(log_id, "")
        if not raw_log:
            print(f"⚠ {log_id}: raw_log 없음, skip")
            continue

        if args.model_mode == "base_llm":
            generated = run_naive_llm(raw_log, llm_name=llm_name)
        else:
            generated = run_rag_agent(
                raw_log,
                label_similarity_threshold=args.label_similarity_threshold,
                label_similarity_margin=args.label_similarity_margin,
                llm_name=llm_name,
            )

        if isinstance(generated, dict):
            generated["_log_id"] = log_id

        fields = extract_fields(generated, args.model_mode)
        inference_error = extract_inference_error(generated)

        cat_score = eval_category_match(expected_oom_type, fields["oom_type"])
        ev_score = eval_evidence_recall(gt_evidence, fields["evidence"])
        ag_score = eval_action_guide_similarity(gt_action, fields["action_guide"])

        row = {
            "log_id": log_id,
            "model_mode": args.model_mode,
            "llm": llm_name,
            "expected_oom_type": expected_oom_type,
            "generated_oom_type": fields["oom_type"],
            "category_match": cat_score,
            "evidence_recall": round(ev_score, 4),
            "action_guide_similarity": round(ag_score, 4),
            "n_gt_evidence": len(gt_evidence),
            "n_gt_action": len(gt_action),
            "n_gen_evidence": len(fields["evidence"]),
            "n_gen_action": len(fields["action_guide"]),
            "had_error": int(bool(inference_error)),
            "inference_error": inference_error,
        }

        if inference_error:
            print(f"⚠ {log_id}: inference failed ({inference_error})")

        new_results.append(row)

        if args.model_mode == "rag_agent":
            embedding_row = extract_embedding_diagnostics(generated, expected_oom_type)
            if embedding_row is not None:
                embedding_rows.append(embedding_row)

    # ==========================================
    # 8. 집계
    # ==========================================

    df = pd.DataFrame(new_results)

    if df.empty:
        print("❌ 평가 가능한 결과가 없습니다.")
        df.to_csv(result_file, index=False, encoding='utf-8-sig')
        summary = pd.DataFrame([
            {
                'model_mode': args.model_mode,
                'llm': llm_name,
                'expected_oom_type': 'Overall',
                'n': 0,
                'category_match': 0.0,
                'evidence_recall': 0.0,
                'action_guide_similarity': 0.0,
            }
        ])
        summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"💾 Saved: {result_file}")
        print(f"💾 Summary saved: {summary_file}")
        if args.model_mode == "rag_agent":
            pd.DataFrame(embedding_rows).to_csv(embedding_diag_file, index=False, encoding='utf-8-sig')
            pd.DataFrame().to_csv(embedding_calibration_file, index=False, encoding='utf-8-sig')
        return

    print(f"\n{'='*60}")
    print(f"Overall ({args.model_mode}, llm={llm_name}, n={len(df)})")
    print(f"{'='*60}")
    print(f"  Category Match:          {df['category_match'].mean():.4f}")
    print(f"  Evidence Recall:         {df['evidence_recall'].mean():.4f}")
    print(f"  Action Guide Similarity: {df['action_guide_similarity'].mean():.4f}")
    print(f"  Inference Error Rate:    {df['had_error'].mean():.4f}")

    print(f"\n{'='*60}")
    print(f"By OOM Type")
    print(f"{'='*60}")
    print(f"{'oom_type':<22} {'n':>4} {'cat':>8} {'evid':>8} {'guide':>8}")
    print("-" * 60)

    for ot in sorted(df['expected_oom_type'].unique()):
        sub = df[df['expected_oom_type'] == ot]
        print(f"{ot:<22} {len(sub):>4} "
              f"{sub['category_match'].mean():>8.4f} "
              f"{sub['evidence_recall'].mean():>8.4f} "
              f"{sub['action_guide_similarity'].mean():>8.4f}")

    df.to_csv(result_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: {result_file}")

    summary = df.groupby('expected_oom_type').agg(
        n=('log_id', 'count'),
        category_match=('category_match', 'mean'),
        evidence_recall=('evidence_recall', 'mean'),
        action_guide_similarity=('action_guide_similarity', 'mean'),
    ).reset_index()
    summary.insert(0, 'llm', llm_name)
    summary.insert(0, 'model_mode', args.model_mode)

    overall_row = pd.DataFrame([{
        'model_mode': args.model_mode,
        'llm': llm_name,
        'expected_oom_type': 'Overall',
        'n': len(df),
        'category_match': df['category_match'].mean(),
        'evidence_recall': df['evidence_recall'].mean(),
        'action_guide_similarity': df['action_guide_similarity'].mean(),
    }])

    summary = pd.concat([summary, overall_row], ignore_index=True)
    summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"💾 Summary saved: {summary_file}")

    if args.model_mode == "rag_agent":
        embedding_df = pd.DataFrame(embedding_rows)
        embedding_df.to_csv(embedding_diag_file, index=False, encoding='utf-8-sig')
        print(f"💾 Embedding diagnostics saved: {embedding_diag_file}")

        calibration_df, best_config = build_embedding_calibration_summary(embedding_df)
        calibration_df.to_csv(embedding_calibration_file, index=False, encoding='utf-8-sig')
        print(f"💾 Embedding calibration saved: {embedding_calibration_file}")

        if best_config is not None:
            print("\n[Node2 Embedding Calibration]")
            print(
                "  Recommended threshold/margin: "
                f"{best_config['threshold']:.2f} / {best_config['margin']:.2f} "
                f"(category_match={best_config['category_match']:.4f}, "
                f"accepted_rate={best_config['accepted_rate']:.4f}, n={int(best_config['n'])})"
            )
        else:
            print("\n[Node2 Embedding Calibration]")
            print("  No threshold-sensitive samples were observed in this run.")

if __name__ == "__main__":
    main()
