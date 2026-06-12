import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# experiments/exp2_generation.py:331 에서 그대로 가져옴
NAIVE_PROMPT_TEMPLATE = """You are a Linux OOM diagnosis expert.
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
"""

def _extract_balanced_json_objects(text: str) -> list[str]:
    """Find every balanced top-level {...} substring in `text`.

    Walks the text tracking brace depth (respecting string literals + escapes).
    Returns each top-level object as a raw substring.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
    return out

def _build_gpt_llm():
    from langchain_community.chat_models.openai import ChatOpenAI
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=(os.environ.get("OPENAI_API_KEY") or "").rstrip("."),
        temperature=0,
        timeout=120,
        max_retries=2,
        model_kwargs={
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 8192,
        },
    )

def run_gpt_baseline(raw_log: str) -> dict | None:
    """GPT를 호출해 동일 스키마로 진단 결과 반환.
    실패 시 None을 반환해서 워커가 우리 결과만 보내도록.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY 미설정 — GPT baseline 스킵")
        return None

    try:
        llm = _build_gpt_llm()
        t0 = time.time()
        response = llm.invoke(NAIVE_PROMPT_TEMPLATE.format(raw_log=raw_log))
        elapsed_ms = int((time.time() - t0) * 1000)
        text = response.content if hasattr(response, "content") else str(response)

        # JSON 파싱
        for raw in reversed(_extract_balanced_json_objects(text)):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and ("classification" in parsed or "final_answer" in parsed):
                    return _to_result_schema(parsed, elapsed_ms)
            except Exception:
                continue
        logger.warning(f"GPT 응답에서 JSON 파싱 실패: {text[:200]}")
        return None
    except Exception as e:
        logger.error(f"GPT baseline 호출 실패: {e}")
        return None

def _to_result_schema(parsed: dict, latency_ms: int) -> dict:
    """GPT raw 응답 → 백엔드 RESULT 스키마와 동일한 형태로 변환."""
    classification = parsed.get("classification", {}) or {}
    final = parsed.get("final_answer", {}) or {}
    diagnosis = final.get("diagnosis", {}) or {}
    action_guide = final.get("action_guide", {}) or {}

    # 우리 worker.py의 normalize_action_guide와 동일한 로직
    ag_list: list[str] = []
    for key in ("immediate", "recommended", "further_investigation"):
        items = action_guide.get(key, []) or []
        if isinstance(items, list):
            ag_list.extend(str(x) for x in items if x)

    return {
        "oom_type": classification.get("oom_type", "UNKNOWN"),
        "constraint_type": "UNKNOWN",  # GPT 프롬프트엔 constraint 필드 없음
        "confidence": 0.9,             # GPT가 confidence 안 줌 → 고정값 or None
        "root_cause": diagnosis.get("root_cause", ""),
        "action_guide": ag_list,
        "latency_ms": latency_ms,
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    }
