import os
import json
import re
from typing import Any, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate

from app.agent.state import OOMState
from app.core.llm_factory import build_node4_synthesizer_llm


# ============================================================
# Node 4: Synthesizer (Schema Repair + Fallback)
# ============================================================
# 설계 원칙
# 1. 정상 경로에서는 LLM이 reasoning_trace + final_answer envelope 전체를 생성한다.
# 2. root_cause / contributing_factors / severity / evidence / actions는
#    LLM 출력을 그대로 사용한다.
# 3. JSON/schema drift가 발생하면 우선 schema repair를 시도한다.
# 4. repair가 불가능하면 contract를 만족하는 fallback bundle을 생성한다.
# 5. 다만 llm.invoke 자체 예외는 그대로 전파한다.
# 6. 기존 호출부 호환을 위해 generate_diagnosis_with_retry() 이름은 유지하지만,
#    실제로는 retry 없이 한 번만 시도한다.
#
# 현재 팀 contract 기준 추가 원칙
# 7. reasoning_trace는 내부 bundle 구조에는 유지한다.
# 8. 하지만 LangGraph state에는 diagnosis만 저장하고,
#    diagnosis_trace는 state에 넣지 않는다.
# 9. fallback 경로가 사용되면 diagnosis.llm_failed=True 로 표시한다.
# ============================================================


# Node 4 프롬프트는 app/agent/prompts/node_4_template.txt에서만 로드된다.
# 치환 변수: {parsed_fields}, {classification}, {tool_results}, {user_metadata}.
# 파일이 없으면 즉시 실패한다 (fallback 없음).


# ------------------------------------------------------------
# 예외 타입
# ------------------------------------------------------------
class Node4Error(Exception):
    """Node 4 전용 기본 예외."""


class Node4PromptError(Node4Error):
    """프롬프트 생성 단계에서 발생한 예외."""


class Node4LLMOutputError(Node4Error):
    """LLM 응답 파싱/추출 단계에서 발생한 예외."""


class Node4SchemaError(Node4Error):
    """LLM 응답 JSON의 스키마가 잘못되었을 때 발생하는 예외."""


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _stringify(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    return str(value).strip() or default


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _ensure_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item.strip() for item in value if _is_non_empty_string(item)]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _normalize_or_repair_string_list(value: Any, fallback: list[str]) -> list[str]:
    """
    문자열 리스트 계약을 맞추되,
    명시적으로 전달된 빈 리스트는 유효 값으로 보존한다.
    """
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if not value:
            return []
        normalized = [item.strip() for item in value if _is_non_empty_string(item)]
        return normalized or fallback

    normalized = _ensure_string_list(value)
    return normalized or fallback


def _normalize_severity(value: Any, default: str = "high") -> str:
    severity = _stringify(value, default=default).lower()
    if severity not in {"high", "medium", "low"}:
        return default
    return severity


def _build_fallback_key_metrics(
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
) -> Dict[str, str]:
    total_ram_pages = _safe_int(parsed_fields.get("total_ram_pages"))
    total_ram = (
        f"{total_ram_pages * 4}KB"
        if total_ram_pages is not None
        else "unknown"
    )

    swap_total = _safe_int(parsed_fields.get("swap_total_kb"))
    swap_free = _safe_int(parsed_fields.get("swap_free_kb"))
    if swap_total is None:
        swap_status = "unknown"
    elif swap_total == 0:
        swap_status = "not configured"
    elif swap_free is None:
        swap_status = f"configured ({swap_total}KB total, free unknown)"
    else:
        swap_status = f"configured ({swap_free}KB free / {swap_total}KB total)"

    killed_process = _stringify(parsed_fields.get("killed_process"), default="unknown")
    constraint_type = _stringify(parsed_fields.get("constraint"), default="unknown")
    kill_reason = _stringify(classification.get("oom_type"), default="unknown")

    return {
        "total_ram": total_ram,
        "swap_status": swap_status,
        "killed_process": killed_process,
        "kill_reason": kill_reason,
        "constraint_type": constraint_type,
    }


def _build_fallback_reasoning_trace(
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
    tool_results: Dict[str, Any],
    failure_reason: str,
) -> Dict[str, list[str]]:
    facts = []
    killed_process = _stringify(parsed_fields.get("killed_process"))
    if killed_process:
        facts.append(f"killed_process={killed_process}")
    oom_type = _stringify(classification.get("oom_type"))
    if oom_type:
        facts.append(f"classified_oom_type={oom_type}")
    constraint = _stringify(parsed_fields.get("constraint"))
    if constraint:
        facts.append(f"constraint={constraint}")

    return {
        "facts": facts or ["Node 4 fallback path was used."],
        "causal_inference": [
            "The OOM diagnosis had to be reconstructed from parsed fields and tool results because the live Node 4 response was not schema-safe.",
        ],
        "kb_application": [
            f"Available tool result groups: {', '.join(sorted(tool_results.keys())) or 'none'}.",
        ],
        "decision_basis": [
            failure_reason,
            "A fallback bundle was generated to preserve the pipeline contract.",
        ],
    }


def _build_fallback_final_answer(
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
    tool_results: Dict[str, Any],
    failure_reason: str,
) -> Dict[str, Any]:
    key_metrics = _build_fallback_key_metrics(parsed_fields, classification)
    oom_type = _stringify(classification.get("oom_type"), default="unknown")
    killed_process = key_metrics["killed_process"]
    constraint_type = key_metrics["constraint_type"]

    evidence = []
    if constraint_type != "unknown":
        evidence.append(f"constraint={constraint_type}")
    if parsed_fields.get("anon_rss_kb") is not None:
        evidence.append(f"anon-rss={parsed_fields['anon_rss_kb']}kB")
    if parsed_fields.get("cgroup_limit_kb") is not None:
        evidence.append(f"cgroup_limit={parsed_fields['cgroup_limit_kb']}kB")

    contributing_factors = []
    if tool_results:
        contributing_factors.append(
            f"Tool results were available from: {', '.join(sorted(tool_results.keys()))}."
        )
    contributing_factors.append("Node 4 response schema drift triggered the fallback path.")

    return {
        "log_analysis": {
            "summary": (
                f"The parsed log indicates a {oom_type} event involving {killed_process}. "
                "This answer was reconstructed from Node 1-3 outputs because the live Node 4 response was not schema-safe."
            ),
            "key_metrics": key_metrics,
        },
        "diagnosis": {
            "root_cause": (
                f"The available parsed evidence is most consistent with {oom_type}. "
                f"Constraint={constraint_type} and the extracted process/memory signals indicate the OOM diagnosis should be based on upstream pipeline results."
            ),
            "contributing_factors": contributing_factors,
            "evidence": evidence or ["Node 4 fallback path was used."],
            "severity": "high",
            "llm_failed": True,
            "llm_failure_reason": failure_reason,
        },
        "action_guide": {
            "immediate": [
                "Review the raw Node 1/2/3 outputs for this case because Node 4 fallback was triggered.",
            ],
            "recommended": [
                "Retry Node 4 generation or inspect the prompt/response pair for schema drift.",
            ],
            "further_investigation": [
                "Capture the raw LLM response when fallback occurs to analyze recurrent schema failures.",
            ],
        },
    }


def _repair_reasoning_trace(
    candidate: Any,
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
    tool_results: Dict[str, Any],
    failure_reason: str,
) -> Dict[str, list[str]]:
    if not isinstance(candidate, dict):
        return _build_fallback_reasoning_trace(parsed_fields, classification, tool_results, failure_reason)

    fallback = _build_fallback_reasoning_trace(parsed_fields, classification, tool_results, failure_reason)
    repaired = _preserve_extra_fields(
        {
            "facts": _normalize_or_repair_string_list(candidate.get("facts"), fallback["facts"]),
            "causal_inference": _normalize_or_repair_string_list(candidate.get("causal_inference"), fallback["causal_inference"]),
            "kb_application": _normalize_or_repair_string_list(candidate.get("kb_application"), fallback["kb_application"]),
            "decision_basis": _normalize_or_repair_string_list(candidate.get("decision_basis"), fallback["decision_basis"]),
        },
        candidate,
    )

    return repaired


def _preserve_extra_fields(
    normalized: Dict[str, Any],
    original: Dict[str, Any],
) -> Dict[str, Any]:
    """
    계약 필드는 normalized 값을 우선 사용하되,
    계약 외 필드는 원본 payload에서 보존한다.
    """
    extras = {key: value for key, value in original.items() if key not in normalized}
    return {**normalized, **extras}


def _repair_final_answer(
    candidate: Any,
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
    tool_results: Dict[str, Any],
    failure_reason: str,
    *,
    mark_llm_failed: bool,
) -> Dict[str, Any]:
    fallback = _build_fallback_final_answer(parsed_fields, classification, tool_results, failure_reason)
    if not isinstance(candidate, dict):
        return fallback

    log_analysis = candidate.get("log_analysis") if isinstance(candidate.get("log_analysis"), dict) else {}
    diagnosis = candidate.get("diagnosis") if isinstance(candidate.get("diagnosis"), dict) else {}
    action_guide = candidate.get("action_guide") if isinstance(candidate.get("action_guide"), dict) else {}
    key_metrics = log_analysis.get("key_metrics") if isinstance(log_analysis.get("key_metrics"), dict) else {}
    diagnosis_original = dict(diagnosis)

    if not mark_llm_failed:
        diagnosis_original.pop("llm_failed", None)
        diagnosis_original.pop("llm_failure_reason", None)

    repaired_diagnosis = {
        "root_cause": _stringify(diagnosis.get("root_cause"), fallback["diagnosis"]["root_cause"]),
        "contributing_factors": _normalize_or_repair_string_list(
            diagnosis.get("contributing_factors"),
            fallback["diagnosis"]["contributing_factors"],
        ),
        "evidence": _normalize_or_repair_string_list(
            diagnosis.get("evidence"),
            fallback["diagnosis"]["evidence"],
        ),
        "severity": _normalize_severity(diagnosis.get("severity"), default=fallback["diagnosis"]["severity"]),
    }
    if mark_llm_failed:
        repaired_diagnosis["llm_failed"] = True
        repaired_diagnosis["llm_failure_reason"] = failure_reason

    repaired = {
        "log_analysis": _preserve_extra_fields(
            {
                "summary": _stringify(
                    log_analysis.get("summary"),
                    default=fallback["log_analysis"]["summary"],
                ),
                "key_metrics": _preserve_extra_fields(
                    {
                        "total_ram": _stringify(key_metrics.get("total_ram"), fallback["log_analysis"]["key_metrics"]["total_ram"]),
                        "swap_status": _stringify(key_metrics.get("swap_status"), fallback["log_analysis"]["key_metrics"]["swap_status"]),
                        "killed_process": _stringify(key_metrics.get("killed_process"), fallback["log_analysis"]["key_metrics"]["killed_process"]),
                        "kill_reason": _stringify(key_metrics.get("kill_reason"), fallback["log_analysis"]["key_metrics"]["kill_reason"]),
                        "constraint_type": _stringify(key_metrics.get("constraint_type"), fallback["log_analysis"]["key_metrics"]["constraint_type"]),
                    },
                    key_metrics,
                ),
            },
            log_analysis,
        ),
        "diagnosis": _preserve_extra_fields(repaired_diagnosis, diagnosis_original),
        "action_guide": _preserve_extra_fields(
            {
                "immediate": _normalize_or_repair_string_list(
                    action_guide.get("immediate"),
                    fallback["action_guide"]["immediate"],
                ),
                "recommended": _normalize_or_repair_string_list(
                    action_guide.get("recommended"),
                    fallback["action_guide"]["recommended"],
                ),
                "further_investigation": _normalize_or_repair_string_list(
                    action_guide.get("further_investigation"),
                    fallback["action_guide"]["further_investigation"],
                ),
            },
            action_guide,
        ),
    }

    return repaired


def _repair_or_build_fallback_bundle(
    parsed: Any,
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
    tool_results: Dict[str, Any],
    failure_reason: str,
    *,
    mark_llm_failed: bool,
) -> Dict[str, Any]:
    original_top_level = parsed if isinstance(parsed, dict) else {}

    if isinstance(parsed, dict):
        reasoning_trace_candidate = parsed.get("reasoning_trace")
        final_answer_candidate = parsed.get("final_answer") if "final_answer" in parsed else parsed
    else:
        final_answer_candidate = parsed
        reasoning_trace_candidate = None

    bundle = {
        "reasoning_trace": _repair_reasoning_trace(
            reasoning_trace_candidate,
            parsed_fields,
            classification,
            tool_results,
            failure_reason,
        ),
            "final_answer": _repair_final_answer(
                final_answer_candidate,
                parsed_fields,
                classification,
                tool_results,
                failure_reason,
                mark_llm_failed=mark_llm_failed,
            ),
    }
    bundle = _preserve_extra_fields(bundle, original_top_level)

    if not _validate_llm_bundle_schema(bundle):
        raise Node4SchemaError(
            "Node 4 fallback bundle could not satisfy the required schema."
        )

    return bundle


def _validate_string_list(candidate: Any) -> bool:
    return isinstance(candidate, list) and all(_is_non_empty_string(item) for item in candidate)


# ------------------------------------------------------------
# 프롬프트 로더
# ------------------------------------------------------------
def load_prompt(filename: str) -> str:
    """
    prompts 폴더에서 텍스트 파일을 읽어온다.
    파일이 없으면 즉시 fail-fast (fallback 없음).
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_path, "..", "prompts", filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------
# 스키마 검증 유틸
# ------------------------------------------------------------
def _validate_reasoning_trace_schema(candidate: Any) -> bool:
    """
    reasoning_trace 구조를 검사한다.
    """
    if not isinstance(candidate, dict):
        return False

    required = {"facts", "causal_inference", "kb_application", "decision_basis"}
    if not required.issubset(candidate.keys()):
        return False

    return (
        _validate_string_list(candidate.get("facts"))
        and _validate_string_list(candidate.get("causal_inference"))
        and _validate_string_list(candidate.get("kb_application"))
        and _validate_string_list(candidate.get("decision_basis"))
    )



def _validate_final_answer_schema(candidate: Any) -> bool:
    """
    사용자에게 보여줄 final_answer 구조를 검사한다.
    """
    if not isinstance(candidate, dict):
        return False

    if not {"log_analysis", "diagnosis", "action_guide"}.issubset(candidate.keys()):
        return False

    log_analysis = candidate.get("log_analysis")
    diagnosis = candidate.get("diagnosis")
    action_guide = candidate.get("action_guide")

    if not isinstance(log_analysis, dict):
        return False
    if not isinstance(diagnosis, dict):
        return False
    if not isinstance(action_guide, dict):
        return False

    if not {"summary", "key_metrics"}.issubset(log_analysis.keys()):
        return False
    if not _is_non_empty_string(log_analysis.get("summary")):
        return False

    key_metrics = log_analysis.get("key_metrics")
    if not isinstance(key_metrics, dict):
        return False

    required_metrics = {
        "total_ram",
        "swap_status",
        "killed_process",
        "kill_reason",
        "constraint_type",
    }
    if not required_metrics.issubset(key_metrics.keys()):
        return False
    if not all(_is_non_empty_string(key_metrics.get(key)) for key in required_metrics):
        return False

    required_diagnosis = {
        "root_cause",
        "contributing_factors",
        "evidence",
        "severity",
    }
    if not required_diagnosis.issubset(diagnosis.keys()):
        return False

    if not _is_non_empty_string(diagnosis.get("root_cause")):
        return False
    if not _validate_string_list(diagnosis.get("contributing_factors")):
        return False
    if not _validate_string_list(diagnosis.get("evidence")):
        return False
    if diagnosis.get("severity") not in {"high", "medium", "low"}:
        return False

    required_action_guide = {"immediate", "recommended", "further_investigation"}
    if not required_action_guide.issubset(action_guide.keys()):
        return False

    if not _validate_string_list(action_guide.get("immediate")):
        return False
    if not _validate_string_list(action_guide.get("recommended")):
        return False
    if not _validate_string_list(action_guide.get("further_investigation")):
        return False

    return True



def _validate_llm_bundle_schema(candidate: Any) -> bool:
    """
    LLM이 반환한 envelope 구조를 검사한다.
    """
    if not isinstance(candidate, dict):
        return False

    if not {"reasoning_trace", "final_answer"}.issubset(candidate.keys()):
        return False

    return (
        _validate_reasoning_trace_schema(candidate.get("reasoning_trace"))
        and _validate_final_answer_schema(candidate.get("final_answer"))
    )


def _get_llm_bundle_failure_reason(candidate: Any) -> str | None:
    if not isinstance(candidate, dict):
        return "Node 4 LLM response did not contain a valid JSON object."

    missing_keys = [key for key in ["reasoning_trace", "final_answer"] if key not in candidate]
    if missing_keys:
        if missing_keys == ["reasoning_trace"]:
            return "Node 4 LLM response JSON is missing the required top-level envelope keys; missing key: reasoning_trace."
        if missing_keys == ["final_answer"]:
            return "Node 4 LLM response JSON is missing the required top-level envelope keys; missing key: final_answer."
        return (
            "Node 4 LLM response JSON is missing the required top-level envelope keys: "
            "reasoning_trace and final_answer."
        )

    if not _validate_reasoning_trace_schema(candidate.get("reasoning_trace")):
        return (
            "Node 4 LLM response contains the top-level envelope, but reasoning_trace failed internal schema/type validation."
        )

    if not _validate_final_answer_schema(candidate.get("final_answer")):
        return (
            "Node 4 LLM response contains the top-level envelope, but final_answer failed internal schema/type validation."
        )

    return None


def _find_json_dict_candidates(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    return candidates


def _select_preferred_json_dict(candidates: list[dict]) -> Optional[dict]:
    if not candidates:
        return None

    for candidate in candidates:
        if {"reasoning_trace", "final_answer"}.issubset(candidate.keys()):
            return candidate

    for candidate in candidates:
        if "final_answer" in candidate and isinstance(candidate.get("final_answer"), dict):
            return candidate

    return candidates[0]


# ------------------------------------------------------------
# JSON 추출 유틸
# ------------------------------------------------------------
def extract_json(text: str) -> Optional[dict]:
    """
    LLM 응답에서 JSON object를 추출한다.

    처리 순서:
    1. 모든 ```json ... ``` code block에서 dict 후보를 수집한다.
    2. 전체 텍스트에서도 dict 후보를 수집한다.
    3. 합쳐진 후보 전체에 대해 envelope 우선 선택을 한 번만 적용한다.
    """
    if not text or not isinstance(text, str):
        return None

    candidates: list[dict] = []

    fenced_block_pattern = re.compile(
        r"```[ \t]*([A-Za-z0-9_+-]+)?[ \t]*\n([\s\S]*?)\s*```",
        flags=re.IGNORECASE,
    )

    for code_block in fenced_block_pattern.finditer(text):
        label = (code_block.group(1) or "").strip().lower()
        if label and label != "json":
            continue

        candidate = code_block.group(2).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                candidates.append(parsed)
        except json.JSONDecodeError:
            pass

        candidates.extend(_find_json_dict_candidates(candidate))

    candidates.extend(_find_json_dict_candidates(text))

    return _select_preferred_json_dict(candidates)


# ------------------------------------------------------------
# LLM 유틸
# ------------------------------------------------------------
def _build_default_llm() -> Any:
    """
    Node 2와 동일한 방식으로 기본 LLM을 생성한다.
    """
    return build_node4_synthesizer_llm()



def _build_prompt(
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
    tool_results: Dict[str, Any],
    user_metadata: str = "",
) -> str:
    """
    prompt file + ChatPromptTemplate를 사용해 최종 프롬프트를 만든다.
    user_metadata는 유저가 자유 형식으로 첨부한 컨텍스트 (없으면 placeholder).
    """
    try:
        system_prompt_text = load_prompt("node_4_template.txt")
        prompt_template = ChatPromptTemplate.from_template(system_prompt_text)

        prompt = prompt_template.format(
            parsed_fields=json.dumps(parsed_fields, indent=2, ensure_ascii=False),
            classification=json.dumps(classification, indent=2, ensure_ascii=False),
            tool_results=json.dumps(tool_results, indent=2, ensure_ascii=False),
            user_metadata=(user_metadata or "").strip() or "None provided.",
        )
        return str(prompt)
    except Exception as exc:
        raise Node4PromptError(f"Failed to build Node 4 prompt: {exc}") from exc



def _serialize_structured_response(value: Any) -> str:
    """
    dict/list 같은 구조화 응답을 문자열(JSON)로 바꾼다.
    """
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)



def _invoke_llm(llm: Any, prompt: str) -> str:
    """
    LLM invoke 결과를 문자열로 정규화한다.
    """
    response = llm.invoke(prompt)

    if isinstance(response, (dict, list)):
        return _serialize_structured_response(response)

    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, (dict, list)):
            return _serialize_structured_response(content)
        return str(content)

    return str(response)


# ------------------------------------------------------------
# bundle 생성 (normalized + fallback)
# ------------------------------------------------------------
def generate_diagnosis_bundle_with_retry(
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
    tool_results: Dict[str, Any],
    llm: Optional[Any] = None,
    max_retries: int = 2,
    user_metadata: str = "",
) -> Dict[str, Any]:
    """
    Node 4의 핵심 bundle 생성 함수.

    함수 이름은 기존 호환을 위해 유지하지만,
    실제 동작은 retry 없이 한 번만 시도한다.
    `max_retries`는 하위 호환용 deprecated 인자다.

    반환 구조:
    {
      "reasoning_trace": {...},
      "final_answer": {...}
    }

    llm.invoke 예외는 그대로 전파한다.
    JSON/schema drift는 repair/fallback bundle로 복구한다.
    """
    del max_retries

    active_llm = llm if llm is not None else _build_default_llm()
    prompt = _build_prompt(parsed_fields, classification, tool_results, user_metadata=user_metadata)
    raw_response = _invoke_llm(active_llm, prompt)
    parsed = extract_json(raw_response)

    if parsed is None:
        return _repair_or_build_fallback_bundle(
            parsed=None,
            parsed_fields=parsed_fields,
            classification=classification,
            tool_results=tool_results,
            failure_reason="Node 4 LLM response did not contain a valid JSON object.",
            mark_llm_failed=True,
        )

    failure_reason = _get_llm_bundle_failure_reason(parsed)

    return _repair_or_build_fallback_bundle(
        parsed=parsed,
        parsed_fields=parsed_fields,
        classification=classification,
        tool_results=tool_results,
        failure_reason=failure_reason or "Node 4 response normalized successfully.",
        mark_llm_failed=failure_reason is not None,
    )


# ------------------------------------------------------------
# 하위 호환용 기존 API
# ------------------------------------------------------------
def generate_diagnosis_with_retry(
    parsed_fields: Dict[str, Any],
    classification: Dict[str, Any],
    tool_results: Dict[str, Any],
    llm: Optional[Any] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    하위 호환용 API.

    기존 호출부/테스트는 최종 diagnosis dict만 기대하므로,
    bundle 생성 결과 중 final_answer만 반환한다.

    llm.invoke 자체 실패가 아니면 normalized final_answer를 반환한다.
    """
    bundle = generate_diagnosis_bundle_with_retry(
        parsed_fields=parsed_fields,
        classification=classification,
        tool_results=tool_results,
        llm=llm,
        max_retries=max_retries,
    )
    return bundle["final_answer"]


# ------------------------------------------------------------
# Node wrapper
# ------------------------------------------------------------
def node_4_synthesizer(state: OOMState) -> OOMState:
    """
    LangGraph 파이프라인용 Node 4 wrapper.

    역할:
    - state에서 parsed_fields / classification / tool_results를 꺼낸다.
    - bundle-first 구조로 generate_diagnosis_bundle_with_retry()를 호출한다.
    - bundle 내부에서는 reasoning_trace + final_answer를 유지한다.
    - 하지만 팀 contract에 맞춰 state에는 diagnosis만 저장한다.

    llm.invoke 자체 실패는 그대로 전파한다.
    schema drift / non-JSON은 내부 fallback으로 복구한다.
    """
    parsed_fields = state.get("parsed_fields", {}) or {}
    classification = state.get("classification", {}) or {}
    tool_results = state.get("tool_results", {}) or {}
    llm = state.get("llm")
    user_metadata = state.get("metadata_text") or ""

    bundle = generate_diagnosis_bundle_with_retry(
        parsed_fields=parsed_fields,
        classification=classification,
        tool_results=tool_results,
        llm=llm,
        user_metadata=user_metadata,
    )

    return {
        **state,
        "diagnosis": bundle["final_answer"],
    }
