import time
import logging
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# AI 파이프라인 함수 임포트
from app.agent.graph import invoke_oom_workflow
from app.core.llm_factory import build_chat_llm
from app.core.settings import get_settings
from app.core.vllm_manager import ensure_vllm_model
from app.network.web_client import WebClient
from app.baseline.gpt_baseline import run_gpt_baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_confidence(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip().lower()
        label_scores = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        }
        if stripped in label_scores:
            return label_scores[stripped]
        try:
            return float(stripped)
        except ValueError:
            return 0.0

    return 0.0

def normalize_action_guide(value) -> list:
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        normalized = []
        for key in ("immediate", "recommended", "further_investigation"):
            items = value.get(key, [])
            if isinstance(items, list):
                normalized.extend(items)
            elif items:
                normalized.append(items)
        return normalized

    if value:
        return [value]

    return []

def run_real_pipeline(raw_log: str, metadata: dict):
    """
    실제 LangGraph 파이프라인을 실행하고 
    결과를 RESULT 테이블 규격에 맞게 변환합니다.
    """
    settings = get_settings()
    served_model = ensure_vllm_model(settings.default_chat_model)
    llm = build_chat_llm(served_model, json_mode=True)

    """
    # 파이프라인 실행 (invoke_oom_workflow 호출)
    # extra_state에 메타데이터 전체를 넣어 Node에서 참조하게 할 수도 있습니다.
    final_state = invoke_oom_workflow(
        raw_log=raw_log, 
        metadata=metadata,
        extra_state={"llm": llm},
    )

    # LangGraph State -> ERD RESULT 테이블 매핑
    # Node 2의 분류 결과와 Node 4의 최종 진단 결과를 합칩니다.
    classification = final_state.get("classification", {})
    diagnosis = final_state.get("diagnosis", {})
    parsed = final_state.get("parsed_fields", {})
    diagnosis_detail = diagnosis.get("diagnosis", {}) if isinstance(diagnosis.get("diagnosis"), dict) else diagnosis
    action_guide = diagnosis.get("action_guide", [])

    # Web 서버가 DB에 바로 넣을 수 있는 형식으로 변환
    result_data = {
        "oom_type": classification.get("oom_type", "UNKNOWN"),
        "constraint_type": parsed.get("constraint", "UNKNOWN"),
        "confidence": normalize_confidence(classification.get("confidence", 0.0)),
        "root_cause": diagnosis_detail.get("root_cause", "진단 결과를 생성할 수 없습니다."),
        "action_guide": normalize_action_guide(action_guide)
    }

    # 중간 결과 (프론트엔드 시각화용)
    intermediate_data = {
        "node1_parsed": parsed,
        "node2_classification": classification,
        "node3_tools": final_state.get("tool_results", {})
    }

    return result_data, intermediate_data
    """
    # 우리 시스템 + GPT 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ours = executor.submit(_run_ours, raw_log, metadata, llm)
        future_gpt  = executor.submit(run_gpt_baseline, raw_log)
        ours_result, ours_intermediate = future_ours.result()
        gpt_result = future_gpt.result()  # 실패 시 None

    return ours_result, ours_intermediate, gpt_result

def _run_ours(raw_log, metadata, llm):
    """기존 run_real_pipeline 본문을 그대로 옮기되 latency_ms 추가."""
    t0 = time.time()
    final_state = invoke_oom_workflow(
        raw_log=raw_log,
        metadata=metadata,
        extra_state={"llm": llm},
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    classification = final_state.get("classification", {})
    diagnosis = final_state.get("diagnosis", {})
    parsed = final_state.get("parsed_fields", {})
    diagnosis_detail = diagnosis.get("diagnosis", {}) if isinstance(diagnosis.get("diagnosis"), dict) else diagnosis
    action_guide = diagnosis.get("action_guide", [])

    result_data = {
        "oom_type": classification.get("oom_type", "UNKNOWN"),
        "constraint_type": parsed.get("constraint", "UNKNOWN"),
        "confidence": normalize_confidence(classification.get("confidence", 0.0)),
        "root_cause": diagnosis_detail.get("root_cause", "진단 결과를 생성할 수 없습니다."),
        "action_guide": normalize_action_guide(action_guide),
        "latency_ms": elapsed_ms,   # 신규
    }
    intermediate_data = {
        "node1_parsed": parsed,
        "node2_classification": classification,
        "node3_tools": final_state.get("tool_results", {}),
    }
    return result_data, intermediate_data

def main():
    logger.info("RAGstar AI Worker가 시작되었습니다.")
    client = WebClient()

    while True:
        try:
            # Web 서버에서 'pending' 상태의 작업 가져오기
            task = client.fetch_pending_task()
            
            if task:
                diagnosis_id = task.get("diagnosis_id")
                raw_log = task.get("raw_log")
                metadata = task.get("metadata", {})
                
                logger.info(f" [Task {diagnosis_id}] 분석 시작...")
                
                # 상태를 'running'으로 업데이트
                client.update_task_status(diagnosis_id, "running")
                
                try:
                    # 실제 파이프라인 실행
                    result_data, intermediate_data, gpt_result = run_real_pipeline(raw_log, metadata)

                    # 결과 전송
                    ok = client.submit_result(diagnosis_id, result_data, intermediate_data, gpt_result)
                    if ok:
                        logger.info(f" [Task {diagnosis_id}] 분석 완료 및 결과 전송 성공")
                    else:
                        logger.warning(f" [Task {diagnosis_id}] 분석은 완료됐으나 백엔드 전송 실패")

                except Exception as e:
                    logger.error(f" [Task {diagnosis_id}] 파이프라인 내부 에러: {e}")
                    client.update_task_status(diagnosis_id, "failed")
            
            else:
                # 대기 작업이 없으면 5초간 휴식 (Polling 간격)
                time.sleep(5)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f" 워커 루프 에러: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
