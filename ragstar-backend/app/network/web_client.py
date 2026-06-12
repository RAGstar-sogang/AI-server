import os
import requests
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

WEB_SERVER_URL = os.getenv("WEB_SERVER_URL", "http://localhost:8000")

logger = logging.getLogger(__name__)

class WebClient:
    def __init__(self):
        self.base_url = f"{WEB_SERVER_URL}/api/v1/diagnosis"

    def fetch_pending_task(self):
        """Web 서버에서 대기 중인(pending) 진단 요청을 가져옵니다."""
        try:
            response = requests.get(f"{self.base_url}/pending", timeout=5)
            if response.status_code == 200:
                return response.json() # {"diagnosis_id": 1, "raw_log": "...", "metadata": {...}}
            elif response.status_code == 404:
                return None # 대기 중인 작업 없음
        except requests.exceptions.RequestException as e:
            logger.error(f"Web 서버 통신 에러 (fetch_pending_task): {e}")
        return None

    def update_task_status(self, diagnosis_id: int, status: str):
        """Web 서버에 작업 상태(running, failed 등)를 업데이트합니다."""
        try:
            response = requests.patch(
                f"{self.base_url}/{diagnosis_id}/status",
                json={"status": status},
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"Task {diagnosis_id} 상태 업데이트 완료: {status}")
        except requests.exceptions.RequestException as e:
            logger.error(f"상태 업데이트 실패 (Task {diagnosis_id}): {e}")

    def submit_result(self, diagnosis_id: int, result_data: dict, intermediate_data: dict = None, gpt_result: dict | None = None,) -> bool:
        """최종 진단 결과와 중간 상태를 Web 서버로 전송합니다."""
        payload = {
            "result": result_data,  # ERD의 RESULT 테이블 규격
            "gpt_result": gpt_result,
            "intermediate_results": intermediate_data or {},
        }
        try:
            response = requests.post(
                f"{self.base_url}/{diagnosis_id}/result",
                json=payload,
                timeout=20, # GPT까지 포함이라 약간 여유
            )
            response.raise_for_status()
            logger.info(f"Task {diagnosis_id} 최종 결과 전송 완료")
            return True
        except requests.exceptions.RequestException as e:
            body = getattr(e.response, "text", "")[:300] if e.response is not None else ""
            logger.error(f"결과 전송 실패 (Task {diagnosis_id}): {e} body={body}")
            return False