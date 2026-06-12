# mock_server.py
from fastapi import FastAPI
import uvicorn
import json

app = FastAPI()

# 1. AI 워커가 일감을 달라고 찌르는 엔드포인트
@app.get("/api/v1/diagnosis/pending")
def get_pending():
    print("▶ [Mock Web] AI 워커가 일감을 가져갔습니다!")
    return {
        "diagnosis_id": 999,
        "raw_log": "Out of memory: Killed process 3201 (java) total-vm:273728kB, anon-rss:875680kB",
        "metadata": {
            "server_info": "Ubuntu 22.04, Kernel 5.15",
            "service": "payment-api v2.3.1",
            "recent_changes": "Increased JVM heap to 8G"
        }
    }

# 2. AI 워커가 상태를 업데이트하는 엔드포인트
@app.patch("/api/v1/diagnosis/{diagnosis_id}/status")
def update_status(diagnosis_id: int, payload: dict):
    print(f"▶ [Mock Web] DB 상태 업데이트 됨 - Task {diagnosis_id} : {payload.get('status')}")
    return {"status": "success"}

# 3. AI 워커가 최종 분석 결과를 쏘는 엔드포인트
@app.post("/api/v1/diagnosis/{diagnosis_id}/result")
def receive_result(diagnosis_id: int, payload: dict):
    print(f"\n🎉 [Mock Web] Task {diagnosis_id} 최종 분석 결과 수신 완료 🎉")
    print("=" * 50)
    # ERD 테이블 규격에 맞게 잘 왔는지 출력해서 확인
    print(json.dumps(payload['result'], indent=2, ensure_ascii=False))
    print("=" * 50)
    return {"status": "success"}

if __name__ == "__main__":
    # 포트 충돌을 피하기 위해 8001번 포트 사용
    uvicorn.run(app, host="127.0.0.1", port=8001)