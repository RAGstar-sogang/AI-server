from typing import TypedDict, Dict, Any, Optional

class OOMState(TypedDict):
    """
    파이프라인에서 노드간 전달되는 상태 정보
    """
    # [입력]
    raw_log: str # 사용자 입력 로그
    metadata: Dict[str, Any] # server_info, service, recent_changes (구조화)
    metadata_text: Optional[str] # 유저가 자유 형식으로 첨부한 컨텍스트 (Node 2/4 프롬프트에 주입)

    # Node 1 출력
    parsed_fields: Dict[str, Any] # 정규식 파싱 결과

    # Node 2 출력
    classification: Dict[str, Any] # LLM 분류 결과

    # Node 3 출력
    tool_results: Dict[str, Any] # 도구 실행 결과

    # Node 4 출력
    # 최종 3단 구조(log_analysis, diagnosis, action_guide) JSON
    diagnosis: Dict[str, Any]

    # 에러 기록
    error: Optional[str]