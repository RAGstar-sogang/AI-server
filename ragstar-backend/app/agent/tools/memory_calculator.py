from typing import Dict, Any

def memory_calculator(parsed_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    로그의 메모리 수치를 계산하여 사용률과 위험도를 분석합니다.
    (Linux 페이지 단위 변환 및 oom_score_adj 기반 보호 여부 확인 포함)
    """
    try:
        def _num(value: Any, default: float = 0.0) -> float:
            try:
                if value is None:
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        def _strict_num(value: Any) -> float:
            if value is None:
                raise ValueError("numeric value is missing")
            return float(value)

        # 1. 시스템 전체 메모리 계산 (Linux 페이지 크기 = 4KB)
        ram_total_kb = _strict_num(parsed_fields.get("total_ram_pages")) * 4
        ram_total_mb = ram_total_kb / 1024

        process_table = parsed_fields.get("process_table", [])
        if not isinstance(process_table, list):
            raise TypeError("process_table must be a list")

        results = []
        # 2. 프로세스 테이블 순회 및 개별 수치 계산
        for proc in process_table:
            if not isinstance(proc, dict):
                raise TypeError("each process entry must be a dict")

            rss_mb = _strict_num(proc.get("rss_kb", 0)) / 1024
            
            # 0 나누기 방지
            ram_pct = round(rss_mb / ram_total_mb * 100, 1) if ram_total_mb > 0 else 0.0

            results.append({
                "name": proc.get("name", "unknown"),
                "pid": proc.get("pid", 0),
                "rss_mb": round(rss_mb, 1),
                "ram_pct": ram_pct,
                "protected": proc.get("oom_score_adj", 0) <= -900 
            })

        # 3. RSS 기준으로 정렬 후 상위 5개 추출
        top5 = sorted(results, key=lambda x: x["rss_mb"], reverse=True)[:5]

        # 4. Swap 상태 계산
        swap_total_raw = parsed_fields.get("swap_total_kb")
        swap_free_raw = parsed_fields.get("swap_free_kb")
        swap_total_kb = _num(swap_total_raw)

        if swap_total_raw is None:
            swap_status = "unknown"
        elif swap_total_kb == 0:
            swap_status = "disabled"
        else:
            swap_status = "enabled"

        return {
            "top_processes": top5,
            "total_top5_pct": round(sum(p["ram_pct"] for p in top5), 1),
            "ram_total_mb": round(ram_total_mb, 1),
            "swap_total_mb": round(swap_total_kb / 1024, 1),
            "swap_free_mb": round(_num(swap_free_raw) / 1024, 1),
            "swap_status": swap_status
        }

    except Exception as e:
        # 에러 발생 시 파이프라인 중단 방지용 Fallback
        return {
            "error": f"Memory calculation error: {str(e)}",
            "top_processes": [],
            "total_top5_pct": 0.0,
            "ram_total_mb": 0.0,
            "swap_total_mb": 0.0,
            "swap_free_mb": 0.0,
            "swap_status": "unknown"
        }