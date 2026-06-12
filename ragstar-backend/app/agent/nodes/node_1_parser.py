import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.agent.state import OOMState


# ---------------------------------------------------------------------
# 1) Regex definitions
# ---------------------------------------------------------------------
# OOM 로그 안에서 필요한 정보를 뽑아내기 위한 정규표현식 모음이다.
# 가능한 한 실제 커널 로그의 여러 변형을 흡수하도록 조금 넉넉하게 잡아 둔다.

# OOM killer를 호출한 "트리거 프로세스"를 찾는다.
# 예: "httpd invoked oom-killer"
TRIGGER_RE = re.compile(r"\b(\S+)\s+invoked\s+oom[- ]killer\b", re.IGNORECASE)

# 트리거 라인에서 세부 정보까지 한 번에 뽑는다.
# 예: trigger_process, gfp_mask, order, oom_score_adj
TRIGGER_DETAIL_RE = re.compile(
    r"\b(\S+)\s+invoked\s+oom[- ]killer:.*?"
    r"gfp_mask=(0x[0-9a-fA-F]+).*?"
    r"order=(\d+).*?"
    r"oom_score_adj\s*[:=]\s*(-?\d+)",
    re.IGNORECASE,
)

# trigger_detail 매치에 실패했을 때 개별 필드를 보완 추출하기 위한 패턴들
GFP_MASK_RE = re.compile(r"\bgfp_mask=(0x[0-9a-fA-F]+)\b", re.IGNORECASE)
ORDER_RE = re.compile(r"\border\s*[:=]\s*(\d+)\b", re.IGNORECASE)
OOM_SCORE_ADJ_RE = re.compile(r"\boom_score_adj\s*[:=]\s*(-?\d+)\b", re.IGNORECASE)
PAGE_ALLOC_FAILURE_RE = re.compile(
    r"^(?P<trigger>.+?):\s+page allocation failure:\s+order:(?P<order>\d+),\s*mode:(?P<mode>0x[0-9a-fA-F]+)",
    re.IGNORECASE,
)

# "Killed process ..." 라인에서 종료된 프로세스 정보와 메모리 수치를 추출한다.
# 예: pid, process name, total-vm, anon-rss
FULL_KILLED_RE = re.compile(
    r"\bKilled process\s+(\d+)\s+\(([^)]+)\)"
    r".*?\btotal-vm:(\d+)kB,\s*anon-rss:(\d+)kB",
    re.IGNORECASE,
)

# total-vm/anon-rss까지 없는 축약형 kill 로그를 위한 fallback 패턴이다.
KILL_FALLBACK_RE = re.compile(
    r"\b(?:Out of memory:\s*)?Kill(?:ed)? process\s+(\d+)\s+\(([^)]+)\)",
    re.IGNORECASE,
)

# 전체 RAM 페이지 수 추출
# 예: "524288 pages RAM"
RAM_RE = re.compile(r"\b(\d+)\s+pages RAM\b", re.IGNORECASE)

# NUMA node 로그에서 free / min 추출
# free 와 min 사이에 boost, low, high 등 다른 필드가 있어도 잡도록 완화
NODE_RE = re.compile(
    r"\bNode\s+\d+\s+\S+\s+free:(\d+)kB\b.*?\bmin:(\d+)kB\b",
    re.IGNORECASE,
)

# SwapTotal / SwapFree 는 로그 포맷이 조금씩 다를 수 있어 둘 다 흡수한다.
SWAP_TOTAL_RE = re.compile(
    r"(?:\bSwapTotal:\s*|(?:\bTotal\s+swap\s*=\s*))(\d+)\s*kB\b",
    re.IGNORECASE,
)
SWAP_FREE_RE = re.compile(
    r"(?:\bSwapFree:\s*|(?:\bFree\s+swap\s*=\s*))(\d+)\s*kB\b",
    re.IGNORECASE,
)

# constraint=CONSTRAINT_NONE / CONSTRAINT_MEMCG 같은 제약 유형 추출
CONSTRAINT_RE = re.compile(r"\bconstraint=(\S+?)(?:[, ]|$)", re.IGNORECASE)

# 구형 memcg OOM 메시지에서 cgroup 경로를 추출
MEMCG_TASK_RE = re.compile(
    r"\bTask in (\S+) killed as a result of limit of (\S+)\b",
    re.IGNORECASE,
)

# 새로 추가: systemd/docker memcg 경로 추출용
MEMCG_STATS_PATH_RE = re.compile(
    r"\bMemory cgroup stats for (\S+?)(?::|$)",
    re.IGNORECASE,
)
OOM_MEMCG_PATH_RE = re.compile(
    r"\boom_memcg=(\S+?)(?:,|\s|$)",
    re.IGNORECASE,
)
TASK_MEMCG_PATH_RE = re.compile(
    r"\btask_memcg=(\S+?)(?:,|\s|$)",
    re.IGNORECASE,
)

# cgroup 사용량 / 제한 / failcnt 추출
# [수정] cgroup v1 형식과 v2 형식 모두 지원
CG_USAGE_RE = re.compile(r"\b(?:memory:\s*usage|memory\.current)\s+(\d+)(kB)?\b", re.IGNORECASE)
CG_LIMIT_RE = re.compile(r"\b(?:memory:.*?\blimit|memory\.max)\s+(\d+)(kB)?\b", re.IGNORECASE)
CG_FAILCNT_RE = re.compile(r"\bfailcnt\s+(\d+)\b", re.IGNORECASE)
CG_SWAP_USAGE_RE = re.compile(r"\b(?:swap:\s*usage|memory\.swap\.current)\s+(\d+)(kB)?\b", re.IGNORECASE)
CG_SWAP_LIMIT_RE = re.compile(r"\b(?:swap:.*?\blimit|memory\.swap\.max)\s+(\d+)(kB)?\b", re.IGNORECASE)
CG_SWAP_FAILCNT_RE = re.compile(r"\bswap:.*?\bfailcnt\s+(\d+)\b", re.IGNORECASE)

# 커널 버전 추출용 패턴들
# 로그 종류에 따라 나타나는 표현이 달라 여러 후보를 준비한다.
LINUX_VERSION_RE = re.compile(r"\bLinux version (\S+)")
TAINTED_VERSION_RE = re.compile(
    r"\b(?:Not tainted|Tainted(?::.*?)?)\s+(\d+\.\d+\.\d+[^\s#]*)\s+#"
)
GENERIC_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+[^\s#]*)\s+#\d")

# syslog prefix 제거용
# 예: "Apr 12 10:00:00 host kernel: ..."
SYSLOG_PREFIX_RE = re.compile(
    r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d\d:\d\d:\d\d\s+\S+\s+kernel:\s+"
)

# 대괄호 timestamp 제거용
# 예: "[11686.040460]" 또는 일부 다른 커널 prefix
BRACKET_TS_RE = re.compile(
    r"^\[(?:\s*\d+\.\d+|[A-Z][a-z]{2}\s+[A-Z][a-z]{2}.*?)\]\s*"
)


# ---------------------------------------------------------------------
# 2) Structured event view
# ---------------------------------------------------------------------
# 하나의 OOM 이벤트를 여러 줄의 묶음으로 표현하기 위한 데이터 클래스다.
# 한 로그 안에 OOM 이벤트가 여러 번 있을 수 있기 때문에,
# 먼저 이벤트 단위로 나누고 그중 "대표 이벤트"를 고르는 흐름으로 간다.

@dataclass
class OOMEvent:
    lines: List[str]

    @property
    def text(self) -> str:
        # 이벤트의 전체 텍스트를 한 문자열로 합쳐 반환한다.
        return "\n".join(self.lines)

    def has_trigger(self) -> bool:
        # 이 이벤트 안에 "invoked oom-killer" 라인이 있는지 검사한다.
        return any(TRIGGER_RE.search(line) for line in self.lines)

    def has_kill(self) -> bool:
        # 이 이벤트 안에 실제 kill line이 있는지 검사한다.
        # FULL_KILLED_RE가 더 강한 매치이고, 없으면 fallback kill line도 허용한다.
        return any(
            FULL_KILLED_RE.search(line) or KILL_FALLBACK_RE.search(line)
            for line in self.lines
        )

    def score(self) -> int:
        # 이벤트의 정보량/신뢰도를 대략 점수화한다.
        # 대표 이벤트를 선택할 때 fallback 기준으로 사용된다.
        text = self.text
        score = 0
        if TRIGGER_RE.search(text):
            score += 3
        if FULL_KILLED_RE.search(text):
            score += 5
        elif KILL_FALLBACK_RE.search(text):
            score += 2
        if CONSTRAINT_RE.search(text):
            score += 1
        if MEMCG_TASK_RE.search(text):
            score += 2
        if RAM_RE.search(text):
            score += 1
        if NODE_RE.search(text):
            score += 1
        return score


# ---------------------------------------------------------------------
# 3) Baseline schema
# ---------------------------------------------------------------------
# 파싱 결과의 기본 스키마를 만든다.
# 파싱에 실패하거나 로그에 값이 없어도 항상 같은 구조를 유지하기 위해
# 모든 키를 미리 채워 둔다.

def _empty_parsed_fields() -> Dict[str, Any]:
    return {
        "trigger_process": None,
        "killed_process": None,
        "killed_pid": None,
        "total_vm_kb": None,
        "anon_rss_kb": None,
        "oom_score_adj": 0,
        "total_ram_pages": None,
        "node_free_kb": None,
        "node_min_kb": None,
        "swap_total_kb": None,
        "swap_free_kb": None,
        "constraint": None,
        "gfp_mask": None,
        "order": None,
        "cgroup_path": None,
        "cgroup_usage_kb": None,
        "cgroup_limit_kb": None,
        "cgroup_failcnt": None,
        "cgroup_swap_usage_kb": None,
        "cgroup_swap_limit_kb": None,
        "cgroup_swap_failcnt": None,
        "process_table": [],
    }


# ---------------------------------------------------------------------
# 4) Normalization
# ---------------------------------------------------------------------
# 원본 로그에서 syslog prefix, bracket timestamp 같은 잡음을 제거해
# 실제 파싱 대상 텍스트를 정리한다.

def _normalize_line(line: str) -> str:
    # 줄 끝 개행 제거
    line = line.rstrip("\n")
    # syslog prefix 제거
    line = SYSLOG_PREFIX_RE.sub("", line)
    # [123.456] 같은 bracket prefix 제거
    line = BRACKET_TS_RE.sub("", line)
    # 좌우 공백 정리
    return line.strip()


def _normalize_log(raw_log: str) -> List[str]:
    # 전체 raw log를 줄 단위로 정규화하고,
    # 비어 있는 줄은 제거한다.
    lines = []
    for raw_line in raw_log.splitlines():
        normalized = _normalize_line(raw_line)
        if normalized:
            lines.append(normalized)
    return lines


# ---------------------------------------------------------------------
# 5) Segmentation / event selection
# ---------------------------------------------------------------------
# 한 로그 안에 OOM 이벤트가 여러 번 섞여 있을 수 있으므로,
# "trigger line" 기준으로 이벤트를 쪼갠 뒤 대표 이벤트를 선택한다.

def _segment_events(lines: List[str]) -> List[OOMEvent]:
    # trigger line의 인덱스를 모두 찾는다.
    trigger_indices = [i for i, line in enumerate(lines) if TRIGGER_RE.search(line)]

    # trigger가 0개 또는 1개면 전체를 하나의 이벤트로 간주한다.
    if len(trigger_indices) <= 1:
        return [OOMEvent(lines=lines)] if lines else []

    # trigger가 여러 개면 trigger ~ 다음 trigger 직전까지를 하나의 이벤트로 자른다.
    events: List[OOMEvent] = []
    for idx, start in enumerate(trigger_indices):
        end = trigger_indices[idx + 1] if idx + 1 < len(trigger_indices) else len(lines)
        chunk = lines[start:end]
        if chunk:
            events.append(OOMEvent(lines=chunk))
    return events


def _select_primary_event(events: List[OOMEvent]) -> Optional[OOMEvent]:
    # 이벤트가 없으면 None 반환
    if not events:
        return None

    # 이벤트가 하나면 그대로 대표 이벤트로 사용
    if len(events) == 1:
        return events[0]

    # 여러 개면 우선 실제 kill line이 있는 이벤트를 고른다.
    for event in events:
        if event.has_kill():
            return event

    # 그래도 애매하면 점수가 가장 높은 이벤트를 대표 이벤트로 선택
    return max(events, key=lambda e: e.score())


# ---------------------------------------------------------------------
# 6) Field extraction helpers
# ---------------------------------------------------------------------
# 각 필드 그룹별로 추출 책임을 나눈 helper들이다.
# parse_oom_log()에서는 이 helper들을 순서대로 호출해 parsed dict를 채운다.

def _extract_trigger_fields(text: str, parsed: Dict[str, Any]) -> None:
    # 가장 정보량이 많은 상세 trigger 패턴부터 시도한다.
    match = TRIGGER_DETAIL_RE.search(text)
    if match:
        parsed["trigger_process"] = match.group(1)
        parsed["gfp_mask"] = match.group(2)
        parsed["order"] = int(match.group(3))
        parsed["oom_score_adj"] = int(match.group(4))
        return

    # page allocation failure 헤더는 invoked oom-killer가 없으므로 별도 처리한다.
    page_alloc = PAGE_ALLOC_FAILURE_RE.search(text)
    if page_alloc:
        parsed["trigger_process"] = page_alloc.group("trigger").strip()
        parsed["gfp_mask"] = page_alloc.group("mode")
        parsed["order"] = int(page_alloc.group("order"))
        return

    # 상세 패턴이 실패하면 개별 패턴들로 부분 보완한다.
    trigger = TRIGGER_RE.search(text)
    if trigger:
        parsed["trigger_process"] = trigger.group(1)

    gfp = GFP_MASK_RE.search(text)
    if gfp:
        parsed["gfp_mask"] = gfp.group(1)

    order = ORDER_RE.search(text)
    if order:
        parsed["order"] = int(order.group(1))

    oom_adj = OOM_SCORE_ADJ_RE.search(text)
    if oom_adj:
        parsed["oom_score_adj"] = int(oom_adj.group(1))


def _extract_killed_fields(text: str, parsed: Dict[str, Any]) -> None:
    # full killed pattern이면 pid/process name 뿐 아니라 vm/rss까지 같이 저장한다.
    match = FULL_KILLED_RE.search(text)
    if match:
        parsed["killed_pid"] = int(match.group(1))
        parsed["killed_process"] = match.group(2)
        parsed["total_vm_kb"] = int(match.group(3))
        parsed["anon_rss_kb"] = int(match.group(4))
        return

    # full pattern이 없으면 최소한 pid와 process name만 잡는다.
    fallback = KILL_FALLBACK_RE.search(text)
    if fallback:
        parsed["killed_pid"] = int(fallback.group(1))
        parsed["killed_process"] = fallback.group(2)


def _extract_memory_fields(text: str, parsed: Dict[str, Any]) -> None:
    # 전체 RAM 페이지 수
    ram = RAM_RE.search(text)
    if ram:
        parsed["total_ram_pages"] = int(ram.group(1))

    # NUMA node free/min
    node = NODE_RE.search(text)
    if node:
        parsed["node_free_kb"] = int(node.group(1))
        parsed["node_min_kb"] = int(node.group(2))

    # swap total/free
    swap_total = SWAP_TOTAL_RE.search(text)
    if swap_total:
        parsed["swap_total_kb"] = int(swap_total.group(1))

    swap_free = SWAP_FREE_RE.search(text)
    if swap_free:
        parsed["swap_free_kb"] = int(swap_free.group(1))


def _extract_cgroup_fields(text: str, parsed: Dict[str, Any]) -> None:
    def _parse_kb_value(match: Optional[re.Match]) -> Optional[int]:
        if not match:
            return None
        val = int(match.group(1))
        unit = match.group(2)
        # 단위가 명시되지 않은 경우(cgroup v2의 바이트 출력) kB로 환산
        if not unit:
            val = val // 1024
        return val
    
    # constraint 종류 추출
    constraint = CONSTRAINT_RE.search(text)
    if constraint:
        parsed["constraint"] = constraint.group(1)

    # cgroup usage / limit / failcnt 추출
    parsed["cgroup_usage_kb"] = _parse_kb_value(CG_USAGE_RE.search(text)) or parsed["cgroup_usage_kb"]
    parsed["cgroup_limit_kb"] = _parse_kb_value(CG_LIMIT_RE.search(text)) or parsed["cgroup_limit_kb"]

    failcnt = CG_FAILCNT_RE.search(text)
    if failcnt:
        parsed["cgroup_failcnt"] = int(failcnt.group(1))

    parsed["cgroup_swap_usage_kb"] = _parse_kb_value(CG_SWAP_USAGE_RE.search(text)) or parsed["cgroup_swap_usage_kb"]
    parsed["cgroup_swap_limit_kb"] = _parse_kb_value(CG_SWAP_LIMIT_RE.search(text)) or parsed["cgroup_swap_limit_kb"]
    
    swap_failcnt = CG_SWAP_FAILCNT_RE.search(text)
    if swap_failcnt:
        parsed["cgroup_swap_failcnt"] = int(swap_failcnt.group(1))

    # 1) 가장 신뢰도 높은 구형 포맷
    # "Task in X killed as a result of limit of Y" 형식이면 바로 경로 사용
    task_line = MEMCG_TASK_RE.search(text)
    if task_line:
        parsed["cgroup_path"] = task_line.group(1)
        return

    # 2) memcg OOM일 때만 아래 경로 후보들을 사용
    # constraint가 MEMCG 이거나, cgroup 수치가 있거나,
    # Memory cgroup stats 블록이 있으면 memcg 관련 OOM으로 판단한다.
    is_memcg_oom = (
        parsed.get("constraint") == "CONSTRAINT_MEMCG"
        or parsed.get("cgroup_usage_kb") is not None
        or parsed.get("cgroup_limit_kb") is not None
        or "Memory cgroup stats for " in text
    )

    if not is_memcg_oom:
        return

    # stats path가 있으면 우선 사용
    stats_path = MEMCG_STATS_PATH_RE.search(text)
    if stats_path:
        parsed["cgroup_path"] = stats_path.group(1)
        return

    # oom_memcg가 / 가 아니면 사용
    oom_memcg = OOM_MEMCG_PATH_RE.search(text)
    if oom_memcg and oom_memcg.group(1) != "/":
        parsed["cgroup_path"] = oom_memcg.group(1)
        return

    # task_memcg가 / 가 아니면 사용
    task_memcg = TASK_MEMCG_PATH_RE.search(text)
    if task_memcg and task_memcg.group(1) != "/":
        parsed["cgroup_path"] = task_memcg.group(1)


def _parse_process_table(lines: List[str]) -> List[Dict[str, Any]]:
    # Tasks state 블록의 각 줄을 파싱해서 process_table을 만든다.
    # 최종 결과는 rss_kb 기준 내림차순 정렬한다.
    processes: List[Dict[str, Any]] = []

    for line in lines:
        # 예:
        # [ 3201] 1000 3201 11111 218920 0 0 0 java
        match = re.match(r"^\[\s*(\d+)\]\s+(.+?)\s+(\S+)$", line)
        if not match:
            continue

        pid = int(match.group(1))
        numeric_blob = match.group(2)
        name = match.group(3)

        # 중간 숫자 덩어리에서 필요한 숫자를 순서로 해석한다.
        ints = [int(x) for x in re.findall(r"-?\d+", numeric_blob)]
        if len(ints) < 4:
            continue

        uid = ints[0]
        rss_pages = ints[3]
        oom_score_adj = ints[-1]

        processes.append(
            {
                "pid": pid,
                "uid": uid,
                "name": name,
                "rss_kb": rss_pages * 4,
                "oom_score_adj": oom_score_adj,
            }
        )

    return sorted(processes, key=lambda x: x["rss_kb"], reverse=True)


def _extract_kernel_version(normalized_text: str) -> Optional[str]:
    # 여러 버전 패턴을 차례대로 시도해서
    # 첫 번째로 성공한 커널 버전을 반환한다.
    for pattern in (LINUX_VERSION_RE, TAINTED_VERSION_RE, GENERIC_VERSION_RE):
        match = pattern.search(normalized_text)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------
# 7) Validation / conservative consistency cleanup
# ---------------------------------------------------------------------
# 파싱 후 최소한의 일관성 정리를 적용한다.
# 공격적으로 값을 바꾸지는 않고, 모순이 뚜렷한 경우만 정리한다.

def _apply_consistency_rules(parsed: Dict[str, Any]) -> None:
    # killed_process가 없으면 killed_pid도 신뢰하지 않는다.
    if parsed["killed_process"] is None:
        parsed["killed_pid"] = None

    # 공백 문자열은 None으로 정리
    for key, value in list(parsed.items()):
        if isinstance(value, str) and not value.strip():
            parsed[key] = None


# ---------------------------------------------------------------------
# 8) Public parser API
# ---------------------------------------------------------------------
# 외부에서 실제로 호출하는 공개 API 구간이다.
# parse_oom_log()는 순수 파서,
# node_1_parser()는 state를 받아 LangGraph 노드처럼 동작하는 wrapper다.

def parse_oom_log(raw_log: str) -> Dict[str, Any]:
    # 기본 스키마 준비
    parsed = _empty_parsed_fields()

    # 로그 정규화
    normalized_lines = _normalize_log(raw_log)

    # 이벤트 단위로 분리 후 대표 이벤트 선택
    events = _segment_events(normalized_lines)
    primary_event = _select_primary_event(events)

    # 대표 이벤트 텍스트와 전체 텍스트를 각각 준비
    # - 대표 이벤트: 실제 핵심 필드 추출용
    # - 전체 텍스트: 커널 버전 같은 전체 로그 범위 정보 추출용
    event_text = primary_event.text if primary_event else ""
    all_text = "\n".join(normalized_lines)

    # 대표 이벤트에서 핵심 필드 추출
    _extract_trigger_fields(event_text, parsed)
    _extract_killed_fields(event_text, parsed)
    _extract_memory_fields(event_text, parsed)
    _extract_cgroup_fields(event_text, parsed)

    # process table은 대표 이벤트의 lines에서만 파싱
    parsed["process_table"] = _parse_process_table(primary_event.lines if primary_event else [])

    # 최소 일관성 정리
    _apply_consistency_rules(parsed)

    # kernel version은 전체 로그 텍스트 기준으로 추출
    parsed["kernel_version"] = _extract_kernel_version(all_text)

    return parsed


def node_1_parser(state: OOMState) -> OOMState:
    # state에서 raw_log를 가져와 순수 파서로 처리
    raw_log = state.get("raw_log", "")
    parsed_fields = parse_oom_log(raw_log)

    # 로그에서 kernel_version을 못 찾았으면
    # 사용자가 별도로 넣어 둔 metadata로 보완
    if parsed_fields["kernel_version"] is None:
        metadata = state.get("metadata", {})
        parsed_fields["kernel_version"] = metadata.get("server_info")

    # 기존 state는 유지하고 parsed_fields만 갱신해서 반환
    return {**state, "parsed_fields": parsed_fields}