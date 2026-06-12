import os
import sys
import json
import pytest

# -----------------------------------------------------------------------------
# 프로젝트 루트 import 경로 설정
# -----------------------------------------------------------------------------
# tests/ 디렉터리에서 실행해도 app/... 모듈을 import할 수 있도록
# 프로젝트 루트를 sys.path에 추가한다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# -----------------------------------------------------------------------------
# 테스트 대상 모듈 import
# -----------------------------------------------------------------------------
from app.agent.nodes.node_1_parser import node_1_parser


# -----------------------------------------------------------------------------
# 디버그 출력 헬퍼
# -----------------------------------------------------------------------------
def _pretty_json(data) -> str:
    """
    JSON 직렬화 가능한 객체를 사람이 읽기 쉬운 문자열로 변환한다.
    """
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except TypeError:
        return str(data)


def dump_case_result(title: str, before_state: dict, parsed_fields: dict):
    """
    테스트 케이스별 입력 state와 Node 1 결과를 출력한다.
    """
    print(f"\n{'=' * 100}")
    print(f"[CASE] {title}")
    print(f"{'=' * 100}")

    print("\n[INPUT STATE]")
    print(_pretty_json(before_state))

    print("\n[OUTPUT parsed_fields]")
    print(_pretty_json(parsed_fields))


def dump_strict_table(title: str, expected: dict, actual: dict, keys_to_compare: list[str]):
    """
    지정한 키들에 대해 expected / actual 비교 표를 출력한다.
    """
    print(f"\n🧪 [Strict Scalar Check] {title}")
    print("=" * 110)
    print(f"{'Field':<24} | {'Expected':<30} | {'Actual':<30} | {'Result'}")
    print("-" * 110)

    for key in keys_to_compare:
        exp_val = expected.get(key)
        act_val = actual.get(key)
        matched = (exp_val == act_val) and (type(exp_val) == type(act_val))
        icon = "✅ Pass" if matched else "❌ Fail"
        print(f"{key:<24} | {str(exp_val):<30} | {str(act_val):<30} | {icon}")

    print("-" * 110)


# -----------------------------------------------------------------------------
# 테스트용 state / baseline 생성기
# -----------------------------------------------------------------------------
def create_mock_state(raw_log: str, server_info: str | None = None) -> dict:
    """
    현재 팀 state.py contract를 따르는 Node 1 입력 state를 생성한다.

    포인트:
    - Node 1만 테스트하더라도 state 전체 형태는 현재 OOMState에 맞춰 둔다.
    - Node 1은 parsed_fields만 갱신하고 나머지 state는 그대로 유지해야 한다.
    """
    return {
        "raw_log": raw_log,
        "metadata": {"server_info": server_info} if server_info else {},
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }


def get_parser_baseline() -> dict:
    """
    현재 node_1_parser의 _empty_parsed_fields() 기준 baseline.

    중요:
        - parser 기본값은 oom_score_adj=0, swap_total_kb=None, swap_free_kb=None,
      process_table=[] 형태다.
    """
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
        "kernel_version": None,
    }


def read_log_from_subdir(filename: str) -> str:
    """
    tests/node_1_logs/<filename> 파일을 읽어 온다.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "node_1_logs", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------------------------------------------------------
# 검증 유틸
# -----------------------------------------------------------------------------
def assert_parser_shape(parsed: dict):
    """
    Node 1이 현재 구현 기준의 최소 parsed_fields 스키마를 만족하는지 검사한다.
    """
    required_keys = {
        "trigger_process",
        "killed_process",
        "killed_pid",
        "total_vm_kb",
        "anon_rss_kb",
        "oom_score_adj",
        "total_ram_pages",
        "node_free_kb",
        "node_min_kb",
        "swap_total_kb",
        "swap_free_kb",
        "constraint",
        "gfp_mask",
        "order",
        "cgroup_path",
        "cgroup_usage_kb",
        "cgroup_limit_kb",
        "cgroup_failcnt",
        "cgroup_swap_usage_kb",
        "cgroup_swap_limit_kb",
        "cgroup_swap_failcnt",
        "process_table",
        "kernel_version",
    }

    assert required_keys.issubset(parsed.keys())
    assert isinstance(parsed["process_table"], list)


def assert_scalar_fields_strict(expected_updates: dict, actual: dict, title: str):
    """
    baseline + expected_updates를 합친 뒤,
    명시적으로 기대한 scalar 필드들만 엄격 비교한다.

    process_table은 구조가 길고 케이스별 변화가 크므로
    이 함수에서는 별도 비교 대상에서 제외한다.
    """
    expected = get_parser_baseline()
    expected.update(expected_updates)

    scalar_keys = [k for k in expected_updates.keys() if k != "process_table"]
    dump_strict_table(title, expected, actual, scalar_keys)

    for key in scalar_keys:
        exp_val = expected[key]
        act_val = actual.get(key)
        assert (exp_val == act_val) and (type(exp_val) == type(act_val)), (
            f"Field mismatch for '{key}': expected={exp_val!r} ({type(exp_val).__name__}), "
            f"actual={act_val!r} ({type(act_val).__name__})"
        )


def assert_process_table_min_length(actual: dict, min_length: int):
    """
    process_table 최소 길이를 검사한다.

    Node 1의 process_table은 로그에 실제 프로세스 행이 있을 때만 채워지므로,
    케이스별로 exact match 대신 최소 길이 검사로 다루는 편이 안정적이다.
    """
    assert isinstance(actual.get("process_table"), list)
    assert len(actual["process_table"]) >= min_length, (
        f"process_table length mismatch: expected at least {min_length}, "
        f"actual={len(actual['process_table'])}"
    )


def assert_state_passthrough(before_state: dict, after_state: dict):
    """
    Node 1 wrapper가 parsed_fields만 갱신하고 나머지 state는 유지하는지 검사한다.
    """
    assert after_state["raw_log"] == before_state["raw_log"]
    assert after_state["metadata"] == before_state["metadata"]
    assert after_state["classification"] == before_state["classification"]
    assert after_state["tool_results"] == before_state["tool_results"]
    assert after_state["diagnosis"] == before_state["diagnosis"]
    assert after_state["error"] == before_state["error"]


# -----------------------------------------------------------------------------
# 공통 케이스 실행기
# -----------------------------------------------------------------------------
def run_case(
    case_id: int,
    filename: str,
    expected_updates: dict,
    server_info: str | None = None,
    min_process_table_len: int | None = None,
):
    """
    로그 파일 기반 Node 1 테스트 공통 실행기.

    매개변수:
    - case_id: 사람이 읽기 쉬운 케이스 번호
    - filename: tests/node_1_logs 안의 로그 파일명
    - expected_updates: baseline 대비 기대값
    - user_kernel_version: 로그에 버전이 없을 때 fallback 검증용 옵션
    - min_process_table_len: process_table 최소 길이 기대치
    """
    raw_content = read_log_from_subdir(filename)
    before_state = create_mock_state(raw_content, server_info=server_info)
    after_state = node_1_parser(before_state)
    parsed = after_state["parsed_fields"]

    dump_case_result(f"Case {case_id}: {filename}", before_state, parsed)

    assert_parser_shape(parsed)
    assert_state_passthrough(before_state, after_state)
    assert_scalar_fields_strict(expected_updates, parsed, f"Case {case_id}: {filename}")

    if min_process_table_len is not None:
        assert_process_table_min_length(parsed, min_process_table_len)


# -----------------------------------------------------------------------------
# 테스트
# -----------------------------------------------------------------------------
def test_node_1_wrapper_preserves_state_and_updates_only_parsed_fields():
    """
    Node 1 wrapper가 현재 state.py contract를 따르면서
    parsed_fields만 채워 넣는지 검증한다.
    """
    raw_log = """
[11686.040460] httpd invoked oom-killer: gfp_mask=0x280da, order=0, oom_score_adj=0
[11686.040466] CPU: 2 PID: 3244 Comm: httpd Not tainted 4.18.0-305.el8.x86_64 #1
[11686.040500] Node 0 Normal free:7296kB min:7360kB low:9200kB high:11040kB
[11686.040515] Swap:  SwapTotal:       0 kB   SwapFree:        0 kB
[11686.040523] 524288 pages RAM
[11686.040530] Tasks state (memory values in pages):
[11686.040532] [  pid  ]   uid  tgid total_vm      rss pgtables_bytes swapents oom_score_adj name
[11686.040549] [   3201]     0  3201    68432   218920  1789952        0             0 java
[11686.040552] [   3244]    48  3244    65120    52340   430080        0             0 httpd
[11686.040567] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,
               global_oom,task_memcg=/,task=java,pid=3201,uid=0,pgtables=1789952,score=836
[11686.040575] Out of memory: Killed process 3201 (java) total-vm:273728kB, anon-rss:875680kB, file-rss:0kB, shmem-rss:0kB, UID:0
    """.strip()

    before_state = create_mock_state(raw_log)
    after_state = node_1_parser(before_state)
    parsed = after_state["parsed_fields"]

    dump_case_result("wrapper passthrough sanity check", before_state, parsed)

    assert_parser_shape(parsed)
    assert_state_passthrough(before_state, after_state)
    assert parsed["trigger_process"] == "httpd"
    assert parsed["killed_process"] == "java"
    assert parsed["killed_pid"] == 3201
    assert parsed["kernel_version"] == "4.18.0-305.el8.x86_64"
    assert len(parsed["process_table"]) >= 2


def test_node_1_uses_user_kernel_version_as_fallback_when_log_has_no_version():
    """
    로그 안에 kernel_version이 없으면 user_kernel_version으로 보완하는지 검증한다.
    """
    raw_log = """
[100.000001] python invoked oom-killer: gfp_mask=0x201da, order=0, oom_score_adj=0
[100.000002] Node 0 Normal free:1024kB min:2048kB low:4096kB high:8192kB
[100.000003] 262144 pages RAM
[100.000004] SwapTotal: 0 kB
[100.000005] SwapFree: 0 kB
[100.000006] Out of memory: Killed process 999 (python) total-vm:123456kB, anon-rss:65432kB, file-rss:0kB, shmem-rss:0kB
    """.strip()

    before_state = create_mock_state(raw_log, server_info="5.15.0-76-generic")
    after_state = node_1_parser(before_state)
    parsed = after_state["parsed_fields"]

    dump_case_result("kernel version fallback", before_state, parsed)

    assert parsed["kernel_version"] == "5.15.0-76-generic"


def test_node_1_keeps_swap_free_unknown_when_free_swap_line_is_missing():
    """
    Total swap만 있고 Free swap 라인이 없으면 swap_free_kb를 0으로 단정하지 않아야 한다.
    """
    raw_log = """
[100.000001] java invoked oom-killer: gfp_mask=0x201da, order=0, oom_score_adj=0
[100.000002] CPU: 0 PID: 1234 Comm: java Not tainted 6.6.0-test #1
[100.000003] Total swap = 2097152kB
[100.000004] 262144 pages RAM
[100.000005] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task=java,pid=1234,uid=0
[100.000006] Out of memory: Killed process 1234 (java) total-vm:123456kB, anon-rss:65432kB, file-rss:0kB, shmem-rss:0kB
    """.strip()

    before_state = create_mock_state(raw_log)
    after_state = node_1_parser(before_state)
    parsed = after_state["parsed_fields"]

    dump_case_result("swap free unknown when line missing", before_state, parsed)

    assert parsed["swap_total_kb"] == 2097152
    assert parsed["swap_free_kb"] is None


def test_node_1_parses_cgroup_swap_metrics_when_present():
    """
    cgroup swap usage/limit/failcnt 라인이 있으면 별도 필드로 파싱해야 한다.
    """
    raw_log = """
[100.000001] python invoked oom-killer: gfp_mask=0x201da, order=0, oom_score_adj=0
[100.000002] memory: usage 2097152kB, limit 2097152kB, failcnt 257237
[100.000003] swap: usage 1048576kB, limit 1048576kB, failcnt 5
[100.000004] oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=/,mems_allowed=0,oom_memcg=/docker/test,task_memcg=/docker/test,task=python,pid=999,uid=0
[100.000005] Memory cgroup out of memory: Killed process 999 (python) total-vm:123456kB, anon-rss:65432kB, file-rss:0kB, shmem-rss:0kB
    """.strip()

    before_state = create_mock_state(raw_log)
    after_state = node_1_parser(before_state)
    parsed = after_state["parsed_fields"]

    dump_case_result("cgroup swap metrics parsing", before_state, parsed)

    assert parsed["cgroup_usage_kb"] == 2097152
    assert parsed["cgroup_limit_kb"] == 2097152
    assert parsed["cgroup_failcnt"] == 257237
    assert parsed["cgroup_swap_usage_kb"] == 1048576
    assert parsed["cgroup_swap_limit_kb"] == 1048576
    assert parsed["cgroup_swap_failcnt"] == 5


def test_node_1_parses_page_allocation_failure_header_fields():
    """
    page allocation failure 로그에서도 trigger/order/mode를 뽑아야 한다.
    """
    raw_log = """
[Wed Mar 12 14:22:31 2025] mlx5_core 0000:04:00.0: page allocation failure: order:3, mode:0x24040c0(GFP_KERNEL|__GFP_COMP|__GFP_NORETRY), nodemask=(null),cpuset=/,mems_allowed=0
[Wed Mar 12 14:22:31 2025] CPU: 14 PID: 3847 Comm: kworker/14:2 Tainted: G E 5.15.0-91-generic #101-Ubuntu
[Wed Mar 12 14:22:31 2025] Call Trace:
[Wed Mar 12 14:22:31 2025]  warn_alloc+0x118/0x1b0
    """.strip()

    before_state = create_mock_state(raw_log)
    after_state = node_1_parser(before_state)
    parsed = after_state["parsed_fields"]

    dump_case_result("page allocation failure header parsing", before_state, parsed)

    assert parsed["trigger_process"] == "mlx5_core 0000:04:00.0"
    assert parsed["gfp_mask"] == "0x24040c0"
    assert parsed["order"] == 3
    assert parsed["kernel_version"] == "5.15.0-91-generic"


def test_node_1_parses_page_allocation_failure_order_colon_format():
    """
    order:2 형식도 잡아서 Node2가 page_alloc_failure를 결정할 수 있어야 한다.
    """
    raw_log = """
[Sat Jun 15 09:14:22 2024] z_wr_iss/0: page allocation failure: order:2, mode:0x40cc0(GFP_KERNEL|__GFP_COMP), nodemask=(null),cpuset=/,mems_allowed=0
[Sat Jun 15 09:14:22 2024] CPU: 3 PID: 1892 Comm: z_wr_iss/0 Tainted: P OE 6.1.0-21-amd64 #1 Debian 6.1.90-1
[Sat Jun 15 09:14:22 2024] Call Trace:
[Sat Jun 15 09:14:22 2024]  warn_alloc+0x118/0x1b0
    """.strip()

    before_state = create_mock_state(raw_log)
    after_state = node_1_parser(before_state)
    parsed = after_state["parsed_fields"]

    dump_case_result("page allocation failure order colon format", before_state, parsed)

    assert parsed["trigger_process"] == "z_wr_iss/0"
    assert parsed["gfp_mask"] == "0x40cc0"
    assert parsed["order"] == 2
    assert parsed["kernel_version"] == "6.1.0-21-amd64"


@pytest.mark.parametrize(
    "case_id, filename, expected_updates, min_process_table_len",
    [
        (
            1,
            "case1_global.txt",
            {
                "trigger_process": "httpd",
                "killed_process": "java",
                "killed_pid": 3201,
                "total_vm_kb": 273728,
                "anon_rss_kb": 875680,
                "oom_score_adj": 0,
                "total_ram_pages": 524288,
                "node_free_kb": 7296,
                "node_min_kb": 7360,
                "swap_total_kb": 0,
                "swap_free_kb": 0,
                "constraint": "CONSTRAINT_NONE",
                "gfp_mask": "0x280da",
                "order": 0,
                "kernel_version": "4.18.0-305.el8.x86_64",
            },
            1,
        ),
        (
            2,
            "case2_cgroup.txt",
            {
                "trigger_process": "s1-agent",
                "killed_process": "s1-agent",
                "killed_pid": 13331,
                "total_vm_kb": 2617284,
                "anon_rss_kb": 1024000,
                "oom_score_adj": 0,
                "cgroup_path": "/agent",
                "cgroup_usage_kb": 1048576,
                "cgroup_limit_kb": 1048576,
                "cgroup_failcnt": 1559756,
                "gfp_mask": "0xd0",
                "order": 0,
                "swap_total_kb": None,
                "swap_free_kb": None,
                "kernel_version": "3.10.0-957.21.3.el7.x86_64",
            },
            0,
        ),
        (
            3,
            "case3_log017.txt",
            {
                "trigger_process": "httpd",
                "killed_process": "httpd",
                "killed_pid": 1842,
                "total_vm_kb": 4523128,
                "anon_rss_kb": 3891204,
                "oom_score_adj": 0,
                "total_ram_pages": 524288,
                "node_free_kb": 8240,
                "node_min_kb": 15280,
                "swap_total_kb": 0,
                "swap_free_kb": 0,
                "constraint": "CONSTRAINT_NONE",
                "gfp_mask": "0x6200ca",
                "order": 0,
                "kernel_version": "5.15.0-91-generic",
            },
            0,
        ),
        (
            4,
            "case4_keystone.txt",
            {
                "trigger_process": "keystone-all",
                "killed_process": "keystone-all",
                "killed_pid": 43805,
                "total_vm_kb": 4446352,
                "anon_rss_kb": 4053140,
                "oom_score_adj": 0,
                "gfp_mask": "0x280da",
                "order": 0,
                "swap_total_kb": None,
                "swap_free_kb": None,
                "kernel_version": "3.10.0-327.13.1.el7.x86_64",
            },
            0,
        ),
        (
            5,
            "case5_flasherav.txt",
            {
                "trigger_process": "flasherav",
                "killed_process": "flasherav",
                "killed_pid": 2603,
                "total_vm_kb": 1498536,
                "anon_rss_kb": 721784,
                "oom_score_adj": 0,
                "total_ram_pages": 262100,
                "swap_total_kb": 524284,
                "swap_free_kb": 0,
                "gfp_mask": "0x201da",
                "order": 0,
                "kernel_version": "3.0.0-12-generic",
            },
            0,
        ),
        (
            6,
            "case6_memleak.txt",
            {
                "killed_process": "invoke_memleak",
                "killed_pid": 1604,
                "oom_score_adj": 0,
                "total_vm_kb": 10281936,
                "anon_rss_kb": 896204,
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": None,
                "swap_free_kb": None,
            },
            0,
        ),
        (
            7,
            "case7_storm.txt",
            {
                "trigger_process": "modprobe",
                "killed_process": "systemd-stdout-",
                "killed_pid": 355,
                "total_vm_kb": 23208,
                "anon_rss_kb": 0,
                "oom_score_adj": 0,
                "total_ram_pages": 2097136,
                "node_free_kb": 15876,
                "node_min_kb": 128,
                "swap_total_kb": 1023996,
                "swap_free_kb": 0,
                "gfp_mask": "0x201da",
                "order": 0,
                "kernel_version": "3.4.0-rc4+",
            },
            1,
        ),
        (
            8,
            "case8_java_high_ram.txt",
            {
                "trigger_process": "telegraf",
                "killed_process": "java",
                "killed_pid": 6033,
                "total_vm_kb": 29930040,
                "anon_rss_kb": 10625048,
                "oom_score_adj": 0,
                "total_ram_pages": 9437070,
                "swap_total_kb": 8191996,
                "swap_free_kb": 0,
                "gfp_mask": "0x201da",
                "order": 0,
            },
            1,
        ),
        (
            9,
            "case9_malloc_global_oom.txt",
            {
                "trigger_process": "node",
                "killed_process": "malloc",
                "killed_pid": 2505,
                "total_vm_kb": 1116860,
                "anon_rss_kb": 1104768,
                "oom_score_adj": 0,
                "total_ram_pages": 524174,
                "node_free_kb": 7840,
                "node_min_kb": 356,
                "swap_total_kb": 0,
                "swap_free_kb": 0,
                "constraint": "CONSTRAINT_NONE",
                "gfp_mask": "0x140cca",
                "order": 0,
                "cgroup_path": None,
                "cgroup_usage_kb": None,
                "cgroup_limit_kb": None,
                "cgroup_failcnt": None,
                "kernel_version": "6.8.0-107-generic",
            },
            1,
        ),
        (
            10,
            "case10_malloc_cgroup_oom.txt",
            {
                "trigger_process": "runc:[2:INIT]",
                "killed_process": "node",
                "killed_pid": 7825,
                "total_vm_kb": 11764120,
                "anon_rss_kb": 39804,
                "oom_score_adj": 0,
                "total_ram_pages": None,
                "node_free_kb": None,
                "node_min_kb": None,
                "swap_total_kb": None,
                "swap_free_kb": None,
                "constraint": "CONSTRAINT_MEMCG",
                "gfp_mask": "0xcc0",
                "order": 0,
                "cgroup_path": "/system.slice/docker-126fd9c6e3bd822b75b1df9547687ab42e8e9ae1379f175087e20065c31bb816.scope",
                "cgroup_usage_kb": 131072,
                "cgroup_limit_kb": 131072,
                "cgroup_failcnt": 4573503,
                "kernel_version": "6.8.0-107-generic",
            },
            1,
        ),
    ],
)
def test_node_1_parser_cases(case_id, filename, expected_updates, min_process_table_len):
    """
    실제 로그 파일 기반 strict regression test.

    각 케이스는 baseline + 기대값 업데이트 방식으로 비교하고,
    process_table은 최소 길이 기준으로 검증한다.
    """
    run_case(
        case_id=case_id,
        filename=filename,
        expected_updates=expected_updates,
        min_process_table_len=min_process_table_len,
    )


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 스크립트 실행 모드
    # -------------------------------------------------------------------------
    # pytest 없이 빠르게 눈으로 확인하고 싶을 때 사용할 수 있다.
    #
    # 예:
    #   python tests/test_node_1.py
    # -------------------------------------------------------------------------
    print("🚀 Node 1 Parser strict regression test를 시작합니다.")

    test_cases = [
        (1, "case1_global.txt", {
            "trigger_process": "httpd", "killed_process": "java", "killed_pid": 3201,
            "total_vm_kb": 273728, "anon_rss_kb": 875680, "oom_score_adj": 0,
            "total_ram_pages": 524288, "node_free_kb": 7296, "node_min_kb": 7360,
            "swap_total_kb": 0, "swap_free_kb": 0, "constraint": "CONSTRAINT_NONE",
            "gfp_mask": "0x280da", "order": 0, "kernel_version": "4.18.0-305.el8.x86_64"
        }, 1),
        (2, "case2_cgroup.txt", {
            "trigger_process": "s1-agent", "killed_process": "s1-agent", "killed_pid": 13331,
            "total_vm_kb": 2617284, "anon_rss_kb": 1024000, "oom_score_adj": 0,
            "cgroup_path": "/agent", "cgroup_usage_kb": 1048576, "cgroup_limit_kb": 1048576,
            "cgroup_failcnt": 1559756, "gfp_mask": "0xd0", "order": 0, "swap_total_kb": None,
            "swap_free_kb": None, "kernel_version": "3.10.0-957.21.3.el7.x86_64"
        }, 0),
        (3, "case3_log017.txt", {
            "trigger_process": "httpd", "killed_process": "httpd", "killed_pid": 1842,
            "total_vm_kb": 4523128, "anon_rss_kb": 3891204, "oom_score_adj": 0,
            "total_ram_pages": 524288, "node_free_kb": 8240, "node_min_kb": 15280,
            "swap_total_kb": 0, "swap_free_kb": 0, "constraint": "CONSTRAINT_NONE",
            "gfp_mask": "0x6200ca", "order": 0, "kernel_version": "5.15.0-91-generic"
        }, 0),
        (4, "case4_keystone.txt", {
            "trigger_process": "keystone-all", "killed_process": "keystone-all", "killed_pid": 43805,
            "total_vm_kb": 4446352, "anon_rss_kb": 4053140, "oom_score_adj": 0,
            "gfp_mask": "0x280da", "order": 0, "swap_total_kb": None, "swap_free_kb": None,
            "kernel_version": "3.10.0-327.13.1.el7.x86_64"
        }, 0),
        (5, "case5_flasherav.txt", {
            "trigger_process": "flasherav", "killed_process": "flasherav", "killed_pid": 2603,
            "total_vm_kb": 1498536, "anon_rss_kb": 721784, "oom_score_adj": 0,
            "total_ram_pages": 262100, "swap_total_kb": 524284, "swap_free_kb": 0,
            "gfp_mask": "0x201da", "order": 0, "kernel_version": "3.0.0-12-generic"
        }, 0),
        (6, "case6_memleak.txt", {
            "killed_process": "invoke_memleak", "killed_pid": 1604, "oom_score_adj": 0,
            "total_vm_kb": 10281936, "anon_rss_kb": 896204, "constraint": "CONSTRAINT_NONE",
            "swap_total_kb": None, "swap_free_kb": None
        }, 0),
        (7, "case7_storm.txt", {
            "trigger_process": "modprobe", "killed_process": "systemd-stdout-", "killed_pid": 355,
            "total_vm_kb": 23208, "anon_rss_kb": 0, "oom_score_adj": 0, "total_ram_pages": 2097136,
            "node_free_kb": 15876, "node_min_kb": 128, "swap_total_kb": 1023996, "swap_free_kb": 0,
            "gfp_mask": "0x201da", "order": 0, "kernel_version": "3.4.0-rc4+"
        }, 1),
        (8, "case8_java_high_ram.txt", {
            "trigger_process": "telegraf", "killed_process": "java", "killed_pid": 6033,
            "total_vm_kb": 29930040, "anon_rss_kb": 10625048, "oom_score_adj": 0,
            "total_ram_pages": 9437070, "swap_total_kb": 8191996, "swap_free_kb": 0,
            "gfp_mask": "0x201da", "order": 0
        }, 1),
        (9, "case9_malloc_global_oom.txt", {
            "trigger_process": "node", "killed_process": "malloc", "killed_pid": 2505,
            "total_vm_kb": 1116860, "anon_rss_kb": 1104768, "oom_score_adj": 0,
            "total_ram_pages": 524174, "node_free_kb": 7840, "node_min_kb": 356,
            "swap_total_kb": 0, "swap_free_kb": 0, "constraint": "CONSTRAINT_NONE",
            "gfp_mask": "0x140cca", "order": 0, "cgroup_path": None,
            "cgroup_usage_kb": None, "cgroup_limit_kb": None, "cgroup_failcnt": None,
            "kernel_version": "6.8.0-107-generic"
        }, 1),
        (10, "case10_malloc_cgroup_oom.txt", {
            "trigger_process": "runc:[2:INIT]", "killed_process": "node", "killed_pid": 7825,
            "total_vm_kb": 11764120, "anon_rss_kb": 39804, "oom_score_adj": 0,
            "total_ram_pages": None, "node_free_kb": None, "node_min_kb": None,
            "swap_total_kb": None, "swap_free_kb": None, "constraint": "CONSTRAINT_MEMCG",
            "gfp_mask": "0xcc0", "order": 0,
            "cgroup_path": "/system.slice/docker-126fd9c6e3bd822b75b1df9547687ab42e8e9ae1379f175087e20065c31bb816.scope",
            "cgroup_usage_kb": 131072, "cgroup_limit_kb": 131072, "cgroup_failcnt": 4573503,
            "kernel_version": "6.8.0-107-generic"
        }, 1),
    ]

    passed = 0
    for case_id, filename, expected_updates, min_process_table_len in test_cases:
        try:
            run_case(
                case_id=case_id,
                filename=filename,
                expected_updates=expected_updates,
                min_process_table_len=min_process_table_len,
            )
            passed += 1
        except AssertionError as exc:
            print(f"❌ Case {case_id} failed: {exc}")
        except FileNotFoundError:
            print(f"❌ Case {case_id} failed: node_1_logs/{filename} 파일을 찾을 수 없습니다.")

    print(f"\n🏁 최종 결과: {passed}/{len(test_cases)} 케이스 통과")