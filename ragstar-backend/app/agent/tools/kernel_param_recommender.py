from typing import Dict, Any, List

# [데이터 출처 및 참고 문헌]
# 1. Overcommit & Admin Reserve: https://www.kernel.org/doc/Documentation/vm/overcommit-accounting
# 2. VM Sysctl & Compaction: https://www.kernel.org/doc/Documentation/sysctl/vm.txt
# 3. Red Hat Swappiness Tuning: https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/6/html/performance_tuning_guide/s-memory-tunables
# 4. CVE-2018-1000200: https://nvd.nist.gov/vuln/detail/cve-2018-1000200
# 5. K8s kmem leak: https://github.com/kubernetes/kubernetes/issues/61937
# 6. Compaction loop bug: https://bugzilla.kernel.org/show_bug.cgi?id=207273
# 7. cgroup v2 Documentation: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html

PARAM_RECOMMENDATIONS = {
    "global_oom": [
        {
            "name": "vm.overcommit_memory",
            "description": "Memory overcommit policy setting",
            "current_implication": "Default value (0) allows the kernel to heuristically estimate available memory and permit allocations",
            "recommendation": "Setting to 2 (Don't overcommit) rejects allocations exceeding CommitLimit (Swap + RAM * ratio), allowing applications to handle errors before the OOM Killer is invoked",
            "command": "sysctl -w vm.overcommit_memory=2"
        },
        {
            "name": "vm.min_free_kbytes",
            "description": "Minimum free memory the system must maintain",
            "current_implication": "If set too low, memory can be exhausted before the kernel begins page reclaim",
            "recommendation": "Increase to 1-3% of physical RAM (e.g. ~160MB-500MB for 16GB RAM). For high-performance servers or JVM environments, 1GB or more may be recommended",
            "command": "sysctl -w vm.min_free_kbytes=1048576"
        },
        {
            "name": "vm.admin_reserve_kbytes",
            "description": "Reserved memory for administrator (root) access",
            "current_implication": "Prevents the system from becoming completely unresponsive during OOM, ensuring admin can SSH in or run diagnostic commands",
            "recommendation": "When using overcommit_memory=2, allocate at least 128MB (131072) on x86-64 for admin recovery operations",
            "command": "sysctl -w vm.admin_reserve_kbytes=131072"
        }
    ],
    "swap_exhaustion": [
        {
            "name": "vm.swappiness",
            "description": "Controls the tendency to swap anonymous pages vs. reclaiming file cache",
            "current_implication": "Default value (60) is typical, but when swap space is limited, frequent swapping causes severe system latency",
            "recommendation": "For database servers (PostgreSQL, MySQL, etc.), lower to 1-10 to suppress swapping and maximize physical memory utilization",
            "command": "sysctl -w vm.swappiness=10"
        }
    ],
    "cgroup_oom": [
        {
            "name": "memory.max",
            "description": "cgroup 내 프로세스가 사용할 수 있는 물리 메모리의 하드 리밋 (cgroup v2)",
            "current_implication": "이 제한에 도달하면 커널이 즉시 OOM Killer를 호출하여 프로세스를 강제 종료함",
            "recommendation": "워크로드의 피크 메모리 사용량을 모니터링한 후, 안전 마진을 포함하여 피크 사용량의 1.2~1.5배로 재설정 검토",
            "command": "echo <bytes> > /sys/fs/cgroup/<path>/memory.max"
        },
        {
            "name": "memory.high",
            "description": "OOM 발생 전 메모리 스로틀링 및 회수를 유도하는 기준점 (cgroup v2)",
            "current_implication": "이 값을 초과하면 커널이 강제로 메모리를 회수하려 시도하여 애플리케이션 응답 지연(latency)이 발생함",
            "recommendation": "OOM 강제 종료를 예방하기 위해 memory.max보다 낮게(예: max의 80~90% 수준) 설정하여 사전 제어",
            "command": "echo <bytes> > /sys/fs/cgroup/<path>/memory.high"
        },
        {
            "name": "memory.oom.group",
            "description": "OOM 발생 시 cgroup 내 전체 프로세스 일괄 종료 여부 (cgroup v2)",
            "current_implication": "기본값(0)에서는 OOM 발생 시 단일 프로세스만 종료되어 서비스가 불완전한 상태로 계속 동작할 수 있음",
            "recommendation": "1로 설정하여 OOM 발생 시 cgroup 내의 모든 프로세스를 한 번에 정리(Kill)하도록 구성",
            "command": "echo 1 > /sys/fs/cgroup/<path>/memory.oom.group"
        }
    ],
    "page_alloc_failure": [
        {
            "name": "vm.min_free_kbytes",
            "description": "Raise watermark to prevent kernel page allocation failures",
            "current_implication": "High-order (contiguous page) allocation requests fail due to memory fragmentation",
            "recommendation": "Increasing this value causes the kernel to start page reclaim and compaction earlier, reducing fragmentation-induced failures",
            "command": "sysctl -w vm.min_free_kbytes=262144"
        },
        {
            "name": "vm.compact_memory",
            "description": "Manual trigger for memory compaction (defragmentation)",
            "current_implication": "Sufficient total free memory exists but lacks contiguous free blocks for high-order allocations",
            "recommendation": "Writing 1 to this file triggers immediate kernel memory compaction to consolidate contiguous free pages",
            "command": "echo 1 > /proc/sys/vm/compact_memory"
        }
    ]
}

def kernel_param_recommender(oom_type: str, parsed_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    OOM 유형에 따라 관련 커널 파라미터 설정을 추천합니다.
    """
    # 유형에 맞는 추천 리스트를 가져오거나 없으면 빈 리스트 반환
    recommendations = PARAM_RECOMMENDATIONS.get(oom_type, [])
    
    return {
        "oom_type": oom_type,
        "recommendations": recommendations,
        "count": len(recommendations)
    }