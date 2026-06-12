import os
import sys
import json
from types import SimpleNamespace

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.nodes.node_2_classifier import (
    DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    inspect_embedding_oom_type,
    node_2_classifier,
    get_available_tools_desc,
    deterministic_oom_classification,
    embedding_normalize_oom_type,
)


class FakeLLM:
    """ChatOllama 대체용 가짜 LLM"""

    def __init__(self, response_text: str):
        self.response_text = response_text


class FakeEmbeddings:
    def _embed(self, text: str):
        normalized = text.lower()
        return [
            float(any(token in normalized for token in ["global", "host", "system"])),
            float(any(token in normalized for token in ["cgroup", "container", "memcg"])),
            float("swap" in normalized),
            float(any(token in normalized for token in ["page", "alloc", "order"])),
        ]

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]


class AmbiguousEmbeddings:
    def embed_query(self, text: str):
        if text == "ambiguous host/container oom":
            return [1.0, 0.0]
        return [0.0, 0.0]

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "global oom host-wide" in normalized:
                vectors.append([1.0, 0.0])
            elif "system-wide oom caused" in normalized:
                vectors.append([0.94, 0.0])
            elif "cgroup oom container" in normalized:
                vectors.append([0.95, 0.0])
            elif "memory cgroup limit hit" in normalized:
                vectors.append([0.93, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class FakePromptTemplate:
    """ChatPromptTemplate.from_template(...) 대체용"""

    def __init__(self, template_text: str):
        self.template_text = template_text

    def __or__(self, llm):
        return FakeChain(llm)


class FakeChain:
    """
    `prompt | llm` 결과를 흉내내는 체인 객체.
    마지막 payload를 저장해 LLM 입력 검증에 활용한다.
    """

    last_payload = None

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, payload: dict):
        FakeChain.last_payload = payload
        return SimpleNamespace(content=self.llm.response_text)


def build_mock_state(
    parsed_fields: dict | None = None,
    raw_log: str = "Task in /agent killed as a result of limit of /agent",
) -> dict:
    """현재 state 스키마에 맞는 기본 mock state 생성"""
    return {
        "raw_log": raw_log,
        "user_kernel_version": "5.15.0-76-generic",
        "parsed_fields": parsed_fields or {},
        "classification": {
            "oom_type": None,
            "tools_needed": [],
            "needs_kb": False,
            "confidence": "low",
        },
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }


def patch_node_2_dependencies(monkeypatch, response_text: str):
    """
    node_2_classifier 내부 외부 의존성을 가짜로 교체
    - 프롬프트 파일 로딩 제거
    - 실제 Ollama 호출 제거
    - LangChain 체인 구성 제거
    """
    import app.agent.nodes.node_2_classifier as module

    FakeChain.last_payload = None

    monkeypatch.setattr(
        module,
        "load_prompt",
        lambda filename: "Dummy prompt: {available_tools_desc}\n{parsed_fields}",
    )

    monkeypatch.setattr(
        module,
        "build_node2_classifier_llm",
        lambda: FakeLLM(response_text),
    )

    monkeypatch.setattr(
        module,
        "build_node2_classifier_embeddings",
        lambda: FakeEmbeddings(),
    )

    monkeypatch.setattr(
        module.ChatPromptTemplate,
        "from_template",
        lambda template_text: FakePromptTemplate(template_text),
    )


def test_get_available_tools_desc_all_candidates():
    """
    process_table이 2개 이상이고 kernel_version이 있으면
    세 도구가 정확히 후보에 포함되어야 한다.
    """
    parsed_fields = {
        "process_table": [
            {"pid": 101, "name": "python", "rss_kb": 100000},
            {"pid": 102, "name": "java", "rss_kb": 200000},
        ],
        "kernel_version": "5.15.0-76-generic",
    }

    tools_desc = get_available_tools_desc(parsed_fields)
    lines = tools_desc.splitlines()

    assert lines == [
        "memory_calculator",
        "kernel_version_check",
        "kernel_param_recommender",
    ]


def test_get_available_tools_desc_minimal_case():
    """
    process_table이 부족하고 kernel_version도 없으면
    kernel_param_recommender만 남아야 한다.
    """
    parsed_fields = {
        "process_table": [],
        "kernel_version": None,
    }

    tools_desc = get_available_tools_desc(parsed_fields)
    lines = tools_desc.splitlines()

    assert lines == ["kernel_param_recommender"]


def test_node_2_classifier_success_cgroup_oom(monkeypatch):
    """
    cgroup OOM 정상 분류 테스트
    """
    fake_llm_json = json.dumps(
        {
            "oom_type": "cgroup_oom",
            "tools_needed": [
                "memory_calculator",
                "kernel_version_check",
                "kernel_param_recommender",
            ],
            "needs_kb": True,
            "confidence": "high",
        },
        ensure_ascii=False,
    )

    patch_node_2_dependencies(monkeypatch, fake_llm_json)

    parsed_fields = {
        "trigger_process": "python",
        "killed_process": "s1-agent",
        "killed_pid": 13331,
        "total_vm_kb": 2617284,
        "anon_rss_kb": 1024000,
        "oom_score_adj": 0,
        "total_ram_pages": 524288,
        "cgroup_path": "/agent",
        "cgroup_usage_kb": 1048576,
        "cgroup_limit_kb": 1048576,
        "constraint": "CONSTRAINT_MEMCG",
        "kernel_version": "5.15.0-76-generic",
        "process_table": [
            {"pid": 13331, "name": "s1-agent", "rss_kb": 1024000, "oom_score_adj": 0},
            {"pid": 13450, "name": "s1-helper", "rss_kb": 20800, "oom_score_adj": 0},
        ],
    }

    state = build_mock_state(parsed_fields=parsed_fields)
    updated_state = node_2_classifier(state)
    result = updated_state["classification"]

    assert updated_state["raw_log"] == state["raw_log"]
    assert updated_state["user_kernel_version"] == state["user_kernel_version"]
    assert updated_state["parsed_fields"] == state["parsed_fields"]
    assert updated_state["tool_results"] == state["tool_results"]
    assert updated_state["diagnosis"] == state["diagnosis"]

    assert result["oom_type"] == "cgroup_oom"
    assert result["tools_needed"] == [
        "memory_calculator",
        "kernel_version_check",
        "kernel_param_recommender",
    ]
    assert result["needs_kb"] is True
    assert result["confidence"] == "high"
    assert updated_state["error"] is None

    assert FakeChain.last_payload is not None
    assert "available_tools_desc" in FakeChain.last_payload
    assert "parsed_fields" in FakeChain.last_payload

    tools_desc = FakeChain.last_payload["available_tools_desc"]
    assert "memory_calculator" in tools_desc
    assert "kernel_version_check" in tools_desc
    assert "kernel_param_recommender" in tools_desc

    parsed_fields_text = FakeChain.last_payload["parsed_fields"]
    assert '"trigger_process": "python"' in parsed_fields_text
    assert '"killed_process": "s1-agent"' in parsed_fields_text
    assert '"killed_pid": 13331' in parsed_fields_text
    assert '"anon_rss_kb": 1024000' in parsed_fields_text
    assert '"constraint": "CONSTRAINT_MEMCG"' in parsed_fields_text
    assert '"cgroup_path": "/agent"' in parsed_fields_text
    assert '"invoked_by"' not in parsed_fields_text
    assert '"killed_rss_kb"' not in parsed_fields_text


def test_node_2_classifier_success_global_oom(monkeypatch):
    """
    global OOM 정상 분류 테스트
    """
    fake_llm_json = json.dumps(
        {
            "oom_type": "global_oom",
            "tools_needed": ["memory_calculator", "kernel_param_recommender"],
            "needs_kb": False,
            "confidence": "medium",
        },
        ensure_ascii=False,
    )

    patch_node_2_dependencies(monkeypatch, fake_llm_json)

    parsed_fields = {
        "trigger_process": "httpd",
        "killed_process": "java",
        "killed_pid": 3201,
        "total_vm_kb": 273728,
        "anon_rss_kb": 875680,
        "oom_score_adj": 0,
        "total_ram_pages": 524288,
        "swap_total_kb": 0,
        "swap_free_kb": 0,
        "constraint": "CONSTRAINT_NONE",
        "kernel_version": None,
        "process_table": [
            {"pid": 3201, "name": "java", "rss_kb": 875680, "oom_score_adj": 0},
            {"pid": 3244, "name": "httpd", "rss_kb": 209360, "oom_score_adj": 0},
        ],
    }

    state = build_mock_state(parsed_fields=parsed_fields)
    updated_state = node_2_classifier(state)
    result = updated_state["classification"]

    assert updated_state["raw_log"] == state["raw_log"]
    assert updated_state["user_kernel_version"] == state["user_kernel_version"]
    assert updated_state["parsed_fields"] == state["parsed_fields"]
    assert updated_state["tool_results"] == state["tool_results"]
    assert updated_state["diagnosis"] == state["diagnosis"]

    assert result["oom_type"] == "global_oom"
    assert result["tools_needed"] == ["memory_calculator", "kernel_param_recommender"]
    assert result["needs_kb"] is False
    assert result["confidence"] == "high"
    assert updated_state["error"] is None

    tools_desc = FakeChain.last_payload["available_tools_desc"]
    assert "memory_calculator" in tools_desc
    assert "kernel_param_recommender" in tools_desc
    assert "kernel_version_check" not in tools_desc

    parsed_fields_text = FakeChain.last_payload["parsed_fields"]
    assert '"trigger_process": "httpd"' in parsed_fields_text
    assert '"killed_process": "java"' in parsed_fields_text
    assert '"killed_pid": 3201' in parsed_fields_text
    assert '"anon_rss_kb": 875680' in parsed_fields_text
    assert '"constraint": "CONSTRAINT_NONE"' in parsed_fields_text
    assert '"invoked_by"' not in parsed_fields_text
    assert '"killed_rss_kb"' not in parsed_fields_text


def test_node_2_classifier_invalid_json_parse_error_path(monkeypatch):
    """
    LLM이 깨진 JSON을 반환하면
    Node 2의 parse-error 방어 경로가 적용되어야 한다.
    """
    patch_node_2_dependencies(monkeypatch, "this is not valid json")

    parsed_fields = {
        "trigger_process": "python",
        "constraint": "CONSTRAINT_MEMCG",
        "kernel_version": "5.15.0-76-generic",
        "process_table": [
            {"pid": 1, "name": "a", "rss_kb": 1000, "oom_score_adj": 0},
            {"pid": 2, "name": "b", "rss_kb": 2000, "oom_score_adj": 0},
        ],
    }

    state = build_mock_state(parsed_fields=parsed_fields)
    updated_state = node_2_classifier(state)
    result = updated_state["classification"]

    assert updated_state["raw_log"] == state["raw_log"]
    assert updated_state["user_kernel_version"] == state["user_kernel_version"]
    assert updated_state["parsed_fields"] == state["parsed_fields"]
    assert updated_state["tool_results"] == state["tool_results"]
    assert updated_state["diagnosis"] == state["diagnosis"]

    assert result["oom_type"] == "cgroup_oom"
    assert result["tools_needed"] == ["kernel_param_recommender"]
    assert result["needs_kb"] is True
    assert result["confidence"] == "low"
    assert updated_state["error"] is not None
    assert "Node 2 Error:" in updated_state["error"]


def test_deterministic_oom_classification_prefers_global_over_swap_exhaustion():
    parsed_fields = {
        "constraint": "CONSTRAINT_NONE",
        "swap_total_kb": 2097152,
        "swap_free_kb": 0,
    }

    assert deterministic_oom_classification(parsed_fields) == "global_oom"


def test_deterministic_oom_classification_detects_memcg_swap_exhaustion():
    parsed_fields = {
        "constraint": "CONSTRAINT_MEMCG",
        "cgroup_path": "/docker/test",
        "cgroup_swap_usage_kb": 1048576,
        "cgroup_swap_limit_kb": 1048576,
    }

    assert deterministic_oom_classification(parsed_fields) == "swap_exhaustion"


def test_deterministic_oom_classification_uses_cgroup_path_when_constraint_missing():
    parsed_fields = {
        "constraint": None,
        "cgroup_path": "/agent",
        "cgroup_usage_kb": 1048576,
        "cgroup_limit_kb": 1048576,
    }

    assert deterministic_oom_classification(parsed_fields) == "cgroup_oom"


def test_embedding_normalize_oom_type_maps_semantic_variant():
    normalized = embedding_normalize_oom_type("host out of memory", FakeEmbeddings())
    assert normalized == "global_oom"


def test_embedding_normalize_oom_type_maps_page_alloc_label_with_underscores():
    normalized = embedding_normalize_oom_type("page_alloc_failure", FakeEmbeddings())
    assert normalized == "page_alloc_failure"


def test_inspect_embedding_oom_type_returns_scores_for_embedding_match():
    diagnostics = inspect_embedding_oom_type(
        "host out of memory",
        FakeEmbeddings(),
        similarity_threshold=DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
        similarity_margin=DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
    )

    assert diagnostics["accepted_label"] == "global_oom"
    assert diagnostics["best_label"] == "global_oom"
    assert diagnostics["passes_threshold"] is True
    assert diagnostics["passes_margin"] is True
    assert diagnostics["normalization_source"] in {"canonical_alias", "embedding_match"}


def test_embedding_normalize_oom_type_rejects_ambiguous_near_tie():
    normalized = embedding_normalize_oom_type(
        "ambiguous host/container oom",
        AmbiguousEmbeddings(),
        similarity_threshold=0.62,
        similarity_margin=0.08,
    )

    assert normalized is None


def test_inspect_embedding_oom_type_marks_below_margin():
    diagnostics = inspect_embedding_oom_type(
        "ambiguous host/container oom",
        AmbiguousEmbeddings(),
        similarity_threshold=0.62,
        similarity_margin=0.08,
    )

    assert diagnostics["accepted_label"] is None
    assert diagnostics["passes_threshold"] is True
    assert diagnostics["passes_margin"] is False
    assert diagnostics["normalization_source"] == "below_margin"


def test_node_2_classifier_uses_embedding_for_non_canonical_llm_label(monkeypatch):
    fake_llm_json = json.dumps(
        {
            "oom_type": "host out of memory",
            "tools_needed": ["memory_calculator", "kernel_param_recommender"],
            "needs_kb": True,
            "confidence": "medium",
        },
        ensure_ascii=False,
    )

    patch_node_2_dependencies(monkeypatch, fake_llm_json)

    parsed_fields = {
        "trigger_process": "telegraf",
        "killed_process": "java",
        "constraint": None,
        "cgroup_path": None,
        "swap_total_kb": None,
        "swap_free_kb": None,
        "order": 0,
        "process_table": [
            {"pid": 1, "name": "java", "rss_kb": 1000, "oom_score_adj": 0},
            {"pid": 2, "name": "telegraf", "rss_kb": 500, "oom_score_adj": 0},
        ],
    }

    updated_state = node_2_classifier(build_mock_state(parsed_fields=parsed_fields))

    assert updated_state["classification"]["oom_type"] == "global_oom"
    assert updated_state["classification"]["raw_llm_oom_type"] == "host out of memory"
    assert updated_state["classification"]["deterministic_oom_type"] == "unknown"
    assert updated_state["classification"]["normalized_llm_oom_type"] == "global_oom"
    assert updated_state["classification"]["tools_needed"] == [
        "memory_calculator",
        "kernel_param_recommender",
    ]
    assert updated_state["classification"]["needs_kb"] is True
