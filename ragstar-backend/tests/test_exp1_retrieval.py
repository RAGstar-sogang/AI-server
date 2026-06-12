import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


"""`exp1_retrieval` 테스트는 전체 4단계 파이프라인이 아니라, 실험용 retrieval harness의 계약을 검증한다."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def load_exp1_module(
    monkeypatch,
    *,
    parser_impl=None,
    search_kb_impl=None,
    get_collection_impl=None,
):
    parser_impl = parser_impl or (lambda state: {"parsed_fields": {}})
    search_kb_impl = search_kb_impl or (lambda *args, **kwargs: None)
    get_collection_impl = get_collection_impl or (lambda: None)

    app_module = types.ModuleType("app")
    app_module.__path__ = []
    agent_module = types.ModuleType("app.agent")
    agent_module.__path__ = []
    tools_module = types.ModuleType("app.agent.tools")
    tools_module.__path__ = []
    search_kb_module = types.ModuleType("app.agent.tools.search_kb")
    search_kb_module.search_kb = search_kb_impl
    database_module = types.ModuleType("app.database")
    database_module.__path__ = []
    chromadb_client_module = types.ModuleType("app.database.chromadb_client")
    chromadb_client_module.get_collection = get_collection_impl
    nodes_module = types.ModuleType("app.agent.nodes")
    nodes_module.__path__ = []
    node_1_parser_module = types.ModuleType("app.agent.nodes.node_1_parser")
    node_1_parser_module.node_1_parser = parser_impl

    monkeypatch.setitem(sys.modules, "app", app_module)
    monkeypatch.setitem(sys.modules, "app.agent", agent_module)
    monkeypatch.setitem(sys.modules, "app.agent.tools", tools_module)
    monkeypatch.setitem(sys.modules, "app.agent.tools.search_kb", search_kb_module)
    monkeypatch.setitem(sys.modules, "app.database", database_module)
    monkeypatch.setitem(sys.modules, "app.database.chromadb_client", chromadb_client_module)
    monkeypatch.setitem(sys.modules, "app.agent.nodes", nodes_module)
    monkeypatch.setitem(sys.modules, "app.agent.nodes.node_1_parser", node_1_parser_module)
    monkeypatch.delitem(sys.modules, "experiments.exp1_retrieval", raising=False)

    return importlib.import_module("experiments.exp1_retrieval")


def load_search_kb_module(monkeypatch, *, get_collection_impl=None):
    get_collection_impl = get_collection_impl or (lambda: None)

    app_module = types.ModuleType("app")
    app_module.__path__ = []
    database_module = types.ModuleType("app.database")
    database_module.__path__ = []
    chromadb_client_module = types.ModuleType("app.database.chromadb_client")
    chromadb_client_module.get_collection = get_collection_impl

    monkeypatch.setitem(sys.modules, "app", app_module)
    monkeypatch.setitem(sys.modules, "app.database", database_module)
    monkeypatch.setitem(sys.modules, "app.database.chromadb_client", chromadb_client_module)
    monkeypatch.delitem(sys.modules, "test_search_kb_runtime", raising=False)

    module_path = PROJECT_ROOT / "app" / "agent" / "tools" / "search_kb.py"
    spec = importlib.util.spec_from_file_location("test_search_kb_runtime", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyCollection:
    def __init__(self, result=None, should_raise=False):
        self.result = result or {
            "documents": [[]],
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }
        self.should_raise = should_raise
        self.last_kwargs = None

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        if self.should_raise:
            raise RuntimeError("query failed")
        return self.result


class RoutingCollection:
    def __init__(self, result_by_query=None, should_raise=False):
        self.result_by_query = result_by_query or {}
        self.should_raise = should_raise
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.should_raise:
            raise RuntimeError("query failed")

        query = kwargs["query_texts"][0]
        return self.result_by_query.get(
            query,
            {
                "documents": [[]],
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
            },
        )


def test_calculate_recall_returns_overlap_fraction(monkeypatch):
    module = load_exp1_module(monkeypatch)

    score = module.calculate_recall(
        retrieved_ids=["chunk_1", "chunk_2", "chunk_9"],
        ground_truth_ids=["chunk_2", "chunk_3"],
    )

    assert score == pytest.approx(0.5)


def test_calculate_recall_returns_zero_when_ground_truth_is_empty(monkeypatch):
    module = load_exp1_module(monkeypatch)

    score = module.calculate_recall(
        retrieved_ids=["chunk_1"],
        ground_truth_ids=[],
    )

    assert score == 0.0


def test_build_parser_input_state_returns_minimum_contract(monkeypatch):
    module = load_exp1_module(monkeypatch)

    state = module.build_parser_input_state("sample raw log")

    assert state == {
        "raw_log": "sample raw log",
        "user_kernel_version": None,
        "parsed_fields": {},
        "classification": {},
        "tool_results": {},
        "diagnosis": {},
        "error": None,
    }


def test_load_oom_logs_by_id_reads_jsonl_and_indexes_by_log_id(monkeypatch, tmp_path):
    module = load_exp1_module(monkeypatch)
    jsonl_path = tmp_path / "oom_logs.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"log_id": "log_1", "raw_log": "alpha"}),
                json.dumps({"log_id": "log_2", "raw_log": "beta"}),
            ]
        ),
        encoding="utf-8",
    )

    rows = module.load_oom_logs_by_id(jsonl_path)

    assert rows == {
        "log_1": {"log_id": "log_1", "raw_log": "alpha"},
        "log_2": {"log_id": "log_2", "raw_log": "beta"},
    }


@pytest.mark.xfail(
    reason="`load_oom_logs_by_id()` silently overwrites duplicate log_id rows instead of rejecting corrupted input",
    strict=True,
)
def test_load_oom_logs_by_id_should_reject_duplicate_log_id(monkeypatch, tmp_path):
    module = load_exp1_module(monkeypatch)
    jsonl_path = tmp_path / "oom_logs.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"log_id": "log_1", "raw_log": "alpha"}),
                json.dumps({"log_id": "log_1", "raw_log": "beta"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate log_id"):
        module.load_oom_logs_by_id(jsonl_path)


def test_get_raw_log_for_log_id_returns_matching_log_or_empty_string(monkeypatch):
    module = load_exp1_module(monkeypatch)
    oom_logs_by_id = {
        "log_1": {"raw_log": "kernel log"},
    }

    assert module.get_raw_log_for_log_id("log_1", oom_logs_by_id) == "kernel log"
    assert module.get_raw_log_for_log_id("missing", oom_logs_by_id) == ""


def test_build_parsed_retrieval_inputs_uses_parser_output_and_expected_label(monkeypatch):
    captured_states = []

    def parser_impl(state):
        captured_states.append(state)
        return {
            "parsed_fields": {
                "constraint": "CONSTRAINT_MEMCG",
                "swap_total_kb": 0,
            }
        }

    module = load_exp1_module(monkeypatch, parser_impl=parser_impl)

    oom_type, parsed_fields = module.build_parsed_retrieval_inputs(
        {"expected_oom_type": "cgroup_oom"},
        "raw log body",
    )

    assert oom_type == "cgroup_oom"
    assert parsed_fields == {
        "constraint": "CONSTRAINT_MEMCG",
        "swap_total_kb": 0,
    }
    assert captured_states == [module.build_parser_input_state("raw log body")]


def test_build_search_kb_query_includes_all_supported_signals(monkeypatch):
    module = load_exp1_module(monkeypatch)

    query = module.build_search_kb_query(
        "cgroup_oom",
        {
            "constraint": "CONSTRAINT_MEMCG",
            "cgroup_path": "/docker/container",
            "swap_total_kb": 0,
            "order": 2,
        },
    )

    assert query == (
        "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space order 2 page allocation"
    )


def test_build_search_kb_query_ignores_malformed_string_order(monkeypatch):
    module = load_exp1_module(monkeypatch)

    query = module.build_search_kb_query(
        "global_oom",
        {
            "constraint": "CONSTRAINT_NONE",
            "swap_total_kb": "0",
            "order": "abc",
        },
    )

    assert query == "global_oom CONSTRAINT_NONE no swap space"


@pytest.mark.parametrize(
    ("oom_type", "parsed_fields"),
    [
        pytest.param(
            "global_oom",
            {"constraint": "CONSTRAINT_NONE", "swap_total_kb": 0, "order": 0},
            id="global_oom_no_swap",
        ),
        pytest.param(
            "cgroup_oom",
            {
                "constraint": "CONSTRAINT_MEMCG",
                "cgroup_path": "/docker/container",
                "swap_total_kb": 0,
                "order": 3,
            },
            id="cgroup_with_order",
        ),
        pytest.param(
            "page_alloc_failure",
            {"constraint": None, "swap_total_kb": 1024, "order": "abc"},
            id="malformed_order_ignored",
        ),
    ],
)
def test_build_search_kb_query_matches_production_search_kb_rules(
    monkeypatch,
    oom_type,
    parsed_fields,
):
    exp1_module = load_exp1_module(monkeypatch)
    search_kb_module = load_search_kb_module(monkeypatch)

    exp1_query = exp1_module.build_search_kb_query(oom_type, parsed_fields)
    prod_result = search_kb_module.search_kb(
        oom_type=oom_type,
        parsed_fields=parsed_fields,
        collection=DummyCollection(should_raise=True),
    )

    assert exp1_query == prod_result["query_used"]


def test_run_parsed_retrieval_query_matches_production_search_kb_contract(monkeypatch):
    exp1_module = load_exp1_module(monkeypatch)
    search_kb_module = load_search_kb_module(monkeypatch)
    collection = DummyCollection(
        result={
            "documents": [["doc one", "doc two"]],
            "ids": [["chunk_1", "chunk_2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[
                {"error_category": "cgroup_oom"},
                {"error_category": "general"},
            ]],
        }
    )
    parsed_fields = {
        "constraint": "CONSTRAINT_MEMCG",
        "cgroup_path": "/docker/container",
        "swap_total_kb": 0,
        "order": 1,
    }

    exp1_result = exp1_module.run_parsed_retrieval_query(
        oom_type="cgroup_oom",
        parsed_fields=parsed_fields,
        collection=collection,
        top_k=5,
    )
    prod_result = search_kb_module.search_kb(
        oom_type="cgroup_oom",
        parsed_fields=parsed_fields,
        collection=collection,
    )

    assert exp1_result == prod_result


def test_run_parsed_retrieval_query_matches_production_error_shape(monkeypatch):
    exp1_module = load_exp1_module(monkeypatch)
    search_kb_module = load_search_kb_module(monkeypatch)
    collection = DummyCollection(should_raise=True)
    parsed_fields = {
        "constraint": "CONSTRAINT_NONE",
        "swap_total_kb": 0,
        "order": 0,
    }

    exp1_result = exp1_module.run_parsed_retrieval_query(
        oom_type="global_oom",
        parsed_fields=parsed_fields,
        collection=collection,
        top_k=5,
    )
    prod_result = search_kb_module.search_kb(
        oom_type="global_oom",
        parsed_fields=parsed_fields,
        collection=collection,
    )

    assert exp1_result == prod_result


def test_run_parsed_retrieval_query_maps_collection_response(monkeypatch):
    module = load_exp1_module(monkeypatch)
    collection = DummyCollection(
        result={
            "documents": [["doc one", "doc two"]],
            "ids": [["chunk_1", "chunk_2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[
                {"error_category": "cgroup_oom"},
                {"error_category": "general"},
            ]],
        }
    )

    result = module.run_parsed_retrieval_query(
        oom_type="cgroup_oom",
        parsed_fields={
            "constraint": "CONSTRAINT_MEMCG",
            "cgroup_path": "/docker/container",
            "swap_total_kb": 0,
            "order": 1,
        },
        collection=collection,
        top_k=3,
    )

    assert collection.last_kwargs == {
        "query_texts": [
            "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space order 1 page allocation"
        ],
        "n_results": 3,
        "where": {"error_category": {"$in": ["cgroup_oom", "general"]}},
    }
    assert result == {
        "query_used": (
            "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space order 1 page allocation"
        ),
        "chunks": [
            {
                "chunk_id": "chunk_1",
                "content": "doc one",
                "score": 0.1,
                "metadata": {"error_category": "cgroup_oom"},
            },
            {
                "chunk_id": "chunk_2",
                "content": "doc two",
                "score": 0.2,
                "metadata": {"error_category": "general"},
            },
        ],
        "total_found": 2,
    }


def test_run_parsed_retrieval_query_returns_structured_error_on_failure(monkeypatch):
    module = load_exp1_module(monkeypatch)
    collection = DummyCollection(should_raise=True)

    result = module.run_parsed_retrieval_query(
        oom_type="global_oom",
        parsed_fields={
            "constraint": "CONSTRAINT_NONE",
            "swap_total_kb": 0,
            "order": 0,
        },
        collection=collection,
        top_k=5,
    )

    assert result["query_used"] == "global_oom CONSTRAINT_NONE no swap space"
    assert result["chunks"] == []
    assert result["total_found"] == 0
    assert result["error"] == "KB 검색 오류: query failed"


def test_build_parsed_retrieval_inputs_defaults_to_unknown_and_empty_fields(monkeypatch):
    module = load_exp1_module(monkeypatch, parser_impl=lambda state: {})

    oom_type, parsed_fields = module.build_parsed_retrieval_inputs({}, "raw log body")

    assert oom_type == "unknown"
    assert parsed_fields == {}


def test_main_rejects_non_positive_top_k(monkeypatch, capsys):
    module = load_exp1_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "0"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "--top-k는 1 이상의 정수" in captured.out


def test_main_raw_mode_joins_logs_and_writes_expected_csv(monkeypatch, tmp_path):
    collection = RoutingCollection(
        result_by_query={
            "raw oom log A": {
                "documents": [["doc one", "doc two"]],
                "ids": [["chunk_hit", "chunk_extra"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [[
                    {"error_category": "global_oom"},
                    {"error_category": "general"},
                ]],
            }
        }
    )
    get_collection_calls = []

    def get_collection_impl():
        get_collection_calls.append("called")
        return collection

    module = load_exp1_module(monkeypatch, get_collection_impl=get_collection_impl)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_1",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_hit", "chunk_other"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_1", "raw_log": "raw oom log A"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "2"],
    )

    module.main()

    assert get_collection_calls == ["called"]
    assert collection.calls == [{"query_texts": ["raw oom log A"], "n_results": 2}]

    output_path = exp_results_dir / "exp1_retrieval_raw_top2_results.csv"
    df = module.pd.read_csv(output_path)
    assert list(df.columns) == [
        "log_id",
        "query_mode",
        "top_k",
        "expected_oom_type",
        "expected_chunks",
        "retrieved_chunks",
        "recall",
    ]
    assert df.loc[0, "log_id"] == "log_1"
    assert df.loc[0, "query_mode"] == "raw"
    assert df.loc[0, "top_k"] == 2
    assert df.loc[0, "expected_oom_type"] == "global_oom"
    assert df.loc[0, "expected_chunks"] == "['chunk_hit', 'chunk_other']"
    assert df.loc[0, "retrieved_chunks"] == "['chunk_hit', 'chunk_extra']"
    assert df.loc[0, "recall"] == pytest.approx(0.5)


def test_main_raw_mode_writes_csv_with_stable_field_types(monkeypatch, tmp_path):
    collection = RoutingCollection(
        result_by_query={
            "raw oom log type check": {
                "documents": [["doc one"]],
                "ids": [["chunk_type"]],
                "distances": [[0.25]],
                "metadatas": [[{"error_category": "global_oom"}]],
            }
        }
    )

    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: collection)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_type_check",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_type"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_type_check", "raw_log": "raw oom log type check"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "7"],
    )

    module.main()

    df = module.pd.read_csv(exp_results_dir / "exp1_retrieval_raw_top7_results.csv")
    assert module.pd.api.types.is_object_dtype(df["log_id"])
    assert module.pd.api.types.is_object_dtype(df["query_mode"])
    assert module.pd.api.types.is_integer_dtype(df["top_k"])
    assert module.pd.api.types.is_object_dtype(df["expected_oom_type"])
    assert module.pd.api.types.is_object_dtype(df["expected_chunks"])
    assert module.pd.api.types.is_object_dtype(df["retrieved_chunks"])
    assert module.pd.api.types.is_float_dtype(df["recall"])


def test_main_parsed_mode_forwards_very_large_top_k(monkeypatch, tmp_path):
    parser_calls = []

    def parser_impl(state):
        parser_calls.append(state)
        return {
            "parsed_fields": {
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": 0,
            }
        }

    query = "global_oom CONSTRAINT_NONE no swap space"
    collection = RoutingCollection(
        result_by_query={
            query: {
                "documents": [["doc one"]],
                "ids": [["chunk_large_topk"]],
                "distances": [[0.01]],
                "metadatas": [[{"error_category": "global_oom"}]],
            }
        }
    )

    module = load_exp1_module(
        monkeypatch,
        parser_impl=parser_impl,
        get_collection_impl=lambda: collection,
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_large_topk",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_large_topk"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_large_topk", "raw_log": "raw oom log M"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "parsed", "--top-k", "999"],
    )

    module.main()

    assert parser_calls == [module.build_parser_input_state("raw oom log M")]
    assert collection.calls == [
        {
            "query_texts": [query],
            "n_results": 999,
            "where": {"error_category": {"$in": ["global_oom", "general"]}},
        }
    ]

    df = module.pd.read_csv(exp_results_dir / "exp1_retrieval_parsed_top999_results.csv")
    assert df.loc[0, "top_k"] == 999
    assert df.loc[0, "recall"] == pytest.approx(1.0)


def test_main_exits_when_ground_truth_file_is_missing(monkeypatch, tmp_path, capsys):
    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: RoutingCollection())
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"])

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "qa_ground_truth.jsonl 파일을 찾을 수 없습니다" in captured.out


def test_main_exits_when_oom_logs_file_is_missing(monkeypatch, tmp_path, capsys):
    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: RoutingCollection())
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"])

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps({"log_id": "log_1", "relevant_chunk_ids": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "oom_logs.jsonl 파일을 찾을 수 없습니다" in captured.out


@pytest.mark.xfail(
    reason="`load_oom_logs_by_id()` does not currently validate malformed `oom_logs.jsonl` rows before indexing by `log_id`",
    strict=True,
)
def test_load_oom_logs_by_id_should_reject_row_missing_log_id(monkeypatch, tmp_path):
    module = load_exp1_module(monkeypatch)
    jsonl_path = tmp_path / "oom_logs.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"raw_log": "alpha"}),
                json.dumps({"log_id": "log_2", "raw_log": "beta"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="log_id.*required|invalid oom_logs row"):
        module.load_oom_logs_by_id(jsonl_path)


@pytest.mark.xfail(
    reason="`load_oom_logs_by_id()` does not currently validate that `raw_log` is a string before returning joined rows",
    strict=True,
)
def test_load_oom_logs_by_id_should_reject_non_string_raw_log(monkeypatch, tmp_path):
    module = load_exp1_module(monkeypatch)
    jsonl_path = tmp_path / "oom_logs.jsonl"
    jsonl_path.write_text(
        json.dumps({"log_id": "log_1", "raw_log": {"message": "not a string"}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_log.*string|invalid oom_logs row"):
        module.load_oom_logs_by_id(jsonl_path)


def test_main_parsed_mode_uses_parser_expected_label_and_writes_csv(monkeypatch, tmp_path):
    parser_calls = []

    def parser_impl(state):
        parser_calls.append(state)
        return {
            "parsed_fields": {
                "constraint": "CONSTRAINT_MEMCG",
                "cgroup_path": "/docker/container",
                "swap_total_kb": 0,
                "order": 2,
            }
        }

    query = "cgroup_oom CONSTRAINT_MEMCG cgroup memory limit no swap space order 2 page allocation"
    collection = RoutingCollection(
        result_by_query={
            query: {
                "documents": [["doc"]],
                "ids": [["chunk_match"]],
                "distances": [[0.05]],
                "metadatas": [[{"error_category": "cgroup_oom"}]],
            }
        }
    )
    get_collection_calls = []

    def get_collection_impl():
        get_collection_calls.append("called")
        return collection

    module = load_exp1_module(
        monkeypatch,
        parser_impl=parser_impl,
        get_collection_impl=get_collection_impl,
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_2",
                "expected_oom_type": "cgroup_oom",
                "relevant_chunk_ids": ["chunk_match"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_2", "raw_log": "raw oom log B"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "parsed", "--top-k", "3"],
    )

    module.main()

    assert get_collection_calls == ["called"]
    assert parser_calls == [module.build_parser_input_state("raw oom log B")]
    assert collection.calls == [
        {
            "query_texts": [query],
            "n_results": 3,
            "where": {"error_category": {"$in": ["cgroup_oom", "general"]}},
        }
    ]

    output_path = exp_results_dir / "exp1_retrieval_parsed_top3_results.csv"
    df = module.pd.read_csv(output_path)
    assert df.loc[0, "log_id"] == "log_2"
    assert df.loc[0, "query_mode"] == "parsed"
    assert df.loc[0, "top_k"] == 3
    assert df.loc[0, "expected_oom_type"] == "cgroup_oom"
    assert df.loc[0, "retrieved_chunks"] == "['chunk_match']"
    assert df.loc[0, "recall"] == pytest.approx(1.0)


def test_main_parsed_mode_preserves_expected_csv_column_order(monkeypatch, tmp_path):
    def parser_impl(state):
        return {
            "parsed_fields": {
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": 0,
            }
        }

    query = "global_oom CONSTRAINT_NONE no swap space"
    collection = RoutingCollection(
        result_by_query={
            query: {
                "documents": [["doc ordered"]],
                "ids": [["chunk_order"]],
                "distances": [[0.11]],
                "metadatas": [[{"error_category": "global_oom"}]],
            }
        }
    )
    module = load_exp1_module(
        monkeypatch,
        parser_impl=parser_impl,
        get_collection_impl=lambda: collection,
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_column_order",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_order"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_column_order", "raw_log": "raw oom log N"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "parsed", "--top-k", "2"],
    )

    module.main()

    df = module.pd.read_csv(exp_results_dir / "exp1_retrieval_parsed_top2_results.csv")
    assert list(df.columns) == [
        "log_id",
        "query_mode",
        "top_k",
        "expected_oom_type",
        "expected_chunks",
        "retrieved_chunks",
        "recall",
    ]


def test_main_parsed_mode_top_k_one_limits_query_and_filename(monkeypatch, tmp_path):
    parser_calls = []

    def parser_impl(state):
        parser_calls.append(state)
        return {
            "parsed_fields": {
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": 0,
            }
        }

    query = "global_oom CONSTRAINT_NONE no swap space"
    collection = RoutingCollection(
        result_by_query={
            query: {
                "documents": [["doc boundary"]],
                "ids": [["chunk_boundary"]],
                "distances": [[0.02]],
                "metadatas": [[{"error_category": "global_oom"}]],
            }
        }
    )
    module = load_exp1_module(
        monkeypatch,
        parser_impl=parser_impl,
        get_collection_impl=lambda: collection,
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_topk_one",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_boundary"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_topk_one", "raw_log": "raw oom log O"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "parsed", "--top-k", "1"],
    )

    module.main()

    assert parser_calls == [module.build_parser_input_state("raw oom log O")]
    assert collection.calls == [
        {
            "query_texts": [query],
            "n_results": 1,
            "where": {"error_category": {"$in": ["global_oom", "general"]}},
        }
    ]

    df = module.pd.read_csv(exp_results_dir / "exp1_retrieval_parsed_top1_results.csv")
    assert df.loc[0, "top_k"] == 1
    assert df.loc[0, "retrieved_chunks"] == "['chunk_boundary']"
    assert df.loc[0, "recall"] == pytest.approx(1.0)


def test_main_parsed_mode_defaults_missing_expected_oom_type_to_unknown(monkeypatch, tmp_path):
    parser_calls = []

    def parser_impl(state):
        parser_calls.append(state)
        return {
            "parsed_fields": {
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": 0,
            }
        }

    query = "unknown CONSTRAINT_NONE no swap space"
    collection = RoutingCollection(
        result_by_query={
            query: {
                "documents": [["doc unknown"]],
                "ids": [["chunk_unknown"]],
                "distances": [[0.07]],
                "metadatas": [[{"error_category": "general"}]],
            }
        }
    )
    module = load_exp1_module(
        monkeypatch,
        parser_impl=parser_impl,
        get_collection_impl=lambda: collection,
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_4",
                "relevant_chunk_ids": ["chunk_unknown"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_4", "raw_log": "raw oom log D"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "parsed", "--top-k", "4"],
    )

    module.main()

    assert parser_calls == [module.build_parser_input_state("raw oom log D")]
    assert collection.calls == [
        {
            "query_texts": [query],
            "n_results": 4,
            "where": {"error_category": {"$in": ["unknown", "general"]}},
        }
    ]

    output_path = exp_results_dir / "exp1_retrieval_parsed_top4_results.csv"
    df = module.pd.read_csv(output_path)
    assert df.loc[0, "expected_oom_type"] == "unknown"
    assert df.loc[0, "retrieved_chunks"] == "['chunk_unknown']"
    assert df.loc[0, "recall"] == pytest.approx(1.0)


@pytest.mark.xfail(
    reason="`main()` does not currently reject dataset rows whose `log_id` cannot be joined to `oom_logs.jsonl`",
    strict=True,
)
def test_main_should_fail_when_log_id_join_is_missing_in_raw_mode(monkeypatch, tmp_path):
    collection = RoutingCollection()
    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: collection)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "missing_log_id",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_expected"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "different_log_id", "raw_log": "raw oom log E"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"],
    )

    with pytest.raises(ValueError, match="log_id.*not found|raw_log.*missing"):
        module.main()

    assert collection.calls == []


@pytest.mark.xfail(
    reason="`main()` raw mode still sends empty raw query strings to the collection instead of rejecting them early",
    strict=True,
)
def test_main_should_fail_when_raw_mode_query_string_is_empty(monkeypatch, tmp_path):
    collection = RoutingCollection()
    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: collection)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_empty_raw_query",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": [],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_empty_raw_query", "raw_log": ""}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"],
    )

    with pytest.raises(ValueError, match="raw query.*empty|raw_log.*empty"):
        module.main()

    assert collection.calls == []


@pytest.mark.xfail(
    reason="`main()` parsed mode still sends empty raw logs through parser/query flow instead of rejecting them early",
    strict=True,
)
def test_main_should_fail_when_raw_log_is_empty_in_parsed_mode(monkeypatch, tmp_path):
    parser_calls = []

    def parser_impl(state):
        parser_calls.append(state)
        return {"parsed_fields": {"constraint": "CONSTRAINT_NONE", "swap_total_kb": 0}}

    collection = RoutingCollection()
    module = load_exp1_module(
        monkeypatch,
        parser_impl=parser_impl,
        get_collection_impl=lambda: collection,
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_empty_raw",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": [],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_empty_raw", "raw_log": ""}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "parsed", "--top-k", "2"],
    )

    with pytest.raises(ValueError, match="raw_log.*empty"):
        module.main()

    assert parser_calls == []
    assert collection.calls == []


@pytest.mark.xfail(
    reason="`main()` parsed mode does not currently validate that parser output `parsed_fields` is a mapping before building a query",
    strict=True,
)
def test_main_should_reject_non_mapping_parsed_fields(monkeypatch, tmp_path):
    parser_calls = []

    def parser_impl(state):
        parser_calls.append(state)
        return {"parsed_fields": ["not", "a", "mapping"]}

    collection = RoutingCollection()
    module = load_exp1_module(
        monkeypatch,
        parser_impl=parser_impl,
        get_collection_impl=lambda: collection,
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_bad_shape",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": [],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_bad_shape", "raw_log": "raw oom log F"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "parsed", "--top-k", "2"],
    )

    with pytest.raises(ValueError, match="parsed_fields.*mapping|invalid parsed_fields"):
        module.main()

    assert parser_calls == [module.build_parser_input_state("raw oom log F")]
    assert collection.calls == []


@pytest.mark.xfail(
    reason="`exp1_retrieval.main()` raw mode still swallows collection.query failures instead of surfacing them",
    strict=True,
)
def test_main_raw_mode_should_surface_query_error(monkeypatch, tmp_path):
    collection = RoutingCollection(should_raise=True)

    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: collection)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_3",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_expected"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_3", "raw_log": "raw oom log C"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"],
    )

    with pytest.raises(RuntimeError, match="query failed"):
        module.main()


@pytest.mark.xfail(
    reason="`main()` raw mode assumes query result ids are complete instead of rejecting partially corrupted responses",
    strict=True,
)
def test_main_raw_mode_should_reject_partially_corrupted_query_response(monkeypatch, tmp_path):
    collection = RoutingCollection(
        result_by_query={
            "raw oom log G": {
                "documents": [["doc one", "doc two"]],
                "ids": [["chunk_1"]],
                "distances": [[0.1]],
                "metadatas": [[{"error_category": "global_oom"}]],
            }
        }
    )

    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: collection)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_corrupt_raw",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_1"],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_corrupt_raw", "raw_log": "raw oom log G"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "2"],
    )

    with pytest.raises(ValueError, match="query result.*corrupt|ids.*incomplete"):
        module.main()


@pytest.mark.xfail(
    reason="`main()` does not currently validate malformed `qa_ground_truth.jsonl` rows before using them",
    strict=True,
)
def test_main_should_reject_ground_truth_row_with_non_list_relevant_chunk_ids(monkeypatch, tmp_path):
    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: RoutingCollection())
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_bad_gt",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": "chunk_1",
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_bad_gt", "raw_log": "raw oom log H"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"],
    )

    with pytest.raises(ValueError, match="relevant_chunk_ids.*list|invalid ground truth row"):
        module.main()


@pytest.mark.xfail(
    reason="`main()` does not currently validate that every `relevant_chunk_ids` item is a string chunk id",
    strict=True,
)
def test_main_should_reject_ground_truth_row_with_non_string_relevant_chunk_id_items(
    monkeypatch,
    tmp_path,
):
    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: RoutingCollection())
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_bad_chunk_item",
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": ["chunk_ok", 123],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_bad_chunk_item", "raw_log": "raw oom log K"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"],
    )

    with pytest.raises(ValueError, match="relevant_chunk_ids.*string|invalid ground truth row"):
        module.main()


@pytest.mark.xfail(
    reason="`main()` does not currently validate that `expected_oom_type` is a string before using it in query/filter construction",
    strict=True,
)
def test_main_should_reject_ground_truth_row_with_non_string_expected_oom_type(monkeypatch, tmp_path):
    parser_calls = []

    def parser_impl(state):
        parser_calls.append(state)
        return {
            "parsed_fields": {
                "constraint": "CONSTRAINT_NONE",
                "swap_total_kb": 0,
            }
        }

    module = load_exp1_module(
        monkeypatch,
        parser_impl=parser_impl,
        get_collection_impl=lambda: RoutingCollection(),
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "log_id": "log_bad_type",
                "expected_oom_type": {"value": "global_oom"},
                "relevant_chunk_ids": [],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_bad_type", "raw_log": "raw oom log I"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "parsed", "--top-k", "2"],
    )

    with pytest.raises(ValueError, match="expected_oom_type.*string|invalid ground truth row"):
        module.main()

    assert parser_calls == []


@pytest.mark.xfail(
    reason="`main()` does not currently reject empty `qa_ground_truth.jsonl` row objects before using default fallbacks",
    strict=True,
)
def test_main_should_reject_empty_ground_truth_row(monkeypatch, tmp_path):
    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: RoutingCollection())
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps({}) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_present", "raw_log": "raw oom log L"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"],
    )

    with pytest.raises(ValueError, match="ground truth row.*empty|invalid ground truth row"):
        module.main()


@pytest.mark.xfail(
    reason="`main()` does not currently reject `qa_ground_truth.jsonl` rows that are missing `log_id`",
    strict=True,
)
def test_main_should_reject_ground_truth_row_missing_log_id(monkeypatch, tmp_path):
    module = load_exp1_module(monkeypatch, get_collection_impl=lambda: RoutingCollection())
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "tqdm", lambda iterable, desc=None: iterable)

    data_dir = tmp_path / "data"
    exp_results_dir = data_dir / "exp_results"
    exp_results_dir.mkdir(parents=True)
    (data_dir / "qa_ground_truth.jsonl").write_text(
        json.dumps(
            {
                "expected_oom_type": "global_oom",
                "relevant_chunk_ids": [],
            }
        ) + "\n",
        encoding="utf-8",
    )
    (data_dir / "oom_logs.jsonl").write_text(
        json.dumps({"log_id": "log_present", "raw_log": "raw oom log J"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp1_retrieval.py", "--query-mode", "raw", "--top-k", "1"],
    )

    with pytest.raises(ValueError, match="log_id.*required|invalid ground truth row"):
        module.main()


@pytest.mark.xfail(
    reason="`run_parsed_retrieval_query()` does not currently defend against malformed collection responses",
    strict=True,
)
def test_run_parsed_retrieval_query_should_handle_malformed_collection_response(monkeypatch):
    module = load_exp1_module(monkeypatch)
    collection = DummyCollection(
        result={
            "documents": [["doc one"]],
            "ids": [[]],
            "distances": [[0.1]],
            "metadatas": [[{"error_category": "global_oom"}]],
        }
    )

    result = module.run_parsed_retrieval_query(
        oom_type="global_oom",
        parsed_fields={"constraint": "CONSTRAINT_NONE", "swap_total_kb": 0},
        collection=collection,
        top_k=5,
    )

    assert result["query_used"] == "global_oom CONSTRAINT_NONE no swap space"
    assert result["chunks"] == []
    assert result["total_found"] == 0
    assert "error" in result


@pytest.mark.xfail(
    reason="`run_parsed_retrieval_query()` assumes query result arrays are complete and aligned instead of handling partial corruption",
    strict=True,
)
def test_run_parsed_retrieval_query_should_handle_partially_corrupted_query_response(monkeypatch):
    module = load_exp1_module(monkeypatch)
    collection = DummyCollection(
        result={
            "documents": [["doc one", "doc two"]],
            "ids": [["chunk_1"]],
            "distances": [[0.1]],
            "metadatas": [[{"error_category": "global_oom"}]],
        }
    )

    result = module.run_parsed_retrieval_query(
        oom_type="global_oom",
        parsed_fields={"constraint": "CONSTRAINT_NONE", "swap_total_kb": 0},
        collection=collection,
        top_k=5,
    )

    assert result["query_used"] == "global_oom CONSTRAINT_NONE no swap space"
    assert result["chunks"] == []
    assert result["total_found"] == 0
    assert "error" in result