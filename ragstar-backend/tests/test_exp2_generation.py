import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))


@pytest.fixture
def exp2_module():
	module = importlib.import_module("experiments.exp2_generation")
	module.get_exp2_llm.cache_clear()
	module.get_exp2_chat_llm.cache_clear()
	module.get_exp2_embeddings.cache_clear()
	yield module
	module.get_exp2_llm.cache_clear()
	module.get_exp2_chat_llm.cache_clear()
	module.get_exp2_embeddings.cache_clear()


class DummyResponse:
	def __init__(self, content):
		self.content = content


class DummyLLM:
	def __init__(self, content):
		self.content = content
		self.prompts = []

	def invoke(self, prompt):
		self.prompts.append(prompt)
		return DummyResponse(self.content)


def test_extract_fields_base_llm_flattens_expected_contract(exp2_module, monkeypatch):
	monkeypatch.setattr(exp2_module, "normalize_generated_oom_type", lambda raw: "global_oom")

	generated_json = {
		"classification": {"oom_type": "Global OOM"},
		"final_answer": {
			"diagnosis": {
				"evidence": ["anon-rss=875680kB", "constraint=CONSTRAINT_NONE"],
			},
			"action_guide": {
				"immediate": ["check top memory process"],
				"recommended": ["tune vm.overcommit_memory"],
				"further_investigation": ["review allocator behavior"],
			},
		},
	}

	fields = exp2_module.extract_fields(generated_json, "base_llm")

	assert fields == {
		"oom_type": "global_oom",
		"evidence": ["anon-rss=875680kB", "constraint=CONSTRAINT_NONE"],
		"action_guide": [
			"check top memory process",
			"tune vm.overcommit_memory",
			"review allocator behavior",
		],
	}


def test_extract_fields_rag_agent_uses_state_diagnosis_path(exp2_module, monkeypatch):
	monkeypatch.setattr(exp2_module, "normalize_generated_oom_type", lambda raw: "cgroup_oom")

	generated_json = {
		"classification": {"oom_type": "cgroup_oom"},
		"diagnosis": {
			"diagnosis": {
				"evidence": "not-a-list",
			},
			"action_guide": {
				"immediate": ["inspect cgroup limit"],
				"recommended": None,
				"further_investigation": ["check container memory spikes"],
			},
		},
	}

	fields = exp2_module.extract_fields(generated_json, "rag_agent")

	assert fields == {
		"oom_type": "cgroup_oom",
		"evidence": [],
		"action_guide": ["inspect cgroup limit", "check container memory spikes"],
	}


def test_run_naive_llm_parses_json_object_from_response_text(exp2_module, monkeypatch):
	llm = DummyLLM(
		'prefix {"classification": {"oom_type": "global_oom"}, "final_answer": {}} suffix'
	)
	monkeypatch.setattr(exp2_module, "get_exp2_llm", lambda model_name=None: llm)

	result = exp2_module.run_naive_llm("kernel log body")

	assert result["classification"]["oom_type"] == "global_oom"
	assert llm.prompts, "LLM should be invoked with the formatted prompt"
	assert "kernel log body" in llm.prompts[0]


def test_run_naive_llm_returns_error_when_no_json_is_present(exp2_module, monkeypatch):
	monkeypatch.setattr(exp2_module, "get_exp2_llm", lambda model_name=None: DummyLLM("no json here"))

	result = exp2_module.run_naive_llm("kernel log body")

	assert result == {"_error": "naive_llm_response_missing_json"}


def test_run_naive_llm_returns_error_when_json_is_malformed(exp2_module, monkeypatch):
	monkeypatch.setattr(
		exp2_module,
		"get_exp2_llm",
		lambda model_name=None: DummyLLM('prefix {"classification": invalid json} suffix'),
	)

	result = exp2_module.run_naive_llm("kernel log body")

	assert result["_error"].startswith("naive_llm_error:")


def test_run_naive_llm_returns_not_dict_when_parsed_json_payload_is_array(exp2_module, monkeypatch):
	monkeypatch.setattr(exp2_module, "get_exp2_llm", lambda model_name=None: DummyLLM('{"wrapped": true}'))
	original_loads = exp2_module.json.loads
	monkeypatch.setattr(exp2_module.json, "loads", lambda text: [original_loads(text)])

	result = exp2_module.run_naive_llm("kernel log body")

	assert result == {"_error": "naive_llm_response_not_dict"}


def test_run_naive_llm_uses_explicit_llm_override(exp2_module, monkeypatch):
	requested_models = []
	llm = DummyLLM('{"classification": {"oom_type": "global_oom"}, "final_answer": {}}')

	def fake_get_exp2_llm(model_name=None):
		requested_models.append(model_name)
		return llm

	monkeypatch.setattr(exp2_module, "get_exp2_llm", fake_get_exp2_llm)

	result = exp2_module.run_naive_llm("kernel log body", llm_name="llama3")

	assert requested_models == ["llama3"]
	assert result["classification"]["oom_type"] == "global_oom"


def test_run_rag_agent_returns_partial_state_when_node4_fails(exp2_module, monkeypatch):
	calls = []

	class FakeEmbeddings:
		pass

	monkeypatch.setattr(exp2_module, "get_exp2_embeddings", lambda: FakeEmbeddings())
	monkeypatch.setattr(exp2_module, "create_initial_state", lambda raw_log: {"raw_log": raw_log})

	def node1(state):
		calls.append("node1")
		return {**state, "parsed_fields": {"constraint": "CONSTRAINT_NONE"}}

	def node2(state):
		calls.append("node2")
		return {**state, "classification": {"oom_type": "global_oom"}}

	def node3(state):
		calls.append("node3")
		return {**state, "tool_results": {"memory": {"top5_total_ratio": 0.54}}}

	def node4(_state):
		calls.append("node4")
		raise RuntimeError("schema drift")

	monkeypatch.setattr(exp2_module, "node_1_parser", node1)
	monkeypatch.setattr(exp2_module, "node_2_classifier", node2)
	monkeypatch.setattr(exp2_module, "node_3_executor", node3)
	monkeypatch.setattr(exp2_module, "node_4_synthesizer", node4)

	result = exp2_module.run_rag_agent(
		"kernel log body",
		label_similarity_threshold=0.73,
		label_similarity_margin=0.05,
	)

	assert calls == ["node1", "node2", "node3", "node4"]
	assert result["parsed_fields"] == {"constraint": "CONSTRAINT_NONE"}
	assert result["classification"] == {"oom_type": "global_oom"}
	assert result["tool_results"] == {"memory": {"top5_total_ratio": 0.54}}
	assert result["_error"] == "rag_agent_error: schema drift"
	assert isinstance(result["label_embeddings"], FakeEmbeddings)
	assert result["label_similarity_threshold"] == 0.73
	assert result["label_similarity_margin"] == 0.05


def test_run_rag_agent_returns_final_state_on_happy_path(exp2_module, monkeypatch):
	calls = []
	final_state = {
		"raw_log": "kernel log body",
		"label_embeddings": object(),
		"label_similarity_threshold": 0.8,
		"label_similarity_margin": 0.1,
		"parsed_fields": {"constraint": "CONSTRAINT_MEMCG"},
		"classification": {"oom_type": "cgroup_oom", "confidence": "high"},
		"tool_results": {"memory": {"top5_total_ratio": 0.61}},
		"diagnosis": {"diagnosis": {"root_cause": "memcg limit hit"}},
	}

	class FakeEmbeddings:
		pass

	monkeypatch.setattr(exp2_module, "get_exp2_embeddings", lambda: FakeEmbeddings())
	monkeypatch.setattr(exp2_module, "create_initial_state", lambda raw_log: {"raw_log": raw_log})

	def node1(state):
		calls.append("node1")
		return {**state, "parsed_fields": final_state["parsed_fields"]}

	def node2(state):
		calls.append("node2")
		return {**state, "classification": final_state["classification"]}

	def node3(state):
		calls.append("node3")
		return {**state, "tool_results": final_state["tool_results"]}

	def node4(state):
		calls.append("node4")
		return {**state, "diagnosis": final_state["diagnosis"]}

	monkeypatch.setattr(exp2_module, "node_1_parser", node1)
	monkeypatch.setattr(exp2_module, "node_2_classifier", node2)
	monkeypatch.setattr(exp2_module, "node_3_executor", node3)
	monkeypatch.setattr(exp2_module, "node_4_synthesizer", node4)

	result = exp2_module.run_rag_agent(
		"kernel log body",
		label_similarity_threshold=0.8,
		label_similarity_margin=0.1,
	)

	assert calls == ["node1", "node2", "node3", "node4"]
	assert result["parsed_fields"] == final_state["parsed_fields"]
	assert result["classification"] == final_state["classification"]
	assert result["tool_results"] == final_state["tool_results"]
	assert result["diagnosis"] == final_state["diagnosis"]
	assert "_error" not in result


def test_run_rag_agent_injects_explicit_chat_llm(exp2_module, monkeypatch):
	chat_llm = object()

	class FakeEmbeddings:
		pass

	monkeypatch.setattr(exp2_module, "get_exp2_embeddings", lambda: FakeEmbeddings())
	monkeypatch.setattr(exp2_module, "get_exp2_chat_llm", lambda model_name=None: chat_llm)
	monkeypatch.setattr(exp2_module, "create_initial_state", lambda raw_log: {"raw_log": raw_log})

	def passthrough(state):
		assert state["llm"] is chat_llm
		return state

	monkeypatch.setattr(exp2_module, "node_1_parser", passthrough)
	monkeypatch.setattr(exp2_module, "node_2_classifier", passthrough)
	monkeypatch.setattr(exp2_module, "node_3_executor", passthrough)
	monkeypatch.setattr(exp2_module, "node_4_synthesizer", passthrough)

	result = exp2_module.run_rag_agent(
		"kernel log body",
		label_similarity_threshold=0.73,
		label_similarity_margin=0.05,
		llm_name="qwen3.5:9b",
	)

	assert result["llm"] is chat_llm


def test_extract_embedding_diagnostics_returns_expected_defaults(exp2_module):
	diagnostics = exp2_module.extract_embedding_diagnostics(
		{
			"_log_id": "log_007",
			"classification": {
				"oom_type": "swap_exhaustion",
				"deterministic_oom_type": "unknown",
				"raw_llm_oom_type": "swap",
				"normalized_llm_oom_type": "swap_exhaustion",
				"embedding_debug": {
					"candidate_text": "swap",
					"best_label": "swap_exhaustion",
					"best_score": 0.88,
					"score_margin": 0.09,
					"passes_threshold": True,
					"passes_margin": True,
				},
			},
		},
		"swap_exhaustion",
	)

	assert diagnostics == {
		"log_id": "log_007",
		"expected_oom_type": "swap_exhaustion",
		"deterministic_oom_type": "unknown",
		"final_oom_type": "swap_exhaustion",
		"raw_llm_oom_type": "swap",
		"normalized_llm_oom_type": "swap_exhaustion",
		"candidate_text": "swap",
		"canonical_label": "",
		"accepted_label": "",
		"best_label": "swap_exhaustion",
		"best_score": 0.88,
		"second_label": "",
		"second_score": -1.0,
		"score_margin": 0.09,
		"passes_threshold": 1,
		"passes_margin": 1,
		"normalization_source": "",
		"similarity_threshold": exp2_module.DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
		"similarity_margin": exp2_module.DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
	}


@pytest.mark.parametrize(
	("row", "threshold", "margin", "expected"),
	[
		pytest.param(
			{"deterministic_oom_type": "global_oom"},
			0.8,
			0.1,
			"global_oom",
			id="deterministic_wins",
		),
		pytest.param(
			{"deterministic_oom_type": "unknown", "canonical_label": "cgroup_oom"},
			0.8,
			0.1,
			"cgroup_oom",
			id="canonical_label_wins",
		),
		pytest.param(
			{
				"deterministic_oom_type": "unknown",
				"canonical_label": "",
				"best_label": "page_alloc_failure",
				"best_score": 0.91,
				"score_margin": 0.12,
			},
			0.8,
			0.1,
			"page_alloc_failure",
			id="embedding_accepts",
		),
		pytest.param(
			{
				"deterministic_oom_type": "unknown",
				"canonical_label": "",
				"best_label": "page_alloc_failure",
				"best_score": 0.74,
				"score_margin": 0.02,
			},
			0.8,
			0.1,
			"unknown",
			id="embedding_rejects",
		),
	],
)
def test_predict_with_threshold_matches_contract(exp2_module, row, threshold, margin, expected):
	predicted = exp2_module._predict_with_threshold(pd.Series(row), threshold, margin)
	assert predicted == expected


def test_build_embedding_calibration_summary_applies_full_tie_break_order(exp2_module):
	embedding_df = pd.DataFrame(
		[
			{
				"expected_oom_type": "global_oom",
				"deterministic_oom_type": "unknown",
				"canonical_label": "",
				"best_label": "global_oom",
				"best_score": 0.84,
				"score_margin": 0.20,
			},
			{
				"expected_oom_type": "cgroup_oom",
				"deterministic_oom_type": "unknown",
				"canonical_label": "",
				"best_label": "cgroup_oom",
				"best_score": 0.84,
				"score_margin": 0.08,
			},
			{
				"expected_oom_type": "swap_exhaustion",
				"deterministic_oom_type": "unknown",
				"canonical_label": "",
				"best_label": "global_oom",
				"best_score": 0.84,
				"score_margin": 0.06,
			},
		]
	)

	calibration_df, best_config = exp2_module.build_embedding_calibration_summary(embedding_df)

	assert not calibration_df.empty
	best_row = calibration_df.iloc[0]
	assert best_row["category_match"] == pytest.approx(2 / 3, rel=1e-3)
	assert best_row["accepted_rate"] == pytest.approx(1.0)
	assert best_row["threshold"] == pytest.approx(0.84)
	assert best_row["margin"] == pytest.approx(0.06)
	assert best_config == {
		"threshold": 0.84,
		"margin": 0.06,
		"category_match": pytest.approx(0.6667, rel=1e-3),
		"accepted_rate": 1.0,
		"n": 3.0,
	}
	assert (
		calibration_df[
			(calibration_df["category_match"] == best_row["category_match"])
			& (calibration_df["accepted_rate"] == best_row["accepted_rate"])
		]["threshold"].max()
	) == pytest.approx(0.84)


def test_main_rag_agent_writes_results_summary_and_embedding_outputs(exp2_module, monkeypatch, tmp_path):
	monkeypatch.setattr(exp2_module, "PROJECT_ROOT", tmp_path)
	monkeypatch.setattr(sys, "argv", ["exp2_generation.py", "--model-mode", "rag_agent", "--llm", "qwen3.5:9b"])
	monkeypatch.setattr(exp2_module, "tqdm", lambda rows, desc=None: rows)

	dataset = [
		{
			"log_id": "log_1",
			"expected_oom_type": "global_oom",
			"ground_truth": {
				"must_include_evidence": ["anon-rss=875680kB"],
				"action_guide": ["check memory pressure"],
			},
		}
	]

	monkeypatch.setattr(exp2_module, "load_jsonl", lambda _path: dataset)
	monkeypatch.setattr(exp2_module, "build_oom_logs_index", lambda _path: {"log_1": "raw oom log"})
	monkeypatch.setattr(
		exp2_module,
		"run_rag_agent",
		lambda raw_log, label_similarity_threshold, label_similarity_margin, llm_name=None: {
			"classification": {
				"oom_type": "global_oom",
				"deterministic_oom_type": "unknown",
				"embedding_debug": {
					"best_label": "global_oom",
					"best_score": 0.88,
					"score_margin": 0.12,
				},
			},
			"diagnosis": {
				"diagnosis": {"evidence": ["anon-rss=875680kB"]},
				"action_guide": {
					"immediate": ["check memory pressure"],
					"recommended": [],
					"further_investigation": [],
				},
			},
		},
	)
	monkeypatch.setattr(exp2_module, "eval_evidence_recall", lambda gt, gen: 1.0)
	monkeypatch.setattr(exp2_module, "eval_action_guide_similarity", lambda gt, gen: 1.0)
	monkeypatch.setattr(
		exp2_module,
		"build_embedding_calibration_summary",
		lambda embedding_df: (
			pd.DataFrame(
				[
					{
						"threshold": 0.81,
						"margin": 0.06,
						"n": 1,
						"category_match": 1.0,
						"accepted_rate": 1.0,
					}
				]
			),
			{
				"threshold": 0.81,
				"margin": 0.06,
				"category_match": 1.0,
				"accepted_rate": 1.0,
				"n": 1.0,
			},
		),
	)

	exp2_module.main()

	result_dir = tmp_path / "data" / "exp_results"
	results_df = pd.read_csv(result_dir / "exp2_generation_rag_agent_qwen3_5_9b_results.csv")
	summary_df = pd.read_csv(result_dir / "exp2_generation_rag_agent_qwen3_5_9b_summary.csv")
	diag_df = pd.read_csv(result_dir / "exp2_generation_rag_agent_qwen3_5_9b_embedding_diagnostics.csv")
	calibration_df = pd.read_csv(result_dir / "exp2_generation_rag_agent_qwen3_5_9b_embedding_calibration.csv")

	row = results_df.iloc[0]
	assert row["log_id"] == "log_1"
	assert row["model_mode"] == "rag_agent"
	assert row["llm"] == "qwen3.5:9b"
	assert row["expected_oom_type"] == "global_oom"
	assert row["generated_oom_type"] == "global_oom"
	assert row["category_match"] == 1
	assert row["evidence_recall"] == pytest.approx(1.0)
	assert row["action_guide_similarity"] == pytest.approx(1.0)
	assert row["n_gt_evidence"] == 1
	assert row["n_gt_action"] == 1
	assert row["n_gen_evidence"] == 1
	assert row["n_gen_action"] == 1
	assert row["had_error"] == 0
	assert pd.isna(row["inference_error"])
	assert set(summary_df["expected_oom_type"]) == {"global_oom", "Overall"}
	assert set(summary_df["llm"]) == {"qwen3.5:9b"}
	assert diag_df.loc[0, "log_id"] == "log_1"
	assert calibration_df.loc[0, "threshold"] == pytest.approx(0.81)


def test_main_base_llm_writes_results_without_embedding_side_outputs(exp2_module, monkeypatch, tmp_path):
	monkeypatch.setattr(exp2_module, "PROJECT_ROOT", tmp_path)
	monkeypatch.setattr(sys, "argv", ["exp2_generation.py", "--model-mode", "base_llm", "--llm", "llama3"])
	monkeypatch.setattr(exp2_module, "tqdm", lambda rows, desc=None: rows)

	dataset = [
		{
			"log_id": "log_base_1",
			"expected_oom_type": "page_alloc_failure",
			"ground_truth": {
				"must_include_evidence": ["order=2"],
				"action_guide": ["check fragmentation"],
			},
		}
	]

	monkeypatch.setattr(exp2_module, "load_jsonl", lambda _path: dataset)
	monkeypatch.setattr(exp2_module, "build_oom_logs_index", lambda _path: {"log_base_1": "raw oom log"})
	monkeypatch.setattr(
		exp2_module,
		"run_naive_llm",
		lambda raw_log, llm_name=None: {
			"classification": {"oom_type": "page_alloc_failure"},
			"final_answer": {
				"diagnosis": {"evidence": ["order=2"]},
				"action_guide": {
					"immediate": ["check fragmentation"],
					"recommended": [],
					"further_investigation": [],
				},
			},
		},
	)
	monkeypatch.setattr(exp2_module, "eval_evidence_recall", lambda gt, gen: 1.0)
	monkeypatch.setattr(exp2_module, "eval_action_guide_similarity", lambda gt, gen: 1.0)

	exp2_module.main()

	result_dir = tmp_path / "data" / "exp_results"
	results_df = pd.read_csv(result_dir / "exp2_generation_base_llm_llama3_results.csv")
	summary_df = pd.read_csv(result_dir / "exp2_generation_base_llm_llama3_summary.csv")

	assert results_df.loc[0, "log_id"] == "log_base_1"
	assert results_df.loc[0, "llm"] == "llama3"
	assert results_df.loc[0, "generated_oom_type"] == "page_alloc_failure"
	assert results_df.loc[0, "had_error"] == 0
	assert set(summary_df["expected_oom_type"]) == {"page_alloc_failure", "Overall"}
	assert not (result_dir / "exp2_generation_base_llm_llama3_embedding_diagnostics.csv").exists()
	assert not (result_dir / "exp2_generation_base_llm_llama3_embedding_calibration.csv").exists()


def test_main_empty_result_path_writes_empty_summary_and_empty_embedding_files(exp2_module, monkeypatch, tmp_path):
	monkeypatch.setattr(exp2_module, "PROJECT_ROOT", tmp_path)
	monkeypatch.setattr(sys, "argv", ["exp2_generation.py", "--model-mode", "rag_agent", "--llm", "gemma4"])
	monkeypatch.setattr(exp2_module, "tqdm", lambda rows, desc=None: rows)

	dataset = [
		{
			"log_id": "missing_log",
			"expected_oom_type": "global_oom",
			"ground_truth": {
				"must_include_evidence": ["anon-rss=875680kB"],
				"action_guide": ["check memory pressure"],
			},
		}
	]

	monkeypatch.setattr(exp2_module, "load_jsonl", lambda _path: dataset)
	monkeypatch.setattr(exp2_module, "build_oom_logs_index", lambda _path: {})

	exp2_module.main()

	result_dir = tmp_path / "data" / "exp_results"
	result_file = result_dir / "exp2_generation_rag_agent_gemma4_results.csv"
	summary_df = pd.read_csv(result_dir / "exp2_generation_rag_agent_gemma4_summary.csv")
	diag_file = result_dir / "exp2_generation_rag_agent_gemma4_embedding_diagnostics.csv"
	calibration_file = result_dir / "exp2_generation_rag_agent_gemma4_embedding_calibration.csv"

	assert result_file.exists()
	assert result_file.read_text(encoding="utf-8-sig").strip() == ""
	assert summary_df.to_dict("records") == [
		{
			"model_mode": "rag_agent",
			"llm": "gemma4",
			"expected_oom_type": "Overall",
			"n": 0,
			"category_match": 0.0,
			"evidence_recall": 0.0,
			"action_guide_similarity": 0.0,
		}
	]
	assert diag_file.exists()
	assert calibration_file.exists()
	assert diag_file.read_text(encoding="utf-8-sig").strip() == ""
	assert calibration_file.read_text(encoding="utf-8-sig").strip() == ""
