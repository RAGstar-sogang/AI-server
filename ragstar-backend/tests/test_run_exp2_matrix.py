import importlib
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_aggregate_matrix_outputs_merges_result_and_summary_files(tmp_path):
    module = importlib.import_module("experiments.run_exp2_matrix")

    result_dir = tmp_path / "data" / "exp_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    model_modes = ["base_llm", "rag_agent"]
    llms = ["llama3", "gemma4"]

    for model_mode in model_modes:
        for llm_name in llms:
            output_paths = module.build_output_paths(result_dir, model_mode, llm_name)
            pd.DataFrame(
                [
                    {
                        "log_id": f"{model_mode}_{llm_name}",
                        "model_mode": model_mode,
                        "llm": llm_name,
                        "expected_oom_type": "global_oom",
                        "generated_oom_type": "global_oom",
                        "category_match": 1,
                        "evidence_recall": 1.0,
                        "action_guide_similarity": 1.0,
                    }
                ]
            ).to_csv(output_paths["results"], index=False)
            pd.DataFrame(
                [
                    {
                        "model_mode": model_mode,
                        "llm": llm_name,
                        "expected_oom_type": "global_oom",
                        "n": 1,
                        "category_match": 1.0,
                        "evidence_recall": 1.0,
                        "action_guide_similarity": 1.0,
                    },
                    {
                        "model_mode": model_mode,
                        "llm": llm_name,
                        "expected_oom_type": "Overall",
                        "n": 1,
                        "category_match": 1.0,
                        "evidence_recall": 1.0,
                        "action_guide_similarity": 1.0,
                    },
                ]
            ).to_csv(output_paths["summary"], index=False)

    aggregate_paths = module.aggregate_matrix_outputs(result_dir, model_modes, llms)

    all_results = pd.read_csv(aggregate_paths["all_results"])
    all_summary = pd.read_csv(aggregate_paths["all_summary"])
    overall_summary = pd.read_csv(aggregate_paths["overall_summary"])

    assert set(all_results["model_mode"]) == {"base_llm", "rag_agent"}
    assert set(all_results["llm"]) == {"llama3", "gemma4"}
    assert len(all_summary) == 8
    assert set(overall_summary["expected_oom_type"]) == {"Overall"}
    assert len(overall_summary) == 4


def test_main_runs_matrix_and_writes_combined_csvs(monkeypatch, tmp_path):
    module = importlib.import_module("experiments.run_exp2_matrix")

    project_root = tmp_path
    result_dir = project_root / "data" / "exp_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(module, "RESULT_DIR", result_dir)
    monkeypatch.setattr(module, "EXP2_SCRIPT", project_root / "experiments" / "exp2_generation.py")

    calls = []

    def fake_run(command, cwd=None, check=None):
        calls.append(command)
        model_mode = command[command.index("--model-mode") + 1]
        llm_name = command[command.index("--llm") + 1]
        output_paths = module.build_output_paths(result_dir, model_mode, llm_name)
        pd.DataFrame(
            [
                {
                    "log_id": f"{model_mode}_{llm_name}",
                    "model_mode": model_mode,
                    "llm": llm_name,
                    "expected_oom_type": "global_oom",
                    "generated_oom_type": "global_oom",
                    "category_match": 1,
                    "evidence_recall": 1.0,
                    "action_guide_similarity": 1.0,
                }
            ]
        ).to_csv(output_paths["results"], index=False)
        pd.DataFrame(
            [
                {
                    "model_mode": model_mode,
                    "llm": llm_name,
                    "expected_oom_type": "Overall",
                    "n": 1,
                    "category_match": 1.0,
                    "evidence_recall": 1.0,
                    "action_guide_similarity": 1.0,
                }
            ]
        ).to_csv(output_paths["summary"], index=False)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_exp2_matrix.py",
        "--model-modes",
        "base_llm",
        "rag_agent",
        "--llms",
        "llama3",
        "gemma4",
    ])

    module.main()

    assert len(calls) == 4
    assert (result_dir / "exp2_generation_all_results.csv").exists()
    assert (result_dir / "exp2_generation_all_summary.csv").exists()
    assert (result_dir / "exp2_generation_all_overall_summary.csv").exists()