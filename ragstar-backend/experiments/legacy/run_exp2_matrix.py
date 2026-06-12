import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP2_SCRIPT = PROJECT_ROOT / "experiments" / "exp2_generation.py"
RESULT_DIR = PROJECT_ROOT / "data" / "exp_results"


def slugify_model_name(model_name: str) -> str:
    return "_".join(
        chunk for chunk in "".join(
            ch.lower() if ch.isalnum() else "_"
            for ch in str(model_name).strip()
        ).split("_")
        if chunk
    ) or "default"


def build_output_paths(result_dir: Path, model_mode: str, llm_name: str) -> dict[str, Path]:
    llm_slug = slugify_model_name(llm_name)
    prefix = result_dir / f"exp2_generation_{model_mode}_{llm_slug}"
    return {
        "results": prefix.with_name(f"{prefix.name}_results.csv"),
        "summary": prefix.with_name(f"{prefix.name}_summary.csv"),
    }


def aggregate_matrix_outputs(result_dir: Path, model_modes: list[str], llms: list[str]) -> dict[str, Path]:
    result_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for model_mode in model_modes:
        for llm_name in llms:
            output_paths = build_output_paths(result_dir, model_mode, llm_name)

            result_file = output_paths["results"]
            summary_file = output_paths["summary"]

            if not result_file.exists():
                raise FileNotFoundError(f"Missing result CSV: {result_file}")
            if not summary_file.exists():
                raise FileNotFoundError(f"Missing summary CSV: {summary_file}")

            result_frames.append(pd.read_csv(result_file))
            summary_frames.append(pd.read_csv(summary_file))

    all_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    all_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

    overall_summary = all_summary[all_summary["expected_oom_type"] == "Overall"].copy() if not all_summary.empty else pd.DataFrame()
    if not overall_summary.empty:
        overall_summary = overall_summary.sort_values(by=["model_mode", "llm"]).reset_index(drop=True)

    all_results_file = result_dir / "exp2_generation_all_results.csv"
    all_summary_file = result_dir / "exp2_generation_all_summary.csv"
    overall_summary_file = result_dir / "exp2_generation_all_overall_summary.csv"

    all_results.to_csv(all_results_file, index=False, encoding="utf-8-sig")
    all_summary.to_csv(all_summary_file, index=False, encoding="utf-8-sig")
    overall_summary.to_csv(overall_summary_file, index=False, encoding="utf-8-sig")

    return {
        "all_results": all_results_file,
        "all_summary": all_summary_file,
        "overall_summary": overall_summary_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2 over model-mode x LLM combinations.")
    parser.add_argument(
        "--model-modes",
        nargs="+",
        default=["base_llm", "rag_agent"],
        choices=["base_llm", "rag_agent"],
        help="Experiment 2 model modes to execute.",
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        default=["llama3", "qwen3.5:9b", "gemma4"],
        help="Inference models to compare. Embeddings remain fixed.",
    )
    parser.add_argument(
        "--label-similarity-threshold",
        type=float,
        default=None,
        help="Optional Node 2 threshold override for rag_agent runs.",
    )
    parser.add_argument(
        "--label-similarity-margin",
        type=float,
        default=None,
        help="Optional Node 2 margin override for rag_agent runs.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke Experiment 2.",
    )
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(args.model_modes) * len(args.llms)
    run_index = 0

    for model_mode in args.model_modes:
        for llm_name in args.llms:
            run_index += 1
            command = [
                args.python,
                str(EXP2_SCRIPT),
                "--model-mode",
                model_mode,
                "--llm",
                llm_name,
            ]

            if model_mode == "rag_agent":
                if args.label_similarity_threshold is not None:
                    command.extend([
                        "--label-similarity-threshold",
                        str(args.label_similarity_threshold),
                    ])
                if args.label_similarity_margin is not None:
                    command.extend([
                        "--label-similarity-margin",
                        str(args.label_similarity_margin),
                    ])

            print(f"[{run_index}/{total_runs}] Running model_mode={model_mode}, llm={llm_name}")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    aggregate_paths = aggregate_matrix_outputs(RESULT_DIR, args.model_modes, args.llms)
    print(f"Combined results saved: {aggregate_paths['all_results']}")
    print(f"Combined summary saved: {aggregate_paths['all_summary']}")
    print(f"Overall-only summary saved: {aggregate_paths['overall_summary']}")


if __name__ == "__main__":
    main()