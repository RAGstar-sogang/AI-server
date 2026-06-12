"""Re-run all 56 Naive logs for a given model and dump every JSON candidate's
classification.oom_type, so we can audit whether the picker is missing a
correct candidate.

Writes data/exp_results/debug_<model_slug>_naive_picker.csv with columns:
  log_id, expected_oom_type, picked_oom_type, all_candidates_oom_type,
  any_candidate_correct, n_candidates, raw_len
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "exp2_generation", str(PROJECT_ROOT / "experiments" / "exp2_generation.py")
)
m = importlib.util.module_from_spec(spec)
sys.modules["exp2_generation"] = m
spec.loader.exec_module(m)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="vLLM model name (e.g. qwen3.5-9b)")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--provider", default="vllm", choices=["vllm", "openai"])
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    oom_logs = {
        r["log_id"]: r["raw_log"]
        for r in (json.loads(l) for l in (data_dir / "oom_logs.jsonl").open())
    }
    qa = {
        r["log_id"]: r
        for r in (json.loads(l) for l in (data_dir / "qa_ground_truth.jsonl").open())
    }

    cond = m.Condition(args.model, "naive", "openai" if args.provider == "openai" else "qwen", 0.0, args.tp, args.provider, "x")
    if cond.provider == "vllm":
        served = m.ensure_vllm_model(cond.model)
        print(f"vLLM ready: served_name={served}")

    slug = re.sub(r"[^a-z0-9]+", "_", args.model.lower()).strip("_")
    out_path = PROJECT_ROOT / "data" / "exp_results" / f"debug_{slug}_naive_picker.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "log_id,expected_oom_type,picked_oom_type,all_candidates_oom_type,"
            "any_candidate_correct,n_candidates,raw_len\n"
        )

    import time as _time
    openai_min_interval = 25 if args.provider == "openai" else 0
    last_call = 0.0

    log_ids = sorted(qa.keys())  # log_000 .. log_055
    for i, lid in enumerate(log_ids, 1):
        raw_log = oom_logs.get(lid)
        if not raw_log:
            continue
        expected = qa[lid]["expected_oom_type"]
        if openai_min_interval > 0 and last_call > 0:
            elapsed = _time.time() - last_call
            if elapsed < openai_min_interval:
                _time.sleep(openai_min_interval - elapsed)
        last_call = _time.time()
        stats = m.CallStats()
        llm = m.build_tracked_chat_llm(cond.model, stats, cond.provider, json_mode=False)
        try:
            generated, raw_text = m.run_naive(llm, raw_log)
        except Exception as exc:
            print(f"  [{i:02d}/{len(log_ids)}] {lid} ERROR: {exc}")
            continue

        picked = m.extract_fields(generated, "naive")["oom_type"]
        candidates = m._extract_balanced_json_objects(raw_text)
        cand_ooms: list[str] = []
        for c in candidates:
            try:
                p = json.loads(c)
            except Exception:
                continue
            if not isinstance(p, dict):
                continue
            cls = p.get("classification")
            if isinstance(cls, dict):
                ot = cls.get("oom_type")
                if ot is not None:
                    cand_ooms.append(str(ot))
        any_correct = expected.strip().lower() in {x.strip().lower() for x in cand_ooms}

        row = (
            f'{lid},{expected},{picked},'
            f'"{ " | ".join(cand_ooms) }",'
            f'{int(any_correct)},{len(cand_ooms)},{len(raw_text)}\n'
        )
        with out_path.open("a", encoding="utf-8") as f:
            f.write(row)
        print(
            f"  [{i:02d}/{len(log_ids)}] {lid} expected={expected} picked={picked!r} "
            f"cand_ooms={cand_ooms} any_correct={any_correct}"
        )

    print(f"\nDONE → {out_path}")


if __name__ == "__main__":
    main()
