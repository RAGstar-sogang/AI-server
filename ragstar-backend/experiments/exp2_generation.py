"""
[Experiment: Model Scaling]
Compare 5 local LLMs (qwen3.5 9B/2B/0.8B, gemma-4 e4b/e2b) + optional GPT 5.2
on the full RAGstar pipeline. Naive (no-RAG) baseline runs only for
qwen3.5-9b among local vLLM models.

Metrics per log
---------------
- accuracy: category_match, evidence_recall, action_guide_similarity
- timing : inference_time_sec, n_llm_calls, input_tokens, output_tokens
- robustness: had_error, error_msg, json_first_try_valid, json_needed_repair
- faithfulness (RAG only): retrieved-chunk evidence overlap (Jaccard on bigrams)

Aggregate per (model, mode)
---------------------------
- mean / p50 / p95 inference time (measured per query)
- mean accuracy / robustness / faithfulness
- mean VRAM per query (measured: nvidia-smi sampled during each query's window)
- estimated cost per query (local: GPU TDP × time × kWh; OpenAI: token price)

Outputs
-------
- data/exp_results/scaling_per_log.csv          # one row per (cond, log)
- data/exp_results/scaling_per_condition.csv    # one row per condition
- experiments/logs/exp2_generation_<TS>.log     # live progress log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load .env into os.environ — sudo strips env vars by default, so callers
# that go through `sudo -E -u smkang ...` may otherwise lose OPENAI_API_KEY.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

# Sweeps swap vLLM models 5+ times per run; eager mode keeps startup fast and
# avoids capture-time OOM on tight 24GB GPUs. start_worker.sh defaults to "0"
# for steady-state serving where compile/CUDA-graph throughput pays off.
os.environ.setdefault("RAGSTAR_VLLM_ENFORCE_EAGER", "1")

from app.agent.graph import create_initial_state
from app.agent.nodes.node_1_parser import node_1_parser
from app.agent.nodes.node_2_classifier import (
    DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    embedding_normalize_oom_type,
    node_2_classifier,
)
from app.agent.nodes.node_3_executor import node_3_executor
from app.agent.nodes.node_4_synthesizer import node_4_synthesizer
from app.agent.rag_runner import run_rag_agent as run_ragstar_agent
from app.core.llm_factory import build_chat_llm, build_exp2_embeddings
from app.core.settings import get_settings
from app.core.vllm_manager import ensure_vllm_model

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
GT_PATH = DATA_DIR / "qa_ground_truth.jsonl"
OOM_LOGS_PATH = DATA_DIR / "oom_logs.jsonl"
KB_CHUNKS_PATH = DATA_DIR / "kb_chunks.jsonl"
RESULT_DIR = DATA_DIR / "exp_results"
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"

# A5000 has 24GB VRAM each (TDP ~230W). Adjust if hardware differs.
GPU_VRAM_GB = 24.0
GPU_TDP_W = 230.0
KWH_PRICE_USD = 0.10  # rough server colocation rate; override via --kwh-price

# OpenAI GPT 5.2 token prices (USD / 1M tokens). Override via env if needed.
GPT_PRICE_INPUT_PER_M = float(os.environ.get("GPT_PRICE_INPUT_PER_M", "5.0"))
GPT_PRICE_OUTPUT_PER_M = float(os.environ.get("GPT_PRICE_OUTPUT_PER_M", "15.0"))


@dataclass
class Condition:
    model: str          # vLLM model name or OpenAI model id
    mode: str           # "rag" or "naive"
    family: str         # "qwen" | "gemma" | "openai"
    size_b: float       # param count in billions (0 for unknown)
    tp: int             # tensor-parallel size used (1 GPU = 1)
    provider: str       # "vllm" | "openai"
    label: str

    @property
    def slug(self) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", f"{self.mode}_{self.model}".lower())
        return s.strip("_")


def build_conditions() -> list[Condition]:
    # Order chosen to minimize vLLM model switches:
    # qwen3.5-9b (rag, naive) -> qwen3.5-2b (rag) -> qwen3.5-0.8b (rag)
    # -> gemma-4-e4b (rag) -> gemma-4-e2b (rag) -> GPT-5.2 (naive, no vLLM switch)
    conds: list[Condition] = [
        Condition("qwen3.5-9b",   "rag",   "qwen",  9.0, 2, "vllm", "Qwen3.5-9B (RAG)"),
        Condition("qwen3.5-9b",   "naive", "qwen",  9.0, 2, "vllm", "Qwen3.5-9B (Naive)"),
        Condition("qwen3.5-2b",   "rag",   "qwen",  2.0, 1, "vllm", "Qwen3.5-2B (RAG)"),
        Condition("qwen3.5-0.8b", "rag",   "qwen",  0.8, 1, "vllm", "Qwen3.5-0.8B (RAG)"),
        Condition("gemma-4-e4b-it", "rag", "gemma", 4.0, 2, "vllm", "Gemma4-E4B (RAG)"),
        Condition("gemma-4-e2b-it", "rag", "gemma", 2.0, 1, "vllm", "Gemma4-E2B (RAG)"),
    ]
    if os.environ.get("OPENAI_API_KEY"):
        gpt_model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
        conds.append(Condition(gpt_model, "naive", "openai", 0.0, 0, "openai", f"{gpt_model} (Naive)"))
    return conds


# =============================================================================
# Logging
# =============================================================================

class TeeLogger:
    """Writes to both a log file and stdout."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("w", encoding="utf-8", buffering=1)  # line-buffered
        self.path = path

    def log(self, msg: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")

    def close(self) -> None:
        self.fh.close()


# =============================================================================
# Tracked LLM wrapper — captures token usage, latency, and JSON-validity
# =============================================================================

@dataclass
class CallStats:
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    raw_responses: list[str] = field(default_factory=list)
    first_try_valid: list[bool] = field(default_factory=list)


def try_parse_json(text: str) -> bool:
    """Lightweight: did the response trivially parse, or after brace-trim?"""
    if not isinstance(text, str):
        return False
    try:
        json.loads(text)
        return True
    except Exception:
        pass
    # Brace trim fallback
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return False
    try:
        json.loads(text[start:end])
        return False  # parsed only after repair
    except Exception:
        return False


def needed_repair_but_parsed(text: str) -> bool:
    if not isinstance(text, str):
        return False
    try:
        json.loads(text)
        return False
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return False
    try:
        json.loads(text[start:end])
        return True
    except Exception:
        return False


try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # langchain<0.1
    from langchain.callbacks.base import BaseCallbackHandler  # type: ignore[no-redef]


class TokenCallback(BaseCallbackHandler):
    """Records token usage and JSON validity for every LLM completion.

    Works with langchain ChatOpenAI's chain composition because we never wrap
    the LLM object — we just attach as a callback that langchain invokes itself.
    """

    def __init__(self, stats: "CallStats"):
        super().__init__()
        self.stats = stats

    def on_llm_end(self, response, **kwargs):  # type: ignore[no-untyped-def]
        generations = getattr(response, "generations", []) or []
        for gen_list in generations:
            for gen in gen_list:
                self.stats.n_calls += 1
                text = getattr(gen, "text", "") or ""
                # AIMessage path: text might be empty when generation is a ChatGeneration
                if not text:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        text = getattr(msg, "content", "") or ""
                self.stats.raw_responses.append(text)
                try:
                    json.loads(text)
                    self.stats.first_try_valid.append(True)
                except Exception:
                    self.stats.first_try_valid.append(False)
        llm_output = getattr(response, "llm_output", None) or {}
        tu = (llm_output.get("token_usage") if isinstance(llm_output, dict) else {}) or {}
        self.stats.input_tokens += int(tu.get("prompt_tokens", 0) or 0)
        self.stats.output_tokens += int(tu.get("completion_tokens", 0) or 0)


MAX_OUTPUT_TOKENS_RAG = 8192
MAX_OUTPUT_TOKENS_NAIVE = 8192
REQUEST_TIMEOUT_SEC = 1800  # very generous; we'd rather wait than retry-burn


def build_tracked_chat_llm(model_name: str, stats: "CallStats", provider: str, json_mode: bool = True):
    """Build a langchain Chat LLM with our TokenCallback attached.

    Bypasses build_chat_llm so we can pass `timeout` directly into the
    ChatOpenAI constructor — setting it post-init via `inner.request_timeout`
    is ignored by ChatOpenAI's pydantic config, and we kept hitting the
    default 120s timeout in Naive runs.
    """
    callback = TokenCallback(stats)
    max_tokens = MAX_OUTPUT_TOKENS_RAG if json_mode else MAX_OUTPUT_TOKENS_NAIVE

    if provider == "openai":
        from langchain_community.chat_models.openai import ChatOpenAI as _Chat
        # GPT-5 family rejects `max_tokens`; use `max_completion_tokens` via model_kwargs.
        model_kwargs: dict[str, Any] = {"max_completion_tokens": max_tokens}
        if json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}
        inner = _Chat(
            model=model_name,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=(os.environ.get("OPENAI_API_KEY") or "").rstrip("."),
            temperature=0,
            timeout=REQUEST_TIMEOUT_SEC,
            request_timeout=REQUEST_TIMEOUT_SEC,
            max_retries=5,  # honor 429s with OpenAI client's built-in backoff
            model_kwargs=model_kwargs,
            callbacks=[callback],
        )
    else:
        from langchain_community.chat_models.openai import ChatOpenAI as _Chat
        settings = get_settings()
        model_kwargs = {}
        if json_mode and settings.vllm_json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}
        inner = _Chat(
            model=model_name,
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            temperature=settings.vllm_temperature,
            timeout=REQUEST_TIMEOUT_SEC,
            request_timeout=REQUEST_TIMEOUT_SEC,
            max_retries=0,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            callbacks=[callback],
        )
    return inner


# =============================================================================
# Data loading
# =============================================================================

def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_oom_logs_index(path: Path) -> dict[str, str]:
    return {r["log_id"]: r.get("raw_log", "") for r in load_jsonl(path)}


def build_kb_index(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in load_jsonl(path):
        cid = r.get("chunk_id") or r.get("id") or ""
        if cid:
            out[cid] = r.get("content") or r.get("text") or ""
    return out


# =============================================================================
# Naive LLM mode (no RAG, single LLM call)
# =============================================================================

NAIVE_PROMPT_TEMPLATE = """You are a Linux OOM diagnosis expert.
Analyze the following OOM kernel log and output a diagnosis.

Output ONLY a JSON object. No explanations, no markdown fences.

The JSON MUST have this exact structure:
{{
  "classification": {{
    "oom_type": "global_oom | cgroup_oom | swap_exhaustion | page_alloc_failure"
  }},
  "final_answer": {{
    "log_analysis": {{
      "summary": "brief factual summary (3-5 sentences)"
    }},
    "diagnosis": {{
      "root_cause": "root cause (1-2 sentences)",
      "contributing_factors": ["factor 1", "factor 2"],
      "evidence": ["specific number or fact 1", "specific number or fact 2"],
      "severity": "high | medium | low"
    }},
    "action_guide": {{
      "immediate": ["immediate action 1"],
      "recommended": ["recommended action 1"],
      "further_investigation": ["item 1"]
    }}
  }}
}}

OOM log:
{raw_log}
"""


def _extract_balanced_json_objects(text: str) -> list[str]:
    """Find every balanced top-level {...} substring in `text`.

    Walks the text tracking brace depth (respecting string literals + escapes).
    Returns each top-level object as a raw substring.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
    return out


def run_naive(llm: Any, raw_log: str) -> tuple[dict, str]:
    """Returns (parsed_or_error_dict, raw_response_text).

    Many local models add a `Thinking Process:` preamble or output multiple
    JSON blocks. Naive-greedy `find("{") -> rfind("}")` then concatenates the
    intermediate prose into the slice, producing `Extra data` parse errors.
    Instead, extract every balanced top-level JSON object and pick the LAST
    parseable dict that has the expected schema keys.
    """
    prompt = NAIVE_PROMPT_TEMPLATE.format(raw_log=raw_log)
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    candidates = _extract_balanced_json_objects(text)
    if not candidates:
        return {"_error": "no_json_braces"}, text

    last_error = ""
    for raw in reversed(candidates):
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            last_error = f"json_decode_error: {exc}"
            continue
        if not isinstance(parsed, dict):
            continue
        # Prefer the answer object (contains `classification` and/or `final_answer`).
        if "classification" in parsed or "final_answer" in parsed:
            return parsed, text
    # No schema-matching object; fall back to the last well-formed dict
    for raw in reversed(candidates):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed, text
        except Exception:
            continue
    return {"_error": last_error or "no_dict_found"}, text


# =============================================================================
# Evaluation metric helpers (reuse style from legacy/exp2_generation.py)
# =============================================================================

_embeddings_cache: Any = None


def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = build_exp2_embeddings()
    return _embeddings_cache


def cosine_sim(v1, v2) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def normalize_oom_type(raw: Any) -> str:
    norm = embedding_normalize_oom_type(raw, get_embeddings())
    if norm:
        return norm
    if raw is None:
        return ""
    return str(raw).strip().lower()


def _find_first_key(obj: Any, target_key: str, predicate=None) -> Any:
    """BFS for the first dict-value at key `target_key` anywhere in `obj`.

    Optional predicate(value) filters candidates (e.g. require list of non-empty
    strings). Returns None if not found. Used to defend against the small-model
    schema-mis-nesting case where a 2B model puts `diagnosis` and `action_guide`
    inside `log_analysis` instead of as siblings of `log_analysis`.
    """
    from collections import deque
    queue = deque([obj])
    while queue:
        cur = queue.popleft()
        if isinstance(cur, dict):
            if target_key in cur:
                val = cur[target_key]
                if predicate is None or predicate(val):
                    return val
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    queue.append(v)
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    queue.append(v)
    return None


def extract_fields(generated: dict, mode: str) -> dict:
    if not isinstance(generated, dict):
        return {"oom_type": "", "evidence": [], "action_guide": []}

    classification = generated.get("classification", {}) or {}
    oom_type = ""
    if isinstance(classification, dict):
        oom_type = normalize_oom_type(classification.get("oom_type", "") or "")

    # Evidence: search recursively for the first non-empty list under any
    # `evidence` key. This covers both the canonical
    # `final_answer.diagnosis.evidence` path AND the small-model mis-nested
    # `final_answer.log_analysis.diagnosis.evidence` path.
    evidence = _find_first_key(
        generated,
        "evidence",
        predicate=lambda v: isinstance(v, list) and len(v) > 0,
    ) or []
    if not isinstance(evidence, list):
        evidence = []

    # Action guide: search for an action_guide dict with {immediate, recommended, ...}.
    ag = _find_first_key(
        generated,
        "action_guide",
        predicate=lambda v: isinstance(v, dict) and any(k in v for k in ("immediate", "recommended", "further_investigation")),
    )
    if not isinstance(ag, dict):
        ag = {}
    action: list[str] = []
    for key in ["immediate", "recommended", "further_investigation"]:
        items = ag.get(key, []) or []
        if isinstance(items, list):
            action.extend(str(x) for x in items if x)

    return {
        "oom_type": str(oom_type).strip().lower(),
        "evidence": [str(e) for e in evidence if e],
        "action_guide": action,
    }


def eval_category_match(expected: str, got: str) -> int:
    if not expected or not got:
        return 0
    return 1 if expected.strip().lower() == got.strip().lower() else 0


def eval_embed_recall(gt: list, gen: list) -> float:
    if not gt:
        return 1.0
    if not gen:
        return 0.0
    try:
        emb = get_embeddings()
        gt_v = [emb.embed_query(str(x)) for x in gt]
        gen_v = [emb.embed_query(str(x)) for x in gen]
    except Exception:
        return 0.0
    per_gt = [max((cosine_sim(g, p) for p in gen_v), default=0.0) for g in gt_v]
    return sum(per_gt) / len(per_gt)


# =============================================================================
# Faithfulness: bigram-overlap between generated evidence and retrieved chunks
# =============================================================================

WORD_RE = re.compile(r"[a-z0-9_/.:-]+", re.IGNORECASE)


def bigrams(text: str) -> set[tuple[str, str]]:
    toks = [t.lower() for t in WORD_RE.findall(text or "")]
    return {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}


def faithfulness_score(
    generated_evidence: list[str],
    retrieved_text: str,
) -> float:
    """Fraction of generated-evidence bigrams that appear in the retrieved corpus.

    1.0 = every bigram in evidence appears in retrieved chunks (grounded);
    0.0 = none (fully hallucinated relative to retrieval).
    """
    if not generated_evidence:
        return float("nan")
    retrieved_bg = bigrams(retrieved_text)
    if not retrieved_bg:
        return 0.0
    scores = []
    for ev in generated_evidence:
        ev_bg = bigrams(ev)
        if not ev_bg:
            continue
        scores.append(len(ev_bg & retrieved_bg) / len(ev_bg))
    return sum(scores) / len(scores) if scores else 0.0


def retrieved_text_from_state(rag_state: dict, kb_index: dict[str, str]) -> str:
    """Concatenate retrieved chunk texts from the RAG run state."""
    if not isinstance(rag_state, dict):
        return ""
    tool_results = rag_state.get("tool_results", {}) or {}
    kb = tool_results.get("kb_chunks", {}) or {}
    parts: list[str] = []
    if isinstance(kb, dict):
        chunks = kb.get("chunks") or kb.get("results") or []
        if isinstance(chunks, list):
            for c in chunks:
                if isinstance(c, dict):
                    txt = c.get("content") or c.get("text") or ""
                    if not txt:
                        cid = c.get("chunk_id") or c.get("id") or ""
                        if cid:
                            txt = kb_index.get(cid, "")
                    if txt:
                        parts.append(txt)
    return "\n".join(parts)


# =============================================================================
# Cost / VRAM estimation
# =============================================================================

class VramSampler:
    """Background thread that polls `nvidia-smi memory.used` on selected GPUs.

    Stores (timestamp, summed_used_GB) tuples so callers can compute the mean
    over any time window — used to attribute VRAM usage to individual queries
    via `mean_between(t_start, t_end)`. Silently no-ops when nvidia-smi is
    unavailable or fails (e.g. broken NVML inside a container) — callers get
    NaN for peak/mean in that case.
    """

    def __init__(self, gpu_indices: list[int] | None = None, interval_sec: float = 1.0):
        self._gpu_indices = gpu_indices if gpu_indices else [0]
        self._interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float]] = []  # (epoch_sec, used_GB)
        self._smi_path = shutil.which("nvidia-smi")
        self._failed = False

    def _query_once(self) -> float | None:
        if not self._smi_path or self._failed:
            return None
        try:
            cp = subprocess.run(
                [self._smi_path,
                 "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits",
                 f"--id={','.join(str(i) for i in self._gpu_indices)}"],
                capture_output=True, text=True, timeout=2,
            )
        except Exception:
            self._failed = True
            return None
        if cp.returncode != 0:
            self._failed = True
            return None
        try:
            total_mib = sum(int(line.strip()) for line in cp.stdout.strip().splitlines() if line.strip())
        except ValueError:
            self._failed = True
            return None
        return total_mib / 1024.0  # MiB -> GB (close enough; nvidia-smi reports MiB as 1024^2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self._query_once()
            if v is not None:
                self.samples.append((time.time(), v))
            self._stop.wait(self._interval_sec)

    def start(self) -> None:
        if not self._smi_path:
            return  # silent no-op when nvidia-smi missing
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vram-sampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def mean_between(self, t_start: float, t_end: float) -> float:
        """Mean VRAM (GB) over samples taken within [t_start, t_end].

        If no sample fell in the window (very short query relative to the
        sampling interval), fall back to the single sample nearest the
        window center so the per-query attribution is non-NaN whenever
        any sample exists.
        """
        if not self.samples:
            return float("nan")
        vals = [v for t, v in self.samples if t_start <= t <= t_end]
        if vals:
            return float(sum(vals) / len(vals))
        mid = 0.5 * (t_start + t_end)
        _, nearest = min(self.samples, key=lambda tv: abs(tv[0] - mid))
        return float(nearest)

    @property
    def peak_gb(self) -> float:
        return float(max(v for _, v in self.samples)) if self.samples else float("nan")

    @property
    def mean_gb(self) -> float:
        if not self.samples:
            return float("nan")
        vals = [v for _, v in self.samples]
        return float(sum(vals) / len(vals))


def _gpu_indices_for_profile(profile_env: dict[str, str] | None) -> list[int]:
    if not profile_env:
        return [0]
    raw = profile_env.get("CUDA_VISIBLE_DEVICES", "0")
    out: list[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out or [0]


def estimate_cost_usd(cond: Condition, inference_time_sec: float, input_tokens: int, output_tokens: int) -> float:
    if cond.provider == "openai":
        return (input_tokens / 1e6) * GPT_PRICE_INPUT_PER_M + (output_tokens / 1e6) * GPT_PRICE_OUTPUT_PER_M
    # Local: energy cost
    watts = cond.tp * GPU_TDP_W
    kwh = (watts * inference_time_sec) / (1000.0 * 3600.0)
    return kwh * KWH_PRICE_USD


# =============================================================================
# OpenAI factory (lazy)
# =============================================================================

def _openai_chat_factory(model_name: str, json_mode: bool = True):
    from langchain_community.chat_models.openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
        timeout=120,
        max_retries=2,
        model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
    )


# =============================================================================
# Per-condition runner
# =============================================================================

def run_one_log_rag(
    raw_log: str,
    cond: Condition,
    stats: CallStats,
) -> dict:
    """Wrap rag_runner.run_rag_agent with a tracked chat LLM factory.

    For OpenAI provider we pre-build the LLM and pass it via `llm=...` so
    rag_runner skips its internal `ensure_vllm_model` call (which would try
    to load "gpt-5.2" as a vLLM profile and fail).
    """

    def chat_factory(model_name: str):
        return build_tracked_chat_llm(model_name, stats, cond.provider, json_mode=True)

    common_kwargs: dict[str, Any] = dict(
        label_similarity_threshold=DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
        label_similarity_margin=DEFAULT_EMBEDDING_SIMILARITY_MARGIN,
        create_state_fn=create_initial_state,
        embeddings_factory=get_embeddings,
        node_1=node_1_parser,
        node_2=node_2_classifier,
        node_3=node_3_executor,
        node_4=node_4_synthesizer,
    )

    if cond.provider == "openai":
        tracked = build_tracked_chat_llm(cond.model, stats, "openai", json_mode=True)
        return run_ragstar_agent(raw_log, llm=tracked, **common_kwargs)

    return run_ragstar_agent(
        raw_log,
        model_name=cond.model,
        llm_name=cond.model,
        chat_llm_factory=chat_factory,
        **common_kwargs,
    )


def run_condition(
    cond: Condition,
    dataset: list[dict],
    oom_logs: dict[str, str],
    kb_index: dict[str, str],
    logger: TeeLogger,
) -> list[dict]:
    logger.log(f"=== Condition: {cond.label} (model={cond.model}, mode={cond.mode}) ===")

    # vLLM: pre-switch model
    profile_env: dict[str, str] | None = None
    if cond.provider == "vllm":
        t0 = time.time()
        served = ensure_vllm_model(cond.model)
        logger.log(f"vLLM ready: served_name={served} (switch={time.time()-t0:.1f}s)")
        # Read profile env for selecting which GPUs to sample
        with open(get_settings().vllm_model_profiles, "r", encoding="utf-8") as f:
            profile_env = json.load(f).get(cond.model, {}).get("env", {})
    elif cond.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            logger.log(f"SKIP {cond.label}: OPENAI_API_KEY not set")
            return []

    # Real VRAM sampling via nvidia-smi (vllm only; openai records NaN).
    # 0.5s interval so even short (~1-2s) queries get multiple samples for
    # a meaningful per-query mean.
    vram_sampler: VramSampler | None = None
    if cond.provider == "vllm":
        vram_sampler = VramSampler(_gpu_indices_for_profile(profile_env), interval_sec=0.5)
        vram_sampler.start()
        if not vram_sampler._smi_path:
            logger.log("VRAM sampler: nvidia-smi not found; per-query VRAM will be NaN")

    # OpenAI free/low tiers have very tight RPM (e.g. 3). Sleep keeps us under.
    openai_min_interval_sec = 25 if cond.provider == "openai" else 0

    rows: list[dict] = []
    n = len(dataset)
    last_call_ts = 0.0
    for i, qa in enumerate(dataset):
        log_id = qa.get("log_id", f"log_{i}")
        expected_oom = qa.get("expected_oom_type", "")
        gt = qa.get("ground_truth", {}) or {}
        gt_evidence = gt.get("must_include_evidence", []) or []
        gt_action = gt.get("action_guide", []) or []

        raw_log = oom_logs.get(log_id, "")
        if not raw_log:
            logger.log(f"  [{i+1:02d}/{n}] {log_id}: raw_log missing, skip")
            continue

        # Throttle for low-RPM OpenAI tiers
        if openai_min_interval_sec > 0 and last_call_ts > 0:
            elapsed_since_last = time.time() - last_call_ts
            if elapsed_since_last < openai_min_interval_sec:
                wait = openai_min_interval_sec - elapsed_since_last
                time.sleep(wait)
        last_call_ts = time.time()

        stats = CallStats()
        t_start = time.time()
        error_msg = ""
        rag_state: dict[str, Any] = {}
        try:
            if cond.mode == "rag":
                rag_state = run_one_log_rag(raw_log, cond, stats)
                generated = rag_state if isinstance(rag_state, dict) else {}
                if "_error" in generated:
                    error_msg = str(generated.get("_error") or "")
            else:
                # Naive: build chat LLM with attached TokenCallback
                llm = build_tracked_chat_llm(cond.model, stats, cond.provider, json_mode=False)
                generated, raw_text = run_naive(llm, raw_log)
                if "_error" in generated:
                    error_msg = str(generated["_error"])
        except Exception as exc:
            generated = {"_error": f"unhandled: {exc}"}
            error_msg = generated["_error"]
        t_end = time.time()
        elapsed = t_end - t_start

        # Per-query VRAM: mean of nvidia-smi samples within this query's window
        per_query_vram = (
            vram_sampler.mean_between(t_start, t_end) if vram_sampler is not None else float("nan")
        )

        fields = extract_fields(generated, cond.mode)
        cat = eval_category_match(expected_oom, fields["oom_type"])
        ev = eval_embed_recall(gt_evidence, fields["evidence"])
        ag = eval_embed_recall(gt_action, fields["action_guide"])

        # JSON validity from the final raw response
        first_try = bool(stats.first_try_valid and stats.first_try_valid[-1])
        repaired = bool(
            stats.raw_responses
            and not first_try
            and needed_repair_but_parsed(stats.raw_responses[-1])
        )

        # Faithfulness (RAG only)
        faith = float("nan")
        if cond.mode == "rag":
            ret_text = retrieved_text_from_state(rag_state, kb_index)
            if ret_text and fields["evidence"]:
                faith = faithfulness_score(fields["evidence"], ret_text)

        cost = estimate_cost_usd(cond, elapsed, stats.input_tokens, stats.output_tokens)

        row = {
            "model": cond.model,
            "mode": cond.mode,
            "family": cond.family,
            "label": cond.label,
            "log_id": log_id,
            "expected_oom_type": expected_oom,
            "generated_oom_type": fields["oom_type"],
            "category_match": cat,
            "evidence_recall": round(ev, 4),
            "action_guide_similarity": round(ag, 4),
            "inference_time_sec": round(elapsed, 3),
            "n_llm_calls": stats.n_calls,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "total_tokens": stats.input_tokens + stats.output_tokens,
            "json_first_try_valid": int(first_try),
            "json_needed_repair": int(repaired),
            "had_error": int(bool(error_msg)),
            "error_msg": error_msg[:200],
            "faithfulness": round(faith, 4) if not np.isnan(faith) else float("nan"),
            "cost_usd_est": round(cost, 6),
            "vram_gb_per_query": (
                round(per_query_vram, 2) if not np.isnan(per_query_vram) else float("nan")
            ),
        }
        rows.append(row)
        vram_str = (
            f"vram={per_query_vram:.1f}GB" if not np.isnan(per_query_vram) else "vram=NaN"
        )
        logger.log(
            f"  [{i+1:02d}/{n}] {log_id} cat={cat} ev={ev:.2f} ag={ag:.2f} "
            f"t={elapsed:.2f}s {vram_str} calls={stats.n_calls} "
            f"tok={stats.input_tokens}/{stats.output_tokens} "
            f"err={'Y' if error_msg else 'N'}"
        )

    # Stop VRAM sampler + back-fill condition-wide peak into every row.
    # (The condition-wide MEAN would be polluted by idle gaps between queries;
    # per-query mean is recorded per row as `vram_gb_per_query` instead.)
    vram_peak = float("nan")
    if vram_sampler is not None:
        vram_sampler.stop()
        vram_peak = vram_sampler.peak_gb
        n_samples = len(vram_sampler.samples)
        per_query_vals = [r["vram_gb_per_query"] for r in rows if not (isinstance(r["vram_gb_per_query"], float) and np.isnan(r["vram_gb_per_query"]))]
        per_query_mean = (sum(per_query_vals) / len(per_query_vals)) if per_query_vals else float("nan")
        per_query_mean_str = (
            f"{per_query_mean:.2f}GB" if not np.isnan(per_query_mean) else "NaN"
        )
        logger.log(
            f"VRAM measured: peak={vram_peak:.2f}GB per_query_mean={per_query_mean_str} "
            f"(n_samples={n_samples}, n_queries={len(per_query_vals)})"
        )
    for r in rows:
        r["vram_gb_peak"] = round(vram_peak, 2) if not np.isnan(vram_peak) else float("nan")

    return rows


# =============================================================================
# Aggregation
# =============================================================================

def aggregate(per_log_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(per_log_rows)
    if df.empty:
        return df
    agg_rows = []
    for (model, mode), sub in df.groupby(["model", "mode"]):
        times = sub["inference_time_sec"].astype(float).tolist()
        # Measured average wall-clock per query for this (model, mode).
        sec_per_query = statistics.mean(times) if times else 0.0
        vram_col = sub["vram_gb_per_query"] if "vram_gb_per_query" in sub.columns else pd.Series(dtype=float)
        agg_rows.append({
            "model": model,
            "mode": mode,
            "family": sub["family"].iloc[0],
            "label": sub["label"].iloc[0],
            "n": len(sub),
            "category_match": sub["category_match"].mean(),
            "evidence_recall": sub["evidence_recall"].mean(),
            "action_guide_similarity": sub["action_guide_similarity"].mean(),
            "faithfulness_mean": sub["faithfulness"].mean(skipna=True),
            "json_first_try_rate": sub["json_first_try_valid"].mean(),
            "json_repair_rate": sub["json_needed_repair"].mean(),
            "error_rate": sub["had_error"].mean(),
            "sec_per_query": sec_per_query,
            "lat_p50": np.percentile(times, 50) if times else 0.0,
            "lat_p95": np.percentile(times, 95) if times else 0.0,
            "avg_input_tokens": sub["input_tokens"].mean(),
            "avg_output_tokens": sub["output_tokens"].mean(),
            "avg_cost_usd": sub["cost_usd_est"].mean(),
            "vram_gb_per_query_mean": vram_col.mean(skipna=True) if not vram_col.empty else float("nan"),
            "vram_gb_peak": sub["vram_gb_peak"].iloc[0] if "vram_gb_peak" in sub.columns else float("nan"),
        })
    return pd.DataFrame(agg_rows).round(4)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    global KWH_PRICE_USD
    parser = argparse.ArgumentParser(description="Model scaling experiment (RAG + Naive)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Filter conditions by slug substring (e.g. --only qwen3.5-9b gemma).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of dataset logs (debug).")
    parser.add_argument("--kwh-price", type=float, default=KWH_PRICE_USD,
                        help="Electricity price USD/kWh for local cost estimate.")
    parser.add_argument("--per-log-csv", type=str,
                        default=str(RESULT_DIR / "scaling_per_log.csv"))
    parser.add_argument("--per-cond-csv", type=str,
                        default=str(RESULT_DIR / "scaling_per_condition.csv"))
    parser.add_argument("--resume", action="store_true",
                        help="Load existing per-log CSV; skip (model, mode) already present.")
    args = parser.parse_args()
    KWH_PRICE_USD = args.kwh_price

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TeeLogger(LOG_DIR / f"exp2_generation_{ts}.log")
    logger.log(f"Project root: {PROJECT_ROOT}")
    logger.log(f"Results dir : {RESULT_DIR}")

    try:
        # Load data
        dataset = load_jsonl(GT_PATH)
        oom_logs = build_oom_logs_index(OOM_LOGS_PATH)
        kb_index = build_kb_index(KB_CHUNKS_PATH)
        if args.limit:
            dataset = dataset[: args.limit]
        logger.log(f"Dataset: {len(dataset)} logs, {len(kb_index)} KB chunks")

        # Conditions
        conditions = build_conditions()
        if args.only:
            keep = []
            for c in conditions:
                hay = " ".join([c.model, c.mode, c.label, c.slug, c.family]).lower()
                if any(token.lower() in hay for token in args.only):
                    keep.append(c)
            conditions = keep
        logger.log(f"Conditions ({len(conditions)}): " + ", ".join(c.label for c in conditions))

        all_rows: list[dict] = []
        per_log_path = Path(args.per_log_csv)
        per_cond_path = Path(args.per_cond_csv)

        # --resume: load prior rows; skip conditions only when they already cover the current dataset size.
        if args.resume and per_log_path.exists():
            try:
                prior = pd.read_csv(per_log_path)
                condition_keys = {(c.model, c.mode) for c in conditions}
                prior = prior[prior.apply(lambda r: (r.get("model"), r.get("mode")) in condition_keys, axis=1)]
                if not prior.empty:
                    all_rows = prior.to_dict("records")
                    done_keys = {
                        (m, md) for (m, md), sub in prior.groupby(["model", "mode"])
                        if len(sub) >= len(dataset)
                    }
                    keep = [c for c in conditions if (c.model, c.mode) not in done_keys]
                    skipped = [c for c in conditions if (c.model, c.mode) in done_keys]
                    for c in skipped:
                        logger.log(f"RESUME skip (already done): {c.label} ({len(prior[(prior['model']==c.model)&(prior['mode']==c.mode)])} rows)")
                    conditions = keep
                    logger.log(f"RESUME loaded {len(all_rows)} prior rows; {len(conditions)} conditions left")
            except Exception as exc:
                logger.log(f"RESUME: could not load prior CSV: {exc}")

        for ci, cond in enumerate(conditions, 1):
            logger.log(f"\n>>> Condition {ci}/{len(conditions)}: {cond.label}")
            t_cond_start = time.time()
            try:
                rows = run_condition(cond, dataset, oom_logs, kb_index, logger)
            except Exception as exc:
                logger.log(f"!!! Condition failed: {cond.label}: {exc}")
                continue
            all_rows.extend(rows)
            # Incrementally flush results so a crash mid-run doesn't lose data
            pd.DataFrame(all_rows).to_csv(per_log_path, index=False, encoding="utf-8-sig")
            agg = aggregate(all_rows)
            if not agg.empty:
                agg.to_csv(per_cond_path, index=False, encoding="utf-8-sig")
            logger.log(f"<<< Condition done in {time.time()-t_cond_start:.1f}s, total rows={len(all_rows)}")

        logger.log("\n=== Final aggregate ===")
        final_agg = aggregate(all_rows)
        if not final_agg.empty:
            for _, r in final_agg.iterrows():
                vram_pq = r.get("vram_gb_per_query_mean", float("nan"))
                vram_str = f"vram={vram_pq:.1f}GB" if not (isinstance(vram_pq, float) and np.isnan(vram_pq)) else "vram=NaN"
                logger.log(
                    f"  {r['label']:30s} n={int(r['n']):3d} cat={r['category_match']:.3f} "
                    f"ev={r['evidence_recall']:.3f} ag={r['action_guide_similarity']:.3f} "
                    f"t/q={r['sec_per_query']:.2f}s {vram_str} err={r['error_rate']:.2f} "
                    f"json1st={r['json_first_try_rate']:.2f} cost=${r['avg_cost_usd']:.4f}"
                )
        logger.log(f"per-log csv : {per_log_path}")
        logger.log(f"per-cond csv: {per_cond_path}")
        logger.log(f"log file    : {logger.path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
