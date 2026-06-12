#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Supported profiles:
#   MODEL_PROFILE=qwen3.5 ./start.sh
#   MODEL_PROFILE=gemma4 MODEL_DIR=/workspace/vllm/models/... ./start.sh
MODEL_PROFILE="${MODEL_PROFILE:-qwen3.5}"

case "${MODEL_PROFILE}" in
  qwen3.5 | qwen3_5 | qwen)
    MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/Qwen/Qwen3.5-2B}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-2b}"
    ;;
  gemma4 | gemma)
    MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/Google/Gemma4}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma4}"
    ;;
  *)
    MODEL_DIR="${MODEL_DIR:-${MODEL_PROFILE}}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_DIR}")}"
    ;;
esac

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Lightweight default: serve the local Qwen 2B profile on one GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# Context limit.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# Aggressive-but-not-insane scheduling/capture settings for 2x24GB A5000.
# If startup OOMs during CUDA graph capture, lower MAX_CUDAGRAPH_CAPTURE_SIZE
# to 8192 or 4096 first.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-16384}"
DTYPE="${DTYPE:-auto}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory does not exist: ${MODEL_DIR}" >&2
  echo "Set MODEL_DIR=/path/to/model or use MODEL_PROFILE=qwen3.5 for the local Qwen model." >&2
  exit 1
fi

# Latest vLLM compilation/CUDA graph path:
# - mode=3: vLLM compile path
# - FULL_AND_PIECEWISE: full cudagraph for decode, piecewise for prefill/mixed
# - max_cudagraph_capture_size=16384: auto-generates dense capture sizes:
#   [1,2,4], multiples of 8 up to 256, then multiples of 16 up to 16384.
COMPILATION_CONFIG="$(python3 - <<PY
import json
max_capture_size = int("${MAX_CUDAGRAPH_CAPTURE_SIZE}")
candidate_compile_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
compile_sizes = [size for size in candidate_compile_sizes if size <= max_capture_size]
if max_capture_size not in compile_sizes:
    compile_sizes.append(max_capture_size)

cfg = {
    "mode": 3,
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "max_cudagraph_capture_size": max_capture_size,
    "compile_sizes": compile_sizes,
}
print(json.dumps(cfg, separators=(",", ":")))
PY
)"

VLLM_ENV_DIR="${VLLM_ENV_DIR:-${SCRIPT_DIR}/.venv-vllm}"
if [[ -n "${VLLM_BIN:-}" ]]; then
  VLLM_CMD=("${VLLM_BIN}")
elif command -v uv >/dev/null 2>&1; then
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${VLLM_ENV_DIR}}"
  VLLM_CMD=(uv run vllm)
elif [[ -x "${VLLM_ENV_DIR}/bin/vllm" ]]; then
  VLLM_CMD=("${VLLM_ENV_DIR}/bin/vllm")
elif command -v vllm >/dev/null 2>&1; then
  VLLM_CMD=(vllm)
else
  echo "Could not find vLLM." >&2
  echo "Install it with uv or create ${VLLM_ENV_DIR}, then rerun this script." >&2
  exit 1
fi

echo "Serving ${SERVED_MODEL_NAME} from ${MODEL_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "true" ]]; then
  echo "ENFORCE_EAGER=${ENFORCE_EAGER}; skipping torch.compile/cudagraph capture for faster startup"
  EAGER_ARGS=(--enforce-eager)
  COMPILATION_ARGS=()
else
  EAGER_ARGS=()
  COMPILATION_ARGS=(--compilation-config "${COMPILATION_CONFIG}")
fi

"${VLLM_CMD[@]}" serve "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype "${DTYPE}" \
  "${EAGER_ARGS[@]}" \
  "${COMPILATION_ARGS[@]}"
