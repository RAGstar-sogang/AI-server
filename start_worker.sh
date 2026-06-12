#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${RAGSTAR_BACKEND_DIR:-${ROOT_DIR}/ragstar-backend}"
VLLM_DIR="${RAGSTAR_VLLM_DIR:-${ROOT_DIR}/vllm}"
VLLM_MANAGE="${VLLM_MANAGE:-${VLLM_DIR}/manage.sh}"

SESSION_NAME="${RAGSTAR_WORKER_SESSION_NAME:-ragstar-worker}"
PYTHON_BIN="${RAGSTAR_WORKER_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
LOG_DIR="${RAGSTAR_WORKER_LOG_DIR:-${BACKEND_DIR}/logs}"
LOG_FILE="${RAGSTAR_WORKER_LOG_FILE:-${LOG_DIR}/worker.log}"

# Steady-state serving: prefer compile/CUDA-graph mode (better sustained
# throughput after warmup). vllm_manager.ensure_vllm_model reads this and
# overrides the model profile's ENFORCE_EAGER. Experiment scripts set "1"
# to favor fast startup + capture-OOM safety for frequent model swaps.
export RAGSTAR_VLLM_ENFORCE_EAGER="${RAGSTAR_VLLM_ENFORCE_EAGER:-0}"

usage() {
  echo "Usage: $0 {start|stop|restart|status|logs|attach}"
}

tmux_has_session() {
  tmux has-session -t "${SESSION_NAME}" 2>/dev/null
}

ensure_vllm() {
  if [[ ! -x "${VLLM_MANAGE}" ]]; then
    echo "vLLM manage script is not executable: ${VLLM_MANAGE}" >&2
    exit 1
  fi

  echo "Ensuring vLLM is running..."
  "${VLLM_MANAGE}" start
}

start_worker() {
  ensure_vllm

  if tmux_has_session; then
    echo "Worker tmux session already running: ${SESSION_NAME}"
    status_worker
    return
  fi

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi

  mkdir -p "${LOG_DIR}"
  touch "${LOG_FILE}"

  tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${BACKEND_DIR}' && '${PYTHON_BIN}' -m app.worker 2>&1 | tee -a '${LOG_FILE}'"

  echo "Started worker in tmux session: ${SESSION_NAME}"
  echo "Log: ${LOG_FILE}"
  echo "Attach: tmux attach -t ${SESSION_NAME}"
}

stop_worker() {
  if tmux_has_session; then
    tmux send-keys -t "${SESSION_NAME}" C-c
    sleep 2
    if tmux_has_session; then
      tmux kill-session -t "${SESSION_NAME}"
    fi
    echo "Stopped worker session: ${SESSION_NAME}"
  else
    echo "Worker tmux session is not running: ${SESSION_NAME}"
  fi
}

status_worker() {
  echo "vLLM:"
  "${VLLM_MANAGE}" status || true

  echo
  echo "Worker:"
  if tmux_has_session; then
    echo "tmux: running (${SESSION_NAME})"
  else
    echo "tmux: stopped (${SESSION_NAME})"
  fi
  echo "log: ${LOG_FILE}"
}

case "${1:-start}" in
  start)
    start_worker
    ;;
  stop)
    stop_worker
    ;;
  restart)
    stop_worker
    start_worker
    ;;
  status)
    status_worker
    ;;
  logs)
    tail -f "${LOG_FILE}"
    ;;
  attach)
    tmux attach -t "${SESSION_NAME}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
