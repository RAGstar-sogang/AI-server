#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${VLLM_SESSION_NAME:-vllm-serving}"
SERVE_ENV_FILE="${VLLM_SERVE_ENV_FILE:-${SCRIPT_DIR}/serve.env}"
LOG_DIR="${VLLM_LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${VLLM_LOG_FILE:-${LOG_DIR}/vllm.log}"

if [[ -f "${SERVE_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SERVE_ENV_FILE}"
  set +a
fi

mkdir -p "${LOG_DIR}"

tmux_has_session() {
  tmux has-session -t "${SESSION_NAME}" 2>/dev/null
}

is_port_open() {
  python3 - <<PY
import socket
host = "127.0.0.1"
port = int("${PORT:-8000}")
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    sock.close()
PY
}

start_server() {
  if tmux_has_session; then
    echo "vLLM tmux session already running: ${SESSION_NAME}"
    status_server
    return
  fi

  touch "${LOG_FILE}"
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${SCRIPT_DIR}' && set -a && source '${SERVE_ENV_FILE}' && set +a && bash start.sh 2>&1 | tee -a '${LOG_FILE}'"

  echo "Started vLLM in tmux session: ${SESSION_NAME}"
  echo "Log: ${LOG_FILE}"
  echo "Attach: tmux attach -t ${SESSION_NAME}"
}

stop_server() {
  if tmux_has_session; then
    tmux send-keys -t "${SESSION_NAME}" C-c
    sleep 2
    if tmux_has_session; then
      tmux kill-session -t "${SESSION_NAME}"
    fi
    echo "Stopped vLLM session: ${SESSION_NAME}"
  else
    echo "vLLM tmux session is not running: ${SESSION_NAME}"
  fi
}

status_server() {
  if tmux_has_session; then
    echo "tmux: running (${SESSION_NAME})"
  else
    echo "tmux: stopped (${SESSION_NAME})"
  fi

  if is_port_open; then
    echo "api: listening on 127.0.0.1:${PORT:-8000}"
  else
    echo "api: not listening on 127.0.0.1:${PORT:-8000}"
  fi
}

case "${1:-status}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    status_server
    ;;
  logs)
    tail -f "${LOG_FILE}"
    ;;
  attach)
    tmux attach -t "${SESSION_NAME}"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|attach}" >&2
    exit 2
    ;;
esac
