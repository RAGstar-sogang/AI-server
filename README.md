# AI-server Workspace

이 저장소는 RAGSTAR AI 서버 운영에 필요한 backend 코드와 vLLM serving 구성을 한곳에서 관리하는 workspace repo입니다.

이전에는 `ragstar-backend`가 별도 Git repo처럼 관리됐지만, 현재는 `AI-server` repo가 workspace 전체를 직접 추적합니다. GitHub에서 `ragstar-backend/`와 `vllm/`이 일반 폴더로 보이는 것이 정상입니다.

## 디렉터리 구성

| 경로 | 설명 |
| --- | --- |
| `ragstar-backend/` | RAGSTAR backend 코드, agent graph, worker, tests, docs, experiments |
| `vllm/` | vLLM OpenAI-compatible serving workspace |
| `ollama-serving/` | Ollama 관련 로컬 디렉터리 |
| `start_worker.sh` | backend worker를 vLLM과 함께 tmux로 띄우는 운영 스크립트 |
| `.gitignore` | workspace 전체 ignore 규칙 |
| `.python-version` | 기본 Python version hint |

## 절대 커밋하지 않는 것

workspace root의 `.gitignore`가 아래 항목들을 막습니다.

- Python 가상환경: `.venv/`, `.venv-*`, `venv/`, `env/`
- vLLM 실행 환경: `vllm/.venv-vllm/`
- 모델 weight: `vllm/models/`, `**/models/`, `*.safetensors`, `*.bin`, `*.pt`, `*.gguf`
- 로컬 env 파일: `.env`, `.env.*`, `*.env`, `vllm/serve.env`, `ragstar-backend/.env`
- 런타임 로그: `logs/`, `*.log`, `ollama.log`
- backend 로컬 데이터: `ragstar-backend/data/`, `ragstar-backend/chroma_db/`
- migration snapshot: `_migration_meta/`
- cache/build 결과물: `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `build/`, `dist/`

커밋 전에 항상 아래를 확인하세요.

```bash
git status --ignored
```

`vllm/models/`, `vllm/.venv-vllm/`, `vllm/serve.env`, `ragstar-backend/.env`, `ragstar-backend/data/`, `ragstar-backend/chroma_db/`가 staged에 올라오면 안 됩니다.

## vLLM 운영

vLLM 운영 문서는 아래 파일을 기준으로 봅니다.

```text
vllm/README.md
```

핵심은 `vllm/serve.env`만 바꾸고 `manage.sh`로 재시작하는 구조입니다.

```bash
cd /workspace/vllm
./manage.sh status
./manage.sh restart
./manage.sh logs
```

`vllm/serve.env`는 로컬 운영 설정이라 git에 올리지 않습니다. 공유 가능한 템플릿은 `vllm/serve.env.example`입니다.

## Backend

Backend 코드는 `ragstar-backend/` 아래에서 관리합니다.

```bash
cd /workspace/ragstar-backend
```

현재 workspace repo가 backend 파일들을 직접 추적하므로 `ragstar-backend/.git`은 없어야 정상입니다. 예전 backend Git metadata는 legacy로 취급합니다.

주요 위치:

- `ragstar-backend/app/agent/`
- `ragstar-backend/app/core/`
- `ragstar-backend/app/database/`
- `ragstar-backend/app/network/`
- `ragstar-backend/tests/`
- `ragstar-backend/docs/`
- `ragstar-backend/experiments/`

## Worker 운영 (`start_worker.sh`)

`start_worker.sh`는 backend worker(`app.worker`)를 **vLLM 의존성 확인 + tmux 세션 분리** 두 가지를 한 번에 처리해서 띄우는 운영 스크립트입니다. SSH 연결이 끊겨도 worker가 살아남도록 `nohup` 대신 tmux를 씁니다.

### 동작 요약

1. `vllm/manage.sh start`를 먼저 호출해 vLLM이 떠 있는지 확인 (이미 실행 중이면 no-op).
2. 지정된 tmux 세션(기본: `ragstar-worker`)이 없으면 새로 만들고 그 안에서 `python -m app.worker` 실행.
3. stdout/stderr를 로그 파일(`ragstar-backend/logs/worker.log`)에 `tee`로 append.
4. 이미 같은 세션이 있으면 새로 띄우지 않고 status만 출력.

worker는 backend API를 polling하면서 들어오는 OOM 이벤트를 RAGSTAR agent graph로 처리하는 데몬입니다 ([ragstar-backend/app/worker.py](ragstar-backend/app/worker.py)).

### 사용법

```bash
cd /workspace
./start_worker.sh start      # vLLM 보장 + worker 기동 (기본값)
./start_worker.sh status     # vLLM + worker tmux 상태
./start_worker.sh logs       # worker.log를 tail -f
./start_worker.sh attach     # tmux 세션에 직접 접속 (이탈: Ctrl+B then D)
./start_worker.sh stop       # Ctrl+C 전송 후 세션 종료
./start_worker.sh restart    # stop → start
```

인자를 안 주고 `./start_worker.sh`만 실행해도 `start`가 됩니다.

### 환경 변수로 덮어쓰기

기본값을 바꾸고 싶을 때 export로 전달합니다.

| 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `RAGSTAR_BACKEND_DIR` | `<workspace>/ragstar-backend` | backend 코드 경로 |
| `RAGSTAR_VLLM_DIR` | `<workspace>/vllm` | vLLM workspace 경로 |
| `VLLM_MANAGE` | `<vllm>/manage.sh` | vLLM 관리 스크립트 경로 |
| `RAGSTAR_WORKER_SESSION_NAME` | `ragstar-worker` | tmux 세션 이름 |
| `RAGSTAR_WORKER_PYTHON` | `<workspace>/.venv/bin/python` | worker가 쓸 python 인터프리터 |
| `RAGSTAR_WORKER_LOG_DIR` | `<backend>/logs` | 로그 디렉터리 |
| `RAGSTAR_WORKER_LOG_FILE` | `<log_dir>/worker.log` | 로그 파일 경로 |

예) 세션 이름과 python 경로를 바꿔서 띄우기:

```bash
RAGSTAR_WORKER_SESSION_NAME=worker-staging \
RAGSTAR_WORKER_PYTHON=/opt/conda/envs/ragstar/bin/python \
./start_worker.sh start
```

### 기동 실패 진단 포인트

- `Python executable not found or not executable` → `.venv`가 아직 만들어지지 않았거나 실행 권한 없음. `python -m venv .venv && .venv/bin/pip install -r ragstar-backend/requirements.txt` 후 재시도.
- `vLLM manage script is not executable` → `chmod +x vllm/manage.sh`.
- `Worker tmux session already running` 메시지 후 정상 동작 안 함 → `./start_worker.sh logs`로 실제 worker 로그를 보고, 필요시 `./start_worker.sh restart`.
- worker가 OOM 이벤트를 못 잡음 → backend API 주소/토큰이 `ragstar-backend/.env`에 맞게 들어있는지 확인.

### vLLM 모델 전환

worker는 요청에 따라 `app.core.vllm_manager.ensure_vllm_model(model_name)`을 호출해 vLLM의 서빙 모델을 자동 교체합니다. 운영 중에 모델을 강제로 바꿔야 할 때만 `vllm/serve.env`를 직접 수정한 뒤 `cd /workspace/vllm && ./manage.sh restart`를 쓰면 됩니다.

### Eager vs Compile 모드 자동 전환

`start_worker.sh`는 `RAGSTAR_VLLM_ENFORCE_EAGER=0`을 export해서 worker가 띄우는 vLLM은 **compile / CUDA-graph 모드**로 돌아갑니다. warmup 후 지속 처리량이 좋아져서 production 서빙에 유리합니다.

실험 스크립트(`experiments/exp2_generation.py` 등)는 같은 변수를 `"1"`로 세팅해서 **eager 모드**로 돌립니다. 모델을 자주 갈아끼는 sweep 시나리오에서 시작 시간 단축 + capture 단계 OOM 회피가 더 중요하기 때문입니다.

`app.core.vllm_manager.ensure_vllm_model()`이 이 변수를 읽어 model profile의 `ENFORCE_EAGER`를 덮어쓰고, 현재 떠 있는 vLLM의 모드가 요청한 것과 다르면 **같은 모델이어도 강제 재시작**합니다. 따라서 worker → 실험 또는 그 반대 순서로 실행하면 자동으로 적절한 모드로 전환됩니다.

## Git 작업 흐름

workspace root에서 작업합니다.

```bash
cd /workspace
git status
git add .
git status
git commit -m "Describe change"
git push
```

`ragstar-backend/` 안에서 별도 git 명령을 실행하는 방식이 아니라, `/workspace`에서 한 번에 관리합니다.

## 새 환경에서 복구할 것

이 repo에는 코드와 운영 스크립트만 들어갑니다. 새 머신에서는 아래 로컬 리소스를 별도로 준비해야 합니다.

- Python 가상환경
- backend `.env`
- vLLM `serve.env`
- vLLM model weights
- Chroma DB 또는 runtime data

vLLM 설정은 다음처럼 시작합니다.

```bash
cd /workspace/vllm
cp serve.env.example serve.env
```

그 뒤 모델 경로와 GPU 설정을 환경에 맞게 수정합니다.

## Push 확인

GitHub에 정상적으로 올라가면 root에는 대략 아래처럼 보입니다.

```text
AI-server/
├── README.md
├── .gitignore
├── ragstar-backend/
├── vllm/
└── ollama-serving/
```

`ragstar-backend/`가 화살표 아이콘이나 submodule처럼 보이면 잘못 올라간 것입니다. 일반 폴더 아이콘으로 보여야 합니다.
