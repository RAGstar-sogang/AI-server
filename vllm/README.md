# vLLM Serving Workspace

이 디렉터리는 RAGSTAR에서 로컬 vLLM 서버를 띄우기 위한 운영용 워크스페이스입니다.

핵심은 단순합니다.

**평소에는 `serve.env`만 수정하면 됩니다.**  
`manage.sh`가 `serve.env`를 자동으로 읽고, 그 값을 환경변수로 export한 뒤 `start.sh`를 실행합니다. 모델 변경, 포트 변경, GPU 개수 변경, context length 변경, batching 튜닝, eager/compile 모드 변경은 거의 전부 `serve.env`에서 처리합니다.

`start.sh`는 실행 로직이고, `serve.env`는 운영 설정입니다. 특별한 이유가 없다면 `start.sh`를 직접 수정하지 말고 `serve.env`를 바꾼 뒤 재시작하세요.

## 파일 구성

| 파일 | 설명 |
| --- | --- |
| `serve.env` | 실제 로컬 실행 설정 파일입니다. git에 올리지 않습니다. |
| `serve.env.example` | 공유용 설정 템플릿입니다. 새 환경에서는 이 파일을 복사해서 `serve.env`를 만듭니다. |
| `manage.sh` | tmux 기반 vLLM 관리 스크립트입니다. start, stop, restart, status, logs, attach를 제공합니다. |
| `start.sh` | 실제 `vllm serve`를 실행하는 스크립트입니다. env 값을 읽어 실행 옵션을 구성합니다. |
| `pyproject.toml` | `uv`로 관리하는 vLLM Python 의존성 정의입니다. |
| `uv.lock` | 고정된 의존성 lock 파일입니다. 재현 가능한 설치를 위해 유지합니다. |
| `model_profiles.json` | 현재 로컬 모델별 권장 설정 참고 파일입니다. 자동 적용되지는 않습니다. |
| `models/` | 로컬 모델 weight 디렉터리입니다. 용량이 크므로 git에 올리지 않습니다. |
| `.venv-vllm/` | vLLM 실행용 Python 가상환경입니다. 용량이 크므로 git에 올리지 않습니다. |
| `logs/` | vLLM 실행 로그 디렉터리입니다. git에 올리지 않습니다. |

## 빠른 실행

```bash
cd /workspace/vllm
cp serve.env.example serve.env
./manage.sh start
./manage.sh status
```

이미 서버가 떠 있고 `serve.env`만 바꾼 경우:

```bash
./manage.sh restart
```

로그 확인:

```bash
./manage.sh logs
```

tmux 세션에 직접 붙기:

```bash
./manage.sh attach
```

서버 중지:

```bash
./manage.sh stop
```

## 실행 흐름

일반적으로는 `./manage.sh start` 또는 `./manage.sh restart`만 사용합니다.

실행 흐름은 다음과 같습니다.

1. `manage.sh`가 `serve.env`를 찾습니다.
2. `serve.env`가 있으면 `source`해서 설정값을 전부 환경변수로 올립니다.
3. `manage.sh`가 `logs/` 디렉터리를 준비합니다.
4. `vllm-serving`이라는 tmux 세션을 띄웁니다.
5. tmux 안에서 `bash start.sh`를 실행합니다.
6. `start.sh`가 `MODEL_DIR`, `SERVED_MODEL_NAME`, GPU, batch, dtype 같은 값을 읽습니다.
7. 마지막으로 `vllm serve ...`가 실행됩니다.

즉, 운영자가 바꿔야 하는 값은 거의 전부 `serve.env`에 있습니다.

## 현재 로컬 모델

현재 이 워크스페이스에는 다음 모델 경로들이 있습니다.

| 모델 | 로컬 경로 | 권장 served name |
| --- | --- | --- |
| Qwen 3.5 9B | `/workspace/vllm/models/Qwen/Qwen3.5-9B` | `qwen3.5-9b` |
| Qwen 3.5 2B | `/workspace/vllm/models/Qwen/Qwen3.5-2B` | `qwen3.5-2b` |
| Qwen 3.5 0.8B | `/workspace/vllm/models/Qwen/Qwen3.5-0.8B` | `qwen3.5-0.8b` |
| Gemma 4 E4B IT | `/workspace/vllm/models/google/gemma-4-E4B-it` | `gemma-4-e4b-it` |
| Gemma 4 E2B IT | `/workspace/vllm/models/google/gemma-4-E2B-it` | `gemma-4-e2b-it` |

`model_profiles.json`에는 위 모델들의 권장 env 값이 들어 있습니다. 다만 `manage.sh`가 이 JSON을 자동으로 읽지는 않습니다. 실제 적용은 `serve.env`에 값을 복사해서 합니다.

## `serve.env` 핵심 설정

### 모델 선택

```bash
MODEL_PROFILE=qwen3.5-9b
MODEL_DIR=/workspace/vllm/models/Qwen/Qwen3.5-9B
SERVED_MODEL_NAME=qwen3.5-9b
```

가장 중요한 값은 `MODEL_DIR`입니다. 반드시 실제 존재하는 로컬 모델 디렉터리를 가리켜야 합니다.

`SERVED_MODEL_NAME`은 클라이언트가 OpenAI-compatible API를 호출할 때 사용하는 모델명입니다. 예를 들어:

```bash
SERVED_MODEL_NAME=qwen3.5-9b
```

이면 API 요청의 `model` 값도 `qwen3.5-9b`여야 합니다.

`MODEL_PROFILE`은 `start.sh` 내부 기본값을 고르는 데 쓰입니다. 하지만 모든 모델명이 자동 매핑되는 구조는 아니기 때문에 운영에서는 `MODEL_DIR`와 `SERVED_MODEL_NAME`을 명시하는 편이 안전합니다.

정리하면 모델을 바꿀 때는 보통 이 세 줄을 같이 바꾸면 됩니다.

```bash
MODEL_PROFILE=gemma-4-e2b-it
MODEL_DIR=/workspace/vllm/models/google/gemma-4-E2B-it
SERVED_MODEL_NAME=gemma-4-e2b-it
```

그리고:

```bash
./manage.sh restart
```

### 네트워크

```bash
HOST=0.0.0.0
PORT=8000
```

`HOST=0.0.0.0`은 현재 머신 또는 컨테이너의 모든 인터페이스에서 listen합니다.

`PORT`는 vLLM OpenAI-compatible API 포트입니다. `manage.sh status`는 내부적으로 `127.0.0.1:${PORT}`에 연결해 API가 떠 있는지 확인합니다.

포트 충돌이 있으면 예를 들어 이렇게 바꿉니다.

```bash
PORT=8001
```

그리고 재시작합니다.

```bash
./manage.sh restart
```

### GPU 선택과 tensor parallel

```bash
CUDA_VISIBLE_DEVICES=0,1
TENSOR_PARALLEL_SIZE=2
```

한 장의 GPU만 쓸 때:

```bash
CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1
```

두 장의 GPU를 쓸 때:

```bash
CUDA_VISIBLE_DEVICES=0,1
TENSOR_PARALLEL_SIZE=2
```

`CUDA_VISIBLE_DEVICES`에 보이는 GPU 개수와 `TENSOR_PARALLEL_SIZE`는 맞춰주는 것이 좋습니다. 예를 들어 GPU는 `0,1` 두 장을 보이게 해놓고 `TENSOR_PARALLEL_SIZE=1`로 두거나, GPU 한 장만 보이는데 `TENSOR_PARALLEL_SIZE=2`로 두면 시작 실패의 원인이 될 수 있습니다.

### Context length와 batching

```bash
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=32
```

`MAX_MODEL_LEN`은 요청 하나가 사용할 수 있는 최대 context length입니다. 값을 키우면 긴 입력을 받을 수 있지만 GPU 메모리 사용량이 늘어납니다.

`MAX_NUM_BATCHED_TOKENS`는 vLLM이 한 번에 스케줄링할 수 있는 token batch 규모입니다. 값을 키우면 처리량이 좋아질 수 있지만, 역시 GPU 메모리를 더 씁니다.

`MAX_NUM_SEQS`는 동시에 스케줄링할 sequence 수입니다. 동시 요청을 더 많이 받으려면 키울 수 있지만, 메모리 여유가 먼저 필요합니다.

안정적으로 시작하려면 먼저 아래처럼 두고 확인하는 것을 추천합니다.

```bash
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
```

서버가 안정적으로 뜨고 GPU 메모리 여유가 있으면 그때 조금씩 올리는 방식이 안전합니다.

### GPU 메모리 사용률

```bash
GPU_MEMORY_UTILIZATION=0.90
```

vLLM이 사용할 수 있는 GPU 메모리 비율입니다.

OOM이 나면 가장 먼저 낮춰볼 값입니다.

```bash
GPU_MEMORY_UTILIZATION=0.85
```

작은 모델을 GPU 한 장에서 돌릴 때는 `0.80`에서 `0.85` 정도가 보수적입니다. 9B처럼 더 큰 모델을 두 장에 나눠 올릴 때는 머신이 조용하다는 전제에서 `0.90`도 쓸 수 있습니다.

### Eager 모드와 compile/CUDA graph

```bash
ENFORCE_EAGER=1
MAX_CUDAGRAPH_CAPTURE_SIZE=1024
```

`ENFORCE_EAGER=1`이면 `start.sh`가 vLLM에 `--enforce-eager`를 넘깁니다. 이 모드는 시작이 빠르고 CUDA graph capture 단계에서 터지는 메모리 문제를 피하기 쉽습니다. 모델을 자주 바꾸거나 설정을 튜닝하는 동안에는 이 값이 안전합니다.

`ENFORCE_EAGER=0`이면 `start.sh`가 compile/CUDA graph 설정을 넘깁니다. warmup 이후 지속 처리량이 좋아질 수 있지만, 시작이 무겁고 GPU 메모리 압박이 커질 수 있습니다.

compile/CUDA graph 모드에서 시작 중 터지면 먼저 capture size를 낮춥니다.

```bash
MAX_CUDAGRAPH_CAPTURE_SIZE=512
```

그래도 불안정하면 eager로 돌립니다.

```bash
ENFORCE_EAGER=1
```

## 권장 `serve.env` 예시

### Qwen 3.5 9B, GPU 2장

```bash
MODEL_PROFILE=qwen3.5-9b
MODEL_DIR=/workspace/vllm/models/Qwen/Qwen3.5-9B
SERVED_MODEL_NAME=qwen3.5-9b
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=0,1
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.90
MAX_CUDAGRAPH_CAPTURE_SIZE=1024
DTYPE=auto
ENFORCE_EAGER=1
```

### Qwen 3.5 2B, GPU 1장

```bash
MODEL_PROFILE=qwen3.5-2b
MODEL_DIR=/workspace/vllm/models/Qwen/Qwen3.5-2B
SERVED_MODEL_NAME=qwen3.5-2b
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.85
MAX_CUDAGRAPH_CAPTURE_SIZE=1024
DTYPE=auto
ENFORCE_EAGER=1
```

### Qwen 3.5 0.8B, GPU 1장

```bash
MODEL_PROFILE=qwen3.5-0.8b
MODEL_DIR=/workspace/vllm/models/Qwen/Qwen3.5-0.8B
SERVED_MODEL_NAME=qwen3.5-0.8b
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=96
GPU_MEMORY_UTILIZATION=0.80
MAX_CUDAGRAPH_CAPTURE_SIZE=1024
DTYPE=auto
ENFORCE_EAGER=1
```

### Gemma 4 E4B IT, GPU 1장

```bash
MODEL_PROFILE=gemma-4-e4b-it
MODEL_DIR=/workspace/vllm/models/google/gemma-4-E4B-it
SERVED_MODEL_NAME=gemma-4-e4b-it
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=48
GPU_MEMORY_UTILIZATION=0.85
MAX_CUDAGRAPH_CAPTURE_SIZE=1024
DTYPE=auto
ENFORCE_EAGER=1
```

### Gemma 4 E2B IT, GPU 1장

```bash
MODEL_PROFILE=gemma-4-e2b-it
MODEL_DIR=/workspace/vllm/models/google/gemma-4-E2B-it
SERVED_MODEL_NAME=gemma-4-e2b-it
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.80
MAX_CUDAGRAPH_CAPTURE_SIZE=1024
DTYPE=auto
ENFORCE_EAGER=1
```

## 운영 명령어

설정 적용:

```bash
./manage.sh restart
```

현재 상태 확인:

```bash
./manage.sh status
```

로그 보기:

```bash
./manage.sh logs
```

tmux 세션 접속:

```bash
./manage.sh attach
```

tmux에서 빠져나오기:

```text
Ctrl-b 입력 후 d
```

서버 중지:

```bash
./manage.sh stop
```

## API 확인

서버가 떠 있으면 모델 목록을 확인합니다.

```bash
curl http://127.0.0.1:8000/v1/models
```

간단한 chat completion 테스트:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64
  }'
```

여기서 `"model"` 값은 반드시 `serve.env`의 `SERVED_MODEL_NAME`과 같아야 합니다.

## vLLM 환경 재설치

이 디렉터리는 `uv`를 사용하고, 기본 vLLM 가상환경은 `.venv-vllm`입니다.

```bash
cd /workspace/vllm
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --frozen
```

`start.sh`는 vLLM 실행 파일을 아래 순서로 찾습니다.

1. `VLLM_BIN`이 설정되어 있으면 그 값을 사용합니다.
2. `uv`가 있으면 `UV_PROJECT_ENVIRONMENT=.venv-vllm` 기준으로 `uv run vllm`을 사용합니다.
3. `.venv-vllm/bin/vllm`이 있으면 직접 사용합니다.
4. 그래도 없으면 `PATH`에 있는 `vllm`을 사용합니다.

일반적인 재설치 흐름:

```bash
cd /workspace/vllm
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --frozen
./manage.sh restart
```

## 자주 나는 문제

### `Model directory does not exist`

`serve.env`의 `MODEL_DIR`가 실제 경로인지 확인합니다.

```bash
ls -la /workspace/vllm/models/Qwen/Qwen3.5-9B
```

모델을 바꿨다면 `MODEL_DIR`도 같이 바꿔야 합니다. `MODEL_PROFILE`만 바꾸는 것보다 `MODEL_DIR`, `SERVED_MODEL_NAME`까지 명시하는 쪽이 안전합니다.

### `status`에서 API가 안 뜬다고 나올 때

```bash
./manage.sh status
./manage.sh logs
```

tmux가 stopped이면:

```bash
./manage.sh start
```

tmux는 running인데 API가 not listening이면, vLLM이 아직 뜨는 중이거나 시작하다가 실패한 상태일 수 있습니다. 이때는 `./manage.sh logs`가 제일 중요합니다.

### CUDA out of memory

먼저 안정 우선 설정으로 낮춥니다.

```bash
ENFORCE_EAGER=1
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=32
```

GPU 2장 모델이면 아래도 확인합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1
TENSOR_PARALLEL_SIZE=2
```

GPU 1장 모델이면 아래처럼 맞춥니다.

```bash
CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1
```

### CUDA graph capture 또는 compile 단계에서 실패

가장 쉬운 해결은 eager 모드입니다.

```bash
ENFORCE_EAGER=1
```

compile/CUDA graph를 유지하고 싶으면 capture size를 낮춥니다.

```bash
MAX_CUDAGRAPH_CAPTURE_SIZE=512
```

변경 후:

```bash
./manage.sh restart
```

### 포트가 이미 사용 중일 때

`serve.env`에서 포트를 바꿉니다.

```bash
PORT=8001
```

그리고:

```bash
./manage.sh restart
```

### 클라이언트에서 model not found가 날 때

클라이언트 요청의 `"model"` 값과 `serve.env`의 `SERVED_MODEL_NAME`이 같은지 확인합니다.

예를 들어 `serve.env`가 아래와 같으면:

```bash
SERVED_MODEL_NAME=gemma-4-e2b-it
```

요청도 아래처럼 보내야 합니다.

```json
{"model": "gemma-4-e2b-it"}
```

## git 관리 기준

git에 올릴 파일:

- `README.md`
- `.gitignore`
- `serve.env.example`
- `start.sh`
- `manage.sh`
- `pyproject.toml`
- `uv.lock`
- `model_profiles.json`

git에 올리지 않을 파일:

- `serve.env`
- `.venv-vllm/`
- `models/`
- `logs/`
- Python cache
- 임시 파일과 로컬 IDE 설정

이 기준으로 관리하면 저장소는 작게 유지하면서도, 각 머신마다 GPU와 모델 설정을 자유롭게 바꿀 수 있습니다.
