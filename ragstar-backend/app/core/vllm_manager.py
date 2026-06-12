import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from app.core.settings import get_settings

ModelProfile = tuple[str, Path, dict[str, str]]


def _model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _served_name_from_dir(model_dir: Path) -> str:
    name = model_dir.name.lower()
    name = re.sub(r"[^a-z0-9.]+", "-", name)
    return name.strip("-")


def _aliases_for(served_name: str, model_dir: Path) -> set[str]:
    aliases = {served_name, model_dir.name, model_dir.name.lower()}
    aliases.add(served_name.replace("-", ":"))
    aliases.add(served_name.replace("-", ""))
    return aliases


def _profile_env_values(profile: dict[str, Any]) -> dict[str, str]:
    env = profile.get("env") or {}
    values = {str(key): str(value) for key, value in env.items() if value is not None}
    if profile.get("model_dir"):
        values["MODEL_DIR"] = str(profile["model_dir"])
    if profile.get("served_model_name"):
        values["SERVED_MODEL_NAME"] = str(profile["served_model_name"])
    return values


def _profile_aliases(profile_name: str, profile: dict[str, Any], served_name: str, model_dir: Path) -> set[str]:
    aliases = set(_aliases_for(served_name, model_dir))
    aliases.update({profile_name, served_name})

    for value in profile.get("aliases") or []:
        aliases.add(str(value))
    if profile.get("hf_id"):
        hf_id = str(profile["hf_id"])
        aliases.add(hf_id)
        aliases.add(hf_id.split("/")[-1])

    return {alias for alias in aliases if alias}


def _load_profiled_vllm_models() -> dict[str, ModelProfile]:
    settings = get_settings()
    profile_path = Path(settings.vllm_model_profiles)
    discovered: dict[str, ModelProfile] = {}

    if not profile_path.exists():
        return discovered

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"vLLM model profile file must contain an object: {profile_path}")

    for profile_name, raw_profile in payload.items():
        if not isinstance(raw_profile, dict):
            continue

        model_dir_value = raw_profile.get("model_dir")
        if not model_dir_value:
            continue

        model_dir = Path(str(model_dir_value))
        served_name = str(raw_profile.get("served_model_name") or profile_name)
        env_values = _profile_env_values(raw_profile)
        for alias in _profile_aliases(str(profile_name), raw_profile, served_name, model_dir):
            discovered[_model_key(alias)] = (served_name, model_dir, env_values)

    return discovered


def discover_local_vllm_models() -> dict[str, ModelProfile]:
    settings = get_settings()
    models_root = Path(settings.vllm_models_root)
    discovered = _load_profiled_vllm_models()

    if not models_root.exists():
        return discovered

    for config_path in models_root.glob("*/*/config.json"):
        model_dir = config_path.parent
        served_name = _served_name_from_dir(model_dir)
        for alias in _aliases_for(served_name, model_dir):
            key = _model_key(alias)
            if key not in discovered:
                discovered[key] = (served_name, model_dir, {})

    return discovered


def resolve_local_vllm_model(model_name: str) -> ModelProfile:
    discovered = discover_local_vllm_models()
    resolved = discovered.get(_model_key(model_name))
    if resolved:
        served_name, model_dir, _env_values = resolved
        if not (model_dir / "config.json").exists():
            raise RuntimeError(
                f"vLLM model profile '{model_name}' points to {model_dir}, "
                "but model files are not downloaded yet. Download it with "
                f"`hf download <repo-id> --local-dir {model_dir}`."
            )
        return resolved

    available = sorted({name for name, path, _env in discovered.values() if (path / "config.json").exists()})
    available_text = ", ".join(available) if available else "none"
    raise RuntimeError(
        f"vLLM model '{model_name}' is not installed under "
        f"{get_settings().vllm_models_root}. Available local models: {available_text}"
    )


def _base_url_for_models() -> str:
    base_url = get_settings().vllm_base_url.rstrip("/") + "/"
    return urljoin(base_url, "models")


def get_served_vllm_models() -> set[str]:
    request = Request(_base_url_for_models(), headers={"Authorization": f"Bearer {get_settings().vllm_api_key}"})
    with urlopen(request, timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")}


def _is_api_port_open() -> bool:
    settings = get_settings()
    host = "127.0.0.1"
    port = 8000
    match = re.search(r":(\d+)(?:/|$)", settings.vllm_base_url)
    if match:
        port = int(match.group(1))

    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _read_env_file(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return lines, values


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    lines, values = _read_env_file(path)
    values.update(updates)
    seen: set[str] = set()
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in values:
            output.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            output.append(line)

    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _restart_vllm() -> None:
    settings = get_settings()
    manage_script = Path(settings.vllm_manage_script)
    if not manage_script.exists():
        raise RuntimeError(f"vLLM manage script does not exist: {manage_script}")

    subprocess.run(
        [str(manage_script), "restart"],
        cwd=str(manage_script.parent),
        check=True,
        text=True,
    )


def _wait_until_served(served_name: str) -> None:
    deadline = time.time() + get_settings().vllm_switch_timeout_seconds
    last_error = ""

    while time.time() < deadline:
        try:
            if served_name in get_served_vllm_models():
                return
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(5)

    port_state = "open" if _is_api_port_open() else "closed"
    raise RuntimeError(
        f"Timed out waiting for vLLM model '{served_name}' to become ready "
        f"(api port is {port_state}; last error: {last_error})"
    )


def ensure_vllm_model(model_name: str) -> str:
    """
    Ensure local vLLM is serving model_name and return the canonical served name.

    The model must already be downloaded under VLLM_MODELS_ROOT. In unit tests we
    skip process management so tests can monkeypatch LLM factories without
    starting/stopping a real vLLM server.

    Caller-level ENFORCE_EAGER override:
        Set RAGSTAR_VLLM_ENFORCE_EAGER=0 (steady-state serving — compile/CUDA
        graph for better throughput) or =1 (sweeps/experiments — eager mode,
        fast startup, no capture-time OOM). When unset, the profile's own
        value wins. If the already-running vLLM disagrees with the desired
        mode, we force a restart so the new mode actually takes effect.
    """
    settings = get_settings()
    if not settings.use_vllm or not settings.vllm_auto_switch_model:
        return model_name

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return model_name

    served_name, model_dir, profile_env = resolve_local_vllm_model(model_name)

    eager_override = os.environ.get("RAGSTAR_VLLM_ENFORCE_EAGER")
    if eager_override is not None:
        profile_env = {**profile_env, "ENFORCE_EAGER": eager_override}

    desired_eager = profile_env.get("ENFORCE_EAGER")
    _, current_serve_env = _read_env_file(Path(settings.vllm_serve_env))
    eager_matches = desired_eager is None or current_serve_env.get("ENFORCE_EAGER") == desired_eager

    try:
        if served_name in get_served_vllm_models() and eager_matches:
            return served_name
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass

    _write_env_file(
        Path(settings.vllm_serve_env),
        {
            **profile_env,
            "MODEL_PROFILE": served_name,
            "MODEL_DIR": str(model_dir),
            "SERVED_MODEL_NAME": served_name,
        },
    )
    _restart_vllm()
    _wait_until_served(served_name)
    return served_name
