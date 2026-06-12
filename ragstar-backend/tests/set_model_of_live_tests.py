from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import pytest
import requests

from app.core.llm_factory import build_chat_ollama
from app.core.settings import get_settings


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live-model",
        action="store",
        default=None,
        help="Fallback chat model override for all live LLM tests.",
    )
    parser.addoption(
        "--llm",
        action="store",
        default=None,
        help="Shorthand alias for --live-model.",
    )
    parser.addoption(
        "--live-node2-model",
        action="store",
        default=None,
        help="Chat model override used only for Node 2 live tests.",
    )
    parser.addoption(
        "--live-node4-model",
        action="store",
        default=None,
        help="Chat model override used only for Node 4 live tests.",
    )
    parser.addoption(
        "--runs",
        action="store",
        type=int,
        default=None,
        help="Override repeat count for live stability tests. Falls back to each test file's environment variable.",
    )


def _resolve_model(config: pytest.Config, option_name: str) -> str | None:
    specific = config.getoption(option_name)
    if specific:
        return str(specific)

    shared = config.getoption("--live-model")
    if shared:
        return str(shared)

    alias = config.getoption("--llm")
    if alias:
        return str(alias)

    return None


def _default_live_node2_model() -> str:
    return get_settings().node2_model


def _default_live_node4_model() -> str:
    return get_settings().node4_model


def _resolve_runs(config: pytest.Config, env_name: str, default_runs: int) -> int:
    cli_runs = config.getoption("--runs")
    if cli_runs is not None:
        return int(cli_runs)

    return int(os.getenv(env_name, str(default_runs)))


@lru_cache(maxsize=1)
def _get_available_ollama_model_names() -> set[str]:
    settings = get_settings()
    response = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()

    payload = response.json()
    return {
        str(model.get("name"))
        for model in payload.get("models", [])
        if model.get("name")
    }


def _match_requested_live_model(model: str, available_models: set[str]) -> str | None:
    requested = str(model).strip()
    if not requested:
        return None

    if requested in available_models:
        return requested

    if ":" not in requested:
        latest_name = f"{requested}:latest"
        if latest_name in available_models:
            return latest_name

    return None


def _validate_requested_live_model(model: str, *, option_name: str) -> str:
    try:
        available_models = _get_available_ollama_model_names()
    except requests.RequestException as exc:
        raise pytest.UsageError(
            "live test 모델 검증 중 Ollama 서버 조회에 실패했습니다. "
            f"option={option_name}, model={model}, error={exc}"
        ) from exc

    matched_model = _match_requested_live_model(model, available_models)
    if matched_model is None:
        raise pytest.UsageError(
            "유효하지 않은 live test LLM 인자입니다. "
            f"option={option_name}, model={model}, available={sorted(available_models)}"
        )

    return matched_model


@pytest.fixture
def live_node2_model(pytestconfig: pytest.Config) -> str:
    override_model = _resolve_model(pytestconfig, "--live-node2-model")
    if override_model:
        return _validate_requested_live_model(override_model, option_name="--live-node2-model/--live-model/--llm")

    return _default_live_node2_model()


@pytest.fixture
def live_node4_model(pytestconfig: pytest.Config) -> str:
    override_model = _resolve_model(pytestconfig, "--live-node4-model")
    if override_model:
        return _validate_requested_live_model(override_model, option_name="--live-node4-model/--live-model/--llm")

    return _default_live_node4_model()


@pytest.fixture
def live_node2_runs(pytestconfig: pytest.Config) -> int:
    return _resolve_runs(pytestconfig, env_name="NODE2_STABILITY_RUNS", default_runs=10)


@pytest.fixture
def live_node4_runs(pytestconfig: pytest.Config) -> int:
    return _resolve_runs(pytestconfig, env_name="NODE4_STABILITY_RUNS", default_runs=5)


@pytest.fixture
def live_node2_llm(live_node2_model: str) -> Any:
    return build_chat_ollama(live_node2_model, json_mode=True)


@pytest.fixture
def live_node4_llm(live_node4_model: str) -> Any:
    return build_chat_ollama(live_node4_model, json_mode=True)