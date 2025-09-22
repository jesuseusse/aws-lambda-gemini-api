import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@pytest.fixture()
def mock_genai():
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.generativeai")

    model_instance = MagicMock()
    genai_module.configure = MagicMock()
    genai_module.GenerativeModel = MagicMock(return_value=model_instance)

    google_module.generativeai = genai_module

    sys.modules["google"] = google_module
    sys.modules["google.generativeai"] = genai_module

    yield {
        "genai": genai_module,
        "model_instance": model_instance
    }

    sys.modules.pop("google.generativeai", None)
    sys.modules.pop("google", None)


@pytest.fixture()
def app_module(mock_genai):
    sys.modules.pop("src.app", None)
    module = importlib.import_module("src.app")
    return module


def _build_event(body):
    return {"body": json.dumps(body)}


def test_missing_api_key_returns_500(app_module, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    response = app_module.lambda_handler(_build_event({"prompt": "hola"}), None)
    assert response["statusCode"] == 500
    assert "GOOGLE_API_KEY" in json.loads(response["body"])['message']


def test_invalid_json_returns_400(app_module, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = app_module.lambda_handler({"body": "{invalid"}, None)
    assert response["statusCode"] == 400


def test_missing_prompt_returns_400(app_module, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = app_module.lambda_handler(_build_event({}), None)
    assert response["statusCode"] == 400


def test_successful_generation_returns_images(app_module, mock_genai, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    inline_data = SimpleNamespace(data="base64img", mime_type="image/png")
    part = SimpleNamespace(inline_data=inline_data)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    result = SimpleNamespace(candidates=[candidate])

    mock_genai["model_instance"].generate_content.return_value = result

    response = app_module.lambda_handler(_build_event({"prompt": "Un paisaje"}), None)
    payload = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert payload["images"] == [{"mimeType": "image/png", "data": "base64img"}]
    mock_genai["genai"].configure.assert_called_once_with(api_key="fake-key")


def test_no_images_returns_502(app_module, mock_genai, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    empty_content = SimpleNamespace(parts=[])
    candidate = SimpleNamespace(content=empty_content)
    result = SimpleNamespace(candidates=[candidate])

    mock_genai["model_instance"].generate_content.return_value = result

    response = app_module.lambda_handler(_build_event({"prompt": "Un paisaje"}), None)

    assert response["statusCode"] == 502
