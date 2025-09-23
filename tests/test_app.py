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
def mock_boto3():
    boto3_module = types.ModuleType("boto3")
    client_instance = MagicMock()
    boto3_module.client = MagicMock(return_value=client_instance)

    botocore_module = types.ModuleType("botocore")
    exceptions_module = types.ModuleType("botocore.exceptions")
    exceptions_module.BotoCoreError = Exception
    exceptions_module.ClientError = Exception
    botocore_module.exceptions = exceptions_module

    sys.modules["boto3"] = boto3_module
    sys.modules["botocore"] = botocore_module
    sys.modules["botocore.exceptions"] = exceptions_module

    yield {
        "boto3": boto3_module,
        "client": client_instance
    }

    sys.modules.pop("boto3", None)
    sys.modules.pop("botocore.exceptions", None)
    sys.modules.pop("botocore", None)


@pytest.fixture()
def app_module(mock_genai, mock_boto3):
    sys.modules.pop("src.app", None)
    module = importlib.import_module("src.app")
    return module


def _build_event(body):
    return {"body": json.dumps(body)}


def test_missing_api_key_returns_500(app_module, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY_PARAM", raising=False)
    response = app_module.lambda_handler(_build_event({"prompt": "hola"}), None)
    assert response["statusCode"] == 500
    assert "API key" in json.loads(response["body"])['message']


def test_invalid_json_returns_400(app_module, mock_boto3, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY_PARAM", "/prod/key")
    mock_boto3["client"].get_parameter.return_value = {"Parameter": {"Value": "fake-key"}}
    response = app_module.lambda_handler({"body": "{invalid"}, None)
    assert response["statusCode"] == 400


def test_missing_prompt_returns_400(app_module, mock_boto3, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY_PARAM", "/prod/key")
    mock_boto3["client"].get_parameter.return_value = {"Parameter": {"Value": "fake-key"}}
    response = app_module.lambda_handler(_build_event({}), None)
    assert response["statusCode"] == 400


def test_successful_generation_returns_images(app_module, mock_genai, mock_boto3, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY_PARAM", "/prod/key")
    mock_boto3["client"].get_parameter.return_value = {"Parameter": {"Value": "fake-key"}}

    inline_data = SimpleNamespace(data="base64img", mime_type="image/png")
    part = SimpleNamespace(inline_data=inline_data)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    result = SimpleNamespace(candidates=[candidate])

    mock_genai["model_instance"].generate_content.return_value = result

    response = app_module.lambda_handler(_build_event({"prompt": "Un paisaje"}), None)
    payload = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert payload["model"] == "gemini-2.5-pro"
    assert payload["images"] == [{"mimeType": "image/png", "data": "base64img"}]
    mock_genai["genai"].configure.assert_called_once_with(api_key="fake-key")
    mock_boto3["boto3"].client.assert_called_once_with("ssm")
    mock_boto3["client"].get_parameter.assert_called_once_with(Name="/prod/key", WithDecryption=True)
    _, kwargs = mock_genai["model_instance"].generate_content.call_args
    assert "generation_config" not in kwargs


def test_generation_config_applied_for_allowed_mimetype(app_module, mock_genai, mock_boto3, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY_PARAM", "/prod/key")
    mock_boto3["client"].get_parameter.return_value = {"Parameter": {"Value": "fake-key"}}

    app_module.lambda_handler(_build_event({
        "prompt": "Describe the image", "mimeType": "application/json"
    }), None)

    _, kwargs = mock_genai["model_instance"].generate_content.call_args
    assert kwargs["generation_config"] == {"response_mime_type": "application/json"}


def test_custom_model_is_used_when_provided(app_module, mock_genai, mock_boto3, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY_PARAM", "/prod/key")
    mock_boto3["client"].get_parameter.return_value = {"Parameter": {"Value": "fake-key"}}

    app_module.lambda_handler(_build_event({
        "prompt": "Render a forest",
        "model": "gemini-1.5-pro-latest"
    }), None)

    mock_genai["genai"].GenerativeModel.assert_called_with("gemini-1.5-pro-latest")


def test_no_images_returns_502(app_module, mock_genai, mock_boto3, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY_PARAM", "/prod/key")
    mock_boto3["client"].get_parameter.return_value = {"Parameter": {"Value": "fake-key"}}

    empty_content = SimpleNamespace(parts=[])
    candidate = SimpleNamespace(content=empty_content)
    result = SimpleNamespace(candidates=[candidate])

    mock_genai["model_instance"].generate_content.return_value = result
    response = app_module.lambda_handler(_build_event({"prompt": "Un paisaje"}), None)

    assert response["statusCode"] == 502
