"""AWS Lambda handler to generate images with Google Gemini."""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import google.generativeai as genai

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DEFAULT_MODEL = "gemini-2.5-flash-image-preview" # or gemini-1.5-pro
DEFAULT_MIME_TYPE = "image/png"

_SSM_CLIENT = boto3.client("ssm")
_CACHED_API_KEY: Optional[str] = None


class BadRequestError(Exception):
    """Raised when the incoming request is invalid."""


@dataclass
class ImagePayload:
    mime_type: str
    data: str

    def as_dict(self) -> Dict[str, str]:
        return {"mimeType": self.mime_type, "data": self.data}


def _build_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload)
    }


def _derive_status_code_from_exception(error: Exception) -> Optional[int]:
    for attr in ("code", "status_code", "status", "http_status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value

    message = str(error).strip()
    match = re.match(r"^(\d{3})\b", message)
    if match:
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover - defensive guard
            return None

    return None


def _parse_body(event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not event:
        raise BadRequestError("El evento de la solicitud está vacío")

    body = event.get("body")
    if isinstance(body, str):
        if not body:
            raise BadRequestError("El cuerpo de la solicitud está vacío")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise BadRequestError("El cuerpo de la solicitud no es un JSON válido") from exc

    raise BadRequestError("Formato de cuerpo no soportado")



def _extract_prompt(payload: Dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        prompt = prompt.strip()
    if not prompt:
        raise BadRequestError('El campo "prompt" es obligatorio')
    return prompt


def _resolve_model(payload: Dict[str, Any]) -> str:
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return DEFAULT_MODEL


def _build_content_parts(prompt: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    parts = [{"text": prompt}]
    negative_prompt = payload.get("negativePrompt") or payload.get("negative_prompt")
    if isinstance(negative_prompt, str) and negative_prompt.strip():
        parts.append({"text": f"Evita: {negative_prompt.strip()}"})
    return parts


def _extract_images(result: Any) -> List[ImagePayload]:
    images: List[ImagePayload] = []

    candidates = getattr(result, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
            if inline_data and getattr(inline_data, "data", None):
                mime = getattr(inline_data, "mime_type", None) or getattr(inline_data, "mimeType", None) or DEFAULT_MIME_TYPE
                images.append(ImagePayload(mime_type=mime, data=inline_data.data))
    return images


def _get_api_key() -> Optional[str]:
    global _CACHED_API_KEY

    if _CACHED_API_KEY:
        return _CACHED_API_KEY

    parameter_name = os.environ.get("GOOGLE_API_KEY_PARAM")
    if not parameter_name:
        LOGGER.error("GOOGLE_API_KEY_PARAM no está configurada")
        return None

    try:
        response = _SSM_CLIENT.get_parameter(Name=parameter_name, WithDecryption=True)
        _CACHED_API_KEY = response["Parameter"]["Value"]
        return _CACHED_API_KEY
    except (BotoCoreError, ClientError) as error:
        LOGGER.exception("No se pudo obtener la API key desde SSM", exc_info=error)
        return None


def lambda_handler(event: Optional[Dict[str, Any]], _context: Any) -> Dict[str, Any]:
    """Entry point for the AWS Lambda runtime."""
    api_key = _get_api_key()
    if not api_key:
        return _build_response(500, {"message": "No se pudo obtener la API key de Gemini"})

    try:
        payload = _parse_body(event)
        prompt = _extract_prompt(payload)
        model_name = _resolve_model(payload)
        parts = _build_content_parts(prompt, payload)

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        request_kwargs = {
            "contents": [{"role": "user", "parts": parts}]
        }

        result = model.generate_content(**request_kwargs)

        images = _extract_images(result)
        if not images:
            LOGGER.error("La respuesta de Gemini no contiene imágenes")
            return _build_response(502, {"message": "Gemini no generó imágenes para el prompt proporcionado"})

        return _build_response(200, {
            "model": model_name,
            "images": [image.as_dict() for image in images]
        })

    except BadRequestError as error:
        LOGGER.warning("Solicitud inválida: %s", error)
        return _build_response(400, {"message": str(error)})
    except Exception as error:  # pragma: no cover - cubierta con logging y manejo genérico
        LOGGER.exception("Error generando imagen")
        status_code = _derive_status_code_from_exception(error) or 502
        message = str(error)

        payload: Dict[str, Any] = {
            "error": "GeminiError",
            "status": status_code,
            "message": message
        }

        detail = getattr(error, "details", None)
        if isinstance(detail, str) and detail.strip() and detail.strip() != message:
            payload["detail"] = detail.strip()

        trace_id = getattr(error, "trace_id", None)
        if isinstance(trace_id, str) and trace_id.strip():
            payload["traceId"] = trace_id.strip()

        return _build_response(status_code, payload)
