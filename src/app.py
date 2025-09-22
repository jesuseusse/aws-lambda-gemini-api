"""AWS Lambda handler to generate images with Google Gemini."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DEFAULT_MODEL = "gemini-1.5-flash-latest"
DEFAULT_MIME_TYPE = "image/png"


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


def _parse_body(event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not event:
        return {}

    body = event.get("body")
    if body is None:
        # Support direct invocation where the event already contains the payload.
        return {key: value for key, value in event.items() if key != "headers"}

    if isinstance(body, dict):
        return body

    if isinstance(body, str):
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:  # pragma: no cover - protects future refactor
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


def _resolve_mime_type(payload: Dict[str, Any]) -> str:
    mime_type = payload.get("mimeType") or payload.get("mime_type")
    if isinstance(mime_type, str) and mime_type.strip():
        return mime_type.strip()
    return DEFAULT_MIME_TYPE


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


def lambda_handler(event: Optional[Dict[str, Any]], _context: Any) -> Dict[str, Any]:
    """Entry point for the AWS Lambda runtime."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        LOGGER.error("GOOGLE_API_KEY no está configurada")
        return _build_response(500, {"message": "Configuración ausente: GOOGLE_API_KEY"})

    try:
        payload = _parse_body(event)
        prompt = _extract_prompt(payload)
        model_name = _resolve_model(payload)
        mime_type = _resolve_mime_type(payload)
        parts = _build_content_parts(prompt, payload)

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        generation_config = {"response_mime_type": mime_type}
        request_contents = [{"role": "user", "parts": parts}]

        result = model.generate_content(
            contents=request_contents,
            generation_config=generation_config
        )

        images = _extract_images(result)
        if not images:
            LOGGER.error("La respuesta de Gemini no contiene imágenes")
            return _build_response(502, {"message": "Gemini no generó imágenes para el prompt proporcionado"})

        return _build_response(200, {"images": [image.as_dict() for image in images]})

    except BadRequestError as error:
        LOGGER.warning("Solicitud inválida: %s", error)
        return _build_response(400, {"message": str(error)})
    except Exception as error:  # pragma: no cover - cubierta con logging y manejo genérico
        LOGGER.exception("Error generando imagen")
        return _build_response(502, {"message": "Error al generar la imagen", "detail": str(error)})
