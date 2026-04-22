"""Helpers for choosing a Gemini model at runtime."""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]


def normalize_gemini_model_name(model_name: str) -> str:
    """Strip API prefixes from Gemini model names returned by list_models()."""
    return model_name.split("/")[-1].strip()


def _iter_available_models(genai) -> list[str]:
    try:
        models = genai.list_models()
    except Exception as exc:
        logger.debug("Could not list Gemini models: %s", exc)
        return []

    available = []
    for model in models:
        methods = getattr(model, "supported_generation_methods", []) or []
        normalized_name = normalize_gemini_model_name(getattr(model, "name", ""))
        if normalized_name and any(
            method.lower() == "generatecontent" or method.lower() == "generate_content"
            for method in methods
        ):
            available.append(normalized_name)
    return available


def select_gemini_model(genai, preferred: str | None = None) -> str:
    """Choose the cheapest supported Gemini model available in the current environment."""
    available = _iter_available_models(genai)

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(DEFAULT_GEMINI_MODELS)

    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for candidate in candidates:
        normalized = normalize_gemini_model_name(candidate)
        if normalized not in seen:
            ordered_candidates.append(normalized)
            seen.add(normalized)

    for candidate in ordered_candidates:
        if candidate in available:
            logger.info("Selected Gemini model: %s", candidate)
            return candidate

    if available:
        fallback = available[0]
        logger.info("Selected fallback Gemini model: %s", fallback)
        return fallback

    fallback = normalize_gemini_model_name(preferred or DEFAULT_GEMINI_MODELS[0])
    logger.info("Gemini model list unavailable; falling back to %s", fallback)
    return fallback
