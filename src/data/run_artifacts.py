from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.data.shared_preprocessing_cache import load_shared_preprocessing_cache

logger = logging.getLogger(__name__)

SHARED_DATA_MANIFEST_NAME = "shared_data_manifest.json"
SHARED_DATA_MANIFEST_VERSION = 1


def shared_data_manifest_path(data_dir: Path) -> Path:
    return data_dir / SHARED_DATA_MANIFEST_NAME


def build_shared_data_manifest(*, cache_dir: Path, cache_key: str, manifest_path: Path) -> dict[str, Any]:
    return {
        "schema_version": SHARED_DATA_MANIFEST_VERSION,
        "shared_preprocessing": {
            "cache_dir": str(cache_dir),
            "cache_key": str(cache_key),
            "manifest_path": str(manifest_path),
        },
    }


def load_shared_data_manifest(data_dir: Path) -> dict[str, Any] | None:
    manifest_file = shared_data_manifest_path(data_dir)
    if manifest_file.exists():
        try:
            return json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read shared data manifest %s: %s", manifest_file, exc)

    run_dir = data_dir.parent
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            preprocessing = summary.get("preprocessing") or {}
            cache_dir = preprocessing.get("cache_dir")
            cache_key = preprocessing.get("cache_key")
            manifest_path = preprocessing.get("manifest_path")
            if cache_dir:
                return build_shared_data_manifest(
                    cache_dir=Path(cache_dir),
                    cache_key=str(cache_key or ""),
                    manifest_path=Path(manifest_path or Path(cache_dir) / "manifest.json"),
                )
        except Exception as exc:
            logger.warning("Failed to infer shared data manifest from %s: %s", summary_path, exc)

    return None


def resolve_shared_cache_dir(data_dir: Path) -> Path | None:
    manifest = load_shared_data_manifest(data_dir)
    if not manifest:
        return None
    shared_preprocessing = manifest.get("shared_preprocessing") or {}
    cache_dir = shared_preprocessing.get("cache_dir")
    return Path(cache_dir) if cache_dir else None


def load_shared_preprocessing_payload_for_run(data_dir: Path) -> dict[str, Any] | None:
    cache_dir = resolve_shared_cache_dir(data_dir)
    if not cache_dir:
        return None
    if not cache_dir.exists():
        logger.warning("Shared preprocessing cache directory does not exist: %s", cache_dir)
        return None
    try:
        return load_shared_preprocessing_cache(cache_dir)
    except Exception as exc:
        logger.warning("Failed to load shared preprocessing cache %s: %s", cache_dir, exc)
        return None
