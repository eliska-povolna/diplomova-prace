"""Streamlit caching and session state management."""

import json
import os
import logging
import pickle
import tempfile
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

    # Define dummy decorator for non-Streamlit contexts
    def cache_resource(func):
        return func


import yaml

from src.ui.services import (
    DataService,
    InferenceService,
    LabelingService,
    WordcloudService,
)
from src.ui.services.coactivation_service import CoactivationService

logger = logging.getLogger(__name__)


def _get_cloud_storage_helper():
    """Return a CloudStorageHelper when GCS is configured, otherwise None."""
    bucket_name = os.getenv("GCS_BUCKET_NAME") or os.getenv("CLOUD_STORAGE_BUCKET")
    if not bucket_name:
        return None

    try:
        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        return CloudStorageHelper(bucket_name=bucket_name)
    except Exception as e:
        logger.debug(f"Cloud Storage helper unavailable: {e}")
        return None


def _find_latest_model_timestamp(cloud_storage) -> Optional[str]:
    """Find the newest timestamped model prefix in GCS."""
    try:
        blobs = cloud_storage.bucket.list_blobs(prefix="outputs/")
    except Exception as e:
        logger.debug(f"Failed to list GCS model artifacts: {e}")
        return None

    timestamps = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) >= 2 and len(parts[1]) == 15:
            timestamps.add(parts[1])

    if not timestamps:
        return None

    return sorted(timestamps)[-1]


def _extract_run_timestamp(selected_output_dir: Optional[str]) -> Optional[str]:
    """Extract timestamp like YYYYMMDD_HHMMSS from a path/string."""
    if not selected_output_dir:
        return None

    raw = str(selected_output_dir).replace("\\", "/")

    match = re.search(r"\d{8}_\d{6}", raw)
    if match:
        return match.group(0)

    logger.warning(
        "Run timestamp was not parsed successfully: raw=%s, normalized=%s, name=%s",
        selected_output_dir,
        raw,
        Path(raw).name,
    )
    return None


def _strict_runtime_error(title: str, issues: List[str]) -> RuntimeError:
    issue_text = "\n".join(f"- {issue}" for issue in issues if issue)
    return RuntimeError(f"{title}\n{issue_text}")


def _find_latest_local_experiment_dir(outputs_base: Path) -> Path:
    latest_pointer = outputs_base / "LATEST_EXPERIMENT.txt"
    if latest_pointer.exists():
        try:
            pointer_text = latest_pointer.read_text(encoding="utf-8").strip()
            pointer_dir = Path(pointer_text)
        except Exception as e:
            raise _strict_runtime_error(
                "Failed to read latest experiment pointer.",
                [f"path={latest_pointer}", f"error={e}"],
            ) from e

        if not pointer_dir.exists():
            raise _strict_runtime_error(
                "Latest experiment directory does not exist.",
                [f"pointer={latest_pointer}", f"resolved_dir={pointer_dir}"],
            )
        return pointer_dir

    experiments_base = outputs_base / "experiments"
    if not experiments_base.exists():
        raise _strict_runtime_error(
            "Missing experiments directory for strict best-run mode.",
            [f"expected_dir={experiments_base}"],
        )

    experiment_dirs = sorted([d for d in experiments_base.iterdir() if d.is_dir()])
    if not experiment_dirs:
        raise _strict_runtime_error(
            "No experiment directories found for strict best-run mode.",
            [f"experiments_dir={experiments_base}"],
        )
    return experiment_dirs[-1]


def _list_local_experiment_dirs(outputs_base: Path) -> List[Path]:
    """List local experiment directories in newest-first order.

    If LATEST_EXPERIMENT.txt exists and resolves to a valid directory, it is
    treated as the first candidate, followed by remaining experiment dirs.
    """
    candidates: List[Path] = []
    latest_pointer = outputs_base / "LATEST_EXPERIMENT.txt"

    if latest_pointer.exists():
        try:
            pointer_text = latest_pointer.read_text(encoding="utf-8").strip()
            pointer_dir = Path(pointer_text)
            if pointer_dir.exists() and pointer_dir.is_dir():
                candidates.append(pointer_dir)
        except Exception:
            # Pointer parsing issues are handled by strict loaders as diagnostic context.
            pass

    experiments_base = outputs_base / "experiments"
    if experiments_base.exists():
        experiment_dirs = sorted(
            [d for d in experiments_base.iterdir() if d.is_dir()],
            reverse=True,
        )
        for exp_dir in experiment_dirs:
            if exp_dir not in candidates:
                candidates.append(exp_dir)

    return candidates


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        raise _strict_runtime_error(
            "Failed to parse JSON artifact.",
            [f"path={path}", f"error={e}"],
        ) from e
    if not isinstance(payload, dict):
        raise _strict_runtime_error(
            "Invalid JSON artifact payload type.",
            [f"path={path}", f"type={type(payload).__name__}"],
        )
    return payload


def _validate_precomputed_matrices_payload(
    payload: Any,
    *,
    run_id: str,
    expected_n_items: Optional[int],
    source_path: Path,
) -> None:
    if not isinstance(payload, dict):
        raise _strict_runtime_error(
            "Invalid precomputed matrix payload.",
            [f"path={source_path}", f"type={type(payload).__name__}"],
        )
    # do not require for backward compatibility
    # required_keys = {"run_id", "n_items", "matrices"}
    # if not required_keys.issubset(payload.keys()):
    #     raise _strict_runtime_error(
    #         "Invalid precomputed matrix payload schema (strict mode requires metadata envelope).",
    #         [
    #             f"path={source_path}",
    #             f"expected_keys={sorted(required_keys)}",
    #             f"actual_keys={sorted(payload.keys())}",
    #         ],
    #     )

    payload_run_id = str(payload.get("run_id"))
    if payload_run_id != run_id:
        raise _strict_runtime_error(
            "Precomputed matrix run_id mismatch.",
            [
                f"path={source_path}",
                f"expected_run_id={run_id}",
                f"actual_run_id={payload_run_id}",
                f"payload={payload}",
            ],
        )

    try:
        payload_n_items = int(payload.get("n_items"))
    except (TypeError, ValueError):
        raise _strict_runtime_error(
            "Invalid precomputed matrix n_items metadata.",
            [f"path={source_path}", f"n_items={payload.get('n_items')}"],
        ) from None

    if expected_n_items is not None and payload_n_items != expected_n_items:
        raise _strict_runtime_error(
            "Precomputed matrix metadata mismatch.",
            [
                f"path={source_path}",
                f"expected_n_items={expected_n_items}",
                f"actual_n_items={payload_n_items}",
            ],
        )

    matrices = payload.get("matrices")
    if not isinstance(matrices, dict) or not matrices:
        raise _strict_runtime_error(
            "Precomputed matrix payload has no matrices.",
            [f"path={source_path}"],
        )

    for user_id, matrix in matrices.items():
        shape = getattr(matrix, "shape", None)
        if not shape or len(shape) != 2:
            raise _strict_runtime_error(
                "Invalid user matrix shape.",
                [f"path={source_path}", f"user={user_id}", f"shape={shape}"],
            )
        if int(shape[1]) != payload_n_items:
            raise _strict_runtime_error(
                "User matrix shape mismatch.",
                [
                    f"path={source_path}",
                    f"user={user_id}",
                    f"expected_second_dim={payload_n_items}",
                    f"actual_shape={shape}",
                ],
            )


def _validate_item2index_mapping(
    item2index: Any,
    *,
    source_path: Path,
    expected_n_items: int,
) -> None:
    """Validate strict run-scoped item2index integrity.

    Contract:
    - mapping size equals expected_n_items
    - values are integer indices
    - indices are unique
    - index range is contiguous [0, expected_n_items - 1]
    """
    if not isinstance(item2index, dict) or not item2index:
        raise _strict_runtime_error(
            "Run-scoped item2index mapping is empty or invalid.",
            [f"path={source_path}"],
        )

    if len(item2index) != expected_n_items:
        raise _strict_runtime_error(
            "Run-scoped item2index mapping size mismatch.",
            [
                f"path={source_path}",
                f"expected_n_items={expected_n_items}",
                f"actual_mapping_size={len(item2index)}",
            ],
        )

    normalized_indices: List[int] = []
    non_int_examples: List[str] = []
    for business_id, raw_idx in item2index.items():
        try:
            idx = int(raw_idx)
            normalized_indices.append(idx)
        except (TypeError, ValueError):
            non_int_examples.append(f"{business_id}:{raw_idx}")
            if len(non_int_examples) >= 5:
                break

    if non_int_examples:
        raise _strict_runtime_error(
            "Run-scoped item2index has non-integer indices.",
            [
                f"path={source_path}",
                f"examples={non_int_examples}",
            ],
        )

    unique_count = len(set(normalized_indices))
    if unique_count != len(normalized_indices):
        raise _strict_runtime_error(
            "Run-scoped item2index has duplicate index values.",
            [
                f"path={source_path}",
                f"mapping_size={len(normalized_indices)}",
                f"unique_indices={unique_count}",
            ],
        )

    min_idx = min(normalized_indices)
    max_idx = max(normalized_indices)
    expected_max = expected_n_items - 1
    if min_idx != 0 or max_idx != expected_max:
        raise _strict_runtime_error(
            "Run-scoped item2index index range mismatch.",
            [
                f"path={source_path}",
                f"expected_min=0",
                f"expected_max={expected_max}",
                f"actual_min={min_idx}",
                f"actual_max={max_idx}",
            ],
        )


def load_run_artifact_bundle(selected_output_dir: Optional[str]) -> Dict[str, Any]:
    """Resolve and validate strict run-scoped artifacts (local or downloaded from GCS)."""
    if not selected_output_dir:
        raise _strict_runtime_error(
            "Strict best-run mode requires an explicit selected run.",
            ["selected_output_dir is empty"],
        )

    run_id = _extract_run_timestamp(selected_output_dir)
    if not run_id:
        run_id = Path(str(selected_output_dir)).name
    if not run_id:
        raise _strict_runtime_error(
            "Could not resolve run_id from selected output directory.",
            [f"selected_output_dir={selected_output_dir}"],
        )

    run_dir = Path(str(selected_output_dir))
    if not run_dir.exists():
        cloud_storage = _get_cloud_storage_helper()
        if not cloud_storage:
            raise _strict_runtime_error(
                "Run directory is missing and GCS is not configured.",
                [
                    f"selected_output_dir={selected_output_dir}",
                    "GCS_BUCKET_NAME is not set",
                ],
            )

        gcs_bases = [f"outputs/{run_id}"]
        required_files = [
            "summary.json",
            "mappings/item2index.pkl",
            "precomputed/user_csr_matrices.pkl",
            "data/test_users_top50.json",
            "neuron_coactivation.json",
            "neuron_category_metadata.json",
            "checkpoints/elsa_best.pt",
        ]
        missing_by_prefix: List[str] = []
        hydrated_run_dir: Optional[Path] = None
        for gcs_base in gcs_bases:
            temp_root = (
                Path(tempfile.mkdtemp(prefix=f"diplomov_run_{run_id}_")) / run_id
            )
            temp_root.mkdir(parents=True, exist_ok=True)
            missing = []
            for relative_path in required_files:
                ok = _download_gcs_file(
                    cloud_storage,
                    f"{gcs_base}/{relative_path}",
                    temp_root / relative_path,
                )
                if not ok:
                    missing.append(
                        f"gs://{cloud_storage.bucket_name}/{gcs_base}/{relative_path}"
                    )

            checkpoints_prefix = f"{gcs_base}/checkpoints/"
            checkpoints_ok = _download_gcs_prefix(
                cloud_storage,
                checkpoints_prefix,
                temp_root / "checkpoints",
            )
            labels_ok = _download_gcs_prefix(
                cloud_storage,
                f"{gcs_base}/neuron_interpretations/",
                temp_root / "neuron_interpretations",
            )
            if not checkpoints_ok:
                missing.append(f"gs://{cloud_storage.bucket_name}/{checkpoints_prefix}")
            if not labels_ok:
                missing.append(
                    f"gs://{cloud_storage.bucket_name}/{gcs_base}/neuron_interpretations/"
                )

            if not missing:
                hydrated_run_dir = temp_root
                break

            missing_by_prefix.append(f"prefix={gcs_base} missing_count={len(missing)}")

        if not hydrated_run_dir:
            raise _strict_runtime_error(
                "Missing required run artifacts in cloud storage.",
                missing_by_prefix,
            )

        run_dir = hydrated_run_dir

    issues: List[str] = []
    required_local_paths = [
        run_dir / "summary.json",
        run_dir / "mappings" / "item2index.pkl",
        run_dir / "precomputed" / "user_csr_matrices.pkl",
        run_dir / "data" / "test_users_top50.json",
        run_dir / "checkpoints" / "elsa_best.pt",
        run_dir / "neuron_coactivation.json",
        run_dir / "neuron_category_metadata.json",
    ]
    for required_path in required_local_paths:
        if not required_path.exists():
            issues.append(f"missing_path={required_path}")

    sae_candidates = sorted((run_dir / "checkpoints").glob("sae_r*_k*_best.pt"))
    if not sae_candidates:
        fallback_sae = run_dir / "checkpoints" / "sae_best.pt"
        if fallback_sae.exists():
            sae_candidates = [fallback_sae]
    if not sae_candidates:
        issues.append(
            f"missing_path={run_dir / 'checkpoints'} (expected sae_r*_k*_best.pt or sae_best.pt)"
        )

    labels_dir = run_dir / "neuron_interpretations"
    if not labels_dir.exists():
        issues.append(f"missing_path={labels_dir}")
    else:
        label_files = list(labels_dir.glob("labels_*.pkl"))
        if not label_files:
            issues.append(f"missing_labels={labels_dir} (expected labels_*.pkl)")

    if issues:
        raise _strict_runtime_error(
            "Strict best-run artifact audit failed.",
            issues,
        )

    summary = _load_json_file(run_dir / "summary.json")
    try:
        expected_n_items = int((summary.get("data") or {}).get("n_items"))
    except (TypeError, ValueError):
        raise _strict_runtime_error(
            "Run summary is missing valid data.n_items metadata.",
            [
                f"path={run_dir / 'summary.json'}",
                f"value={(summary.get('data') or {}).get('n_items')}",
            ],
        ) from None

    try:
        with (run_dir / "mappings" / "item2index.pkl").open("rb") as f:
            item2index = pickle.load(f)
    except Exception as e:
        raise _strict_runtime_error(
            "Failed to load run-scoped item2index mapping.",
            [f"path={run_dir / 'mappings' / 'item2index.pkl'}", f"error={e}"],
        ) from e

    _validate_item2index_mapping(
        item2index,
        source_path=run_dir / "mappings" / "item2index.pkl",
        expected_n_items=expected_n_items,
    )

    precomputed_path = run_dir / "precomputed" / "user_csr_matrices.pkl"
    try:
        with precomputed_path.open("rb") as f:
            precomputed_payload = pickle.load(f)
    except Exception as e:
        raise _strict_runtime_error(
            "Failed to load precomputed user matrices.",
            [f"path={precomputed_path}", f"error={e}"],
        ) from e

    # _validate_precomputed_matrices_payload(
    #     precomputed_payload,
    #     run_id=run_id,
    #     expected_n_items=expected_n_items,
    #     source_path=precomputed_path,
    # )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": summary,
        "expected_n_items": expected_n_items,
        "sae_checkpoint": str(sae_candidates[0]),
    }


def _download_gcs_file(cloud_storage, gcs_path: str, local_path: Path) -> bool:
    """Download one GCS object to a local path."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = cloud_storage.bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))
        return local_path.exists()
    except Exception as e:
        logger.debug(
            f"Failed to download gs://{cloud_storage.bucket_name}/{gcs_path}: {e}"
        )
        return False


def _download_gcs_prefix(cloud_storage, gcs_prefix: str, local_root: Path) -> bool:
    """Download all objects under a GCS prefix into a local directory."""
    try:
        downloaded = False
        for gcs_path in cloud_storage.list_files(prefix=gcs_prefix):
            if not gcs_path.startswith(gcs_prefix):
                continue
            relative_path = gcs_path[len(gcs_prefix) :].lstrip("/")
            if not relative_path:
                continue
            destination = local_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob = cloud_storage.bucket.blob(gcs_path)
            blob.download_to_filename(str(destination))
            downloaded = True
        return downloaded
    except Exception as e:
        logger.debug(
            f"Failed to download GCS prefix gs://{cloud_storage.bucket_name}/{gcs_prefix}: {e}"
        )
        return False


def validate_cloud_run_artifacts(selected_output_dir: Optional[str]) -> Dict[str, Any]:
    """Validate presence of required strict-run artifacts in GCS (non-mutating)."""
    run_id = (
        _extract_run_timestamp(selected_output_dir)
        or Path(str(selected_output_dir or "")).name
    )
    if not run_id:
        return {"status": "invalid", "missing": ["run_id could not be resolved"]}

    cloud_storage = _get_cloud_storage_helper()
    if not cloud_storage:
        return {"status": "skipped", "run_id": run_id, "reason": "GCS not configured"}

    required_relative_paths = [
        "summary.json",
        "mappings/item2index.pkl",
        "precomputed/user_csr_matrices.pkl",
        "data/test_users_top50.json",
        "checkpoints/elsa_best.pt",
        "neuron_coactivation.json",
        "neuron_category_metadata.json",
    ]
    for gcs_base in [f"outputs/{run_id}"]:
        missing: List[str] = []
        present: List[str] = []
        for relative_path in required_relative_paths:
            path = f"{gcs_base}/{relative_path}"
            try:
                if cloud_storage.exists(path):
                    present.append(path)
                else:
                    missing.append(path)
            except Exception:
                missing.append(path)

        try:
            label_files = [
                p
                for p in cloud_storage.list_files(
                    prefix=f"{gcs_base}/neuron_interpretations/"
                )
                if p.endswith(".pkl") and Path(p).name.startswith("labels_")
            ]
        except Exception:
            label_files = []
        if not label_files:
            missing.append(f"{gcs_base}/neuron_interpretations/labels_*.pkl")

        if not missing:
            logger.info(
                "Cloud strict-run artifact check (%s): complete (prefix=%s)",
                run_id,
                gcs_base,
            )
            return {
                "status": "ok",
                "run_id": run_id,
                "bucket": cloud_storage.bucket_name,
                "missing": [],
                "present_count": len(present),
                "resolved_prefix": gcs_base,
            }

    logger.warning("Cloud strict-run artifact check (%s): incomplete", run_id)
    return {
        "status": "missing",
        "run_id": run_id,
        "bucket": cloud_storage.bucket_name,
        "missing": ["Required artifacts not complete in outputs/"],
        "present_count": 0,
    }


def _has_all_required_artifacts(run_dir: Path) -> bool:
    """Check if a run directory has all required strict-mode artifacts."""
    required_paths = [
        run_dir / "summary.json",
        run_dir / "mappings" / "item2index.pkl",
        run_dir / "precomputed" / "user_csr_matrices.pkl",
        run_dir / "data" / "test_users_top50.json",
        run_dir / "checkpoints" / "elsa_best.pt",
        run_dir / "neuron_coactivation.json",
        run_dir / "neuron_category_metadata.json",
    ]
    for required_path in required_paths:
        if not required_path.exists():
            return False

    # Check for SAE checkpoint
    sae_candidates = sorted((run_dir / "checkpoints").glob("sae_r*_k*_best.pt"))
    if not sae_candidates:
        fallback_sae = run_dir / "checkpoints" / "sae_best.pt"
        if not fallback_sae.exists():
            return False

    # Check for neuron interpretation labels
    labels_dir = run_dir / "neuron_interpretations"
    if not labels_dir.exists() or not labels_dir.is_dir():
        return False
    label_files = list(labels_dir.glob("labels_*.pkl"))
    if not label_files:
        return False

    return True


def _has_all_required_gcs_artifacts(cloud_storage, run_id: str) -> bool:
    if not run_id:
        return False

    gcs_base = f"outputs/{run_id}"

    file_paths = [
        "summary.json",
        "mappings/item2index.pkl",
        "precomputed/user_csr_matrices.pkl",
        "data/test_users_top50.json",
        "checkpoints/elsa_best.pt",
        "neuron_coactivation.json",
        "neuron_category_metadata.json",
    ]

    try:
        results = []

        # 🔹 check files
        for rel in file_paths:
            full_path = f"{gcs_base}/{rel}"
            exists = cloud_storage.exists(full_path)
            logger.error(f"FILE {full_path}: {exists}")
            results.append(exists)

        # 🔹 check checkpoints folder (must contain at least something)
        checkpoints_prefix = f"{gcs_base}/checkpoints/"
        has_checkpoints = False
        for path in cloud_storage.list_files(prefix=checkpoints_prefix):
            has_checkpoints = True
            break
        logger.error(f"DIR {checkpoints_prefix}: {has_checkpoints}")
        results.append(has_checkpoints)

        # 🔹 check neuron_interpretations folder (must contain labels_*.pkl)
        labels_prefix = f"{gcs_base}/neuron_interpretations/"
        has_labels = False
        for path in cloud_storage.list_files(prefix=labels_prefix):
            if path.endswith(".pkl") and "labels_" in path:
                has_labels = True
                break
        logger.error(f"DIR {labels_prefix}: {has_labels}")
        results.append(has_labels)

        return all(results)

    except Exception as e:
        logger.error(f"GCS artifact check failed: {e}")
        return False


def _build_experiment_results(
    manifest: dict,
    *,
    source: str,
    experiment_dir: Path,
    cloud_storage=None,
) -> Optional[Dict]:
    """Normalize an experiment manifest into the structure expected by the UI.

    Skips runs that don't have all required artifacts for strict mode.
    """

    def _ndcg_at_20(summary: Optional[dict]) -> float:
        ranking_metrics = (
            summary.get("ranking_metrics_sae", {}) if isinstance(summary, dict) else {}
        )
        try:
            return float(ranking_metrics.get("ndcg", {}).get("@20", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    runs = []

    for raw_run in manifest.get("runs", []):
        run = dict(raw_run)
        summary = run.get("summary")
        logger.error(f"RUN RAW: {run.get('run_dir')}")
        run_id = _extract_run_timestamp(run.get("run_dir"))
        logger.error(f"RUN ID: {run_id}")
        if not summary:
            summary_path_str = run.get("summary_path")
            if summary_path_str:
                summary_path = Path(summary_path_str)
                if summary_path.exists():
                    try:
                        with summary_path.open("r", encoding="utf-8") as f:
                            summary = json.load(f)
                    except Exception as e:
                        logger.debug(
                            "Could not load run summary from %s: %s", summary_path, e
                        )

        run["summary"] = summary
        run["ndcg_at_20"] = _ndcg_at_20(summary)

        # Skip runs that don't have all required artifacts
        if run_id:
            if cloud_storage is not None:
                if not _has_all_required_gcs_artifacts(cloud_storage, str(run_id)):
                    logger.debug(
                        "Incomplete GCS run (missing artifacts): %s",
                        run_id,
                    )

            else:
                run_dir = Path(run_id)
                if not _has_all_required_artifacts(run_dir):
                    logger.debug("Incomplete run (missing artifacts): %s", run_dir)

        runs.append(run)

    if not runs:
        logger.warning("No runs found in the manifest.")
        return None

    runs = [run for run in runs if math.isfinite(run.get("ndcg_at_20", float("-inf")))]
    if not runs:
        logger.warning("Could not create the run list.")
        return None

    runs = sorted(
        runs, key=lambda run: run.get("ndcg_at_20", float("-inf")), reverse=True
    )
    best_run = runs[0]
    for idx, run in enumerate(runs):
        run["is_best_run"] = idx == 0

    selected_summary = best_run.get("summary") or {}
    return {
        "summary": selected_summary,
        "ranking_metrics": selected_summary.get("ranking_metrics_sae"),
        "source": source,
        "default_run_dir": best_run.get("run_dir"),
        "experiment": {
            "experiment_id": manifest.get("experiment_id"),
            "created": manifest.get("created"),
            "source_config": manifest.get("source_config"),
            "base_config": manifest.get("base_config"),
            "experiment_dir": str(experiment_dir),
            "manifest": manifest,
            "best_run_dir": best_run.get("run_dir"),
            "selection_metric": "ndcg@20",
        },
        "runs": runs,
        "experiment_runs": runs,
    }


def _load_local_experiment_results(outputs_base: Path) -> Dict:
    candidates = _list_local_experiment_dirs(outputs_base)
    if not candidates:
        raise _strict_runtime_error(
            "No experiment directories found for strict best-run mode.",
            [f"experiments_dir={outputs_base / 'experiments'}"],
        )

    issues: List[str] = []
    for idx, experiment_dir in enumerate(candidates):
        manifest_path = experiment_dir / "manifest.json"
        if not manifest_path.exists():
            issues.append(
                f"missing_manifest experiment_dir={experiment_dir} path={manifest_path}"
            )
            continue

        try:
            manifest = _load_json_file(manifest_path)
        except Exception as e:
            issues.append(f"invalid_manifest experiment_dir={experiment_dir} error={e}")
            continue

        results = _build_experiment_results(
            manifest,
            source="Local Experiment",
            experiment_dir=experiment_dir,
        )
        if not results:
            issues.append(f"no_usable_runs manifest={manifest_path}")
            continue

        if idx > 0:
            latest = candidates[0]
            results["startup_notice"] = (
                "Latest experiment is not yet usable (likely still running). "
                f"Falling back to previous completed experiment: {experiment_dir.name} "
                f"(latest candidate was {latest.name})."
            )
            logger.warning(results["startup_notice"])

        logger.info("✓ Loaded experiment manifest from local: %s", experiment_dir)
        return results

    raise _strict_runtime_error(
        "No usable completed runs found in local experiment manifests.",
        issues[:20],
    )


def _load_gcs_experiment_results() -> Optional[Dict]:
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        return None

    try:
        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)
        blobs = cloud_storage.bucket.list_blobs(prefix="experiments/")
        experiment_ids = set()

        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) >= 3 and parts[2] == "manifest.json" and len(parts[1]) == 15:
                experiment_ids.add(parts[1])

        if not experiment_ids:
            raise _strict_runtime_error(
                "No experiment manifests found in GCS (strict mode).",
                [f"bucket={gcs_bucket_name}", "prefix=experiments/"],
            )

        sorted_ids = sorted(experiment_ids, reverse=True)
        issues: List[str] = []
        for idx, experiment_id in enumerate(sorted_ids):
            try:
                manifest = cloud_storage.read_json(
                    f"experiments/{experiment_id}/manifest.json"
                )
            except Exception as e:
                issues.append(f"read_error experiment_id={experiment_id} error={e}")
                continue

            if not isinstance(manifest, dict):
                issues.append(f"invalid_payload experiment_id={experiment_id}")
                continue

            results = _build_experiment_results(
                manifest,
                source="GCS Experiment",
                experiment_dir=Path(f"experiments/{experiment_id}"),
                cloud_storage=cloud_storage,
            )
            if not results:
                issues.append(f"no_usable_runs experiment_id={experiment_id}")
                continue

            if idx > 0:
                latest = sorted_ids[0]
                results["startup_notice"] = (
                    "Latest GCS experiment is not yet usable (likely still running). "
                    f"Falling back to previous completed experiment: {experiment_id} "
                    f"(latest candidate was {latest})."
                )
                logger.warning(results["startup_notice"])

            logger.info("Loaded experiment manifest from GCS: %s", experiment_id)
            return results

        raise _strict_runtime_error(
            "No usable completed runs found in GCS experiment manifests.",
            [f"bucket={gcs_bucket_name}", *issues[:20]],
        )
    except Exception as e:
        raise _strict_runtime_error(
            "Failed to load latest experiment manifest from GCS.",
            [f"bucket={gcs_bucket_name}", f"error={e}"],
        ) from e


def _load_summary_from_run_dir(run_dir: Path) -> Optional[Dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary_data = json.load(f)
        return {
            "summary": summary_data,
            "ranking_metrics": summary_data.get("ranking_metrics"),
            "source": f"Local Run: {run_dir.name}",
            "run_dir": str(run_dir),
        }
    except Exception as e:
        logger.warning("Failed to load run summary %s: %s", summary_path, e)
        return None


# Use streamlit's cache decorator if available, otherwise use dummy
if HAS_STREAMLIT:
    st_cache_resource = st.cache_resource
else:
    st_cache_resource = cache_resource


@st_cache_resource
def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML and flatten for UI services.

    Note: Cache is based on function signature only (not file mtime).
    To invalidate on config changes, restart the Streamlit app.
    """
    config_path = Path(config_path)
    project_root = config_path.parent.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    def _resolve_to_project_root(path_value: str) -> str:
        """Resolve path values relative to project root unless already absolute."""
        if not path_value:
            return ""
        candidate = Path(path_value)
        if candidate.is_absolute():
            return str(candidate)
        return str((project_root / candidate).resolve())

    # Flatten nested config structure for UI services
    config = {}
    config["project_root"] = str(project_root.resolve())
    config["config_path"] = str(config_path.resolve())

    # Data paths (from data section)
    if "data" in raw_config:
        config["duckdb_path"] = _resolve_to_project_root(
            raw_config["data"].get("db_path", "")
        )
        config["preprocess_dir"] = _resolve_to_project_root("data/preprocessed_yelp")

    # ELSA hyperparameters
    if "elsa" in raw_config:
        config["latent_dim"] = raw_config["elsa"].get("latent_dim", 512)
        config["device"] = raw_config["elsa"].get("device", "cpu")

    # SAE hyperparameters (k = sparsity level)
    if "sae" in raw_config:
        config["k"] = raw_config["sae"].get("k", 32)
        config["width_ratio"] = raw_config["sae"].get("width_ratio", 4)

    # Output & steering defaults
    config["steering_alpha"] = 0.3  # Default steering interpolation
    checkpoint_dir = raw_config.get("model", {}).get(
        "checkpoint_dir",
        raw_config.get("output", {}).get("base_dir", "outputs"),
    )
    config["model_checkpoint_dir"] = _resolve_to_project_root(checkpoint_dir)
    config["neuron_labels_path"] = _resolve_to_project_root(
        "outputs/neuron_labels.json"
    )

    # Compute n_items from database (apply same filters as training)
    # NOTE: This is informational only; the inference service reads n_items from checkpoint metadata
    config["n_items"] = None  # Will be read from checkpoint by inference service

    try:
        from src.ui.services.secrets_helper import get_cloudsql_config

        cloudsql_cfg = get_cloudsql_config()
        state_filter = raw_config.get("data", {}).get("state_filter")

        # Try Cloud SQL first
        if all(
            [
                cloudsql_cfg.get("instance"),
                cloudsql_cfg.get("database"),
                cloudsql_cfg.get("user"),
                cloudsql_cfg.get("password"),
            ]
        ):
            try:
                from src.ui.services.cloud_sql_helper import CloudSQLHelper

                sql_helper = CloudSQLHelper(
                    instance_connection_name=cloudsql_cfg["instance"],
                    database=cloudsql_cfg["database"],
                    user=cloudsql_cfg["user"],
                    password=cloudsql_cfg["password"],
                )
                with sql_helper.engine.connect() as conn:
                    if state_filter:
                        query = f"SELECT COUNT(*) FROM review WHERE state = '{state_filter}'"
                        logger.info(
                            f"   Counting items from Cloud SQL with state_filter='{state_filter}'..."
                        )
                    else:
                        query = "SELECT COUNT(*) FROM review"
                        logger.info(
                            "   Counting all items from Cloud SQL (no state filter)..."
                        )

                    result = conn.execute(query).scalar()
                    config["n_items"] = result if result else None
                    logger.info(
                        f"   Found {config['n_items']} items in Cloud SQL dataset"
                    )
            except Exception as e:
                logger.debug(f"Cloud SQL count failed: {e}, trying local DuckDB...")
                config["n_items"] = None

        # Fall back to local DuckDB if Cloud SQL unavailable
        if config["n_items"] is None:
            duckdb_path = Path(config["duckdb_path"])
            if duckdb_path.exists():
                import duckdb

                conn = duckdb.connect(str(duckdb_path))
                try:
                    if state_filter:
                        query = f"SELECT COUNT(*) FROM review WHERE state = '{state_filter}'"
                        logger.info(
                            f"   Counting items from DuckDB with state_filter='{state_filter}'..."
                        )
                    else:
                        query = "SELECT COUNT(*) FROM review"
                        logger.info(
                            "   Counting all items from DuckDB (no state filter)..."
                        )

                    result = conn.execute(query).fetchall()
                    config["n_items"] = result[0][0] if result else None
                    logger.info(f"   Found {config['n_items']} items in DuckDB dataset")
                except Exception as e:
                    logger.debug(f"Local DuckDB count failed: {e}")
                    config["n_items"] = None
                finally:
                    conn.close()
            else:
                logger.debug(
                    f"DuckDB not found at {duckdb_path}, will use checkpoint metadata"
                )

    except Exception as e:
        logger.debug(f"Could not count items from database: {e}")
        config["n_items"] = None  # Will be read from checkpoint by inference service

    # Include state_filter in config for DataService
    config["state_filter"] = raw_config.get("data", {}).get("state_filter")

    logger.info(f"✓ Loaded config from {config_path}")
    logger.info(
        f"   Device: {config['device']}, Latent dim: {config['latent_dim']}, SAE k: {config['k']}"
    )
    if config.get("state_filter"):
        logger.info(f"   State filter: {config['state_filter']}")
    return config


@st_cache_resource
def load_inference_service(
    config: Dict,
    selected_output_dir: Optional[str] = None,
) -> InferenceService:
    """
    Load ELSA+SAE models once per session.

    Streamlit will call this once and reuse result across page refreshes.

    Model metadata (n_items, latent_dim, k, width_ratio) is read from checkpoint
    files, NOT from config. This ensures consistency regardless of how data is
    filtered or configured on the inference machine.
    """
    bundle = load_run_artifact_bundle(selected_output_dir)
    run_dir = Path(bundle["run_dir"])
    checkpoint_dir = run_dir / "checkpoints"

    elsa_ckpt = checkpoint_dir / "elsa_best.pt"
    sae_ckpt = Path(bundle["sae_checkpoint"])

    logger.info("Loading strict best-run checkpoints from %s", checkpoint_dir)
    logger.info("Loading ELSA from %s", elsa_ckpt)
    logger.info("Loading SAE from %s", sae_ckpt)

    service = InferenceService(
        elsa_ckpt, sae_ckpt, config, labels=None, data_service=None
    )
    if HAS_STREAMLIT:
        if not hasattr(st.session_state, "_startup_diagnostics"):
            st.session_state._startup_diagnostics = {}
        st.session_state._startup_diagnostics["models_loaded"] = True
        st.session_state._startup_diagnostics["run_id"] = bundle.get("run_id")
        st.session_state._startup_diagnostics["n_items"] = service.n_items
    logger.info("Models loaded successfully")
    return service


@st_cache_resource
def load_data_service(
    config: Dict,
    selected_output_dir: Optional[str] = None,
    expected_n_items: Optional[int] = None,
):
    """Load strict run-scoped POI data service."""
    logger.info("Initializing Data Service (strict best-run mode)")
    bundle = load_run_artifact_bundle(selected_output_dir)
    active_run_dir = Path(bundle["run_dir"])
    item2index_path = active_run_dir / "mappings" / "item2index.pkl"
    if not item2index_path.exists():
        raise _strict_runtime_error(
            "Strict best-run mode requires run-scoped mapping artifact.",
            [f"missing_path={item2index_path}"],
        )

    duckdb_path = Path(config["duckdb_path"])
    data_available_locally = duckdb_path.exists()

    project_root = Path(__file__).parent.parent.parent
    photo_candidates = [
        project_root / "yelp_photos",
        project_root / "Yelp-Photos",
    ]
    local_photos_path = next((p for p in photo_candidates if p.exists()), None)

    from src.ui.services.secrets_helper import get_cloudsql_config

    cloudsql_config = get_cloudsql_config()
    cloudsql_available = all(
        [
            cloudsql_config.get("instance"),
            cloudsql_config.get("database"),
            cloudsql_config.get("user"),
            cloudsql_config.get("password"),
        ]
    )

    if not data_available_locally and not cloudsql_available:
        raise _strict_runtime_error(
            "Data backend not available.",
            [
                f"duckdb_missing={duckdb_path}",
                "cloudsql_credentials=missing",
            ],
        )

    service = DataService(
        duckdb_path=duckdb_path,
        config=config,
        item2index_path=item2index_path,
        local_photos_dir=local_photos_path,
        active_run_dir=active_run_dir,
        expected_n_items=expected_n_items or bundle.get("expected_n_items"),
        strict_run_artifacts=True,
    )

    backend_info = getattr(service, "backend_type", "unknown")
    if HAS_STREAMLIT:
        if not hasattr(st.session_state, "_startup_diagnostics"):
            st.session_state._startup_diagnostics = {}
        st.session_state._startup_diagnostics["backend"] = (
            "Cloud SQL"
            if backend_info == "cloudsql"
            else f"DuckDB ({service.num_pois} POIs)"
        )

    logger.info("Data Service ready (%s backend)", backend_info)
    return service


@st_cache_resource
def get_precomputed_cache_dir() -> Optional[Path]:
    """Detect if precomputed UI cache exists and return its path.

    Looks for precomputed_ui_cache/neuron_wordclouds/ in the latest outputs/*/ directories.
    Returns the neuron_wordclouds subdirectory or None if not found (app will compute on-demand).
    """
    project_root = Path(__file__).parent.parent.parent
    outputs_dir = project_root / "outputs"

    if not outputs_dir.exists():
        return None

    # Find all output directories, sorted by modification time
    import os

    output_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir()],
        key=lambda d: os.path.getmtime(d),
        reverse=True,
    )

    for output_dir in output_dirs:
        cache_dir = output_dir / "precomputed_ui_cache" / "neuron_wordclouds"
        if cache_dir.exists():
            logger.info(f"✓ Found precomputed cache at {cache_dir}")
            return cache_dir

    logger.info("No precomputed cache found (app will compute on-demand)")
    return None


@st_cache_resource
def load_labeling_service(
    config: Dict,
    _data_service=None,
    selected_output_dir: Optional[str] = None,
) -> LabelingService:
    """Load strict run-scoped neuron labeling artifacts.

    Notes:
    - `_data_service` intentionally starts with underscore so Streamlit's
      cache key builder does not try to hash the live DataService object.
    """
    bundle = load_run_artifact_bundle(selected_output_dir)
    output_dir = Path(bundle["run_dir"])
    interpretations_dir = output_dir / "neuron_interpretations"
    if not interpretations_dir.exists():
        raise _strict_runtime_error(
            "Strict best-run mode requires neuron_interpretations artifacts.",
            [f"missing_path={interpretations_dir}"],
        )

    method_files = sorted(interpretations_dir.glob("labels_*.pkl"))
    if not method_files:
        raise _strict_runtime_error(
            "Missing run-scoped label files.",
            [f"expected_pattern={interpretations_dir / 'labels_*.pkl'}"],
        )

    service = LabelingService(
        labels_json_path=interpretations_dir,
        config=config,
        data_service=_data_service,
    )
    return service


@st_cache_resource
def load_wordcloud_service(
    config: Dict,
    selected_output_dir: Optional[str] = None,
) -> "WordcloudService":
    """Load wordcloud service from strict run-scoped artifacts."""
    try:
        from src.ui.services import WordcloudService
    except ImportError:
        logger.error("WordcloudService not available")
        return None

    bundle = load_run_artifact_bundle(selected_output_dir)
    output_dir = Path(bundle["run_dir"])
    labels_path = output_dir / "neuron_interpretations"
    metadata_path = output_dir / "neuron_category_metadata.json"

    if not metadata_path.exists():
        raise _strict_runtime_error(
            "Missing run-scoped neuron category metadata.",
            [f"missing_path={metadata_path}"],
        )

    service = WordcloudService(
        category_metadata_path=metadata_path,
        labels_path=labels_path if labels_path.exists() else None,
    )
    logger.info("Wordcloud service initialized")
    return service


@st_cache_resource
def load_coactivation_service(
    config: Dict,
    selected_output_dir: Optional[str] = None,
) -> Optional["CoactivationService"]:
    """Load strict run-scoped co-activation service."""
    bundle = load_run_artifact_bundle(selected_output_dir)
    output_dir = Path(bundle["run_dir"])
    coactivation_path = output_dir / "neuron_coactivation.json"

    if not coactivation_path.exists():
        raise _strict_runtime_error(
            "Missing run-scoped coactivation artifact.",
            [f"missing_path={coactivation_path}"],
        )

    service = CoactivationService(coactivation_path=coactivation_path)
    logger.info("Co-activation service initialized")
    return service


@st_cache_resource
def load_training_results(
    config: Dict,
    selected_output_dir: Optional[str] = None,
) -> Optional[Dict]:
    """Load latest-experiment results and block startup on contract failure."""
    local_error: Optional[Exception] = None
    gcs_error: Optional[Exception] = None
    try:
        gcs_results = _load_gcs_experiment_results()
        if gcs_results:
            return gcs_results
    except Exception as e:
        gcs_error = e
        logger.warning("GCS strict experiment loading failed: %s", e)
    try:
        local_results = _load_local_experiment_results(Path("outputs"))
        if local_results:
            return local_results
    except Exception as e:
        local_error = e
        logger.warning("Local strict experiment loading failed: %s", e)

    issues = [
        "Strict best-run mode could not resolve the latest experiment manifest.",
    ]
    if local_error:
        issues.append(f"local_error={local_error}")
    if gcs_error:
        issues.append(f"gcs_error={gcs_error}")
    raise _strict_runtime_error(
        "Startup blocked: strict best-run contract failed.",
        issues,
    )


@st_cache_resource
def load_semantic_search_model():
    """Load sentence transformer model for semantic search (cached at startup).

    This model is cached as a resource, so it's loaded once at startup
    and reused for all semantic search queries. Uses the lightweight
    all-MiniLM-L6-v2 model (~22MB) for fast inference.

    Returns:
        SentenceTransformer model or None if loading fails
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✓ Semantic search model loaded and cached")
        return model
    except Exception as e:
        logger.warning(f"Could not load semantic search model: {e}")
        return None


def cache_all_label_embeddings(labels_service, max_neuron: int):
    """Cache all label embeddings in session state for efficient semantic search.

    Encodes all feature labels once using batch encoding and stores them
    in session_state to avoid re-encoding on every search.

    Args:
        labels_service: LabelingService for fetching labels
        max_neuron: Maximum neuron index (0 to max_neuron inclusive)
    """
    cache_key = "all_label_embeddings"

    # Return cached version if available
    if cache_key in st.session_state and st.session_state[cache_key] is not None:
        logger.info(
            f"✓ Using cached label embeddings: {len(st.session_state[cache_key])} embeddings"
        )
        return st.session_state[cache_key]

    try:
        semantic_model = load_semantic_search_model()
        if semantic_model is None:
            logger.warning("Cannot cache embeddings: semantic model is None")
            return None

        logger.info(
            f"Encoding all {max_neuron + 1} feature labels for semantic search..."
        )

        # Get all labels and their indices
        all_labels = []
        label_indices = []
        for idx in range(max_neuron + 1):
            try:
                label = labels_service.get_label(idx)
                all_labels.append(label)
                label_indices.append(idx)
            except Exception as e:
                logger.debug(f"Failed to get label for {idx}: {e}")

        logger.info(f"Found {len(all_labels)} labels to encode")

        if not all_labels:
            logger.warning("No labels found for embedding")
            return None

        # Batch encode all labels at once (much faster than one-by-one)
        logger.info(f"Batch encoding {len(all_labels)} labels...")
        label_embeddings = semantic_model.encode(all_labels, show_progress_bar=False)
        logger.info(
            f"Batch encoding complete. Shape: {label_embeddings.shape}, dtype: {label_embeddings.dtype}"
        )

        # Create dictionary mapping label index to embedding
        # Keep as numpy arrays for consistency and easy computation
        embeddings_dict = {
            idx: embedding for idx, embedding in zip(label_indices, label_embeddings)
        }

        # Cache in session state
        st.session_state[cache_key] = embeddings_dict
        logger.info(
            f"✓ Cached {len(embeddings_dict)} label embeddings in session state"
        )
        return embeddings_dict

    except Exception as e:
        logger.error(f"Failed to cache label embeddings: {e}", exc_info=True)
        return None


def init_session_state():
    """
    Initialize Streamlit session state.

    Called once per session to set up variables persisted across reruns.
    """
    if not HAS_STREAMLIT:
        logger.debug("Streamlit not available, skipping session state init")
        return

    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = None

    if "current_recommendations" not in st.session_state:
        st.session_state.current_recommendations = []

    if "steering_modified" not in st.session_state:
        st.session_state.steering_modified = False

    if "user_history" not in st.session_state:
        st.session_state.user_history = {}


def get_services() -> tuple:
    """
    Retrieve cached services from session state.

    Must be called after main.py initializes them.

    Returns:
        (inference, data, labels)
    """
    if not HAS_STREAMLIT:
        raise RuntimeError(
            "Streamlit not available - cannot retrieve services from session"
        )

    inference = st.session_state.get("inference")
    data = st.session_state.get("data")
    labels = st.session_state.get("labels")

    if not all([inference, data, labels]):
        st.error("Services not initialized. Check main.py setup.")
        st.stop()

    return inference, data, labels
