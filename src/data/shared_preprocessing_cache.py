from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix

from src.data.preprocessing import load_dataset, save_dataset
from src.data.yelp_loader import load_businesses, load_reviews
from src.data.preprocessing import build_csr

logger = logging.getLogger(__name__)

PREPROCESSING_CACHE_VERSION = 2
PREPROCESSING_K_CORE = 5


def _apply_kcore_filtering_to_reviews(
    reviews: pd.DataFrame,
    *,
    k: int = PREPROCESSING_K_CORE,
    max_iterations: int = 10,
) -> pd.DataFrame:
    """Iteratively keep only user-item pairs that satisfy k-core filtering.

    This preserves a concrete filtered interaction table, which lets us build a
    final CSR matrix whose row ordering and saved user IDs come from the exact
    same dataset rather than reconstructing them afterward.
    """
    filtered = reviews[["user_id", "business_id"]].dropna().drop_duplicates()

    for iteration in range(max_iterations):
        old_count = len(filtered)

        user_counts = filtered["user_id"].value_counts()
        item_counts = filtered["business_id"].value_counts()

        keep_users = set(user_counts[user_counts >= k].index)
        keep_items = set(item_counts[item_counts >= k].index)

        filtered = filtered[
            filtered["user_id"].isin(keep_users)
            & filtered["business_id"].isin(keep_items)
        ].drop_duplicates()

        logger.info(
            "Review-level k-core (k=%d, iter=%d): %d interactions (removed %d)",
            k,
            iteration + 1,
            len(filtered),
            old_count - len(filtered),
        )

        if len(filtered) == old_count:
            logger.info(
                "Review-level k-core filtering converged after %d iterations",
                iteration + 1,
            )
            break

    return filtered.reset_index(drop=True)


def extract_preprocessing_signature(config_dict: dict) -> dict:
    data_cfg = config_dict.get("data", {})
    db_path = data_cfg.get("db_path")
    db_path_str = str(Path(db_path).resolve()) if db_path else None
    return {
        "cache_version": PREPROCESSING_CACHE_VERSION,
        "db_path": db_path_str,
        "state_filter": data_cfg.get("state_filter"),
        "min_review_count": data_cfg.get("min_review_count"),
        "pos_threshold": data_cfg.get("pos_threshold"),
        "year_min": data_cfg.get("year_min"),
        "year_max": data_cfg.get("year_max"),
        "k_core": PREPROCESSING_K_CORE,
    }


def build_preprocessing_cache_key(config_dict: dict) -> str:
    signature = extract_preprocessing_signature(config_dict)
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()[:16]


def get_shared_preprocessing_cache_dir(config_dict: dict) -> Path:
    output_base = Path(config_dict.get("output", {}).get("base_dir", "outputs"))
    cache_key = build_preprocessing_cache_key(config_dict)
    return output_base / "_shared_preprocessed" / cache_key


def shared_preprocessing_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_shared_preprocessing_cache(
    cache_dir: Path,
    *,
    config_dict: dict,
    reviews: Any,
    raw_dataset: Any,
    final_csr: csr_matrix,
    item_map_after_kcore: dict[str, int],
    final_user_ids: list[str],
    universal_user_map: dict[str, int],
    universal_business_map: dict[str, int],
) -> dict:
    raw_dir = cache_dir / "raw_filtered"
    final_dir = cache_dir / "kcore_filtered"
    cache_dir.mkdir(parents=True, exist_ok=True)

    save_dataset(raw_dataset, raw_dir)
    save_dataset(
        type(raw_dataset)(
            user_map={uid: idx for idx, uid in enumerate(final_user_ids)},
            item_map=item_map_after_kcore,
            csr=final_csr,
        ),
        final_dir,
    )

    with (cache_dir / "_universal_mappings.pkl").open("wb") as f:
        pickle.dump((universal_user_map, universal_business_map), f)
    with (cache_dir / "reviews_df.pkl").open("wb") as f:
        pickle.dump(reviews, f)
    with (cache_dir / "final_user_ids.json").open("w", encoding="utf-8") as f:
        json.dump(final_user_ids, f, indent=2)

    manifest = {
        "cache_key": build_preprocessing_cache_key(config_dict),
        "cache_version": PREPROCESSING_CACHE_VERSION,
        "created_at": datetime.now().isoformat(),
        "signature": extract_preprocessing_signature(config_dict),
        "artifacts": {
            "raw_dataset_dir": str(raw_dir),
            "final_dataset_dir": str(final_dir),
            "universal_mappings": str(cache_dir / "_universal_mappings.pkl"),
            "reviews_df": str(cache_dir / "reviews_df.pkl"),
            "final_user_ids": str(cache_dir / "final_user_ids.json"),
        },
        "counts": {
            "raw_n_users": int(raw_dataset.csr.shape[0]),
            "raw_n_items": int(raw_dataset.csr.shape[1]),
            "raw_n_interactions": int(raw_dataset.csr.nnz),
            "final_n_users": int(final_csr.shape[0]),
            "final_n_items": int(final_csr.shape[1]),
            "final_n_interactions": int(final_csr.nnz),
        },
    }
    _save_json(shared_preprocessing_manifest_path(cache_dir), manifest)
    return manifest


def load_shared_preprocessing_cache(cache_dir: Path) -> dict:
    manifest = _load_json(shared_preprocessing_manifest_path(cache_dir))
    raw_dataset = load_dataset(cache_dir / "raw_filtered")
    final_dataset = load_dataset(cache_dir / "kcore_filtered")
    with (cache_dir / "_universal_mappings.pkl").open("rb") as f:
        universal_user_map, universal_business_map = pickle.load(f)
    with (cache_dir / "reviews_df.pkl").open("rb") as f:
        reviews = pickle.load(f)
    final_user_ids = _load_json(cache_dir / "final_user_ids.json")
    if not isinstance(final_user_ids, list):
        raise ValueError(f"Invalid final_user_ids.json in {cache_dir}")

    return {
        "manifest": manifest,
        "reviews": reviews,
        "raw_dataset": raw_dataset,
        "final_dataset": final_dataset,
        "item_map_before_kcore": raw_dataset.item_map,
        "item_map_after_kcore": final_dataset.item_map,
        "final_user_ids": [str(uid) for uid in final_user_ids],
        "universal_user_map": universal_user_map,
        "universal_business_map": universal_business_map,
    }


def build_shared_preprocessing_cache(config_dict: dict) -> dict:
    data_cfg = config_dict["data"]
    db_path = data_cfg["db_path"]
    logger.info("Creating shared preprocessing cache from raw data...")

    all_reviews = load_reviews(
        db_path=db_path,
        pos_threshold=data_cfg["pos_threshold"],
        year_min=data_cfg.get("year_min"),
        year_max=data_cfg.get("year_max"),
    )

    all_users = all_reviews["user_id"].unique()
    all_businesses = all_reviews["business_id"].unique()
    universal_user_map = {uid: idx for idx, uid in enumerate(all_users)}
    universal_business_map = {bid: idx for idx, bid in enumerate(all_businesses)}

    logger.info("Universal mappings created:")
    logger.info("  Total unique users: %d", len(universal_user_map))
    logger.info("  Total unique businesses: %d", len(universal_business_map))

    reviews = all_reviews.copy()
    state_filter = data_cfg.get("state_filter")
    if state_filter:
        businesses = load_businesses(
            db_path=db_path,
            state_filter=state_filter,
            min_review_count=data_cfg.get("min_review_count", 5),
        )
        business_ids = set(businesses["business_id"].values)
        logger.info(
            "Filtering by state %s: %d businesses", state_filter, len(business_ids)
        )
        reviews = reviews[reviews["business_id"].isin(business_ids)]

    logger.info("Loaded %d reviews (after state filtering)", len(reviews))
    logger.info("Building raw CSR matrix from filtered data...")
    raw_dataset = build_csr(reviews)
    raw_csr = raw_dataset.csr
    logger.info(
        "Built CSR: %d users x %d items, %d interactions",
        raw_csr.shape[0],
        raw_csr.shape[1],
        raw_csr.nnz,
    )

    logger.info(
        "Applying review-level k-core filtering (k=%d)...", PREPROCESSING_K_CORE
    )
    kcore_reviews = _apply_kcore_filtering_to_reviews(reviews, k=PREPROCESSING_K_CORE)
    final_dataset = build_csr(kcore_reviews)
    final_csr = final_dataset.csr
    item_map_after_kcore = final_dataset.item_map
    logger.info(
        "Item mapping: %d items -> %d items (after k-core)",
        len(raw_dataset.item_map),
        len(item_map_after_kcore),
    )
    logger.info(
        "After k-core: %d users x %d items, %d interactions",
        final_csr.shape[0],
        final_csr.shape[1],
        final_csr.nnz,
    )

    final_user_ids = list(final_dataset.user_map.keys())

    return {
        "reviews": reviews,
        "raw_dataset": raw_dataset,
        "final_csr": final_csr,
        "item_map_before_kcore": raw_dataset.item_map,
        "item_map_after_kcore": item_map_after_kcore,
        "final_user_ids": [str(uid) for uid in final_user_ids],
        "universal_user_map": universal_user_map,
        "universal_business_map": universal_business_map,
    }


def prepare_shared_preprocessing_cache(
    config_dict: dict, *, require_existing: bool = False
) -> tuple[dict, str, Path]:
    cache_dir = get_shared_preprocessing_cache_dir(config_dict)
    cache_key = build_preprocessing_cache_key(config_dict)

    if cache_dir.exists():
        try:
            payload = load_shared_preprocessing_cache(cache_dir)
            logger.info("Loaded shared preprocessing cache: %s", cache_dir)
            return payload, "loaded_shared_cache", cache_dir
        except Exception as e:
            if require_existing:
                raise RuntimeError(
                    f"--skip-preprocessing requires an existing valid shared preprocessing cache at {cache_dir}: {e}"
                ) from e
            logger.warning(
                "Shared preprocessing cache at %s is invalid (%s). Rebuilding it.",
                cache_dir,
                e,
            )
    elif require_existing:
        raise FileNotFoundError(
            f"--skip-preprocessing requires a shared preprocessing cache, but none exists for key {cache_key} at {cache_dir}"
        )

    built_payload = build_shared_preprocessing_cache(config_dict)
    manifest = _save_shared_preprocessing_cache(
        cache_dir,
        config_dict=config_dict,
        reviews=built_payload["reviews"],
        raw_dataset=built_payload["raw_dataset"],
        final_csr=built_payload["final_csr"],
        item_map_after_kcore=built_payload["item_map_after_kcore"],
        final_user_ids=built_payload["final_user_ids"],
        universal_user_map=built_payload["universal_user_map"],
        universal_business_map=built_payload["universal_business_map"],
    )
    payload = load_shared_preprocessing_cache(cache_dir)
    payload["manifest"] = manifest
    logger.info("Built shared preprocessing cache: %s", cache_dir)
    return payload, "built_shared_cache", cache_dir
