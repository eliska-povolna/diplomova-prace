"""
Streamlit Secrets Helper

Utilities to retrieve secrets from Streamlit secrets and environment variables.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)
_CLOUDSQL_CONFIG_LOGGED = False


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve secret by key from Streamlit first, then environment."""
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except (ImportError, RuntimeError, AttributeError):
        pass

    return os.getenv(key, default)


def get_gcp_credentials_path() -> Optional[str]:
    """Path to GOOGLE_APPLICATION_CREDENTIALS."""
    return get_secret("GOOGLE_APPLICATION_CREDENTIALS")


def get_gemini_api_key() -> Optional[str]:
    """Gemini API key."""
    return get_secret("GEMINI_API_KEY")


def get_google_api_key() -> Optional[str]:
    """Google API key for Generative AI."""
    return get_secret("GOOGLE_API_KEY")


def get_gcp_project() -> Optional[str]:
    """GCP project ID."""
    return get_secret("GOOGLE_CLOUD_PROJECT")


def get_gcp_region() -> str:
    """GCP region, defaults to us-central1."""
    return get_secret("GOOGLE_CLOUD_REGION", "us-central1")


def get_cloudsql_config() -> dict[str, Optional[str]]:
    """Cloud SQL configuration from secrets/env."""
    config = {
        "instance": get_secret("CLOUDSQL_INSTANCE"),
        "database": get_secret("CLOUDSQL_DATABASE", "postgres"),
        "user": get_secret("CLOUDSQL_USER", "postgres"),
        "password": get_secret("CLOUDSQL_PASSWORD"),
        "credentials_path": get_gcp_credentials_path(),
    }

    global _CLOUDSQL_CONFIG_LOGGED
    if not _CLOUDSQL_CONFIG_LOGGED:
        logger.info("Cloud SQL config loaded from secrets")
        logger.debug("  instance: %s", config["instance"])
        logger.debug("  database: %s", config["database"])
        logger.debug("  user: %s", config["user"])
        logger.debug("  password: %s", "SET" if config["password"] else "NOT SET")
        logger.debug("  credentials_path: %s", config["credentials_path"])
        _CLOUDSQL_CONFIG_LOGGED = True

    return config


def get_cloud_storage_bucket() -> Optional[str]:
    """Cloud Storage bucket name from CLOUD_STORAGE_BUCKET or GCS_BUCKET_NAME."""
    bucket = get_secret("CLOUD_STORAGE_BUCKET")
    if not bucket:
        bucket = get_secret("GCS_BUCKET_NAME")
    return bucket
