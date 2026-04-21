"""
Streamlit Secrets Helper

Provides utilities to safely retrieve secrets from Streamlit's secrets management.
Works with both .streamlit/secrets.toml (local dev) and Streamlit Cloud secrets.
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret from Streamlit secrets or environment variables.

    Tries (in order):
    1. Streamlit secrets (st.secrets[key]) - if Streamlit is available
    2. Environment variable (os.getenv)
    3. Default value

    Args:
        key: Secret key name (e.g., "GEMINI_API_KEY")
        default: Default value if secret not found

    Returns:
        Secret value or default, or None if not found
    """
    # Try Streamlit secrets first (if available)
    try:
        import streamlit as st

        if key in st.secrets:
            logger.debug(f"✓ Found {key} in Streamlit secrets")
            return st.secrets[key]
        else:
            logger.debug(f"✗ {key} not in Streamlit secrets")
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug(
            f"✗ Streamlit not available or st.secrets not accessible: {type(e).__name__}"
        )

    # Fall back to environment variables (for CI/CD, training scripts, or when Streamlit unavailable)
    env_val = os.getenv(key)
    if env_val:
        logger.debug(f"✓ Found {key} in environment variables")
        return env_val
    else:
        logger.debug(f"✗ {key} not in environment variables")

    logger.debug(f"✗ {key} not found anywhere, using default: {default}")
    return default


def get_gcp_credentials_path() -> Optional[str]:
    """
    Get Google Cloud credentials path from secrets or environment.

    Returns the path to GOOGLE_APPLICATION_CREDENTIALS file.
    """
    return get_secret("GOOGLE_APPLICATION_CREDENTIALS")


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from secrets."""
    return get_secret("GEMINI_API_KEY")


def get_google_api_key() -> Optional[str]:
    """Get Google API key from secrets (for Generative AI)."""
    return get_secret("GOOGLE_API_KEY")


def get_gcp_project() -> Optional[str]:
    """Get GCP project ID from secrets."""
    return get_secret("GOOGLE_CLOUD_PROJECT")


def get_gcp_region() -> str:
    """Get GCP region from secrets, default to us-central1."""
    return get_secret("GOOGLE_CLOUD_REGION", "us-central1")


def get_cloudsql_config() -> dict[str, Optional[str]]:
    """
    Get Cloud SQL configuration from secrets.

    Returns:
        Dict with keys: instance, database, user, password, credentials_path
    """
    logger.info("🔍 Reading Cloud SQL configuration from secrets...")

    config = {
        "instance": get_secret("CLOUDSQL_INSTANCE"),
        "database": get_secret("CLOUDSQL_DATABASE", "postgres"),
        "user": get_secret("CLOUDSQL_USER", "postgres"),
        "password": get_secret("CLOUDSQL_PASSWORD"),
        "credentials_path": get_gcp_credentials_path(),
    }

    logger.info(f"Cloud SQL config result:")
    logger.info(f"  instance: {config['instance']}")
    logger.info(f"  database: {config['database']}")
    logger.info(f"  user: {config['user']}")
    logger.info(f"  password: {'SET' if config['password'] else 'NOT SET'}")
    logger.info(f"  credentials_path: {config['credentials_path']}")

    return config


def get_cloud_storage_bucket() -> Optional[str]:
    """
    Get Google Cloud Storage bucket name from secrets.

    Checks both CLOUD_STORAGE_BUCKET and GCS_BUCKET_NAME for compatibility.

    Returns:
        Bucket name (e.g., 'diplomova-prace') or None if not configured
    """
    # Try both naming conventions
    bucket = get_secret("CLOUD_STORAGE_BUCKET")
    if not bucket:
        bucket = get_secret("GCS_BUCKET_NAME")
    return bucket
