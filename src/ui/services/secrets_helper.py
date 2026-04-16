"""
Streamlit Secrets Helper

Provides utilities to safely retrieve secrets from Streamlit's secrets management.
Works with both .streamlit/secrets.toml (local dev) and Streamlit Cloud secrets.
"""

import os
from typing import Any, Optional

import streamlit as st


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret from Streamlit secrets or environment variables.

    Tries (in order):
    1. Streamlit secrets (st.secrets[key])
    2. Environment variable (os.getenv)
    3. Default value

    Args:
        key: Secret key name (e.g., "GEMINI_API_KEY")
        default: Default value if secret not found

    Returns:
        Secret value or default, or None if not found
    """
    try:
        # Try Streamlit secrets first (works in both local + Cloud)
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fall back to environment variables (for CI/CD or manual env setup)
    if env_val := os.getenv(key):
        return env_val

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
    return {
        "instance": get_secret("CLOUDSQL_INSTANCE"),
        "database": get_secret("CLOUDSQL_DATABASE", "postgres"),
        "user": get_secret("CLOUDSQL_USER", "postgres"),
        "password": get_secret("CLOUDSQL_PASSWORD"),
        "credentials_path": get_gcp_credentials_path(),
    }


def get_cloud_storage_bucket() -> Optional[str]:
    """
    Get Google Cloud Storage bucket name from secrets.

    Returns:
        Bucket name (e.g., 'neuronsae-photos') or None if not configured
    """
    return get_secret("CLOUD_STORAGE_BUCKET")
