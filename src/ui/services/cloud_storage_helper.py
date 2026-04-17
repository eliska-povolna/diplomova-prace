"""Google Cloud Storage helper for metadata and model exports."""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Any

try:
    from google.cloud import storage
    from google.oauth2 import service_account

    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

logger = logging.getLogger(__name__)


class CloudStorageHelper:
    """Helper for reading/writing to Google Cloud Storage."""

    def __init__(
        self,
        bucket_name: str,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """
        Initialize GCS helper.

        Args:
            bucket_name: GCS bucket name (e.g., 'neuronsae-data-myproject')
            credentials_path: Path to service account JSON key file (optional)
                             - If not provided, tries local key file (streamlit-deploy-key.json)
                             - Then tries Streamlit secrets
                             - Then falls back to Application Default Credentials
            project_id: Google Cloud Project ID
        """
        if not HAS_GCS:
            raise ImportError(
                "google-cloud-storage not installed. Install with: "
                "pip install google-cloud-storage"
            )

        self.bucket_name = bucket_name
        from .secrets_helper import get_secret, get_gcp_project

        self.project_id = project_id or get_gcp_project()

        # Initialize credentials
        credentials = None

        # Try 1: Explicit credentials file path
        if credentials_path and os.path.exists(credentials_path):
            logger.info(f"Loading GCS credentials from: {credentials_path}")
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )

        # Try 2: Local key file (streamlit-deploy-key.json in project root)
        if not credentials:
            local_key_paths = [
                Path("streamlit-deploy-key.json"),  # Current dir
                Path(__file__).parent.parent.parent
                / "streamlit-deploy-key.json",  # Project root
            ]
            for key_path in local_key_paths:
                if key_path.exists():
                    try:
                        logger.info(
                            f"Loading GCS credentials from local key: {key_path}"
                        )
                        credentials = (
                            service_account.Credentials.from_service_account_file(
                                str(key_path)
                            )
                        )
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load from {key_path}: {e}")

        # Try 3: Streamlit Secrets (for deployed apps)
        if not credentials and HAS_STREAMLIT:
            try:
                if "google_credentials" in st.secrets:
                    logger.info("Loading GCS credentials from Streamlit Secrets")
                    creds_json = st.secrets["google_credentials"]
                    if isinstance(creds_json, str):
                        creds_json = json.loads(creds_json)
                    credentials = service_account.Credentials.from_service_account_info(
                        creds_json
                    )
            except Exception as e:
                logger.debug(f"Could not load Streamlit secrets: {e}")

        # Try 4: Environment variable path or secrets
        if not credentials:
            from .secrets_helper import get_gcp_credentials_path

            env_path = get_gcp_credentials_path()
            if env_path and os.path.exists(env_path):
                logger.info(f"Loading GCS credentials from {env_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    env_path
                )

        # Initialize client with credentials or use Application Default Credentials
        if credentials:
            self.client = storage.Client(
                credentials=credentials, project=self.project_id
            )
            logger.info("Using explicit credentials")
        else:
            logger.info("Using Application Default Credentials")
            self.client = storage.Client(project=self.project_id)

        self.bucket = self.client.bucket(bucket_name)

        # Verify bucket exists
        if not self.bucket.exists():
            logger.warning(f"Bucket {bucket_name} not found. Will attempt to create...")
            raise RuntimeError(f"Bucket {bucket_name} does not exist")

        logger.info(f"✅ Connected to GCS bucket: {bucket_name}")

    def upload_json(
        self, local_path: Path, gcs_path: str, metadata: Optional[Dict] = None
    ) -> bool:
        """
        Upload JSON file to GCS.

        Args:
            local_path: Local file path
            gcs_path: GCS path (e.g., 'metadata/neuron_labels.json')
            metadata: Optional metadata dict for GCS object

        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path), content_type="application/json")

            if metadata:
                blob.metadata = metadata
                blob.patch()

            logger.info(
                f"✅ Uploaded {local_path.name} → gs://{self.bucket_name}/{gcs_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to GCS: {e}")
            return False

    def download_json(self, gcs_path: str, local_path: Optional[Path] = None) -> Dict:
        """
        Download and parse JSON file from GCS.

        Args:
            gcs_path: GCS path to JSON file
            local_path: Optional local path to save to

        Returns:
            Parsed JSON dict
        """
        try:
            blob = self.bucket.blob(gcs_path)
            content = blob.download_as_string()
            data = json.loads(content)

            if local_path:
                with open(local_path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"✅ Downloaded and saved {gcs_path} → {local_path}")
            else:
                logger.info(f"✅ Downloaded {gcs_path} from GCS")

            return data
        except Exception as e:
            logger.error(f"Failed to download {gcs_path} from GCS: {e}")
            raise

    def read_json(self, gcs_path: str) -> Dict:
        """Read JSON from GCS without saving locally."""
        return self.download_json(gcs_path)

    def read_pickle(self, gcs_path: str) -> Any:
        """
        Download and unpickle binary object from GCS.

        Args:
            gcs_path: GCS path to pickle file

        Returns:
            Unpickled Python object
        """
        try:
            blob = self.bucket.blob(gcs_path)
            content = blob.download_as_bytes()
            data = pickle.loads(content)
            logger.info(f"✅ Downloaded pickle from {gcs_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to download pickle from {gcs_path}: {e}")
            return None

    def list_files(self, prefix: str = "") -> list:
        """List all files in GCS bucket with optional prefix."""
        try:
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=prefix))
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list files in GCS: {e}")
            return []

    def delete_file(self, gcs_path: str) -> bool:
        """Delete file from GCS."""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.delete()
            logger.info(f"✅ Deleted {gcs_path} from GCS")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {gcs_path} from GCS: {e}")
            return False

    def exists(self, gcs_path: str) -> bool:
        """Check if file exists in GCS."""
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except Exception as e:
            logger.warning(f"Error checking existence of {gcs_path}: {e}")
            return False

    def get_photo_url(self, gcs_path: str, expiration_hours: int = 24) -> Optional[str]:
        """
        Get a signed URL for a photo in GCS (valid for specified hours).

        Args:
            gcs_path: GCS path to photo (e.g., 'photos/photo_id.jpg')
            expiration_hours: URL expiration time in hours (default 24)

        Returns:
            Signed URL string or None if photo doesn't exist
        """
        try:
            blob = self.bucket.blob(gcs_path)
            if not blob.exists():
                return None

            from datetime import timedelta

            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=expiration_hours),
                method="GET",
            )
            logger.debug(f"Generated signed URL for {gcs_path}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {gcs_path}: {e}")
            return None

    def download_photo_bytes(self, gcs_path: str) -> Optional[bytes]:
        """
        Download photo bytes from GCS (useful for Streamlit st.image()).

        Args:
            gcs_path: GCS path to photo

        Returns:
            Photo bytes or None if failed
        """
        try:
            blob = self.bucket.blob(gcs_path)
            photo_bytes = blob.download_as_bytes()
            logger.debug(f"Downloaded photo bytes from {gcs_path}")
            return photo_bytes
        except Exception as e:
            logger.error(f"Failed to download photo from {gcs_path}: {e}")
            return None
