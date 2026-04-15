"""Google Cloud SQL helper for PostgreSQL database operations."""

import json
import logging
from typing import Dict, List, Optional
import os

try:
    from google.cloud.sql.connector import Connector
    import sqlalchemy

    HAS_CLOUD_SQL = True
except ImportError:
    HAS_CLOUD_SQL = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

logger = logging.getLogger(__name__)


class CloudSQLHelper:
    """Helper for connecting to Google Cloud SQL PostgreSQL instances."""

    def __init__(
        self,
        instance_connection_name: str,
        database: str,
        user: str,
        password: str,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        """
        Initialize Cloud SQL helper.

        Args:
            instance_connection_name: Cloud SQL instance connection string (project:region:instance-id)
            database: PostgreSQL database name
            user: Database user
            password: Database password
            project_id: Google Cloud Project ID (optional)
            credentials_path: Path to service account JSON (optional)
        """
        if not HAS_CLOUD_SQL:
            raise ImportError(
                "google-cloud-sql-connector not installed. Install with:\n"
                "  pip install cloud-sql-python-connector sqlalchemy"
            )

        self.instance_connection_name = instance_connection_name
        self.database = database
        self.user = user
        self.password = password
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

        # Initialize connector
        try:
            from google.oauth2 import service_account
            from pathlib import Path

            credentials = None

            # Try 1: Explicit credentials file path
            if credentials_path and os.path.exists(credentials_path):
                logger.info(f"Loading SQL credentials from: {credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(credentials_path)

            # Try 2: Local key file
            if not credentials:
                local_key_paths = [
                    Path("streamlit-deploy-key.json"),
                    Path(__file__).parent.parent.parent / "streamlit-deploy-key.json",
                ]
                for key_path in local_key_paths:
                    if key_path.exists():
                        try:
                            logger.info(f"Loading SQL credentials from local key: {key_path}")
                            credentials = service_account.Credentials.from_service_account_file(str(key_path))
                            break
                        except Exception as e:
                            logger.debug(f"Failed to load SQL credentials from {key_path}: {e}")

            # Try 3: Streamlit Secrets
            if not credentials and HAS_STREAMLIT:
                try:
                    if "google_credentials" in st.secrets:
                        logger.info("Loading SQL credentials from Streamlit Secrets")
                        creds_json = st.secrets["google_credentials"]
                        if isinstance(creds_json, str):
                            creds_json = json.loads(creds_json)
                        credentials = service_account.Credentials.from_service_account_info(creds_json)
                except Exception as e:
                    logger.debug(f"Could not load Streamlit secrets: {e}")

            # Try 4: Environment variable path
            if not credentials:
                env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if env_path and os.path.exists(env_path):
                    logger.info(f"Loading SQL credentials from {env_path}")
                    credentials = service_account.Credentials.from_service_account_file(env_path)

            if credentials:
                self.connector = Connector(credentials=credentials)
                logger.info("Cloud SQL using explicit credentials")
            else:
                self.connector = Connector()
                logger.info("Cloud SQL using Application Default Credentials")

        except Exception as e:
            logger.warning(f"Could not load credentials for SQL connector: {e}")
            self.connector = Connector()

        self.engine = None
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize SQLAlchemy connection pool with production-grade configuration."""
        try:
            from sqlalchemy.pool import QueuePool

            def getconn():
                return self.connector.connect(
                    self.instance_connection_name,
                    "pg8000",
                    user=self.user,
                    password=self.password,
                    db=self.database,
                )

            self.engine = sqlalchemy.create_engine(
                "postgresql+pg8000://",
                creator=getconn,
                poolclass=QueuePool,
                pool_size=5,           # Keep 5 connections in pool
                max_overflow=10,       # Allow up to 10 additional connections if needed
                pool_recycle=3600,     # Recycle connections every hour to avoid timeout
                isolation_level="AUTOCOMMIT",  # For Streamlit use case
                echo=False,
            )

            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text("SELECT 1"))
                logger.info(f"✅ Connected to Cloud SQL: {self.instance_connection_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Cloud SQL: {e}")
            raise

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute SELECT query and return results as list of dicts.

        Args:
            query: SQL query string with :param placeholders
            params: Optional dict of parameter values

        Returns:
            List of result rows as dicts
        """
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(sqlalchemy.text(query), params)
                else:
                    result = conn.execute(sqlalchemy.text(query))
                
                rows = []
                for row in result:
                    if isinstance(row, dict):
                        rows.append(row)
                    else:
                        rows.append(dict(row._mapping))
                return rows
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def execute_update(self, query: str, params: Optional[Dict] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query string with :param placeholders
            params: Optional dict of parameter values

        Returns:
            Number of rows affected
        """
        try:
            with self.engine.begin() as conn:
                if params:
                    result = conn.execute(sqlalchemy.text(query), params)
                else:
                    result = conn.execute(sqlalchemy.text(query))
                logger.info(f"✅ Updated {result.rowcount} rows")
                return result.rowcount
        except Exception as e:
            logger.error(f"Update failed: {e}")
            raise

    def close(self) -> None:
        """Close Cloud SQL connection and dispose of the connection pool."""
        if self.engine:
            self.engine.dispose()
        logger.info("✅ Cloud SQL connection closed")


# Singleton instance management
_cloud_sql_instance: Optional[CloudSQLHelper] = None


def get_cloud_sql_instance() -> CloudSQLHelper:
    """Get or create singleton Cloud SQL helper instance."""
    global _cloud_sql_instance

    if _cloud_sql_instance is None:
        # Load from environment
        instance_connection_name = os.getenv("CLOUDSQL_INSTANCE")
        database = os.getenv("CLOUDSQL_DATABASE", "postgres")
        user = os.getenv("CLOUDSQL_USER", "postgres")
        password = os.getenv("CLOUDSQL_PASSWORD")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not instance_connection_name or not password:
            raise ValueError(
                "CLOUDSQL_INSTANCE and CLOUDSQL_PASSWORD must be set in .env"
            )

        _cloud_sql_instance = CloudSQLHelper(
            instance_connection_name=instance_connection_name,
            database=database,
            user=user,
            password=password,
            credentials_path=credentials_path,
        )

    return _cloud_sql_instance