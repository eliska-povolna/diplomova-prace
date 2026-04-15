# Cloud Data Integration

## Overview

The application supports dual backends:
- **Cloud SQL (Primary)** - PostgreSQL in GCP, stores Yelp data (businesses, users, reviews)
- **DuckDB (Local Fallback)** - Offline-capable, automatic fallback if Cloud SQL unavailable

## Architecture

```
Streamlit UI
    ↓
DataService (auto-detects backend)
    ├─→ Cloud SQL (if CLOUDSQL_* env vars set)
    └─→ DuckDB (fallback/offline)
```

## Setup

Set these environment variables in `.env`:
```bash
CLOUDSQL_INSTANCE=project:region:instance-name
CLOUDSQL_DATABASE=postgres
CLOUDSQL_USER=postgres
CLOUDSQL_PASSWORD=your-password
```

If not set, the app falls back to local DuckDB automatically.

## Services

- **DataService** - Unified data access (both backends)
- **CloudSQLHelper** - Cloud SQL credential management
- **CloudStorageHelper** - GCS file operations (planned)
- **GeminiLabelingService** - LLM-based neuron labeling (planned)

For detailed setup and architecture, see `deployment/` folder.
