# Secrets Configuration Guide

This guide explains how to configure secrets for local development and production (Streamlit Cloud).

## Local Development

### 1. Create `.streamlit/secrets.toml`

This file already exists as a template. Edit it with your credentials:

```bash
# Open the file in your editor
.streamlit/secrets.toml
```

### 2. Configure Your Secrets

Fill in the template with your actual values. The main secrets needed are:

```toml
# Google Cloud Project
GOOGLE_CLOUD_PROJECT = "your-gcp-project-id"
GOOGLE_CLOUD_REGION = "us-central1"

# Gemini API (easiest for local dev)
GEMINI_API_KEY = "your-gemini-api-key"

# Google API Key (for other services)
GOOGLE_API_KEY = "your-google-api-key"

# Service account JSON (either as multi-line string or path)
google_credentials = """
{
  "type": "service_account",
  ...
}
"""

# Cloud SQL (if using Cloud SQL backend)
CLOUDSQL_INSTANCE = "project:region:instance-id"
CLOUDSQL_DATABASE = "postgres"
CLOUDSQL_USER = "postgres"
CLOUDSQL_PASSWORD = "your-db-password"
```

### 3. Restart Streamlit

Changes take ~1 minute to propagate:

```bash
streamlit run src/ui/main.py
```

## Getting Your Credentials

### Gemini API Key (Recommended for local dev)

1. Go to [ai.google.dev](https://ai.google.dev/)
2. Click "Get API Key"
3. Copy your API key
4. Paste into `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   ```

### Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Service Accounts**
3. Click **Create Service Account**
4. Give it a name (e.g., "streamlit-app")
5. Click "Create and Continue"
6. Assign roles (at minimum):
   - Storage Object Viewer (for GCS)
   - Cloud SQL Client (for Cloud SQL)
   - Vertex AI User (if using Vertex AI)
7. Click "Continue" and then "Create Key"
8. Choose **JSON** format and download
9. Open the JSON file and copy its entire content
10. Paste into `.streamlit/secrets.toml`:
    ```toml
    google_credentials = """
    {
      "type": "service_account",
      ...entire JSON content...
    }
    """
    ```

### Google API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **APIs & Services** → **Credentials**
3. Click **Create Credentials** → **API Key**
4. Copy the key
5. Paste into `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "AIzaSy..."
   ```

## Production (Streamlit Cloud)

### 1. Deploy Your App

Push your code to GitHub (`.streamlit/secrets.toml` is in `.gitignore` so it won't be committed):

```bash
git push origin feat/streamlit-ui
```

### 2. Add Secrets on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click your app
3. Click **Settings** (gear icon)
4. Click **Secrets**
5. Paste your secrets in TOML format (same as local `secrets.toml`)
6. Click **Save**

Secrets are automatically encrypted and injected at runtime.

### 3. Verify in Logs

Once deployed, check app logs to confirm secrets are loaded:

```
✅ Initialized Gemini with direct API key
✅ Using Cloud SQL as backend
```

## Important: `.env` vs `.streamlit/secrets.toml`

The app **no longer uses `.env` files**. If you have an old `.env` file:

```bash
# You can delete it or keep it for other tools
rm .env

# The app will automatically use .streamlit/secrets.toml instead
```

For **CI/CD pipelines** (GitHub Actions, etc.), set environment variables directly in your CI configuration:

```yaml
# .github/workflows/ci.yml
env:
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  GOOGLE_CLOUD_PROJECT: ${{ secrets.GOOGLE_CLOUD_PROJECT }}
```

## Accessing Secrets in Code

If you need to use secrets in your Python code:

```python
import streamlit as st
from services.secrets_helper import (
    get_gemini_api_key,
    get_gcp_project,
    get_cloudsql_config,
)

# Get individual secrets
api_key = get_gemini_api_key()
project = get_gcp_project()

# Get grouped config
sql_config = get_cloudsql_config()
# Returns: {"instance": "...", "database": "...", "user": "...", "password": "..."}

# Or access directly
# Direct access via st.secrets (useful in Streamlit components)
gemini_key = st.secrets["GEMINI_API_KEY"]
```

## Troubleshooting

### Secret not found error

**Problem**: `KeyError: 'GEMINI_API_KEY'`

**Solution**: 
1. Check `.streamlit/secrets.toml` exists
2. Verify the key name matches exactly (case-sensitive)
3. Restart Streamlit: `streamlit run src/ui/main.py`
4. Wait ~1 minute for changes to propagate

### Import errors

**Problem**: `ModuleNotFoundError: No module named 'services'`

**Solution**: Ensure you're running from the project root:
```bash
cd c:\Users\elisk\Desktop\2024-25\Diplomka\Github\Diplomov-pr-ce
streamlit run src/ui/main.py
```

### Secrets not loading in CI/CD

**Problem**: Tests or CI pipeline can't find secrets

**Solution**: Set environment variables in your CI configuration:
```bash
export GEMINI_API_KEY="..."
export GOOGLE_CLOUD_PROJECT="..."
```

## Security Best Practices

✅ **DO:**
- Store `.streamlit/secrets.toml` only locally (it's in `.gitignore`)
- Use separate service accounts for different environments
- Rotate API keys periodically
- Use strong passwords for database credentials
- Set appropriate IAM roles (principle of least privilege)

❌ **DON'T:**
- Commit `.streamlit/secrets.toml` to git
- Share secrets in chat or email
- Use the same service account for production and development
- Expose secrets in logs or error messages

## Next Steps

1. **Local Setup**: Fill in `.streamlit/secrets.toml` with your credentials
2. **Test Locally**: `streamlit run src/ui/main.py`
3. **Deploy**: Push to GitHub and add secrets on Streamlit Cloud
4. **Verify**: Check app logs to confirm secrets loaded correctly

For more info: [Streamlit Secrets Documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
