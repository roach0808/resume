# Streamlit Secrets Setup Guide

This project uses Streamlit's secrets feature for secure API key management, especially for cloud deployments.

## Setup for Cloud Deployment (Streamlit Cloud, etc.)

1. **Create the secrets file:**
   - In your Streamlit Cloud dashboard, go to your app settings
   - Navigate to "Secrets" section
   - Add the following secret:
   ```
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

2. **Alternative: Using secrets.toml file (for local testing):**
   - Create a `.streamlit` directory in your project root (if it doesn't exist)
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Replace `your_openai_api_key_here` with your actual OpenAI API key
   - **IMPORTANT:** Add `.streamlit/secrets.toml` to `.gitignore` to avoid committing secrets

## Setup for Local Development

The code automatically falls back to environment variables if Streamlit secrets are not available:

1. **Option 1: Environment Variable**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

2. **Option 2: .env file (still supported)**
   - Create a `.env` file in your project root
   - Add: `OPENAI_API_KEY=your_openai_api_key_here`
   - The code will use this as a fallback

## How It Works

The `get_openai_api_key()` function in `config.py`:
1. First tries to get the API key from Streamlit secrets (for cloud deployment)
2. Falls back to environment variables (for local development)
3. Raises a clear error if neither is found

## Files Updated

All files that previously used `os.getenv("OPENAI_API_KEY")` or `load_dotenv()` have been updated to use the new `get_openai_api_key()` function:

- `app.py`
- `app_minimal.py`
- `test_minimal.py`
- `config.py`
- `resume_builder/utils/openai_llm_inference.py`
- `resume_builder/parser/resume_parser.py`

## Security Notes

- Never commit `.streamlit/secrets.toml` to version control
- Never commit `.env` files to version control
- Use Streamlit Cloud's secrets management for production deployments
- Keep your API keys secure and rotate them regularly

