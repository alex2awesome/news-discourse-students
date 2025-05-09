import os
from dotenv import load_dotenv

load_dotenv()

# API Keys and Secrets
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Application Settings
USE_SPACY = True
CLEAN_TEXT = False
LLM_CLIENT = "openai" 
DEFAULT_CLAUDE_MODEL = "claude-3-5-haiku-latest"
DEFAULT_TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
BATCH_SIZE = 2
SKIP_LOGIN = False 