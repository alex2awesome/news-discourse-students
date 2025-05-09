class Config:
    # API Keys and Secrets
    GOOGLE_CLIENT_ID = None
    GOOGLE_CLIENT_SECRET = None
    FLASK_SECRET_KEY = None
    OPENAI_API_KEY = None
    TOGETHER_API_KEY = None
    ANTHROPIC_API_KEY = None

    # Application Settings
    USE_SPACY = True
    CLEAN_TEXT = False
    LLM_CLIENT = "openai"
    DEFAULT_CLAUDE_MODEL = "claude-3-5-haiku-latest"
    DEFAULT_TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
    BATCH_SIZE = 2
    SKIP_LOGIN = True