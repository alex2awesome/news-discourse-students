class Config:
    # Flask settings
    SECRET_KEY = 'your-secret-key'  # Replace with env variable in production
    
    # Application settings
    USE_SPACY = True
    CLEAN_TEXT = False
    LLM_CLIENT = "openai"
    DEFAULT_CLAUDE_MODEL = "claude-3-5-haiku-latest"
    DEFAULT_TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
    BATCH_SIZE = 2
    SKIP_LOGIN = True