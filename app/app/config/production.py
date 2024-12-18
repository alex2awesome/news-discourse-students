from .default import Config

class ProductionConfig(Config):
    DEBUG = False
    SKIP_LOGIN = False
    # Add production-specific settings