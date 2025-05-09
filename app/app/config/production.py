from .default import Config

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SKIP_LOGIN = False
    # Add production-specific settings