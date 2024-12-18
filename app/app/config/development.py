from .default import Config

class DevelopmentConfig(Config):
    DEBUG = True
    SKIP_LOGIN = True
    # Add development-specific settings