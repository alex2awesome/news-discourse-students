from .default import Config

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    SKIP_LOGIN = True
    # Add development-specific settings