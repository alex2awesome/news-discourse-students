import os
import logging
from flask import Flask
from authlib.integrations.flask_client import OAuth
from .config import DevelopmentConfig, ProductionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    logger.info("Starting application initialization...")
    
    app = Flask(__name__, static_url_path='/', static_folder='static')
    logger.info("Flask app created")
    
    # Load environment-specific config
    env = os.getenv('FLASK_ENV', 'development')
    logger.info(f"Loading {env} configuration...")
    
    if env == 'production':
        app.config.from_object(ProductionConfig)
    else:
        app.config.from_object(DevelopmentConfig)
    logger.info("Configuration loaded")

    # Load environment variables
    logger.info("Loading environment variables...")
    for key in app.config:
        env_val = os.getenv(key)
        if env_val is not None:
            app.config[key] = env_val
    logger.info("Environment variables loaded")

    app.secret_key = app.config['FLASK_SECRET_KEY']
    logger.info("Secret key configured")

    # Configure OAuth
    logger.info("Configuring OAuth...")
    oauth = OAuth(app)
    google = oauth.register(
        name='google',
        client_id=app.config['GOOGLE_CLIENT_ID'],
        client_secret=app.config['GOOGLE_CLIENT_SECRET'],
        access_token_url='https://oauth2.googleapis.com/token',
        access_token_params=None,
        authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
        authorize_params={
            'scope': 'openid email profile',
            'prompt': 'consent',
            'access_type': 'offline'
        },
        api_base_url='https://www.googleapis.com/oauth2/v1/',
        jwks_uri='https://www.googleapis.com/oauth2/v3/certs'
    )
    app.google = google
    app.analyzing_requests = set()
    app.oauth = oauth
    logger.info("OAuth configured")

    # Load configuration from environment variable
    if 'APP_CONFIG_FILE' in os.environ:
        logger.info("Loading additional configuration from APP_CONFIG_FILE...")
        app.config.from_envvar('APP_CONFIG_FILE')
        logger.info("Additional configuration loaded")

    logger.info("Registering blueprints...")
    from .auth import auth_bp
    from .app import main_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    logger.info("Blueprints registered")

    logger.info("Application initialization complete")
    return app