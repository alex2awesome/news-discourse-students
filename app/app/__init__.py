import os
from flask import Flask
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
import spacy

load_dotenv()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

USE_SPACY = True
CLEAN_TEXT = False
LLM_CLIENT = "openai" 
DEFAULT_CLAUDE_MODEL = "claude-3-5-haiku-latest"
DEFAULT_TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
BATCH_SIZE = 2
SKIP_LOGIN = True

spacy_model = None
def load_spacy_model():
    global spacy_model
    if spacy_model is None:
        spacy_model = spacy.load("en_core_web_lg")
    return spacy_model

def create_app():
    app = Flask(__name__, static_url_path='/', static_folder='static')
    
    # Load environment-specific config
    env = os.getenv('FLASK_ENV', 'development')
    if env == 'production':
        app.config.from_object('app.config.production.ProductionConfig')
    else:
        app.config.from_object('app.config.development.DevelopmentConfig')

    # Override with environment variables
    for key in app.config:
        env_val = os.getenv(key)
        if env_val is not None:
            app.config[key] = env_val

    app.secret_key = FLASK_SECRET_KEY

    # Configure OAuth
    oauth = OAuth(app)
    google = oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
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
    # Load configuration from environment variable
    if 'APP_CONFIG_FILE' in os.environ:
        app.config.from_envvar('APP_CONFIG_FILE')

    from .auth import auth_bp
    from .main_app import main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    return app