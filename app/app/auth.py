from flask import Blueprint, session, redirect, url_for, request, jsonify
from authlib.integrations.flask_client import OAuth
import os 
from dotenv import load_dotenv
from flask import current_app
app = current_app

load_dotenv()

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login')
def login():
    # Redirect to Google's OAuth 2.0 server for login.
    if app.config.get('SKIP_LOGIN', False):
        session['logged_in'] = True
        return redirect(url_for('main.analysis_page'))
    else:
        return app.google.authorize_redirect(url_for('auth.authorize', _external=True)) 

@auth_bp.route('/authorize')
def authorize():
    google = app.google
    token = google.authorize_access_token()
    if token is None:
        return "Authorization failed.", 401

    resp = google.get('userinfo')
    user_info = resp.json()
    user_email = user_info.get('email')
    allowed_domain = "usc.edu"
    if not user_email.endswith(allowed_domain):
        return "Unauthorized", 401

    session['logged_in'] = True
    session['user'] = {
        'email': user_info.get('email'),
        'name': user_info.get('name')
    }
    return redirect(url_for('main.analysis_page'))