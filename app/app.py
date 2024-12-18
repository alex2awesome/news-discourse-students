from flask import Flask, session, redirect, url_for, request, jsonify, send_from_directory, abort
import os
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from prompts import SENTENCIZE_PROMPT, TEST_PROMPT, LABELING_PROMPT, CLEAN_TEXT_PROMPT
import json, ast 
import re
import spacy 
from flask import Response
import os, asyncio
from together import AsyncTogether
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI


load_dotenv()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
USE_SPACY = True
CLEAN_TEXT = False
LLM_CLIENT = "openai"  # flag: "openai", "together", or "claude"
DEFAULT_CLAUDE_MODEL = "claude-3-5-haiku-latest"
DEFAULT_TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
BATCH_SIZE = 2
SKIP_LOGIN = True


app = Flask(__name__, static_url_path='/', static_folder='static')
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
        'prompt': 'consent',  # This may or may not be needed depending on desired behavior.
        'access_type': 'offline'
    },
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs'
)

spacy_model = None
def load_spacy_model():
    global spacy_model
    if spacy_model is None:
        spacy_model = spacy.load("en_core_web_lg")
    return spacy_model

def robust_parse_answer(answer):
    parsed_sentences = re.search(r'\[.*?\]', answer)
    if parsed_sentences:
        answer = parsed_sentences.group(0)
    try:
        answer = answer.strip()
        return json.loads(answer)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(answer)
        except Exception as e:
            raise e

def call_openai_batch(prompts, model="gpt-4o-mini", max_retries=3, post_process=None):
    client = AsyncOpenAI()
    async def async_chat_completion(messages):
        tasks = [
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}]
            )
            for message in messages
        ]
        responses = await asyncio.gather(*tasks)
        return list(map(lambda x: x.choices[0].message.content, responses))
    return asyncio.run(async_chat_completion(prompts))

def call_together_batch(prompts, model, max_retries=3, post_process=None):
    async_client = AsyncTogether()
    async def async_chat_completion(messages):
        tasks = [
            async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
            )
            for message in messages
        ]
        responses = await asyncio.gather(*tasks)
        return list(map(lambda x: x.choices[0].message.content, responses))
    return asyncio.run(async_chat_completion(prompts))

def call_claude_batch(prompts, model, max_retries=3, post_process=None):
    async_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def async_chat_completion(messages):
        tasks = [
            async_client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            for message in messages
        ]
        responses = await asyncio.gather(*tasks)
        return list(map(lambda x: x.content[0].text, responses))
    return asyncio.run(async_chat_completion(prompts))

def call_together(prompt, model, max_retries=3, post_process=None):
    for attempt in range(max_retries):
        error = None
        try:
            response = get_together_client().chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1,
            )
            answer = response.choices[0].message.content
            if post_process:
                answer = post_process(answer)
            return answer
        except Exception as e:
            error = str(e)
        if attempt == max_retries - 1:
            raise Exception(f"Failed to get a response from Together: {error}")

def call_claude(prompt, model, max_retries=3, post_process=None):
    anthropic = Anthropic()
    
    for attempt in range(max_retries):
        error = None
        try:
            response = anthropic.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.content[0].text
            if post_process:
                answer = post_process(answer)
            return answer
        except Exception as e:
            error = str(e)
        if attempt == max_retries - 1:
            raise Exception(f"Failed to get a response from Claude: {error}")

def call_llm(prompt, model=None, max_retries=3, post_process=None):
    """Wrapper function to choose between OpenAI, Together, and Claude APIs"""
    match LLM_CLIENT:
        case "claude":
            return call_claude(prompt, model or DEFAULT_CLAUDE_MODEL, max_retries, post_process)
        case "together":
            return call_together(prompt, model or DEFAULT_TOGETHER_MODEL, max_retries, post_process)
        case _:  # default to openai
            return call_openai_async(prompt, model or DEFAULT_OPENAI_MODEL, max_retries, post_process)

def call_llm_batch(prompts, model=None, max_retries=3, post_process=None):
    match LLM_CLIENT:
        case "claude":
            return call_claude_batch(prompts, model or DEFAULT_CLAUDE_MODEL, max_retries, post_process)
        case "together":
            return call_together_batch(prompts, model or DEFAULT_TOGETHER_MODEL, max_retries, post_process)
        case _:  # default to openai
            return call_openai_batch(prompts, model or DEFAULT_OPENAI_MODEL, max_retries, post_process)

def process_text(story):
    """Common text processing logic for both endpoints"""
    if CLEAN_TEXT:
        clean_text_prompt = CLEAN_TEXT_PROMPT.format(story=story)
        story = call_llm(clean_text_prompt)
    return story

def get_sentences(story):
    """Common sentence parsing logic"""
    if not USE_SPACY:
        sentencizer_prompt = SENTENCIZE_PROMPT.format(story=story)
        return call_llm(sentencizer_prompt, post_process=robust_parse_answer)
    else:
        nlp = load_spacy_model()
        parsed_sentences = nlp(story).sents
        return list(map(str, parsed_sentences))

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/login')
def login():
    # Redirect to Google's OAuth 2.0 server for login.
    return google.authorize_redirect(url_for('authorize', _external=True))

@app.route('/authorize')
def authorize():
    if SKIP_LOGIN:
        session['logged_in'] = True
        return redirect(url_for('index'))
    
    # This route is called by Google after the user authenticates.
    # Exchange the authorization code for a token
    token = google.authorize_access_token()
    if token is None:
        return "Authorization failed.", 401

    # Fetch user info from Google
    resp = google.get('userinfo')
    user_info = resp.json()
    user_email = user_info.get('email')
    allowed_domain = "usc.edu"
    if not user_email.endswith(allowed_domain):
        return "Unauthorized", 401

    # Store user info in session
    session['logged_in'] = True
    session['user'] = {
        'email': user_info.get('email'),
        'name': user_info.get('name')
    }

    # Redirect back to the homepage or some protected page
    return redirect(url_for('index'))


@app.route('/api/ask', methods=['POST'])
def ask():    
    if not session.get('logged_in'):
        return jsonify({"message": "Unauthorized"}), 401

    data = request.json
    story = data.get('story')
    if not story:
        return jsonify({"message": "No story provided"}), 400

    def generate():
        request_id = id(request)
        app.analyzing_requests = getattr(app, 'analyzing_requests', set())
        app.analyzing_requests.add(request_id)
        
        try:
            # Process text
            current_story = process_text(story)
            if CLEAN_TEXT and request_id in app.analyzing_requests:
                yield f"data: {json.dumps({'type': 'clean_text', 'text': current_story})}\n\n"

            # Get sentences
            if request_id in app.analyzing_requests:
                parsed_sentences = get_sentences(current_story)
                yield f"data: {json.dumps({'type': 'sentences', 'sentences': parsed_sentences})}\n\n"

            # Process sentences in batches
            for i in range(0, len(parsed_sentences), BATCH_SIZE):
                if request_id not in app.analyzing_requests:
                    yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                    return

                batch = parsed_sentences[i:i + BATCH_SIZE]
                labeling_prompts = list(map(lambda x: LABELING_PROMPT.format(story=current_story, sentence=x), batch))
                results = call_llm_batch(labeling_prompts, max_retries=1)
                
                # Stream results
                for idx, result in zip(range(i, i + len(results)), results):
                    if request_id not in app.analyzing_requests:
                        yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                        return
                    if isinstance(result, str) and result.startswith("Error:"):
                        yield f"data: {json.dumps({'type': 'error', 'message': result, 'index': idx})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'analysis', 'index': idx, 'analysis': result})}\n\n"

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
        finally:
            if request_id in app.analyzing_requests:
                app.analyzing_requests.remove(request_id)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/ask_static', methods=['POST'])
def ask_static():    
    if not session.get('logged_in'):
        return jsonify({"message": "Unauthorized"}), 401

    data = request.json
    story = data.get('story')

    if not story:
        return jsonify({"message": "No story provided"}), 400
    
    # Process text and get sentences
    current_story = process_text(story)
    parsed_sentences = get_sentences(current_story)

    # Analyze all sentences
    analysis = []
    for sentence in parsed_sentences:
        try:
            labeling_prompt = LABELING_PROMPT.format(sentence=sentence)
            answer = call_llm(labeling_prompt, max_retries=1)
            analysis.append(answer)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({
        "analysis": analysis,
        "sentences": parsed_sentences
    }), 200

@app.route('/api/stop', methods=['POST'])
def stop_analysis():
    request_id = id(request)
    if hasattr(app, 'analyzing_requests') and request_id in app.analyzing_requests:
        app.analyzing_requests.remove(request_id)
    return jsonify({"status": "stopped"}), 200

if __name__ == '__main__':
    # Run the Flask app
    # In production, run behind a WSGI server (e.g., gunicorn) and use HTTPS.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)#, debug=True)
