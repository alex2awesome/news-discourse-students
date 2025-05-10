from flask import session, request, jsonify, render_template, redirect, g
import json
from flask import Response
import os
from flask import Blueprint
from app.utils import process_text, get_sentences, call_llm, call_llm_batch
from flask import current_app
from .prompts import LABELING_PROMPT, COMPARISON_PROMPT
from .config.default import Config
from .retriever import SimpleRetriever
import logging
from threading import Lock

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

# Lazy initialization of retrievers
_retrievers = {}
def get_retriever(source='annenberg'):
    global _retrievers
    if source not in _retrievers:
        logger.info(f"Initializing FAISS retriever for {source}...")
        _retrievers[source] = SimpleRetriever()
        index_path = os.path.join(os.path.dirname(__file__), f"{source}_index")
        logger.info(f"Looking for index at: {index_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Directory contents: {os.listdir(os.path.dirname(__file__))}")
        if os.path.exists(index_path):
            logger.info(f"Loading FAISS index from {index_path}")
            _retrievers[source].load(index_path)
            logger.info(f"{source} FAISS index loaded successfully")
        else:
            logger.warning(f"FAISS index not found at {index_path}")
    return _retrievers[source]

# Create a thread-safe dictionary to track requests
active_requests = {}
request_lock = Lock()

@main_bp.route('/')
def index():
    return render_template('login.html')

@main_bp.route('/analysis')
def analysis_page():
    if not session.get('logged_in'):
        return redirect('/')
    layout = request.args.get('layout', 'default')
    if layout == 'right-split':
        return render_template('analysis-right-side.html')
    return render_template('analysis.html')

@main_bp.route('/api/ask', methods=['GET', 'POST'])
def ask():    
    if not session.get('logged_in'):
        return jsonify({"message": "Unauthorized"}), 401

    # Get story from either POST data or query parameters
    story = None
    source = 'annenberg'  # Default source
    
    if request.method == 'POST':
        data = request.json
        story = data.get('story')
        source = data.get('source', 'annenberg')
    else:  # GET
        story = request.args.get('story')
        source = request.args.get('source', 'annenberg')

    if not story:
        return jsonify({"message": "No story provided"}), 400

    # Create request ID and store it
    request_id = id(request)
    with request_lock:
        active_requests[request_id] = True
    
    def generate():
        try:
            # Process text
            current_story = process_text(story)
            yield f"data: {json.dumps({'type': 'clean_text', 'text': current_story})}\n\n"

            # Get sentences
            parsed_sentences = get_sentences(current_story)
            yield f"data: {json.dumps({'type': 'sentences', 'sentences': parsed_sentences})}\n\n"

            # Find similar articles
            try:
                retriever = get_retriever(source)
                similar_articles = retriever.search(current_story, top_k=3)
                yield f"data: {json.dumps({'type': 'similar_articles', 'articles': similar_articles, 'source': source})}\n\n"
            except Exception as e:
                logger.error(f"Error finding similar articles: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Error finding similar articles: {str(e)}'})}\n\n"

            # Process sentences in batches
            for i in range(0, len(parsed_sentences), Config.BATCH_SIZE):
                with request_lock:
                    if request_id not in active_requests:
                        yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                        return

                # Check if request is still active before processing batch
                with request_lock:
                    if request_id not in active_requests:
                        yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                        return

                batch = parsed_sentences[i:i + Config.BATCH_SIZE]
                labeling_prompts = [LABELING_PROMPT.format(story=current_story, sentence=x) for x in batch]
                results = call_llm_batch(labeling_prompts, max_retries=1)
                
                # Stream results
                for idx, result in zip(range(i, i + len(results)), results):
                    with request_lock:
                        if request_id not in active_requests:
                            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                            return
                    if isinstance(result, str) and result.startswith("Error:"):
                        yield f"data: {json.dumps({'type': 'error', 'message': result, 'index': idx})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'analysis', 'index': idx, 'analysis': result})}\n\n"

            # Final check before completing
            with request_lock:
                if request_id not in active_requests:
                    yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                    return
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
        finally:
            with request_lock:
                if request_id in active_requests:
                    del active_requests[request_id]

    return Response(generate(), mimetype='text/event-stream')

@main_bp.route('/api/ask_static', methods=['POST'])
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

@main_bp.route('/api/stop', methods=['POST'])
def stop_analysis():
    request_id = id(request)
    with request_lock:
        if request_id in active_requests:
            del active_requests[request_id]
    return jsonify({"status": "stopped"}), 200

@main_bp.route('/api/find_similar', methods=['POST'])
def find_similar():
    if not session.get('logged_in'):
        return jsonify({"message": "Unauthorized"}), 401

    data = request.json
    story = data.get('story')
    source = data.get('source', 'annenberg')  # Default to annenberg if not specified
    
    if not story:
        return jsonify({"message": "No story provided"}), 400

    try:
        # Get retriever only when needed
        retriever = get_retriever(source)
        results = retriever.search(story, top_k=3)
        return jsonify({"articles": results, "source": source}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@main_bp.route('/api/compare_articles', methods=['POST'])
def compare_articles():
    if not session.get('logged_in'):
        return jsonify({"message": "Unauthorized"}), 401

    try:
        data = request.json
        student_article = data.get('student_article')
        professional_article = data.get('professional_article')

        if not student_article or not professional_article:
            return jsonify({"message": "Both articles are required"}), 400

        # Format the prompt using the imported template
        prompt = COMPARISON_PROMPT.format(
            student_article=student_article,
            professional_article=professional_article
        )

        # Call the LLM using the existing function
        response = call_llm(prompt, max_retries=1)
        return jsonify({
            "comparison": response,
            "prompt": prompt
        }), 200
    except Exception as e:
        logger.error(f"Error in compare_articles: {str(e)}")
        return jsonify({"message": str(e)}), 500

@main_bp.route('/api/check_login', methods=['GET'])
def check_login():
    if not session.get('logged_in'):
        return jsonify({"message": "Unauthorized"}), 401
    return jsonify({"message": "Authorized"}), 200

