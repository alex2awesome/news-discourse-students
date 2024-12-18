from flask import session, request, jsonify, render_template, redirect, g
import json
from flask import Response
import os
from flask import Blueprint
from .utils import process_text, get_sentences, call_llm, call_llm_batch, generate
from flask import current_app


main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('login.html')

@main_bp.route('/analysis')
def analysis_page():
    if not session.get('logged_in'):
        return redirect('/')
    return render_template('analysis.html')

@main_bp.route('/api/ask', methods=['POST'])
def ask():    
    if not session.get('logged_in'):
        return jsonify({"message": "Unauthorized"}), 401
    data = request.json
    story = data.get('story')
    if not story:
        return jsonify({"message": "No story provided"}), 400
    with current_app.app_context():
        analyzing_requests = current_app.analyzing_requests
        return Response(generate(story, id(request), analyzing_requests), mimetype='text/event-stream')

@main_bp.route('/api/stop', methods=['POST'])
def stop_analysis():
    request_id = id(request)
    with current_app.app_context():
        analyzing_requests = current_app.analyzing_requests
        if request_id in analyzing_requests:
            analyzing_requests.remove(request_id)
    return jsonify({"status": "stopped"}), 200

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

