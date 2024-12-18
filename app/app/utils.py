import re
import json
import ast
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from together import AsyncTogether
from . import (
    USE_SPACY, CLEAN_TEXT,
    LLM_CLIENT, DEFAULT_CLAUDE_MODEL, DEFAULT_TOGETHER_MODEL, DEFAULT_OPENAI_MODEL,
    load_spacy_model
)
from .prompts import SENTENCIZE_PROMPT, CLEAN_TEXT_PROMPT
import spacy
import os 
from .prompts import LABELING_PROMPT


BATCH_SIZE = os.environ.get('BATCH_SIZE', 2)
CLEAN_TEXT = os.environ.get('CLEAN_TEXT', False)


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


def generate(story, request_id, analyzing_requests):
    analyzing_requests.add(request_id)
    try:
        # Process text
        current_story = process_text(story)
        if CLEAN_TEXT and request_id in analyzing_requests:
            yield f"data: {json.dumps({'type': 'clean_text', 'text': current_story})}\n\n"

        # Get sentences
        if request_id in analyzing_requests:
            parsed_sentences = get_sentences(current_story)
            yield f"data: {json.dumps({'type': 'sentences', 'sentences': parsed_sentences})}\n\n"

        # Process sentences in batches
        for i in range(0, len(parsed_sentences), BATCH_SIZE):
            if request_id not in analyzing_requests:
                yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                return

            batch = parsed_sentences[i:i + BATCH_SIZE]
            labeling_prompts = list(map(lambda x: LABELING_PROMPT.format(story=current_story, sentence=x), batch))
            results = call_llm_batch(labeling_prompts, max_retries=1)
            
            for idx, result in zip(range(i, i + len(results)), results):
                if request_id not in analyzing_requests:
                    yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                    return
                if isinstance(result, str) and result.startswith("Error:"):
                    yield f"data: {json.dumps({'type': 'error', 'message': result, 'index': idx})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'analysis', 'index': idx, 'analysis': result})}\n\n"

        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    finally:
        if request_id in analyzing_requests:
            analyzing_requests.remove(request_id)

