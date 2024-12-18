import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata

import os
import json
import torch
import ast
import logging
import re 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
import os
here = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f'{here}/../../config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

BATCH_SIZE = 500
CLEAN_ARTICLE_TEXT_PROMPT = """
You are a helpful editor assistant. I scraped this news article from the web and removed HTML tags, however, 
there is a lot of extraneous text from the HTML page that remains. Please only extract text related to the news article.
Do NOT miss any article text. Here is the news article:

```{article_text}```

Return only the CLEANED text of the news article, nothing else.
"""

DISCOURSE_PROMPT = """
You will receive a sentence from a {document_type}. Your task is to identify the structural role that this sentence plays in the overall {document_type}.
Please generate a generalizable keyword label to describe the structural role. The label must be as generalizable as possible and should not be document-specific. 

{definitions}

For reference, I'll also give you the full {document_type}.

[Examples]
Example 1:

Sentence: "This annual festival, which welcomes and celebrates large, gay men -- 'bears' -- on the tip of Cape Cod, ended Saturday."
Your response: 
"Main Event": This sentence directly describes a primary event, noting its conclusion, which is the focal point of the document.

Example 2:
Sentence: "But the league has also become one of the most prestigious summer internships for aspiring broadcasters."
Your response:
"Current Context": This sentence explains the background that contributes to the league's reputation, making it a relevant aspect of the broader discussion within the document.

Example 3:
Sentence: "Photo by Alex Lam)  By   Chuck White May 08, 2024 at  8:35 am PDT."
Your response:
"Error": This sentence was misparsed from the original news article and does not serve a discourse role in the overall article.


[Instructions]
Again, determine a generalizable structure labels in the document.

Each keyword label must reflect a SINGLE structural feature instead of a combination of features.
If you add a new label, add a SHORT GENERAL LABEL and a DESCRIPTION OF THAT LABEL.
If the sentence is an error, return "Error". 
Return in the same format as the examples: "LABEL": DESCRIPTION. Please feel free to add a new label if it helps describe a sentence.

Now it's your turn. Here's a {document_type}:

[{document_type}]
```{document}```

What role does the following sentence play in the {document_type}?

[Sentence]
```{sentence}```

Please only state your response as "LABEL": DESCRIPTION. Say nothing else."""

def robust_extract_json_str(lm_string):
    
    if not lm_string:
        return None
    # Use regular expressions to search for list brackets across multiple lines
    match = re.search(r'\[.*?\]', lm_string, re.DOTALL)
    if match:
        lm_string = match.group(0)
    try:
        return json.loads(lm_string)
    except:
        try:
            return ast.literal_eval(lm_string)
        except:
            pass
    logging.error(f"Could not extract json string from: {lm_string}")
    return None


def format_prompt(prompt: str, json_str: str) -> str:
    message = [
        {
            "role": "system",
            "content": "You are an experienced journalist.",
        },

        {
            "role": "user",
            "content": prompt.format(json_str=json_str)
        },
    ]
    formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return formatted_prompt

_spacy_model = None
def load_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        import spacy
        _spacy_model = spacy.load("en_core_web_lg")
        _spacy_model.disable_pipes(*[pipe for pipe in _spacy_model.pipe_names if pipe != 'senter'])
        _spacy_model.add_pipe('sentencizer')
    return _spacy_model


def sentencize_text_column(s):
    nlp = load_spacy_model()
    all_sents = []
    for doc in tqdm(nlp.pipe(s.tolist(), batch_size=100, n_process=1), total=len(s)):
        all_sents.append(list(map(str, doc.sents)))
    return all_sents


def load_model(model_name: str):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        max_model_len=60_000
    )
    return tokenizer, model


def write_to_file(fname, urls, outputs, id_col_name='key', response_col_name='response'):
    with open(fname, 'wb') as file:
        for url, output in zip(urls, outputs):
            response = output.outputs[0].text
            response = unicodedata.normalize('NFKC', response)
            if response and url:
                output = {}
                output[id_col_name] = str(url)
                output[response_col_name] = str(response)
                file.write(json.dumps(output).encode('utf-8'))
                file.write(b'\n')


def batchify_dataframe(df, batch_size):
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def get_text_and_sources(text_fname, summaries_fname):
    cleaned_articles = pd.read_json(text_fname, lines=True).rename(columns={'response': 'article_text'})
    source_summaries = pd.read_json(summaries_fname, lines=True).rename(columns={'response': 'source_summaries'})
    combined_df = cleaned_articles.merge(source_summaries, on='url')
    combined_df['target_sources'] = (
        combined_df['source_summaries']
            .apply(robust_extract_json_str)
            .apply(lambda x: [s['Name'] for s in x] if x else [])
            .apply(lambda x: ', '.join(map(lambda y: '"%s"' % y , x)))
    )
    return combined_df


def sentencize_text(text):
    return [sent for sent in text.split('.') if sent]

# def write_batch_prompt(prompt_col, key_col):
#     url_batch = prompt_df['url'].tolist()
#     message_batch = prompt_df['message'].tolist()
#     return url_batch, message_batch



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--data_dir', type=str, default=f'{here}/../data')
    parser.add_argument('--source_data_file', type=str, default='full-source-scored-data.jsonl')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--input_data_file', type=str, default=None)
    parser.add_argument('--id_col', type=str, default='article_url')
    parser.add_argument('--text_col', type=str, default='article_text')
    parser.add_argument('--output_file', type=str, default='sources_data_70b.txt')
    parser.add_argument('--write_prompt_only', action='store_true')
    # prompts
    parser.add_argument('--do_article_gen', action='store_true')
    parser.add_argument('--do_discourse', action='store_true')

    args = parser.parse_args()
    article_df = pd.read_csv(args.input_data_file, index_col=0)

    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(article_df)
    
    # load the model
    sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)
    tokenizer, model = load_model(args.model)
    # store each article_url, annoted_sentences pair
    # hold the batches
    url_batches, message_batches = [], []
    df_batches = batchify_dataframe(article_df, BATCH_SIZE)

    # generate the summaries
    start_idx = args.start_idx
    end_idx = start_idx + BATCH_SIZE
    for df in tqdm(df_batches):
        dirname = os.path.dirname(args.output_file)
        if (dirname != '') and not os.path.exists(dirname):
            os.makedirs(dirname)

        out_dirname, out_fname = os.path.split(args.output_file)
        if (out_dirname != ''):
            os.makedirs(out_dirname, exist_ok=True)
            if args.do_article_gen:
                os.makedirs(f'{out_dirname}/article_text', exist_ok=True)
            if args.do_discourse:
                os.makedirs(f'{out_dirname}/news-discourse', exist_ok=True)

        fname, fext = os.path.splitext(out_fname)
        text_fname = f'{out_dirname}/article_text/{fname}__article_text__{start_idx}_{end_idx}{fext}'
        discourse_fname = f'{out_dirname}/news-discourse/{fname}__news-discourse__{start_idx}_{end_idx}{fext}'

        # clean the article text
        if args.do_article_gen and not os.path.exists(text_fname):
            # write to article as marker 
            with open(text_fname, 'w') as f:
                f.write('')
            logging.info(f"Generating cleaned article text for batch {start_idx} to {end_idx}")
            clean_prompts = df[args.text_col].apply(lambda x: CLEAN_ARTICLE_TEXT_PROMPT.format(article_text=x)).tolist()
            cleaned_article_outputs = model.generate(clean_prompts, sampling_params)
            write_to_file(text_fname, df[args.id_col], cleaned_article_outputs)

        # generate discourse
        if args.do_discourse and not os.path.exists(discourse_fname):
            with open(discourse_fname, 'w') as f:
                f.write('')                
            logging.info(f"Generating discourse for batch {start_idx} to {end_idx}")
            if not os.path.exists(text_fname):
                raise ValueError("You need to generate the cleaned article text first.")
            cleaned_articles = pd.read_json(text_fname, lines=True)
            cleaned_articles['response'] = cleaned_articles['response'].str.replace('`','').str.strip()
            cleaned_articles['sentences'] = cleaned_articles['response'].pipe(sentencize_text_column)
            
            # explode cleaned_articles and generate a unique key for each sentence
            cleaned_articles_sents = cleaned_articles.explode('sentences')
            cleaned_articles_sents['sent_idx'] = cleaned_articles_sents.groupby('key').cumcount()
            cleaned_articles_sents['key'] = cleaned_articles_sents['key'] + '__' + cleaned_articles_sents['sent_idx'].astype(str)

            discourse_prompts = cleaned_articles_sents.apply(lambda x: DISCOURSE_PROMPT.format(
                document_type='news article', 
                sentence=x['sentences'],
                document=x['response'],
                definitions='',
                ), axis=1).tolist()
            discourse_outputs = model.generate(discourse_prompts, sampling_params)
            write_to_file(discourse_fname, cleaned_articles_sents['key'], discourse_outputs, id_col_name='key', response_col_name='discourse_label')
        
        # update the indices
        start_idx = end_idx
        end_idx = start_idx + BATCH_SIZE


"""
    python score_discourse.py \
      --start_idx 0 \
      --end_idx 5000 \
      --id_col url \
      --text_col article_text \
      --input_data_file ../../data/batch_article_text.csv \
      --output_file  ../../data/v3_discourse_summaries/extracted_discourse.jsonl \
      --do_article_gen \
      --do_discourse

"""        