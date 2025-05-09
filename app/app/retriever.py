import faiss
import numpy as np
import pandas as pd
import pickle
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from sentence_transformers import SentenceTransformer
import torch


class SimpleRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Get the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize model with device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.doc_ids = []
        self.sentences = []
        self.labels = []
        self.headlines = []
        self.subheaders = []
        self.justifications = []
        self.urls = []
        self.embeddings = None

    def build(
        self, 
        dataframe: pd.DataFrame, 
        text_col="article_text", 
        id_col="doc_id", 
        sentences_col="sent", 
        labels_col="label",
        headline_col="article_headline",
        subheader_col="article_subheader",
        justification_col="justification",
        url_col="article_url",
        normalize=True
    ):
        texts = dataframe[text_col].tolist()
        self.doc_ids = dataframe[id_col].tolist()
        self.sentences = dataframe[sentences_col].tolist() if sentences_col else None
        self.labels = dataframe[labels_col].tolist() if labels_col else None
        self.headlines = dataframe[headline_col].tolist() if headline_col in dataframe.columns else None
        self.subheaders = dataframe[subheader_col].tolist() if subheader_col in dataframe.columns else None
        self.justifications = dataframe[justification_col].tolist() if justification_col in dataframe.columns else None
        self.urls = dataframe[url_col].tolist() if url_col in dataframe.columns else None
        self.embeddings = self.model.encode(texts, normalize_embeddings=normalize)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity if normalized
        self.index.add(np.array(self.embeddings).astype("float32"))

    def search(self, query: str, top_k: int = 5):
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_vec).astype("float32"), top_k)
        results = []
        for i, score in zip(indices[0], scores[0]):
            if i == -1:
                continue
            results.append({
                "id": self.doc_ids[i],
                "score": float(score),
                "sentences": self.sentences[i] if self.sentences else None,
                "labels": self.labels[i] if self.labels else None,
                "headline": self.headlines[i] if self.headlines else None,
                "subheader": self.subheaders[i] if self.subheaders else None,
                "justification": self.justifications[i] if self.justifications else None,
                "url": self.urls[i] if self.urls else None,
            })
        return results

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(dir_path, "faiss.index"))

        # Save metadata
        metadata = {
            "doc_ids": self.doc_ids,
            "sentences": self.sentences,
            "labels": self.labels,
            "headlines": self.headlines,
            "subheaders": self.subheaders,
            "justifications": self.justifications,
            "urls": self.urls
        }
        with open(os.path.join(dir_path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    def load(self, dir_path: str):
        self.index = faiss.read_index(os.path.join(dir_path, "faiss.index"))

        with open(os.path.join(dir_path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            self.doc_ids = metadata["doc_ids"]
            self.sentences = metadata["sentences"]
            self.labels = metadata["labels"]
            self.headlines = metadata.get("headlines")
            self.subheaders = metadata.get("subheaders")
            self.justifications = metadata.get("justifications")
            self.urls = metadata.get("urls")


"""
# 
# example usage to build and save the index
# 

# annenberg 
from retriever import SimpleRetriever
import pandas as pd
df =  pd.read_json('../../data/for_retriv/annenberg-full-articles-w-headline.jsonl', lines=True)
retriever = SimpleRetriever()
retriever.build(df)
retriever.save("annenberg_index")
results = retriever.search("media bias in US elections", top_k=3)


# latimes 
from retriever import SimpleRetriever
import pandas as pd
df =  pd.read_json('../../data/for_retriv/latimes-full-articles.jsonl', lines=True)
retriever = SimpleRetriever()
retriever.build(df)
retriever.save("latimes_index")
results = retriever.search("media bias in US elections", top_k=3)
"""