from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"ISSN:?\s*\d{4}-\d{4}", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class LSAService:
    """
    LSA = TF-IDF -> SVD. 
    - Top words tiap topik berdasarkan |loading| (absolut) agar istilah dengan loading negatif tetap tertangkap.
    - Top terms tiap dokumen dihitung dari aproksimasi X_hat = UΣV^T (di sini: doc_topic @ topic_term), lalu dinormalisasi.
    """

    def __init__(
        self,
        n_topics: int = 10,
        n_top_terms_per_doc: int = 12,
        max_features: int = 20000,
        stopwords_lang: str = "english",
        random_state: int = 42,
        ngram_range=(1, 2),
        min_df: int | float = 2,     # remove words that are too rare
        max_df: float = 0.9,         # remove words that are too common
    ):
        self.n_topics = n_topics
        self.n_top_terms_per_doc = n_top_terms_per_doc
        self.max_features = max_features
        self.stopwords_lang = stopwords_lang
        self.random_state = random_state
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

    def run(self, pdf_texts: Dict[str, str]) -> Dict[str, Any]:
        # 1) Prepare documents
        filenames, docs = [], []
        for fn, txt in pdf_texts.items():
            c = _clean_text(txt)
            if c:
                filenames.append(fn)
                docs.append(c)
        if not docs:
            return {"doc_terms": [], "topics": [], "n_docs": 0, "n_topics": 0}

        # 2) TF-IDF
        vec = TfidfVectorizer(
            stop_words=self.stopwords_lang,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        X = vec.fit_transform(docs)
        terms = vec.get_feature_names_out()
        if X.shape[1] == 0:
            return {"doc_terms": [], "topics": [], "n_docs": len(filenames), "n_topics": 0}

        # 3) SVD
        # Adjust number of topics if needed
        max_possible_topics = min(self.n_topics, max(1, min(X.shape[0], X.shape[1])))
        n_topics_eff = max(1, max_possible_topics)
        svd = TruncatedSVD(n_components=n_topics_eff, random_state=self.random_state)
        doc_topic = svd.fit_transform(X)          # shape: (n_docs, k)
        topic_term = svd.components_              # shape: (k, n_terms)

        # 4) Compute document-term scores
        doc_term_scores = doc_topic @ topic_term  # (n_docs, n_terms)
        doc_term_scores = np.abs(doc_term_scores) # absolute values
        doc_term_scores = normalize(doc_term_scores, norm="l2", axis=1)

        # 5) Get top terms per document
        doc_terms = []
        top_k_doc = min(self.n_top_terms_per_doc, len(terms))
        for i, fn in enumerate(filenames):
            row = doc_term_scores[i]
            idx = np.argsort(-row)[:top_k_doc]
            terms_i = [(terms[j], float(row[j])) for j in idx]
            doc_terms.append({"filename": fn, "model": "LSA", "terms": terms_i})

        # 6) Get top words per topic based on |loading|
        topics = []
        top_k_topic = min(self.n_top_terms_per_doc, len(terms))
        for k in range(n_topics_eff):
            loadings = topic_term[k]
            idx = np.argsort(-np.abs(loadings))[:top_k_topic]
            topics.append({
                "topic_id": k,
                "top_words": [terms[j] for j in idx],
                "weights": [float(loadings[j]) for j in idx],
            })

        print(f"Generated {n_topics_eff} LSA topics from {len(filenames)} documents")
        return {
            "doc_terms": doc_terms,
            "topics": topics,
            "n_docs": len(filenames),
            "n_topics": n_topics_eff,
        }
