# services/lsa_service.py
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
    def __init__(self, n_topics: int = 10, n_top_terms_per_doc: int = 12,
                 max_features: int = 20000, stopwords_lang: str = "english",
                 random_state: int = 42, ngram_range=(1,2)):
        self.n_topics = n_topics
        self.n_top_terms_per_doc = n_top_terms_per_doc
        self.max_features = max_features
        self.stopwords_lang = stopwords_lang
        self.random_state = random_state
        self.ngram_range = ngram_range

    def run(self, pdf_texts: Dict[str, str]) -> Dict[str, Any]:
        # prepare documents
        filenames, docs = [], []
        for fn, txt in pdf_texts.items():
            c = _clean_text(txt)
            if c:
                filenames.append(fn)
                docs.append(c)
        if not docs:
            return {"doc_terms": [], "topics": []}

        # TF-IDF
        vec = TfidfVectorizer(stop_words=self.stopwords_lang,
                              max_features=self.max_features,
                              ngram_range=self.ngram_range)
        X = vec.fit_transform(docs)
        terms = vec.get_feature_names_out()

        # SVD
        max_possible_topics = min(X.shape[0], X.shape[1], self.n_topics)
        n_topics_eff = max(1, max_possible_topics) 
        svd = TruncatedSVD(n_components=n_topics_eff, random_state=self.random_state)
        doc_topic = svd.fit_transform(X)
        topic_term = svd.components_

        # Normalisasi skor
        doc_term_scores = doc_topic @ topic_term
        doc_term_scores = np.abs(doc_term_scores)
        doc_term_scores = normalize(doc_term_scores, norm="l2", axis=1)

        doc_terms = []
        for i, fn in enumerate(filenames):
            row = doc_term_scores[i]
            top_k = min(self.n_top_terms_per_doc, len(terms))
            idx = np.argsort(-row)[:top_k]
            terms_i = [(terms[j], float(row[j])) for j in idx]
            doc_terms.append({"filename": fn, "model": "LSA", "terms": terms_i})

        topics = []
        for k in range(n_topics_eff):
            top_k = min(self.n_top_terms_per_doc, len(terms))
            idx = np.argsort(-topic_term[k])[:top_k]
            topics.append({
                "topic_id": k,
                "top_words": [terms[j] for j in idx],
                "weights": [float(topic_term[k, j]) for j in idx]
                })

        print(f"Generated {n_topics_eff} topics from {len(docs)} documents")
        return {
            "doc_terms": doc_terms, 
            "topics": topics, 
            "n_docs": len(filenames), 
            "n_topics": n_topics_eff
        }