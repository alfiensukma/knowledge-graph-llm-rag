# services/lda_service.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"ISSN:?\s*\d{4}-\d{4}", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class LDAService:
    def __init__(self, n_topics: int = 10, n_top_terms_per_doc: int = 12,
                 max_features: int = 20000, stopwords_lang: str = "english",
                 random_state: int = 42):
        self.n_topics = n_topics
        self.n_top_terms_per_doc = n_top_terms_per_doc
        self.max_features = max_features
        self.stopwords_lang = stopwords_lang
        self.random_state = random_state

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

        # CountVectorizer
        vec = CountVectorizer(stop_words=self.stopwords_lang,
                              max_features=self.max_features)
        X = vec.fit_transform(docs)
        terms = vec.get_feature_names_out()

        # LDA fit
        n_topics_eff = max(2, min(self.n_topics, min(X.shape[0], X.shape[1])))
        lda = LatentDirichletAllocation(n_components=n_topics_eff,
                                        learning_method="batch",
                                        random_state=self.random_state)
        doc_topic = lda.fit_transform(X)
        topic_term = lda.components_

        # distribusi term per dokumen
        topic_term_norm = topic_term / (topic_term.sum(axis=1, keepdims=True) + 1e-12)
        doc_terms = []
        for i, fn in enumerate(filenames):
            weights = doc_topic[i]
            weights = weights / (weights.sum() + 1e-12)
            doc_term_dist = weights @ topic_term_norm
            idx = np.argsort(-doc_term_dist)[: self.n_top_terms_per_doc]
            terms_i = [(terms[j], float(doc_term_dist[j])) for j in idx]
            doc_terms.append({"filename": fn, "model": "LDA", "terms": terms_i})

        topics = []
        for k in range(n_topics_eff):
            tt = topic_term_norm[k]
            idx = np.argsort(-tt)[: self.n_top_terms_per_doc]
            topics.append({
                "topic_id": k,
                "top_words": [terms[j] for j in idx],
                "weights": [float(tt[j]) for j in idx]
            })

        return {"doc_terms": doc_terms, "topics": topics, "n_docs": len(filenames), "n_topics": n_topics_eff}
