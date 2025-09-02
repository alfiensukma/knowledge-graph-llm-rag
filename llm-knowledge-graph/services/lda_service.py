from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"ISSN:?\s*\d{4}-\d{4}", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class LDAService:
    def __init__(
        self,
        n_topics: int,
        n_top_terms_per_doc: int,
        max_features: int,
        stopwords_lang: str = "english",
        custom_stopwords: List[str] | None = None,
        random_state: int = 0,
        ngram_range=(1, 2),
        min_df: int | float = 0,
        max_df: float = 0,
    ):
        self.n_topics = n_topics
        self.n_top_terms_per_doc = n_top_terms_per_doc
        self.max_features = max_features
        self.stopwords_lang = stopwords_lang
        self.custom_stopwords = custom_stopwords
        self.random_state = random_state
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

    def run(self, pdf_texts: Dict[str, str]) -> Dict[str, Any]:
        filenames, docs = [], []
        for fn, txt in pdf_texts.items():
            c = _clean_text(txt)
            if c:
                filenames.append(fn)
                docs.append(c)
        if not docs:
            return {"doc_terms": [], "topics": [], "n_docs": 0, "n_topics": 0}
        
        stop_words = list(ENGLISH_STOP_WORDS.union(self.custom_stopwords or []))

        vec = CountVectorizer(
            stop_words=stop_words,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        X = vec.fit_transform(docs)
        terms = vec.get_feature_names_out()
        if X.shape[1] == 0:
            return {"doc_terms": [], "topics": [], "n_docs": len(filenames), "n_topics": 0}

        # LDA fit
        # safe limit: k <= number of features, and minimum 1
        max_possible_topics = min(self.n_topics, max(1, min(X.shape[0], X.shape[1])))
        n_topics_eff = max(1, max_possible_topics)

        lda = LatentDirichletAllocation(
            n_components=n_topics_eff,
            learning_method="batch",
            random_state=self.random_state,
        )
        doc_topic = lda.fit_transform(X)  # θ_dk
        topic_term = lda.components_      # β_kv (unnormalized)

        # Normalize β per topic
        topic_term_norm = topic_term / (topic_term.sum(axis=1, keepdims=True) + 1e-12)

        # doc_terms
        doc_terms = []
        top_k_doc = min(self.n_top_terms_per_doc, len(terms))
        for i, fn in enumerate(filenames):
            theta = doc_topic[i]
            theta = theta / (theta.sum() + 1e-12)
            doc_term_dist = theta @ topic_term_norm  # (n_terms,)
            idx = np.argsort(-doc_term_dist)[:top_k_doc]
            terms_i = [(terms[j], float(doc_term_dist[j])) for j in idx]
            doc_terms.append({
                "filename": fn,
                "model": "LDA",
                "terms": terms_i,
                "distribution": theta.tolist()
            })

        # Top words per topic
        topics = []
        top_k_topic = min(self.n_top_terms_per_doc, len(terms))
        for k in range(n_topics_eff):
            tt = topic_term_norm[k]
            idx = np.argsort(-tt)[:top_k_topic]
            topics.append({
                "topic_id": k,
                "top_words": [terms[j] for j in idx],
                "weights": [float(tt[j]) for j in idx],
            })

        print(f"Generated {n_topics_eff} LDA topics from {len(filenames)} documents")
        return {
            "doc_terms": doc_terms,
            "topics": topics,
            "n_docs": len(filenames),
            "n_topics": n_topics_eff,
        }
