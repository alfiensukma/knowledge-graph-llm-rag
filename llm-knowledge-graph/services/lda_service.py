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
    """
    LDA = Bag-of-Words -> LDA (sklearn).
    - Use min_df/max_df to remove too common/rare words.
    - doc_terms calculated from document topic mixture against word distribution per topic.
    """

    def __init__(
        self,
        n_topics: int = 10,
        n_top_terms_per_doc: int = 12,
        max_features: int = 20000,
        stopwords_lang: str = "english",
        random_state: int = 42,
        ngram_range=(1, 2),
        min_df: int | float = 2,
        max_df: float = 0.9,
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

        # 2) CountVectorizer
        vec = CountVectorizer(
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

        # 3) LDA fit
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

        # 4) Normalize β per topic
        topic_term_norm = topic_term / (topic_term.sum(axis=1, keepdims=True) + 1e-12)

        # 5) doc_terms from document topic mixture
        doc_terms = []
        top_k_doc = min(self.n_top_terms_per_doc, len(terms))
        for i, fn in enumerate(filenames):
            theta = doc_topic[i]
            theta = theta / (theta.sum() + 1e-12)
            doc_term_dist = theta @ topic_term_norm  # (n_terms,)
            idx = np.argsort(-doc_term_dist)[:top_k_doc]
            terms_i = [(terms[j], float(doc_term_dist[j])) for j in idx]
            doc_terms.append({"filename": fn, "model": "LDA", "terms": terms_i})

        # 6) Top words per topic
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
