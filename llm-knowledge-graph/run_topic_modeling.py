import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from services.lsa_service import LSAService
from services.lda_service import LDAService

DOCS_PATH = os.path.join("data", "pdfs")

def _clean_page(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def load_pdfs(folder: str) -> Dict[str, str]:
    loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    docs = loader.load()
    pdf_docs: Dict[str, List[str]] = {}
    for d in docs:
        fn = os.path.basename(d.metadata["source"])
        pdf_docs.setdefault(fn, [])
        if d.page_content:
            pdf_docs[fn].append(_clean_page(d.page_content))
    return {fn: "\n".join(pages) for fn, pages in pdf_docs.items()}


def choose_files(pdfs: Dict[str, str]) -> Dict[str, str]:
    if not pdfs:
        print(" > No PDFs found in:", DOCS_PATH)
        return {}
    
    pdf_info = []
    for name in sorted(pdfs.keys()):
        file_path = os.path.join(DOCS_PATH, name)
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            size_mb = len(pdfs[name]) / (1024 * 1024 * 0.5)
        pdf_info.append({"name": name, "size_mb": size_mb})
    
    print("\n" + "=" * 90)
    print("PDF FILES (choose files to run topic modeling)")
    print("=" * 90)
    for i, info in enumerate(pdf_info, 1):
        print(f"{i:2d}. {info['name']:<75} {info['size_mb']:6.2f}MB")
    print("=" * 90)
    print(f"Total: {len(pdf_info)}")
    
    raw = input("Enter PDF number (or 'q' to quit): ").strip()
    
    idxs = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            j = int(token)
            if 1 <= j <= len(pdf_info):
                idxs.add(j-1)
    
    if not idxs:
        print(" > Nothing selected; exiting.")
        return {}
    
    selected = {pdf_info[j]["name"]: pdfs[pdf_info[j]["name"]] for j in sorted(idxs)}
    selected_names = [pdf_info[j]["name"] for j in sorted(idxs)]
    print(f"\nSelected: {', '.join(selected_names)}")
    return selected


def print_model_results(model_name: str, results: Dict[str, Any], top_terms_preview: int = 10):
    print(f"\n=== {model_name} Results ===")
    print(f"Docs: {results.get('n_docs', 0)} | Topics: {results.get('n_topics', 0)}")

    # Topics
    topics = results.get("topics", [])
    if topics:
        print("\nGenerated Topics:")
        for topic in topics:
            words = topic.get("top_words", [])
            weights = topic.get("weights", [])
            print(f"\nTopic {topic['topic_id']}:")
            for w, ww in zip(words[:top_terms_preview], weights[:top_terms_preview]):
                print(f"  - {w}: {ww:.4f}")
    else:
        print("\n(no topics)")

    # Terms per document
    docs_terms = results.get("doc_terms", [])
    if docs_terms:
        print("\nDocument Terms:")
        for doc in docs_terms:
            print(f"\nDocument: {doc['filename']}")
            print("Terms and weights:")
            for term, weight in doc["terms"][:top_terms_preview]:
                print(f"  - {term}: {weight:.4f}")
    else:
        print("\n(no doc terms)")


def main():
    load_dotenv()

    N_TOPICS = 8
    N_TOP_TERMS = 12
    MAX_FEATURES = 20000
    MIN_DF = 2
    MAX_DF = 0.9
    NGRAM_RANGE = (1, 2)
    RANDOM_STATE = 42
    RUN_LSA = True
    RUN_LDA = True

    all_pdfs = load_pdfs(DOCS_PATH)
    selected = choose_files(all_pdfs)
    if not selected:
        return
    
    total_chars = sum(len(text) for text in selected.values())
    print(f" > Loaded {len(selected)} selected PDFs for topic modeling.")
    print(f" > Total content: {total_chars:,} characters")

    print("\n=== Running Topic Modeling (LSA & LDA) ===")

    # Adjust min_df based on number of documents to avoid sklearn errors
    n_docs = len(selected)
    adjusted_min_df = min(MIN_DF, max(1, n_docs // 2)) if n_docs > 1 else 1
    adjusted_max_df = MAX_DF if n_docs > 2 else 1.0
    
    print(f" > Documents: {n_docs}, using min_df={adjusted_min_df}, max_df={adjusted_max_df}")

    # LSA
    if RUN_LSA:
        print("\n=== Running LSA ===")
        lsa = LSAService(
            n_topics=N_TOPICS,
            n_top_terms_per_doc=N_TOP_TERMS,
            max_features=MAX_FEATURES,
            stopwords_lang="english",
            random_state=RANDOM_STATE,
            ngram_range=NGRAM_RANGE,
            min_df=adjusted_min_df,
            max_df=adjusted_max_df,
        )
        lsa_res = lsa.run(selected)
        print_model_results("LSA", lsa_res, top_terms_preview=min(10, N_TOP_TERMS))

    # LDA
    if RUN_LDA:
        print("\n=== Running LDA ===")
        lda = LDAService(
            n_topics=N_TOPICS,
            n_top_terms_per_doc=N_TOP_TERMS,
            max_features=MAX_FEATURES,
            stopwords_lang="english",
            random_state=RANDOM_STATE,
            ngram_range=NGRAM_RANGE,
            min_df=adjusted_min_df,
            max_df=adjusted_max_df,
        )
        lda_res = lda.run(selected)
        print_model_results("LDA", lda_res, top_terms_preview=min(10, N_TOP_TERMS))


if __name__ == "__main__":
    main()