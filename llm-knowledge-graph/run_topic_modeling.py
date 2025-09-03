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


def load_pdf_names(folder: str) -> List[str]:
    pdf_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(file)
    return sorted(pdf_files)


def load_selected_pdfs(folder: str, selected_names: List[str]) -> Dict[str, str]:
    print(f"\nLoading content for {len(selected_names)} selected PDF(s)...")
    selected_docs = {}
    for name in selected_names:
        file_path = os.path.join(folder, name)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            content = "\n".join([_clean_page(p.page_content) for p in pages if p.page_content])
            selected_docs[name] = content
        except Exception as e:
            print(f"  - Warning: Could not load {name}. Error: {e}")
    return selected_docs


def choose_files(pdf_names: List[str]) -> List[str]:
    if not pdf_names:
        print(" > No PDFs found in:", DOCS_PATH)
        return []

    pdf_info = []
    for name in pdf_names:
        file_path = os.path.join(DOCS_PATH, name)
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            size_mb = 0
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
        return []

    selected_names = [pdf_info[j]["name"] for j in sorted(idxs)]
    print(f"\nSelected: {', '.join(selected_names)}")
    return selected_names


def print_model_results(model_name: str, results: Dict[str, Any], top_terms_preview: int = 10):
    print("\n" + "="*25 + f" {model_name} Results " + "="*25)

    topics = results.get("topics", [])
    docs_terms = results.get("doc_terms", [])

    if not topics and not docs_terms:
        print("(No results generated)")
        return
    
    topic_names = {}
    for topic in topics:
        top_words = topic.get("top_words", [])
        name = " ".join(top_words[:3]).title()
        topic_names[topic['topic_id']] = name

    for doc in docs_terms:
        print(f"\n--- Document: {doc['filename']} ---")

        if model_name == "LDA" and "distribution" in doc:
            print("\n  Document Topic Distribution:")
            dist_str = []
            for i, p in enumerate(doc["distribution"]):
                topic_name = f"\"{topic_names.get(i, f'Topic {i}')}\""
                dist_str.append(f"{p*100:.1f}% {topic_name}")
            print(f"    └─ Composed of: {', '.join(dist_str)}")

        print(f"\n  Document Top Terms (Top {top_terms_preview}):")
        for term, weight in doc["terms"][:top_terms_preview]:
            print(f"    {term:<30} {weight:.4f}")

    print("\n" + "-"*20 + " Global Latent Topics " + "-"*20)
    if topics:
        for topic in topics:
            topic_id = topic['topic_id']
            suggested_name = topic_names.get(topic_id, "")
            print(f"\n  Topic {topic_id}: \"{suggested_name}\"")
            words = " ".join([f"{word}({w:.2f})" for word, w in zip(topic.get("top_words", []), topic.get("weights", []))][:top_terms_preview])
            print(f"    └─ Keywords: {words}")
    else:
        print("\n(No global topics generated)")
    print("\n" + "="*65)


def main():
    load_dotenv()

    N_TOPICS = 5
    N_TOP_TERMS = 10
    MAX_FEATURES = 20000
    MIN_DF = 2
    MAX_DF = 0.9
    NGRAM_RANGE = (1, 2)
    RANDOM_STATE = 42
    RUN_LSA = True
    RUN_LDA = True
    CUSTOM_STOP_WORDS = ['et', 'al', 'et al', 'fig', 'figure', 'table', 'doi', 'https', 'www', 'org', '000']

    all_pdf_names = load_pdf_names(DOCS_PATH)
    selected_names = choose_files(all_pdf_names)
    if not selected_names:
        return

    selected_pdfs = load_selected_pdfs(DOCS_PATH, selected_names)
    if not selected_pdfs:
        print("No PDF content could be loaded. Exiting.")
        return

    total_chars = sum(len(text) for text in selected_pdfs.values())
    print(f" > Loaded {len(selected_pdfs)} selected PDFs for topic modeling.")
    print(f" > Total content: {total_chars:,} characters")

    print("\n=== Running Topic Modeling (LSA & LDA) ===")


    # Adjust min_df based on number of documents to avoid sklearn errors
    n_docs = len(selected_pdfs)
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
            custom_stopwords=CUSTOM_STOP_WORDS,
        )
        lsa_res = lsa.run(selected_pdfs)
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
            custom_stopwords=CUSTOM_STOP_WORDS,
        )
        lda_res = lda.run(selected_pdfs)
        print_model_results("LDA", lda_res, top_terms_preview=min(10, N_TOP_TERMS))


if __name__ == "__main__":
    main()