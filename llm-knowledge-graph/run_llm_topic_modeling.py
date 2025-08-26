import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.llm_topic_modeling_service import LLMTopicModelingService

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'ISSN:?\s*\d{4}-\d{4}', ' ', text, flags=re.I)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def list_pdf_files(pdf_dir: str):
    items = []
    if not os.path.isdir(pdf_dir):
        return items
    for name in sorted(os.listdir(pdf_dir)):
        if name.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, name)
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
            except OSError:
                size_mb = 0
            items.append({
                "filename": name,
                "path": path,
                "size_mb": size_mb
            })
    return items

def display_pdfs(pdfs):
    print("\n" + "=" * 90)
    print("PDF FILES (choose one to run LLM topic modeling)")
    print("=" * 90)
    if not pdfs:
        print(" > No PDF files found in data/pdfs.")
        return
    for idx, p in enumerate(pdfs, 1):
        print(f"{idx:2d}. {p['filename']:<75} {p['size_mb']:6.2f}MB")
    print("=" * 90)
    print(f"Total: {len(pdfs)}")

def select_pdf(pdfs):
    if not pdfs:
        return None
    while True:
        choice = input("\nEnter PDF number (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return None
        if not choice.isdigit():
            print(" > Invalid input.")
            continue
        num = int(choice)
        if 1 <= num <= len(pdfs):
            return pdfs[num - 1]
        print(" > Number out of range.")

def load_pdf_or_fallback(selected):
    path = selected["path"]
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        pages = []
        for d in docs:
            if d.page_content and isinstance(d.page_content, str):
                pages.append(clean_text(d.page_content))
        full_text = "\n".join(pages)
        if not full_text.strip():
            print(" > Empty PDF content after cleaning. Using filename only as context.")
            return selected["filename"], 0, False
        return full_text, len(pages), True
    except Exception as e:
        print(f" > Failed to read PDF: {e}. Using filename only.")
        return selected["filename"], 0, False
    
def run_model(service: LLMTopicModelingService, filename: str, full_text: str):
    context = service._make_context(full_text, filename)
    print("\n=== Running LLM Topic Modeling (LSA-like & LDA-like) ===")
    out = service._run_lsa_lda_like(context)

    print("\n--- LSA-like: Document Top Terms ---")
    for term, w in out.lsa.doc_terms[: service.n_top_terms_per_doc]:
        print(f"  {term:<30} {w:.4f}")

    print("\n--- LSA-like: Topics ---")
    for tv in out.lsa.topics:
        zipped = list(zip(tv.top_words, tv.weights))
        # Show up to first 10 terms
        line = ", ".join(f"{w}({wt:.2f})" for w, wt in zipped[:10])
        print(f"  Topic {tv.topic_id}: {line}")

    print("\n--- LDA-like: Document Topic Distribution ---")
    dist_str = ", ".join(f"{x:.4f}" for x in out.lda.doc_topic)
    print(f"  [{dist_str}] (sum≈{sum(out.lda.doc_topic):.4f})")

    print("\n--- LDA-like: Document Top Terms ---")
    for term, w in out.lda.doc_terms[: service.n_top_terms_per_doc]:
        print(f"  {term:<30} {w:.4f}")

    print("\n--- LDA-like: Topics ---")
    for tv in out.lda.topics:
        zipped = list(zip(tv.top_words, tv.weights))
        line = ", ".join(f"{w}({wt:.2f})" for w, wt in zipped[:10])
        print(f"  Topic {tv.topic_id}: {line}")

    print("\n=== Completed ===")

    return {
        "lsa": out.lsa.model_dump(),
        "lda": out.lda.model_dump()
    }
    
def main():
    load_dotenv()

    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    print("Initializing services ...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    # Fetch existing pdf
    pdf_dir = os.path.join("data", "pdfs")
    pdfs = list_pdf_files(pdf_dir)
    display_pdfs(pdfs)
    selected = select_pdf(pdfs)
    if not selected:
        print(" > No selection. Exiting.")
        return

    print(f"\nSelected: {selected['filename']}")
    full_text, pages, used_pdf = load_pdf_or_fallback(selected)
    if not full_text.strip():
        print(" > No usable text. Abort.")
        return
    print(f" > Source: {'PDF content' if used_pdf else 'Title only'} | pages={pages} | chars={len(full_text)}")

    modeling_service = LLMTopicModelingService(
        llm=llm,
        graph_service=graph_service,
        max_topics_in_prompt=40,
        n_topics=5,
        n_top_terms_per_doc=10,
        min_confidence=0.9,
        top_k_map_each=5,
        use_full_document=True,
        max_context_chars=80000
    )

    _ = run_model(modeling_service, selected["filename"], full_text)

    print("\nDone (results printed above).")

if __name__ == "__main__":
    main()