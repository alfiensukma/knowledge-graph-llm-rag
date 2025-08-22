import os, re, time
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from services.graph_service import GraphService
from services.lsa_service import LSAService
from services.lda_service import LDAService
from services.topic_mapper_service import TopicMapperService
from langchain_google_genai import ChatGoogleGenerativeAI

def _clean_page(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def load_pdfs(folder: str):
    loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    docs = loader.load()
    pdf_docs = {}
    for d in docs:
        fn = os.path.basename(d.metadata["source"])
        pdf_docs.setdefault(fn, [])
        if d.page_content:
            pdf_docs[fn].append(_clean_page(d.page_content))
    return {fn: "\n".join(pages) for fn, pages in pdf_docs.items()}

def print_model_results(model_name: str, results: Dict[str, Any]):
    print(f"\n=== {model_name} Results ===")
    
    print("\nGenerated Topics:")
    for topic in results.get("topics", []):
        print(f"\nTopic {topic['topic_id']}:")
        for word, weight in zip(topic["top_words"], topic["weights"]):
            print(f"  - {word}: {weight:.4f}")
    
    print("\nDocument Terms:")
    for doc in results.get("doc_terms", []):
        print(f"\nDocument: {doc['filename']}")
        print("Terms and weights:")
        for term, weight in doc["terms"]:
            print(f"  - {term}: {weight:.4f}")

class TokenManager:
    def __init__(self, tokens_per_minute: int = 1_000_000):
        self.tokens_per_minute = tokens_per_minute
        self.safety_margin = 0.8  # Use 80% of limit
        self.effective_limit = int(tokens_per_minute * self.safety_margin)
        self.current_minute_tokens = 0
        self.minute_start_time = time.time()
        
    def can_proceed(self, estimated_tokens: int) -> bool:
        current_time = time.time()
        
        # Reset if new minute
        if current_time - self.minute_start_time >= 60:
            self.current_minute_tokens = 0
            self.minute_start_time = current_time
            
        return self.current_minute_tokens + estimated_tokens <= self.effective_limit
    
    def add_tokens(self, tokens: int):
        self.current_minute_tokens += tokens
        
    def wait_if_needed(self, estimated_tokens: int):
        if not self.can_proceed(estimated_tokens):
            wait_time = 60 - (time.time() - self.minute_start_time)
            if wait_time > 0:
                print(f"Rate limit approaching. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.current_minute_tokens = 0
                self.minute_start_time = time.time()

def main():
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    RUN_LSA = True
    RUN_LDA = False

    if RUN_LSA and RUN_LDA:
        raise ValueError("Hanya boleh salah satu yang True: LSA atau LDA.")
    if not RUN_LSA and not RUN_LDA:
        raise ValueError("Pilih salah satu: set RUN_LSA=True atau RUN_LDA=True")

    # Initialize services
    gs = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    
    # Check if papers exist in database
    papers_count = gs.graph.query("MATCH (p:Paper) RETURN count(p) AS count")[0]["count"]
    if papers_count == 0:
        print("No papers found in database. Please run create_paper_nodes.py first.")
        return
    
    print(f"Found {papers_count} papers in database. Proceeding with topic modeling...")

    # Load PDFs for topic modeling
    pdfs = load_pdfs(os.path.join("data", "pdfs"))
    print(f"Loaded {len(pdfs)} PDFs for topic modeling")

    terms_by_doc = {}

    if RUN_LSA:
        print("\n=== Running LSA ===")
        lsa = LSAService(
            n_topics=5,
            n_top_terms_per_doc=10,
            max_features=20000,
            stopwords_lang="english",
            random_state=42,
            ngram_range=(1, 2),
        )
        lsa_res = lsa.run(pdfs)
        terms_by_doc = {d["filename"]: d["terms"] for d in lsa_res.get("doc_terms", [])}
        print_model_results("LSA", lsa_res)

    if RUN_LDA:
        print("\n=== Running LDA ===")
        lda = LDAService(
            n_topics=5,
            n_top_terms_per_doc=10,
            max_features=20000,
            stopwords_lang="english",
            random_state=42,
        )
        lda_res = lda.run(pdfs)
        terms_by_doc = {d["filename"]: d["terms"] for d in lda_res.get("doc_terms", [])}
        print_model_results("LDA", lda_res)
        
        
# --------------------------------------------------- #
    # Mapping to topic (OPTIONAL)
    print(f"\n=== Starting Topic Mapping ===")
    
    # Initialize token manager
    token_manager = TokenManager(tokens_per_minute=1_000_000)
    
    mapper = TopicMapperService(graph_service=gs, llm=llm)
    mapper.token_manager = token_manager
    
    linked = mapper.map_and_link(
        lsa_terms_by_doc=terms_by_doc if RUN_LSA else {},
        lda_terms_by_doc=terms_by_doc if RUN_LDA else {},
        top_k_each=10,
    )

    # Results
    print("\n=== Matched Topics with CSO ===")
    total_matches = 0
    for fn, topics in linked.items():
        print(f"\nDocument: {fn}")
        if topics:
            print("Matched Topics:")
            for topic in topics:
                print(f"  - {topic}")
            total_matches += len(topics)
        else:
            print("  No topics matched")
    
    print(f"\n=== Summary ===")
    print(f"Total documents processed: {len(linked)}")
    print(f"Total topic matches created: {total_matches}")
    print(f"Average topics per document: {total_matches/len(linked) if linked else 0:.1f}")

if __name__ == "__main__":
    main()