import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.llm_graph_service import LLMGraphExtractionService
from services.topic_service import TopicExtractionService

def clean_text(text: str) -> str:
    """Clean text by removing irrelevant metadata, ISSN, and excessive whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'ISSN:?\s*\d{4}-\d{4}', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def main():
    load_dotenv()
    
    # Konfigurasi
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    
    # Inisialisasi Services
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    llm_graph_extractor = LLMGraphExtractionService(llm=llm, graph_service=graph_service)
    topic_service = TopicExtractionService(llm=llm, graph_service=graph_service)
    
    # Load semua dokumen dari folder menggunakan PyPDFLoader
    DOCS_PATH = os.path.join("data", "pdfs")
    try:
        loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        docs = loader.load()
        print(f"Found {len(docs)} documents to process.")
    except Exception as e:
        print(f"Failed to load documents: {e}")
        return
    
    # Kelompokkan dokumen berdasarkan file PDF
    pdf_docs = {}
    for doc in docs:
        filename = os.path.basename(doc.metadata["source"])
        pdf_path = doc.metadata["source"]
        if filename not in pdf_docs:
            pdf_docs[filename] = {"path": pdf_path, "pages": []}
        if doc.page_content and isinstance(doc.page_content, str):
            pdf_docs[filename]["pages"].append(clean_text(doc.page_content))
    
    # Proses satu dokumen utuh pada satu waktu
    for filename, data in pdf_docs.items():
        print(f"\n--- Processing Document: {filename} ---")
        full_text = "\n".join(data["pages"])
        if not full_text.strip():
            print(f"  > No valid text after cleaning for {filename}")
            continue
        
        # Ekstrak struktur graph
        result = llm_graph_extractor.process_document(data["path"], filename, full_text)
        
        if result:
            print(f"  > Extracted paper data: {result['graph_data']}")
            paper_id = result["paper_id"]
            
            # Ekstrak dan validasi topik
            try:
                validated_topics = topic_service.get_validated_topics_for_text(full_text)
                print(f"  > Validated topics: {validated_topics}")
                if validated_topics:
                    graph_service.link_paper_to_topics(paper_id, validated_topics)
                    print(f"  > Linked paper to {len(validated_topics)} topics: {validated_topics}")
                else:
                    print("  > No validated CSO topics found for this paper.")
            except Exception as e:
                print(f"  > Failed to process topics for {filename}: {e}")
        else:
            print(f"  > Could not process document {filename}, skipping.")
    
    print("\nAll documents processed successfully!")

if __name__ == "__main__":
    main()