import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.llm_graph_service import LLMGraphExtractionService
from services.topic_service import TopicExtractionService
from services.llm_topic_modeling_service import LLMTopicModelingService

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

    USE_TOPIC_SERVICE = False
    USE_LLM_TOPIC_MODELING = True

    if USE_TOPIC_SERVICE and USE_LLM_TOPIC_MODELING:
        raise ValueError("Hanya boleh salah satu True: USE_TOPIC_SERVICE atau USE_LLM_TOPIC_MODELING.")
    if not USE_TOPIC_SERVICE and not USE_LLM_TOPIC_MODELING:
        raise ValueError("Set salah satu ke True: USE_TOPIC_SERVICE atau USE_LLM_TOPIC_MODELING.")

    # Konfigurasi
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)

    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    llm_graph_extractor = LLMGraphExtractionService(llm=llm, graph_service=graph_service)

    if USE_TOPIC_SERVICE:
        print(">> Mode: TopicExtractionService")
        topic_handler = TopicExtractionService(llm=llm, graph_service=graph_service)
    else:
        print(">> Mode: LLMTopicModelingService (LSA/LDA)")
        topic_handler = LLMTopicModelingService(
            llm=llm,
            graph_service=graph_service,
            n_topics=5,
            n_top_terms_per_doc=10,
            min_confidence=0.9,
            top_k_map_each=10,
            max_topics_in_prompt=100,
            use_full_document=True,
            max_context_chars=80000,
        )

    # Load document
    DOCS_PATH = os.path.join("data", "pdfs")
    try:
        loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        docs = loader.load()
        print(f"Found {len(docs)} documents to process.")
    except Exception as e:
        print(f"Failed to load documents: {e}")
        return

    # merge PDF
    pdf_docs = {}
    for doc in docs:
        filename = os.path.basename(doc.metadata["source"])
        pdf_path = doc.metadata["source"]
        if filename not in pdf_docs:
            pdf_docs[filename] = {"path": pdf_path, "pages": []}
        if doc.page_content and isinstance(doc.page_content, str):
            pdf_docs[filename]["pages"].append(clean_text(doc.page_content))

    for filename, data in pdf_docs.items():
        print(f"\n--- Processing Document: {filename} ---")
        full_text = "\n".join(data["pages"])
        if not full_text.strip():
            print(f"  > No valid text after cleaning for {filename}")
            continue

        result = llm_graph_extractor.process_document(data["path"], filename, full_text)

        if result:
            print(f"  > Extracted paper data: {result['graph_data']}")
            paper_id = result["paper_id"]

            # ADD TOPIC
            try:
                if USE_TOPIC_SERVICE:
                    validated_topics = topic_handler.get_validated_topics_for_text(full_text)
                    print(f"  > Validated topics: {validated_topics}")
                    
                    if validated_topics:
                        graph_service.link_paper_to_topics(paper_id, validated_topics)
                        print(f"  > Linked paper to {len(validated_topics)} topics: {validated_topics}")
                    else:
                        print("  > No validated CSO topics found for this paper.")
                else:
                    print(f"  > Processing topics with LLM Topic Modeling...")
                    topic_result = topic_handler.process_document(
                        filename=filename, 
                        full_text=full_text, 
                        link_to_graph=False
                    )
                    
                    validated_topics = topic_result.get("mapped_topics", [])
                    print(f"  > Mapped topics from LLM: {validated_topics}")
                    
                    if validated_topics:
                        # Manual linking
                        graph_service.link_paper_to_topics(paper_id, validated_topics)
                        print(f"  > Linked paper to {len(validated_topics)} topics: {validated_topics}")
                    else:
                        print("  > No CSO topics mapped from LLM topic modeling.")
                        
            except Exception as e:
                print(f"  > Failed to process topics for {filename}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  > Could not process document {filename}, skipping.")

    print("\nAll documents processed successfully!")

if __name__ == "__main__":
    main()
