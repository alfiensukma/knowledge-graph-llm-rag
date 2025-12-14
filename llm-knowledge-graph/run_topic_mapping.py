import os
import re
from typing import Dict, List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from services.graph_service import GraphService
from services.topic_service import TopicExtractionService, MAX_TOPICS_LLM
from services.lsa_service import LSAService, MAX_TOPICS_LSA
from services.lda_service import LDAService, MAX_TOPICS_LDA
from services.topic_mapping_service import TopicMappingService

PDF_FOLDER = "data/pdfs"


def get_mapped_papers(graph_service) -> set:
    try:
        result = graph_service.graph.query("""
            MATCH (p:Paper)-[:HAS_TOPIC]->(:Topic)
            RETURN DISTINCT p.filename AS filename
        """)
        mapped_files = set()
        
        for record in result:
            if record.get("filename"):
                mapped_files.add(record["filename"])
        
        if mapped_files:
            print(f"Found {len(mapped_files)} papers with HAS_TOPIC relationships:")
            for i, filename in enumerate(sorted(mapped_files), 1):
                print(f"  {i}. {filename}")
        else:
            print("No papers with HAS_TOPIC relationships found.")
        
        return mapped_files
    except Exception as e:
        print(f"Error checking mapped papers: {e}")
        return set()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'ISSN:?\s*\d{4}-\d{4}', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def load_pdf_names(folder: str) -> List[str]:
    pdf_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(file)
    return sorted(pdf_files)


def load_pdf_text(folder: str, filename: str) -> tuple:
    file_path = os.path.join(folder, filename)
    if not os.path.isfile(file_path):
        print(f" > PDF file '{filename}' not found in {folder}.")
        return "", 0, False
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        pages = []
        for d in docs:
            if d.page_content and isinstance(d.page_content, str):
                pages.append(clean_text(d.page_content))
        full_text = "\n".join(pages)
        
        if not full_text.strip():
            print(" > Empty PDF content after cleaning.")
            return "", len(pages), False
        
        return full_text, len(pages), True
    except Exception as e:
        print(f" > Failed to read PDF: {e}")
        return "", 0, False


def display_pdf_list(pdf_names: List[str], mapped_files: set) -> List[tuple]:
    available_files = []
    
    print("\n" + "=" * 110)
    print("PDF FILES STATUS (select file that has NOT been mapped)")
    print("=" * 110)
    
    if not pdf_names:
        print("No PDF files found.")
        return []
    
    for idx, name in enumerate(pdf_names, 1):
        file_path = os.path.join(PDF_FOLDER, name)
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if name in mapped_files:
                status = " - (MAPPED)"
                color = "\033[92m"
            else:
                status = " - (NOT MAPPED)"
                color = "\033[93m"
                available_files.append((idx, name))
            
            reset_color = "\033[0m"
            print(f"{color}{idx:2d}. {name:<75} {size_mb:6.2f}MB {status}{reset_color}")
        except OSError:
            print(f"{idx:2d}. {name:<75} (size unknown)")
    
    print("=" * 110)
    print(f"Summary: {len(pdf_names)} total | {len(mapped_files)} mapped | {len(available_files)} available")
    
    return available_files


def select_pdf(pdf_names: List[str], available_files: List[tuple], mapped_files: set) -> str:
    if not available_files:
        print("\n > No unmapped PDF files available!")
        return None
    
    valid_choices = {idx: name for idx, name in available_files}
    valid_numbers = sorted(valid_choices.keys())
    
    print(f"\n > Valid choices (NOT MAPPED): {', '.join(map(str, valid_numbers))}")
    
    while True:
        choice = input("\nEnter PDF number (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return None
        
        if not choice.isdigit():
            print(" > Invalid input. Please enter a number.")
            continue
        
        num = int(choice)
        if num in valid_choices:
            return valid_choices[num]
        
        if 1 <= num <= len(pdf_names):
            filename = pdf_names[num - 1]
            if filename in mapped_files:
                print(f" > File '{filename}' has already been mapped (HAS_TOPIC relationships exist)!")
                print(f" > Please select from available choices: {', '.join(map(str, valid_numbers))}")
            else:
                print(f" > Invalid choice! Available choices: {', '.join(map(str, valid_numbers))}")
        else:
            print(f" > Out of range. Please enter a number from: {', '.join(map(str, valid_numbers))}")


def select_method() -> str:
    print("\n" + "=" * 60)
    print("SELECT TOPIC EXTRACTION METHOD")
    print("=" * 60)
    print("1. LLM (Gemini) - AI-based extraction")
    print("2. LSA (Latent Semantic Analysis) - Statistical method")
    print("3. LDA (Latent Dirichlet Allocation) - Probabilistic method")
    print("=" * 60)
    
    while True:
        choice = input("Enter method number (1-3, or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return None
        
        if choice in ['1', '2', '3']:
            methods = {'1': 'LLM', '2': 'LSA', '3': 'LDA'}
            return methods[choice]
        else:
            print(" > Invalid choice. Please enter 1, 2, or 3.")


def ask_mapping() -> bool:
    print("\n" + "=" * 60)
    print("MAP TOPICS TO CSO (Computer Science Ontology)?")
    print("=" * 60)
    print("This will create HAS_TOPIC relationships in Neo4j.")
    
    while True:
        choice = input("Map topics to CSO? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print(" > Invalid choice. Please enter 'y' or 'n'.")


def extract_topics_with_llm(llm, full_text: str) -> List[str]:
    print(f"\n=== Extracting Topics with LLM (max {MAX_TOPICS_LLM} topics) ===")
    topic_service = TopicExtractionService(llm)
    topics = topic_service.extract_topics_from_text(full_text, max_topics=MAX_TOPICS_LLM)
    return topics


def extract_topics_with_lsa(filename: str, full_text: str) -> List[str]:
    print(f"\n=== Extracting Topics with LSA (max {MAX_TOPICS_LSA} topics) ===")
    lsa_service = LSAService()
    
    # Adjust min_df/max_df for single document
    lsa_service.min_df = 1
    lsa_service.max_df = 1.0
    result = lsa_service.run({filename: full_text})
    
    if result['doc_terms']:
        doc_result = result['doc_terms'][0]
        topics = doc_result.get('top_terms', [])
        print(f" > LSA extracted {len(topics)} topics")
        return topics
    return []


def extract_topics_with_lda(filename: str, full_text: str) -> List[str]:
    print(f"\n=== Extracting Topics with LDA (max {MAX_TOPICS_LDA} topics) ===")
    lda_service = LDAService()
    
    # Adjust min_df/max_df for single document
    lda_service.min_df = 1
    lda_service.max_df = 1.0
    result = lda_service.run({filename: full_text})
    
    if result['doc_terms']:
        doc_result = result['doc_terms'][0]
        topics = doc_result.get('top_terms', [])
        print(f" > LDA extracted {len(topics)} topics")
        return topics
    return []


def main():
    load_dotenv()

    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL = os.getenv("MODEL", "gemini-2.5-flash")

    print("=" * 110)
    print("TOPIC MODELING & MAPPING")
    print("This tool extracts topics from PDFs using LLM, LSA, or LDA,")
    print("and optionally maps them to CSO (Computer Science Ontology) in Neo4j.")
    print("=" * 110)
    print("\n > Initializing Neo4j connection...")
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    
    print("\n > Checking existing paper mappings in Neo4j...")
    mapped_files = get_mapped_papers(graph_service)

    pdf_names = load_pdf_names(PDF_FOLDER)
    if not pdf_names:
        print(f"\n > No PDF files found in {PDF_FOLDER}")
        return

    available_files = display_pdf_list(pdf_names, mapped_files)
    selected_pdf = select_pdf(pdf_names, available_files, mapped_files)
    if not selected_pdf:
        print("\n > No PDF selected. Exiting.")
        return
    
    print(f"\n > Loading PDF: {selected_pdf}")
    full_text, page_count, success = load_pdf_text(PDF_FOLDER, selected_pdf)
    if not success or not full_text.strip():
        print(" > Failed to load PDF content. Exiting.")
        return    
    print(f" > Loaded {page_count} pages, {len(full_text):,} characters")

    method = select_method()
    if not method:
        print("\n > No method selected. Exiting.")
        return

    extracted_topics = []
    if method == 'LLM':
        llm = ChatGoogleGenerativeAI(
            model=MODEL, 
            google_api_key=GEMINI_API_KEY, 
            temperature=0
        )
        extracted_topics = extract_topics_with_llm(llm, full_text)
    elif method == 'LSA':
        extracted_topics = extract_topics_with_lsa(selected_pdf, full_text)
    elif method == 'LDA':
        extracted_topics = extract_topics_with_lda(selected_pdf, full_text)

    print("\n" + "=" * 60)
    print(f"EXTRACTED TOPICS ({method})")
    print("=" * 60)
    if extracted_topics:
        for idx, topic in enumerate(extracted_topics, 1):
            print(f"{idx:2d}. {topic}")
        print("=" * 60)
        print(f"Total: {len(extracted_topics)} topics extracted")
    else:
        print("No topics extracted.")
        return

    do_mapping = ask_mapping()
    if do_mapping:
        print("\n=== Mapping Topics to CSO ===")
        llm = ChatGoogleGenerativeAI(
            model=MODEL, 
            google_api_key=GEMINI_API_KEY, 
            temperature=0
        )
        mapping_service = TopicMappingService(llm, graph_service)
        validated_topics = mapping_service.map_topics_to_cso(extracted_topics)
        
        print("\n" + "=" * 60)
        print("VALIDATED CSO TOPICS")
        print("=" * 60)
        if validated_topics:
            for idx, topic in enumerate(validated_topics, 1):
                print(f"{idx:2d}. {topic}")
            print("=" * 60)
            print(f"Total: {len(validated_topics)} topics mapped to CSO")
            result = mapping_service.create_paper_topic_relationships(selected_pdf, validated_topics)
            print("\n" + "=" * 60)
            print("MAPPING SUMMARY")
            print("=" * 60)
            print(f"Linked topics: {len(result['linked'])}")
            print(f"Missing topics: {len(result['missing'])}")
            if result['linked']:
                print(f"Successfully linked: {', '.join(result['linked'][:5])}...")
        else:
            print("No topics could be mapped to CSO.")
    else:
        print("\n > Skipping topic mapping.")

    print("\n" + "=" * 110)
    print("PROCESS COMPLETED")
    print("=" * 110)
    print(f"PDF: {selected_pdf}")
    print(f"Method: {method}")
    print(f"Topics extracted: {len(extracted_topics)}")
    if do_mapping and 'validated_topics' in locals():
        print(f"Topics mapped to CSO: {len(validated_topics)}")
    print("=" * 110)


if __name__ == "__main__":
    main()
