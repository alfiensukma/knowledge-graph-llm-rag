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

def get_existing_papers(graph_service) -> set:
    """Get list of papers that already exist in Neo4j"""
    try:
        result = graph_service.graph.query("""
            MATCH (p:Paper)
            RETURN p.filename AS filename, p.title AS title
        """)
        existing_files = set()
        existing_titles = set()
        
        for record in result:
            if record.get("filename"):
                existing_files.add(record["filename"])
            if record.get("title"):
                existing_titles.add(record["title"].lower().strip())
        
        print(f"Found {len(existing_files)} existing papers in Neo4j:")
        for i, filename in enumerate(sorted(existing_files), 1):
            print(f"  {i}. {filename}")
        
        return existing_files, existing_titles
    except Exception as e:
        print(f"Error checking existing papers: {e}")
        return set(), set()

def list_pdf_files(docs_path: str) -> list:
    """List all PDF files in the directory"""
    pdf_files = []
    try:
        for filename in os.listdir(docs_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(docs_path, filename)
                file_size = os.path.getsize(file_path)
                pdf_files.append({
                    'filename': filename,
                    'path': file_path,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
        return sorted(pdf_files, key=lambda x: x['filename'])
    except Exception as e:
        print(f"Error listing PDF files: {e}")
        return []

def display_pdf_status(pdf_files: list, existing_files: set) -> list:
    """Display PDF files with their processing status"""
    available_files = []
    
    print("\n" + "="*80)
    print("PDF FILES STATUS (select file that has not been processed)")
    print("="*80)
    
    if not pdf_files:
        print(" > No PDF files found in data/pdfs/ directory")
        return []
    
    for i, pdf_info in enumerate(pdf_files, 1):
        filename = pdf_info['filename']
        size_mb = pdf_info['size_mb']
        
        if filename in existing_files:
            status = " - (PROCESSED)"
            color = "\033[92m"
        else:
            status = " - (NOT PROCESSED)"
            color = "\033[93m"
            available_files.append((i, pdf_info))
        
        reset_color = "\033[0m"
        print(f"{color}{i:2d}. {filename:<60} {size_mb:>6.2f}MB {status}{reset_color}")
    
    print("="*80)
    print(f"Summary: {len(pdf_files)} total, {len(existing_files)} processed, {len(available_files)} available")
    
    return available_files

def select_pdf_file(available_files: list, all_pdf_files: list, existing_files: set) -> dict:
    """Let user select a PDF file to process"""
    if not available_files:
        print("\n > No unprocessed PDF files available!")
        return None
    
    valid_choices = {original_index: pdf_info for original_index, pdf_info in available_files}
    valid_numbers = sorted(valid_choices.keys())
    
    print(f"\n > Available choices: {', '.join(map(str, valid_numbers))}")

    while True:
        try:
            choice = input(f"\nEnter your choice or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print(" > Exiting...")
                return None
            
            choice_num = int(choice)
            
            # Check if choice is valid (unprocessed file)
            if choice_num in valid_choices:
                selected_pdf = valid_choices[choice_num]
                print(f"\n Selected: {selected_pdf['filename']}")
                return selected_pdf
            
            # Check if choice is a processed file
            elif 1 <= choice_num <= len(all_pdf_files):
                filename = all_pdf_files[choice_num - 1]['filename']
                if filename in existing_files:
                    print(f" > File '{filename}' has already been processed!")
                    print(f" > Please select from available choices: {', '.join(map(str, valid_numbers))}")
                else:
                    print(f" > Invalid choice! Available choices: {', '.join(map(str, valid_numbers))}")
            else:
                print(f" > Invalid choice! Please enter a number from: {', '.join(map(str, valid_numbers))}")
                
        except ValueError:
            print(" > Invalid input! Please enter a number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n Interrupted by user. Exiting...")
            return None

def load_selected_pdf(pdf_info: dict) -> tuple:
    """Load the selected PDF file"""
    try:
        print(f"\n Loading PDF: {pdf_info['filename']}...")
        
        loader = PyPDFLoader(pdf_info['path'])
        docs = loader.load()
        
        if not docs:
            print(f" No content found in {pdf_info['filename']}")
            return None, None
        
        # Merge all pages
        pages = []
        for doc in docs:
            if doc.page_content and isinstance(doc.page_content, str):
                pages.append(clean_text(doc.page_content))
        
        full_text = "\n".join(pages)
        if not full_text.strip():
            print(f" No valid text after cleaning for {pdf_info['filename']}")
            return None, None
        
        print(f" Loaded {len(pages)} pages, {len(full_text)} characters")
        return pdf_info['filename'], full_text
        
    except Exception as e:
        print(f" Error loading PDF {pdf_info['filename']}: {e}")
        return None, None

def main():
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    print(" > Initializing services...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    llm_graph_extractor = LLMGraphExtractionService(llm=llm, graph_service=graph_service)

    print("\n Checking existing papers in Neo4j...")
    existing_files, existing_titles = get_existing_papers(graph_service)
    
    DOCS_PATH = os.path.join("data", "pdfs")
    if not os.path.exists(DOCS_PATH):
        print(f" > Directory {DOCS_PATH} does not exist!")
        return

    pdf_files = list_pdf_files(DOCS_PATH)
    available_files = display_pdf_status(pdf_files, existing_files)

    selected_pdf = select_pdf_file(available_files, pdf_files, existing_files)
    if not selected_pdf:
        return

    filename, full_text = load_selected_pdf(selected_pdf)
    if not filename or not full_text:
        return

    # Process the document
    print(f"\n Processing Document: {filename}")
    print("-" * 60)
    
    try:
        result = llm_graph_extractor.process_document(selected_pdf['path'], filename, full_text)

        if result:
            print(f"Extracted paper data:")
            print(f" > Title: {result['graph_data'].get('title', 'N/A')}")
            print(f" > Authors: {len(result['graph_data'].get('authors', []))} found")
            print(f" > References: {len(result['graph_data'].get('references', []))} found")
        else:
            print(f" > Could not process document {filename}")

    except Exception as e:
        print(f" > Error processing document: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n > Processing completed for: {filename}")

if __name__ == "__main__":
    main()
