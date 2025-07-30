import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from services.cso_service import CSOService

def main():
    load_dotenv()
    
    CSO_FILE_PATH = os.path.join("data", "cso.ttl")
    
    # Inisialisasi LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv('GEMINI_API_KEY'))
    
    # Inisialisasi CSOService
    cso_service = CSOService(
        neo4j_uri=os.getenv('NEO4J_URI'),
        neo4j_username=os.getenv('NEO4J_USERNAME'),
        neo4j_password=os.getenv('NEO4J_PASSWORD'),
        llm=llm
    )
    
    # Ekstrak dan impor topik CSO
    topics, hierarchy_data = cso_service.extract_topics_with_hierarchy(CSO_FILE_PATH, max_depth=4)
    cso_service.import_to_neo4j(topics, hierarchy_data)
    
    print("\nCSO graph complete!")

if __name__ == "__main__":
    main()