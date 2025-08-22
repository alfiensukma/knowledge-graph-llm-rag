import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.recommendation_service import RecommendationService

def main():
    load_dotenv()
    
    # Konfigurasi
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Inisialisasi LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    
    # Inisialisasi Services
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    recommendation_service = RecommendationService(llm=llm, graph_service=graph_service)
    
    # Dapatkan rekomendasi untuk paper dengan ID tertentu
    user_paper_ids = [
        "23faa0a0-0750-4669-ae96-8040ea82e043",
        "e0755a18-c7bf-4bc7-b315-693bdfa0362d"
    ]
    print(f"\n--- Generating Recommendations for Paper IDs: {user_paper_ids} ---")
    recommendations = recommendation_service.get_llm_recommendations(user_paper_ids)
    
    print("--- LLM Recommendations ---")
    if recommendations:
        for rec in recommendations:
            print(f"  - {rec['filename']}: {rec['title']} (Topics: {rec['topics']})")
    else:
        print("  > No recommendations generated.")
    
    print("\nRecommendations generated successfully!")

if __name__ == "__main__":
    main()