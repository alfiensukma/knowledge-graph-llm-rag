import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.llm_combination_service import LLMCombinationService

def main():
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    svc = LLMCombinationService(llm=llm, graph_service=graph_service)

    # proses satu per satu (kamu masukkan id paper-nya)
    svc.generate_combinations_for_paper("ef372d69-04d9-4ae7-bddb-4df67d9cd9a4", max_k=5)

    # atau batch (tetap satu-per-satu, urut ASC)
    # svc.generate_combinations_for_papers(
    #     ["21c791b5-d21c-4581-a2e7-66151ef7c3b8", "8c6f5081-8050-4470-ac18-2eb9637bef02"],
    #     max_k=5
    # )

if __name__ == "__main__":
    main()