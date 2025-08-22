import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from services.graph_service import GraphService
from services.llm_apriori_service import LLMAprioriService

def main():
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    svc = LLMAprioriService(llm=llm, graph_service=graph_service)
    svc.build_llm_apriori_graph(
        min_support_count=2,
        min_confidence=0.7,
        max_itemset_size=5
    )

if __name__ == "__main__":
    main()
