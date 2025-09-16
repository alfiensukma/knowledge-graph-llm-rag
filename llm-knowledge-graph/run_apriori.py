import os
from dotenv import load_dotenv
from services.graph_service import GraphService
from services.apriori_service import AprioriService

def main():
    load_dotenv()

    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    MIN_SUPPORT_COUNT = 2

    print("Initializing services...")
    graph_service = GraphService(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    apriori_service = AprioriService(graph_service)

    try:
        apriori_service.create_graph_projection()

        apriori_service.run_full_apriori_pipeline(
            min_support_count=MIN_SUPPORT_COUNT
        )

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
