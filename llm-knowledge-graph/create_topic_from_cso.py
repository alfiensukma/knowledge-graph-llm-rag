import os
from dotenv import load_dotenv
from services.cso_service import CSOService

def ask(prompt: str, default: str = "n") -> bool:
    default = default.lower()
    suffix = "[Y/n]" if default == "y" else "[y/N]"
    while True:
        ans = input(f"{prompt} {suffix}: ").strip().lower()
        if not ans:
            ans = default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")

def main():
    load_dotenv()

    CSO_FILE_PATH = os.path.join("data", "cso.ttl")
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

    cso_service = CSOService(
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        llm=None,
        embed_model=os.getenv("CSO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )

    if ask("Do you want to CLEAR existing TOPIC nodes before import? ", default="n"):
        cso_service.clear_existing_data()
        
    cso_service.ensure_constraints()

    print("Extracting CSO topics + hierarchy...")
    topics, hierarchy = cso_service.extract_topics_with_hierarchy(CSO_FILE_PATH, max_depth=4)

    print(f"Importing {len(topics)} topics into Neo4j...")
    cso_service.import_to_neo4j(topics, hierarchy)

    cso_service.merge_duplicates()

    index_path = os.getenv("CSO_INDEX_PATH", "data/cso_topics.faiss")
    labels_path = os.getenv("CSO_LABELS_PATH", "data/cso_labels.json")
    cso_service.build_and_save_cso_index(
        topics=topics,
        index_path=index_path,
        labels_path=labels_path,
        use_normalized=True,
        batch_size=512,
    )

    print("\nCSO graph + embedding index complete!")

if __name__ == "__main__":
    main()
