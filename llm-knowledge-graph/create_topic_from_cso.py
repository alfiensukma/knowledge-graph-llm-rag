import os
from dotenv import load_dotenv
import faiss
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
    use_llm = True

    svc = CSOService(
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        llm=None,
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        cluster_threshold=0.9,
        use_llm=use_llm,
    )

    if ask("Clear existing Topic nodes before import?", default="n"):
        svc.clear_existing_data()

    svc.ensure_constraints()

    print("Extracting CSO topics + hierarchy...")
    topics, hierarchy = svc.extract_topics_with_hierarchy(CSO_FILE_PATH, max_depth=4)

    print("Preparing (normalize + abbrev expand + dedup + cluster)...")
    processed = svc.prepare_topics(topics)

    print(f"Importing {len(processed)} canonical topics...")
    svc.import_to_neo4j(processed, hierarchy)

    print("Running APOC duplicate merge by label_norm...")
    svc.merge_duplicates_apoc()

    # Build embedding index
    index_path = os.getenv("CSO_INDEX_PATH", "data/cso_topics.faiss")
    labels_path = os.getenv("CSO_LABELS_PATH", "data/cso_labels.json")
    if faiss is not None:
        svc.build_and_save_cso_index(
            topics=processed,
            index_path=index_path,
            labels_path=labels_path,
            use_normalized=True,
            batch_size=512,
        )
        print("FAISS index built.")
    else:
        print("FAISS not installed; skip index build.")

    print("\nCSO graph build complete.")

if __name__ == "__main__":
    main()
