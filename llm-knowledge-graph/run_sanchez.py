import os
import json
import argparse
import datetime
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from services.ontology_index_service import OntologyIndexService
from services.itemset_similarity_service import ItemsetSimilarityService, SetSimConfig
from services.neo4j_sanchez_service import Neo4jIOService
from services.reporting_sanchez_service import ReportingService

def analyze_and_save(
    uri: str,
    username: str,
    password: str,
    name: str,
    out_dir: str,
    min_support: int = 2,
):
    """Run Sanchez cohesion on a Neo4j connection and save results."""
    graph = Neo4jGraph(url=uri, username=username, password=password)
    ont = OntologyIndexService(graph)
    ic_index = ont.build_ic_index()
    set_sim = ItemsetSimilarityService(ic_index)
    io = Neo4jIOService(graph, ic_index)
    reporting = ReportingService(ic_index, set_sim)

    F = io.fetch_frequent_itemsets(min_support=min_support)
    
    # Default Sanchez config
    cfg = SetSimConfig(method="bma", weight_by_ic=True, min_lcs_depth=3)
    
    # Get similarity metrics
    metrics = reporting.cohesion(F, cfg)
    metrics["name"] = name
    metrics["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    
    # Save results
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, name + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {json_path}")
    
    # Print summary
    print(f"\n=== Sanchez Similarity Results for {name} ===")
    print(f"Number of itemsets: {metrics['n_itemsets']}")
    print(f"Number of pairs compared: {metrics['n_pairs']}")
    print(f"Mean similarity: {metrics['mean_similarity']:.3f}")
    print(f"% High similarity (>= 0.8): {metrics['pct_high_sim']*100:.1f}%")
    print(f"% Low similarity (< 0.2): {metrics['pct_low_sim']*100:.1f}%")
    
    return metrics

def _parse_args():
    p = argparse.ArgumentParser(description="Run Sanchez cohesion and save results.")
    p.add_argument("filename", help="Output name without extension (e.g. llm_run1)")
    return p.parse_args()


if __name__ == "__main__":
    load_dotenv()

    args = _parse_args()

    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    out_dir = os.path.join("data", "sanchez-result")
    
    res = analyze_and_save(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        name=args.filename,
        out_dir=out_dir
    )