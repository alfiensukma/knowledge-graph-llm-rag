from __future__ import annotations
import os, json, re
from typing import List, Dict, Tuple, Optional
import rdflib
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

from sentence_transformers import SentenceTransformer
from langchain_neo4j import Neo4jGraph


def _normalize_label(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s*\([^)]+\)\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _canonical_form(label: str) -> str:
    s = _normalize_label(label)
    parts = s.split()
    if not parts:
        return s
    last = parts[-1]
    if last.endswith("ies") and len(last) > 3:
        last = last[:-3] + "y"
    elif last.endswith(("sses", "shes", "ches")) and len(last) > 4:
        last = last[:-2]
    elif last.endswith("es") and len(last) > 3:
        last = last[:-2]
    elif last.endswith("s") and len(last) > 3:
        last = last[:-1]
    parts[-1] = last
    return " ".join(parts)

class CSOService:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        llm=None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        self.llm = llm
        self.embed_model_name = embed_model
        self.embedder: Optional[SentenceTransformer] = None
        self._ensure_embedder()

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.embed_model_name)
            
    def ensure_constraints(self):
        self.graph.query("""
        CREATE CONSTRAINT topic_labelnorm_unique IF NOT EXISTS
        FOR (t:Topic) REQUIRE t.label_norm IS UNIQUE
        """)
        self.graph.query("""
        CREATE INDEX topic_label_idx IF NOT EXISTS
        FOR (t:Topic) ON (t.label)
        """)
        print("Neo4j constraints & indexes ensured.")

    def clear_existing_data(self):
        print("Clearing existing Topic nodes and relationships...")
        try:
            self.graph.query("DROP CONSTRAINT topic_labelnorm_unique IF EXISTS")
        except:
            pass
        self.graph.query("MATCH (t:Topic) DETACH DELETE t")
        print("Database cleared.")

    def extract_topics_with_hierarchy(self, cso_file_path: str, max_depth: int = 4) -> Tuple[List[Dict], List[Dict]]:
        print(f"Loading CSO ontology from {cso_file_path}...")
        g = rdflib.Graph()
        g.parse(cso_file_path, format="turtle")
        print("CSO ontology loaded.")

        topic_query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX cso: <http://cso.kmi.open.ac.uk/schema/cso#>
        SELECT ?uri ?label
        WHERE {
            ?uri a cso:Topic ;
                 rdfs:label ?label .
            FILTER (?label != "computer science")
        }
        """
        results = g.query(topic_query)
        topic_data = [{"uri": str(r.uri), "label": str(r.label)} for r in results]
        print(f"Found {len(topic_data)} topics. Loading hierarchy...")

        hierarchy_query = """
        PREFIX cso: <http://cso.kmi.open.ac.uk/schema/cso#>
        SELECT ?sub_topic ?super_topic
        WHERE {
            ?super_topic cso:superTopicOf ?sub_topic .
        }
        """
        hierarchy_results = g.query(hierarchy_query)
        hierarchy_data = [{"sub": str(r.sub_topic), "super": str(r.super_topic)} for r in hierarchy_results]

        # Filter by depth
        if max_depth is None:
            print("No depth filter applied (max_depth=None).")
            filtered = topic_data
        else:
            valid_uris = set()
            for t in topic_data:
                d = self._calculate_depth(t["uri"], hierarchy_data, max_depth=max_depth)
                if d <= max_depth:
                    valid_uris.add(t["uri"])
            filtered = [t for t in topic_data if t["uri"] in valid_uris]
            print(f"Filtered to {len(filtered)} topics with depth ≤ {max_depth}.")

        return filtered, hierarchy_data

    def _calculate_depth(self, topic_uri: str, hierarchy_data: List[Dict], max_depth: int = 4) -> int:
        if not hierarchy_data:
            return 1
        depth = 1
        cur = topic_uri
        while depth <= max_depth:
            parents = [h["super"] for h in hierarchy_data if h["sub"] == cur]
            if not parents:
                break
            depth += 1
            cur = parents[0]
        return depth

    def import_to_neo4j(self, topics: List[Dict], hierarchy_data: List[Dict]):
        self.ensure_constraints()
        
        print(f"Importing {len(topics)} topics to Neo4j (merge by label_norm)...")
        enriched = []
        for t in topics:
            enriched.append({
                "uri": t["uri"], 
                "label": t["label"],
                "label_norm": _canonical_form(t["label"]),
            })
            
        self.graph.query(
            """
            UNWIND $topics AS topic
            MERGE (t:Topic {label_norm: topic.label_norm})
            ON CREATE SET 
                t.label = topic.label,
                t.uris = [topic.uri]
            ON MATCH SET 
                t.uris = CASE 
                    WHEN topic.uri IN t.uris THEN t.uris 
                    ELSE t.uris + topic.uri 
                END,
                t.label = CASE 
                    WHEN size(topic.label) < size(t.label) THEN topic.label 
                    ELSE t.label 
                END
            """,
            {"topics": enriched},
        )
        print("Topics imported successfully (merged by canonical form).")

        if hierarchy_data:
            print(f"Importing {len(hierarchy_data)} hierarchical relationships...")
            # Map hierarchy by label_norm instead of URI
            hierarchy_mapped = []
            uri_to_norm = {t["uri"]: _canonical_form(t["label"]) for t in topics}
            
            for rel in hierarchy_data:
                sub_norm = uri_to_norm.get(rel["sub"])
                super_norm = uri_to_norm.get(rel["super"])
                if sub_norm and super_norm and sub_norm != super_norm:
                    hierarchy_mapped.append({
                        "sub_norm": sub_norm,
                        "super_norm": super_norm
                    })
            
            self.graph.query(
                """
                UNWIND $relations AS rel
                MATCH (sub:Topic {label_norm: rel.sub_norm})
                MATCH (super:Topic {label_norm: rel.super_norm})
                MERGE (sub)-[:SUB_TOPIC_OF]->(super)
                """,
                {"relations": hierarchy_mapped},
            )
            print("Hierarchy imported successfully.")

    def merge_duplicates(self):
        print("Merging duplicate Topic nodes by label_norm...")

        # Find groups of duplicates
        duplicate_groups = self.graph.query("""
            MATCH (t:Topic)
            WITH t.label_norm AS norm, collect(t) AS nodes
            WHERE size(nodes) > 1
            RETURN norm, [n IN nodes | {id: elementId(n), label: n.label, uris: n.uris}] AS nodeInfo
        """)
        
        if not duplicate_groups:
            print("No duplicate topics found.")
            return
        
        print(f"Found {len(duplicate_groups)} groups of duplicates to merge...")
        
        for group in duplicate_groups:
            norm = group["norm"]
            nodes = group["nodeInfo"]
            
            # Keep the first node, merge others into it
            keep_id = nodes[0]["id"]
            merge_ids = [n["id"] for n in nodes[1:]]
            
            print(f"Merging {len(merge_ids)} duplicates for '{norm}' into {keep_id}")
            
            all_uris = []
            for n in nodes:
                if n["uris"]:
                    if isinstance(n["uris"], list):
                        all_uris.extend(n["uris"])
                    else:
                        all_uris.append(n["uris"])
            
            self.graph.query("""
                MATCH (keep:Topic) WHERE elementId(keep) = $keepId
                SET keep.uris = $allUris
            """, {"keepId": keep_id, "allUris": list(set(all_uris))})
            
            # Transfer all relationships from duplicates to the kept node
            for merge_id in merge_ids:
                # Transfer incoming relationships
                self.graph.query("""
                    MATCH (source)-[r]->(dup:Topic) WHERE elementId(dup) = $dupId
                    MATCH (keep:Topic) WHERE elementId(keep) = $keepId
                    WITH source, r, keep, type(r) AS relType, properties(r) AS props
                    CALL apoc.create.relationship(source, relType, props, keep) YIELD rel
                    DELETE r
                """, {"dupId": merge_id, "keepId": keep_id})
                
                # Transfer outgoing relationships  
                self.graph.query("""
                    MATCH (dup:Topic)-[r]->(target) WHERE elementId(dup) = $dupId
                    MATCH (keep:Topic) WHERE elementId(keep) = $keepId
                    WITH keep, r, target, type(r) AS relType, properties(r) AS props
                    CALL apoc.create.relationship(keep, relType, props, target) YIELD rel
                    DELETE r
                """, {"dupId": merge_id, "keepId": keep_id})
                
                # Delete the duplicate node
                self.graph.query("""
                    MATCH (dup:Topic) WHERE elementId(dup) = $dupId
                    DELETE dup
                """, {"dupId": merge_id})
        
        print("Duplicate merging completed.")

    def build_and_save_cso_index(
        self,
        topics: List[Dict],
        index_path: str = "data/cso_topics.faiss",
        labels_path: str = "data/cso_labels.json",
        use_normalized: bool = True,
        batch_size: int = 512,
    ):
        if faiss is None:
            raise RuntimeError("faiss is not available. please install it with: pip install faiss-cpu")

        labels = [t["label"] for t in topics]
        labels_for_embed = [_normalize_label(l) if use_normalized else l for l in labels]

        print(f"Embedding {len(labels_for_embed)} CSO topics with '{self.embed_model_name}' ...")
        vecs = self.embedder.encode(
            labels_for_embed,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vecs = np.asarray(vecs, dtype="float32")
        dim = vecs.shape[1]

        index = faiss.IndexFlatIP(dim)  # cosine similarity via dot-product on normalized vectors
        index.add(vecs)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        print(f"Saved FAISS index -> {index_path}")
        print(f"Saved labels      -> {labels_path}")

    def load_index(self, index_path: str, labels_path: str):
        if faiss is None:
            raise RuntimeError("faiss is not available. please install it with: pip install faiss-cpu")
        index = faiss.read_index(index_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return index, labels

    def search_topics(self, terms: List[str], top_k: int, index, labels: List[str]):
        if not terms:
            return []
        q = self.embedder.encode(
            [_normalize_label(t) for t in terms],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        D, I = index.search(q, top_k)
        out = []
        for i, term in enumerate(terms):
            cand = [(labels[j], float(D[i, k])) for k, j in enumerate(I[i])]
            out.append(cand)
        return out
