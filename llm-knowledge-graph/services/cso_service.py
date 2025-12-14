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

MAX_DEPTH = 6
ABBREV_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cs_abbreviations.json")

def _load_abbreviations() -> Dict[str, str]:
    try:
        with open(ABBREV_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  > Warning: Abbreviations file not found at {ABBREV_FILE}")
        return {}
    except Exception as e:
        print(f"  > Warning: Error loading abbreviations: {e}")
        return {}

_ABBREV_MAP = _load_abbreviations()

def _normalize_label(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s*\([^)]+\)\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _singularize_last_token(text: str) -> str:
    parts = text.split()
    if not parts:
        return text
    last = parts[-1]
    if len(last) > 3 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 4 and last.endswith(("sses", "shes", "ches")):
        last = last[:-2]
    elif len(last) > 3 and last.endswith("es"):
        last = last[:-2]
    elif len(last) > 3 and last.endswith("s"):
        last = last[:-1]
    parts[-1] = last
    return " ".join(parts)

def _canonical_label(label: str) -> str:
    return _singularize_last_token(_normalize_label(label))


def _expand_abbrev(label: str) -> str:
    raw = label.strip()
    base = _normalize_label(raw)
    key = base.replace(" ", "")
    if key in _ABBREV_MAP:
        return _ABBREV_MAP[key]
    if raw.isupper() and 2 <= len(raw) <= 6 and key in _ABBREV_MAP:
        return _ABBREV_MAP[key]
    return label


class CSOService:
    """
    Token-efficient CSO import:
    1) Parse RDF → topics + hierarchy.
    2) Expand known abbreviations (rule-based).
    3) Canonicalize labels (normalize + singularize).
    4) Deduplicate exact same canonicals.
    5) Embedding-based near-duplicate clustering (FAISS + cosine).
    6) (Optional) LLM only for small ambiguous clusters (few items) to pick a canonical title.
    7) Import to Neo4j with constraints; label_norm stored for future joins/merges.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        llm=None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cluster_threshold: float = 0,   # cosine threshold to join topics
        use_llm: bool = False
    ):
        self.graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        self.llm = llm
        self.embed_model_name = embed_model
        self.embedder: Optional[SentenceTransformer] = None
        self.cluster_threshold = float(cluster_threshold)
        self.use_llm = bool(use_llm)
        self._ensure_embedder()

    def _ensure_embedder(self):
        self.embedder = SentenceTransformer(self.embed_model_name)

    def ensure_constraints(self):
        self.graph.query("CREATE CONSTRAINT topic_uri_if_not_exists IF NOT EXISTS FOR (t:Topic) REQUIRE t.uri IS UNIQUE")
        self.graph.query("CREATE CONSTRAINT topic_label_norm_if_not_exists IF NOT EXISTS FOR (t:Topic) REQUIRE t.label_norm IS UNIQUE")

    def clear_existing_data(self):
        print("Clearing existing Topic nodes and relationships...")
        self.graph.query("MATCH (t:Topic) DETACH DELETE t")
        print("Database cleared.")
        
    def build_and_save_cso_index(
        self,
        topics: List[Dict],
        index_path: str = "data/cso_topics.faiss",
        labels_path: str = "data/cso_labels.json",
        use_normalized: bool = True,
        batch_size: int = 512,
    ):
        if faiss is None:
            raise RuntimeError("faiss is not available. Install with: pip install faiss-cpu")

        if not topics:
            print("No topics provided for index build.")
            return

        labels = [t["label_norm"] if use_normalized else t["label"] for t in topics]
        texts_for_embed = labels

        print(f"Embedding {len(texts_for_embed)} topics with '{self.embed_model_name}' ...")
        vecs = self.embedder.encode(
            texts_for_embed,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)

        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

    def load_index(self, index_path: str, labels_path: str):
        if faiss is None:
            raise RuntimeError("faiss is not available. Install first")
        index = faiss.read_index(index_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return index, labels

    def extract_topics_with_hierarchy(self, cso_file_path: str, max_depth: int = MAX_DEPTH) -> Tuple[List[Dict], List[Dict]]:
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
        topics = [{"uri": str(r.uri), "label": str(r.label)} for r in results]

        hierarchy_query = """
        PREFIX cso: <http://cso.kmi.open.ac.uk/schema/cso#>
        SELECT ?sub_topic ?super_topic
        WHERE {
            ?super_topic cso:superTopicOf ?sub_topic .
        }
        """
        h = g.query(hierarchy_query)
        hierarchy = [{"sub": str(r.sub_topic), "super": str(r.super_topic)} for r in h]

        # ilter by depth
        valid = set()
        for t in topics:
            d = self._calculate_depth(t["uri"], hierarchy, max_depth=max_depth)
            if d <= max_depth:
                valid.add(t["uri"])
        topics = [t for t in topics if t["uri"] in valid]
        print(f"Collected {len(topics)} topics (depth ≤ {max_depth}).")
        return topics, hierarchy

    def _calculate_depth(self, topic_uri: str, hierarchy: List[Dict], max_depth: int = MAX_DEPTH) -> int:
        if not hierarchy:
            return 1
        depth = 1
        cur = topic_uri
        while depth <= max_depth:
            parents = [r["super"] for r in hierarchy if r["sub"] == cur]
            if not parents:
                break
            depth += 1
            cur = parents[0]
        return depth

    def prepare_topics(self, raw_topics: List[Dict]) -> List[Dict]:
        if not raw_topics:
            return []

        enriched: List[Dict] = []
        for t in raw_topics:
            orig = t["label"]
            expanded = _expand_abbrev(orig)
            canon = _canonical_label(expanded)
            enriched.append({
                "uri": t["uri"],
                "label": expanded,
                "label_norm": canon,
                "orig_label": orig
            })

        dedup_map: Dict[str, Dict] = {}
        for t in enriched:
            key = t["label_norm"]
            if key not in dedup_map or len(t["label"]) < len(dedup_map[key]["label"]):
                dedup_map[key] = t
        dedup_list = list(dedup_map.values())
        print(f"Exact canonical dedup → {len(dedup_list)} unique labels.")

        clustered = self._cluster_by_embeddings(dedup_list)

        if self.use_llm and self.llm:
            clustered = self._llm_refine_cluster_titles(clustered)

        final_topics = []
        for c in clustered:
            rep = c["representative"]
            final_topics.append({
                "uri": rep["uri"],
                "label": rep["label"],
                "label_norm": rep["label_norm"],
            })
        print(f"Embedding clustering → {len(final_topics)} canonical topics.")
        return final_topics

    def _cluster_by_embeddings(self, topics: List[Dict]) -> List[Dict]:
        if not topics:
            return []

        texts = [t["label_norm"] for t in topics]
        vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        n, d = vecs.shape

        # Build index
        if faiss is None:
            # Fallback: O(n^2) cosine via dot product
            sims = vecs @ vecs.T
            used = np.zeros(n, dtype=bool)
            clusters = []
            for i in range(n):
                if used[i]:
                    continue
                idx = np.where(sims[i] >= self.cluster_threshold)[0].tolist()
                idx.sort()
                for j in idx:
                    used[j] = True
                members = [topics[j] for j in idx]
                rep = self._pick_representative(members)
                clusters.append({"representative": rep, "members": members})
            return clusters

        index = faiss.IndexFlatIP(d)  # cosine = dot product (normalized)
        index.add(vecs)

        visited = np.zeros(n, dtype=bool)
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            # retrieve top neighbors (k heuristic)
            k = min(32, n)
            D, I = index.search(vecs[i:i+1], k)
            cand = [j for j, s in zip(I[0].tolist(), D[0].tolist()) if s >= self.cluster_threshold]
            cand = [j for j in cand if not visited[j]]
            for j in cand:
                visited[j] = True
            members = [topics[j] for j in cand]
            rep = self._pick_representative(members)
            clusters.append({"representative": rep, "members": members})
        return clusters

    def _pick_representative(self, members: List[Dict]) -> Dict:
        members_sorted = sorted(members, key=lambda x: (len(x["label"]), x["label"]))
        rep = {
            "uri": members_sorted[0]["uri"],
            "label": members_sorted[0]["label"],
            "label_norm": members_sorted[0]["label_norm"],
        }
        return rep

    def _llm_refine_cluster_titles(self, clusters: List[Dict]) -> List[Dict]:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        prompt = ChatPromptTemplate.from_template(
            "Anda adalah ahli ontologi ilmu komputer. Diberikan daftar label topik yang sangat mirip:\n"
            "{labels}\n\n"
            "Pilih satu nama kanonik terbaik (jelas, baku, bahasa Inggris konsisten, tanpa plural jika tidak perlu). "
            "Output JSON: {{\"canonical\": \"<nama_terpilih>\"}}"
        )
        parser = JsonOutputParser(pydantic_object=dict)
        chain = prompt | self.llm | parser

        for c in clusters:
            names = sorted({m["label"] for m in c["members"]})
            try:
                out = chain.invoke({"labels": ", ".join(names)})
                canonical = (out or {}).get("canonical")
                if isinstance(canonical, str) and canonical.strip():
                    can = canonical.strip()
                    c["representative"]["label"] = can
                    c["representative"]["label_norm"] = _canonical_label(can)
            except Exception:
                pass
        return clusters

    def import_to_neo4j(self, topics: List[Dict], hierarchy: List[Dict]):
        if not topics:
            print("No topics to import.")
            return
        print(f"Importing {len(topics)} topics to Neo4j...")
        self.graph.query(
            """
            UNWIND $rows AS row
            MERGE (t:Topic {uri: row.uri})
            SET t.label = row.label,
                t.label_norm = row.label_norm
            """,
            {"rows": topics},
        )
        print("Topics imported.")

        if hierarchy:
            print("Importing hierarchy...")
            self.graph.query(
                """
                UNWIND $rels AS rel
                MATCH (sub:Topic {uri: rel.sub})
                MATCH (sup:Topic {uri: rel.super})
                MERGE (sub)-[:SUB_TOPIC_OF]->(sup)
                """,
                {"rels": hierarchy},
            )
            print("Hierarchy imported.")

    def merge_duplicates_apoc(self):
        self.graph.query(
            """
            CALL apoc.periodic.iterate(
              'MATCH (t:Topic) RETURN t.label_norm AS ln, collect(t) AS nodes',
              'WITH ln, nodes WHERE size(nodes) > 1
               WITH ln, nodes, nodes[0] AS keep, nodes[1..] AS dups
               CALL apoc.refactor.mergeNodes(dups + keep, {properties:"discard", mergeRels:true}) YIELD node
               SET node.label_norm = ln
               RETURN count(*)',
              {batchSize:50, parallel:false}
            )
            YIELD batches, total
            RETURN batches, total
            """
        )
        print("APOC merge by label_norm completed.")
