import os
import rdflib
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict
import re

class CSOService:
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str, llm):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=List[dict])
        self.prompt = ChatPromptTemplate.from_template(
            """Anda adalah ahli ontologi ilmu komputer. 
            - Berikan nama panjang (expanded form) dari daftar topik atau singkatan berikut dalam konteks Computer 
              Science Ontology (CSO). 
            - Jika topik adalah singkatan, kembalikan nama panjangnya. 
            - Jika topik memiliki akhiran 's' atau 'es', kembalikan bentuk tunggalnya (misalnya, 'networks' menjadi 'network',
              'databases' menjadi 'database'). 
            - Jika topik tidak valid atau tidak diketahui (tidak sesuai dengan konteks CSO), tandai sebagai 'unknown'. 
            - Jangan sertakan topik 'computer science' karena itu adalah ontologi root; gunakan topik anak yang lebih spesifik. 
            - Pastikan tidak ada duplikasi dalam output. 
            Kembalikan hasil dalam format JSON: [{{"label": "<original_label>", "expanded_label": "<expanded_label>"}}].
            Daftar topik: ```{topics}```\n\nJSON Output: """
        )
        self.chain = self.prompt | self.llm | self.parser

    def normalize_topic(self, topic: str) -> str:
        """Normalize topic by converting to lowercase and removing 's' or 'es' plural forms."""
        if not topic or not isinstance(topic, str):
            return "unknown"
        normalized = topic.lower().strip()
        normalized = re.sub(r'\b(s|es)$', '', normalized)
        if normalized == "computer science":
            return "unknown"
        return normalized

    def get_existing_topics(self, topics: List[str]) -> Dict[str, str]:
        """Check for existing topics in Neo4j and return a mapping of normalized to existing labels."""
        normalized_to_existing = {}
        normalized_topics = [self.normalize_topic(topic) for topic in topics]
        if not normalized_topics:
            return normalized_to_existing
        
        result = self.graph.query(
            """
            MATCH (t:Topic)
            WHERE t.normalized_label IN $normalized_labels
            RETURN t.label, t.normalized_label AS normalized
            """,
            {"normalized_labels": [t for t in normalized_topics if t != "unknown"]}
        )
        for record in result:
            normalized_to_existing[record["normalized"]] = record["t.label"]
        
        return normalized_to_existing

    def extract_topics_with_hierarchy(self, cso_file_path: str, max_depth: int = 4) -> List[Dict]:
        """Extracts topics from CSO RDF file, limited to max_depth hierarchy levels, excluding root topics like 'computer science'."""
        print(f"Loading CSO ontology from {cso_file_path}...")
        g = rdflib.Graph()
        try:
            g.parse(cso_file_path, format="turtle")
            print("CSO ontology loaded.")
        except Exception as e:
            print(f"Failed to load CSO ontology: {e}")
            return [], []

        # Ambil semua topik
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
        topic_data = [
            {
                "uri": str(r.uri),
                "label": str(r.label),
                "normalized": self.normalize_topic(str(r.label))
            }
            for r in results
        ]
        print(f"Found {len(topic_data)} topics. Filtering by hierarchy depth...")

        # Ambil hubungan hierarki
        hierarchy_query = """
            PREFIX cso: <http://cso.kmi.open.ac.uk/schema/cso#>
            SELECT ?sub_topic ?super_topic
            WHERE {
                ?super_topic cso:superTopicOf ?sub_topic .
            }
        """
        hierarchy_results = g.query(hierarchy_query)
        hierarchy_data = [{"sub": str(r.sub_topic), "super": str(r.super_topic)} for r in hierarchy_results]

        # Filter topik dengan kedalaman <= 4
        valid_topics = set()
        topic_depth = {}
        for topic in topic_data:
            topic_uri = topic["uri"]
            depth = self._calculate_depth(topic_uri, hierarchy_data)
            if depth <= max_depth:
                valid_topics.add(topic_uri)
                topic_depth[topic_uri] = depth

        filtered_topics = [t for t in topic_data if t["uri"] in valid_topics]
        print(f"Filtered to {len(filtered_topics)} topics with depth <= {max_depth}.")

        # Check for existing topics in Neo4j
        existing_topics = self.get_existing_topics([t["label"] for t in filtered_topics])

        # Proses topik dengan LLM untuk nama panjang, hindari duplikasi
        batch_size = 50
        validated_topics = []
        seen_normalized = set()
        for i in range(0, len(filtered_topics), batch_size):
            batch = filtered_topics[i:i + batch_size]
            topics_to_process = []
            for topic in batch:
                normalized = topic["normalized"]
                if normalized in seen_normalized or normalized == "unknown":
                    continue
                if normalized in existing_topics:
                    validated_topics.append({
                        "uri": topic["uri"],
                        "label": existing_topics[normalized],
                        "normalized_label": normalized
                    })
                    seen_normalized.add(normalized)
                    continue
                topics_to_process.append(topic["label"])
                seen_normalized.add(normalized)
                
            if topics_to_process:
                try:
                    llm_output = self.chain.invoke({"topics": ", ".join(topics_to_process)})
                    print(f"  > LLM output for batch {i//batch_size + 1}: {llm_output}")
                    for topic, llm_result in zip([t for t in batch if t["label"] in topics_to_process], llm_output):
                        if llm_result["expanded_label"] != "unknown":
                            normalized = self.normalize_topic(llm_result["expanded_label"])
                            validated_topics.append({
                                "uri": topic["uri"],
                                "label": llm_result["expanded_label"],
                                "normalized_label": normalized
                            })
                except Exception as e:
                    print(f"  > Error processing batch {i//batch_size + 1}: {e}")

        print(f"Validated {len(validated_topics)} topics.")
        return validated_topics, hierarchy_data

    def _calculate_depth(self, topic_uri: str, hierarchy_data: List[Dict]) -> int:
        """Calculates the depth of a topic in the hierarchy."""
        if not hierarchy_data:
            return 1
        depth = 1
        current = topic_uri
        while True:
            parents = [h["super"] for h in hierarchy_data if h["sub"] == current]
            if not parents:
                break
            depth += 1
            current = parents[0]  # Ambil parent pertama
            if depth > 4:  # Batasi hingga kedalaman 4
                break
        return depth

    def import_to_neo4j(self, topics: List[Dict], hierarchy_data: List[Dict]):
        """Imports validated topics and hierarchy to Neo4j, avoiding duplicates."""
        print(f"Importing {len(topics)} topics to Neo4j...")
        self.graph.query(
            """
            UNWIND $topics AS topic
            MERGE (t:Topic {normalized_label: topic.normalized_label})
            SET t.label = topic.label,
                t.uri = topic.uri
            """, {"topics": topics}
        )
        print("Topics imported successfully.")

        print(f"Importing {len(hierarchy_data)} hierarchical relationships to Neo4j...")
        self.graph.query(
            """
            UNWIND $relations AS rel
            MATCH (sub:Topic {uri: rel.sub})
            MATCH (super:Topic {uri: rel.super})
            MERGE (sub)-[:SUB_TOPIC_OF]->(super)
            """, {"relations": hierarchy_data}
        )
        print("Hierarchy imported successfully.")