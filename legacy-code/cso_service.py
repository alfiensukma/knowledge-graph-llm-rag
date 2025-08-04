import os
import rdflib
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict

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
            """Anda adalah ahli ontologi ilmu komputer. Berikan nama panjang (expanded form) dari daftar topik 
            atau singkatan berikut dalam konteks Computer Science Ontology (CSO). Jika topik adalah singkatan, 
            kembalikan nama panjangnya. Jika Anda tidak tahu nama panjangnya atau topik tidak valid, tandai sebagai 
            'unknown'. Jangan sertakan topik 'computer science' karena itu adalah ontologi root; gunakan topik anak 
            yang lebih spesifik. Kembalikan hasil dalam format JSON: [{{"label": "<original_label>", "expanded_label": "<expanded_label>"}}].
            Daftar topik: ```{topics}```\n\nJSON Output: """
        )
        self.chain = self.prompt | self.llm | self.parser

    def extract_topics_with_hierarchy(self, cso_file_path: str, max_depth: int = 4) -> List[Dict]:
        """Extracts topics from CSO RDF file, limited to max_depth hierarchy levels, excluding root topics like 'computer science'."""
        print(f"Loading CSO ontology from {cso_file_path}...")
        g = rdflib.Graph()
        g.parse(cso_file_path, format="turtle")
        print("CSO ontology loaded.")

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
        topic_data = [{"uri": str(r.uri), "label": str(r.label)} for r in results]
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

        # Proses topik dengan LLM untuk nama panjang
        batch_size = 50
        validated_topics = []
        for i in range(0, len(filtered_topics), batch_size):
            batch = filtered_topics[i:i + batch_size]
            topic_labels = [t["label"] for t in batch]
            try:
                llm_output = self.chain.invoke({"topics": ", ".join(topic_labels)})
                print(f"  > LLM output for batch {i//batch_size + 1}: {llm_output}")
                for topic, llm_result in zip(batch, llm_output):
                    if llm_result["expanded_label"] != "unknown":
                        validated_topics.append({
                            "uri": topic["uri"],
                            "label": llm_result["expanded_label"]
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
        """Imports validated topics and hierarchy to Neo4j."""
        print(f"Importing {len(topics)} topics to Neo4j...")
        self.graph.query(
            """
            UNWIND $topics AS topic
            MERGE (t:Topic {uri: topic.uri})
            SET t.label = topic.label
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