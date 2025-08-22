import os
import rdflib
import time
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Tuple
import re
from tqdm import tqdm

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
            """Anda adalah ahli ontologi ilmu komputer yang bertugas memvalidasi dan menormalisasi topik-topik CSO.
            TUGAS:
            Periksa daftar topik berikut dan untuk setiap topik:
            1. Validasi apakah ini topik ilmu komputer yang valid
            2. Normalisasi nama (singkatan, plural, formatting)
            3. Berikan nama lengkap jika diperlukan
            ATURAN:
            - Jika topik adalah singkatan, berikan nama lengkapnya (contoh: 'ai' -> 'artificial intelligence')
            - Hilangkan bentuk jamak ('networks' -> 'network')
            - Normalisasi formatting (lowercase, hapus karakter khusus)
            - JANGAN ubah makna asli dari topik
            - Jika tidak valid/tidak dikenal, kembalikan 'unknown'
            - Jangan proses 'computer science' (root topic)
            INPUT TOPICS:
            {topics}
            OUTPUT FORMAT (JSON):
            [
                {{
                    "label": "<original_label>",
                    "expanded_label": "<normalized/expanded_label>",
                    "reason": "<alasan perubahan atau 'valid' jika tidak ada perubahan>"
                }}
            ]
            JSON Output:""")
        
        self.chain = self.prompt | self.llm | self.parser
        self.TOKENS_PER_MINUTE_LIMIT = 1_000_000
        self.PROMPT_TOKENS = 200  # Base prompt tokens
        self.AVG_TOKENS_PER_TOPIC = 15  # Average tokens per topic (label + expanded form)
        self.SAFETY_MARGIN = 0.9  # 90% of limit to be safe
        
        # Calculate optimal batch size
        self.BATCH_SIZE = int(
            (self.TOKENS_PER_MINUTE_LIMIT * self.SAFETY_MARGIN - self.PROMPT_TOKENS) 
            // self.AVG_TOKENS_PER_TOPIC
        )
        
        # Token tracking
        self.current_minute_tokens = 0
        self.last_request_time = time.time()
        
    def track_token_usage(self, text_length: int) -> None:
        current_time = time.time()
        # Reset counter if new minute
        if current_time - self.last_request_time >= 60:
            self.current_minute_tokens = 0
            self.last_request_time = current_time
        
        # Estimate tokens (rough approximation: 4 chars = 1 token)
        estimated_tokens = (text_length // 4) + self.PROMPT_TOKENS
        self.current_minute_tokens += estimated_tokens
        
        # Check if we need to wait
        if self.current_minute_tokens >= self.TOKENS_PER_MINUTE_LIMIT * self.SAFETY_MARGIN:
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                print(f"\nApproaching rate limit. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.current_minute_tokens = 0
                self.last_request_time = time.time()
                
    def process_topic_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of topics with improved error handling"""
        if not batch:
            return []
            
        topics_to_process = [t["label"] for t in batch]
        total_text_length = sum(len(t) for t in topics_to_process)
        
        # Track token usage
        self.track_token_usage(total_text_length)
        
        try:
            # Process in smaller sub-batches if needed
            MAX_TOPICS_PER_REQUEST = 50
            if len(topics_to_process) > MAX_TOPICS_PER_REQUEST:
                results = []
                for i in range(0, len(topics_to_process), MAX_TOPICS_PER_REQUEST):
                    sub_batch = topics_to_process[i:i + MAX_TOPICS_PER_REQUEST]
                    sub_results = self.chain.invoke({"topics": ", ".join(sub_batch)})
                    results.extend(sub_results)
                    time.sleep(1)  # delay
                return results
            
            return self.chain.invoke({"topics": ", ".join(topics_to_process)})
            
        except Exception as e:
            print(f"\nError processing batch: {e}")
            if "rate limit" in str(e).lower():
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                return self.process_topic_batch(batch)
            if len(batch) > 1:
                # Split batch and retry
                mid = len(batch) // 2
                return (
                    self.process_topic_batch(batch[:mid]) +
                    self.process_topic_batch(batch[mid:])
                )
            raise e
    
    def clear_existing_data(self):
        """Clear all Topic nodes and relationships"""
        print("Clearing existing Topic nodes and relationships...")
        self.graph.query("""
            MATCH (t:Topic)
            DETACH DELETE t
        """)
        print("Database cleared.")

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

    def extract_topics_with_hierarchy(self, cso_file_path: str) -> Tuple[List[Dict], List[Dict]]:
        print(f"Loading CSO ontology from {cso_file_path}...")
        g = rdflib.Graph()
        try:
            g.parse(cso_file_path, format="turtle")
            print("CSO ontology loaded successfully.")
        except Exception as e:
            print(f"Failed to load CSO ontology: {e}")
            return [], []

        # Extract topics
        topic_query = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX cso: <http://cso.kmi.open.ac.uk/schema/cso#>
            SELECT DISTINCT ?uri ?label
            WHERE {
                ?uri a cso:Topic ;
                    rdfs:label ?label .
                FILTER (LCASE(str(?label)) != "computer science")
            }
            ORDER BY ?label
        """
        topic_data = [
            {
                "uri": str(r.uri),
                "label": str(r.label),
                "normalized": self.normalize_topic(str(r.label))
            }
            for r in g.query(topic_query)
        ]
        print(f"Found {len(topic_data)} unique topics.")

        # Extract hierarchy
        hierarchy_query = """
            PREFIX cso: <http://cso.kmi.open.ac.uk/schema/cso#>
            SELECT DISTINCT ?sub_topic ?super_topic
            WHERE {
                ?super_topic cso:superTopicOf ?sub_topic .
            }
        """
        hierarchy_data = [
            {"sub": str(r.sub_topic), "super": str(r.super_topic)} 
            for r in g.query(hierarchy_query)
        ]
        print(f"Found {len(hierarchy_data)} hierarchical relationships.")

        # Calculate optimal batch size based on token limits
        estimated_chars_per_topic = 50  # Average characters per topic including JSON structure
        max_chars_per_batch = (self.TOKENS_PER_MINUTE_LIMIT * self.SAFETY_MARGIN - self.PROMPT_TOKENS) * 4
        optimal_batch_size = min(
            50,  # Maximum batch size for reliable processing
            int(max_chars_per_batch // estimated_chars_per_topic)
        )
        
        # Track processed topics and results
        validated_topics = []
        seen_uris = set()
        failed_topics = []
        
        # Process topics in optimized batches
        batches = [
            topic_data[i:i + optimal_batch_size] 
            for i in range(0, len(topic_data), optimal_batch_size)
        ]
        
        print(f"\nProcessing {len(topic_data)} topics in {len(batches)} batches "
            f"(max {optimal_batch_size} topics per batch)")
        
        with tqdm(total=len(batches), desc="Processing topics") as pbar:
            for batch_idx, batch in enumerate(batches, 1):
                try:
                    to_process = [
                        t for t in batch 
                        if t["uri"] not in seen_uris
                    ]
                    
                    if to_process:
                        results = self.process_topic_batch(to_process)
                        
                        # Handle results
                        for topic, result in zip(to_process, results):
                            seen_uris.add(topic["uri"])
                            
                            if (result.get("expanded_label") and 
                                result["expanded_label"].lower() != "unknown"):
                                
                                normalized_label = self.normalize_topic(result["expanded_label"])
                                if normalized_label != "unknown":
                                    validated_topics.append({
                                        "uri": topic["uri"],
                                        "label": result["expanded_label"],
                                        "normalized_label": normalized_label,
                                        "original_label": topic["label"]
                                    })
                                    
                                    if topic["label"] != result["expanded_label"]:
                                        print(f"\nTransformed: {topic['label']} -> {result['expanded_label']}"
                                            f"\nReason: {result.get('reason', 'No reason provided')}")
                            else:
                                failed_topics.append(topic["label"])
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "Validated": len(validated_topics),
                        "Failed": len(failed_topics),
                        "Batch": f"{batch_idx}/{len(batches)}"
                    })
                    
                    # Rate limiting
                    if batch_idx < len(batches):
                        time.sleep(0.5)
                    
                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {e}")
                    failed_topics.extend([t["label"] for t in to_process])
                    time.sleep(5)
                    continue
        
        # Final statistics
        print(f"\nProcessing complete:")
        print(f"- Successfully validated: {len(validated_topics)} topics")
        print(f"- Failed to validate: {len(failed_topics)} topics")
        if failed_topics:
            print("\nSample of failed topics:")
            for topic in failed_topics[:5]:
                print(f"- {topic}")
        
        # Filter hierarchy to only include validated topics
        valid_uris = {t["uri"] for t in validated_topics}
        filtered_hierarchy = [
            rel for rel in hierarchy_data
            if rel["sub"] in valid_uris and rel["super"] in valid_uris
        ]
        print(f"\nFiltered hierarchy relationships: {len(filtered_hierarchy)}")
        
        return validated_topics, filtered_hierarchy
    
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