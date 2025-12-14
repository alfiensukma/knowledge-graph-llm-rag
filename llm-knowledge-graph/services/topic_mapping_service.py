from typing import List, Set, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re
import json
import os

VALIDATION_SEMANTIC_THRESHOLD = 90
ABBREV_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cs_abbreviations.json")

def load_abbreviations() -> Dict[str, str]:
    try:
        with open(ABBREV_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  > Warning: Abbreviations file not found at {ABBREV_FILE}")
        return {}
    except Exception as e:
        print(f"  > Warning: Error loading abbreviations: {e}")
        return {}

ABBREV_MAP = load_abbreviations()
SKIP_TERMS = {
    "data", "algorithm", "algorithms", "method", "methods", "system", "systems",
    "model", "models", "technique", "techniques", "approach", "approaches",
    "computer", "science", "computer science", "technology", "application",
}


def normalize_text(text: str) -> str:
    text = re.sub(r'\s*\([^)]+\)\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()


def expand_abbreviation(text: str) -> str:
    """Expand known abbreviations to full form."""
    normalized = normalize_text(text)
    # Check exact match
    if normalized in ABBREV_MAP:
        return ABBREV_MAP[normalized]
    # Check if it's uppercase abbreviation
    if text.isupper() and 2 <= len(text) <= 6:
        key = normalized.replace(" ", "")
        if key in ABBREV_MAP:
            return ABBREV_MAP[key]
    return text


class TopicMappingService:
    def __init__(self, llm, graph_service):
        self.llm = llm
        self.graph_service = graph_service
        
        print("Fetching existing topics and hierarchy from Neo4j for validation...")
        self.cso_topics, self.hierarchy = self._fetch_topics_and_hierarchy_from_neo4j()
        print(f"-> Found {len(self.cso_topics)} topics and {len(self.hierarchy)} hierarchical relationships in the graph.")
        
        self.validate_parser = JsonOutputParser(pydantic_object=dict)
        self.validate_prompt = ChatPromptTemplate.from_template(
            """Anda adalah ahli ontologi ilmu komputer. Validasi topik kandidat terhadap Computer Science Ontology (CSO).

KANDIDAT: "{candidate}"
EXPANDED FORM: "{expanded_candidate}"
CSO TOPICS (sample): {cso_topics}

INSTRUKSI:
1. Cari kecocokan semantik antara kandidat (atau expanded form-nya) dengan topik CSO
2. PENTING: Perhatikan akronim/singkatan (contoh: CNN = convolutional neural network, NLP = natural language processing)
3. Prioritaskan kecocokan yang spesifik dan relevan dalam konteks ilmu komputer
4. Jangan match dengan topik yang terlalu umum seperti 'data', 'algorithm', 'method' saja
5. Threshold minimum: {threshold}% similarity

CONTOH MATCHING:
- "cnn" atau "CNN" → "convolutional neural network"
- "nlp" → "natural language processing"
- "encrypted data" → "data privacy"
- "data" saja → None (terlalu umum)

OUTPUT FORMAT (JSON):
{{"candidate": "<original_candidate>", "matched_topic": "<matched_cso_topic or None>", "reason": "<explanation if None>"}}
"""
        )
        self.validate_chain = self.validate_prompt | self.llm | self.validate_parser

    def _fetch_topics_and_hierarchy_from_neo4j(self) -> tuple:
        try:
            topic_results = self.graph_service.graph.query(
                "MATCH (t:Topic) WHERE t.label <> 'computer science' RETURN t.label AS label"
            )
            topics = [record['label'] for record in topic_results]
            if not topics:
                print("  > Warning: No topics found in Neo4j database!")

            # Get hierarchy
            hierarchy_results = self.graph_service.graph.query(
                """
                MATCH (sub:Topic)-[:SUB_TOPIC_OF]->(super:Topic)
                WHERE sub.label <> 'computer science' AND super.label <> 'computer science'
                RETURN sub.label AS sub_topic, super.label AS super_topic
                """
            )
            hierarchy = [f"{record['sub_topic']} -> {record['super_topic']}" for record in hierarchy_results]
            if not hierarchy:
                print("  > Warning: No hierarchy found in Neo4j database!")
            
            return topics, hierarchy
        except Exception as e:
            print(f"  > Error fetching topics/hierarchy from Neo4j: {e}")
            return [], []

    def map_topics_to_cso(self, candidate_topics: List[str]) -> List[str]:
        """
        Maps candidate topics to CSO topics using:
        1. Abbreviation expansion
        2. Exact matching with normalization
        3. LLM-based semantic matching
        """
        if not candidate_topics:
            print("  > No candidate topics to validate.")
            return []

        print(f"  > Mapping {len(candidate_topics)} candidate topics to CSO...")
        validated_topics: Set[str] = set()
        normalized_cso_topics = {t: normalize_text(t) for t in self.cso_topics}

        for candidate in candidate_topics:
            try:
                # Skip if too generic
                if normalize_text(candidate) in SKIP_TERMS:
                    print(f"  > Skipping '{candidate}' (too generic)")
                    continue
                
                # Step 1: Expand abbreviation
                expanded = expand_abbreviation(candidate)
                if expanded != candidate:
                    print(f"  > Expanded '{candidate}' -> '{expanded}'")
                
                # Step 2: Try exact match with original candidate
                normalized_candidate = normalize_text(candidate)
                matched = False
                
                for original_topic, normalized_topic in normalized_cso_topics.items():
                    if normalized_candidate == normalized_topic:
                        print(f"  > Exact match: '{candidate}' -> '{original_topic}'")
                        validated_topics.add(original_topic)
                        matched = True
                        break
                
                if matched:
                    continue
                
                # Step 3: Try exact match with expanded form
                if expanded != candidate:
                    normalized_expanded = normalize_text(expanded)
                    for original_topic, normalized_topic in normalized_cso_topics.items():
                        if normalized_expanded == normalized_topic:
                            print(f"  > Exact match (expanded): '{candidate}' ({expanded}) -> '{original_topic}'")
                            validated_topics.add(original_topic)
                            matched = True
                            break
                    
                    if matched:
                        continue
                
                # Step 4: Partial matching (contains)
                for original_topic, normalized_topic in normalized_cso_topics.items():
                    if normalized_candidate in normalized_topic or normalized_topic in normalized_candidate:
                        # Avoid too short matches
                        if len(normalized_candidate) >= 4 and len(normalized_topic) >= 4:
                            print(f"  > Partial match: '{candidate}' ≈ '{original_topic}'")
                            validated_topics.add(original_topic)
                            matched = True
                            break
                
                if matched:
                    continue
                
                # Step 5: LLM-based semantic matching
                print(f"  > No direct match for '{candidate}', trying LLM validation...")
                
                # Find relevant CSO topics based on keyword overlap for better context
                relevant_topics = self._find_relevant_topics(expanded if expanded != candidate else candidate, limit=100)
                
                validation_result = self.validate_chain.invoke({
                    "candidate": candidate,
                    "expanded_candidate": expanded,
                    "cso_topics": ", ".join(relevant_topics),
                    "threshold": VALIDATION_SEMANTIC_THRESHOLD
                })
                
                print(f"  > Validation result: {validation_result}")
                
                matched_topic = validation_result.get("matched_topic")
                if matched_topic and matched_topic != "None" and matched_topic.lower() != "none":
                    validated_topics.add(matched_topic)
                    print(f"  > LLM match: '{candidate}' -> '{matched_topic}'")
                else:
                    reason = validation_result.get("reason", "No match found")
                    print(f"  > No match for '{candidate}': {reason}")
                    
            except Exception as e:
                print(f"  > Error validating topic '{candidate}': {e}")
        
        result = list(validated_topics)
        print(f"  > Successfully mapped {len(result)}/{len(candidate_topics)} topics to CSO")
        return result

    def _find_relevant_topics(self, query: str, limit: int = 100) -> List[str]:
        """Find CSO topics that might be relevant to the query based on keyword overlap."""
        query_words = set(normalize_text(query).split())
        
        scored_topics = []
        for topic in self.cso_topics:
            topic_words = set(normalize_text(topic).split())
            overlap = len(query_words & topic_words)
            if overlap > 0:
                scored_topics.append((topic, overlap))
        
        # Sort by overlap score and return top N
        scored_topics.sort(key=lambda x: x[1], reverse=True)
        relevant = [t[0] for t in scored_topics[:limit]]
        
        # If not enough relevant topics, add some random ones
        if len(relevant) < limit:
            remaining = [t for t in self.cso_topics if t not in relevant]
            relevant.extend(remaining[:limit - len(relevant)])
        
        return relevant

    def create_paper_topic_relationships(self, paper_filename: str, validated_topics: List[str]) -> dict:
        if not validated_topics:
            print("  > No validated topics to link.")
            return {"linked": [], "missing": []}

        print(f"  > Creating HAS_TOPIC relationships for {len(validated_topics)} topics...")
        
        try:
            # Link topics that exist in database
            link_result = self.graph_service.graph.query("""
                MATCH (p:Paper {filename: $filename})
                WITH p
                UNWIND $topics AS topicLabel
                OPTIONAL MATCH (t:Topic {label: topicLabel})
                WITH p, topicLabel, t
                CALL {
                  WITH p, t
                  WITH p, t
                  WHERE t IS NOT NULL
                  MERGE (p)-[:HAS_TOPIC]->(t)
                  RETURN count(*) AS _c
                }
                RETURN collect({topic: topicLabel, exists: t IS NOT NULL}) AS status
            """, {"filename": paper_filename, "topics": validated_topics})

            status = link_result[0]["status"] if link_result else []
            linked = [s["topic"] for s in status if s["exists"]]
            missing = [s["topic"] for s in status if not s["exists"]]

            print(f"  > Linked topics: {linked}")
            if missing:
                print(f"  > Skipped (no matching Topic node): {missing}")

            # Verification
            verify = self.graph_service.graph.query("""
                MATCH (p:Paper {filename:$filename})-[:HAS_TOPIC]->(t:Topic)
                RETURN count(t) AS total, collect(t.label) AS labels
            """, {"filename": paper_filename})
            
            if verify:
                print(f"  > Verification: {verify[0]['total']} HAS_TOPIC relationships created")
            
            return {"linked": linked, "missing": missing}
            
        except Exception as e:
            print(f"  > Error creating relationships: {e}")
            import traceback
            traceback.print_exc()
            return {"linked": [], "missing": validated_topics}
