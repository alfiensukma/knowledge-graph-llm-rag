from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.graphs import Neo4jGraph
import re

def normalize_text(text: str) -> str:
    text = re.sub(r'\s*\([^)]+\)\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()

class TopicExtractionService:
    def __init__(self, llm, graph_service):
        self.llm = llm
        self.graph_service = graph_service
        print("Fetching existing topics and hierarchy from Neo4j for validation...")
        self.cso_topics, self.hierarchy = self._fetch_topics_and_hierarchy_from_neo4j()
        print(f"-> Found {len(self.cso_topics)} topics and {len(self.hierarchy)} hierarchical relationships in the graph.")
        
        self.parser = JsonOutputParser(pydantic_object=List[str])
        self.validate_parser = JsonOutputParser(pydantic_object=dict)
        self.extract_prompt = ChatPromptTemplate.from_template(
            """Berdasarkan teks paper akademik berikut, identifikasi hingga **10 topik ilmiah utama** yang dibahas.
            Fokus pada konsep ilmiah spesifik dalam ilmu komputer, contohnya 'Content-Based Filtering', 'Information 
            Retrieval', 'Text Mining', atau 'Machine Learning'. Hindari topik umum seperti 'ilmu komputer' dan/atau 
            'computer science'. Kembalikan topik dalam bentuk daftar JSON berisi string. Teks: ```{text}```\n\nJSON Output: """
        )
        self.validate_prompt = ChatPromptTemplate.from_template(
            """Anda adalah ahli ontologi ilmu komputer. Validasi topik kandidat: ```{candidate}``` terhadap daftar topik Computer 
            Science Ontology CSO: ```{cso_topics}```, dengan hierarki (sub_topic -> super_topic): ```{hierarchy}```. 
            - Jika topik kandidat (setelah normalisasi: lowercase, tanpa spasi ekstra, tanpa tanda kurung) ada di CSO, 
              kembalikan topik asli dari database.
            - Jika tidak, cari topik CSO yang relevan secara semantik dalam konteks ilmu komputer atau aplikasinya (misalnya, MCDM 
              dipetakan ke 'decision support systems'). Contoh:
                - Kandidat: "encrypted data", CSO: ["data privacy", "security"], Hasil: {{"candidate": "encrypted data", 
                  "matched_topic": "data privacy"}}
                - Kandidat: "Multi-Criteria Decision-Making (MCDM)", CSO: ["decision support systems"], 
                  Hasil: {{"candidate": "Multi-Criteria Decision-Making (MCDM)", "matched_topic": "decision support systems"}}
            - Jangan gunakan 'computer science' sebagai kecocokan.
            - Jika tidak ada kecocokan semantik (skor < 90%), kembalikan 'None' dengan alasan.
            Output JSON: {{"candidate": "<candidate>", "matched_topic": "<matched_topic>", "reason": "<alasan jika None>"}}.
            """
        )
        self.extract_chain = self.extract_prompt | self.llm | self.parser
        self.validate_chain = self.validate_prompt | self.llm | self.validate_parser

    def _fetch_topics_and_hierarchy_from_neo4j(self) -> tuple:
        """Fetches all topic labels and hierarchy from Neo4j."""
        try:
            # take all topics, except "computer science"
            topic_results = self.graph_service.graph.query(
                "MATCH (t:Topic) WHERE t.label <> 'computer science' RETURN t.label AS label"
            )
            topics = [record['label'] for record in topic_results]
            if not topics:
                print("  > Warning: No topics found in Neo4j database!")

            # hierarchy
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

    def get_validated_topics_for_text(self, full_text: str) -> list:
        """Extracts candidate topics with LLM and validates them against CSO topics."""
        try:
            candidate_topics = self.extract_chain.invoke({"text": full_text})
            print(f"  > LLM candidate topics: {candidate_topics}")
        except Exception as e:
            print(f"  > LLM topic extraction failed: {e}")
            return []

        if not candidate_topics:
            print("  > No candidate topics extracted by LLM.")
            return []

        validated_topics = set()
        # normalize
        normalized_cso_topics = {t: normalize_text(t) for t in self.cso_topics}

        for candidate in candidate_topics:
            try:
                # Cek kecocokan langsung dengan normalisasi
                normalized_candidate = normalize_text(candidate)
                for original_topic, normalized_topic in normalized_cso_topics.items():
                    if normalized_candidate == normalized_topic:
                        print(f"  > Candidate '{candidate}' matches database topic '{original_topic}' after normalization.")
                        validated_topics.add(original_topic)
                        print(f"  > Validated topic: {candidate} -> {original_topic}")
                        break
                else:
                    # if no direct match, validate with LLM
                    validation_result = self.validate_chain.invoke({
                        "candidate": candidate,
                        "cso_topics": ", ".join(self.cso_topics),
                        "hierarchy": "; ".join(self.hierarchy)
                    })
                    print(f"  > Validation result: {validation_result}")
                    if validation_result["matched_topic"] != "None":
                        validated_topics.add(validation_result["matched_topic"])
                        print(f"  > Validated topic: {candidate} -> {validation_result['matched_topic']}")
                    else:
                        print(f"  > No match for candidate topic: {candidate}")
            except Exception as e:
                print(f"  > Error validating topic {candidate}: {e}")
        
        if not validated_topics:
            print(f"  > No validated topics found. Sample CSO topics: {self.cso_topics[:10]}")
        return list(validated_topics)