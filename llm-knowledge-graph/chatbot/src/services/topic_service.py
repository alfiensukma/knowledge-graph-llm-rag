from typing import List
from .graph_service import GraphService
from .llm_service import LLMService

class TopicService:
    """Service for validating topics against existing CSO data."""
    
    def __init__(self, graph_service: GraphService, llm_service: LLMService):
        self.graph_service = graph_service
        self.llm_service = llm_service
        self.cso_topics, self.hierarchy = self._fetch_cso_data()
    
    def _fetch_cso_data(self) -> tuple:
        """Fetch CSO topics and hierarchy from Neo4j."""
        try:
            topic_results = self.graph_service.run_query("MATCH (t:Topic) RETURN t.label AS label")
            topics = [record["label"].lower() for record in topic_results]
            hierarchy_results = self.graph_service.run_query(
                "MATCH (sub:Topic)-[:SUB_TOPIC_OF]->(super:Topic) RETURN sub.label AS sub_topic, super.label AS super_topic"
            )
            hierarchy = [f"{record['sub_topic']} -> {record['super_topic']}" for record in hierarchy_results]
            return topics, hierarchy
        except Exception as e:
            print(f"Error fetching CSO data: {e}")
            return [], []
    
    def get_validated_topics(self, text: str) -> List[str]:
        """Extract and validate topics from text."""
        try:
            candidate_topics = self.llm_service.extract_topics(text)
            validated_topics = set()
            for candidate in candidate_topics:
                result = self.llm_service.validate_topic(candidate, self.cso_topics, self.hierarchy)
                if result["matched_topic"] != "None":
                    validated_topics.add(result["matched_topic"])
            return list(validated_topics)[:5]
        except Exception as e:
            print(f"Error validating topics: {e}")
            return []