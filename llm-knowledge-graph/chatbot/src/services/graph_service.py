from langchain_neo4j import Neo4jGraph
from config.settings import Settings

class GraphService:
    """Service for managing Neo4j graph operations."""
    
    def __init__(self):
        self.graph = Neo4jGraph(
            url=Settings.NEO4J_URI,
            username=Settings.NEO4J_USERNAME,
            password=Settings.NEO4J_PASSWORD,
            database=Settings.DATABASE_NAME
        )
    
    def run_query(self, query: str, params: dict = None) -> list:
        """Execute a Cypher query and return results."""
        try:
            return self.graph.query(query, params or {})
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def get_paper_metadata(self, paper_id: str) -> dict:
        """Get metadata for a paper."""
        query = """
        MATCH (p:Paper {id: $paper_id})
        OPTIONAL MATCH (p)-[:WROTE]->(a:Author)
        OPTIONAL MATCH (p)-[:CITED_BY]->(r:Reference)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
        RETURN p.id, p.title, p.filename, p.abstract,
               collect(DISTINCT a.name) AS authors,
               collect(DISTINCT r.title) AS references,
               j.name AS journal,
               collect(DISTINCT t.label) AS topics
        """
        result = self.run_query(query, {"paper_id": paper_id})
        return result[0] if result else {}