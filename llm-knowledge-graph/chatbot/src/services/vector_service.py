from langchain_neo4j import Neo4jVector
from config.settings import Settings

class VectorService:
    def __init__(self, graph_service, llm_service):
        self.graph_service = graph_service
        self.llm_service = llm_service
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> None:
        """Initialize Neo4j vector index."""
        try:
            self.vector_store = Neo4jVector.from_existing_index(
                embedding=self.llm_service.embedding,
                graph=self.graph_service.graph,
                index_name="chunkVector",
                embedding_node_property="textEmbedding",
                text_node_property="text",
                retrieval_query="""
                MATCH (node:Chunk)-[:PART_OF]->(p:Paper)
                OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
                OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
                OPTIONAL MATCH (p)-[:HAS_RESULT]->(r:Result)
                OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
                WITH node, score, p, 
                     collect(DISTINCT m.name) AS methods,
                     collect(DISTINCT d.name) AS datasets,
                     collect(DISTINCT r.description) AS results,
                     collect(DISTINCT k.term) AS keywords
                RETURN node.text AS text, score,
                       {paper_id: p.id, paper_title: p.title, chunk_id: node.id, section: node.section,
                        methods: methods, datasets: datasets, results: results, keywords: keywords} AS metadata
                ORDER BY score DESC
                LIMIT 5
                """
            )
            print("  > Vector store initialized successfully.")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> list:
        """Search for chunks similar to the query."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [
                {
                    "text": result[0].page_content,
                    "score": result[1],
                    "metadata": {
                        "paper_id": result[0].metadata.get("paper_id", ""),
                        "paper_title": result[0].metadata.get("paper_title", ""),
                        "chunk_id": result[0].metadata.get("chunk_id", ""),
                        "section": result[0].metadata.get("section", ""),
                        "methods": result[0].metadata.get("methods", []),
                        "datasets": result[0].metadata.get("datasets", []),
                        "results": result[0].metadata.get("results", []),
                        "keywords": result[0].metadata.get("keywords", [])
                    }
                } for result in results
            ]
        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []