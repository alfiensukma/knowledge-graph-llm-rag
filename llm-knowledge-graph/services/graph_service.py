from langchain_neo4j import Neo4jGraph
import uuid

class GraphService:
    def __init__(self, url, username, password):
        self.graph = Neo4jGraph(url=url, username=username, password=password)
        print("GraphService connected to Neo4j.")

    def import_paper_graph(self, paper_data: dict, filename: str):
        paper_uuid = str(uuid.uuid4())
        venue = paper_data.get('venue')
        if isinstance(venue, dict):
            venue = venue.get('name', venue.get('title', str(venue)))
        if not isinstance(venue, str):
            venue = str(venue) if venue else "Unknown Journal or Conference"

        import_query = """
        MERGE (p:Paper {id: $uuid})
        SET p.title = coalesce($paper.title, 'Untitled Paper'),
            p.abstract = coalesce($paper.abstract, 'No abstract provided'),
            p.publisher = $paper.publisher,
            p.venue = $paper.venue,
            p.publicationDate = $paper.publication_date,
            p.filename = $filename,
            p.topics = $paper.topics
        WITH p, $paper.venue AS venueName
        FOREACH (_ IN CASE WHEN venueName IS NOT NULL THEN [1] ELSE [] END |
            MERGE (j:Journal {id: apoc.text.slug(venueName, '_')})
            ON CREATE SET j.name = venueName
            MERGE (p)-[:PUBLISHED_IN]->(j)
        )

        WITH p
        UNWIND $paper.authors AS author_data
        MERGE (a:Author {id: coalesce(author_data.email, author_data.name)})
        ON CREATE SET a.name = author_data.name, a.email = author_data.email
        MERGE (a)-[:AUTHORED]->(p)
        WITH p, collect(a) AS authors 

        WITH p, authors 
        UNWIND $paper.references AS ref_data
        WITH p, authors, ref_data
        WHERE ref_data.title IS NOT NULL
        MERGE (r:Reference {id: coalesce(ref_data.doi, ref_data.title)})
        ON CREATE SET r.title = ref_data.title, r.doi = ref_data.doi
        MERGE (p)-[:CITES]->(r)

        WITH authors
        UNWIND authors AS a1
        UNWIND authors AS a2
        WITH a1, a2
        WHERE a1.id < a2.id
        MERGE (a1)-[:CO_AUTHOR]-(a2)
        """
        
        params = {
            "uuid": paper_uuid,
            "filename": filename,
            "paper": {
                "title": paper_data.get('title', 'Untitled Paper'),
                "abstract": paper_data.get('abstract', 'No abstract provided'),
                "publisher": paper_data.get('publisher', None),
                "venue": venue,
                "publication_date": paper_data.get('publication_date', None),
                "authors": paper_data.get('authors', []),
                "references": paper_data.get('references', []),
                "topics": paper_data.get('topics', [])
            }
        }
        try:
            self.graph.query(import_query, params)
            print(f"  > Successfully imported graph for paper: {filename} with UUID: {paper_uuid}")
            return paper_uuid
        except Exception as e:
            print(f"  > Failed to import graph for {filename}. Error: {e}")
            return None

    def link_paper_to_topics(self, paper_uuid: str, topic_labels: list):
        """Links the Paper node to its validated Topic nodes using UUID and updates topics property."""
        if not topic_labels:
            print(f"  > No topics to link for paper with UUID: {paper_uuid}")
            return
        
        try:
            existing_topics = self.graph.query(
                """
                MATCH (t:Topic)
                WHERE t.label IN $labels
                RETURN t.label AS label
                """,
                {"labels": topic_labels}
            )
            existing_labels = [record['label'] for record in existing_topics]
            print(f"  > Found {len(existing_labels)} existing topics in Neo4j: {existing_labels}")

            if existing_labels:
                self.graph.query(
                    """
                    MATCH (p:Paper {id: $uuid})
                    SET p.topics = $labels
                    WITH p
                    UNWIND $labels AS topic_label
                    MATCH (t:Topic {label: topic_label})
                    MERGE (p)-[:HAS_TOPIC]->(t)
                    """,
                    {"uuid": paper_uuid, "labels": existing_labels}
                )
                print(f"  > Linked paper with UUID {paper_uuid} to {len(existing_labels)} topics and updated topics property: {existing_labels}")
            else:
                print(f"  > No matching topics found in Neo4j for paper with UUID {paper_uuid}.")
        except Exception as e:
            print(f"  > Failed to link topics for paper with UUID {paper_uuid}. Error: {e}")