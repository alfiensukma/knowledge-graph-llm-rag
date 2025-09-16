from typing import Dict, Any

class AprioriService:
    def __init__(self, graph_service):
        self.graph_service = graph_service

    def _run_cypher(self, query: str, params: Dict[str, Any] = None):
        print(f"  > Executing Cypher query...")
        try:
            results = self.graph_service.graph.query(query, params)
            print("  > Query executed successfully.")
            return results
        except Exception as e:
            print(f"  > Cypher query failed: {e}")
            raise

    def create_frequent_itemsets(self, min_support_count: int = 2):
        print("\n--- Step 1: Creating Frequent Itemsets ---")
        query = """
        CALL gds.degree.stream('paperGraph')
        YIELD nodeId, score
        WHERE score >= $min_support_count
        WITH gds.util.asNode(nodeId).items AS items, toInteger(score) AS supportCount
        
        MERGE (f:FrequentTopicSet {items: items})
        SET f.supportCount = supportCount
        
        MERGE (sv:SupportValue {value: supportCount})
        MERGE (f)-[:HAS_SUPPORT_VALUE]->(sv)
        
        RETURN count(f) AS created_count
        """
        result = self._run_cypher(query, {"min_support_count": min_support_count})
        count = result[0]['created_count'] if result else 0
        print(f"  > Created or merged {count} FrequentTopicSet nodes.")

    def generate_association_rules(self, min_support_count):
        print("\n--- Step 2: Generating Association Rules ---")
        query = """
        // N = total transaction
        MATCH (p:Paper) WITH count(DISTINCT p) AS N

        // Enumerate A,B from each FrequentTopicSet
        MATCH (f:FrequentTopicSet)
        WITH N, apoc.coll.toSet(f.items) AS items
        WITH N, items, apoc.coll.combinations(items, 1, size(items)-1) AS subs
        UNWIND subs AS A
        WITH N, apoc.coll.sort(A) AS A,
            apoc.coll.sort(apoc.coll.subtract(items, A)) AS B
        WHERE size(A) > 0 AND size(B) > 0

        // (optional) DON'T use DISTINCT if want to "no deduplication across itemsets"
        // WITH DISTINCT N, A, B

        // Calculate support(A∪B)
        WITH N, A, B, apoc.coll.sort(apoc.coll.toSet(apoc.coll.union(A,B))) AS AB
        MATCH (pAB:Paper)
        WHERE ALL(x IN AB WHERE (pAB)-[:HAS_TOPIC]->(:Topic {label: x}))
        WITH N, A, B, count(DISTINCT pAB) AS cntAB
        WHERE cntAB >= $min_support_count  // min_support

        // Calculate support(A)
        MATCH (pA:Paper)
        WHERE ALL(x IN A WHERE (pA)-[:HAS_TOPIC]->(:Topic {label: x}))
        WITH N, A, B, cntAB, count(DISTINCT pA) AS cntA

        // Persist rules
        MERGE (l:LeftTopicSet  {items: A})
        MERGE (r:RightTopicSet {items: B})
        MERGE (l)-[rel:RULES]->(r)
        SET rel.support    = toFloat(cntAB)/N,
            rel.confidence = CASE WHEN cntA>0 THEN toFloat(cntAB)/cntA ELSE 0 END,
            rel.supportCount = cntAB

        RETURN count(rel) AS rules_created;
        """
        result = self._run_cypher(query, {"min_support_count": min_support_count})
        count = result[0]['rules_created'] if result else 0
        print(f"  > Created or merged {count} association rules.")

    def run_full_apriori_pipeline(self, min_support_count: int = 2):
        print("--- Starting GDS-based Apriori Pipeline ---")
        self.create_frequent_itemsets(min_support_count)
        self.generate_association_rules(min_support_count)
        print("\n--- GDS-based Apriori Pipeline Finished ---")

    def create_graph_projection(self):
        print("\n--- Step 0: Creating GDS Graph Projection 'paperGraph' ---")
        query = """
        CALL gds.graph.project(
            'paperGraph',
            ['Paper', 'TopicCombination'],
            {
                HAS_TOPIC_COMBINATION: {
                    orientation: 'REVERSE'
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        """
        try:
            self.graph_service.graph.query("CALL gds.graph.drop('paperGraph', false) YIELD graphName;")
            print("  > Dropped existing 'paperGraph' projection.")
        except Exception:
            print("  > No existing 'paperGraph' projection to drop. Continuing.")
        
        result = self._run_cypher(query)
        if result:
            print(f"  > Projection '{result[0]['graphName']}' created with {result[0]['nodeCount']} nodes and {result[0]['relationshipCount']} relationships.")

