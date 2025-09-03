from typing import List, Dict, Any, Optional, Tuple, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import re
import json
from itertools import combinations, chain
from collections import defaultdict

def _normalize_item(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s*\([^)]+\)\s*", " ", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def _canonicalize_items(items: List[str]) -> List[str]:
    return sorted({_normalize_item(i) for i in items if isinstance(i, str) and i.strip()})

class FrequentItemset(BaseModel):
    items: List[str] = Field(description="Daftar topik (itemset) yang dinilai sering muncul bersama.")
    support_count: int = Field(description="Jumlah paper yang mengandung seluruh items.")
    support: Optional[float] = Field(default=None, description="Rasio support = support_count / total_papers.")

class AssociationRule(BaseModel):
    antecedent: List[str] = Field(description="Kiri (LHS) itemset.")
    consequent: List[str] = Field(description="Kanan (RHS) itemset.")
    support: float = Field(description="Support rule adalah proporsi semua paper yang mengandung seluruh item baik di sisi kiri maupun sisi kanan aturan, dibagi dengan total jumlah paper")
    confidence: float = Field(description="Confidence adalah persentase paper yang memuat seluruh item di sisi kanan aturan (consequent) di antara semua paper yang sudah memuat seluruh item di sisi kiri aturan (antecedent)")

class LLMAprioriOutput(BaseModel):
    frequent_itemsets: List[FrequentItemset] = Field(description="Himpunan item-set yang dianggap sering.")
    rules: List[AssociationRule] = Field(description="Aturan asosiasi (A -> B) yang relevan.")
    
class SupportCountResult(BaseModel):
    items: List[str]
    support_count: int

class LLMSupportCountOutput(BaseModel):
    counts: List[SupportCountResult]

class LLMAprioriService:
    def __init__(self, llm, graph_service):
        self.llm = llm
        self.graph_service = graph_service

        self.support_counter_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """Anda adalah asisten penghitung yang sangat akurat. Tugas Anda SATU: menghitung berapa kali setiap 
                'candidate_itemset' muncul dalam daftar 'transactions'.
             - Sebuah itemset dianggap muncul dalam sebuah transaksi jika SEMUA item di dalamnya adalah subset dari 
                topik transaksi tersebut.
             - Normalisasi item tidak diperlukan, data sudah bersih.
             - Kembalikan HANYA objek JSON yang valid dengan daftar hasil.
             - Jika sebuah kandidat tidak pernah muncul, support_count-nya adalah 0.
             """),
            ("human",
             """Berikut adalah data dan tugasnya:
             Transactions:
             ```json
             {transactions}
             ```
             Candidate Itemsets to Count:
             ```json
             {candidate_itemsets}
             ```
             Kembalikan JSON dengan format: {{"counts": [{{"items": ["item1", "item2"], "support_count": 5}}, ...]}}
             """)
        ])
        
        self.support_counter_parser = JsonOutputParser(pydantic_object=LLMSupportCountOutput)
        self.support_counter_chain = self.support_counter_prompt | self.llm | self.support_counter_parser
        
    def _fetch_transactions(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
        RETURN p.id AS id, collect(DISTINCT t.label) AS topics
        """
        rows = self.graph_service.graph.query(query)
        tx = []
        for r in rows:
            topics = [t for t in r["topics"] if isinstance(t, str)]
            topics = _canonicalize_items(topics)
            if topics:
                tx.append({"paper_id": r["id"], "topics": topics})
        print(f"  > Loaded {len(tx)} transactions from Neo4j.")
        return tx
    
    def _generate_candidates(self, Lk_minus_1: Set[frozenset], k: int) -> Set[frozenset]:
        candidates = set()
        for i1 in Lk_minus_1:
            for i2 in Lk_minus_1:
                if len(i1.union(i2)) == k:
                    candidates.add(i1.union(i2))
        return candidates

    def _run_hybrid_apriori_loop(self,
                                 transactions: List[Dict[str, Any]],
                                 min_support_count: int) -> List[FrequentItemset]:
        
        all_frequent_itemsets = []
        
        item_counts = defaultdict(int)
        for tx in transactions:
            for item in tx['topics']:
                item_counts[item] += 1
        
        L1_sets = set()
        for item, count in item_counts.items():
            if count >= min_support_count:
                L1_sets.add(frozenset([item]))
                all_frequent_itemsets.append(FrequentItemset(items=[item], support_count=count))
        
        print(f"  > [Python] Found {len(L1_sets)} frequent 1-itemsets (L1).")

        k = 2
        Lk_minus_1 = L1_sets
        while Lk_minus_1:
            Ck = self._generate_candidates(Lk_minus_1, k)
            if not Ck:
                break
            
            print(f"  > [Python] Generated {len(Ck)} candidates for C{k}.")
            
            candidate_list_for_llm = [sorted(list(c)) for c in Ck]
            print(f"  > [LLM] Sending {len(candidate_list_for_llm)} candidates to LLM for support counting...")
            
            llm_output = self.support_counter_chain.invoke({
                "transactions": json.dumps(transactions, indent=2),
                "candidate_itemsets": json.dumps(candidate_list_for_llm, indent=2)
            })
            
            support_counts_map = {frozenset(res['items']): res['support_count'] for res in llm_output['counts']}

            Lk = set()
            for candidate in Ck:
                count = support_counts_map.get(candidate, 0)
                if count >= min_support_count:
                    Lk.add(candidate)
                    all_frequent_itemsets.append(FrequentItemset(items=sorted(list(candidate)), support_count=count))
            
            print(f"  > [Python] Filtered to {len(Lk)} frequent {k}-itemsets (L{k}).")

            if not Lk:
                break
            
            Lk_minus_1 = Lk
            k += 1
            
        return all_frequent_itemsets

    def _generate_rules(self, 
                        all_frequent_itemsets: List[FrequentItemset], 
                        min_confidence: float) -> List[AssociationRule]:
        
        print("  > [Python] Generating association rules...")
        itemset_support_map = {frozenset(it.items): it.support_count for it in all_frequent_itemsets}
        rules = []

        for itemset in all_frequent_itemsets:
            if len(itemset.items) < 2:
                continue
            
            all_subsets = chain.from_iterable(combinations(itemset.items, r) for r in range(1, len(itemset.items)))
            
            for antecedent_tuple in all_subsets:
                antecedent = frozenset(antecedent_tuple)
                consequent = frozenset(itemset.items) - antecedent
                
                if not antecedent or not consequent:
                    continue
                
                support_itemset = itemset_support_map.get(frozenset(itemset.items), 0)
                support_antecedent = itemset_support_map.get(antecedent, 0)
                
                if support_antecedent == 0:
                    continue
                    
                confidence = support_itemset / support_antecedent
                
                if confidence >= min_confidence:
                    total_papers = len(self._fetch_transactions()) # Re-fetch for total count
                    rules.append(AssociationRule(
                        antecedent=sorted(list(antecedent)),
                        consequent=sorted(list(consequent)),
                        support=support_itemset / total_papers,
                        confidence=confidence
                    ))
        print(f"  > [Python] Found {len(rules)} rules meeting min_confidence.")
        return rules

    def build_llm_apriori_graph(self,
                                min_support_count: int,
                                min_confidence: float) -> Optional[Dict[str, Any]]:
        try:
            transactions = self._fetch_transactions()
            if not transactions:
                return {"transactions": 0, "itemsets": 0, "rules": 0}

            all_frequent_itemsets = self._run_hybrid_apriori_loop(transactions, min_support_count)
            
            total_papers = len(transactions)
            for it in all_frequent_itemsets:
                it.support = it.support_count / total_papers

            all_rules = self._generate_rules(all_frequent_itemsets, min_confidence)

            self._persist_frequent_itemsets(all_frequent_itemsets)
            self._persist_rules(all_rules)

            summary = {
                "transactions": len(transactions),
                "itemsets": len(all_frequent_itemsets),
                "rules": len(all_rules)
            }
            print(f"  > Hybrid Apriori summary: {summary}")
            return summary

        except Exception as e:
            print(f"  > Failed to build Hybrid Apriori graph: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _persist_frequent_itemsets(self, itemsets: List[FrequentItemset]):
        payload = []
        for it in itemsets:
            items = _canonicalize_items(it.items)
            if not items:
                continue
            payload.append({
                "items": items,
                "support_count": int(it.support_count),
                "support": float(it.support) if it.support is not None else None
            })

        if not payload:
            print("  > No frequent itemsets to persist.")
            return

        cypher = """
        UNWIND $itemsets AS row
        MERGE (f:FrequentTopicSet {items: row.items})
        SET f.support_count = row.support_count,
            f.support = coalesce(row.support, f.support)
        """
        self.graph_service.graph.query(cypher, {"itemsets": payload})
        print(f"  > Persisted {len(payload)} FrequentTopicSet nodes.")

    def _persist_rules(self, rules: List[AssociationRule]):
        payload = []
        for r in rules:
            lhs = _canonicalize_items(r.antecedent)
            rhs = _canonicalize_items(r.consequent)
            if not lhs or not rhs:
                continue
            payload.append({
                "lhs": lhs, "rhs": rhs,
                "support": float(r.support),
                "confidence": float(r.confidence)
            })

        if not payload:
            print("  > No association rules to persist.")
            return

        cypher = """
        UNWIND $rules AS row
        MERGE (l:LeftTopicSet {items: row.lhs})
        MERGE (r:RightTopicSet {items: row.rhs})
        MERGE (l)-[rel:RULES]->(r)
        SET rel.support = row.support,
            rel.confidence = row.confidence
        """
        self.graph_service.graph.query(cypher, {"rules": payload})
        print(f"  > Persisted {len(payload)} rules (LeftTopicSet)-[:RULES]->(RightTopicSet).")