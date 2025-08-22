from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import re
import json

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

class LLMAprioriService:
    def __init__(self, llm, graph_service):
        self.llm = llm
        self.graph_service = graph_service

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
            """Anda adalah analis data yang menjalankan algoritma seperti Apriori di atas daftar transaksi topik paper.
            Instruksi:
            - Input adalah daftar transaksi: [{{"paper_id": "...", "topics": ["topic a", "topic b", ...]}}, ...].
            - Hitung frequent itemsets (ukuran 1..max_k) berbasis support_count minimal (min_support_count).
            - Pastikan semua kombinasi topik dari transaksi input dipertimbangkan, termasuk topik seperti 'large language model' dan 'object-oriented programming'.
            - Normalisasi topik serupa (misalnya, 'large language models' = 'large language model') menggunakan lowercase dan hapus duplikasi.
            - Hasilkan aturan asosiasi A->B dengan confidence >= min_confidence. Hindari aturan trivial (antecedent atau consequent kosong).
            - Jika ragu pada rasio support, tetap isi support_count dengan benar (jumlah paper yang memuat seluruh item) dan support = support_count/total_papers.
            - Kembalikan JSON sesuai skema.
            """),
            ("human",
            """Transaksi paper (JSON): {transactions}
            Parameter:
            - total_papers={total_papers}
            - min_support_count={min_support_count}
            - min_confidence={min_confidence}
            - max_itemset_size={max_itemset_size}

            Kembalikan JSON dengan format:
            {{
            "frequent_itemsets": [{{"items": [...], "support_count": 3, "support": 0.2}}, ...],
            "rules": [{{"antecedent": [...], "consequent": [...], "support": 0.15, "confidence": 0.7}}, ...]
            }}
            """)
        ])
        self.parser = JsonOutputParser(pydantic_object=LLMAprioriOutput)
        self.chain = self.prompt | self.llm | self.parser

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
    
    def _run_llm_apriori(self,
                         transactions: List[Dict[str, Any]],
                         min_support_count: int,
                         min_confidence: float,
                         max_itemset_size: int) -> LLMAprioriOutput:
        total_papers = len(transactions)
        print("  > Sending transactions to LLM for Apriori-like mining...")

        raw = self.chain.invoke({
            "transactions": transactions,
            "total_papers": total_papers,
            "min_support_count": min_support_count,
            "min_confidence": min_confidence,
            "max_itemset_size": max_itemset_size
        })

        # Normalisasi output ke Pydantic model
        try:
            if isinstance(raw, LLMAprioriOutput):
                return raw
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict):
                return LLMAprioriOutput.model_validate(raw)
            raise TypeError(f"Unexpected LLM output type: {type(raw)}")
        except Exception as e:
            print(f"  > Failed to parse LLM output into LLMAprioriOutput: {e}")
            raise

    def _print_step2_frequent_itemsets(self, itemsets: List[FrequentItemset]):
        if not itemsets:
            print("[Step2] No frequent itemsets.")
            return
        data = sorted(
            itemsets,
            key=lambda it: (it.support_count, -len(it.items), it.items)
        )
        for it in data:
            print(f"[Step2] itemset={it.items}, paperCount={it.support_count}, length={len(it.items)}")

    def _print_step3_candidate_rules(self, itemsets: List[FrequentItemset], min_support_count: int = 2):
        filt = [it for it in itemsets if it.support_count >= min_support_count]
        if not filt:
            print(f"[Step3] No itemsets with support_count >= {min_support_count}.")
            return

        max_len = max(len(it.items) for it in filt)
        max_sets = [it for it in filt if len(it.items) == max_len]
        if not max_sets or max_len < 2:
            print(f"[Step3] No candidate rules from max n-itemset (max_len={max_len}).")
            return

        # generate semua subset A -> B untuk tiap base itemset terbesar
        from itertools import combinations
        for it in max_sets:
            items = it.items
            for r in range(1, len(items)):
                for A in combinations(items, r):
                    A = list(A)
                    B = [x for x in items if x not in A]
                    if not B:
                        continue
                    print(f"[Step3] base_itemset={items}, antecedent={A}, consequent={B}")

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

    def build_llm_apriori_graph(self,
                                min_support_count: int,
                                min_confidence: float,
                                max_itemset_size: int) -> Optional[Dict[str, Any]]:
        try:
            transactions = self._fetch_transactions()
            if not transactions:
                print("  > No transactions available in the database.")
                return {"transactions": 0, "itemsets": 0, "rules": 0}

            output: LLMAprioriOutput = self._run_llm_apriori(
                transactions=transactions,
                min_support_count=min_support_count,
                min_confidence=min_confidence,
                max_itemset_size=max_itemset_size
            )

            self._print_step2_frequent_itemsets(output.frequent_itemsets)
            self._print_step3_candidate_rules(output.frequent_itemsets, min_support_count=min_support_count)

            # Persist hasil LLM
            self._persist_frequent_itemsets(output.frequent_itemsets)
            self._persist_rules(output.rules)

            summary = {
                "transactions": len(transactions),
                "itemsets": len(output.frequent_itemsets),
                "rules": len(output.rules)
            }
            print(f"  > LLM Apriori summary: {summary}")
            return summary

        except Exception as e:
            print(f"  > Failed to build LLM Apriori graph: {e}")
            return None
