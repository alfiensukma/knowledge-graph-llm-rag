from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import re
import json

ITEM_MAP = {
    "analytic hierarchy process": "anhipr"
}

def _apply_item_mapping(items: List[str]) -> List[str]:
    mapped = []
    for x in items:
        x0 = _normalize_item(x)
        x1 = ITEM_MAP.get(x0, x0)
        mapped.append(x1)
    return sorted(set(mapped))

def _normalize_item(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s*\([^)]+\)\s*", " ", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def _canonicalize_items(items: List[str]) -> List[str]:
    return sorted({_normalize_item(i) for i in items if isinstance(i, str) and i.strip()})

class FrequentItemset(BaseModel):
    items: List[str] = Field(..., min_length=1,
        description="Daftar topik (itemset) yang dinilai sering muncul bersama.")
    support_count: int = Field(..., ge=0,
        description="Jumlah transaksi/paper yang mengandung seluruh items.")
    support: Optional[float] = Field(default=None, ge=0, le=1,
        description="Rasio support = support_count / total_papers.")

class AssociationRule(BaseModel):
    antecedent: List[str] = Field(..., min_length=1,
        description="Kiri (LHS) itemset.")
    consequent: List[str] = Field(..., min_length=1,
        description="Kanan (RHS) itemset.")
    support: float = Field(..., ge=0, le=1,
        description="Proporsi transaksi yang memuat seluruh item A∪B.")
    confidence: float = Field(..., ge=0, le=1,
        description="P(P(B|A)) = support(A∪B) / support(A).")

class CandidateReport(BaseModel):
    total: int = Field(..., ge=0)
    by_k: Dict[str, int] = Field(...,
        description="Jumlah kandidat per ukuran k; k sebagai string: {'1': 66, '2': 153, ...}")

class FrequentReport(BaseModel):
    total: int = Field(..., ge=0)
    by_k: Dict[str, int] = Field(...,
        description="Jumlah frequent set per ukuran k")

class RulesReport(BaseModel):
    total: int = Field(..., ge=0)
    unique_lhs: int = Field(..., ge=0)
    unique_rhs: int = Field(..., ge=0)

class AprioriReport(BaseModel):
    candidates: CandidateReport = Field(...)
    frequents: FrequentReport = Field(...)
    rules: RulesReport = Field(...)

class LLMAprioriOutput(BaseModel):
    frequent_itemsets: List[FrequentItemset] = Field(...,
        description="Daftar frequent itemset (ukuran >=1) yang lolos min_support.")
    rules: List[AssociationRule] = Field(...,
        description="Daftar aturan asosiasi (A -> B) yang dibangkitkan dari frequent itemset berukuran >=2.")
    report: AprioriReport = Field(...,
        description="Ringkasan total kandidat/frequent/rules (termasuk rincian per k).")

class LLMAprioriService:
    def __init__(self, llm, graph_service):
        self.llm = llm
        self.graph_service = graph_service

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
                """Anda mensimulasikan Apriori klasik stepwise secara deterministik di atas transaksi topik paper.

                Aturan keras (ikuti persis, tanpa improvisasi):
                - Item = string persis dari input setelah lowercase+trim dan hapus teks dalam tanda kurung (...).
                DILARANG menciptakan item baru atau menggabungkan sinonim. Gunakan hanya item yang muncul di transaksi.
                - Gunakan join–prune standar Apriori:
                • C1 = semua item unik dari transaksi.
                • L1 = item dengan support_count ≥ min_support_count (absolut).
                • Untuk k ≥ 2: bangkitkan Ck dari join Lk−1 saja (bukan semua item global);
                    prune kandidat yang punya subset tidak frequent; lalu Lk = kandidat dengan support_count ≥ min_support_count.
                • Berhenti saat Lk kosong.
                - Aturan asosiasi: dari setiap frequent itemset berukuran ≥2, enumerasi SEMUA subset non-kosong A ⊂ S dan B = S \\ A; laporkan 
                  SEMUA aturan A→B. Jangan deduplikasi lintas itemset.
                - Definisi:
                • support_count(S) = jumlah transaksi yang memuat seluruh item S
                • support(S) = support_count(S) / total_papers
                • confidence(A→B) = support(A∪B) / support(A)
                - Urutkan setiap daftar item di dalam itemset/rule secara leksikografis ascending.
                - Angka float laporkan maksimal 6 desimal.
                - Kembalikan HANYA JSON valid sesuai skema Pydantic yang diberikan. Jangan sertakan narasi/kode.

                Wajib laporkan secara konsisten:
                - Total kandidat Ck dan rincian per ukuran k (by_k).
                - Total frequent Lk dan rincian per ukuran k (by_k).
                - Jumlah aturan: total, jumlah LHS unik, jumlah RHS unik.

                Lakukan self-check internal sebelum menjawab agar semua metrik konsisten dengan definisi di atas.
            """),
            ("human",
                """Diketahui transaksi dari {total_papers} paper (JSON): {transactions}

                Parameter:
                - total_papers={total_papers}
                - min_support_count={min_support_count}
                - min_confidence={min_confidence}

                KELUARAN (JSON-only):
                {{
                    "frequent_itemsets": [
                        {{"items": ["item1","item2",...], "support_count": <int>, "support": <float>}}
                    ],
                    "rules": [
                        {{"antecedent": ["..."], "consequent": ["..."], "support": <float>, "confidence": <float>}}
                    ],
                    "report": {{
                        "candidates": {{"total": <int>, "by_k": {{"1": <int>, "2": <int>, "3": <int>, ...}}}},
                        "frequents":  {{"total": <int>, "by_k": {{"1": <int>, "2": <int>, "3": <int>, ...}}}},
                        "rules":      {{"total": <int>, "unique_lhs": <int>, "unique_rhs": <int>}}
                    }}
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
            topics = _apply_item_mapping(topics)
            if topics:
                tx.append({"paper_id": r["id"], "topics": topics})
        print(f"  > Loaded {len(tx)} transactions from Neo4j.")
        return tx
    
    def _run_llm_apriori(self,
                         transactions: List[Dict[str, Any]],
                         min_support_count: int,
                         min_confidence: float) -> LLMAprioriOutput:
        total_papers = len(transactions)
        print("  > Sending transactions to LLM for Apriori-like mining...")
        raw = self.chain.invoke({
            "transactions": transactions,
            "total_papers": total_papers,
            "min_support_count": min_support_count,
            "min_confidence": min_confidence
        })
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
        data = sorted(itemsets, key=lambda it: (it.support_count, -len(it.items), it.items))
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
                "lhs": lhs,
                "rhs": rhs,
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
        
    def _tx_to_sets(self, transactions: List[Dict[str, Any]]) -> List[set]:
        # convert list of topics to set for subset checks
        tx_sets = []
        for tx in transactions:
            topics = tx.get("topics") or []
            topics = _apply_item_mapping(topics)
            tx_sets.append(set(topics))
        return tx_sets
    
    def _support_count_for(self, items: List[str], tx_sets: List[set]) -> int:
        S = set(_apply_item_mapping(items))
        if not S:
            return 0
        return sum(1 for t in tx_sets if S.issubset(t))

    def _validate_and_correct_output(self,
                                     output: LLMAprioriOutput,
                                     transactions: List[Dict[str, Any]],
                                     min_support_count: int) -> LLMAprioriOutput:
        # recompute support/support_count/confidence from ground truth
        tx_sets = self._tx_to_sets(transactions)
        total_papers = len(tx_sets)

        fixed_itemsets: List[FrequentItemset] = []
        for it in output.frequent_itemsets:
            items = _canonicalize_items(it.items)
            sc = self._support_count_for(items, tx_sets)
            if sc >= min_support_count:
                sp = round(sc / total_papers, 6)
                fixed_itemsets.append(FrequentItemset(items=items, support_count=sc, support=sp))
        fixed_itemsets.sort(key=lambda x: (len(x.items), x.items))

        fixed_rules: List[AssociationRule] = []
        for r in output.rules:
            lhs = _canonicalize_items(r.antecedent)
            rhs = _canonicalize_items(r.consequent)
            if not lhs or not rhs:
                continue
            sc_union = self._support_count_for(sorted(set(lhs) | set(rhs)), tx_sets)
            sc_lhs = self._support_count_for(lhs, tx_sets)
            sp = round(sc_union / total_papers, 6)
            cf = round((sc_union / sc_lhs) if sc_lhs > 0 else 0.0, 6)
            fixed_rules.append(AssociationRule(antecedent=lhs, consequent=rhs, support=sp, confidence=cf))

        by_k_freq: Dict[str, int] = {}
        for it in fixed_itemsets:
            k = str(len(it.items))
            by_k_freq[k] = by_k_freq.get(k, 0) + 1
        frequents_report = FrequentReport(total=sum(by_k_freq.values()), by_k=by_k_freq)

        unique_lhs = {tuple(r.antecedent) for r in fixed_rules}
        unique_rhs = {tuple(r.consequent) for r in fixed_rules}
        rules_report = RulesReport(total=len(fixed_rules), unique_lhs=len(unique_lhs), unique_rhs=len(unique_rhs))

        candidates_report = output.report.candidates if output.report else CandidateReport(total=0, by_k={})
        fixed_report = AprioriReport(candidates=candidates_report, frequents=frequents_report, rules=rules_report)

        return LLMAprioriOutput(frequent_itemsets=fixed_itemsets, rules=fixed_rules, report=fixed_report)

    def build_llm_apriori_graph(self,
                                min_support_count: int,
                                min_confidence: float):
        try:
            transactions = self._fetch_transactions()
            if not transactions:
                print("  > No transactions available in the database.")
                return {"transactions": 0, "itemsets": 0, "rules": 0}

            output: LLMAprioriOutput = self._run_llm_apriori(
                transactions=transactions,
                min_support_count=min_support_count,
                min_confidence=min_confidence
            )

            # validate using real transactions
            output = self._validate_and_correct_output(
                output=output,
                transactions=transactions,
                min_support_count=min_support_count
            )

            self._print_step2_frequent_itemsets(output.frequent_itemsets)
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