from typing import List, Dict, Any, Optional, Tuple, Set
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


class ComboResult(BaseModel):
    paper_id: str = Field(description="ID paper yang diproses.")
    combos: List[List[str]] = Field(description="Daftar kombinasi (1..max_k) dari topik paper tersebut.")


class LLMCombinationService:

    def __init__(self, llm, graph_service):
        self.llm = llm
        self.graph_service = graph_service

        self.combo_prompt = ChatPromptTemplate.from_messages([
            ("system",
                """Anda adalah asisten yang tepat dan lengkap dalam menghasilkan semua kombinasi dari daftar topik.
                Instruksi:
                - Input: paper_id, topics (list string), max_k (batas ukuran kombinasi).
                - Kembalikan semua kombinasi ukuran 1..max_k (tidak boleh ada yang terlewat).
                - Jangan menambahkan item di luar 'topics'.
                - Jangan ada duplikasi kombinasi.
                - Jawab hanya dengan JSON yang valid.
                Contoh format:
                {{"paper_id":"<id>", "combos":[["a"],["b"],["a","b"]]}}
            """),
            ("human",
                """paper_id={paper_id}
                topics={topics}
                max_k={max_k}
                Kembalikan JSON persis dengan format:
                {{"paper_id":"{paper_id}", "combos":[["item1"],["item2"],["item1","item2"], ...]}}
            """)
        ])

        self.combo_parser = JsonOutputParser(pydantic_object=ComboResult)
        self.combo_chain = self.combo_prompt | self.llm | self.combo_parser

    def _fetch_topics_for_paper(self, paper_id: str) -> List[str]:
        rows = self.graph_service.graph.query(
            """
            MATCH (p:Paper {id: $pid})-[:HAS_TOPIC]->(t:Topic)
            RETURN collect(DISTINCT t.label) AS topics
            """,
            {"pid": paper_id}
        )
        topics = rows[0]["topics"] if rows else []
        return _canonicalize_items([t for t in topics if isinstance(t, str)])

    def _validate_and_canonicalize_combos(
        self,
        paper_id: str,
        topics: List[str],
        combos: List[List[str]],
        max_k: int
    ) -> List[List[str]]:
        topic_set: Set[str] = set(topics)
        cleaned: Set[Tuple[str, ...]] = set()

        for combo in combos or []:
            if not isinstance(combo, list) or not combo:
                continue
            canon = _canonicalize_items(combo)
            if not canon:
                continue
            if not set(canon).issubset(topic_set):
                continue
            if len(canon) < 1 or len(canon) > max_k:
                continue
            cleaned.add(tuple(canon))

        return [list(t) for t in sorted(cleaned)]

    def _persist_combos_for_paper(self, paper_id: str, combos: List[List[str]], topics: List[str]) -> None:
        topic_count = len(topics)
        combo_count = len(combos or [])
        if not combos:
            print(f"[Step1] paperId={paper_id}, topicCount={topic_count}, combinationCount={combo_count}\n"
                f"         topics={topics}\n"
                f"         combinations=[]")
            return

        self.graph_service.graph.query(
            """
            UNWIND $rows AS row
            MATCH (p:Paper {id: row.paper_id})
            UNWIND row.combos AS combo
            WITH p, combo
            MERGE (c:TopicCombination {items: combo})
            MERGE (p)-[:HAS_TOPIC_COMBINATION]->(c)
            """,
            {"rows": [{"paper_id": paper_id, "combos": combos}]}
        )
        print(f"[Step1] paperId={paper_id}, topicCount={topic_count}, combinationCount={combo_count}\n"
            f"         topics={topics}\n"
            f"         combinations={combos}")

    def generate_combinations_for_paper(
        self,
        paper_id: str,
        max_k: Optional[int] = None,
        repair_missing: bool = False  # kalau True, isi kekurangan (jika LLM miss) pakai kombinasi Python (opsional)
    ) -> Optional[List[List[str]]]:
        topics = self._fetch_topics_for_paper(paper_id)
        if not topics:
            print(f"[Step1] paperId={paper_id}\n         combinations=[]  (No topics found)")
            return None

        k = min(len(topics), max_k) if max_k else len(topics)

        #LLM
        raw = self.combo_chain.invoke({
            "paper_id": paper_id,
            "topics": topics,
            "max_k": k
        })

        if isinstance(raw, ComboResult):
            llm_combos = raw.combos
        elif isinstance(raw, dict):
            llm_combos = raw.get("combos", [])
        else:
            try:
                data = json.loads(raw)
                llm_combos = data.get("combos", [])
            except Exception:
                print(f"  > Unexpected LLM output for {paper_id}: {type(raw)}")
                return None

        # Validasi pasca-LLM
        combos = self._validate_and_canonicalize_combos(paper_id, topics, llm_combos, k)

        if repair_missing:
            import itertools
            full = set()
            for r in range(1, k + 1):
                for combo in itertools.combinations(topics, r):
                    full.add(tuple(combo))
            have = set(tuple(c) for c in combos)
            missing = [list(c) for c in sorted(full - have)]
            if missing:
                print(f"  > LLM missed {len(missing)} combos; repairing due to repair_missing=True.")
                combos = [list(c) for c in sorted(have | set(tuple(m) for m in missing))]

        # Persist & print
        self._persist_combos_for_paper(paper_id, combos, topics)
        return combos

    def generate_combinations_for_papers(
        self,
        paper_ids: List[str],
        max_k: Optional[int] = None,
        repair_missing: bool = False
    ) -> Dict[str, List[List[str]]]:
        out: Dict[str, List[List[str]]] = {}
        for pid in sorted(set(paper_ids)):
            combos = self.generate_combinations_for_paper(pid, max_k=max_k, repair_missing=repair_missing)
            out[pid] = combos or []
        return out
