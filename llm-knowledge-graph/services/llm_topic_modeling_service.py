from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re
import json

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"ISSN:?\s*\d{4}-\d{4}", " ", s, flags=re.I)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_label(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = s.replace("-", " ")
    s = re.sub(r"\s*\([^)]+\)\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    if s.endswith("es") and len(s) > 4:
        s = s[:-2]
    elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        s = s[:-1]
    return s

def _extract_title_and_abstract(full_text: str, filename: str) -> str:
    if not full_text:
        return filename.replace(".pdf", "").replace("_", " ").replace("-", " ").strip()

    text = full_text.strip()
    lines = text.split("\n")

    title = ""
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 20 and not line.lower().startswith(("abstract", "introduction", "keywords")):
            title = line
            break
    if not title:
        title = filename.replace(".pdf", "").replace("_", " ").replace("-", " ").strip()

    abstract_lines, abstract_found = [], False
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if "abstract" in low and len(low) < 50:
            abstract_found = True
            if len(line.strip()) > len("abstract") + 5:
                abstract_lines.append(line.strip())
            continue
        if abstract_found:
            line = line.strip()
            if not line:
                continue
            if (low.startswith(("keywords", "key words", "1.", "1 ", "introduction", "© ", "doi:", "index terms"))
                or len(abstract_lines) > 20):
                break
            abstract_lines.append(line)
    abstract = " ".join(abstract_lines)

    context = f"Title: {title}"
    if abstract:
        context += f"\nAbstract: {abstract}"

    max_chars = 1200
    if len(context) > max_chars:
        if abstract:
            head = f"Title: {title}\nAbstract: "
            remain = max_chars - len(head) - 10
            if remain > 100:
                context = head + abstract[:remain] + "..."
            else:
                context = f"Title: {title}"
        else:
            context = context[:max_chars] + "..."
    return context.strip()

class TopicVector(BaseModel):
    topic_id: int
    top_words: List[str]
    weights: List[float]

class LSAResult(BaseModel):
    doc_terms: List[Tuple[str, float]] = Field(description="Top terms untuk dokumen (mirip TF-IDF/SVD proyeksi), urut desc.")
    topics: List[TopicVector] = Field(description="K topik laten versi LSA-like (dimensi SVD).")

class LDAResult(BaseModel):
    topics: List[TopicVector] = Field(description="K topik laten versi LDA-like (distribusi kata per topik).")
    doc_topic: List[float] = Field(description="Distribusi topik untuk dokumen (panjang K, sum≈1).")
    doc_terms: List[Tuple[str, float]] = Field(description="Top terms untuk dokumen berdasar campuran topik LDA.")

class LLMTopicsOutput(BaseModel):
    lsa: LSAResult
    lda: LDAResult

class LLMTopicModelingService:
    def __init__(
        self,
        llm,
        graph_service,
        max_topics_in_prompt,
        n_topics: int = 8,
        n_top_terms_per_doc: int = 12,
        min_confidence: float = 0.9,
        top_k_map_each: int = 5,
        use_full_document: bool = True,
        max_context_chars: Optional[int] = None,
    ):
        self.llm = llm
        self.graph_service = graph_service
        self.n_topics = n_topics
        self.n_top_terms_per_doc = n_top_terms_per_doc
        self.min_confidence = min_confidence
        self.top_k_map_each = top_k_map_each
        self.MAX_TOPICS_IN_PROMPT = max_topics_in_prompt

        self._cso_topics, self._hier = self._fetch_topics_and_hierarchy()
        self._cso_map = {_normalize_label(t): t for t in self._cso_topics}

        # ---------- letakkan skema JSON sebagai string variabel ----------
        self._schema_tm = (
            '{\n'
            '  "lsa": {\n'
            '    "doc_terms": [["term", 0.123], ...],\n'
            '    "topics": [{"topic_id": 0, "top_words": ["..."], "weights": [0.1, ...]}]\n'
            '  },\n'
            '  "lda": {\n'
            '    "doc_topic": [0.4, 0.3, ...],\n'
            '    "topics": [{"topic_id": 0, "top_words": ["..."], "weights": [0.12, ...]}],\n'
            '    "doc_terms": [["term", 0.123], ...]\n'
            '  }\n'
            '}'
        )

        self._json_map_format = (
            '{\n'
            '  "term": "<input-term>",\n'
            '  "matched_topic": "<topic|None>",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "reason": "<brief>"\n'
            '}'
        )
        # -----------------------------------------------------------------

        self.tm_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Anda meniru dua metode topic modeling untuk SATU dokumen:\n"
             "1) LSA-like:\n"
             "- Anggap Anda membentuk TF-IDF, lakukan SVD rank-K.\n"
             "- Kembalikan top terms dokumen (doc_terms) hasil proyeksi, dan K topik laten (topics) berupa top_words + weights.\n"
             "2) LDA-like:\n"
             "- Anggap Anda menjalankan LDA dengan K topik.\n"
             "- Kembalikan distribusi topik dokumen (doc_topic: panjang K, sum≈1), daftar topik (top_words + weights), dan doc_terms yang diperoleh dari campuran topik.\n"
             "ATURAN UMUM:\n"
             "- Jangan menciptakan istilah yang sama sekali tidak tampak relevan dari konteks; fokus istilah teknis di dokumen.\n"
             "- Semua terms lowercase, tanpa duplikasi, rapikan spasi.\n"
             "- Weights dalam [0,1], urut menurun, jumlah doc_topic mendekati 1.\n"
             "- Batasi doc_terms pada masing-masing metode ke N teratas.\n"
             "FORMAT:\n"
             "{schema_tm}"
            ),
            ("human",
             "Dokumen (title+abstract/bagian awal):\n```{context}```\n\n"
             "Parameter:\n"
             "- K (jumlah topik laten) = {k}\n"
             "- N (top terms per doc) = {n_top}\n\n"
             "Kembalikan JSON PENUH persis sesuai skema di atas."
            )
        ])
        self.tm_parser = JsonOutputParser(pydantic_object=LLMTopicsOutput)
        self.tm_chain = self.tm_prompt | self.llm | self.tm_parser

        self.map_prompt = ChatPromptTemplate.from_template(
            "Anda ahli ontologi CS. Pilih topik CSO yang paling sesuai untuk TERM berikut,\n"
            "dengan mempertimbangkan konteks dokumen:\n"
            "TERM: \"{term}\"\n\n"
            "KONTEKS:\n"
            "{context}\n\n"
            "KANDIDAT TOPIK CSO (subset):\n"
            "{cso_candidates}\n\n"
            "ATURAN:\n"
            "- Jika ada match langsung (string/istilah standar), pilih itu.\n"
            "- Jika tidak ada, pilih topik semantik paling dekat yang didukung konteks.\n"
            "- Kembalikan None jika tidak yakin (confidence < {min_conf}).\n\n"
            "Output JSON (format):\n"
            "{json_map_format}"
        )
        
        self.use_full_document = use_full_document
        self.max_context_chars = max_context_chars

    def _fetch_topics_and_hierarchy(self) -> Tuple[List[str], List[str]]:
        try:
            topic_rows = self.graph_service.graph.query(
                "MATCH (t:Topic) WHERE t.label <> 'computer science' RETURN t.label AS label"
            )
            topics = [r["label"] for r in topic_rows]
            hier_rows = self.graph_service.graph.query(
                """MATCH (a:Topic)-[:SUB_TOPIC_OF]->(b:Topic)
                   WHERE a.label <> 'computer science' AND b.label <> 'computer science'
                   RETURN a.label AS sub, b.label AS sup"""
            )
            hier = [f"{r['sub']} -> {r['sup']}" for r in hier_rows]
            return topics, hier
        except Exception as e:
            print(f"  > Error fetching CSO topics/hierarchy: {e}")
            return [], []

    def _get_cso_candidates(self, term: str) -> List[str]:
        term_norm = _normalize_label(term)
        cands = []
        if term_norm in self._cso_map:
            cands.append(self._cso_map[term_norm])
        # quick substring scan
        low = term_norm
        for t in self._cso_topics:
            tl = t.lower()
            if low in tl or any(w in tl for w in low.split() if len(w) > 3):
                cands.append(t)
            if len(cands) >= self.MAX_TOPICS_IN_PROMPT:
                break
        seen, out = set(), []
        for x in cands:
            if x not in seen:
                seen.add(x); out.append(x)
        return out[: self.MAX_TOPICS_IN_PROMPT]
    
    def _make_context(self, full_text: str, filename: str) -> str:
        txt = _clean_text(full_text or "")
        if self.max_context_chars and len(txt) > self.max_context_chars:
            return txt[: self.max_context_chars]
        return txt

    def _run_lsa_lda_like(self, context: str) -> LLMTopicsOutput:
        raw = self.tm_chain.invoke({
            "context": context,
            "k": self.n_topics,
            "n_top": self.n_top_terms_per_doc,
            "schema_tm": self._schema_tm,
        })

        try:
            if isinstance(raw, LLMTopicsOutput):
                return raw
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict):
                return LLMTopicsOutput.model_validate(raw)
            raise TypeError(f"Unexpected LLM output type: {type(raw)}")
        except Exception as e:
            print(f"  > Failed to parse LLM output into LLMTopicsOutput: {e}")
            try:
                snippet = str(raw)[:500]
                print(f"  > Raw output snippet: {snippet}")
            except:
                pass
            raise

    def _map_terms(self, filename: str, full_text: str,
                   lsa_terms: List[Tuple[str, float]],
                   lda_terms: List[Tuple[str, float]]) -> List[str]:
        terms = []
        for seq in (lsa_terms[: self.top_k_map_each], lda_terms[: self.top_k_map_each]):
            for t, w in seq:
                if isinstance(t, str) and len(t.strip()) >= 3:
                    terms.append((t.strip().lower(), float(w)))
        # unique by term
        seen, uniq = set(), []
        for t, w in terms:
            if t not in seen:
                seen.add(t); uniq.append((t, w))

        context = self._make_context(full_text, filename)
        mapped = []

        for term, weight in uniq:
            # exact-normalized match
            norm = _normalize_label(term)
            if norm in self._cso_map:
                mapped.append(self._cso_map[norm])
                print(f"  Exact match: {term} → {self._cso_map[norm]}")
                continue

            # candidate subset for prompt
            cands = self._get_cso_candidates(term)
            if not cands:
                continue

            # LLM
            parser = JsonOutputParser(pydantic_object=dict)
            chain = self.map_prompt | self.llm | parser
            res = chain.invoke({
                "term": term,
                "context": context,
                "cso_candidates": ", ".join(cands),
                "min_conf": f"{self.min_confidence:.2f}",
                "json_map_format": self._json_map_format,
            })

            mt = (res or {}).get("matched_topic")
            conf = float((res or {}).get("confidence", 0.0) or 0.0)
            if mt and mt != "None" and conf >= self.min_confidence:
                mapped.append(mt)
                print(f"  Semantic match: {term} → {mt} (conf: {conf:.2f})")

        # unique & return
        out = []
        seen = set()
        for t in mapped:
            if t not in seen:
                seen.add(t); out.append(t)
        return out

    def link_has_topic(self, filename: str, topics: List[str]) -> None:
        if not topics:
            return
        try:
            self.graph_service.graph.query(
                """UNWIND $rows AS row
                   MATCH (p:Paper {filename: row.filename})
                   MATCH (t:Topic {label: row.topic})
                   MERGE (p)-[:HAS_TOPIC]->(t)""",
                {"rows": [{"filename": filename, "topic": t} for t in topics]}
            )
            print(f"  Created {len(topics)} HAS_TOPIC relationships")
        except Exception as e:
            print(f"  Error creating HAS_TOPIC: {e}")

    def process_document(self, filename: str, full_text: str, link_to_graph: bool = True) -> Dict[str, Any]:
        ctx = self._make_context(full_text, filename)

        print(f"\n=== LLM Topic Modeling (LSA-like & LDA-like) ===")
        out: LLMTopicsOutput = self._run_lsa_lda_like(ctx)

        print("\n[LSA-like] Top terms:")
        for term, w in out.lsa.doc_terms[: self.n_top_terms_per_doc]:
            print(f"  - {term}: {w:.4f}")
        print("\n[LSA-like] Topics:")
        for t in out.lsa.topics:
            words = ", ".join(t.top_words[:8])
            print(f"  Topic {t.topic_id}: {words}")

        print("\n[LDA-like] doc_topic:", [round(x, 4) for x in out.lda.doc_topic])
        print("\n[LDA-like] Top terms:")
        for term, w in out.lda.doc_terms[: self.n_top_terms_per_doc]:
            print(f"  - {term}: {w:.4f}")
        print("\n[LDA-like] Topics:")
        for t in out.lda.topics:
            words = ", ".join(t.top_words[:8])
            print(f"  Topic {t.topic_id}: {words}")

        # mapping to CSO
        mapped = self._map_terms(filename, full_text, out.lsa.doc_terms, out.lda.doc_terms)
        print("\n[Mapping] Matched CSO topics:")
        for m in mapped:
            print(f"  - {m}")

        if link_to_graph and mapped:
            self.link_has_topic(filename, mapped)

        return {
            "lsa": out.lsa.model_dump(),
            "lda": out.lda.model_dump(),
            "mapped_topics": mapped
        }

    def process_pdfs(self, pdfs: Dict[str, str], link_to_graph: bool = True) -> Dict[str, Any]:
        results = {}
        for fn, txt in pdfs.items():
            res = self.process_document(fn, txt, link_to_graph=link_to_graph)
            results[fn] = res
        return results
