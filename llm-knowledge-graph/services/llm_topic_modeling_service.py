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

class TopicVector(BaseModel):
    topic_id: int
    suggested_name: str = Field(description="Satu nama singkat dan deskriptif untuk topik ini.")
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
        n_topics: int = 8,
        n_top_terms_per_doc: int = 10,
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

        self._schema_tm = (
            '{\n'
            '  "lsa": {\n'
            '    "doc_terms": [["term", 0.123], ...],\n'
            '    "topics": [{"topic_id": 0, "suggested_name": "...", "top_words": ["..."], "weights": [0.1, ...]}]\n'
            '  },\n'
            '  "lda": {\n'
            '    "doc_topic": [0.4, 0.3, ...],\n'
            '    "topics": [{"topic_id": 0, "suggested_name": "...", "top_words": ["..."], "weights": [0.12, ...]}],\n'
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

        self.tm_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Anda adalah ahli topic modeling yang sangat presisi. Anda akan meniru dua metode untuk SATU dokumen. Ikuti langkah-langkah ini dengan ketat:\n"
             "1. **Analisis Dokumen**: Identifikasi semua istilah teknis dan konsep utama dalam dokumen yang diberikan.\n"
             "2. **Simulasi LSA-like**:\n"
             "   - Bayangkan Anda membentuk matriks TF-IDF dan melakukan SVD rank-K.\n"
             "   - Hasilkan `doc_terms` (istilah teratas dokumen) dan `topics` (K topik laten dengan `top_words` dan `weights`).\n"
             "3. **Simulasi LDA-like**:\n"
             "   - Bayangkan Anda menjalankan model LDA dengan K topik.\n"
             "   - Hasilkan `doc_topic` (distribusi topik untuk dokumen), `topics` (K topik laten), dan `doc_terms`.\n"
             "\nATURAN UMUM YANG SANGAT PENTING:\n"
             "- **JANGAN MENGARANG**: Semua istilah (`terms` dan `words`) HARUS berasal dari atau sangat relevan dengan konteks dokumen. Jangan menciptakan istilah baru.\n"
             "- **BERI NAMA TOPIK**: Untuk setiap topik laten di LSA dan LDA, berikan `suggested_name` yang singkat dan deskriptif berdasarkan kata-katanya.\n"
             "- **FORMAT KETAT**: Semua istilah harus lowercase, tanpa duplikasi, dan spasi rapi. Bobot (`weights`) harus dalam [0,1] dan diurutkan menurun. Jumlah `doc_topic` harus mendekati 1.\n"
             "- **BATASI JUMLAH**: Batasi `doc_terms` pada masing-masing metode ke N teratas.\n"
             "- **OUTPUT JSON**: Respons Anda HARUS berupa satu objek JSON yang valid, tanpa teks atau penjelasan lain di luar JSON tersebut.\n"
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
        self.use_full_document = use_full_document
        self.max_context_chars = max_context_chars
    
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

    def process_document(self, filename: str, full_text: str, link_to_graph: bool = True) -> Dict[str, Any]:
        ctx = self._make_context(full_text, filename)

        print(f"\n>>> Running Topic Modeling for: {filename} <<<")
        out: LLMTopicsOutput = self._run_lsa_lda_like(ctx)

        # --- LSA ---
        print("\n" + "="*25 + " LSA-like Results " + "="*25)
        print("\n--- LSA: Document Top Terms ---")
        for term, w in out.lsa.doc_terms:
            print(f"  {term:<30} {w:.4f}")

        print("\n--- LSA: Latent Topics ---")
        for t in out.lsa.topics:
            print(f"  Topic {t.topic_id}: \"{t.suggested_name}\"")
            words = " ".join([f"{word}({weight:.2f})" for word, weight in zip(t.top_words, t.weights)])
            print(f"    └─ Keywords: {words}")

        # --- LDA ---
        print("\n" + "="*25 + " LDA-like Results " + "="*25)
        print("\n--- LDA: Document Topic Distribution ---")
        dist_str = []
        for i, p in enumerate(out.lda.doc_topic):
            topic_name = f"Topic {i}"
            for t in out.lda.topics:
                if t.topic_id == i:
                    topic_name = f"\"{t.suggested_name}\""
                    break
            dist_str.append(f"{p*100:.1f}% {topic_name}")
        print(f"  Document is composed of: {', '.join(dist_str)}")


        print("\n--- LDA: Document Top Terms (from topic mix) ---")
        for term, w in out.lda.doc_terms:
            print(f"  {term:<30} {w:.4f}")

        print("\n--- LDA: Latent Topics ---")
        for t in out.lda.topics:
            print(f"  Topic {t.topic_id}: \"{t.suggested_name}\"")
            words = " ".join([f"{word}({weight:.2f})" for word, weight in zip(t.top_words, t.weights)])
            print(f"    └─ Keywords: {words}")

        print("\n" + "="*68)
        print("=== Completed ===")


        return {
            "lsa": out.lsa.model_dump(),
            "lda": out.lda.model_dump()
        }

    def process_pdfs(self, pdfs: Dict[str, str], link_to_graph: bool = True) -> Dict[str, Any]:
        results = {}
        for fn, txt in pdfs.items():
            res = self.process_document(fn, txt, link_to_graph=link_to_graph)
            results[fn] = res
        return results
