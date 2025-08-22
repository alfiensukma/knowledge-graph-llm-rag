from __future__ import annotations
from typing import Dict, List, Tuple, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re
import time
from tqdm import tqdm

def _normalize_label(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s*\([^)]+\)\s*", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    if s.endswith("es") and len(s) > 4:
        s = s[:-2]
    elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        s = s[:-1]
    return s

def _extract_title_and_abstract(full_text: str, filename: str) -> str:
    """Extract title and abstract from document text"""
    if not full_text:
        return filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    
    text = full_text.strip()
    lines = text.split('\n')
    
    title = ""
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 20 and not line.lower().startswith(('abstract', 'introduction', 'keywords')):
            title = line
            break
    
    if not title:
        title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    
    abstract = ""
    abstract_found = False
    abstract_lines = []
    
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        
        if 'abstract' in line_lower and len(line_lower) < 50:
            abstract_found = True
            if len(line.strip()) > len('abstract') + 5:
                abstract_lines.append(line.strip())
            continue
        
        if abstract_found:
            line = line.strip()
            if not line:
                continue
                
            if (line_lower.startswith(('keywords', 'key words', '1.', '1 ', 'introduction', 
                                     '© ', 'doi:', 'index terms')) or
                len(abstract_lines) > 20):  # Limit abstract length
                break
                
            abstract_lines.append(line)
    
    abstract = ' '.join(abstract_lines)
    
    # Combine title and abstract
    context = f"Title: {title}"
    if abstract:
        context += f"\nAbstract: {abstract}"
    
    # Limit total length to save tokens
    max_chars = 1200
    if len(context) > max_chars:
        if abstract:
            available_for_abstract = max_chars - len(f"Title: {title}\nAbstract: ") - 10
            if available_for_abstract > 100:
                context = f"Title: {title}\nAbstract: {abstract[:available_for_abstract]}..."
            else:
                context = f"Title: {title}"
        else:
            context = context[:max_chars] + "..."
    
    return context.strip()

class TopicMapperService:
    def __init__(self, graph_service, llm, min_hits_per_term: int = 1):
        self.graph_service = graph_service
        self.llm = llm
        self.min_hits_per_term = min_hits_per_term
        
        # Token management
        self.TOKENS_PER_MINUTE_LIMIT = 1_000_000
        self.ESTIMATED_TOKENS_PER_CHAR = 0.25
        self.SAFETY_MARGIN = 0.8
        self.MAX_TOPICS_IN_PROMPT = 100
        
        print("Fetching topics from Neo4j...")
        self._cso_topics = self._fetch_topics()
        self._cso_map = {_normalize_label(topic): topic for topic in self._cso_topics}
        
        print(f"Loaded {len(self._cso_topics)} topics")
        self.document_texts = {}
        self._setup_search_index()
        self.match_parser = JsonOutputParser(pydantic_object=dict)
        self.match_prompt = ChatPromptTemplate.from_template(
            """Anda adalah ahli ontologi ilmu komputer. Cocokkan term berikut dengan topik CSO yang paling sesuai.
            TERM: "{term}"
            KONTEKS DOKUMEN:
            {document_context}
            KANDIDAT TOPIK CSO:
            {cso_topics}
            ATURAN MATCHING:
            1. Prioritaskan konteks dari judul dan abstract dokumen
            2. Pertimbangkan domain penelitian dan metodologi yang disebutkan
            3. Urutan prioritas:
               a. Match langsung dengan topik di CSO
               b. Singkatan standar CS (e.g., 'lstm' → 'long short term memory') 
               c. Konsep yang relevan dengan konteks penelitian
               d. Sinonim dalam domain yang sama
            PANDUAN SPESIFIK:
            - Gunakan konteks abstract untuk memahami fokus penelitian
            - Untuk term umum seperti 'model': pilih jenis model yang sesuai konteks
            - Untuk neural network terms: cocokkan dengan arsitektur spesifik jika disebutkan
            - Jangan generalisasi tanpa dukungan konteks yang kuat
            CONFIDENCE SCORE:
            1.0 = Match langsung/singkatan standar
            0.95 = Konsep eksplisit dalam abstract
            0.9 = Sinonim dalam konteks yang tepat  
            < 0.9 = Kembalikan None
            Output JSON:
            {{
                "term": "{term}",
                "matched_topic": "<topic atau None>",
                "confidence": <0.0-1.0>,
                "reason": "<alasan dengan rujukan ke konteks dokumen>"
            }}"""
        )
        self.match_chain = self.match_prompt | self.llm | self.match_parser
    
    def set_document_texts(self, pdfs_dict: Dict[str, str]):
        """Store full document texts for context extraction"""
        self.document_texts = pdfs_dict
        print(f"Stored {len(pdfs_dict)} document texts for full context processing")
        
    def _get_document_context(self, filename: str) -> str:
        """Get title + abstract context from full document text"""
        if filename in self.document_texts:
            full_text = self.document_texts[filename]
            return _extract_title_and_abstract(full_text, filename)
        
        return filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').strip()

    def _fetch_topics(self) -> List[str]:
        try:
            results = self.graph_service.graph.query("""
                MATCH (t:Topic)
                WHERE t.label <> 'computer science'
                RETURN t.label AS label
                ORDER BY t.label
            """)
            return [r["label"] for r in results]
        except Exception as e:
            print(f"Error fetching topics: {e}")
            return []

    def _setup_search_index(self):
        try:
            self.graph_service.graph.query("""
                CREATE FULLTEXT INDEX topic_fulltext_index IF NOT EXISTS
                FOR (t:Topic) ON EACH [t.label]
                OPTIONS {
                    indexConfig: {
                        `fulltext.analyzer`: 'english'
                    }
                }
            """)
            print("Topic search index ready")
        except Exception as e:
            print(f"Warning: Could not create search index: {e}")

    def _get_candidate_topics(self, term: str, document_context: str) -> List[str]:
        """Get candidate topics using multiple strategies"""
        candidates = set()

        norm_term = _normalize_label(term)
        if norm_term in self._cso_map:
            candidates.add(self._cso_map[norm_term])

        try:
            context_words = re.findall(r'\b[a-zA-Z]{4,}\b', document_context.lower())
            search_terms = [term] + context_words[:5]  # Limit context words
            search_query = " ".join(search_terms)
            
            results = self.graph_service.graph.query("""
                CALL db.index.fulltext.queryNodes("topic_fulltext_index", $query)
                YIELD node, score
                WHERE score > 0.2
                RETURN node.label AS label, score
                ORDER BY score DESC
                LIMIT 15
            """, {"query": search_query})
            
            for r in results:
                candidates.add(r["label"])
                
        except Exception as e:
            print(f"Warning: Full-text search failed: {e}")

        term_lower = term.lower()
        for topic in self._cso_topics:
            topic_lower = topic.lower()
            if (term_lower in topic_lower or 
                any(word in topic_lower for word in term_lower.split() if len(word) > 2)):
                candidates.add(topic)
                if len(candidates) >= self.MAX_TOPICS_IN_PROMPT:
                    break
        
        return list(candidates)[:self.MAX_TOPICS_IN_PROMPT]

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) * self.ESTIMATED_TOKENS_PER_CHAR)

    def _semantic_match(self, term: str, weight: float, filename: str) -> Tuple[str, float]:
        """Perform semantic matching with full document context"""
        try:
            document_context = self._get_document_context(filename)
            candidates = self._get_candidate_topics(term, document_context)
            
            if not candidates:
                return None, 0.0
            
            candidate_text = ", ".join(candidates)
            total_text = f"{term} {document_context} {candidate_text}"
            estimated_tokens = self._estimate_tokens(total_text)
            
            # Token management
            max_tokens_per_request = 5000
            if estimated_tokens > max_tokens_per_request:
                candidates = candidates[:10]
                candidate_text = ", ".join(candidates)
                
                # If still too long, trim context but keep title
                if self._estimate_tokens(f"{term} {document_context} {candidate_text}") > max_tokens_per_request:
                    lines = document_context.split('\n')
                    title_line = lines[0] if lines else ""
                    document_context = title_line[:300] + "..."
            
            # Perform matching with full context
            result = self.match_chain.invoke({
                "term": term,
                "document_context": document_context,
                "cso_topics": candidate_text
            })
            
            if (result.get("matched_topic") and 
                result["matched_topic"] != "None" and 
                result.get("confidence", 0) >= 0.9):
                
                print(f"  * Semantic match: {term} → {result['matched_topic']}"
                      f" (conf: {result['confidence']:.2f})")
                print(f"    Reason: {result.get('reason', 'No reason provided')}")
                
                return result["matched_topic"], weight * result["confidence"]
            
            return None, 0.0
            
        except Exception as e:
            print(f"  * Error matching '{term}': {e}")
            time.sleep(2)
            return None, 0.0

    def _select_candidates(self, terms_with_scores: List[Tuple[str, float]], 
                         filename: str, top_k: int) -> List[str]:
        """Select best matching topics with full document context"""
        matched_topics = {}
        
        print(f"  Processing {len(terms_with_scores)} terms with full document context...")
        
        for term, weight in terms_with_scores:
            # Skip very short terms or common words
            if len(term) < 3 or term.lower() in {
                'the', 'and', 'for', 'with', 'are', 'this', 'that', 'from', 
                'been', 'have', 'will', 'can', 'may', 'use', 'used', 'using'
            }:
                continue
            
            # Try exact match first
            norm_term = _normalize_label(term)
            if norm_term in self._cso_map:
                topic = self._cso_map[norm_term]
                matched_topics[topic] = max(matched_topics.get(topic, 0.0), weight)
                print(f"  ✓ Exact match: {term} → {topic}")
                continue
            
            # Try semantic matching with full document context
            matched_topic, confidence = self._semantic_match(term, weight, filename)
            if matched_topic:
                matched_topics[matched_topic] = max(
                    matched_topics.get(matched_topic, 0.0), 
                    confidence
                )
            
            time.sleep(1.2)
        
        # Sort by confidence and return top-k
        sorted_topics = sorted(
            matched_topics.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [topic for topic, _ in sorted_topics[:top_k]]

    def map_and_link(self, lsa_terms_by_doc: Dict[str, List[Tuple[str, float]]],
                     lda_terms_by_doc: Dict[str, List[Tuple[str, float]]],
                     top_k_each: int) -> Dict[str, List[str]]:
        result = {}
        
        all_files = sorted(set(list(lsa_terms_by_doc.keys()) + list(lda_terms_by_doc.keys())))
        
        print(f"Processing {len(all_files)} documents with FULL DOCUMENT CONTEXT...")
        
        with tqdm(total=len(all_files), desc="Processing documents") as pbar:
            for filename in all_files:
                print(f"\n--- Processing: {filename} ---")
                
                context = self._get_document_context(filename)
                print(f"Document context (first 150 chars): {context[:150]}...")
                
                terms_lsa = lsa_terms_by_doc.get(filename, [])
                terms_lda = lda_terms_by_doc.get(filename, [])
                
                cand_lsa = self._select_candidates(terms_lsa, filename, top_k_each)
                cand_lda = self._select_candidates(terms_lda, filename, top_k_each)
                
                all_matched = list(dict.fromkeys(cand_lsa + cand_lda))
                
                if all_matched:
                    try:
                        self.graph_service.graph.query("""
                            UNWIND $rows AS row
                            MATCH (p:Paper {filename: row.filename})
                            MATCH (t:Topic {label: row.topic})
                            MERGE (p)-[:HAS_TOPIC]->(t)
                        """, {
                            "rows": [
                                {"filename": filename, "topic": topic}
                                for topic in all_matched
                            ]
                        })
                        print(f"  Created {len(all_matched)} HAS_TOPIC relationships")
                    except Exception as e:
                        print(f"  Error creating relationships: {e}")
                
                result[filename] = all_matched
                pbar.update(1)
                pbar.set_postfix({"Topics": len(all_matched)})
                
                time.sleep(2.0)
        
        return result