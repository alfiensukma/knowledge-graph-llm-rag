from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

MAX_TOPICS_LLM = 15

class TopicExtractionService:
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=List[str])
        
        self.extract_prompt = ChatPromptTemplate.from_template(
            """Berdasarkan teks paper akademik berikut, identifikasi hingga **{max_topics} topik ilmiah utama** yang dibahas.
            Fokus pada konsep ilmiah spesifik dalam ilmu komputer, contohnya 'Content-Based Filtering', 'Information 
            Retrieval', 'Text Mining', atau 'Machine Learning'. Hindari topik umum seperti 'ilmu komputer' dan/atau 
            'computer science'. Kembalikan topik dalam bentuk daftar JSON berisi string. Teks: ```{text}```\n\nJSON Output: """
        )
        
        self.extract_chain = self.extract_prompt | self.llm | self.parser

    def extract_topics_from_text(self, full_text: str, max_topics: int = MAX_TOPICS_LLM) -> List[str]:
        try:
            print(f"  > Extracting up to {max_topics} topics using LLM...")
            candidate_topics = self.extract_chain.invoke({
                "text": full_text,
                "max_topics": max_topics
            })
            print(f"  > LLM extracted {len(candidate_topics)} topics: {candidate_topics}")
            return candidate_topics if candidate_topics else []
        except Exception as e:
            print(f"  > LLM topic extraction failed: {e}")
            return []