from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Any
import json

class RecommendationResult(BaseModel):
    filename: str = Field(description="Nama file PDF dari paper yang direkomendasikan.")
    title: str = Field(description="Judul paper yang direkomendasikan.")
    topics: List[str] = Field(description="Daftar topik yang relevan dengan paper.")

class RecommendationService:
    def __init__(self, llm, graph_service):
        self.llm = llm
        self.graph_service = graph_service
        self.prompt = self._create_prompt()
        self.parser = JsonOutputParser(pydantic_object=List[RecommendationResult])
        self.chain = self.prompt | self.llm | self.parser

    def _create_prompt(self):
        return ChatPromptTemplate.from_messages([
            (
                "system",
                """Anda adalah asisten riset yang merekomendasikan paper akademik menggunakan logika Apriori.
                - Input: topik dari paper user (`user_topics`) dan semua paper di database (`all_papers`, 
                  berisi id, filename, title, topics).
                - Tugas: Rekomendasikan paper berdasarkan pola co-occurrence topik (seperti Apriori), prioritaskan 
                  paper dengan topik yang sering muncul bersama `user_topics`.
                - Cara memilih:
                  - Gunakan topik per paper sebagai transaksi, cari itemset (kombinasi topik) yang sering muncul 
                    dengan `user_topics`.
                  - Normalisasi topik serupa (misalnya, "neural networks" = "neural network").
                  - Pertimbangkan hubungan hierarki (misalnya, "neural networks" terkait "Machine Learning").
                  - Pilih paper dengan >=2 topik yang sama atau terkait erat dengan `user_topics`, dengan co-occurrence tinggi.
                  - Abaikan duplikasi topik dalam satu paper.
                - Jangan rekomendasikan paper dengan ID di `user_paper_ids`.
                - Output: JSON berisi daftar paper: [{{"filename": "<nama_file>", "title": "<judul>", 
                  "topics": ["<topik1>", "<topik2>"]}}], atau [] jika tidak ada.
                Contoh:
                  - Input: user_topics=["machine learning", "decision support systems"], 
                    all_papers=[{{"id": "1", "filename": "paper1.pdf", "title": "ML Study", 
                    "topics": ["machine learning", "neural networks"]}}, {{"id": "2", "filename": "paper2.pdf", 
                    "title": "Decision Systems", "topics": ["decision support systems", "neural networks"]}}]
                  - Output: [{{"filename": "paper1.pdf", "title": "ML Study", "topics": ["machine learning", "neural networks"]}},
                    {{"filename": "paper2.pdf", "title": "Decision Systems", "topics": ["decision support systems", "neural networks"]}}]
                """
            ),
            (
                "human",
                """Berdasarkan topik user: {user_topics}, rekomendasikan paper dari: {all_papers}. Jangan sertakan paper dengan ID: {user_paper_ids}. Kembalikan JSON."""
            ),
        ])

    def get_llm_recommendations(self, user_paper_ids: List[str]) -> List[Dict]:
        """Get paper recommendations using LLM based on user paper IDs."""
        try:
            # Ambil topik dari paper yang dibaca user
            user_topics_query = """
            MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
            WHERE p.id IN $paper_ids
            RETURN collect(DISTINCT t.label) AS userTopics
            """
            user_topics_result = self.graph_service.graph.query(user_topics_query, {"paper_ids": user_paper_ids})
            user_topics = user_topics_result[0]["userTopics"] if user_topics_result else []
            print(f"  > User topics: {user_topics}")

            if not user_topics:
                print("  > No topics found for the given paper IDs. Returning empty recommendations.")
                return []

            # Ambil semua paper dan topiknya (kecuali paper yang dibaca user)
            all_papers_query = """
            MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
            WHERE NOT p.id IN $paper_ids
            RETURN p.id AS id, p.filename AS filename, p.title AS title, collect(t.label) AS topics
            """
            all_papers_result = self.graph_service.graph.query(all_papers_query, {"paper_ids": user_paper_ids})
            all_papers = [
                {"id": record["id"], "filename": record["filename"], "title": record["title"], "topics": record["topics"]}
                for record in all_papers_result
            ]
            print(f"  > All papers: {len(all_papers)} papers retrieved")

            if not all_papers:
                print("  > No other papers found in the database. Returning empty recommendations.")
                return []

            # Log input ke LLM
            input_data = {
                "user_topics": user_topics,
                "all_papers": json.dumps(all_papers, ensure_ascii=False),
                "user_paper_ids": user_paper_ids
            }
            print(f"  > Input to LLM: {input_data}")

            # Dapatkan rekomendasi dari LLM
            recommendations = self.chain.invoke(input_data)

            # Filter untuk memastikan ID paper tidak termasuk dalam rekomendasi
            recommendations = [
                rec for rec in recommendations
                if not any(paper["id"] in user_paper_ids for paper in all_papers if paper["filename"] == rec["filename"])
            ]

            print(f"  > LLM recommendations: {recommendations}")
            return recommendations
        except Exception as e:
            print(f"  > Failed to get LLM recommendations: {str(e)}")
            return []