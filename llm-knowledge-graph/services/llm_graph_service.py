from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Any, Optional

class Author(BaseModel):
    name: str = Field(description="Nama lengkap penulis.")
    email: Optional[str] = Field(description="Alamat email penulis jika tersedia, kosongkan jika tidak ada.")

class Reference(BaseModel):
    title: str = Field(description="Judul paper atau sumber yang direferensikan.")
    doi: Optional[str] = Field(description="DOI dari referensi jika tersedia.")

class Paper(BaseModel):
    title: str = Field(description="Judul utama dari paper, wajib diisi.")
    abstract: str = Field(description="Abstrak dari paper, wajib diisi, jika tidak ada gunakan ringkasan singkat.")
    publisher: Optional[str] = Field(description="Nama penerbit, jika disebutkan.")
    venue: str = Field(description="Nama konferensi atau jurnal, wajib diisi.")
    publication_date: Optional[str] = Field(description="Tanggal publikasi, format YYYY-MM-DD jika memungkinkan.")
    authors: List[Author] = Field(description="Daftar penulis paper, minimal satu penulis.")
    references: List[Reference] = Field(description="Daftar pustaka atau referensi.")

class LLMGraphExtractionService:
    def __init__(self, llm, graph_service):
        self.llm = llm
        self.graph_service = graph_service
        self.prompt = self._create_prompt()
        self.parser = JsonOutputParser(pydantic_object=Paper)
        self.chain = self.prompt | self.llm | self.parser

    def _create_prompt(self):
        return ChatPromptTemplate.from_messages([
            (
                "system",
                """Anda adalah asisten riset yang ahli menganalisis dokumen akademik.
                Tugas Anda adalah mengekstrak informasi terstruktur ke dalam format JSON untuk paper ilmiah.
                Fokus pada: judul, abstrak, penulis, venue (jurnal/konferensi), penerbit, tanggal publikasi, dan referensi.
                - Judul wajib diisi; jika tidak jelas, gunakan frase awal atau deskripsi singkat dari teks.
                - Abstrak wajib diisi; jika tidak ada, buat ringkasan singkat (2-3 kalimat) berdasarkan teks.
                - Penulis minimal satu; ambil dari teks (misalnya, header atau bagian author); jika tidak ada, gunakan 'Unknown Author'.
                - Venue wajib diisi sebagai string (nama jurnal/konferensi); jika tidak ada, gunakan 'Unknown Journal/Conference'.
                - Penerbit: Jika tidak ada, gunakan 'Unknown Publisher'.
                - Tanggal publikasi: Jika tidak ada, gunakan 'Unknown Date'.
                - Referensi: Ambil judul dan DOI (jika ada); jika tidak ada, gunakan 'Unknown Reference'.
                - Pastikan data diambil hanya dari teks paper yang dianalisis, bukan dari referensi.
                """
            ),
            ("human", "Ekstrak informasi terstruktur dari teks paper berikut:\n\n```{text}```"),
        ])

    def process_document(self, pdf_path: str, filename: str, full_text: str) -> Optional[dict]:
        print(f"  > Processing PDF: {filename}")
        if not full_text or not isinstance(full_text, str) or not full_text.strip():
            print(f"  > No valid text provided for {filename}")
            return None
        
        # Extract structured data
        print("  > Sending document to LLM for structured graph extraction...")
        try:
            graph_data = self.chain.invoke({"text": full_text})
            # Validate and normalize data
            if isinstance(graph_data.get('venue'), dict):
                graph_data['venue'] = graph_data['venue'].get('name', graph_data['venue'].get('title', str(graph_data['venue'])))
            if not graph_data.get('venue'):
                graph_data['venue'] = "Unknown Journal or Conference"
            if not graph_data.get('title'):
                graph_data['title'] = "Untitled Paper"
            if not graph_data.get('abstract'):
                graph_data['abstract'] = "No abstract provided; paper discusses academic research."
            if not graph_data.get('authors') or not isinstance(graph_data.get('authors'), list):
                graph_data['authors'] = [{"name": "Unknown Author", "email": None}]
            else:
                valid_authors = []
                for author in graph_data['authors']:
                    if isinstance(author, dict) and 'name' in author:
                        valid_authors.append({
                            "name": author.get('name', "Unknown Author"),
                            "email": author.get('email', None)
                        })
                    elif isinstance(author, str):
                        valid_authors.append({"name": author, "email": None})
                graph_data['authors'] = valid_authors if valid_authors else [{"name": "Unknown Author", "email": None}]
            if not graph_data.get('references') or not isinstance(graph_data.get('references'), list):
                graph_data['references'] = []
            else:
                valid_references = []
                for ref in graph_data['references']:
                    if isinstance(ref, dict) and ref.get('title'):
                        valid_references.append({
                            "title": ref.get('title', "Unknown Reference"),
                            "doi": ref.get('doi', None)
                        })
                    elif isinstance(ref, str):
                        valid_references.append({"title": ref, "doi": None})
                graph_data['references'] = valid_references
            
            # Import paper graph and get UUID
            paper_id = self.graph_service.import_paper_graph(graph_data, filename)
            if not paper_id:
                print(f"  > Failed to import graph for {filename}")
                return None
                
            return {"paper_id": paper_id, "graph_data": graph_data}
        except Exception as e:
            print(f"  > Failed to process document {filename}: {e}")
            return None