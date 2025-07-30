import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    DATABASE_NAME = os.getenv("NEO4J_DATABASE", "neo4j")
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_LLM_MODEL = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL = "models/embedding-001"
    
    VECTOR_DIMENSIONS = 768
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200