import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DOCS_PATH = "llm-knowledge-graph/data/pdfs"
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/embedding-001"
VECTOR_DIMENSIONS = 768  # Gemini-embedding-001 uses 768 dimensions
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

# Initialize embedding provider
embedding_provider = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)

# Initialize text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def embed_and_store_papers():
    try:
        # Load and split documents
        loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            filename = os.path.basename(chunk.metadata["source"])
            chunk_id = f"{filename}.{chunk.metadata['page']}"
            print(f"Processing - {chunk_id}")

            # Get title from metadata (if available)
            title = chunk.metadata.get("title", filename).strip().lower()

            # Embed the chunk
            chunk_embedding = embedding_provider.embed_query(chunk.page_content)

            # Store chunk and link to Paper node
            properties = {
                "filename": filename,
                "chunk_id": chunk_id,
                "text": chunk.page_content,
                "embedding": chunk_embedding,
                "title": title
            }

            graph.query("""
                MERGE (p:Paper {title: $title})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text, c.filename = $filename
                MERGE (p)<-[:PART_OF]-(c)
                WITH c
                CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
            """, properties)

        # Drop existing vector index (to fix dimension mismatch)
        graph.query("DROP INDEX chunkVector IF EXISTS")

        # Create new vector index
        graph.query("""
            CREATE VECTOR INDEX chunkVector
            IF NOT EXISTS
            FOR (c:Chunk) ON (c.textEmbedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $vector_dimensions,
                `vector.similarity_function`: 'cosine'
            }}
        """, {"vector_dimensions": VECTOR_DIMENSIONS})

        print("Embedding and storage completed successfully.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def main():
    print("Starting embedding and storage process...")
    embed_and_store_papers()

if __name__ == "__main__":
    main()