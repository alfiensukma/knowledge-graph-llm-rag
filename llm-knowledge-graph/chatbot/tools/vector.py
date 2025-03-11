import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class VectorSearch:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv('OPENAI_API_KEY'), 
            temperature=0
        )
        
        self.embedding_provider = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        self.graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        
        self.chunk_vector = self._initialize_vector_store()
        self.retriever_chain = self._setup_retriever_chain()

    def _initialize_vector_store(self):
        """Initialize Neo4j vector store"""
        return Neo4jVector.from_existing_index(
            self.embedding_provider,
            graph=self.graph,
            index_name="chunkVector",  # Make sure this matches your Neo4j vector index name
            embedding_node_property="textEmbedding",
            text_node_property="text",
            retrieval_query=self._get_retrieval_query()
        )

    def _get_retrieval_query(self):
        """Define the retrieval query"""
        return """
        MATCH (node)-[:PART_OF]->(d:Document)
        WITH node, score, d
        MATCH (node)-[:HAS_ENTITY]->(e)
        MATCH p = (e)-[r]-(e2)
        WHERE (node)-[:HAS_ENTITY]->(e2)
        UNWIND relationships(p) as rels
        WITH 
            node, 
            score, 
            d, 
            collect(apoc.text.join(
                [labels(startNode(rels))[0], startNode(rels).id, type(rels), labels(endNode(rels))[0], endNode(rels).id]
                ," ")) as kg
        RETURN
            node.text as text, score,
            { 
                document: d.id,
                entities: kg
            } AS metadata
        """

    def _setup_retriever_chain(self):
        """Setup the retrieval chain"""
        instructions = (
            "Use the given context to answer the question. "
            "Reply with an answer that includes the id of the document and other relevant information from the text. "
            "If you don't know the answer, say you don't know. "
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", instructions),
            ("human", "{input}"),
        ])
        
        chunk_retriever = self.chunk_vector.as_retriever()
        chunk_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(chunk_retriever, chunk_chain)

    def find_chunk(self, query):
        """Search for relevant chunks based on query"""
        return self.retriever_chain.invoke({"input": query})

# Initialize vector search
vector_search = VectorSearch()

# Export the find_chunk function to maintain compatibility
def find_chunk(query):
    return vector_search.find_chunk(query)