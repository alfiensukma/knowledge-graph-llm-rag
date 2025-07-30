from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from src.utils.helpers import get_session_id
from src.services.graph_service import GraphService
from src.services.vector_service import VectorService
from src.services.llm_service import LLMService
from src.services.topic_service import TopicService
from config.settings import Settings
import json

class ChatbotAgent:
    """Agent for handling user queries and generating responses."""
    
    def __init__(self, graph_service: GraphService, vector_service: VectorService, llm_service: LLMService, topic_service: TopicService):
        self.graph_service = graph_service
        self.vector_service = vector_service
        self.llm_service = llm_service
        self.topic_service = topic_service
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> list:
        """Create tools for the agent."""
        return [
            Tool.from_function(
                name="Paper Search",
                description="Search for papers by title or keywords",
                func=lambda x: self.graph_service.run_query(
                    """
                    MATCH (p:Paper)
                    WHERE toLower(p.title) CONTAINS toLower($keyword)
                    RETURN p.id, p.title, p.filename, p.abstract
                    LIMIT 5
                    """,
                    {"keyword": x}
                )
            ),
            Tool.from_function(
                name="Content Search",
                description="Search within paper content for specific information",
                func=lambda x: self.vector_service.search_similar_chunks(x)
            ),
            Tool.from_function(
                name="Metadata Search",
                description="Get metadata for a specific paper",
                func=lambda x: self.graph_service.get_paper_metadata(x)
            ),
            Tool.from_function(
                name="Paper Recommendation",
                description="Recommend papers based on user topics using Apriori logic",
                func=self.recommend_papers
            )
        ]
    
    def recommend_papers(self, input_str: str) -> str:
        """Recommend papers based on user topics using Apriori logic."""
        try:
            # Extract topics from input
            user_topics = self.topic_service.get_validated_topics(input_str)
            if not user_topics:
                return json.dumps({"response": "No valid topics found for recommendation.", "papers": []})
            
            # Get all papers
            all_papers = self.graph_service.run_query(
                """
                MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
                RETURN p.id, p.title, p.filename, collect(DISTINCT t.label) AS topics
                """
            )
            
            # Prepare prompt for LLM
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are an academic research assistant using Apriori logic.
                    - Input: user topics ({user_topics}) and all papers ({all_papers}).
                    - Task: Recommend papers based on topic co-occurrence with user topics.
                    - Rules:
                      - Normalize similar topics (e.g., 'neural networks' = 'neural network').
                      - Consider hierarchy (e.g., 'neural networks' related to 'Machine Learning').
                      - Select papers with >=2 topics matching or closely related to user topics.
                      - Exclude papers with IDs in {user_paper_ids}.
                    - Output: JSON list of papers: [{"filename": "<name>", "title": "<title>", "topics": ["<topic1>", "<topic2>"]}]
                    Example:
                      Input: user_topics=["machine learning", "decision support systems"], all_papers=[{"id": "1", "filename": "paper1.pdf", "title": "ML Study", "topics": ["machine learning", "neural networks"]}, {"id": "2", "filename": "paper2.pdf", "title": "Decision Systems", "topics": ["decision support systems", "neural networks"]}]
                      Output: [{"filename": "paper1.pdf", "title": "ML Study", "topics": ["machine learning", "neural networks"]}, {"filename": "paper2.pdf", "title": "Decision Systems", "topics": ["decision support systems", "neural networks"]}]
                    """
                ),
                ("human", "User topics: {user_topics}\nAll papers: {all_papers}\nExclude IDs: {user_paper_ids}")
            ])
            
            chain = prompt | self.llm_service.llm | StrOutputParser()
            response = chain.invoke({
                "user_topics": user_topics,
                "all_papers": all_papers,
                "user_paper_ids": []
            })
            # Ensure response is valid JSON
            try:
                json.loads(response)
                return response
            except json.JSONDecodeError:
                return json.dumps({"response": "Invalid recommendation format.", "papers": []})
        except Exception as e:
            print(f"Error recommending papers: {e}")
            return json.dumps({"response": f"Error: {str(e)}", "papers": []})
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with tools and prompt."""
        prompt = PromptTemplate.from_template(
            """
            You are an academic research assistant specializing in computer science papers.
            Provide detailed responses including paper titles, relevant content quotes, and metadata (authors, journal, topics).
            Use tools to search for papers, content, metadata, or recommendations as needed.
            
            TOOLS:
            ------
            {tools}
            
            TOOL NAMES:
            ------
            {tool_names}
            
            Use this format for tool usage:
            ```
            Thought: Do I need to use a tool? Yes
            Action: [tool_name]
            Action Input: [input]
            Observation: [result]
            ```
            
            For direct responses:
            ```
            Thought: Do I need to use a tool? No
            Final Answer: [response]
            ```
            
            Previous conversation history:
            {chat_history}
            
            New input: {input}
            {agent_scratchpad}
            """
        )
        agent = create_react_agent(self.llm_service.llm, self.tools, prompt, tools_renderer=lambda tools: ", ".join([t.name for t in tools]))
        return AgentExecutor(agent=agent, tools=self.tools, handle_parsing_errors=True, verbose=True)
    
    def get_memory(self, session_id: str) -> Neo4jChatMessageHistory:
        """Get chat history for a session."""
        return Neo4jChatMessageHistory(
            session_id=session_id,
            graph=self.graph_service.graph,
            database=Settings.DATABASE_NAME
        )
    
    def handle_query(self, query: str, session_id: str = None) -> dict:
        """Handle a user query and return a response."""
        session_id = session_id or get_session_id()
        try:
            # Search for relevant chunks
            chunks = self.vector_service.search_similar_chunks(query)
            context = "\n".join([f"Paper: {c['metadata']['paper_title']}\nSection: {c['metadata']['section']}\nText: {c['text']}" for c in chunks])
            
            # Add metadata for relevant papers
            paper_ids = list(set(c["metadata"]["paper_id"] for c in chunks if c["metadata"]["paper_id"]))
            if paper_ids:
                metadata = self.graph_service.get_paper_metadata(paper_ids[0])
                context += f"\n\nMetadata:\nTitle: {metadata.get('title', 'N/A')}\nAuthors: {', '.join(metadata.get('authors', []))}\nJournal: {metadata.get('journal', 'N/A')}\nTopics: {', '.join(metadata.get('topics', []))}"
            
            # Check if query asks for recommendation
            if "recommend" in query.lower() or "suggest" in query.lower():
                response = self.recommend_papers(query)
                return {"response": response, "is_recommendation": True}
            
            # Generate response
            response = self.llm_service.generate_response(context, query)
            return {"response": response, "is_recommendation": False}
        except Exception as e:
            print(f"Error handling query: {e}")
            return {"response": f"Sorry, I couldn't process your query: {str(e)}", "is_recommendation": False}