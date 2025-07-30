from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.api_core import retry

class LLMService:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0
        )
        self.embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    
    @retry.Retry(predicate=retry.if_transient_error)
    def generate_response(self, context: str, query: str) -> str:
        """Generate a response with retry mechanism."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. Use the context to answer in detail, including relevant quotes.\nContext: {context}"),
            ("human", "{query}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"context": context, "query": query})
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response."
    
    @retry.Retry(predicate=retry.if_transient_error)
    def extract_topics(self, text: str) -> list:
        """Extract topics from text."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract up to 5 key computer science topics from the text. Return as a list of strings.\nText: {text}"),
            ("human", "Extract topics")
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            return eval(chain.invoke({"text": text}))  # Assuming output is a valid Python list
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
    
    @retry.Retry(predicate=retry.if_transient_error)
    def validate_topic(self, topic: str, cso_topics: list, hierarchy: list) -> dict:
        """Validate topic against CSO topics and hierarchy."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Validate the topic against CSO topics and hierarchy. Return {'matched_topic': 'topic' or 'None'}.\nCSO Topics: {cso_topics}\nHierarchy: {hierarchy}\nTopic: {topic}"),
            ("human", "Validate topic")
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            return eval(chain.invoke({"topic": topic, "cso_topics": cso_topics, "hierarchy": hierarchy}))
        except Exception as e:
            print(f"Error validating topic: {e}")
            return {"matched_topic": "None"}