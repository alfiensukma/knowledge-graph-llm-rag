from config.settings import Settings
from src.services.graph_service import GraphService
from src.services.llm_service import LLMService
from src.services.vector_service import VectorService
from src.services.topic_service import TopicService
from src.agent.chatbot_agent import ChatbotAgent
from src.ui.streamlit_ui import StreamlitUI

def main():
    # Initialize Services
    graph_service = GraphService()
    llm_service = LLMService(api_key=Settings.GEMINI_API_KEY, model_name=Settings.GEMINI_LLM_MODEL)
    topic_service = TopicService(graph_service=graph_service, llm_service=llm_service)
    vector_service = VectorService(graph_service=graph_service, llm_service=llm_service)
    agent = ChatbotAgent(graph_service=graph_service, vector_service=vector_service, llm_service=llm_service, topic_service=topic_service)
    
    # Initialize UI
    ui = StreamlitUI(agent=agent)
    ui.render()
    
if __name__ == "__main__":
    main()