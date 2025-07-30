import streamlit as st
import json
from src.utils.helpers import write_message, get_session_id
from src.agent.chatbot_agent import ChatbotAgent
from langchain_core.messages import AIMessage, HumanMessage

class StreamlitUI:
    """Streamlit UI for the chatbot."""
    
    def __init__(self, agent: ChatbotAgent):
        self.agent = agent
        st.set_page_config(page_title="Research Chatbot", page_icon="📚")
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def render(self) -> None:
        """Render the Streamlit UI."""
        session_id = get_session_id()
        
        # Load chat history from Neo4j and sync with session state
        chat_history = self.agent.get_memory(session_id).messages
        if len(st.session_state.messages) < len(chat_history):
            st.session_state.messages = [
                {"role": msg.type, "content": msg.content}
                for msg in chat_history
            ]
        
        # Render all messages as chat bubbles
        for message in st.session_state.messages:
            write_message(message["role"], message["content"], save=False)
        
        # Chat input
        if prompt := st.chat_input("Ask about papers, their content, or recommendations"):
            write_message("user", prompt)
            # Save user message to Neo4j
            self.agent.get_memory(session_id).add_message(HumanMessage(content=prompt))
            with st.spinner("Thinking..."):
                # Handle query
                result = self.agent.handle_query(prompt, session_id)
                response = result["response"]
                is_recommendation = result["is_recommendation"]
                
                # Save assistant response to session state and Neo4j
                write_message("assistant", response)
                self.agent.get_memory(session_id).add_message(AIMessage(content=response))
                
                # Render recommendation list if applicable
                if is_recommendation:
                    try:
                        parsed_response = json.loads(response)
                        papers = parsed_response.get("papers", [])
                        if papers:
                            with st.container():
                                st.subheader("Recommended Papers")
                                for paper in papers:
                                    st.write(f"- **{paper['title']}** (File: {paper['filename']})")
                                    st.write(f"  Topics: {', '.join(paper['topics'])}")
                    except json.JSONDecodeError:
                        st.error("Error parsing recommendation response.")