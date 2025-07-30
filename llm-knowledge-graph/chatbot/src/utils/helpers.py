import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

def write_message(role: str, content: str, save: bool = True) -> None:
    if save:
        st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)

def get_session_id() -> str:
    return get_script_run_ctx().session_id