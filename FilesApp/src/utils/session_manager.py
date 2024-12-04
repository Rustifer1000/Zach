import streamlit as st
from typing import Dict
from ..constants.messages import SYSTEM_MESSAGE, INITIAL_GREETING

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "assistant", "content": INITIAL_GREETING}
        ]
        st.session_state.user_name = ""
        st.session_state.spouse_name = ""
        st.session_state.current_issue = None
        st.session_state.issues_discussed = []
        st.session_state.backend_messages = []
        st.session_state.current_response = INITIAL_GREETING
