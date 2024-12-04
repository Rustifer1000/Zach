import sys
print("Python path:", sys.path)

import streamlit as st
from pinecone import Pinecone
print("Imported pinecone")

# Import one at a time to see which import might be causing the issue
from services.api_manager import APIManager
print("Imported APIManager")

from utils.session_manager import initialize_session_state
print("Imported initialize_session_state")

def main():
    st.title("Collins Family Mediation AI Intermediary")
    print("Starting main function")
    
    # Just basic initialization for testing
    initialize_session_state()
    print("Session state initialized")
    
    st.write("Debug mode - basic functionality only")
    
if __name__ == "__main__":
    main()
