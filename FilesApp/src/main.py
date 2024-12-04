import streamlit as st
from pinecone import Pinecone
from src.services.api_manager import APIManager
from src.utils.session_manager import initialize_session_state

def main():
    st.title("Collins Family Mediation AI Intermediary")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    index = pc.Index("mediation4")
    api_manager = APIManager(index)
    
    # Create containers for layout
    main_container = st.container()
    input_container = st.container()
    
    # Display current response
    with main_container:
        st.write(st.session_state.current_response)
        st.markdown("<br>" * 2, unsafe_allow_html=True)
    
    # Handle user input
    with input_container:
        user_input = st.chat_input("Your response:")
    
    if user_input:
        # Add user message to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get relevant context from vector store
        relevant_info = api_manager.query_context(
            user_input,
            st.session_state.messages
        )
        
        if relevant_info:
            st.session_state.backend_messages.append({
                "role": "system",
                "content": relevant_info
            })
            
        # Combine messages for API call
        full_context = (
            [st.session_state.messages[0]]  # System message
            + st.session_state.backend_messages  # Backend context
            + st.session_state.messages[1:]  # Conversation history
        )
        
        # Generate response and update state
        response = api_manager.generate_response(full_context)
        st.session_state.current_response = response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Force a rerun to update the display
        st.rerun()

if __name__ == "__main__":
    main()
