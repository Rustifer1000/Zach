import openai
import streamlit as st
from openai import OpenAI
import pinecone
from pinecone import Pinecone
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import json

# Initialize OpenAI Client
client = OpenAI()

# ---- Constants and Configuration ----
class ModelTier(Enum):
    ANALYSIS = "analysis"
    RESPONSE = "response"
    EMBEDDING = "embedding"

@dataclass
class ModelConfig:
    name: str
    tokens_per_second: float
    cost_per_1k_tokens: float
    max_context_tokens: int
    latency_ms: float
    quality_score: float

MODEL_CONFIGS = {
    "gpt-4o": ModelConfig("gpt-4o", 12, 0.03, 3072, 2000, 1.0),
    "gpt-4o": ModelConfig("gpt-4o", 30, 0.0015, 16385, 500, 0.85),
    "text-embedding-3-large": ModelConfig("text-embedding-3-large", 100, 0.0004, 3072, 100, 0.90)
}

INITIAL_GREETING = """Hello! I'm the Collins Family Mediation Intermediary. I'm here to gather information and clarify issues to help you get a head start on your mediation sessions with the Collinses. To get started, could you please tell me your first name?"""

SYSTEM_MESSAGE = """
You are an AI intermediary for Collins Family Mediation, gathering information for the mediators and providing information to the user in preparation for in-person sessions with the Collinses
Your goals are:
- Collect first names of each spouse.
- Inquire about history of marriage, separation and divorce.  
    - Current living arrangement
    - legal action, contact with attorneys etc.
    - Children, number and ages, special needs etc
- Ask for main concerns and goals.
- Explore each issue in depth to fully understand this user's concerns and hopes for mediation
- Regularly check for this user's perception of the other spouse's perspective.
- Perform rough calculations of potential outcomes.
- Offer tentative proposals for settlement.
- Offer to explore with the other spouse one of the ideas or goals discussed .
- Summarize discussions.
- Identify other major concerns and goals and explore them (loop through each issue, as above in lines 46 to 51)
- Conclude with an acknowlegement of the progress you have made together and sign off

Periodically let the user know that you are updating the Collinses for the user's in-person mediation sessions
Please be empathetic and professional while gathering necessary information.
Ask for one piece of information at a time.
Identify complex financial and legal issues, and areas of high conflict to be discussed with the Collinses.
At appropriate intervals, offer to pause and reconvene as user gathers relevant information. Or to continue on.
Track the conversation to be sure all issues are addressed
Refer to the vector store for phrasing and tone.
When a user first connects, begin by asking for their first name.
follow-up by requesting the first name of the user's spouse.
"""

class APIManager:
    def __init__(self, pinecone_index):
        self.pinecone_index = pinecone_index

    def query_pinecone(self, user_input: str) -> str:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=user_input
            )
            
            embedding = response.data[0].embedding
            
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )

            relevant_info = []
            for match in results["matches"]:
                metadata = match.get('metadata', {})
                info_parts = []
                for field in ['title', 'category1', 'category2', 'priority']:
                    if field in metadata:
                        info_parts.append(f"{field.title()}: {metadata[field]}")
                
                if info_parts:
                    relevant_info.append("\n".join(info_parts))

            return "\n\n".join(relevant_info) if relevant_info else ""

        except Exception as e:
            print(f"Error in query_pinecone: {str(e)}")
            return ""

    def generate_response(self, messages: List[Dict]) -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
            )
            
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return "I apologize, but I encountered an error processing your request."

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
        st.session_state.current_response = INITIAL_GREETING  # New: store current response

def main():
    st.title("Collins Family Mediation AI Intermediary")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    index = pc.Index("mediation4")
    api_manager = APIManager(index)

    # Create a container for the main content
    main_container = st.container()
    
    # Create a container for the chat input with custom height
    input_container = st.container()
    
    # Use the main container for the response
    with main_container:
        st.write(st.session_state.current_response)
        
        # Add some spacing
        st.markdown("<br>" * 2, unsafe_allow_html=True)
    
    # Use the input container for the chat input
    with input_container:
        user_input = st.chat_input("Your response:")
    
    if user_input:
        # Add user message to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Query Pinecone and add to backend context
        relevant_info = api_manager.query_pinecone(user_input)
        if relevant_info:
            st.session_state.backend_messages.append({
                "role": "system", 
                "content": f"Consider this relevant information when responding:\n{relevant_info}"
            })
        # Combine messages for API call
        full_context = (
            [st.session_state.messages[0]]  # System message
            + st.session_state.backend_messages  # Backend context
            + st.session_state.messages[1:]  # Conversation history
        )
        # Generate response and update current_response
        response = api_manager.generate_response(full_context)
        st.session_state.current_response = response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Force a rerun to update the display
        st.rerun()

if __name__ == "__main__":
    main()
