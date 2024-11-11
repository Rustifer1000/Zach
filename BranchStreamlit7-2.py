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
    "gpt-40": ModelConfig("gpt-4o", 30, 0.0015, 16385, 500, 0.85),
    "text-embedding-3-large": ModelConfig("text-embedding-3-large", 100, 0.0004, 3072, 100, 0.90)
}

INITIAL_GREETING = """Hello! I'm the Collins Family Mediation Intermediary. I'm here to help you get a head start on your mediation sessions with the Collinses. To get started, could you please tell me your first name?"""

SYSTEM_MESSAGE = """
You are an AI assistant for Collins Family Mediation, helping user.
Your goals are:
- Collect first names of both parties.
- Discuss main concerns and goals.
- Explore issues in depth to fully understand what motivates this user.
- Explore this user's values, priorities, and goals.
- Explore this user's perception of the other spouse's perspective.
- Offer to explore some of the user's ideas with the other spouse.
- Offer to reconvene as user gathers relevant information, and assistant gets the perspective of other spouse.
- Perform rough calculations of potential outcomes.
- Offer tentative proposals for settlement.
- Summarize discussions.
- Suggest next steps.

Please be empathetic and professional while gathering necessary information.
When a user first connects, begin by asking for their first name.
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
                for field in ['title', 'author', 'category1', 'category2', 'date', 'priority']:
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
        st.session_state.backend_messages = []  # New: separate list for backend context

def main():
    st.title("Collins Family Mediation Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    index = pc.Index("mediation4")
    api_manager = APIManager(index)

    # Display conversation history (only visible messages)
    for message in st.session_state.messages[1:]:  # Skip the system message
        role = "You" if message['role'] == 'user' else "Assistant"
        with st.chat_message(role.lower()):
            st.write(message['content'])

    # Handle user input
    user_input = st.chat_input("Your response:")
    
    if user_input:
        # Add user message to conversation
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Query Pinecone and add to backend context
        relevant_info = api_manager.query_pinecone(user_input)
        if relevant_info:
            # Add to backend messages instead of visible messages
            st.session_state.backend_messages.append({
                "role": "system", 
                "content": f"Consider this relevant information when responding:\n{relevant_info}"
            })

        # Combine visible messages and backend context for API call
        full_context = (
            [st.session_state.messages[0]]  # System message
            + st.session_state.backend_messages  # Backend context
            + st.session_state.messages[1:]  # Visible conversation
        )

        # Generate and display assistant's response
        response = api_manager.generate_response(full_context)
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()